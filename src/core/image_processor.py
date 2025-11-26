"""
Image Processing Application - Core Image Processor
Project ID: Image Processing App 20251119
Created: 2025-11-19 07:08:42 UTC
Author: The-Sage-Mage
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import time
import traceback
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure NumPy threading before any NumPy operations
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
# Prevent threading conflicts in scientific libraries
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Optional imports
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    
    # Create a simple fallback progress bar
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.count = 0
            self.desc = kwargs.get('desc', 'Processing')
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            self.count += n
            if self.count % 10 == 0 or self.count == self.total:
                print(f"{self.desc}: {self.count}/{self.total}")

from src.transforms.basic_transforms import BasicTransforms
from src.transforms.artistic_transforms import ArtisticTransforms
from src.transforms.activity_transforms import ActivityTransforms
from src.models.blur_detector import BlurDetector
from src.models.caption_generator import CaptionGenerator
from src.core.metadata_handler import MetadataHandler
from src.utils.monitoring import EnhancedProcessingMonitor, HeartbeatLogger

from .file_manager import FileManager
from ..utils.database import DatabaseManager


class ImageProcessor:
    """Main image processing orchestrator with comprehensive monitoring and QA/QC."""
    
    def __init__(self, file_manager: FileManager, config: dict, logger: logging.Logger):
        self.file_manager = file_manager
        self.config = config
        self.logger = logger
        
        # Configure threading safety before initializing transforms
        self._configure_threading_safety()
        
        self.basic_transforms = BasicTransforms(config, logger)
        self.artistic_transforms = ArtisticTransforms(config, logger)
        
        # Initialize comprehensive monitoring system
        self.monitor = EnhancedProcessingMonitor(logger, config)
        self.heartbeat = HeartbeatLogger(logger, interval=config.get('monitoring', {}).get('heartbeat_interval', 30))
        
        # Initialize database if checkpoint enabled
        if config.get('general', {}).get('checkpoint_enabled', True):
            self.db_manager = DatabaseManager(
                file_manager.admin_path,
                config,
                logger
            )
        else:
            self.db_manager = None
        
        # Performance settings - reduced for thread safety
        configured_workers = config.get('general', {}).get('max_parallel_workers', 4)
        # Limit to 2 workers maximum to prevent NumPy threading conflicts
        self.max_workers = min(configured_workers, 2)
        if configured_workers > 2:
            self.logger.warning(f"Reducing max_workers from {configured_workers} to {self.max_workers} for thread safety")
        
        self.enable_gpu = config.get('general', {}).get('enable_gpu', False)
        
        # Statistics tracking (now enhanced with monitoring)
        self.stats = {
            'total_processed': 0,
            'total_failed': 0,
            'processing_times': [],
            'file_sizes': []
        }
        
        # Initialize GPU if available and enabled
        if self.enable_gpu:
            self._initialize_gpu()
    
    def _configure_threading_safety(self):
        """Configure threading safety for NumPy, OpenCV, and related libraries."""
        try:
            # Additional NumPy threading configuration
            import numpy as np
            # Set NumPy to use single-threaded BLAS
            np.seterr(all='warn')  # Convert errors to warnings to prevent crashes
            
            # Configure OpenCV threading
            cv2.setNumThreads(1)  # Force OpenCV to single-threaded mode
            
            # Log threading configuration
            self.logger.info("Threading safety configured:")
            self.logger.info(f"  - OpenCV threads: {cv2.getNumThreads()}")
            self.logger.info(f"  - NumPy threading environment variables set")
            self.logger.info(f"  - Max parallel workers limited to prevent conflicts")
            
        except Exception as e:
            self.logger.warning(f"Could not fully configure threading safety: {e}")
    
    def _initialize_gpu(self):
        """Initialize GPU/CUDA support if available."""
        try:
            # Check for CUDA availability with OpenCV
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.logger.info(f"CUDA enabled devices found: {cv2.cuda.getCudaEnabledDeviceCount()}")
                self.gpu_available = True
                
                # Set CUDA device
                cv2.cuda.setDevice(0)
                device_info = cv2.cuda.getDevice()
                self.logger.info(f"Using CUDA device {device_info}")
                
                # Test CUDA functionality with thread safety
                try:
                    # Create a test GPU memory allocation
                    test_gpu_mat = cv2.cuda_GpuMat()
                    test_cpu_mat = np.zeros((100, 100), dtype=np.uint8)
                    test_gpu_mat.upload(test_cpu_mat)
                    test_result = test_gpu_mat.download()
                    
                    # Clean up immediately to prevent memory issues
                    del test_gpu_mat, test_cpu_mat, test_result
                    
                    self.logger.info("CUDA memory operations test passed")
                    self.gpu_available = True
                except Exception as cuda_test_error:
                    self.logger.warning(f"CUDA test failed, falling back to CPU: {cuda_test_error}")
                    self.gpu_available = False
                    self.enable_gpu = False
            else:
                self.logger.warning("CUDA enabled but no devices found")
                self.gpu_available = False
                self.enable_gpu = False
        except Exception as e:
            self.logger.warning(f"GPU initialization failed: {e}")
            self.gpu_available = False
            self.enable_gpu = False
        
        # Additional check for PyTorch CUDA if available
        try:
            import torch
            if torch.cuda.is_available() and self.enable_gpu:
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                self.logger.info(f"PyTorch CUDA available: {device_count} devices, using {device_name}")
                self.torch_gpu_available = True
                
                # Set PyTorch to use limited threads for compatibility
                torch.set_num_threads(1)
            else:
                self.torch_gpu_available = False
        except ImportError:
            self.torch_gpu_available = False
    
    def process_with_progress(self, process_func, items: List[Any], 
                            description: str, use_parallel: bool = True) -> List[Any]:
        """
        Process items with progress bar, comprehensive monitoring, and thread-safe QA/QC.
        
        Args:
            process_func: Function to apply to each item
            items: List of items to process
            description: Description for progress bar
            use_parallel: Whether to use parallel processing
            
        Returns:
            List of results with comprehensive monitoring
        """
        results = []
        failed_count = 0
        
        # Force sequential processing if only 1 worker to avoid threading overhead
        if self.max_workers <= 1:
            use_parallel = False
        
        # Start comprehensive monitoring
        self.monitor.start_operation(description, len(items))
        
        self.logger.info(f">> Starting {description} with thread-safe monitoring")
        self.logger.info(f">> Processing {len(items)} items with {self.max_workers if use_parallel else 1} workers (thread-safe mode)")
        
        # Create progress bar
        with tqdm(total=len(items), desc=description, unit='file') as pbar:
            if use_parallel and self.max_workers > 1:
                # Thread-safe parallel processing
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(self._thread_safe_process_item, process_func, item): item for item in items}
                    
                    for future in as_completed(futures):
                        try:
                            result, processing_metrics = future.result()
                            if result:
                                results.append(result)
                            else:
                                failed_count += 1
                            
                            # Record metrics with monitoring system
                            item = futures[future]
                            if processing_metrics:
                                self.monitor.record_processing_result(
                                    item if isinstance(item, Path) else Path(str(item)),
                                    processing_metrics.get('original_size', 0),
                                    processing_metrics.get('processed_path'),
                                    processing_metrics.get('processing_time', 0),
                                    result is not None,
                                    processing_metrics.get('original_hash')
                                )
                                
                        except Exception as e:
                            self.logger.error(f"Thread-safe processing error: {e}")
                            failed_count += 1
                            item = futures[future]
                            self.monitor.record_processing_result(
                                item if isinstance(item, Path) else Path(str(item)),
                                0, None, 0, False
                            )
                        finally:
                            pbar.update(1)
                            
                            # Enhanced progress message with monitoring
                            if pbar.n % 10 == 0:
                                monitoring_summary = self.monitor.get_monitoring_summary()
                                rate = monitoring_summary.get('timing', {}).get('items_per_hour', 0)
                                self.logger.info(f">> Progress: {pbar.n}/{len(items)} files processed "
                                               f"({pbar.n/len(items)*100:.1f}%) - "
                                               f"Rate: {rate:.0f} items/hour - "
                                               f"QA Issues: {sum(monitoring_summary.get('qa_checks', {}).values())}")
                            
                            # Send heartbeat
                            self.heartbeat.beat(f"Processing {description}")
                            
            else:
                # Sequential processing with enhanced monitoring
                for i, item in enumerate(items):
                    try:
                        result, processing_metrics = self._thread_safe_process_item(process_func, item)
                        if result:
                            results.append(result)
                        else:
                            failed_count += 1
                        
                        # Record metrics with monitoring system
                        if processing_metrics:
                            self.monitor.record_processing_result(
                                item if isinstance(item, Path) else Path(str(item)),
                                processing_metrics.get('original_size', 0),
                                processing_metrics.get('processed_path'),
                                processing_metrics.get('processing_time', 0),
                                result is not None,
                                processing_metrics.get('original_hash')
                            )
                                
                    except Exception as e:
                        self.logger.error(f"Sequential processing error: {e}")
                        failed_count += 1
                        self.monitor.record_processing_result(
                            item if isinstance(item, Path) else Path(str(item)),
                            0, None, 0, False
                        )
                    finally:
                        pbar.update(1)
                        
                        # Enhanced progress message
                        if (i + 1) % 10 == 0:
                            monitoring_summary = self.monitor.get_monitoring_summary()
                            rate = monitoring_summary.get('timing', {}).get('items_per_hour', 0)
                            self.logger.info(f">> Progress: {i+1}/{len(items)} files processed "
                                           f"({(i+1)/len(items)*100:.1f}%) - "
                                           f"Rate: {rate:.0f} items/hour - "
                                           f"QA Issues: {sum(monitoring_summary.get('qa_checks', {}).values())}")
                        
                        # Send heartbeat
                        self.heartbeat.beat(f"Processing {description}")
        
        # Complete monitoring and log comprehensive summary
        self.monitor.complete_operation()
        
        # Update statistics
        self.stats['total_processed'] += len(results)
        self.stats['total_failed'] += failed_count
        
        return results
    
    def _thread_safe_process_item(self, process_func, item) -> tuple:
        """
        Process a single item with thread-safe monitoring and memory management.
        
        Returns:
            Tuple of (result, processing_metrics)
        """
        start_time = time.time()
        original_size = 0
        original_hash = None
        processed_path = None
        
        try:
            # Get original file metrics if it's a file path
            if isinstance(item, Path) and item.exists():
                original_size = item.stat().st_size
                original_hash = self.monitor._calculate_file_hash(item)
            
            # Process the item with thread safety
            result = process_func(item)
            
            # Try to find the processed file path for QA analysis
            if result and hasattr(result, '__iter__') and not isinstance(result, str):
                # If result is a tuple/list, look for Path objects
                for r in result if isinstance(result, (list, tuple)) else [result]:
                    if isinstance(r, Path) and r.exists():
                        processed_path = r
                        break
            elif isinstance(result, Path) and result.exists():
                processed_path = result
            elif isinstance(result, bool) and result:
                # For boolean returns, try to infer the output path
                if isinstance(item, Path):
                    # Common output patterns based on operation type
                    base_name = item.stem
                    for prefix in ['CLR_ORIG_', 'BWG_ORIG_', 'SEP_ORIG_', 'PSK_ORIG_', 'BK_Coloring_', 'BK_CTD_', 'BK_CBN_']:
                        potential_path = self.file_manager.output_path / prefix.rstrip('_') / f"{prefix}{item.name}"
                        if potential_path.exists():
                            processed_path = potential_path
                            break
            
            processing_time = time.time() - start_time
            
            metrics = {
                'processing_time': processing_time,
                'original_size': original_size,
                'original_hash': original_hash,
                'processed_path': processed_path
            }
            
            return result, metrics
            
        except Exception as e:
            self.logger.error(f"Thread-safe processing error for {item}: {e}")
            processing_time = time.time() - start_time
            metrics = {
                'processing_time': processing_time,
                'original_size': original_size,
                'original_hash': original_hash,
                'processed_path': None
            }
            return None, metrics

    def convert_pencil_sketch(self):
        """Execute pencil sketch conversion (Menu Item 9)."""
        self.logger.info("Starting pencil sketch conversion (Menu Item 9)")
        
        # Get all image files
        image_files = self.file_manager.get_image_files()
        if not image_files:
            self.logger.warning("No image files found for processing")
            return
        
        self.logger.info(f"Found {len(image_files)} image files for pencil sketch conversion")
        
        # Process with progress tracking
        results = self.process_with_progress(
            self._process_single_pencil_sketch,
            image_files,
            "Converting to Pencil Sketch",
            use_parallel=True
        )
        
        self.logger.info(f"Pencil sketch conversion completed: {len(results)} successful")
    
    def _process_single_pencil_sketch(self, image_path: Path) -> bool:
        """Process single image for pencil sketch conversion."""
        try:
            # Use artistic transforms
            sketch_image = self.artistic_transforms.convert_to_pencil_sketch(image_path)
            
            if sketch_image is None:
                self.logger.warning(f"Failed to create pencil sketch for: {image_path}")
                return False
            
            # Create output path with PSK_ORIG folder structure
            relative_path = self.file_manager.get_relative_path(image_path)
            output_folder = self.file_manager.output_path / "PSK_ORIG" / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Create filename with PSK_ORIG_ prefix
            new_filename = f"PSK_ORIG_{image_path.name}"
            output_path = output_folder / new_filename
            
            # Handle duplicate filenames with sequence numbers
            counter = 1
            while output_path.exists():
                name_stem = image_path.stem
                name_suffix = image_path.suffix
                new_filename = f"PSK_ORIG_{name_stem}_{counter}{name_suffix}"
                output_path = output_folder / new_filename
                counter += 1
            
            # Save the image
            import cv2
            success = cv2.imwrite(str(output_path), sketch_image)
            
            if success:
                self.logger.debug(f"Saved pencil sketch: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to save pencil sketch: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing pencil sketch for {image_path}: {e}")
            return False
    
    def convert_coloring_book(self):
        """Execute coloring book conversion (Menu Item 10)."""
        self.logger.info("Starting coloring book conversion (Menu Item 10)")
        
        # Get all image files
        image_files = self.file_manager.get_image_files()
        if not image_files:
            self.logger.warning("No image files found for processing")
            return
        
        self.logger.info(f"Found {len(image_files)} image files for coloring book conversion")
        
        # Process with progress tracking
        results = self.process_with_progress(
            self._process_single_coloring_book,
            image_files,
            "Converting to Coloring Book",
            use_parallel=True
        )
        
        self.logger.info(f"Coloring book conversion completed: {len(results)} successful")
    
    def _process_single_coloring_book(self, image_path: Path) -> bool:
        """Process single image for coloring book conversion."""
        try:
            # Use artistic transforms
            coloring_image = self.artistic_transforms.convert_to_coloring_book(image_path)
            
            if coloring_image is None:
                self.logger.warning(f"Failed to create coloring book page for: {image_path}")
                return False
            
            # Create output path with BK_Coloring folder structure
            relative_path = self.file_manager.get_relative_path(image_path)
            output_folder = self.file_manager.output_path / "BK_Coloring" / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Create filename with BK_Coloring_ prefix
            new_filename = f"BK_Coloring_{image_path.name}"
            output_path = output_folder / new_filename
            
            # Handle duplicate filenames with sequence numbers
            counter = 1
            while output_path.exists():
                name_stem = image_path.stem
                name_suffix = image_path.suffix
                new_filename = f"BK_Coloring_{name_stem}_{counter}{name_suffix}"
                output_path = output_folder / new_filename
                counter += 1
            
            # Save the image
            import cv2
            success = cv2.imwrite(str(output_path), coloring_image)
            
            if success:
                self.logger.debug(f"Saved coloring book page: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to save coloring book page: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing coloring book for {image_path}: {e}")
            return False
    
    def convert_connect_dots(self):
        """Execute connect-the-dots conversion (Menu Item 11)."""
        self.logger.info("Starting connect-the-dots conversion (Menu Item 11)")
        
        # Get all image files
        image_files = self.file_manager.get_image_files()
        if not image_files:
            self.logger.warning("No image files found for processing")
            return
        
        self.logger.info(f"Found {len(image_files)} image files for connect-the-dots conversion")
        
        # Initialize activity transforms if not already done
        if not hasattr(self, 'activity_transforms'):
            from ..transforms.activity_transforms import ActivityTransforms
            self.activity_transforms = ActivityTransforms(self.config, self.logger)
        
        # Process with progress tracking
        results = self.process_with_progress(
            self._process_single_connect_dots,
            image_files,
            "Converting to Connect-the-Dots",
            use_parallel=False  # Sequential for complex processing
        )
        
        self.logger.info(f"Connect-the-dots conversion completed: {len(results)} successful")
    
    def _process_single_connect_dots(self, image_path: Path) -> bool:
        """Process single image for connect-the-dots conversion."""
        try:
            # Use activity transforms
            dots_image = self.activity_transforms.convert_to_connect_dots(image_path)
            
            if dots_image is None:
                self.logger.warning(f"Failed to create connect-the-dots for: {image_path} (image may be unsuitable)")
                return False
            
            # Create output path with BK_CTD folder structure
            relative_path = self.file_manager.get_relative_path(image_path)
            output_folder = self.file_manager.output_path / "BK_CTD" / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Create filename with BK_CTD_ prefix
            new_filename = f"BK_CTD_{image_path.name}"
            output_path = output_folder / new_filename
            
            # Handle duplicate filenames with sequence numbers
            counter = 1
            while output_path.exists():
                name_stem = image_path.stem
                name_suffix = image_path.suffix
                new_filename = f"BK_CTD_{name_stem}_{counter}{name_suffix}"
                output_path = output_folder / new_filename
                counter += 1
            
            # Save the image
            import cv2
            success = cv2.imwrite(str(output_path), dots_image)
            
            if success:
                self.logger.debug(f"Saved connect-the-dots: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to save connect-the-dots: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing connect-the-dots for {image_path}: {e}")
            return False
    
    def convert_color_by_numbers(self):
        """Execute color-by-numbers conversion (Menu Item 12)."""
        self.logger.info("Starting color-by-numbers conversion (Menu Item 12)")
        
        # Get all image files
        image_files = self.file_manager.get_image_files()
        if not image_files:
            self.logger.warning("No image files found for processing")
            return
        
        self.logger.info(f"Found {len(image_files)} image files for color-by-numbers conversion")
        
        # Initialize activity transforms if not already done
        if not hasattr(self, 'activity_transforms'):
            from ..transforms.activity_transforms import ActivityTransforms
            self.activity_transforms = ActivityTransforms(self.config, self.logger)
        
        # Process with progress tracking
        results = self.process_with_progress(
            self._process_single_color_by_numbers,
            image_files,
            "Converting to Color-by-Numbers",
            use_parallel=False  # Sequential for complex processing
        )
        
        self.logger.info(f"Color-by-numbers conversion completed: {len(results)} successful")
    
    def _process_single_color_by_numbers(self, image_path: Path) -> bool:
        """Process single image for color-by-numbers conversion."""
        try:
            # Use activity transforms
            cbn_image = self.activity_transforms.convert_to_color_by_numbers(image_path)
            
            if cbn_image is None:
                self.logger.warning(f"Failed to create color-by-numbers for: {image_path} (image may be unsuitable)")
                return False
            
            # Create output path with BK_CBN folder structure
            relative_path = self.file_manager.get_relative_path(image_path)
            output_folder = self.file_manager.output_path / "BK_CBN" / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Create filename with BK_CBN_ prefix
            new_filename = f"BK_CBN_{image_path.name}"
            output_path = output_folder / new_filename
            
            # Handle duplicate filenames with sequence numbers
            counter = 1
            while output_path.exists():
                name_stem = image_path.stem
                name_suffix = image_path.suffix
                new_filename = f"BK_CBN_{name_stem}_{counter}{name_suffix}"
                output_path = output_folder / new_filename
                counter += 1
            
            # Save the image
            import cv2
            success = cv2.imwrite(str(output_path), cbn_image)
            
            if success:
                self.logger.debug(f"Saved color-by-numbers: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to save color-by-numbers: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing color-by-numbers for {image_path}: {e}")
            return False
    
    def detect_blur(self):
        """Execute blur detection (Menu Item 2)."""
        self.logger.info("Starting blur detection (Menu Item 2)")
        
        # Get all image files
        image_files = self.file_manager.get_image_files()
        if not image_files:
            self.logger.warning("No image files found for blur detection")
            return
        
        self.logger.info(f"Found {len(image_files)} image files for blur detection")
        
        # Initialize blur detector if not already done
        if not hasattr(self, 'blur_detector'):
            self.blur_detector = BlurDetector(self.config, self.logger)
        
        # Process with progress tracking
        results = self.process_with_progress(
            self._process_single_blur_detection,
            image_files,
            "Detecting Blur",
            use_parallel=True
        )
        
        self.logger.info(f"Blur detection completed: {len(results)} images analyzed")
    
    def _process_single_blur_detection(self, image_path: Path) -> bool:
        """Process single image for blur detection."""
        try:
            # Use blur detector
            blur_score = self.blur_detector.analyze_blur(image_path)
            
            if blur_score is not None:
                self.logger.debug(f"Blur score for {image_path.name}: {blur_score:.2f}")
                return True
            else:
                self.logger.warning(f"Failed to analyze blur for: {image_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error detecting blur for {image_path}: {e}")
            return False
    
    def extract_metadata(self):
        """Execute metadata extraction (Menu Item 3)."""
        self.logger.info("Starting metadata extraction (Menu Item 3)")
        
        # Get all image files
        image_files = self.file_manager.get_image_files()
        if not image_files:
            self.logger.warning("No image files found for metadata extraction")
            return
        
        self.logger.info(f"Found {len(image_files)} image files for metadata extraction")
        
        # Initialize metadata handler if not already done
        if not hasattr(self, 'metadata_handler'):
            self.metadata_handler = MetadataHandler(self.config, self.logger)
        
        # Process with progress tracking
        results = self.process_with_progress(
            self._process_single_metadata_extraction,
            image_files,
            "Extracting Metadata",
            use_parallel=True
        )
        
        self.logger.info(f"Metadata extraction completed: {len(results)} images processed")
    
    def _process_single_metadata_extraction(self, image_path: Path) -> bool:
        """Process single image for metadata extraction."""
        try:
            # Use metadata handler
            metadata = self.metadata_handler.extract_metadata(image_path)
            
            if metadata:
                self.logger.debug(f"Extracted metadata for {image_path.name}: {len(metadata)} fields")
                return True
            else:
                self.logger.warning(f"Failed to extract metadata for: {image_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error extracting metadata for {image_path}: {e}")
            return False
    
    def generate_captions(self):
        """Execute caption generation (Menu Item 4)."""
        self.logger.info("Starting caption generation (Menu Item 4)")
        
        # Get all image files
        image_files = self.file_manager.get_image_files()
        if not image_files:
            self.logger.warning("No image files found for caption generation")
            return
        
        self.logger.info(f"Found {len(image_files)} image files for caption generation")
        
        # Initialize caption generator if not already done
        if not hasattr(self, 'caption_generator'):
            self.caption_generator = CaptionGenerator(self.config, self.logger)
        
        # Process with progress tracking
        results = self.process_with_progress(
            self._process_single_caption_generation,
            image_files,
            "Generating Captions",
            use_parallel=False  # Sequential for AI operations
        )
        
        self.logger.info(f"Caption generation completed: {len(results)} captions generated")
    
    def _process_single_caption_generation(self, image_path: Path) -> bool:
        """Process single image for caption generation."""
        try:
            # Use caption generator
            caption = self.caption_generator.generate_caption(image_path)
            
            if caption:
                self.logger.debug(f"Generated caption for {image_path.name}: {caption[:50]}...")
                return True
            else:
                self.logger.warning(f"Failed to generate caption for: {image_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error generating caption for {image_path}: {e}")
            return False
    
    def execute_menu_option_5(self):
        """Execute comprehensive color analysis (Menu Item 5)."""
        self.logger.info("Starting color analysis (Menu Item 5)")
        
        # Get all image files
        image_files = self.file_manager.get_image_files()
        if not image_files:
            self.logger.warning("No image files found for color analysis")
            return
        
        self.logger.info(f"Found {len(image_files)} image files for color analysis")
        
        # Process with progress tracking
        results = self.process_with_progress(
            self._process_single_color_analysis,
            image_files,
            "Analyzing Colors",
            use_parallel=True
        )
        
        self.logger.info(f"Color analysis completed: {len(results)} images analyzed")
    
    def _process_single_color_analysis(self, image_path: Path) -> bool:
        """Process single image for color analysis."""
        try:
            # Use basic transforms for color analysis
            analysis = self.basic_transforms.analyze_colors(image_path)
            
            if analysis:
                self.logger.debug(f"Color analysis for {image_path.name} completed")
                return True
            else:
                self.logger.warning(f"Failed to analyze colors for: {image_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error analyzing colors for {image_path}: {e}")
            return False
    
    def copy_color_images(self):
        """Execute color copy (Menu Item 6)."""
        self.logger.info("Starting color copy (Menu Item 6)")
        
        # Get all image files
        image_files = self.file_manager.get_image_files()
        if not image_files:
            self.logger.warning("No image files found for color copy")
            return
        
        self.logger.info(f"Found {len(image_files)} image files for color copy")
        
        # Process with progress tracking
        results = self.process_with_progress(
            self._process_single_color_copy,
            image_files,
            "Copying Color Images",
            use_parallel=True
        )
        
        self.logger.info(f"Color copy completed: {len(results)} images copied")
    
    def _process_single_color_copy(self, image_path: Path) -> bool:
        """Process single image for color copy."""
        try:
            # Create output path with CLR_ORIG folder structure
            relative_path = self.file_manager.get_relative_path(image_path)
            output_folder = self.file_manager.output_path / "CLR_ORIG" / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Create filename with CLR_ORIG_ prefix
            new_filename = f"CLR_ORIG_{image_path.name}"
            output_path = output_folder / new_filename
            
            # Handle duplicate filenames with sequence numbers
            counter = 1
            while output_path.exists():
                name_stem = image_path.stem
                name_suffix = image_path.suffix
                new_filename = f"CLR_ORIG_{name_stem}_{counter}{name_suffix}"
                output_path = output_folder / new_filename
                counter += 1
            
            # Copy the image
            import shutil
            shutil.copy2(str(image_path), str(output_path))
            
            self.logger.debug(f"Copied color image: {output_path}")
            return True
                
        except Exception as e:
            self.logger.error(f"Error copying color image {image_path}: {e}")
            return False
    
    def convert_grayscale(self):
        """Execute grayscale conversion (Menu Item 7)."""
        self.logger.info("Starting grayscale conversion (Menu Item 7)")
        
        # Get all image files
        image_files = self.file_manager.get_image_files()
        if not image_files:
            self.logger.warning("No image files found for grayscale conversion")
            return
        
        self.logger.info(f"Found {len(image_files)} image files for grayscale conversion")
        
        # Process with progress tracking
        results = self.process_with_progress(
            self._process_single_grayscale,
            image_files,
            "Converting to Grayscale",
            use_parallel=True
        )
        
        self.logger.info(f"Grayscale conversion completed: {len(results)} successful")
    
    def _process_single_grayscale(self, image_path: Path) -> bool:
        """Process single image for grayscale conversion."""
        try:
            # Use basic transforms
            gray_image = self.basic_transforms.convert_to_grayscale(image_path)
            
            if gray_image is None:
                self.logger.warning(f"Failed to convert to grayscale: {image_path}")
                return False
            
            # Create output path with BWG_ORIG folder structure
            relative_path = self.file_manager.get_relative_path(image_path)
            output_folder = self.file_manager.output_path / "BWG_ORIG" / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Create filename with BWG_ORIG_ prefix
            new_filename = f"BWG_ORIG_{image_path.name}"
            output_path = output_folder / new_filename
            
            # Handle duplicate filenames with sequence numbers
            counter = 1
            while output_path.exists():
                name_stem = image_path.stem
                name_suffix = image_path.suffix
                new_filename = f"BWG_ORIG_{name_stem}_{counter}{name_suffix}"
                output_path = output_folder / new_filename
                counter += 1
            
            # Save the image
            import cv2
            success = cv2.imwrite(str(output_path), gray_image)
            
            if success:
                self.logger.debug(f"Saved grayscale image: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to save grayscale image: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error converting to grayscale {image_path}: {e}")
            return False
    
    def convert_sepia(self):
        """Execute sepia conversion (Menu Item 8)."""
        self.logger.info("Starting sepia conversion (Menu Item 8)")
        
        # Get all image files
        image_files = self.file_manager.get_image_files()
        if not image_files:
            self.logger.warning("No image files found for sepia conversion")
            return
        
        self.logger.info(f"Found {len(image_files)} image files for sepia conversion")
        
        # Process with progress tracking
        results = self.process_with_progress(
            self._process_single_sepia,
            image_files,
            "Converting to Sepia",
            use_parallel=True
        )
        
        self.logger.info(f"Sepia conversion completed: {len(results)} successful")
    
    def _process_single_sepia(self, image_path: Path) -> bool:
        """Process single image for sepia conversion."""
        try:
            # Use basic transforms
            sepia_image = self.basic_transforms.convert_to_sepia(image_path)
            
            if sepia_image is None:
                self.logger.warning(f"Failed to convert to sepia: {image_path}")
                return False
            
            # Create output path with SEP_ORIG folder structure
            relative_path = self.file_manager.get_relative_path(image_path)
            output_folder = self.file_manager.output_path / "SEP_ORIG" / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Create filename with SEP_ORIG_ prefix
            new_filename = f"SEP_ORIG_{image_path.name}"
            output_path = output_folder / new_filename
            
            # Handle duplicate filenames with sequence numbers
            counter = 1
            while output_path.exists():
                name_stem = image_path.stem
                name_suffix = image_path.suffix
                new_filename = f"SEP_ORIG_{name_stem}_{counter}{name_suffix}"
                output_path = output_folder / new_filename
                counter += 1
            
            # Save the image
            import cv2
            success = cv2.imwrite(str(output_path), sepia_image)
            
            if success:
                self.logger.debug(f"Saved sepia image: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to save sepia image: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error converting to sepia {image_path}: {e}")
            return False
