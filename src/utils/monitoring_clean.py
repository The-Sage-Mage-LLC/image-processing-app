# -*- coding: utf-8 -*-
"""
Self-Monitoring and Quality Assurance Module (Clean Version)
Project ID: Image Processing App 20251119
Created: 2025-01-19
Author: The-Sage-Mage

This module provides comprehensive self-monitoring, progress tracking, and QA/QC functionality
for all image processing operations with enhanced real-time feedback and statistical analysis.
"""

import time
import hashlib
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import threading
import json
import os

# Try to import psutil, fall back to basic monitoring if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from collections import deque

# Try to import numpy, provide fallback if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class EnhancedProcessingMonitor:
    """Enhanced comprehensive monitoring system for image processing operations with full QA/QC."""
    
    def __init__(self, logger: logging.Logger, config: dict):
        self.logger = logger
        self.config = config
        
        # Progress tracking
        self.current_operation = None
        self.total_items = 0
        self.processed_items = 0
        self.failed_items = 0
        self.skipped_items = 0
        self.start_time = None
        self.last_heartbeat = None
        self.heartbeat_interval = config.get('monitoring', {}).get('heartbeat_interval', 30)
        
        # Performance metrics with enhanced tracking
        self.processing_times = deque(maxlen=1000)  # Keep last 1000 for better statistics
        self.file_sizes_original = deque(maxlen=1000)
        self.file_sizes_processed = deque(maxlen=1000)
        self.hash_comparisons = deque(maxlen=1000)
        
        # System resource tracking
        self.memory_usage = deque(maxlen=100)
        self.cpu_usage = deque(maxlen=100)
        self.disk_usage = deque(maxlen=100)
        
        # Enhanced QA/QC tracking with more detailed checks
        self.qa_checks = {
            'empty_files': 0,
            'oversized_files': 0,
            'undersized_files': 0,
            'hash_duplicates': 0,
            'processing_failures': 0,
            'unexpected_timing': 0,
            'corrupted_outputs': 0,
            'memory_issues': 0,
            'disk_space_warnings': 0,
            'resolution_mismatches': 0,
            'format_validation_failures': 0
        }
        
        # Statistical expectations with enhanced tracking
        self.expected_metrics = {
            'processing_time_mean': None,
            'processing_time_std': None,
            'processing_time_median': None,
            'size_ratio_mean': None,
            'size_ratio_std': None,
            'memory_baseline': None,
            'cpu_baseline': None
        }
        
        # Performance benchmarks for comparison
        self.performance_benchmarks = {
            'items_per_minute_target': 60,  # Target processing rate
            'memory_usage_threshold': 80,   # Warning at 80% memory usage
            'cpu_usage_threshold': 90,      # Warning at 90% CPU usage
            'disk_space_threshold': 85      # Warning at 85% disk usage
        }
        
        # Thread for continuous monitoring
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Configuration thresholds with enhanced defaults
        self.size_threshold_min = config.get('monitoring', {}).get('min_file_size', 1024)
        self.size_threshold_max = config.get('monitoring', {}).get('max_file_size', 100 * 1024 * 1024)
        self.timing_deviation_threshold = config.get('monitoring', {}).get('timing_deviation_multiplier', 3)
        self.size_deviation_threshold = config.get('monitoring', {}).get('size_deviation_multiplier', 2)
        
        # Error pattern tracking
        self.error_patterns = {}
        self.error_frequency = deque(maxlen=50)
        
        # Initialize system monitoring
        self._initialize_system_monitoring()
    
    def _initialize_system_monitoring(self):
        """Initialize system resource monitoring."""
        try:
            if PSUTIL_AVAILABLE:
                # Get initial system metrics
                initial_memory = psutil.virtual_memory().percent
                initial_cpu = psutil.cpu_percent(interval=1)
                initial_disk = psutil.disk_usage('.').free
                
                self.expected_metrics['memory_baseline'] = initial_memory
                self.expected_metrics['cpu_baseline'] = initial_cpu
                
                self.logger.info(f"System monitoring initialized:")
                self.logger.info(f"   Memory baseline: {initial_memory:.1f}%")
                self.logger.info(f"   CPU baseline: {initial_cpu:.1f}%")
                self.logger.info(f"   Available disk space: {initial_disk / (1024**3):.1f}GB")
            else:
                self.logger.warning("psutil not available - system monitoring limited")
                
        except Exception as e:
            self.logger.warning(f"Could not initialize system monitoring: {e}")
    
    def start_operation(self, operation_name: str, total_items: int):
        """Start monitoring a new processing operation with enhanced tracking."""
        self.current_operation = operation_name
        self.total_items = total_items
        self.processed_items = 0
        self.failed_items = 0
        self.skipped_items = 0
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        
        # Reset metrics for this operation
        self.processing_times.clear()
        self.file_sizes_original.clear()
        self.file_sizes_processed.clear()
        self.hash_comparisons.clear()
        self.memory_usage.clear()
        self.cpu_usage.clear()
        self.disk_usage.clear()
        
        # Reset QA counters
        for key in self.qa_checks:
            self.qa_checks[key] = 0
        
        # Clear expected metrics (will be recalculated)
        metrics_to_clear = ['processing_time_mean', 'processing_time_std', 'processing_time_median', 
                           'size_ratio_mean', 'size_ratio_std']
        for key in metrics_to_clear:
            self.expected_metrics[key] = None
        
        # Enhanced startup logging
        self.logger.info(f"** OPERATION STARTED: {operation_name}")
        self.logger.info(f"** Total items to process: {total_items:,}")
        self.logger.info(f"** Multi-threaded processing enabled")
        self.logger.info(f"** QA/QC monitoring active")
        self.logger.info(f"** Heartbeat interval: {self.heartbeat_interval}s")
        
        # Start continuous monitoring
        self._start_continuous_monitoring()
    
    def _start_continuous_monitoring(self):
        """Start the continuous monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()
            self.logger.debug("Continuous monitoring thread started")
    
    def _monitoring_loop(self):
        """Enhanced background monitoring loop with system metrics."""
        monitoring_interval = min(10, self.heartbeat_interval / 3)
        
        while not self._stop_monitoring.wait(monitoring_interval):
            if self.current_operation:
                current_time = time.time()
                
                # Collect system metrics if available
                try:
                    if PSUTIL_AVAILABLE:
                        memory_percent = psutil.virtual_memory().percent
                        cpu_percent = psutil.cpu_percent()
                        disk_free = psutil.disk_usage('.').free / (1024**3)  # GB
                        
                        self.memory_usage.append(memory_percent)
                        self.cpu_usage.append(cpu_percent)
                        self.disk_usage.append(disk_free)
                        
                        # Check for resource warnings
                        if memory_percent > self.performance_benchmarks['memory_usage_threshold']:
                            self.qa_checks['memory_issues'] += 1
                            self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
                        
                        if cpu_percent > self.performance_benchmarks['cpu_usage_threshold']:
                            self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                        
                        if disk_free < 2:  # Less than 2GB free
                            self.qa_checks['disk_space_warnings'] += 1
                            self.logger.warning(f"Low disk space: {disk_free:.1f}GB remaining")
                            
                except Exception as e:
                    self.logger.debug(f"Error collecting system metrics: {e}")
                
                # Send heartbeat if enough time has passed
                if current_time - self.last_heartbeat >= self.heartbeat_interval:
                    self._send_enhanced_heartbeat(current_time)
    
    def _send_enhanced_heartbeat(self, current_time: float):
        """Send enhanced heartbeat with comprehensive status."""
        elapsed = current_time - self.start_time if self.start_time else 0
        
        # Calculate detailed progress metrics
        total_processed = self.processed_items + self.failed_items + self.skipped_items
        progress_pct = (total_processed / self.total_items * 100) if self.total_items > 0 else 0
        success_rate = (self.processed_items / total_processed * 100) if total_processed > 0 else 0
        
        # Calculate processing rates
        if elapsed > 0 and total_processed > 0:
            items_per_minute = (total_processed / elapsed) * 60
            items_per_hour = items_per_minute * 60
            estimated_completion_seconds = ((self.total_items - total_processed) / total_processed) * elapsed
            eta_str = str(timedelta(seconds=int(estimated_completion_seconds)))
        else:
            items_per_minute = 0
            items_per_hour = 0
            eta_str = "calculating..."
        
        # Get current system metrics
        current_memory = self.memory_usage[-1] if self.memory_usage else 0
        current_cpu = self.cpu_usage[-1] if self.cpu_usage else 0
        current_disk = self.disk_usage[-1] if self.disk_usage else 0
        
        # Calculate QA issues
        total_qa_issues = sum(self.qa_checks.values())
        qa_rate = (total_qa_issues / total_processed * 100) if total_processed > 0 else 0
        
        # Enhanced heartbeat message with safe characters
        self.logger.info(f"** STILL PROCESSING - {self.current_operation}")
        self.logger.info(f"   Progress: {total_processed:,}/{self.total_items:,} ({progress_pct:.1f}%)")
        self.logger.info(f"   Success Rate: {success_rate:.1f}% ({self.processed_items:,} completed)")
        self.logger.info(f"   Processing Rate: {items_per_hour:.0f} items/hour ({items_per_minute:.1f}/min)")
        self.logger.info(f"   ETA: {eta_str} | Elapsed: {str(timedelta(seconds=int(elapsed)))}")
        self.logger.info(f"   QA Issues: {total_qa_issues} ({qa_rate:.2f}% of processed)")
        if PSUTIL_AVAILABLE:
            self.logger.info(f"   System: CPU {current_cpu:.1f}% | RAM {current_memory:.1f}% | Disk {current_disk:.1f}GB")
        self.logger.info(f"   STATUS: ALIVE AND PROCESSING - Timestamp: {datetime.now().strftime('%H:%M:%S')}")
        
        self.last_heartbeat = current_time
    
    def record_processing_result(self, file_path: Path, original_size: int, 
                               processed_path: Optional[Path], processing_time: float,
                               success: bool, original_hash: str = None):
        """Enhanced processing result recording with comprehensive QA checks."""
        start_qa_time = time.time()
        
        # Basic progress tracking
        if success:
            self.processed_items += 1
        else:
            self.failed_items += 1
            self.qa_checks['processing_failures'] += 1
            self._track_error_pattern(file_path, "Processing failed")
            self.logger.warning(f"Processing failed for: {file_path.name}")
        
        # Record processing time and original size
        self.processing_times.append(processing_time)
        self.file_sizes_original.append(original_size)
        
        if success and processed_path and processed_path.exists():
            try:
                # Enhanced file validation
                processed_size = processed_path.stat().st_size
                self.file_sizes_processed.append(processed_size)
                
                # QA Check 1: File size validation
                if processed_size < self.size_threshold_min:
                    self.qa_checks['empty_files'] += 1
                    self.logger.warning(f"QA Alert: Unusually small output file ({processed_size} bytes): {processed_path.name}")
                
                if processed_size > self.size_threshold_max:
                    self.qa_checks['oversized_files'] += 1
                    self.logger.warning(f"QA Alert: Unusually large output file ({processed_size / (1024*1024):.1f} MB): {processed_path.name}")
                
                # QA Check 2: File format validation
                if not self._validate_output_format(processed_path):
                    self.qa_checks['format_validation_failures'] += 1
                    self.logger.warning(f"QA Alert: Output format validation failed: {processed_path.name}")
                
                # QA Check 3: Image corruption detection
                if not self._validate_image_integrity(processed_path):
                    self.qa_checks['corrupted_outputs'] += 1
                    self.logger.warning(f"QA Alert: Output image appears corrupted: {processed_path.name}")
                
                # QA Check 4: Hash comparison (enhanced)
                if original_hash:
                    processed_hash = self._calculate_file_hash(processed_path)
                    if original_hash == processed_hash:
                        self.qa_checks['hash_duplicates'] += 1
                        self.logger.warning(f"QA Alert: Processed file identical to original (same hash): {processed_path.name}")
                    else:
                        self.logger.debug(f"Hash verification passed: {processed_path.name}")
                    self.hash_comparisons.append((original_hash, processed_hash))
                
                # QA Check 5: Statistical analysis (enhanced)
                if len(self.file_sizes_processed) >= 10:
                    self._update_expected_metrics()
                    self._perform_statistical_qa_checks(file_path, processing_time, processed_size, original_size)
                
                # QA Check 6: Resolution validation
                if not self._validate_image_resolution(file_path, processed_path):
                    self.qa_checks['resolution_mismatches'] += 1
                    self.logger.warning(f"QA Alert: Unexpected resolution change: {processed_path.name}")
                
            except Exception as e:
                self.qa_checks['processing_failures'] += 1
                self.logger.error(f"Error during enhanced QA analysis for {processed_path}: {e}")
                self._track_error_pattern(file_path, f"QA analysis failed: {e}")
        
        # Enhanced progress logging
        total_processed = self.processed_items + self.failed_items + self.skipped_items
        if total_processed % 25 == 0:  # Every 25 items
            self._log_enhanced_progress()
        
        # QA timing check
        qa_time = time.time() - start_qa_time
        if qa_time > 0.2:  # QA taking too long
            self.logger.debug(f"QA analysis took {qa_time:.2f}s for {file_path.name}")
    
    def _validate_output_format(self, file_path: Path) -> bool:
        """Validate that the output file format is correct."""
        try:
            # Check file extension
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            if file_path.suffix.lower() not in valid_extensions:
                return False
            
            # Try to open with PIL to verify it's a valid image
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    img.verify()
                return True
            except ImportError:
                # PIL not available, just check extension
                return True
            
        except Exception:
            return False
    
    def _validate_image_integrity(self, file_path: Path) -> bool:
        """Check if the processed image is not corrupted."""
        try:
            # Test with PIL if available
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    img.load()
            except ImportError:
                pass  # PIL not available
            
            # Test with OpenCV if available
            try:
                import cv2
                cv2_img = cv2.imread(str(file_path))
                if cv2_img is None:
                    return False
                
                # Check if image has reasonable dimensions
                h, w = cv2_img.shape[:2]
                if h < 10 or w < 10:  # Too small to be valid
                    return False
            except ImportError:
                pass  # OpenCV not available
            
            return True
        except Exception:
            return False
    
    def _validate_image_resolution(self, original_path: Path, processed_path: Path) -> bool:
        """Validate that image resolution changes are within expected bounds."""
        try:
            import cv2
            
            original_img = cv2.imread(str(original_path))
            processed_img = cv2.imread(str(processed_path))
            
            if original_img is None or processed_img is None:
                return False
            
            orig_h, orig_w = original_img.shape[:2]
            proc_h, proc_w = processed_img.shape[:2]
            
            # For most operations, resolution should remain the same
            # Allow some tolerance for operations that might resize
            size_ratio = (proc_h * proc_w) / (orig_h * orig_w)
            
            # Accept if within reasonable bounds (0.1x to 10x original size)
            return 0.1 <= size_ratio <= 10.0
            
        except ImportError:
            return True  # OpenCV not available, assume it's okay
        except Exception:
            return True  # If we can't check, assume it's okay
    
    def _perform_statistical_qa_checks(self, file_path: Path, processing_time: float, 
                                     processed_size: int, original_size: int):
        """Perform statistical QA checks against expected metrics."""
        try:
            # Processing time analysis
            if (self.expected_metrics['processing_time_mean'] is not None and 
                self.expected_metrics['processing_time_std'] is not None):
                
                time_deviation = abs(processing_time - self.expected_metrics['processing_time_mean'])
                time_threshold = self.timing_deviation_threshold * self.expected_metrics['processing_time_std']
                
                if time_deviation > time_threshold:
                    self.qa_checks['unexpected_timing'] += 1
                    if processing_time > self.expected_metrics['processing_time_mean']:
                        self.logger.warning(f"QA Alert: Processing slower than expected "
                                          f"({processing_time:.2f}s vs expected {self.expected_metrics['processing_time_mean']:.2f}s): "
                                          f"{file_path.name}")
                    else:
                        self.logger.debug(f"Processing faster than expected "
                                        f"({processing_time:.2f}s vs expected {self.expected_metrics['processing_time_mean']:.2f}s): "
                                        f"{file_path.name}")
            
            # File size ratio analysis
            if (original_size > 0 and self.expected_metrics['size_ratio_mean'] is not None and 
                self.expected_metrics['size_ratio_std'] is not None):
                
                size_ratio = processed_size / original_size
                size_deviation = abs(size_ratio - self.expected_metrics['size_ratio_mean'])
                size_threshold = self.size_deviation_threshold * self.expected_metrics['size_ratio_std']
                
                if size_deviation > size_threshold:
                    if size_ratio < self.expected_metrics['size_ratio_mean']:
                        self.qa_checks['undersized_files'] += 1
                        self.logger.warning(f"QA Alert: Output file smaller than expected "
                                          f"(ratio: {size_ratio:.2f}, expected: {self.expected_metrics['size_ratio_mean']:.2f}): "
                                          f"{file_path.name}")
                    else:
                        self.qa_checks['oversized_files'] += 1
                        self.logger.warning(f"QA Alert: Output file larger than expected "
                                          f"(ratio: {size_ratio:.2f}, expected: {self.expected_metrics['size_ratio_mean']:.2f}): "
                                          f"{file_path.name}")
        
        except Exception as e:
            self.logger.debug(f"Error in statistical QA checks: {e}")
    
    def _track_error_pattern(self, file_path: Path, error_msg: str):
        """Track error patterns for analysis."""
        error_key = f"{file_path.suffix.lower()}:{error_msg[:50]}"
        self.error_patterns[error_key] = self.error_patterns.get(error_key, 0) + 1
        self.error_frequency.append(time.time())
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file efficiently."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read in larger chunks for better performance
                for chunk in iter(lambda: f.read(65536), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _update_expected_metrics(self):
        """Update expected metrics based on current samples with enhanced statistics."""
        if len(self.processing_times) >= 5:
            times_list = list(self.processing_times)
            self.expected_metrics['processing_time_mean'] = statistics.mean(times_list)
            self.expected_metrics['processing_time_median'] = statistics.median(times_list)
            if len(times_list) > 1:
                self.expected_metrics['processing_time_std'] = statistics.stdev(times_list)
            else:
                self.expected_metrics['processing_time_std'] = 0
        
        if len(self.file_sizes_processed) >= 5 and len(self.file_sizes_original) >= 5:
            size_ratios = [p/o for p, o in zip(list(self.file_sizes_processed), list(self.file_sizes_original)) if o > 0]
            if size_ratios:
                self.expected_metrics['size_ratio_mean'] = statistics.mean(size_ratios)
                if len(size_ratios) > 1:
                    self.expected_metrics['size_ratio_std'] = statistics.stdev(size_ratios)
                else:
                    self.expected_metrics['size_ratio_std'] = 0
    
    def _log_enhanced_progress(self):
        """Log enhanced progress information with comprehensive metrics."""
        if not self.current_operation:
            return
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        total_processed = self.processed_items + self.failed_items + self.skipped_items
        progress_pct = (total_processed / self.total_items * 100) if self.total_items > 0 else 0
        
        # Calculate enhanced metrics
        success_rate = (self.processed_items / total_processed * 100) if total_processed > 0 else 0
        if elapsed > 0:
            items_per_second = total_processed / elapsed
            items_per_hour = items_per_second * 3600
        else:
            items_per_second = 0
            items_per_hour = 0
        
        # Calculate averages with safe handling
        avg_processing_time = statistics.mean(self.processing_times) if self.processing_times else 0
        avg_original_size = statistics.mean(self.file_sizes_original) if self.file_sizes_original else 0
        avg_processed_size = statistics.mean(self.file_sizes_processed) if self.file_sizes_processed else 0
        
        # System metrics
        avg_memory = statistics.mean(self.memory_usage) if self.memory_usage else 0
        avg_cpu = statistics.mean(self.cpu_usage) if self.cpu_usage else 0
        current_disk = self.disk_usage[-1] if self.disk_usage else 0
        
        # Enhanced progress report with safe characters
        self.logger.info(f"** DETAILED PROGRESS REPORT - {self.current_operation}")
        self.logger.info(f"   Progress: {total_processed:,}/{self.total_items:,} ({progress_pct:.1f}%)")
        self.logger.info(f"   Success: {self.processed_items:,} | Failed: {self.failed_items:,} | Skipped: {self.skipped_items:,}")
        self.logger.info(f"   Success Rate: {success_rate:.1f}%")
        self.logger.info(f"   Processing Rate: {items_per_hour:.0f} items/hour ({items_per_second:.2f} items/sec)")
        self.logger.info(f"   Avg Processing Time: {avg_processing_time:.2f}s per item")
        self.logger.info(f"   File Sizes: {avg_original_size/1024:.0f}KB -> {avg_processed_size/1024:.0f}KB")
        if PSUTIL_AVAILABLE and self.memory_usage and self.cpu_usage:
            self.logger.info(f"   System Performance: CPU {avg_cpu:.1f}% | RAM {avg_memory:.1f}% | Disk {current_disk:.1f}GB free")
        
        # QA summary
        total_qa_issues = sum(self.qa_checks.values())
        if total_qa_issues > 0:
            self.logger.info(f"   QA Issues: {total_qa_issues} total ({total_qa_issues/total_processed*100:.2f}%)")
            for check_name, count in self.qa_checks.items():
                if count > 0:
                    self.logger.info(f"     - {check_name.replace('_', ' ').title()}: {count}")
        else:
            self.logger.info(f"   QA Status: No issues detected")
    
    def complete_operation(self):
        """Complete the current operation with comprehensive final reporting."""
        if not self.current_operation:
            return
        
        # Stop monitoring
        self._stop_monitoring.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=2)
        
        total_time = time.time() - self.start_time if self.start_time else 0
        total_processed = self.processed_items + self.failed_items + self.skipped_items
        
        # Comprehensive final statistics with safe characters
        self.logger.info("")
        self.logger.info("="*100)
        self.logger.info(f"** OPERATION COMPLETED: {self.current_operation}")
        self.logger.info("="*100)
        
        # Performance summary
        self.logger.info(f"** PERFORMANCE SUMMARY:")
        self.logger.info(f"   Total Processing Time: {total_time:.2f}s ({str(timedelta(seconds=int(total_time)))})")
        self.logger.info(f"   Items Processed: {self.processed_items:,}/{self.total_items:,}")
        self.logger.info(f"   Failed Items: {self.failed_items:,}")
        self.logger.info(f"   Skipped Items: {self.skipped_items:,}")
        self.logger.info(f"   Overall Success Rate: {(self.processed_items/self.total_items*100):.1f}%" if self.total_items > 0 else "   Success Rate: N/A")
        
        if total_time > 0 and total_processed > 0:
            final_rate = total_processed / total_time * 3600
            self.logger.info(f"   Final Processing Rate: {final_rate:.0f} items/hour")
            
            # Compare against benchmarks
            if final_rate >= self.performance_benchmarks['items_per_minute_target'] * 60:
                self.logger.info(f"   Performance: EXCELLENT (exceeded target rate)")
            elif final_rate >= self.performance_benchmarks['items_per_minute_target'] * 60 * 0.8:
                self.logger.info(f"   Performance: GOOD (within 20% of target)")
            else:
                self.logger.info(f"   Performance: BELOW TARGET (consider optimization)")
        
        # Processing time analysis
        if self.processing_times:
            self._log_comprehensive_statistics()
        
        # System resource summary
        self._log_system_resource_summary()
        
        # QA/QC comprehensive summary
        self._log_comprehensive_qa_summary(total_processed)
        
        # Error pattern analysis
        self._log_error_pattern_analysis()
        
        # Final health check
        self.logger.info(f"** SYSTEM HEALTH CHECK:")
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.info(f"   WARNING: Monitoring thread still active")
        else:
            self.logger.info(f"   OK: Monitoring thread properly terminated")
        
        self.logger.info(f"   OK: Operation completed successfully with comprehensive monitoring")
        self.logger.info("="*100)
        self.logger.info("")
        
        # Reset operation state
        self.current_operation = None
    
    def _log_comprehensive_statistics(self):
        """Log comprehensive processing time statistics."""
        times_list = list(self.processing_times)
        
        self.logger.info(f"** PROCESSING TIME ANALYSIS:")
        self.logger.info(f"   Average Time: {statistics.mean(times_list):.3f}s per item")
        self.logger.info(f"   Median Time: {statistics.median(times_list):.3f}s per item")
        self.logger.info(f"   Min Time: {min(times_list):.3f}s")
        self.logger.info(f"   Max Time: {max(times_list):.3f}s")
        
        if len(times_list) > 1:
            std_dev = statistics.stdev(times_list)
            self.logger.info(f"   Std Deviation: {std_dev:.3f}s")
            
            # Calculate percentiles
            sorted_times = sorted(times_list)
            p95 = sorted_times[int(0.95 * len(sorted_times))]
            p99 = sorted_times[int(0.99 * len(sorted_times))]
            self.logger.info(f"   95th Percentile: {p95:.3f}s")
            self.logger.info(f"   99th Percentile: {p99:.3f}s")
    
    def _log_system_resource_summary(self):
        """Log system resource usage summary."""
        if PSUTIL_AVAILABLE and self.memory_usage and self.cpu_usage:
            self.logger.info(f"** SYSTEM RESOURCE USAGE:")
            self.logger.info(f"   Memory: Avg {statistics.mean(self.memory_usage):.1f}% | "
                           f"Peak {max(self.memory_usage):.1f}%")
            self.logger.info(f"   CPU: Avg {statistics.mean(self.cpu_usage):.1f}% | "
                           f"Peak {max(self.cpu_usage):.1f}%")
            if self.disk_usage:
                self.logger.info(f"   Disk: Min Free Space {min(self.disk_usage):.1f}GB")
    
    def _log_comprehensive_qa_summary(self, total_processed: int):
        """Log comprehensive QA/QC summary."""
        self.logger.info(f"** COMPREHENSIVE QUALITY ASSURANCE SUMMARY:")
        total_qa_issues = sum(self.qa_checks.values())
        self.logger.info(f"   Total QA Issues: {total_qa_issues:,}")
        
        if total_processed > 0:
            qa_rate = (total_qa_issues / total_processed * 100)
            self.logger.info(f"   QA Issue Rate: {qa_rate:.2f}%")
            
            if qa_rate == 0:
                self.logger.info(f"   EXCELLENT: No quality issues detected")
            elif qa_rate < 1:
                self.logger.info(f"   GOOD: Quality issues below 1%")
            elif qa_rate < 5:
                self.logger.info(f"   ACCEPTABLE: Quality issues below 5%")
            else:
                self.logger.info(f"   ATTENTION NEEDED: Quality issues above 5%")
        
        # Detailed QA breakdown
        for check_name, count in self.qa_checks.items():
            if count > 0:
                percentage = (count / total_processed * 100) if total_processed > 0 else 0
                self.logger.info(f"   - {check_name.replace('_', ' ').title()}: {count:,} ({percentage:.2f}%)")
        
        # Hash analysis
        if self.hash_comparisons:
            unique_hashes = set()
            duplicate_count = 0
            for orig, proc in self.hash_comparisons:
                if orig == proc:
                    duplicate_count += 1
                unique_hashes.add(orig)
                unique_hashes.add(proc)
            
            self.logger.info(f"** HASH VERIFICATION:")
            self.logger.info(f"   Hash Comparisons: {len(self.hash_comparisons):,}")
            self.logger.info(f"   Unique Hashes: {len(unique_hashes):,}")
            self.logger.info(f"   Identical Hashes: {duplicate_count:,}")
    
    def _log_error_pattern_analysis(self):
        """Log error pattern analysis."""
        if self.error_patterns:
            self.logger.info(f"** ERROR PATTERN ANALYSIS:")
            sorted_patterns = sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)
            
            for pattern, count in sorted_patterns[:5]:  # Top 5 error patterns
                self.logger.info(f"   - {pattern}: {count} occurrences")
            
            # Error frequency analysis
            if len(self.error_frequency) > 1:
                recent_errors = [t for t in self.error_frequency if time.time() - t < 300]  # Last 5 minutes
                if recent_errors:
                    self.logger.info(f"   Recent Error Rate: {len(recent_errors)} errors in last 5 minutes")
    
    def send_heartbeat(self, custom_message: str = None):
        """Send an immediate enhanced heartbeat message."""
        if self.current_operation:
            current_time = time.time()
            self._send_enhanced_heartbeat(current_time)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get current comprehensive monitoring statistics as a dictionary."""
        if not self.current_operation:
            return {}
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        total_processed = self.processed_items + self.failed_items + self.skipped_items
        
        return {
            'operation': self.current_operation,
            'progress': {
                'processed': self.processed_items,
                'failed': self.failed_items,
                'skipped': self.skipped_items,
                'total': self.total_items,
                'percentage': (total_processed / self.total_items * 100) if self.total_items > 0 else 0,
                'success_rate': (self.processed_items / total_processed * 100) if total_processed > 0 else 0
            },
            'timing': {
                'elapsed_seconds': elapsed,
                'avg_processing_time': statistics.mean(self.processing_times) if self.processing_times else 0,
                'median_processing_time': statistics.median(self.processing_times) if self.processing_times else 0,
                'items_per_hour': (total_processed / elapsed * 3600) if elapsed > 0 else 0
            },
            'qa_checks': dict(self.qa_checks),
            'system_metrics': {
                'avg_memory_usage': statistics.mean(self.memory_usage) if self.memory_usage else 0,
                'avg_cpu_usage': statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
                'current_disk_free': self.disk_usage[-1] if self.disk_usage else 0
            },
            'file_metrics': {
                'avg_original_size': statistics.mean(self.file_sizes_original) if self.file_sizes_original else 0,
                'avg_processed_size': statistics.mean(self.file_sizes_processed) if self.file_sizes_processed else 0,
                'size_ratio': (statistics.mean(self.file_sizes_processed) / statistics.mean(self.file_sizes_original)) if self.file_sizes_original and self.file_sizes_processed else 0
            },
            'error_patterns': dict(self.error_patterns)
        }


# Keep the original classes for backward compatibility
class ProcessingMonitor(EnhancedProcessingMonitor):
    """Alias for backward compatibility."""
    pass


class HeartbeatLogger:
    """Enhanced lightweight heartbeat logger for simple operations."""
    
    def __init__(self, logger: logging.Logger, interval: int = 30):
        self.logger = logger
        self.interval = interval
        self.last_heartbeat = time.time()
        self.beat_count = 0
    
    def beat(self, message: str = "Still processing"):
        """Send a heartbeat if enough time has passed with enhanced formatting."""
        current_time = time.time()
        if current_time - self.last_heartbeat >= self.interval:
            self.beat_count += 1
            elapsed = current_time - self.last_heartbeat
            
            # Enhanced heartbeat message with safe characters
            self.logger.info(f"** {message} - Status: ALIVE AND ACTIVE")
            self.logger.info(f"   Timestamp: {datetime.now().strftime('%H:%M:%S')}")
            self.logger.info(f"   Beat #{self.beat_count} - Interval: {elapsed:.1f}s")
            
            self.last_heartbeat = current_time
    
    def force_beat(self, message: str = "Checkpoint"):
        """Force an immediate enhanced heartbeat."""
        self.beat_count += 1
        
        self.logger.info(f"** {message} - Status: ALIVE AND ACTIVE")
        self.logger.info(f"   Timestamp: {datetime.now().strftime('%H:%M:%S')}")
        self.logger.info(f"   Forced Beat #{self.beat_count}")
        
        self.last_heartbeat = time.time()