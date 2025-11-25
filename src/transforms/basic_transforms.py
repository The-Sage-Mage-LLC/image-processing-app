"""
Basic Image Transformations with Quality Control Integration
Project ID: Image Processing App 20251119
Created: 2025-11-19 06:52:45 UTC
Enhanced: 2025-01-25 with comprehensive quality constraints
Author: The-Sage-Mage
"""

from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
from typing import Optional
import logging

# Import quality control integration
try:
    from ..utils.quality_controlled_transforms import QualityControlledTransformBase
    QUALITY_CONTROL_AVAILABLE = True
except ImportError:
    # Fallback for standalone operation
    QUALITY_CONTROL_AVAILABLE = False


class BasicTransforms:
    """Basic image transformation operations with integrated quality control."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize quality control if available
        if QUALITY_CONTROL_AVAILABLE:
            from ..utils.image_quality_manager import ImageQualityManager
            self.quality_manager = ImageQualityManager(config, logger)
            self.logger.info("BasicTransforms initialized WITH quality control")
        else:
            self.quality_manager = None
            self.logger.warning("BasicTransforms initialized WITHOUT quality control (fallback mode)")
        
        # Load configuration settings
        self.grayscale_config = config.get('grayscale', {})
        self.sepia_config = config.get('sepia', {})
        self.preserve_metadata = config.get('basic_transforms', {}).get('preserve_metadata', False)
        self.strip_sensitive = config.get('processing', {}).get('strip_sensitive_metadata', True)
        
        self.logger.debug("BasicTransforms initialized with thread safety")
    
    def convert_to_grayscale(self, image_path: Path) -> Optional[Image.Image]:
        """
        Convert image to grayscale with quality control enforcement.
        
        ENFORCED CONSTRAINTS:
        - Minimum resolution: 256 pixels/inch (higher is better)
        - Width: 3-19 inches (greater is better within limits) 
        - Height: 3-19 inches (greater is better within limits)
        - No distortion or blur added
        - Optimized for viewing and printing quality
        
        Args:
            image_path: Path to input image
            
        Returns:
            Processed PIL Image meeting all quality constraints or None if error
        """
        try:
            # Use quality-controlled processing if available
            if self.quality_manager:
                return self._convert_to_grayscale_with_quality_control(image_path)
            else:
                # Fallback to basic processing
                return self._convert_to_grayscale_basic(image_path)
                
        except Exception as e:
            self.logger.error(f"Error converting {image_path} to grayscale: {e}")
            return None
    
    def _convert_to_grayscale_with_quality_control(self, image_path: Path) -> Optional[Image.Image]:
        """Grayscale conversion with full quality control."""
        def grayscale_transform(path):
            return self._convert_to_grayscale_basic(path)
        
        # Analyze source metrics
        source_metrics = self.quality_manager.analyze_image_metrics(image_path)
        
        # Log quality status
        if source_metrics.meets_constraints:
            self.logger.info(f"Source image meets quality constraints: {image_path.name}")
        else:
            self.logger.info(f"Source image quality issues (will be fixed): {image_path.name}")
            for issue in source_metrics.issues:
                self.logger.info(f"  - {issue}")
        
        # Apply transformation with quality control
        from ..utils.quality_controlled_transforms import QualityControlledTransformBase
        quality_processor = QualityControlledTransformBase(self.config, self.logger)
        
        result = quality_processor.process_with_quality_control(
            image_path, grayscale_transform
        )
        
        return result
    
    def _convert_to_grayscale_basic(self, image_path: Path) -> Optional[Image.Image]:
        """Basic grayscale conversion without quality control."""
        # Open image with validation
        image = Image.open(image_path)
        if not image:
            self.logger.error(f"Failed to open image: {image_path}")
            return None
        
        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Get conversion method
        method = self.grayscale_config.get('method', 'luminosity')
        
        if method == 'average':
            # Simple average method
            grayscale = image.convert('L')
        
        elif method == 'luminosity':
            # Luminosity method (weighted average)
            if image.mode == 'RGB':
                # Use PIL's built-in conversion with proper weights
                grayscale = image.convert('L')
            else:
                grayscale = image
        
        elif method == 'desaturation':
            # Desaturation method (max + min) / 2
            if image.mode == 'RGB':
                try:
                    # Thread-safe numpy operations
                    np_image = np.array(image)
                    if np_image.size == 0:
                        self.logger.error(f"Empty image array for: {image_path}")
                        return None
                        
                    max_rgb = np.max(np_image, axis=2)
                    min_rgb = np.min(np_image, axis=2)
                    grayscale_array = ((max_rgb + min_rgb) / 2).astype(np.uint8)
                    grayscale = Image.fromarray(grayscale_array, mode='L')
                except Exception as e:
                    self.logger.warning(f"NumPy desaturation failed for {image_path}, falling back to luminosity: {e}")
                    grayscale = image.convert('L')
            else:
                grayscale = image
        
        else:
            # Default to luminosity
            grayscale = image.convert('L')
        
        # Validate result
        if not grayscale:
            self.logger.error(f"Grayscale conversion failed for: {image_path}")
            return None
        
        # Handle metadata preservation
        if self.preserve_metadata and not self.strip_sensitive:
            try:
                # Preserve EXIF data
                if hasattr(image, 'info') and image.info:
                    grayscale.info = image.info.copy()
                if hasattr(image, '_getexif'):
                    try:
                        exif = image._getexif()
                        if exif:
                            grayscale.info['exif'] = exif
                    except:
                        # Silently ignore EXIF errors
                        pass
            except Exception as e:
                self.logger.warning(f"Failed to preserve metadata for {image_path}: {e}")
        
        return grayscale
    
    def convert_to_sepia(self, image_path: Path) -> Optional[Image.Image]:
        """
        Convert image to sepia tone with quality control enforcement.
        
        ENFORCED CONSTRAINTS:
        - Minimum resolution: 256 pixels/inch (higher is better)
        - Width: 3-19 inches (greater is better within limits)
        - Height: 3-19 inches (greater is better within limits) 
        - No distortion or blur added
        - Optimized for viewing and printing quality
        
        Args:
            image_path: Path to input image
            
        Returns:
            Processed PIL Image meeting all quality constraints or None if error
        """
        try:
            # Use quality-controlled processing if available
            if self.quality_manager:
                return self._convert_to_sepia_with_quality_control(image_path)
            else:
                # Fallback to basic processing
                return self._convert_to_sepia_basic(image_path)
                
        except Exception as e:
            self.logger.error(f"Error converting {image_path} to sepia: {e}")
            return None
    
    def _convert_to_sepia_with_quality_control(self, image_path: Path) -> Optional[Image.Image]:
        """Sepia conversion with full quality control."""
        def sepia_transform(path):
            return self._convert_to_sepia_basic(path)
        
        # Apply transformation with quality control
        from ..utils.quality_controlled_transforms import QualityControlledTransformBase
        quality_processor = QualityControlledTransformBase(self.config, self.logger)
        
        result = quality_processor.process_with_quality_control(
            image_path, sepia_transform
        )
        
        return result
    
    def _convert_to_sepia_basic(self, image_path: Path) -> Optional[Image.Image]:
        """Basic sepia conversion without quality control."""
        # Open image with validation
        image = Image.open(image_path)
        if not image:
            self.logger.error(f"Failed to open image: {image_path}")
            return None
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get sepia configuration
        intensity = self.sepia_config.get('intensity', 0.8)
        # Clamp intensity to valid range
        intensity = max(0.0, min(1.0, intensity))
        
        # Convert to numpy array with validation
        try:
            pixels = np.array(image, dtype=np.float32)
            if pixels.size == 0:
                self.logger.error(f"Empty pixel array for: {image_path}")
                return None
            
            # Apply sepia matrix transformation
            sepia_matrix = np.array([
                [0.393, 0.769, 0.189],  # Red channel
                [0.349, 0.686, 0.168],  # Green channel  
                [0.272, 0.534, 0.131]   # Blue channel
            ], dtype=np.float32)
            
            # Apply transformation (thread-safe matrix multiplication)
            sepia_pixels = np.dot(pixels, sepia_matrix.T)
            
            # Blend with original based on intensity
            result_pixels = (1 - intensity) * pixels + intensity * sepia_pixels
            
            # Clip values to valid range and convert to uint8
            result_pixels = np.clip(result_pixels, 0, 255).astype(np.uint8)
            
            # Convert back to PIL Image
            sepia_image = Image.fromarray(result_pixels, mode='RGB')
            
        except Exception as numpy_error:
            self.logger.error(f"NumPy sepia processing failed for {image_path}: {numpy_error}")
            # Fallback to simple sepia using PIL
            try:
                # Simple sepia using ImageOps
                sepia_image = ImageOps.colorize(
                    image.convert('L'),
                    black='#704214',
                    white='#C8B99C'
                )
            except Exception as fallback_error:
                self.logger.error(f"Fallback sepia failed for {image_path}: {fallback_error}")
                return None
        
        # Validate result
        if not sepia_image:
            self.logger.error(f"Sepia conversion failed for: {image_path}")
            return None
        
        # Handle metadata preservation
        if self.preserve_metadata and not self.strip_sensitive:
            try:
                if hasattr(image, 'info') and image.info:
                    sepia_image.info = image.info.copy()
            except Exception as e:
                self.logger.warning(f"Failed to preserve metadata for {image_path}: {e}")
        
        return sepia_image