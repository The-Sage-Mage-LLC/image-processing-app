# -*- coding: utf-8 -*-
"""
Image Quality Constraints and Resolution Management Module
Project ID: Image Processing App 20251119
Created: 2025-01-25
Author: The-Sage-Mage

This module ensures that ALL image transformations meet the specified quality requirements:
- Minimum resolution: 256 pixels/inch (higher is better)
- Width constraints: 3-19 inches (greater is better within limits)
- Height constraints: 3-19 inches (greater is better within limits)
- Optimal quality for viewing and printing
- No distortion or blur introduction
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass


@dataclass
class ImageQualityConstraints:
    """Image quality constraints configuration."""
    min_dpi: int = 256
    max_dpi: int = 600
    min_width_inches: float = 3.0
    max_width_inches: float = 19.0
    min_height_inches: float = 3.0
    max_height_inches: float = 19.0
    target_dpi: int = 300
    preserve_aspect_ratio: bool = True
    prevent_distortion: bool = True
    prevent_blur: bool = True
    optimize_for_printing: bool = True


@dataclass
class ImageMetrics:
    """Current image metrics."""
    width_pixels: int
    height_pixels: int
    dpi: float
    width_inches: float
    height_inches: float
    aspect_ratio: float
    total_pixels: int
    meets_constraints: bool
    issues: list


class ImageQualityManager:
    """Manages image quality constraints and transformations."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Load constraints from config or use defaults
        constraint_config = config.get('image_quality', {})
        self.constraints = ImageQualityConstraints(
            min_dpi=constraint_config.get('min_dpi', 256),
            max_dpi=constraint_config.get('max_dpi', 600),
            min_width_inches=constraint_config.get('min_width_inches', 3.0),
            max_width_inches=constraint_config.get('max_width_inches', 19.0),
            min_height_inches=constraint_config.get('min_height_inches', 3.0),
            max_height_inches=constraint_config.get('max_height_inches', 19.0),
            target_dpi=constraint_config.get('target_dpi', 300),
            preserve_aspect_ratio=constraint_config.get('preserve_aspect_ratio', True),
            prevent_distortion=constraint_config.get('prevent_distortion', True),
            prevent_blur=constraint_config.get('prevent_blur', True),
            optimize_for_printing=constraint_config.get('optimize_for_printing', True)
        )
        
        self.logger.info(f"Image Quality Manager initialized:")
        self.logger.info(f"  DPI range: {self.constraints.min_dpi}-{self.constraints.max_dpi} (target: {self.constraints.target_dpi})")
        self.logger.info(f"  Width range: {self.constraints.min_width_inches}-{self.constraints.max_width_inches} inches")
        self.logger.info(f"  Height range: {self.constraints.min_height_inches}-{self.constraints.max_height_inches} inches")
    
    def analyze_image_metrics(self, image_path: Path) -> ImageMetrics:
        """Analyze current image metrics against constraints."""
        try:
            # Read image with PIL to get DPI information
            with Image.open(image_path) as pil_image:
                width_pixels, height_pixels = pil_image.size
                
                # Get DPI (PIL stores as tuple, we want the actual DPI)
                dpi_info = pil_image.info.get('dpi', (72, 72))
                dpi = float(dpi_info[0] if isinstance(dpi_info, tuple) else dpi_info)
                
                # Calculate physical dimensions
                width_inches = width_pixels / dpi
                height_inches = height_pixels / dpi
                aspect_ratio = width_pixels / height_pixels
                total_pixels = width_pixels * height_pixels
                
                # Check constraints
                issues = []
                
                if dpi < self.constraints.min_dpi:
                    issues.append(f"DPI too low: {dpi:.1f} < {self.constraints.min_dpi}")
                
                if width_inches < self.constraints.min_width_inches:
                    issues.append(f"Width too small: {width_inches:.2f} < {self.constraints.min_width_inches}")
                elif width_inches > self.constraints.max_width_inches:
                    issues.append(f"Width too large: {width_inches:.2f} > {self.constraints.max_width_inches}")
                
                if height_inches < self.constraints.min_height_inches:
                    issues.append(f"Height too small: {height_inches:.2f} < {self.constraints.min_height_inches}")
                elif height_inches > self.constraints.max_height_inches:
                    issues.append(f"Height too large: {height_inches:.2f} > {self.constraints.max_height_inches}")
                
                meets_constraints = len(issues) == 0
                
                return ImageMetrics(
                    width_pixels=width_pixels,
                    height_pixels=height_pixels,
                    dpi=dpi,
                    width_inches=width_inches,
                    height_inches=height_inches,
                    aspect_ratio=aspect_ratio,
                    total_pixels=total_pixels,
                    meets_constraints=meets_constraints,
                    issues=issues
                )
        
        except Exception as e:
            self.logger.error(f"Error analyzing image metrics for {image_path}: {e}")
            return ImageMetrics(
                width_pixels=0, height_pixels=0, dpi=72.0,
                width_inches=0.0, height_inches=0.0, aspect_ratio=1.0,
                total_pixels=0, meets_constraints=False,
                issues=[f"Analysis failed: {e}"]
            )
    
    def calculate_optimal_dimensions(self, current_metrics: ImageMetrics) -> Tuple[int, int, float]:
        """
        Calculate optimal dimensions that meet all constraints.
        
        Returns:
            (optimal_width_pixels, optimal_height_pixels, optimal_dpi)
        """
        # Start with current dimensions
        width_pixels = current_metrics.width_pixels
        height_pixels = current_metrics.height_pixels
        current_dpi = current_metrics.dpi
        aspect_ratio = current_metrics.aspect_ratio
        
        # Determine target DPI
        target_dpi = self.constraints.target_dpi
        
        # If current DPI is below minimum, scale up
        if current_dpi < self.constraints.min_dpi:
            scale_factor = self.constraints.min_dpi / current_dpi
            width_pixels = int(width_pixels * scale_factor)
            height_pixels = int(height_pixels * scale_factor)
            target_dpi = self.constraints.min_dpi
            self.logger.info(f"Upscaling image: DPI {current_dpi:.1f} -> {target_dpi}")
        
        # Check physical size constraints at target DPI
        width_inches = width_pixels / target_dpi
        height_inches = height_pixels / target_dpi
        
        # Adjust if physical dimensions are outside constraints
        needs_adjustment = False
        
        if width_inches < self.constraints.min_width_inches:
            width_pixels = int(self.constraints.min_width_inches * target_dpi)
            if self.constraints.preserve_aspect_ratio:
                height_pixels = int(width_pixels / aspect_ratio)
            needs_adjustment = True
        elif width_inches > self.constraints.max_width_inches:
            width_pixels = int(self.constraints.max_width_inches * target_dpi)
            if self.constraints.preserve_aspect_ratio:
                height_pixels = int(width_pixels / aspect_ratio)
            needs_adjustment = True
        
        if height_inches < self.constraints.min_height_inches:
            height_pixels = int(self.constraints.min_height_inches * target_dpi)
            if self.constraints.preserve_aspect_ratio:
                width_pixels = int(height_pixels * aspect_ratio)
            needs_adjustment = True
        elif height_inches > self.constraints.max_height_inches:
            height_pixels = int(self.constraints.max_height_inches * target_dpi)
            if self.constraints.preserve_aspect_ratio:
                width_pixels = int(height_pixels * aspect_ratio)
            needs_adjustment = True
        
        if needs_adjustment:
            # Recalculate physical dimensions
            width_inches = width_pixels / target_dpi
            height_inches = height_pixels / target_dpi
            self.logger.info(f"Adjusted dimensions: {width_pixels}x{height_pixels} pixels ({width_inches:.2f}x{height_inches:.2f} @ {target_dpi} DPI)")
        
        return width_pixels, height_pixels, target_dpi
    
    def apply_quality_constraints(self, image: np.ndarray, source_metrics: ImageMetrics, 
                                  transformation_type: str = "standard") -> np.ndarray:
        """
        Apply quality constraints to ensure output meets requirements.
        
        Args:
            image: Input image as numpy array
            source_metrics: Original image metrics
            transformation_type: Type of transformation being applied
            
        Returns:
            Processed image meeting all quality constraints
        """
        try:
            # Calculate optimal dimensions
            target_width, target_height, target_dpi = self.calculate_optimal_dimensions(source_metrics)
            current_height, current_width = image.shape[:2]
            
            # Check if resizing is needed
            if current_width != target_width or current_height != target_height:
                self.logger.info(f"Resizing image for quality constraints: {current_width}x{current_height} -> {target_width}x{target_height}")
                
                # Choose interpolation method based on scale direction
                if target_width > current_width or target_height > current_height:
                    # Upscaling - use high-quality interpolation
                    if self.constraints.prevent_blur:
                        interpolation = cv2.INTER_LANCZOS4
                    else:
                        interpolation = cv2.INTER_CUBIC
                else:
                    # Downscaling - use area interpolation to prevent aliasing
                    interpolation = cv2.INTER_AREA
                
                # Apply resize
                resized_image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
                
                # Verify no distortion introduced
                if self.constraints.prevent_distortion:
                    original_aspect = current_width / current_height
                    new_aspect = target_width / target_height
                    aspect_diff = abs(original_aspect - new_aspect) / original_aspect
                    
                    if aspect_diff > 0.02:  # More than 2% difference
                        self.logger.warning(f"Aspect ratio changed by {aspect_diff*100:.1f}% - potential distortion")
                
                return resized_image
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error applying quality constraints: {e}")
            return image
    
    def optimize_for_print_quality(self, image: np.ndarray) -> np.ndarray:
        """Apply additional optimizations for print quality."""
        if not self.constraints.optimize_for_printing:
            return image
        
        try:
            # Apply subtle sharpening for print quality
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend with original to avoid over-sharpening
            result = cv2.addWeighted(image, 0.8, sharpened, 0.2, 0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing for print quality: {e}")
            return image
    
    def save_with_quality_metadata(self, image: np.ndarray, output_path: Path, 
                                   target_dpi: float, quality: int = 95) -> bool:
        """
        Save image with proper DPI metadata and quality settings.
        
        Args:
            image: Image to save
            output_path: Output file path  
            target_dpi: Target DPI for the image
            quality: JPEG quality (1-100)
            
        Returns:
            True if saved successfully
        """
        try:
            # Convert OpenCV image to PIL
            if len(image.shape) == 3:
                # Color image - convert BGR to RGB
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                # Grayscale image
                pil_image = Image.fromarray(image)
            
            # Prepare save parameters
            save_kwargs = {
                'dpi': (int(target_dpi), int(target_dpi)),
                'optimize': True
            }
            
            # Format-specific parameters
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                save_kwargs['quality'] = quality
                save_kwargs['progressive'] = True
            elif output_path.suffix.lower() == '.png':
                save_kwargs['compress_level'] = 6
            
            # Save with metadata
            pil_image.save(output_path, **save_kwargs)
            
            # Verify the save
            if output_path.exists() and output_path.stat().st_size > 0:
                self.logger.debug(f"Saved image with {target_dpi} DPI: {output_path.name}")
                return True
            else:
                self.logger.error(f"Failed to save image: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving image with quality metadata: {e}")
            return False
    
    def validate_output_quality(self, output_path: Path) -> Dict[str, Any]:
        """
        Validate that the output image meets all quality requirements.
        
        Returns:
            Dictionary with validation results
        """
        try:
            output_metrics = self.analyze_image_metrics(output_path)
            
            validation_results = {
                'meets_all_constraints': output_metrics.meets_constraints,
                'metrics': {
                    'width_pixels': output_metrics.width_pixels,
                    'height_pixels': output_metrics.height_pixels,
                    'width_inches': output_metrics.width_inches,
                    'height_inches': output_metrics.height_inches,
                    'dpi': output_metrics.dpi,
                    'aspect_ratio': output_metrics.aspect_ratio,
                    'total_pixels': output_metrics.total_pixels
                },
                'constraints_check': {
                    'dpi_ok': output_metrics.dpi >= self.constraints.min_dpi,
                    'width_ok': (self.constraints.min_width_inches <= 
                               output_metrics.width_inches <= 
                               self.constraints.max_width_inches),
                    'height_ok': (self.constraints.min_height_inches <= 
                                output_metrics.height_inches <= 
                                self.constraints.max_height_inches),
                },
                'issues': output_metrics.issues
            }
            
            if output_metrics.meets_constraints:
                self.logger.info(f"Output quality validation PASSED: {output_path.name}")
                self.logger.info(f"  Dimensions: {output_metrics.width_inches:.2f}x{output_metrics.height_inches:.2f} @ {output_metrics.dpi:.0f} DPI")
            else:
                self.logger.warning(f"Output quality validation FAILED: {output_path.name}")
                for issue in output_metrics.issues:
                    self.logger.warning(f"  Issue: {issue}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating output quality: {e}")
            return {
                'meets_all_constraints': False,
                'error': str(e)
            }


def apply_quality_constraints_to_transform(transform_func):
    """
    Decorator to apply quality constraints to any image transformation function.
    
    Usage:
        @apply_quality_constraints_to_transform
        def convert_to_grayscale(self, image_path: Path) -> Optional[Image.Image]:
            # ... transformation code ...
    """
    def wrapper(self, image_path: Path, *args, **kwargs):
        # Check if quality manager is available
        quality_manager = getattr(self, 'quality_manager', None)
        if not quality_manager:
            # Fall back to original function if no quality manager
            return transform_func(self, image_path, *args, **kwargs)
        
        try:
            # Analyze source image
            source_metrics = quality_manager.analyze_image_metrics(image_path)
            quality_manager.logger.info(f"Source: {image_path.name} - {source_metrics.width_inches:.2f}x{source_metrics.height_inches:.2f} @ {source_metrics.dpi:.0f} DPI")
            
            # Apply original transformation
            result = transform_func(self, image_path, *args, **kwargs)
            
            if result is None:
                return None
            
            # Convert PIL to numpy if needed
            if isinstance(result, Image.Image):
                if result.mode == 'RGB':
                    result_array = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                else:
                    result_array = np.array(result)
            else:
                result_array = result
            
            # Apply quality constraints
            constrained_result = quality_manager.apply_quality_constraints(
                result_array, source_metrics, transform_func.__name__
            )
            
            # Apply print optimizations
            optimized_result = quality_manager.optimize_for_print_quality(constrained_result)
            
            # Convert back to PIL if original was PIL
            if isinstance(result, Image.Image):
                if len(optimized_result.shape) == 3:
                    # Color image
                    final_result = Image.fromarray(cv2.cvtColor(optimized_result, cv2.COLOR_BGR2RGB))
                else:
                    # Grayscale
                    final_result = Image.fromarray(optimized_result)
            else:
                final_result = optimized_result
            
            return final_result
            
        except Exception as e:
            quality_manager.logger.error(f"Error applying quality constraints: {e}")
            # Fall back to original result
            return transform_func(self, image_path, *args, **kwargs)
    
    return wrapper