# -*- coding: utf-8 -*-
"""
Quality-Controlled Transform Base Classes
Project ID: Image Processing App 20251119
Created: 2025-01-25
Author: The-Sage-Mage

Base classes that integrate image quality management into all transformations.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Union
import logging

from .image_quality_manager import ImageQualityManager


class QualityControlledTransformBase:
    """Base class for all image transformations with quality control integration."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize quality manager
        self.quality_manager = ImageQualityManager(config, logger)
        
        self.logger.info(f"Quality-controlled {self.__class__.__name__} initialized")
    
    def process_with_quality_control(self, image_path: Path, transform_func, 
                                   output_path: Optional[Path] = None, 
                                   *args, **kwargs) -> Optional[Union[Image.Image, np.ndarray]]:
        """
        Process image with automatic quality control integration.
        
        Args:
            image_path: Source image path
            transform_func: Transformation function to apply
            output_path: Optional output path for saving
            *args, **kwargs: Additional arguments for transform function
            
        Returns:
            Processed image meeting quality constraints
        """
        try:
            # Analyze source image metrics
            source_metrics = self.quality_manager.analyze_image_metrics(image_path)
            
            if not source_metrics.meets_constraints:
                self.logger.warning(f"Source image quality issues for {image_path.name}:")
                for issue in source_metrics.issues:
                    self.logger.warning(f"  - {issue}")
            
            self.logger.info(f"Processing: {image_path.name}")
            self.logger.info(f"  Source: {source_metrics.width_inches:.2f}\"×{source_metrics.height_inches:.2f}\" @ {source_metrics.dpi:.0f} DPI")
            
            # Apply the transformation
            result = transform_func(image_path, *args, **kwargs)
            
            if result is None:
                self.logger.error(f"Transformation failed for {image_path.name}")
                return None
            
            # Convert to numpy array for quality processing
            if isinstance(result, Image.Image):
                if result.mode == 'RGB':
                    result_array = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                elif result.mode == 'L':
                    result_array = np.array(result)
                else:
                    # Convert other modes to RGB first
                    result_rgb = result.convert('RGB')
                    result_array = cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)
                was_pil = True
                original_mode = result.mode
            else:
                result_array = result
                was_pil = False
                original_mode = 'BGR' if len(result.shape) == 3 else 'L'
            
            # Apply quality constraints
            self.logger.debug("Applying quality constraints...")
            constrained_result = self.quality_manager.apply_quality_constraints(
                result_array, source_metrics, transform_func.__name__
            )
            
            # Apply print optimizations
            self.logger.debug("Optimizing for print quality...")
            optimized_result = self.quality_manager.optimize_for_print_quality(constrained_result)
            
            # Calculate target DPI for output
            _, _, target_dpi = self.quality_manager.calculate_optimal_dimensions(source_metrics)
            
            # Save with quality metadata if output path provided
            if output_path:
                success = self.quality_manager.save_with_quality_metadata(
                    optimized_result, output_path, target_dpi
                )
                
                if success:
                    # Validate output quality
                    validation = self.quality_manager.validate_output_quality(output_path)
                    
                    if validation['meets_all_constraints']:
                        self.logger.info(f"? Quality validation PASSED for {output_path.name}")
                        metrics = validation['metrics']
                        self.logger.info(f"  Output: {metrics['width_inches']:.2f}\"×{metrics['height_inches']:.2f}\" @ {metrics['dpi']:.0f} DPI")
                    else:
                        self.logger.warning(f"? Quality validation FAILED for {output_path.name}")
                        if 'issues' in validation:
                            for issue in validation['issues']:
                                self.logger.warning(f"  Issue: {issue}")
                else:
                    self.logger.error(f"Failed to save {output_path.name}")
            
            # Convert back to original format if needed
            if was_pil:
                if len(optimized_result.shape) == 3 and original_mode != 'L':
                    # Color image
                    final_result = Image.fromarray(cv2.cvtColor(optimized_result, cv2.COLOR_BGR2RGB))
                    if original_mode != 'RGB':
                        final_result = final_result.convert(original_mode)
                else:
                    # Grayscale
                    final_result = Image.fromarray(optimized_result)
                    if original_mode == 'L':
                        final_result = final_result.convert('L')
            else:
                final_result = optimized_result
            
            self.logger.info(f"? Processing completed for {image_path.name}")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in quality-controlled processing of {image_path}: {e}")
            return None


class EnhancedBasicTransforms(QualityControlledTransformBase):
    """Enhanced basic transforms with quality control."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)
        
        # Load configuration settings with quality-aware defaults
        self.grayscale_config = config.get('grayscale', {})
        self.sepia_config = config.get('sepia', {})
        self.preserve_metadata = config.get('basic_transforms', {}).get('preserve_metadata', True)
        self.strip_sensitive = config.get('processing', {}).get('strip_sensitive_metadata', True)
    
    def convert_to_grayscale(self, image_path: Path, output_path: Optional[Path] = None) -> Optional[Image.Image]:
        """Convert image to grayscale with quality control."""
        def grayscale_transform(path):
            try:
                image = Image.open(path)
                if not image:
                    return None
                
                # Convert to RGB if necessary
                if image.mode not in ('RGB', 'L'):
                    image = image.convert('RGB')
                
                # Get conversion method
                method = self.grayscale_config.get('method', 'luminosity')
                
                if method == 'average':
                    grayscale = image.convert('L')
                elif method == 'luminosity':
                    grayscale = image.convert('L')
                elif method == 'desaturation':
                    if image.mode == 'RGB':
                        np_image = np.array(image)
                        max_rgb = np.max(np_image, axis=2)
                        min_rgb = np.min(np_image, axis=2)
                        grayscale_array = ((max_rgb + min_rgb) / 2).astype(np.uint8)
                        grayscale = Image.fromarray(grayscale_array, mode='L')
                    else:
                        grayscale = image
                else:
                    grayscale = image.convert('L')
                
                # Handle metadata preservation
                if self.preserve_metadata and not self.strip_sensitive:
                    try:
                        if hasattr(image, 'info') and image.info:
                            grayscale.info = image.info.copy()
                    except Exception as e:
                        self.logger.warning(f"Failed to preserve metadata: {e}")
                
                return grayscale
                
            except Exception as e:
                self.logger.error(f"Grayscale transformation error: {e}")
                return None
        
        return self.process_with_quality_control(image_path, grayscale_transform, output_path)
    
    def convert_to_sepia(self, image_path: Path, output_path: Optional[Path] = None) -> Optional[Image.Image]:
        """Convert image to sepia with quality control."""
        def sepia_transform(path):
            try:
                image = Image.open(path)
                if not image:
                    return None
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Get sepia configuration
                intensity = self.sepia_config.get('intensity', 0.8)
                intensity = max(0.0, min(1.0, intensity))
                
                # Apply sepia transformation
                pixels = np.array(image, dtype=np.float32)
                
                sepia_matrix = np.array([
                    [0.393, 0.769, 0.189],
                    [0.349, 0.686, 0.168],
                    [0.272, 0.534, 0.131]
                ], dtype=np.float32)
                
                sepia_pixels = np.dot(pixels, sepia_matrix.T)
                result_pixels = (1 - intensity) * pixels + intensity * sepia_pixels
                result_pixels = np.clip(result_pixels, 0, 255).astype(np.uint8)
                
                sepia_image = Image.fromarray(result_pixels, mode='RGB')
                
                # Handle metadata preservation
                if self.preserve_metadata and not self.strip_sensitive:
                    try:
                        if hasattr(image, 'info') and image.info:
                            sepia_image.info = image.info.copy()
                    except Exception as e:
                        self.logger.warning(f"Failed to preserve metadata: {e}")
                
                return sepia_image
                
            except Exception as e:
                self.logger.error(f"Sepia transformation error: {e}")
                return None
        
        return self.process_with_quality_control(image_path, sepia_transform, output_path)