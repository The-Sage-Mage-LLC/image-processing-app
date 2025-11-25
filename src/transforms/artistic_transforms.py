"""
Artistic Image Transformations
Project ID: Image Processing App 20251119
Created: 2025-11-19 06:52:45 UTC
Author: The-Sage-Mage
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from pathlib import Path
from typing import Optional, Tuple
import logging


class ArtisticTransforms:
    """Artistic and activity book image transformations with thread safety."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Configure OpenCV for thread safety
        cv2.setNumThreads(1)
        
        # Load configuration settings
        self.pencil_config = config.get('pencil_sketch', {})
        self.coloring_config = config.get('coloring_book', {})
        self.dots_config = config.get('connect_the_dots', {})
        self.color_numbers_config = config.get('color_by_numbers', {})
        
        self.logger.debug("ArtisticTransforms initialized with thread safety")
    
    def convert_to_pencil_sketch(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Convert image to pencil sketch style with thread safety.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Processed image as numpy array or None if error
        """
        try:
            # Read image with error handling
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to read image: {image_path}")
                return None
            
            # Ensure we have a valid image array
            if not isinstance(image, np.ndarray) or image.size == 0:
                self.logger.error(f"Invalid image array for: {image_path}")
                return None
            
            # Get configuration
            radius = self.pencil_config.get('radius', 15)
            strength = self.pencil_config.get('strength', 0.5)
            clarity = self.pencil_config.get('clarity', 0.8)
            blur_amount = self.pencil_config.get('blur_amount', 0.2)
            
            # Create copies to avoid modifying original arrays
            working_image = image.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
            
            # Invert the grayscale image
            inverted = 255 - gray
            
            # Apply Gaussian blur with broad pencil tip simulation
            kernel_size = radius * 2 + 1
            blurred = cv2.GaussianBlur(inverted, (kernel_size, kernel_size), 0)
            
            # Invert the blurred image
            inverted_blur = 255 - blurred
            
            # Create the sketch by dividing (thread-safe)
            # Add small epsilon to avoid division by zero
            inverted_blur_safe = np.where(inverted_blur == 0, 1, inverted_blur)
            sketch = cv2.divide(gray, inverted_blur_safe, scale=256.0)
            
            # Apply clarity enhancement
            if clarity > 0:
                # Enhance edges for more clarity
                edges = cv2.Canny(gray, 50, 150)
                edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
                
                # Blend edges with sketch
                sketch = cv2.addWeighted(
                    sketch, 1 - clarity * 0.3,
                    255 - edges_dilated, clarity * 0.3,
                    0
                )
            
            # Apply strength adjustment
            if strength != 1.0:
                # Adjust contrast based on strength
                sketch = cv2.convertScaleAbs(sketch, alpha=strength, beta=128 * (1 - strength))
            
            # Apply subtle blur if configured
            if blur_amount > 0:
                blur_kernel = int(blur_amount * 5) * 2 + 1
                sketch = cv2.GaussianBlur(sketch, (blur_kernel, blur_kernel), 0)
            
            # Ensure output is valid
            if not isinstance(sketch, np.ndarray) or sketch.size == 0:
                self.logger.error(f"Invalid sketch output for: {image_path}")
                return None
                
            return sketch
            
        except Exception as e:
            self.logger.error(f"Error converting {image_path} to pencil sketch: {e}")
            return None
    
    def convert_to_coloring_book(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Convert image to coloring book style with strong outlines and thread safety.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Processed image as numpy array or None if error
        """
        try:
            # Read image with validation
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to read image: {image_path}")
                return None
                
            # Validate image array
            if not isinstance(image, np.ndarray) or image.size == 0:
                self.logger.error(f"Invalid image array for: {image_path}")
                return None
            
            # Get configuration
            method = self.coloring_config.get('edge_detection_method', 'canny')
            lower_threshold = self.coloring_config.get('lower_threshold', 50)
            upper_threshold = self.coloring_config.get('upper_threshold', 150)
            line_thickness = self.coloring_config.get('line_thickness', 2)
            simplification_level = self.coloring_config.get('simplification_level', 3)
            
            # Create working copy
            working_image = image.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter for noise reduction while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Simplify image to reduce complexity
            if simplification_level > 0:
                # Quantize the image to reduce number of gray levels
                n_levels = max(2, 10 - simplification_level * 2)
                quantized = (filtered // (256 // n_levels)) * (256 // n_levels)
                filtered = quantized.astype(np.uint8)
            
            # Detect edges using configured method
            if method == 'canny':
                edges = cv2.Canny(filtered, lower_threshold, upper_threshold)
            elif method == 'laplacian':
                laplacian = cv2.Laplacian(filtered, cv2.CV_64F)
                edges = np.uint8(np.absolute(laplacian))
                _, edges = cv2.threshold(edges, lower_threshold, 255, cv2.THRESH_BINARY)
            elif method == 'sobel':
                sobelx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
                # Thread-safe sqrt operation
                magnitude = np.sqrt(np.add(np.square(sobelx), np.square(sobely)))
                edges = np.uint8(magnitude / magnitude.max() * 255)
                _, edges = cv2.threshold(edges, lower_threshold, 255, cv2.THRESH_BINARY)
            else:
                edges = cv2.Canny(filtered, lower_threshold, upper_threshold)
            
            # Apply morphological operations to clean up edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Dilate edges to make lines thicker
            if line_thickness > 1:
                kernel = np.ones((line_thickness, line_thickness), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find and draw contours for additional detail
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create white background with same shape as gray
            result = np.full_like(gray, 255, dtype=np.uint8)
            
            # Draw edges (black lines on white background)
            result[edges > 0] = 0
            
            # Optionally draw simplified contours
            if len(contours) > 0:
                # Filter small contours
                min_area = gray.shape[0] * gray.shape[1] * 0.001
                filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
                
                # Draw contours
                if filtered_contours:
                    cv2.drawContours(result, filtered_contours, -1, 0, line_thickness)
            
            # Validate result
            if not isinstance(result, np.ndarray) or result.size == 0:
                self.logger.error(f"Invalid coloring book output for: {image_path}")
                return None
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error converting {image_path} to coloring book: {e}")
            return None
    
    def convert_to_connect_dots(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Convert image to connect-the-dots style.
        Full implementation for Phase 4.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Processed image as numpy array or None if error
        """
        try:
            from .activity_transforms import ActivityTransforms
            
            # Initialize activity transforms
            activity = ActivityTransforms(self.config, self.logger)
            
            # Process the image
            return activity.convert_to_connect_dots(image_path)
            
        except ImportError as e:
            self.logger.error(f"ActivityTransforms not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error in connect_dots conversion: {e}")
            return None

    def convert_to_color_by_numbers(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Convert image to color-by-numbers style.
        Full implementation for Phase 4.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Processed image as numpy array or None if error
        """
        try:
            from .activity_transforms import ActivityTransforms
            
            # Initialize activity transforms
            activity = ActivityTransforms(self.config, self.logger)
            
            # Process the image
            return activity.convert_to_color_by_numbers(image_path)
            
        except ImportError as e:
            self.logger.error(f"ActivityTransforms not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error in color_by_numbers conversion: {e}")
            return None
