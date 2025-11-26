"""
Blur Detection Module using Multiple Methods
Project ID: Image Processing App 20251119
Created: 2025-11-19 07:08:42 UTC
Author: The-Sage-Mage
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime


class BlurDetector:
    """Detects blur in images using multiple algorithms with thread safety."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Configure OpenCV for thread safety
        cv2.setNumThreads(1)
        
        # Load blur detection configuration
        self.blur_config = config.get('blur_detection', {})
        self.enabled_models = self.blur_config.get('models', ['laplacian', 'variance_of_laplacian', 'gradient_magnitude'])
        self.center_weight = self.blur_config.get('center_weight', 1.5)
        self.peripheral_weight = self.blur_config.get('peripheral_weight', 0.5)
        
        # Thresholds for each method
        self.thresholds = {
            'laplacian': self.blur_config.get('blur_threshold_laplacian', 100.0),
            'variance_of_laplacian': self.blur_config.get('blur_threshold_variance', 50.0),
            'gradient_magnitude': self.blur_config.get('blur_threshold_gradient', 75.0),
            'tenengrad': self.blur_config.get('blur_threshold_tenengrad', 500.0),
            'brenner': self.blur_config.get('blur_threshold_brenner', 10.0)
        }
        
        self.consensus_threshold = self.blur_config.get('consensus_threshold', 2)
        
        self.logger.debug("BlurDetector initialized with thread safety")
        
    def detect_blur(self, image_path: Path) -> Dict[str, Any]:
        """
        Detect if an image is blurry using multiple methods with thread safety.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing blur detection results
        """
        results = {
            'file_path': str(image_path),
            'file_name': image_path.name,
            'is_blurry': False,
            'consensus_score': 0,
            'weighted_score': 0.0,
            'detection_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            # Read image with validation
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to read image: {image_path}")
                results['error'] = 'Failed to read image'
                return results
            
            # Validate image array
            if not isinstance(image, np.ndarray) or image.size == 0:
                self.logger.error(f"Invalid image array for: {image_path}")
                results['error'] = 'Invalid image data'
                return results
            
            # Create working copy to avoid modifying original
            working_image = image.copy()
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
            
            # Validate grayscale conversion
            if not isinstance(gray, np.ndarray) or gray.size == 0:
                self.logger.error(f"Failed to convert to grayscale: {image_path}")
                results['error'] = 'Failed to convert to grayscale'
                return results
            
            # Create weight mask for center vs peripheral
            weight_mask = self._create_weight_mask(gray.shape)
            
            # Run each detection method
            method_results = {}
            blur_votes = 0
            
            if 'laplacian' in self.enabled_models:
                try:
                    score, is_blurry = self._laplacian_variance(gray, weight_mask)
                    method_results['laplacian_score'] = score
                    method_results['laplacian_blurry'] = is_blurry
                    if is_blurry:
                        blur_votes += 1
                except Exception as e:
                    self.logger.warning(f"Laplacian method failed for {image_path}: {e}")
            
            if 'variance_of_laplacian' in self.enabled_models:
                try:
                    score, is_blurry = self._variance_of_laplacian(gray, weight_mask)
                    method_results['variance_score'] = score
                    method_results['variance_blurry'] = is_blurry
                    if is_blurry:
                        blur_votes += 1
                except Exception as e:
                    self.logger.warning(f"Variance of Laplacian method failed for {image_path}: {e}")
            
            if 'gradient_magnitude' in self.enabled_models:
                try:
                    score, is_blurry = self._gradient_magnitude(gray, weight_mask)
                    method_results['gradient_score'] = score
                    method_results['gradient_blurry'] = is_blurry
                    if is_blurry:
                        blur_votes += 1
                except Exception as e:
                    self.logger.warning(f"Gradient magnitude method failed for {image_path}: {e}")
            
            if 'tenengrad' in self.enabled_models:
                try:
                    score, is_blurry = self._tenengrad(gray, weight_mask)
                    method_results['tenengrad_score'] = score
                    method_results['tenengrad_blurry'] = is_blurry
                    if is_blurry:
                        blur_votes += 1
                except Exception as e:
                    self.logger.warning(f"Tenengrad method failed for {image_path}: {e}")
            
            if 'brenner' in self.enabled_models:
                try:
                    score, is_blurry = self._brenner_gradient(gray, weight_mask)
                    method_results['brenner_score'] = score
                    method_results['brenner_blurry'] = is_blurry
                    if is_blurry:
                        blur_votes += 1
                except Exception as e:
                    self.logger.warning(f"Brenner gradient method failed for {image_path}: {e}")
            
            # Update results with method scores
            results.update(method_results)
            
            # Calculate consensus
            results['consensus_score'] = blur_votes
            results['total_models'] = len([m for m in self.enabled_models if m in method_results])
            
            # Determine if image is blurry based on consensus
            if results['total_models'] > 0:
                results['is_blurry'] = blur_votes >= self.consensus_threshold
                results['blur_percentage'] = (blur_votes / results['total_models']) * 100
                
                # Calculate weighted average score (thread-safe)
                scores = []
                if 'laplacian_score' in method_results and np.isfinite(method_results['laplacian_score']):
                    scores.append(method_results['laplacian_score'] / self.thresholds['laplacian'])
                if 'variance_score' in method_results and np.isfinite(method_results['variance_score']):
                    scores.append(method_results['variance_score'] / self.thresholds['variance_of_laplacian'])
                if 'gradient_score' in method_results and np.isfinite(method_results['gradient_score']):
                    scores.append(method_results['gradient_score'] / self.thresholds['gradient_magnitude'])
                
                if scores:
                    results['weighted_score'] = float(np.mean(scores)) * 100  # Convert to percentage
            
            # Add image properties
            results['image_width'] = image.shape[1]
            results['image_height'] = image.shape[0]
            
        except Exception as e:
            self.logger.error(f"Error detecting blur in {image_path}: {e}")
            results['error'] = str(e)
        
        return results
    
    def _create_weight_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a weight mask that gives more importance to the center of the image with thread safety.
        
        Args:
            shape: Image shape (height, width)
            
        Returns:
            Weight mask array
        """
        try:
            h, w = shape
            center_y, center_x = h // 2, w // 2
            
            # Create coordinate grids
            y, x = np.ogrid[:h, :w]
            
            # Calculate distance from center (thread-safe)
            distance = np.sqrt(np.add(np.square(x - center_x), np.square(y - center_y)))
            
            # Normalize distance
            max_distance = np.sqrt(center_x**2 + center_y**2)
            if max_distance == 0:
                # Handle edge case for single pixel images
                return np.ones(shape, dtype=np.float32)
                
            normalized_distance = distance / max_distance
            
            # Create weight mask (higher weight in center)
            weight = self.peripheral_weight + (self.center_weight - self.peripheral_weight) * (1 - normalized_distance)
            
            return weight.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Failed to create weight mask: {e}, using uniform weights")
            return np.ones(shape, dtype=np.float32)
    
    def _laplacian_variance(self, gray: np.ndarray, weight_mask: np.ndarray) -> Tuple[float, bool]:
        """
        Detect blur using Laplacian variance method with thread safety.
        
        Args:
            gray: Grayscale image
            weight_mask: Weight mask for center emphasis
            
        Returns:
            Tuple of (score, is_blurry)
        """
        try:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Ensure arrays are compatible
            if laplacian.shape != weight_mask.shape:
                weight_mask = cv2.resize(weight_mask, (laplacian.shape[1], laplacian.shape[0]))
            
            weighted_laplacian = laplacian * weight_mask
            score = float(weighted_laplacian.var())
            
            # Ensure finite result
            if not np.isfinite(score):
                score = 0.0
                
            is_blurry = score < self.thresholds['laplacian']
            
            return score, is_blurry
            
        except Exception as e:
            self.logger.error(f"Laplacian variance calculation failed: {e}")
            return 0.0, True
    
    def _variance_of_laplacian(self, gray: np.ndarray, weight_mask: np.ndarray) -> Tuple[float, bool]:
        """
        Detect blur using variance of Laplacian method with thread safety.
        
        Args:
            gray: Grayscale image
            weight_mask: Weight mask for center emphasis
            
        Returns:
            Tuple of (score, is_blurry)
        """
        try:
            # Apply different kernel size for variance calculation
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            
            # Ensure arrays are compatible
            if laplacian.shape != weight_mask.shape:
                weight_mask = cv2.resize(weight_mask, (laplacian.shape[1], laplacian.shape[0]))
            
            # Apply weight mask
            weighted_laplacian = np.abs(laplacian) * weight_mask
            
            # Calculate variance in local regions
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
            local_variance = cv2.filter2D(weighted_laplacian, -1, kernel)
            
            score = float(local_variance.mean())
            
            # Ensure finite result
            if not np.isfinite(score):
                score = 0.0
                
            is_blurry = score < self.thresholds['variance_of_laplacian']
            
            return score, is_blurry
            
        except Exception as e:
            self.logger.error(f"Variance of Laplacian calculation failed: {e}")
            return 0.0, True
    
    def _gradient_magnitude(self, gray: np.ndarray, weight_mask: np.ndarray) -> Tuple[float, bool]:
        """
        Detect blur using gradient magnitude method with thread safety.
        
        Args:
            gray: Grayscale image
            weight_mask: Weight mask for center emphasis
            
        Returns:
            Tuple of (score, is_blurry)
        """
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate magnitude (thread-safe)
            magnitude = np.sqrt(np.add(np.square(grad_x), np.square(grad_y)))
            
            # Ensure arrays are compatible
            if magnitude.shape != weight_mask.shape:
                weight_mask = cv2.resize(weight_mask, (magnitude.shape[1], magnitude.shape[0]))
            
            # Apply weight mask
            weighted_magnitude = magnitude * weight_mask
            
            score = float(weighted_magnitude.mean())
            
            # Ensure finite result
            if not np.isfinite(score):
                score = 0.0
                
            is_blurry = score < self.thresholds['gradient_magnitude']
            
            return score, is_blurry
            
        except Exception as e:
            self.logger.error(f"Gradient magnitude calculation failed: {e}")
            return 0.0, True
    
    def _tenengrad(self, gray: np.ndarray, weight_mask: np.ndarray) -> Tuple[float, bool]:
        """
        Detect blur using Tenengrad method with thread safety.
        
        Args:
            gray: Grayscale image
            weight_mask: Weight mask for center emphasis
            
        Returns:
            Tuple of (score, is_blurry)
        """
        try:
            # Sobel operators
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Tenengrad calculation (thread-safe)
            tenengrad = np.sqrt(np.add(np.square(gx), np.square(gy)))
            
            # Ensure arrays are compatible
            if tenengrad.shape != weight_mask.shape:
                weight_mask = cv2.resize(weight_mask, (tenengrad.shape[1], tenengrad.shape[0]))
            
            # Apply weight mask
            weighted_tenengrad = tenengrad * weight_mask
            
            score = float(weighted_tenengrad.sum())
            
            # Ensure finite result
            if not np.isfinite(score):
                score = 0.0
                
            is_blurry = score < self.thresholds.get('tenengrad', 500.0)
            
            return score, is_blurry
            
        except Exception as e:
            self.logger.error(f"Tenengrad calculation failed: {e}")
            return 0.0, True
    
    def _brenner_gradient(self, gray: np.ndarray, weight_mask: np.ndarray) -> Tuple[float, bool]:
        """
        Detect blur using Brenner gradient method with thread safety.
        
        Args:
            gray: Grayscale image
            weight_mask: Weight mask for center emphasis
            
        Returns:
            Tuple of (score, is_blurry)
        """
        try:
            # Brenner gradient (difference between pixel and pixel 2 positions away)
            # Use safe array slicing
            gray_float = gray.astype(np.float32)
            
            if gray_float.shape[1] > 2 and gray_float.shape[0] > 2:
                diff_h = np.abs(gray_float[:, 2:] - gray_float[:, :-2])
                diff_v = np.abs(gray_float[2:, :] - gray_float[:-2, :])
                
                # Pad to match original size
                diff_h = np.pad(diff_h, ((0, 0), (1, 1)), mode='edge')
                diff_v = np.pad(diff_v, ((1, 1), (0, 0)), mode='edge')
                
                # Combine horizontal and vertical differences
                brenner = diff_h + diff_v
            else:
                # Handle small images
                brenner = np.zeros_like(gray_float)
            
            # Ensure arrays are compatible
            if brenner.shape != weight_mask.shape:
                weight_mask = cv2.resize(weight_mask, (brenner.shape[1], brenner.shape[0]))
            
            # Apply weight mask
            weighted_brenner = brenner * weight_mask
            
            score = float(weighted_brenner.mean())
            
            # Ensure finite result
            if not np.isfinite(score):
                score = 0.0
                
            is_blurry = score < self.thresholds.get('brenner', 10.0)
            
            return score, is_blurry
            
        except Exception as e:
            self.logger.error(f"Brenner gradient calculation failed: {e}")
            return 0.0, True
    
    def analyze_blur(self, image_path: Path) -> float:
        """
        Analyze blur in an image and return a simple blur score.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Float blur score (higher = less blurry)
        """
        try:
            # Use the existing detect_blur method
            results = self.detect_blur(image_path)
            
            # Return weighted score, defaulting to 0 if error
            if 'error' in results:
                self.logger.warning(f"Blur detection error for {image_path}: {results['error']}")
                return 0.0
            
            return results.get('weighted_score', 0.0)
            
        except Exception as e:
            self.logger.error(f"Error analyzing blur for {image_path}: {e}")
            return 0.0