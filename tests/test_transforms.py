"""
Test suite for image transformation modules
Project ID: Image Processing App 20251119
Created: 2025-11-19 08:50:29 UTC
Author: The-Sage-Mage
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
import cv2
from PIL import Image
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transforms.basic_transforms import BasicTransforms
from src.transforms.artistic_transforms import ArtisticTransforms
from src.transforms.activity_transforms import ActivityTransforms
from src.utils.logger import setup_logging


class TestBasicTransforms:
    """Test suite for basic image transformations."""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.config = {
            'grayscale': {
                'method': 'luminosity'
            },
            'sepia': {
                'intensity': 0.8,
                'red_factor': 0.393,
                'green_factor': 0.769,
                'blue_factor': 0.189
            },
            'basic_transforms': {
                'jpeg_quality': 95,
                'png_compression': 6,
                'preserve_metadata': False
            },
            'processing': {
                'strip_sensitive_metadata': True
            },
            'logging': {
                'log_to_console': False,
                'log_to_file': False
            }
        }
        
        # Create logger
        cls.logger = setup_logging(cls.test_dir, cls.config)
        
        # Initialize transform classes
        cls.basic_transforms = BasicTransforms(cls.config, cls.logger)
        
        # Create test image
        cls.test_image_path = cls.test_dir / "test_image.jpg"
        cls.create_test_image(cls.test_image_path)
    
    @classmethod
    def teardown_class(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @staticmethod
    def create_test_image(path: Path, size: tuple = (640, 480)):
        """Create a test image with various colors."""
        # Create image with color gradients
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Add red gradient
        for x in range(size[0] // 3):
            img[:, x] = [int(x * 255 / (size[0] // 3)), 0, 0]
        
        # Add green gradient
        for x in range(size[0] // 3, 2 * size[0] // 3):
            img[:, x] = [0, int((x - size[0] // 3) * 255 / (size[0] // 3)), 0]
        
        # Add blue gradient
        for x in range(2 * size[0] // 3, size[0]):
            img[:, x] = [0, 0, int((x - 2 * size[0] // 3) * 255 / (size[0] // 3))]
        
        # Save as JPEG
        cv2.imwrite(str(path), img)
        
        return img
    
    def test_grayscale_conversion(self):
        """Test grayscale conversion."""
        result = self.basic_transforms.convert_to_grayscale(self.test_image_path)
        
        assert result is not None, "Grayscale conversion failed"
        assert result.mode == 'L', f"Expected mode 'L', got {result.mode}"
        
        # Check dimensions
        original = Image.open(self.test_image_path)
        assert result.size == original.size, "Image dimensions changed"
        
        # Verify it's actually grayscale
        np_array = np.array(result)
        assert len(np_array.shape) == 2 or (len(np_array.shape) == 3 and np_array.shape[2] == 1), \
            "Image is not grayscale"
    
    def test_sepia_conversion(self):
        """Test sepia tone conversion."""
        result = self.basic_transforms.convert_to_sepia(self.test_image_path)
        
        assert result is not None, "Sepia conversion failed"
        assert result.mode == 'RGB', f"Expected mode 'RGB', got {result.mode}"
        
        # Check dimensions
        original = Image.open(self.test_image_path)
        assert result.size == original.size, "Image dimensions changed"
        
        # Verify sepia characteristics (reddish-brown tint)
        np_array = np.array(result)
        mean_colors = np_array.mean(axis=(0, 1))
        
        # Sepia images typically have R > G > B
        assert mean_colors[0] >= mean_colors[1], "Sepia red channel issue"
        assert mean_colors[1] >= mean_colors[2], "Sepia green/blue channel issue"
    
    def test_invalid_image_path(self):
        """Test handling of invalid image path."""
        invalid_path = Path("nonexistent.jpg")
        
        result = self.basic_transforms.convert_to_grayscale(invalid_path)
        assert result is None, "Should return None for invalid path"
        
        result = self.basic_transforms.convert_to_sepia(invalid_path)
        assert result is None, "Should return None for invalid path"


class TestArtisticTransforms:
    """Test suite for artistic transformations."""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.config = {
            'pencil_sketch': {
                'pencil_tip_size': 'broad',
                'radius': 15,
                'clarity': 0.8,
                'blur_amount': 0.2,
                'strength': 0.5
            },
            'coloring_book': {
                'edge_detection_method': 'canny',
                'lower_threshold': 50,
                'upper_threshold': 150,
                'line_thickness': 2,
                'simplification_level': 3,
                'background_color': [255, 255, 255],
                'line_color': [0, 0, 0]
            },
            'logging': {
                'log_to_console': False,
                'log_to_file': False
            }
        }
        
        # Create logger
        cls.logger = setup_logging(cls.test_dir, cls.config)
        
        # Initialize transform class
        cls.artistic_transforms = ArtisticTransforms(cls.config, cls.logger)
        
        # Create test image
        cls.test_image_path = cls.test_dir / "test_image.jpg"
        cls.create_test_image(cls.test_image_path)
    
    @classmethod
    def teardown_class(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @staticmethod
    def create_test_image(path: Path):
        """Create a test image with shapes."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Draw some shapes for edge detection
        cv2.circle(img, (160, 240), 80, (255, 0, 0), -1)
        cv2.rectangle(img, (320, 160), (480, 320), (0, 255, 0), -1)
        cv2.ellipse(img, (480, 240), (60, 40), 45, 0, 360, (0, 0, 255), -1)
        
        cv2.imwrite(str(path), img)
        return img
    
    def test_pencil_sketch_conversion(self):
        """Test pencil sketch conversion."""
        result = self.artistic_transforms.convert_to_pencil_sketch(self.test_image_path)
        
        assert result is not None, "Pencil sketch conversion failed"
        assert isinstance(result, np.ndarray), "Result should be numpy array"
        
        # Check if it's grayscale
        assert len(result.shape) == 2 or (len(result.shape) == 3 and result.shape[2] == 1), \
            "Pencil sketch should be grayscale"
        
        # Check dimensions
        original = cv2.imread(str(self.test_image_path))
        assert result.shape[:2] == original.shape[:2], "Image dimensions changed"
    
    def test_coloring_book_conversion(self):
        """Test coloring book conversion."""
        result = self.artistic_transforms.convert_to_coloring_book(self.test_image_path)
        
        assert result is not None, "Coloring book conversion failed"
        assert isinstance(result, np.ndarray), "Result should be numpy array"
        
        # Check if it's basically binary (edges)
        unique_values = np.unique(result)
        assert len(unique_values) <= 256, "Too many unique values for edge image"
        
        # Should have mostly white background
        white_pixels = np.sum(result > 250)
        total_pixels = result.size
        assert white_pixels / total_pixels > 0.5, "Coloring book should be mostly white background"


class TestActivityTransforms:
    """Test suite for activity book transformations."""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.config = {
            'connect_the_dots': {
                'max_dots_per_image': 50,
                'min_dots_per_image': 10,
                'min_distance_between_dots': 10,
                'max_distance_between_dots': 50,
                'dot_size': 5,
                'number_font_size': 10,
                'edge_detection_sensitivity': 0.7
            },
            'color_by_numbers': {
                'max_distinct_colors': 10,
                'min_distinct_colors': 3,
                'min_area_size': 100,
                'max_area_size': 10000,
                'smoothing_kernel_size': 5,
                'color_similarity_threshold': 30,
                'number_font_size': 12,
                'border_thickness': 1
            },
            'logging': {
                'log_to_console': False,
                'log_to_file': False
            }
        }
        
        # Create logger
        cls.logger = setup_logging(cls.test_dir, cls.config)
        
        # Initialize transform class
        cls.activity_transforms = ActivityTransforms(cls.config, cls.logger)
        
        # Create test image
        cls.test_image_path = cls.test_dir / "test_image.jpg"
        cls.create_test_image(cls.test_image_path)
    
    @classmethod
    def teardown_class(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @staticmethod
    def create_test_image(path: Path):
        """Create a test image with distinct regions."""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw distinct colored shapes
        cv2.circle(img, (100, 100), 50, (255, 0, 0), -1)  # Blue circle
        cv2.rectangle(img, (200, 50), (350, 150), (0, 255, 0), -1)  # Green rectangle
        cv2.ellipse(img, (300, 300), (60, 40), 0, 0, 360, (0, 0, 255), -1)  # Red ellipse
        
        # Draw triangle
        pts = np.array([[100, 300], [50, 380], [150, 380]], np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 0))  # Yellow triangle
        
        cv2.imwrite(str(path), img)
        return img
    
    def test_connect_dots_conversion(self):
        """Test connect-the-dots conversion."""
        result = self.activity_transforms.convert_to_connect_dots(self.test_image_path)
        
        assert result is not None, "Connect-the-dots conversion failed"
        assert isinstance(result, np.ndarray), "Result should be numpy array"
        
        # Check if it's RGB
        assert len(result.shape) == 3 and result.shape[2] == 3, \
            "Connect-the-dots should be RGB image"
        
        # Should have mostly white background
        white_pixels = np.sum(np.all(result > 250, axis=2))
        total_pixels = result.shape[0] * result.shape[1]
        assert white_pixels / total_pixels > 0.8, "Should be mostly white background"
        
        # Should have some black dots
        black_pixels = np.sum(np.all(result < 10, axis=2))
        assert black_pixels > 0, "Should have black dots"
    
    def test_color_by_numbers_conversion(self):
        """Test color-by-numbers conversion."""
        result = self.activity_transforms.convert_to_color_by_numbers(self.test_image_path)
        
        assert result is not None, "Color-by-numbers conversion failed"
        assert isinstance(result, np.ndarray), "Result should be numpy array"
        
        # Check if it's RGB
        assert len(result.shape) == 3 and result.shape[2] == 3, \
            "Color-by-numbers should be RGB image"
        
        # Result should be wider than original (legend added)
        original = cv2.imread(str(self.test_image_path))
        assert result.shape[1] > original.shape[1], "Should have legend on the side"
        
        # Should have region boundaries (black lines)
        black_pixels = np.sum(np.all(result < 10, axis=2))
        assert black_pixels > 0, "Should have black boundary lines"
    
    def test_dot_placement_optimization(self):
        """Test dot placement optimization."""
        # Create simple points
        points = [(10, 10), (20, 20), (11, 11), (100, 100), (200, 200)]
        
        optimized = self.activity_transforms._optimize_dot_placement(points)
        
        assert len(optimized) <= len(points), "Should not add points"
        assert len(optimized) >= self.config['connect_the_dots']['min_dots_per_image'] or \
               len(optimized) == len(points), "Should maintain minimum dots if possible"
        
        # Check minimum distance constraint
        for i, p1 in enumerate(optimized):
            for j, p2 in enumerate(optimized):
                if i != j:
                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    assert dist >= self.config['connect_the_dots']['min_distance_between_dots'] - 1, \
                        "Points too close together"


class TestTransformIntegration:
    """Integration tests for transform pipeline."""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.config = {
            'grayscale': {'method': 'luminosity'},
            'sepia': {'intensity': 0.8},
            'pencil_sketch': {'radius': 15, 'strength': 0.5},
            'coloring_book': {'lower_threshold': 50, 'upper_threshold': 150},
            'connect_the_dots': {'max_dots_per_image': 50, 'min_dots_per_image': 10},
            'color_by_numbers': {'max_distinct_colors': 10, 'min_distinct_colors': 3},
            'logging': {'log_to_console': False, 'log_to_file': False}
        }
        
        cls.logger = setup_logging(cls.test_dir, cls.config)
        cls.basic_transforms = BasicTransforms(cls.config, cls.logger)
        cls.artistic_transforms = ArtisticTransforms(cls.config, cls.logger)
        cls.activity_transforms = ActivityTransforms(cls.config, cls.logger)
    
    @classmethod
    def teardown_class(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    def test_all_transforms_on_single_image(self):
        """Test all transformations on a single image."""
        # Create test image
        test_image = self.test_dir / "integration_test.jpg"
        self.create_complex_test_image(test_image)
        
        # Test all transformations
        transforms_to_test = [
            (self.basic_transforms.convert_to_grayscale, "grayscale"),
            (self.basic_transforms.convert_to_sepia, "sepia"),
            (self.artistic_transforms.convert_to_pencil_sketch, "pencil"),
            (self.artistic_transforms.convert_to_coloring_book, "coloring"),
            (self.activity_transforms.convert_to_connect_dots, "dots"),
            (self.activity_transforms.convert_to_color_by_numbers, "numbers")
        ]
        
        for transform_func, name in transforms_to_test:
            print(f"Testing {name} transformation...")
            result = transform_func(test_image)
            assert result is not None, f"{name} transformation failed"
            
            # Save result for manual inspection if needed
            if isinstance(result, np.ndarray):
                output_path = self.test_dir / f"result_{name}.jpg"
                cv2.imwrite(str(output_path), result)
            elif isinstance(result, Image.Image):
                output_path = self.test_dir / f"result_{name}.jpg"
                result.save(output_path)
    
    @staticmethod
    def create_complex_test_image(path: Path):
        """Create a complex test image with various features."""
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Add gradient background
        for y in range(600):
            img[y, :, :] = [255 - y // 3, 255 - y // 4, 255 - y // 5]
        
        # Add various shapes
        cv2.circle(img, (200, 150), 80, (255, 0, 0), -1)
        cv2.rectangle(img, (400, 100), (600, 200), (0, 255, 0), -1)
        cv2.ellipse(img, (400, 400), (100, 60), 30, 0, 360, (0, 0, 255), -1)
        
        # Add some lines
        cv2.line(img, (50, 50), (750, 550), (128, 128, 128), 3)
        cv2.line(img, (750, 50), (50, 550), (64, 64, 64), 2)
        
        # Add polygon
        pts = np.array([[600, 300], [700, 350], [700, 450], [600, 500], [500, 450], [500, 350]], np.int32)
        cv2.fillPoly(img, [pts], (255, 128, 0))
        
        cv2.imwrite(str(path), img)
        return img


# Performance tests
class TestPerformance:
    """Performance and stress tests."""
    
    @classmethod
    def setup_class(cls):
        """Setup for performance tests."""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.config = {
            'grayscale': {'method': 'luminosity'},
            'logging': {'log_to_console': False, 'log_to_file': False}
        }
        cls.logger = setup_logging(cls.test_dir, cls.config)
        cls.basic_transforms = BasicTransforms(cls.config, cls.logger)
    
    @classmethod
    def teardown_class(cls):
        """Clean up."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    def test_large_image_handling(self):
        """Test handling of large images."""
        # Create a large test image (4K resolution)
        large_image_path = self.test_dir / "large_test.jpg"
        large_img = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        cv2.imwrite(str(large_image_path), large_img)
        
        # Test transformation
        result = self.basic_transforms.convert_to_grayscale(large_image_path)
        assert result is not None, "Failed to process large image"
        assert result.size == (3840, 2160), "Large image dimensions incorrect"
    
    def test_batch_processing(self):
        """Test batch processing of multiple images."""
        # Create multiple test images
        image_paths = []
        for i in range(10):
            path = self.test_dir / f"batch_test_{i}.jpg"
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(path), img)
            image_paths.append(path)
        
        # Process all images
        results = []
        for path in image_paths:
            result = self.basic_transforms.convert_to_grayscale(path)
            results.append(result)
        
        assert len(results) == 10, "Not all images processed"
        assert all(r is not None for r in results), "Some images failed to process"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])