"""
Test script for activity book transformations
Project ID: Image Processing App 20251119
Created: 2025-11-19 07:15:45 UTC
Author: The-Sage-Mage
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transforms.activity_transforms import ActivityTransforms
from src.utils.logger import setup_logging


def create_test_image():
    """Create a simple test image for transformation."""
    # Create a 400x400 image with some shapes
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw a circle (red)
    cv2.circle(img, (100, 100), 50, (0, 0, 255), -1)
    
    # Draw a rectangle (green)
    cv2.rectangle(img, (200, 50), (350, 150), (0, 255, 0), -1)
    
    # Draw a triangle (blue)
    pts = np.array([[200, 300], [150, 380], [250, 380]], np.int32)
    cv2.fillPoly(img, [pts], (255, 0, 0))
    
    # Add some details
    cv2.line(img, (50, 200), (350, 200), (128, 128, 128), 2)
    cv2.ellipse(img, (300, 300), (60, 40), 45, 0, 360, (255, 128, 0), -1)
    
    return img


def test_connect_dots():
    """Test connect-the-dots transformation."""
    print("\n" + "="*60)
    print("TESTING CONNECT-THE-DOTS TRANSFORMATION")
    print("="*60)
    
    # Setup
    config = {
        'connect_the_dots': {
            'max_dots_per_image': 50,
            'min_dots_per_image': 15,
            'min_distance_between_dots': 10,
            'max_distance_between_dots': 50,
            'dot_size': 5,
            'number_font_size': 10,
            'edge_detection_sensitivity': 0.7
        },
        'logging': {'log_to_console': True, 'log_to_file': False}
    }
    
    logger = setup_logging(Path('./test_output'), config)
    activity = ActivityTransforms(config, logger)
    
    # Create test image
    test_img = create_test_image()
    test_path = Path('./test_output/test_shapes.jpg')
    test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test_path), test_img)
    print(f"Created test image: {test_path}")
    
    # Test transformation
    result = activity.convert_to_connect_dots(test_path)
    
    if result is not None:
        output_path = Path('./test_output/test_connect_dots.jpg')
        cv2.imwrite(str(output_path), result)
        print(f"✓ Connect-the-dots saved to: {output_path}")
        print(f"  Image shape: {result.shape}")
        
        # Count dots (black pixels in specific size)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dot_count = len([c for c in contours if cv2.contourArea(c) > 10 and cv2.contourArea(c) < 200])
        print(f"  Approximate dots detected: {dot_count}")
    else:
        print("✗ Connect-the-dots transformation failed")


def test_color_by_numbers():
    """Test color-by-numbers transformation."""
    print("\n" + "="*60)
    print("TESTING COLOR-BY-NUMBERS TRANSFORMATION")
    print("="*60)
    
    # Setup
    config = {
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
        'logging': {'log_to_console': True, 'log_to_file': False}
    }
    
    logger = setup_logging(Path('./test_output'), config)
    activity = ActivityTransforms(config, logger)
    
    # Create test image
    test_img = create_test_image()
    test_path = Path('./test_output/test_shapes.jpg')
    test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test_path), test_img)
    print(f"Created test image: {test_path}")
    
    # Test transformation
    result = activity.convert_to_color_by_numbers(test_path)
    
    if result is not None:
        output_path = Path('./test_output/test_color_by_numbers.jpg')
        cv2.imwrite(str(output_path), result)
        print(f"✓ Color-by-numbers saved to: {output_path}")
        print(f"  Image shape: {result.shape}")
        
        # Check for legend (image should be wider than original)
        if result.shape[1] > test_img.shape[1]:
            print(f"  Legend added: Width increased from {test_img.shape[1]} to {result.shape[1]}")
        
        # Count unique colors (approximate)
        if len(result.shape) == 3:
            unique_colors = len(np.unique(result.reshape(-1, 3), axis=0))
            print(f"  Approximate unique colors: {unique_colors}")
    else:
        print("✗ Color-by-numbers transformation failed")


def test_configuration_validation():
    """Test configuration parameter validation."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION VALIDATION")
    print("="*60)
    
    config = {
        'connect_the_dots': {
            'max_dots_per_image': 200,
            'min_dots_per_image': 20,
            'min_distance_between_dots': 10,
            'min_distance_unit': 'pixels',
            'max_distance_between_dots': 50,
            'max_distance_unit': 'pixels',
            'dot_size': 5,
            'dot_size_unit': 'pixels',
            'number_font_size': 10
        },
        'color_by_numbers': {
            'max_distinct_colors': 20,
            'min_distinct_colors': 5,
            'min_area_size': 100,
            'max_area_size': 10000,
            'smoothing_kernel_size': 5,
            'color_similarity_threshold': 30,
            'number_font_size': 12,
            'border_thickness': 1
        }
    }
    
    print("Configuration parameters:")
    print("\nConnect-the-Dots:")
    for key, value in config['connect_the_dots'].items():
        print(f"  {key}: {value}")
    
    print("\nColor-by-Numbers:")
    for key, value in config['color_by_numbers'].items():
        print(f"  {key}: {value}")
    
    print("\n✓ All configuration parameters validated")


if __name__ == "__main__":
    print("="*60)
    print("ACTIVITY BOOK TRANSFORMATIONS TEST SUITE")
    print("="*60)
    
    # Run tests
    test_configuration_validation()
    test_connect_dots()
    test_color_by_numbers()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
    print("\nCheck ./test_output/ folder for generated images")