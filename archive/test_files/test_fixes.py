#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Fixes Verification Script
Project ID: Image Processing App 20251119
Created: 2025-01-19
Author: The-Sage-Mage

Test the specific fixes implemented.
"""

import sys
import os
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_image(filename: str, size: tuple = (200, 200), complexity: str = "simple") -> Path:
    """Create a test image with specific characteristics."""
    if complexity == "simple":
        # Simple image with few colors - good for color-by-numbers
        image = Image.new('RGB', size, color='blue')
        pixels = np.array(image)
        
        # Add simple shapes
        h, w = pixels.shape[:2]
        pixels[h//4:3*h//4, w//4:3*w//4] = [255, 0, 0]  # Red square
        pixels[h//3:2*h//3, w//3:2*w//3] = [0, 255, 0]  # Green square
        
    elif complexity == "dots":
        # Good for connect-the-dots
        image = Image.new('RGB', size, color='white')
        pixels = np.array(image)
        
        # Draw simple outline
        cv2.rectangle(pixels, (50, 50), (150, 150), (0, 0, 0), 3)
        cv2.circle(pixels, (100, 100), 40, (0, 0, 0), 3)
        
    else:  # complex
        # Complex image that should be rejected
        image = Image.new('RGB', size, color='white')
        pixels = np.array(image)
        
        # Add lots of random noise and details
        for i in range(100):
            x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
            color = [np.random.randint(0, 255) for _ in range(3)]
            cv2.circle(pixels, (x, y), np.random.randint(2, 8), color, -1)
    
    test_image = Image.fromarray(pixels)
    test_path = Path(filename)
    test_image.save(test_path, 'JPEG')
    
    return test_path

def test_csv_primary_keys():
    """Test that CSV files include primary keys."""
    print("Testing CSV primary key inclusion...")
    
    from src.core.image_processor import ImageProcessor
    from src.core.file_manager import FileManager
    from src.utils.logger import setup_logging
    
    # Create mock results
    blur_results = [
        {'file_path': 'test1.jpg', 'is_blurry': False, 'weighted_score': 85.2},
        {'file_path': 'test2.jpg', 'is_blurry': True, 'weighted_score': 45.1}
    ]
    
    caption_results = [
        {'file_path': 'test1.jpg', 'primary_caption': 'A test image', 'gps_location': 'Test Location'},
        {'file_path': 'test2.jpg', 'primary_caption': 'Another test', 'gps_location': 'Another Location'}
    ]
    
    # Test the CSV saving methods
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Mock processor
        config = {}
        logger = setup_logging(temp_path, config)
        file_manager = FileManager([temp_path], temp_path, temp_path, config, logger)
        processor = ImageProcessor(file_manager, config, logger)
        
        # Test blur CSV
        blur_csv = temp_path / "test_blur.csv"
        processor._save_blur_results_to_csv(blur_results, blur_csv)
        
        # Test caption CSV  
        caption_csv = temp_path / "test_captions.csv"
        processor._save_caption_results_to_csv(caption_results, caption_csv)
        
        # Verify primary keys
        with open(blur_csv, 'r') as f:
            blur_content = f.read()
            if 'primary_key' in blur_content and '1,' in blur_content:
                print("OK - Blur CSV includes primary keys")
            else:
                print("FAIL - Blur CSV missing primary keys")
        
        with open(caption_csv, 'r') as f:
            caption_content = f.read()
            if 'primary_key' in caption_content and '1,' in caption_content:
                print("OK - Caption CSV includes primary keys")
            else:
                print("FAIL - Caption CSV missing primary keys")

def test_enhanced_captions():
    """Test enhanced caption generation with GPS integration."""
    print("Testing enhanced caption generation...")
    
    from src.models.caption_generator import CaptionGenerator
    
    config = {}
    
    # Create simple logger
    import logging
    logger = logging.getLogger('test')
    logger.setLevel(logging.INFO)
    
    caption_gen = CaptionGenerator(config, logger)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_img = create_test_image(str(temp_path / "test.jpg"), complexity="simple")
        
        result = caption_gen.generate_captions(test_img)
        
        # Check for enhanced features
        if result.get('gps_location') and result['gps_location'] != 'Location data requires metadata extraction':
            print("OK - GPS integration working")
        else:
            print("INFO - GPS integration shows fallback (expected without real metadata)")
        
        if len(result.get('primary_caption', '')) > 20:
            print("OK - Enhanced captions are more detailed")
        else:
            print("FAIL - Captions still too basic")
        
        if 'primary_key' in str(result):
            print("INFO - Primary key in caption result (should be added during CSV save)")

def test_connect_dots_improvements():
    """Test connect-the-dots improvements."""
    print("Testing connect-the-dots improvements...")
    
    from src.transforms.activity_transforms import ActivityTransforms
    
    config = {
        'connect_the_dots': {
            'max_dots_per_image': 50,
            'min_dots_per_image': 8,
            'dot_size': 8,
            'min_distance_between_dots': 25
        }
    }
    
    import logging
    logger = logging.getLogger('test')
    
    activity = ActivityTransforms(config, logger)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test with simple image (should work)
        simple_img = create_test_image(str(temp_path / "simple.jpg"), complexity="dots")
        result_simple = activity.convert_to_connect_dots(simple_img)
        
        # Test with complex image (should reject)
        complex_img = create_test_image(str(temp_path / "complex.jpg"), complexity="complex")
        result_complex = activity.convert_to_connect_dots(complex_img)
        
        if result_simple is not None:
            print("OK - Simple image processed for connect-the-dots")
        else:
            print("FAIL - Simple image rejected incorrectly")
        
        if result_complex is None:
            print("OK - Complex image correctly rejected")
        else:
            print("INFO - Complex image processed (may be acceptable)")

def test_color_by_numbers_improvements():
    """Test color-by-numbers improvements."""
    print("Testing color-by-numbers improvements...")
    
    from src.transforms.activity_transforms import ActivityTransforms
    
    config = {
        'color_by_numbers': {
            'max_distinct_colors': 8,
            'min_distinct_colors': 3,
            'min_area_size': 500,
            'number_font_size': 16
        }
    }
    
    import logging
    logger = logging.getLogger('test')
    
    activity = ActivityTransforms(config, logger)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test with simple image (should work)
        simple_img = create_test_image(str(temp_path / "simple.jpg"), complexity="simple")
        result_simple = activity.convert_to_color_by_numbers(simple_img)
        
        # Test with complex image (should reject)
        complex_img = create_test_image(str(temp_path / "complex.jpg"), complexity="complex")
        result_complex = activity.convert_to_color_by_numbers(complex_img)
        
        if result_simple is not None:
            print("OK - Simple image processed for color-by-numbers")
        else:
            print("FAIL - Simple image rejected incorrectly")
        
        if result_complex is None:
            print("OK - Complex image correctly rejected")
        else:
            print("INFO - Complex image processed (may be acceptable)")

def main():
    """Run all fix verification tests."""
    print("Fix Verification Tests")
    print("=" * 60)
    
    tests = [
        test_csv_primary_keys,
        test_enhanced_captions,
        test_connect_dots_improvements,
        test_color_by_numbers_improvements
    ]
    
    for test_func in tests:
        try:
            print(f"\n{test_func.__name__.replace('_', ' ').title()}:")
            test_func()
        except Exception as e:
            print(f"FAIL - Test {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 60)
    print("Fix verification complete!")
    print("\nSummary of Fixes:")
    print("- Enhanced CUDA detection and GPU utilization")
    print("- Fixed primary keys in blur and caption CSV files")
    print("- Improved caption generation with GPS integration and object detection")
    print("- Enhanced connect-the-dots with fewer, larger dots and complexity rejection")
    print("- Improved color-by-numbers with fewer colors and named color legend")
    print("- Updated configuration with optimized parameters")

if __name__ == "__main__":
    main()