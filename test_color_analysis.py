#!/usr/bin/env python3
"""
Color Analysis Test Script
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Test the comprehensive color analysis functionality (Menu Item 5).
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

def create_color_test_image(filename: str, colors: list) -> Path:
    """Create a test image with specific colors for testing."""
    size = (200, 200)
    image = Image.new('RGB', size, color='white')
    pixels = np.array(image)
    
    h, w = size
    sections = len(colors)
    section_width = w // sections
    
    for i, color in enumerate(colors):
        start_x = i * section_width
        end_x = min((i + 1) * section_width, w)
        pixels[:, start_x:end_x] = color
    
    test_image = Image.fromarray(pixels)
    test_path = Path(filename)
    test_image.save(test_path, 'JPEG')
    
    return test_path

def test_color_analysis():
    """Test the color analysis functionality."""
    print("Testing Color Analysis (Menu Item 5)")
    print("=" * 50)
    
    try:
        from src.models.color_analyzer import ColorAnalyzer
        from src.utils.logger import setup_logging
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Setup logger
            logger = setup_logging(temp_path, {})
            
            # Initialize color analyzer
            config = {
                'color_analysis': {
                    'num_dominant_colors': 4,
                    'enable_advanced_cv': True,
                    'color_similarity_threshold': 30,
                    'validate_color_presence': True,
                    'merge_similar_colors': True,
                    'use_ensemble_methods': True
                }
            }
            
            color_analyzer = ColorAnalyzer(config, logger)
            
            # Test with simple color image
            test_colors = [
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green  
                [0, 0, 255],    # Blue
                [255, 255, 0]   # Yellow
            ]
            
            test_image = create_color_test_image(str(temp_path / "test_colors.jpg"), test_colors)
            
            print("Analyzing test image with known colors...")
            result = color_analyzer.analyze_colors(test_image)
            
            if 'error' in result:
                print(f"FAIL - Analysis failed: {result['error']}")
                return
            
            # Check results
            print("\nAnalysis Results:")
            print(f"File: {result.get('file_name')}")
            print(f"Image size: {result.get('image_width')}x{result.get('image_height')}")
            
            # Check each color
            for i in range(1, 5):
                rgb_r = result.get(f'color_{i}_rgb_r', 'N/A')
                rgb_g = result.get(f'color_{i}_rgb_g', 'N/A')
                rgb_b = result.get(f'color_{i}_rgb_b', 'N/A')
                hex_val = result.get(f'color_{i}_hex_value', 'N/A')
                percentage = result.get(f'color_{i}_percentage', 'N/A')
                color_name = result.get(f'color_{i}_nearest_color_name', 'N/A')
                
                print(f"\nColor {i}:")
                print(f"  RGB: {rgb_r}, {rgb_g}, {rgb_b}")
                print(f"  HEX: {hex_val}")
                print(f"  Percentage: {percentage}%")
                print(f"  Name: {color_name}")
            
            # Test CSV output
            print("\nTesting CSV output...")
            csv_path = temp_path / "test_colors.csv"
            color_analyzer.save_color_analysis_to_csv([result], csv_path)
            
            if csv_path.exists():
                print("OK - CSV file created successfully")
                
                # Read first line to verify structure
                with open(csv_path, 'r', encoding='utf-8') as f:
                    header = f.readline().strip()
                
                # Check for primary key
                if header.startswith('primary_key'):
                    print("OK - CSV includes primary_key as first column")
                else:
                    print("FAIL - CSV missing primary_key as first column")
                
                print(f"CSV saved to: {csv_path}")
                
            else:
                print("FAIL - CSV file not created")
            
            print("\n" + "=" * 50)
            print("SUCCESS - Color Analysis test completed!")
                
    except ImportError as e:
        print(f"FAIL - Required modules not available: {e}")
    except Exception as e:
        print(f"FAIL - Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_color_analysis()