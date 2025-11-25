#!/usr/bin/env python3
"""
Minimal ColorAnalyzer test - bypass all utils to find the root cause
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_opencv():
    """Test basic OpenCV functionality."""
    try:
        import cv2
        import numpy as np
        
        # Try to create a simple image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        print("+ Basic numpy array creation works")
        
        # Try simple OpenCV operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("+ OpenCV color conversion works")
        
        return True
    except Exception as e:
        print(f"X Basic OpenCV test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_color_analyzer_minimal():
    """Test minimal ColorAnalyzer functionality."""
    try:
        # Avoid importing from utils entirely
        import logging
        
        # Create a basic logger without using utils
        logger = logging.getLogger('test')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        
        print("+ Basic logging setup works")
        
        # Now try the ColorAnalyzer import
        from src.models.color_analyzer import ColorAnalyzer
        print("+ ColorAnalyzer import successful")
        
        # Create instance
        config = {}
        analyzer = ColorAnalyzer(config, logger)
        print("+ ColorAnalyzer instance created")
        
        # Test basic functionality
        h, s, v = analyzer._rgb_to_hsv(255, 0, 0)
        print(f"+ RGB to HSV works: {h}, {s}, {v}")
        
        return True
        
    except Exception as e:
        print(f"X ColorAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Minimal ColorAnalyzer Test")
    print("=" * 30)
    
    # Test OpenCV first
    opencv_ok = test_basic_opencv()
    
    if opencv_ok:
        print("\\nTesting ColorAnalyzer...")
        analyzer_ok = test_color_analyzer_minimal()
        
        if analyzer_ok:
            print("\\n+ SUCCESS: ColorAnalyzer working at basic level")
        else:
            print("\\nX FAILED: ColorAnalyzer has issues")
    else:
        print("\\nX FAILED: Basic OpenCV not working")