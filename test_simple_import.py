#!/usr/bin/env python3
"""
Simple import test to identify the issue
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test individual imports to find the problem."""
    print("Testing individual imports...")
    
    # Test 1: Basic imports
    try:
        import cv2
        print("? OpenCV imported successfully")
    except Exception as e:
        print(f"? OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("? NumPy imported successfully")
    except Exception as e:
        print(f"? NumPy import failed: {e}")
        return False
    
    # Test 2: Try importing ColorAnalyzer
    try:
        print("Attempting ColorAnalyzer import...")
        from src.models.color_analyzer import ColorAnalyzer
        print("? ColorAnalyzer imported successfully")
        return True
    except Exception as e:
        print(f"? ColorAnalyzer import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n? All imports successful!")
    else:
        print("\n? Import failures detected")