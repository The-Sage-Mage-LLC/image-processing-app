#!/usr/bin/env python3
"""
Comprehensive ColorAnalyzer diagnostic test
Bypasses problematic imports to isolate the root cause of OpenCV issues
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic imports without triggering any array safety."""
    print("Testing Basic Imports")
    print("-" * 25)
    
    try:
        import json
        print("+ JSON import successful")
    except Exception as e:
        print(f"X JSON import failed: {e}")
        return False
    
    try:
        import logging
        print("+ Logging import successful")
    except Exception as e:
        print(f"X Logging import failed: {e}")
        return False
        
    try:
        from pathlib import Path
        print("+ Pathlib import successful")
    except Exception as e:
        print(f"X Pathlib import failed: {e}")
        return False
        
    return True

def test_numpy_import():
    """Test numpy import separately."""
    print("\nTesting NumPy Import")
    print("-" * 22)
    
    try:
        import numpy as np
        print("+ NumPy import successful")
        
        # Test basic numpy functionality
        arr = np.zeros((10, 10), dtype=np.uint8)
        print("+ NumPy array creation successful")
        
        # Test array operations
        mean_val = np.mean(arr)
        print(f"+ NumPy operations work (mean: {mean_val})")
        
        return True
    except Exception as e:
        print(f"X NumPy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opencv_import():
    """Test OpenCV import separately."""
    print("\nTesting OpenCV Import")
    print("-" * 23)
    
    try:
        import cv2
        print("+ OpenCV import successful")
        
        # Get OpenCV version
        version = cv2.__version__
        print(f"+ OpenCV version: {version}")
        
        return True
    except Exception as e:
        print(f"X OpenCV import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opencv_numpy_interaction():
    """Test OpenCV and NumPy working together."""
    print("\nTesting OpenCV-NumPy Interaction")
    print("-" * 35)
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        print("+ Created NumPy image array")
        
        # Set some pixels to test
        img[25:75, 25:75] = [255, 0, 0]  # Red square
        print("+ Modified NumPy array")
        
        # Test OpenCV operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("+ OpenCV color conversion successful")
        
        # Test more OpenCV operations
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        print("+ OpenCV blur operation successful")
        
        # Test histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        print("+ OpenCV histogram calculation successful")
        
        return True
    except Exception as e:
        print(f"X OpenCV-NumPy interaction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_src_path_setup():
    """Test setting up src path without importing anything problematic."""
    print("\nTesting Src Path Setup")
    print("-" * 25)
    
    try:
        # Add src to path
        src_path = str(Path(__file__).parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        print(f"+ Added src path: {src_path}")
        
        # Check if path exists
        if Path(src_path).exists():
            print("+ Src directory exists")
        else:
            print("X Src directory not found")
            return False
        
        # Check for ColorAnalyzer file
        color_analyzer_path = Path(src_path) / "models" / "color_analyzer.py"
        if color_analyzer_path.exists():
            print("+ color_analyzer.py found")
        else:
            print("X color_analyzer.py not found")
            return False
            
        return True
    except Exception as e:
        print(f"X Src path setup failed: {e}")
        return False

def test_standalone_color_functions():
    """Test color conversion functions without importing ColorAnalyzer class."""
    print("\nTesting Standalone Color Functions")
    print("-" * 37)
    
    try:
        import colorsys
        print("+ Colorsys import successful")
        
        # Test RGB to HSV conversion
        r, g, b = 255, 0, 0  # Red
        r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        h_deg = int(h * 360)
        s_pct = int(s * 100)
        v_pct = int(v * 100)
        
        print(f"+ RGB({r},{g},{b}) -> HSV({h_deg},{s_pct}%,{v_pct}%)")
        
        # Test CMYK conversion
        k = 1 - max(r_norm, g_norm, b_norm)
        if k == 1:
            c, m, y = 0, 0, 0
        else:
            c = (1 - r_norm - k) / (1 - k)
            m = (1 - g_norm - k) / (1 - k)
            y = (1 - b_norm - k) / (1 - k)
        
        c_pct = int(c * 100)
        m_pct = int(m * 100)
        y_pct = int(y * 100)
        k_pct = int(k * 100)
        
        print(f"+ RGB({r},{g},{b}) -> CMYK({c_pct}%,{m_pct}%,{y_pct}%,{k_pct}%)")
        
        # Test HEX conversion
        hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
        print(f"+ RGB({r},{g},{b}) -> HEX({hex_color})")
        
        return True
        
    except Exception as e:
        print(f"X Color function tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_color_loading():
    """Test loading external JSON color dictionaries."""
    print("\nTesting JSON Color Dictionary Loading")
    print("-" * 42)
    
    try:
        import json
        
        # Check for color dictionary files
        config_path = Path(__file__).parent / "config" / "colors"
        
        if not config_path.exists():
            print("X Color config directory not found")
            return False
        
        print(f"+ Color config directory found: {config_path}")
        
        # Look for JSON files
        json_files = list(config_path.glob("*.json"))
        print(f"+ Found {len(json_files)} JSON color files")
        
        if json_files:
            # Try to load one file as a test
            test_file = json_files[0]
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    color_data = json.load(f)
                    
                print(f"+ Successfully loaded: {test_file.name}")
                
                if 'colors' in color_data:
                    color_count = len(color_data['colors'])
                    print(f"+ Found {color_count} colors in {test_file.name}")
                else:
                    print("X Invalid color file format (no 'colors' key)")
                    
            except Exception as e:
                print(f"X Failed to load {test_file.name}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"X JSON color loading test failed: {e}")
        return False

def test_color_analyzer_import_only():
    """Test just importing ColorAnalyzer without creating instance."""
    print("\nTesting ColorAnalyzer Import Only")
    print("-" * 36)
    
    try:
        # Ensure src is in path
        src_path = str(Path(__file__).parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            
        # Try the import
        from src.models.color_analyzer import ColorAnalyzer
        print("+ ColorAnalyzer import successful")
        
        # Check if class is properly defined
        if hasattr(ColorAnalyzer, '__init__'):
            print("+ ColorAnalyzer class structure looks good")
        else:
            print("X ColorAnalyzer class structure incomplete")
            return False
            
        return True
        
    except Exception as e:
        print(f"X ColorAnalyzer import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_color_analyzer_basic_creation():
    """Test creating ColorAnalyzer instance with minimal configuration."""
    print("\nTesting ColorAnalyzer Basic Creation")
    print("-" * 38)
    
    try:
        import logging
        
        # Create minimal logger
        logger = logging.getLogger('test_minimal')
        logger.setLevel(logging.WARNING)  # Reduce noise
        
        # Import ColorAnalyzer
        from src.models.color_analyzer import ColorAnalyzer
        
        # Create with minimal config
        config = {}
        
        print("+ Creating ColorAnalyzer instance...")
        analyzer = ColorAnalyzer(config, logger)
        print("+ ColorAnalyzer instance created successfully")
        
        # Test basic methods
        if hasattr(analyzer, '_rgb_to_hsv'):
            h, s, v = analyzer._rgb_to_hsv(255, 0, 0)
            print(f"+ RGB to HSV method works: {h}, {s}, {v}")
        else:
            print("X RGB to HSV method missing")
            return False
            
        if hasattr(analyzer, 'color_databases'):
            db_count = len(analyzer.color_databases)
            print(f"+ Color databases loaded: {db_count}")
        else:
            print("X Color databases not initialized")
            return False
            
        return True
        
    except Exception as e:
        print(f"X ColorAnalyzer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive diagnostic tests."""
    print("ColorAnalyzer Comprehensive Diagnostic Test")
    print("=" * 50)
    print("This test will help identify exactly where the issue occurs.\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("NumPy Import", test_numpy_import),
        ("OpenCV Import", test_opencv_import),
        ("OpenCV-NumPy Interaction", test_opencv_numpy_interaction),
        ("Src Path Setup", test_src_path_setup),
        ("Standalone Color Functions", test_standalone_color_functions),
        ("JSON Color Dictionary Loading", test_json_color_loading),
        ("ColorAnalyzer Import Only", test_color_analyzer_import_only),
        ("ColorAnalyzer Basic Creation", test_color_analyzer_basic_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            
            if not result:
                print(f"\n!!! STOPPING HERE - {test_name} failed")
                print("!!! This is likely where the issue begins")
                break
                
        except Exception as e:
            print(f"\nX EXCEPTION in {test_name}: {e}")
            results.append((test_name, False))
            break
    
    # Print summary
    print(f"\n{'='*50}")
    print("DIAGNOSTIC RESULTS SUMMARY")
    print(f"{'='*50}")
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    if passed == total:
        print(f"\n+ ALL TESTS PASSED ({passed}/{len(tests)})!")
        print("+ Your ColorAnalyzer should be working correctly")
    else:
        print(f"\n! TESTS PASSED: {passed}/{total}")
        print("! Check the first failed test above for the root cause")
        
        # Provide specific guidance
        first_failure = next((name for name, result in results if not result), None)
        if first_failure:
            print(f"\n*** ROOT CAUSE: Issue appears to be in '{first_failure}'")
            
            if "NumPy" in first_failure:
                print("*** SOLUTION: NumPy installation or compatibility issue")
                print("*** Try: pip install --upgrade numpy")
            elif "OpenCV" in first_failure:
                print("*** SOLUTION: OpenCV installation or compatibility issue") 
                print("*** Try: pip install --upgrade opencv-python")
            elif "Interaction" in first_failure:
                print("*** SOLUTION: NumPy-OpenCV compatibility issue")
                print("*** Try: pip install --upgrade numpy opencv-python")
            elif "ColorAnalyzer" in first_failure:
                print("*** SOLUTION: Issue with ColorAnalyzer implementation")
                print("*** Check the code in src/models/color_analyzer.py")

if __name__ == "__main__":
    main()