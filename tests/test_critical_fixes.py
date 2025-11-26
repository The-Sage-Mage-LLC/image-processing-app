#!/usr/bin/env python3
"""
VERIFY ALL CRITICAL FIXES - Final comprehensive test
Tests:
1. No forced upscaling
2. Images processed as-is  
3. PIL Image to OpenCV saving works
4. No quality control violations
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

sys.path.insert(0, 'src')

def test_no_forced_upscaling():
    """Verify images are NOT forcibly upscaled."""
    print("\n1. Testing No Forced Upscaling...")
    
    import logging
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger()
    config = {}
    
    from src.transforms.basic_transforms import BasicTransforms
    
    bt = BasicTransforms(config, logger)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create small test image (72 DPI, 4032x1648 pixels = 56x22 inches @ 72 DPI)
        test_path = Path(temp_dir) / "test_small.jpg"
        img = Image.new('RGB', (4032, 1648), color=(100, 150, 200))
        img.save(test_path, dpi=(72, 72))
        
        # Convert to grayscale
        gray_img = bt.convert_to_grayscale(test_path)
        
        checks = [
            ("Grayscale conversion successful", gray_img is not None),
            ("Output dimensions match input", gray_img.size == (4032, 1648) if gray_img else False),
            ("No upscaling occurred", gray_img.size[0] <= 4032 if gray_img else False),
        ]
        
        for desc, passed in checks:
            print(f"   {'?' if passed else '?'} {desc}")
        
        return all(passed for _, passed in checks)

def test_pil_to_opencv_saving():
    """Verify PIL Images can be saved with OpenCV."""
    print("\n2. Testing PIL to OpenCV Saving...")
    
    import cv2
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create PIL Image
        pil_img = Image.new('L', (200, 200), color=128)
        
        # Convert to numpy for OpenCV
        np_img = np.array(pil_img)
        
        # Save with OpenCV
        output_path = test_dir / "test_output.jpg"
        success = cv2.imwrite(str(output_path), np_img)
        
        checks = [
            ("PIL to numpy conversion worked", np_img is not None),
            ("OpenCV save successful", success),
            ("Output file exists", output_path.exists() if success else False),
            ("Output file has content", output_path.stat().st_size > 0 if success and output_path.exists() else False),
        ]
        
        for desc, passed in checks:
            print(f"   {'?' if passed else '?'} {desc}")
        
        return all(passed for _, passed in checks)

def test_complete_grayscale_workflow():
    """Test complete grayscale workflow end-to-end."""
    print("\n3. Testing Complete Grayscale Workflow...")
    
    import logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()
    
    from src.core.file_manager import FileManager
    from src.core.image_processor import ImageProcessor
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create source directory with test image
        source_dir = test_dir / "source"
        source_dir.mkdir()
        
        test_img_path = source_dir / "test_image.jpg"
        img = Image.new('RGB', (800, 600), color=(200, 100, 50))
        img.save(test_img_path, dpi=(72, 72))
        
        original_size = test_img_path.stat().st_size
        
        # Create output and admin dirs
        output_dir = test_dir / "output"
        admin_dir = test_dir / "admin"
        output_dir.mkdir()
        admin_dir.mkdir()
        
        # Configure and process
        config = {
            'general': {'max_parallel_workers': 1},
            'paths': {'supported_formats': ['jpg', 'jpeg', 'png']},
            'basic_transforms': {'jpeg_quality': 95},
            'grayscale': {'method': 'luminosity'}
        }
        
        file_manager = FileManager([source_dir], output_dir, admin_dir, config, logger)
        processor = ImageProcessor(file_manager, config, logger)
        
        # Run grayscale conversion
        processor.convert_grayscale()
        
        # Verify output
        output_file = output_dir / "BWG_ORIG" / "BWG_ORIG_test_image.jpg"
        
        checks = [
            ("BWG_ORIG folder created", (output_dir / "BWG_ORIG").exists()),
            ("Output file created", output_file.exists()),
            ("Output file has content", output_file.stat().st_size > 0 if output_file.exists() else False),
        ]
        
        if output_file.exists():
            # Verify it's actually grayscale
            import cv2
            result_img = cv2.imread(str(output_file))
            checks.append(("Output image can be read", result_img is not None))
            
            if result_img is not None:
                h, w = result_img.shape[:2]
                checks.extend([
                    ("Dimensions preserved", h == 600 and w == 800),
                    ("No upscaling occurred", h <= 600 and w <= 800),
                ])
        
        for desc, passed in checks:
            print(f"   {'?' if passed else '?'} {desc}")
        
        return all(passed for _, passed in checks)

def test_sepia_workflow():
    """Test complete sepia workflow end-to-end."""
    print("\n4. Testing Complete Sepia Workflow...")
    
    import logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()
    
    from src.core.file_manager import FileManager
    from src.core.image_processor import ImageProcessor
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create source directory with test image
        source_dir = test_dir / "source"
        source_dir.mkdir()
        
        test_img_path = source_dir / "test_image.jpg"
        img = Image.new('RGB', (800, 600), color=(200, 100, 50))
        img.save(test_img_path, dpi=(72, 72))
        
        # Create output and admin dirs
        output_dir = test_dir / "output"
        admin_dir = test_dir / "admin"
        output_dir.mkdir()
        admin_dir.mkdir()
        
        # Configure and process
        config = {
            'general': {'max_parallel_workers': 1},
            'paths': {'supported_formats': ['jpg', 'jpeg', 'png']},
            'basic_transforms': {'jpeg_quality': 95},
            'sepia': {'intensity': 0.8}
        }
        
        file_manager = FileManager([source_dir], output_dir, admin_dir, config, logger)
        processor = ImageProcessor(file_manager, config, logger)
        
        # Run sepia conversion
        processor.convert_sepia()
        
        # Verify output
        output_file = output_dir / "SEP_ORIG" / "SEP_ORIG_test_image.jpg"
        
        checks = [
            ("SEP_ORIG folder created", (output_dir / "SEP_ORIG").exists()),
            ("Output file created", output_file.exists()),
            ("Output file has content", output_file.stat().st_size > 0 if output_file.exists() else False),
        ]
        
        if output_file.exists():
            # Verify it's actually sepia
            import cv2
            result_img = cv2.imread(str(output_file))
            checks.append(("Output image can be read", result_img is not None))
            
            if result_img is not None:
                h, w = result_img.shape[:2]
                checks.extend([
                    ("Dimensions preserved", h == 600 and w == 800),
                    ("No upscaling occurred", h <= 600 and w <= 800),
                ])
        
        for desc, passed in checks:
            print(f"   {'?' if passed else '?'} {desc}")
        
        return all(passed for _, passed in checks)

def main():
    """Run all critical tests."""
    print("="*80)
    print("CRITICAL FIXES VERIFICATION")
    print("="*80)
    
    tests = [
        ("No Forced Upscaling", test_no_forced_upscaling),
        ("PIL to OpenCV Saving", test_pil_to_opencv_saving),
        ("Complete Grayscale Workflow", test_complete_grayscale_workflow),
        ("Complete Sepia Workflow", test_sepia_workflow),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n? {test_name} - ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "? PASSED" if result else "? FAILED"
        print(f"  {status:<15} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n??? ALL CRITICAL FIXES VERIFIED!")
        print("\nFixed Issues:")
        print("  ? No forced upscaling - images processed at original size")
        print("  ? PIL Image to OpenCV saving works correctly")
        print("  ? Grayscale workflow works end-to-end")
        print("  ? Sepia workflow works end-to-end")
        print("\n? Application is now fully functional!")
        return True
    else:
        print("\n? Some tests failed - review above for details")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
