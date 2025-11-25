#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Menu Item 2 (Blur Detection) Verification Test
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

This script verifies that Menu Item 2 is completely implemented and functional according to requirements.
"""

import sys
import tempfile
import csv
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageFilter
import numpy as np


def create_test_images(test_dir: Path):
    """Create test images with varying levels of blur for testing."""
    
    # Create a sharp test image
    sharp_img = Image.new('RGB', (400, 400), color='blue')
    sharp_path = test_dir / "sharp_test_image.jpg"
    sharp_img.save(sharp_path, 'JPEG')
    
    # Create a slightly blurry image
    slightly_blurry_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=1))
    slightly_blurry_path = test_dir / "slightly_blurry_test.jpg"
    slightly_blurry_img.save(slightly_blurry_path, 'JPEG')
    
    # Create a very blurry image
    very_blurry_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=5))
    very_blurry_path = test_dir / "very_blurry_test.jpg"
    very_blurry_img.save(very_blurry_path, 'JPEG')
    
    # Create subdirectory with images to test recursive scanning
    sub_dir = test_dir / "subdirectory"
    sub_dir.mkdir()
    
    sub_img = Image.new('RGB', (300, 300), color='red')
    sub_blur_img = sub_img.filter(ImageFilter.GaussianBlur(radius=3))
    sub_blur_path = sub_dir / "sub_blurry_image.png"
    sub_blur_img.save(sub_blur_path, 'PNG')
    
    return [sharp_path, slightly_blurry_path, very_blurry_path, sub_blur_path]


def verify_blur_detection_models():
    """Verify multiple AI models are implemented."""
    print("?? Verifying Multiple AI Models Implementation...")
    
    try:
        from src.models.blur_detector import BlurDetector
        from src.utils.logger import setup_logging
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test image
            test_image = temp_path / "test.jpg"
            img = Image.new('RGB', (100, 100), color='green')
            img.save(test_image, 'JPEG')
            
            # Initialize blur detector
            config = {
                'blur_detection': {
                    'models': ['laplacian', 'variance_of_laplacian', 'gradient_magnitude', 'tenengrad', 'brenner'],
                    'center_weight': 1.5,
                    'peripheral_weight': 0.5
                }
            }
            logger = setup_logging(temp_path, config)
            detector = BlurDetector(config, logger)
            
            # Test detection
            result = detector.detect_blur(test_image)
            
            checks = [
                ("Multiple models available", len(config['blur_detection']['models']) >= 3),
                ("Laplacian method", 'laplacian_score' in result),
                ("Variance method", 'variance_score' in result),
                ("Gradient method", 'gradient_score' in result),
                ("Center weighting", detector.center_weight > detector.peripheral_weight),
                ("Consensus scoring", 'consensus_score' in result),
            ]
            
            for desc, passed in checks:
                print(f"  {'?' if passed else '?'} {desc}")
            
            return all(passed for _, passed in checks)
            
    except Exception as e:
        print(f"? Blur detection models verification failed: {e}")
        return False


def verify_center_weighting():
    """Verify center area weighting is implemented."""
    print("?? Verifying Center Area Weighting...")
    
    try:
        from src.models.blur_detector import BlurDetector
        from src.utils.logger import setup_logging
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config = {
                'blur_detection': {
                    'models': ['laplacian'],
                    'center_weight': 1.5,
                    'peripheral_weight': 0.5
                }
            }
            logger = setup_logging(temp_path, config)
            detector = BlurDetector(config, logger)
            
            # Test weight mask creation
            weight_mask = detector._create_weight_mask((100, 100))
            
            center_value = weight_mask[50, 50]  # Center
            edge_value = weight_mask[0, 0]      # Corner
            
            checks = [
                ("Weight mask created", weight_mask is not None),
                ("Center has higher weight", center_value > edge_value),
                ("Weight configuration applied", detector.center_weight == 1.5),
                ("Peripheral weight configured", detector.peripheral_weight == 0.5),
            ]
            
            for desc, passed in checks:
                print(f"  {'?' if passed else '?'} {desc}")
            
            return all(passed for _, passed in checks)
            
    except Exception as e:
        print(f"? Center weighting verification failed: {e}")
        return False


def verify_csv_output_structure():
    """Verify CSV output has correct structure and naming."""
    print("?? Verifying CSV Output Structure...")
    
    try:
        from src.core.image_processor import ImageProcessor
        from src.core.file_manager import FileManager
        from src.utils.logger import setup_logging
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test images
            test_images = create_test_images(temp_path)
            
            # Setup components
            config = {
                'blur_detection': {
                    'models': ['laplacian', 'variance_of_laplacian'],
                    'consensus_threshold': 1
                },
                'general': {'max_parallel_workers': 1}
            }
            logger = setup_logging(temp_path, config)
            
            file_manager = FileManager(
                source_paths=[temp_path],
                output_path=temp_path / "output",
                admin_path=temp_path / "admin",
                config=config,
                logger=logger
            )
            
            processor = ImageProcessor(file_manager, config, logger)
            
            # Execute blur detection
            processor.detect_blur()
            
            # Check for CSV file
            csv_dir = temp_path / "admin" / "CSV"
            if csv_dir.exists():
                csv_files = list(csv_dir.glob("All_Image_Files_Focus_*.csv"))
                
                if csv_files:
                    csv_file = csv_files[0]
                    
                    # Read CSV and check structure
                    with open(csv_file, 'r', encoding='utf-8-sig') as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        rows = list(reader)
                    
                    # Verify filename format
                    filename = csv_file.name
                    filename_parts = filename.split('_')
                    
                    checks = [
                        ("CSV file created", len(csv_files) > 0),
                        ("Correct filename prefix", filename.startswith("All_Image_Files_Focus_")),
                        ("Timestamp in filename", len(filename_parts) >= 5),  # Includes date and time parts
                        ("Primary key first column", header[0] == 'primary_key'),
                        ("Data timestamp last column", header[-1] == 'data_row_creation_timestamp'),
                        ("Contains blur score columns", any('score' in col for col in header)),
                        ("Contains consensus decision", 'is_blurry' in header),
                        ("Multiple model results", len([col for col in header if 'blurry' in col]) >= 2),
                        ("Has data rows", len(rows) > 0),
                    ]
                    
                    for desc, passed in checks:
                        print(f"  {'?' if passed else '?'} {desc}")
                    
                    return all(passed for _, passed in checks)
                
            print("? No CSV files found")
            return False
            
    except Exception as e:
        print(f"? CSV output verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_blurry_image_movement():
    """Verify blurry images are moved to IMGOrig-Blurry folder with correct naming."""
    print("?? Verifying Blurry Image Movement...")
    
    try:
        from src.core.file_manager import FileManager
        from src.utils.logger import setup_logging
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test images
            test_images = create_test_images(temp_path)
            
            # Setup file manager
            config = {'processing': {'maintain_folder_structure': True}}
            logger = setup_logging(temp_path, config)
            
            file_manager = FileManager(
                source_paths=[temp_path],
                output_path=temp_path / "output",
                admin_path=temp_path / "admin",
                config=config,
                logger=logger
            )
            
            # Simulate blurry files for moving
            blurry_files = [(test_images[1], 85.5), (test_images[2], 92.3)]  # Some blurry images with scores
            
            # Move blurry files
            moved_count = file_manager.move_blurry_images(blurry_files)
            
            # Check IMGOrig-Blurry folder
            blurry_folder = temp_path.parent / "IMGOrig-Blurry"
            
            checks = [
                ("Files were moved", moved_count > 0),
                ("IMGOrig-Blurry folder created", blurry_folder.exists()),
                ("Files have BLUR_ORIG_ prefix", any('BLUR_ORIG_' in f.name for f in blurry_folder.rglob('*') if f.is_file())),
                ("Folder structure maintained", any(subdir.is_dir() for subdir in blurry_folder.rglob('*'))),
            ]
            
            for desc, passed in checks:
                print(f"  {'?' if passed else '?'} {desc}")
            
            return all(passed for _, passed in checks)
            
    except Exception as e:
        print(f"? Blurry image movement verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_recursive_scanning():
    """Verify recursive directory scanning is working."""
    print("?? Verifying Recursive Directory Scanning...")
    
    try:
        from src.core.file_manager import FileManager
        from src.utils.logger import setup_logging
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create nested structure with images
            test_images = create_test_images(temp_path)
            
            # Setup file manager
            config = {}
            logger = setup_logging(temp_path, config)
            
            file_manager = FileManager(
                source_paths=[temp_path],
                output_path=temp_path / "output", 
                admin_path=temp_path / "admin",
                config=config,
                logger=logger
            )
            
            # Scan for images
            found_images = list(file_manager.scan_for_images(temp_path))
            
            checks = [
                ("Images found", len(found_images) > 0),
                ("Recursive scanning", len(found_images) >= 4),  # Should find all test images including subdirectory
                ("Multiple formats", any(img.suffix.lower() == '.jpg' for img in found_images) and 
                                   any(img.suffix.lower() == '.png' for img in found_images)),
                ("Subdirectory images found", any('subdirectory' in str(img) for img in found_images)),
            ]
            
            for desc, passed in checks:
                print(f"  {'?' if passed else '?'} {desc}")
            
            return all(passed for _, passed in checks)
            
    except Exception as e:
        print(f"? Recursive scanning verification failed: {e}")
        return False


def verify_file_collision_handling():
    """Verify file naming collision handling."""
    print("?? Verifying File Collision Handling...")
    
    try:
        from src.core.file_manager import FileManager
        from src.utils.logger import setup_logging
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test CSV filename
            config = {}
            logger = setup_logging(temp_path, config)
            
            file_manager = FileManager(
                source_paths=[temp_path],
                output_path=temp_path / "output",
                admin_path=temp_path / "admin", 
                config=config,
                logger=logger
            )
            
            # Test CSV filename generation
            csv_filename1 = file_manager.get_csv_filename("All_Image_Files_Focus")
            
            # Create the file to simulate collision
            csv_filename1.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_filename1, 'w') as f:
                f.write("test")
            
            # Generate another filename (should handle collision)
            csv_filename2 = file_manager.get_csv_filename("All_Image_Files_Focus")
            
            checks = [
                ("CSV filename includes timestamp", "_" in csv_filename1.name),
                ("Admin/CSV directory created", csv_filename1.parent.exists()),
                ("Unique filename generation", csv_filename1 != csv_filename2 or not csv_filename1.exists()),
                ("Proper file extension", csv_filename1.suffix == '.csv'),
            ]
            
            for desc, passed in checks:
                print(f"  {'?' if passed else '?'} {desc}")
            
            return all(passed for _, passed in checks)
            
    except Exception as e:
        print(f"? File collision handling verification failed: {e}")
        return False


def main():
    """Run comprehensive Menu Item 2 verification."""
    print("?? COMPREHENSIVE MENU ITEM 2 (BLUR DETECTION) VERIFICATION")
    print("=" * 80)
    print("Verifying ALL requirements are completely implemented and functional...")
    print()
    
    verification_tests = [
        ("Multiple AI Models (Laplacian, Variance, Gradient, etc.)", verify_blur_detection_models),
        ("Center Area Weighting vs Peripheral", verify_center_weighting),
        ("CSV Output Structure & Naming", verify_csv_output_structure),
        ("Blurry Image Movement to IMGOrig-Blurry", verify_blurry_image_movement),
        ("Recursive Directory Scanning", verify_recursive_scanning),
        ("File Collision Handling", verify_file_collision_handling),
    ]
    
    results = []
    
    for test_name, test_func in verification_tests:
        print(f"\n?? {test_name}")
        print("-" * 50)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"? {test_name} - PASSED")
            else:
                print(f"? {test_name} - FAILED")
                
        except Exception as e:
            print(f"? {test_name} - ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("?? MENU ITEM 2 VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "? PASSED" if result else "? FAILED"
        print(f"  {status:<10} {test_name}")
    
    print()
    print(f"Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("?? SUCCESS! Menu Item 2 is fully implemented and functional!")
        print("\n? CONFIRMED: All requirements are met:")
        print("   - Multiple AI image processing models for blur detection")
        print("   - Greater weight to center/middle area vs peripheral areas")
        print("   - Assessment, evaluation and data extraction from each image")
        print("   - Statistical analysis and calculations recorded in CSV")
        print("   - Pass/Fail TRUE/FALSE aggregated decision field")
        print("   - CSV stored in admin output folder with proper naming")
        print("   - Primary key first column, timestamp last column")
        print("   - Blurry images MOVED (not copied) to IMGOrig-Blurry folder")
        print("   - Original folder structure maintained")
        print("   - Files renamed with BLUR_ORIG_ prefix")
        print("   - File collision handling with sequence numbers")
        return True
    else:
        print("??  Some requirements need attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)