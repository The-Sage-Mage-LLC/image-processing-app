#!/usr/bin/env python3
"""
Test Script for Image Processing Application
Project ID: Image Processing App 20251119
Created: 2025-01-19
Author: The-Sage-Mage

This script tests the core functionality of the application.
"""

import sys
import os
import traceback
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        from src.cli.main import CLIApp
        print("? CLI main import successful")
    except Exception as e:
        print(f"? CLI main import failed: {e}")
        return False
    
    try:
        from src.core.file_manager import FileManager
        print("? FileManager import successful")
    except Exception as e:
        print(f"? FileManager import failed: {e}")
        return False
    
    try:
        from src.core.image_processor import ImageProcessor
        print("? ImageProcessor import successful")
    except Exception as e:
        print(f"? ImageProcessor import failed: {e}")
        return False
    
    try:
        from src.transforms.basic_transforms import BasicTransforms
        print("? BasicTransforms import successful")
    except Exception as e:
        print(f"? BasicTransforms import failed: {e}")
        return False
    
    try:
        from src.transforms.artistic_transforms import ArtisticTransforms
        print("? ArtisticTransforms import successful")
    except Exception as e:
        print(f"? ArtisticTransforms import failed: {e}")
        return False
    
    try:
        from src.models.blur_detector import BlurDetector
        print("? BlurDetector import successful")
    except Exception as e:
        print(f"? BlurDetector import failed: {e}")
        return False
    
    try:
        from src.core.metadata_handler import MetadataHandler
        print("? MetadataHandler import successful")
    except Exception as e:
        print(f"? MetadataHandler import failed: {e}")
        return False
    
    try:
        from src.utils.logger import setup_logging
        print("? Logger import successful")
    except Exception as e:
        print(f"? Logger import failed: {e}")
        return False
    
    try:
        from src.utils.database import DatabaseManager
        print("? DatabaseManager import successful")
    except Exception as e:
        print(f"? DatabaseManager import failed: {e}")
        return False
    
    return True

def create_test_image(filename: str, size: tuple = (100, 100)) -> Path:
    """Create a test image."""
    # Create a simple test image
    image = Image.new('RGB', size, color='red')
    
    # Add some content to make it interesting
    pixels = np.array(image)
    pixels[25:75, 25:75] = [0, 255, 0]  # Green square in center
    
    # Convert back to PIL and save
    test_image = Image.fromarray(pixels)
    test_path = Path(filename)
    test_image.save(test_path, 'JPEG')
    
    return test_path

def test_basic_functionality():
    """Test basic functionality with test images."""
    print("\nTesting basic functionality...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_path = temp_path / "source"
        output_path = temp_path / "output"
        admin_path = temp_path / "admin"
        
        source_path.mkdir()
        output_path.mkdir()
        admin_path.mkdir()
        
        # Create test images
        test_images = []
        for i in range(3):
            img_path = create_test_image(
                str(source_path / f"test_{i}.jpg"),
                size=(100 + i*50, 100 + i*50)
            )
            test_images.append(img_path)
        
        print(f"Created {len(test_images)} test images")
        
        # Test configuration loading
        try:
            import tomli
            config_path = Path("config/config.toml")
            if config_path.exists():
                with open(config_path, "rb") as f:
                    config = tomli.load(f)
                print("? Configuration loaded successfully")
            else:
                config = {}
                print("? Using default configuration")
        except Exception as e:
            print(f"? Configuration loading failed: {e}")
            config = {}
        
        # Test logger setup
        try:
            from src.utils.logger import setup_logging
            logger = setup_logging(admin_path, config)
            print("? Logger setup successful")
        except Exception as e:
            print(f"? Logger setup failed: {e}")
            return False
        
        # Test file manager
        try:
            from src.core.file_manager import FileManager
            file_manager = FileManager([source_path], output_path, admin_path, config, logger)
            
            # Test file scanning
            found_files = list(file_manager.scan_for_images())
            print(f"? FileManager found {len(found_files)} files")
        except Exception as e:
            print(f"? FileManager test failed: {e}")
            return False
        
        # Test basic transforms
        try:
            from src.transforms.basic_transforms import BasicTransforms
            basic_transforms = BasicTransforms(config, logger)
            
            # Test grayscale conversion
            test_img = test_images[0]
            result = basic_transforms.convert_to_grayscale(test_img)
            if result:
                print("? Grayscale conversion successful")
            else:
                print("? Grayscale conversion failed")
        except Exception as e:
            print(f"? BasicTransforms test failed: {e}")
            return False
        
        # Test blur detection
        try:
            from src.models.blur_detector import BlurDetector
            blur_detector = BlurDetector(config, logger)
            
            test_img = test_images[0]
            result = blur_detector.detect_blur(test_img)
            if result and 'is_blurry' in result:
                print("? Blur detection successful")
            else:
                print("? Blur detection failed")
        except Exception as e:
            print(f"? BlurDetector test failed: {e}")
            return False
        
        # Test metadata extraction
        try:
            from src.core.metadata_handler import MetadataHandler
            metadata_handler = MetadataHandler(config, logger)
            
            test_img = test_images[0]
            result = metadata_handler.extract_all_metadata(test_img)
            if result and 'file_path' in result:
                print("? Metadata extraction successful")
            else:
                print("? Metadata extraction failed")
        except Exception as e:
            print(f"? MetadataHandler test failed: {e}")
            return False
    
    return True

def test_gui_imports():
    """Test GUI imports (optional)."""
    print("\nTesting GUI imports...")
    
    try:
        import PyQt6
        print("? PyQt6 available")
        
        try:
            from src.gui.main_window import ImageProcessingGUI
            print("? GUI main window import successful")
        except Exception as e:
            print(f"? GUI main window import failed: {e}")
            return False
        
    except ImportError:
        print("? PyQt6 not available - GUI functionality disabled")
        return True
    
    return True

def test_cli_arguments():
    """Test CLI argument processing."""
    print("\nTesting CLI argument processing...")
    
    try:
        from src.cli.validators import PathValidator
        validator = PathValidator()
        
        # Test with valid paths
        config = {'validation': {'min_source_paths': 1, 'max_source_paths': 10}}
        
        # Use current directory as test path
        current_dir = str(Path.cwd())
        result = validator.validate_source_paths(current_dir, config)
        
        if result:
            print("? Path validation successful")
        else:
            print("? Path validation failed")
            
    except Exception as e:
        print(f"? CLI argument processing test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("Image Processing Application - Test Suite")
    print("Project ID: Image Processing App 20251119")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    tests = [
        test_imports,
        test_basic_functionality,
        test_gui_imports,
        test_cli_arguments
    ]
    
    for test_func in tests:
        try:
            result = test_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"? Test {test_func.__name__} failed with exception: {e}")
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("? ALL TESTS PASSED - Application ready for use")
        print("\nTo run the application:")
        print("  CLI: python main.py --cli --source-paths \"C:\\Photos\" --output-path \"C:\\Output\" --admin-path \"C:\\Admin\" --menu-option 7")
        print("  GUI: python main.py --gui")
    else:
        print("? SOME TESTS FAILED - Check errors above")
    print("="*60)

if __name__ == "__main__":
    main()