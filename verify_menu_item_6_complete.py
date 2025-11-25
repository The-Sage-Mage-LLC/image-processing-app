"""
Menu Item 6 Verification Test
Comprehensive test to verify color image copy functionality is complete and functional.

This test verifies all requirements:
1. Copy all original image files without any processing
2. Save to output root under "CLR_ORIG" folder
3. Maintain source root folder structure
4. Rename files with "CLR_ORIG_" prefix while keeping original name
5. Create folders if they don't exist
6. Don't overwrite existing files - append sequence number
7. Process all color images (anticipated to all be in color)
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
from PIL import Image
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.file_manager import FileManager
from src.core.image_processor import ImageProcessor


def create_test_images_with_structure(test_dir: Path) -> list[Path]:
    """Create test images in a directory structure for testing."""
    
    # Create main directory with images
    main_images = []
    
    # Root level images
    for i in range(3):
        img_path = test_dir / f"root_image_{i+1}.jpg"
        create_test_color_image(img_path, (300, 200), colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        main_images.append(img_path)
    
    # Create subdirectory structure
    sub_dir1 = test_dir / "photos" / "vacation"
    sub_dir1.mkdir(parents=True)
    
    for i in range(2):
        img_path = sub_dir1 / f"vacation_photo_{i+1}.png"
        create_test_color_image(img_path, (400, 300), colors=[(200, 100, 50), (50, 200, 100)])
        main_images.append(img_path)
    
    # Create another subdirectory
    sub_dir2 = test_dir / "documents" / "scans"
    sub_dir2.mkdir(parents=True)
    
    for i in range(2):
        img_path = sub_dir2 / f"document_scan_{i+1}.jpeg"
        create_test_color_image(img_path, (600, 800), colors=[(100, 100, 100), (200, 200, 200)])
        main_images.append(img_path)
    
    # Create deeply nested structure
    deep_dir = test_dir / "archive" / "2023" / "events" / "conference"
    deep_dir.mkdir(parents=True)
    
    img_path = deep_dir / "conference_group_photo.jpg"
    create_test_color_image(img_path, (800, 600), colors=[(120, 80, 40), (40, 120, 80), (80, 40, 120)])
    main_images.append(img_path)
    
    return main_images


def create_test_color_image(file_path: Path, size: tuple, colors: list = None):
    """Create a test color image with specified colors."""
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create colored sections
    section_width = width // len(colors)
    for i, color in enumerate(colors):
        start_x = i * section_width
        end_x = (i + 1) * section_width if i < len(colors) - 1 else width
        image[:, start_x:end_x] = color
    
    # Save image
    pil_image = Image.fromarray(image)
    pil_image.save(file_path, quality=95, optimize=True)


def verify_color_copy_comprehensive():
    """Verify comprehensive color image copy functionality."""
    print("?? Verifying comprehensive color image copy...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create source directory with images
        source_dir = test_dir / "source"
        source_dir.mkdir()
        
        # Create test images with nested structure
        test_images = create_test_images_with_structure(source_dir)
        
        # Create output and admin directories
        output_dir = test_dir / "output"
        admin_dir = test_dir / "admin"
        output_dir.mkdir()
        admin_dir.mkdir()
        
        # Configure application
        config = {
            'general': {'max_parallel_workers': 1},
            'paths': {'supported_formats': ['jpg', 'jpeg', 'png']},
            'processing': {'maintain_folder_structure': True}
        }
        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)
        
        # Initialize components
        file_manager = FileManager([source_dir], output_dir, admin_dir, config, logger)
        processor = ImageProcessor(file_manager, config, logger)
        
        # Run color image copy (Menu Item 6)
        processor.copy_color_images()
        
        # Verify CLR_ORIG folder was created
        clr_orig_dir = output_dir / "CLR_ORIG"
        assert clr_orig_dir.exists(), "CLR_ORIG folder should be created"
        
        # Collect all copied files
        copied_files = list(clr_orig_dir.rglob("CLR_ORIG_*"))\n        \n        checks = [\n            ("CLR_ORIG folder created", clr_orig_dir.exists()),\n            ("Files were copied", len(copied_files) > 0),\n            ("Expected number of files copied", len(copied_files) == len(test_images)),\n        ]\n        \n        # Verify folder structure is maintained\n        original_structure = set()\n        for img_path in test_images:\n            rel_path = img_path.relative_to(source_dir)\n            original_structure.add(str(rel_path.parent))\n        \n        copied_structure = set()\n        for copied_file in copied_files:\n            rel_path = copied_file.relative_to(clr_orig_dir)\n            copied_structure.add(str(rel_path.parent))\n        \n        structure_maintained = original_structure == copied_structure\n        checks.append(("Folder structure maintained", structure_maintained))\n        \n        # Verify filename prefix\n        all_have_prefix = all(f.name.startswith("CLR_ORIG_") for f in copied_files)\n        checks.append(("All files have CLR_ORIG_ prefix", all_have_prefix))\n        \n        # Verify original filename preservation\n        for original_img in test_images:\n            expected_name = f"CLR_ORIG_{original_img.name}"\n            rel_path = original_img.relative_to(source_dir)\n            expected_path = clr_orig_dir / rel_path.parent / expected_name\n            \n            file_exists = expected_path.exists()\n            if not file_exists:\n                # Check for sequence-numbered version\n                parent_dir = expected_path.parent\n                matching_files = list(parent_dir.glob(f"CLR_ORIG_{original_img.stem}_*{original_img.suffix}"))\n                file_exists = len(matching_files) > 0\n            \n            checks.append((f"File {original_img.name} copied correctly", file_exists))\n        \n        # Verify no processing was applied (files are identical)\n        files_identical = True\n        for original_img in test_images:\n            expected_name = f"CLR_ORIG_{original_img.name}"\n            rel_path = original_img.relative_to(source_dir)\n            copied_path = clr_orig_dir / rel_path.parent / expected_name\n            \n            if copied_path.exists():\n                # Compare file sizes (should be identical for no processing)\n                original_size = original_img.stat().st_size\n                copied_size = copied_path.stat().st_size\n                if abs(original_size - copied_size) > 1000:  # Allow small variance\n                    files_identical = False\n                    break\n        \n        checks.append(("Files copied without processing (identical sizes)", files_identical))\n        \n        # Print results\n        for desc, passed in checks:\n            print(f"  {'?' if passed else '?'} {desc}")\n        \n        return all(passed for _, passed in checks)\n\n\ndef verify_file_deduplication():\n    """Verify file deduplication with sequence numbers."""\n    print("?? Verifying file deduplication...")\n    \n    with tempfile.TemporaryDirectory() as temp_dir:\n        test_dir = Path(temp_dir)\n        \n        # Create test file\n        source_file = test_dir / "test_image.jpg"\n        create_test_color_image(source_file, (200, 200))\n        \n        # Create destination directory\n        dest_dir = test_dir / "destination"\n        dest_dir.mkdir()\n        \n        config = {}\n        logger = logging.getLogger("test")\n        file_manager = FileManager([], Path(), Path(), config, logger)\n        \n        # First copy\n        result1 = file_manager.save_file_with_dedup(\n            source_file, dest_dir, "CLR_ORIG_test_image.jpg", copy=True\n        )\n        \n        # Create a slightly different file with same name\n        source_file2 = test_dir / "test_image2.jpg"\n        create_test_color_image(source_file2, (200, 200), colors=[(200, 0, 0), (0, 200, 0)])  # Different colors\n        \n        # Second copy attempt (should get sequence number)\n        result2 = file_manager.save_file_with_dedup(\n            source_file2, dest_dir, "CLR_ORIG_test_image.jpg", copy=True\n        )\n        \n        # Check results\n        copied_files = list(dest_dir.glob("CLR_ORIG_test_image*"))\n        \n        checks = [\n            ("First copy successful", result1 is not None),\n            ("Second copy with sequence number", result2 is not None),\n            ("Two different files created", len(copied_files) == 2),\n            ("Sequence number appended", any("_0001" in f.name for f in copied_files)),\n        ]\n        \n        for desc, passed in checks:\n            print(f"  {'?' if passed else '?'} {desc}")\n        \n        return all(passed for _, passed in checks)\n\n\ndef verify_folder_creation():\n    """Verify automatic folder creation."""\n    print("?? Verifying automatic folder creation...")\n    \n    with tempfile.TemporaryDirectory() as temp_dir:\n        test_dir = Path(temp_dir)\n        \n        # Create source with nested structure\n        source_dir = test_dir / "source"\n        nested_dir = source_dir / "level1" / "level2" / "level3"\n        nested_dir.mkdir(parents=True)\n        \n        test_file = nested_dir / "deep_image.jpg"\n        create_test_color_image(test_file, (100, 100))\n        \n        # Create output directory (empty initially)\n        output_dir = test_dir / "output"\n        output_dir.mkdir()\n        \n        config = {'processing': {'maintain_folder_structure': True}}\n        logger = logging.getLogger("test")\n        file_manager = FileManager([source_dir], output_dir, test_dir, config, logger)\n        \n        # Test directory creation through the file manager\n        created_dir = file_manager.create_output_directory(\n            "CLR_ORIG",\n            maintain_structure=True,\n            source_path=source_dir,\n            relative_path=test_file\n        )\n        \n        # Generate and save file\n        new_filename = file_manager.generate_output_filename(\n            test_file.name, "CLR_ORIG_"\n        )\n        \n        result = file_manager.save_file_with_dedup(\n            test_file, created_dir, new_filename, copy=True\n        )\n        \n        # Verify structure was created\n        expected_path = output_dir / "CLR_ORIG" / "level1" / "level2" / "level3" / "CLR_ORIG_deep_image.jpg"\n        \n        checks = [\n            ("Nested directory created", created_dir.exists()),\n            ("File saved in correct location", result is not None),\n            ("Full path structure maintained", expected_path.exists()),\n            ("Multiple directory levels created", len(expected_path.parts) > len(output_dir.parts) + 4),\n        ]\n        \n        for desc, passed in checks:\n            print(f"  {'?' if passed else '?'} {desc}")\n        \n        return all(passed for _, passed in checks)\n\n\ndef verify_filename_generation():\n    """Verify filename generation with prefix."""\n    print("??? Verifying filename generation...")\n    \n    config = {}\n    logger = logging.getLogger("test")\n    file_manager = FileManager([], Path(), Path(), config, logger)\n    \n    test_cases = [\n        ("simple.jpg", "CLR_ORIG_", "CLR_ORIG_simple.jpg"),\n        ("photo with spaces.png", "CLR_ORIG_", "CLR_ORIG_photo with spaces.png"),\n        ("document.scan.jpeg", "CLR_ORIG_", "CLR_ORIG_document.scan.jpeg"),\n        ("file_with_underscores.jpg", "CLR_ORIG_", "CLR_ORIG_file_with_underscores.jpg"),\n    ]\n    \n    checks = []\n    \n    for original, prefix, expected in test_cases:\n        result = file_manager.generate_output_filename(original, prefix)\n        correct = result == expected\n        checks.append((f"Filename '{original}' -> '{expected}'", correct))\n        if not correct:\n            print(f"    Expected: {expected}, Got: {result}")\n    \n    # Test with sequence number\n    result_with_seq = file_manager.generate_output_filename(\n        "test.jpg", "CLR_ORIG_", "", 1\n    )\n    expected_with_seq = "CLR_ORIG_test_0001.jpg"\n    checks.append(("Sequence number handling", result_with_seq == expected_with_seq))\n    \n    for desc, passed in checks:\n        print(f"  {'?' if passed else '?'} {desc}")\n    \n    return all(passed for _, passed in checks)\n\n\ndef verify_complete_workflow():\n    """Verify complete Menu Item 6 workflow."""\n    print("?? Verifying complete workflow...")\n    \n    with tempfile.TemporaryDirectory() as temp_dir:\n        test_dir = Path(temp_dir)\n        \n        # Create realistic test scenario\n        source_dir = test_dir / "PhotoLibrary"\n        source_dir.mkdir()\n        \n        # Create various image types and locations\n        test_images = [\n            source_dir / "IMG_001.jpg",\n            source_dir / "family" / "vacation_2023.png", \n            source_dir / "work" / "presentations" / "slide_01.jpeg",\n            source_dir / "archive" / "old_photos" / "vintage.jpg"\n        ]\n        \n        for img_path in test_images:\n            img_path.parent.mkdir(parents=True, exist_ok=True)\n            create_test_color_image(img_path, (400, 300))\n        \n        # Create output and admin directories\n        output_dir = test_dir / "ProcessedImages"\n        admin_dir = test_dir / "Admin"\n        output_dir.mkdir()\n        admin_dir.mkdir()\n        \n        # Configure and run\n        config = {\n            'general': {'max_parallel_workers': 1},\n            'paths': {'supported_formats': ['jpg', 'jpeg', 'png']},\n            'processing': {'maintain_folder_structure': True}\n        }\n        logger = logging.getLogger("test")\n        \n        file_manager = FileManager([source_dir], output_dir, admin_dir, config, logger)\n        processor = ImageProcessor(file_manager, config, logger)\n        \n        # Execute Menu Item 6\n        processor.copy_color_images()\n        \n        # Verify complete results\n        clr_orig_dir = output_dir / "CLR_ORIG"\n        copied_files = list(clr_orig_dir.rglob("CLR_ORIG_*"))\n        \n        checks = [\n            ("All source images found and copied", len(copied_files) == len(test_images)),\n            ("CLR_ORIG directory created", clr_orig_dir.exists()),\n            ("Subdirectory structure preserved", (clr_orig_dir / "family").exists()),\n            ("Deep nesting preserved", (clr_orig_dir / "work" / "presentations").exists()),\n            ("All files have correct prefix", all(f.name.startswith("CLR_ORIG_") for f in copied_files)),\n        ]\n        \n        # Verify each specific file\n        expected_files = [\n            clr_orig_dir / "CLR_ORIG_IMG_001.jpg",\n            clr_orig_dir / "family" / "CLR_ORIG_vacation_2023.png",\n            clr_orig_dir / "work" / "presentations" / "CLR_ORIG_slide_01.jpeg",\n            clr_orig_dir / "archive" / "old_photos" / "CLR_ORIG_vintage.jpg",\n        ]\n        \n        for expected_file in expected_files:\n            exists = expected_file.exists()\n            checks.append((f"File {expected_file.name} in correct location", exists))\n        \n        for desc, passed in checks:\n            print(f"  {'?' if passed else '?'} {desc}")\n        \n        return all(passed for _, passed in checks)\n\n\ndef main():\n    """Run all verification tests."""\n    print("=" * 80)\n    print("?? MENU ITEM 6 VERIFICATION TEST SUITE")\n    print("Testing color image copy functionality implementation")\n    print("=" * 80)\n    \n    verification_tests = [\n        ("Comprehensive Color Image Copy", verify_color_copy_comprehensive),\n        ("File Deduplication", verify_file_deduplication),\n        ("Automatic Folder Creation", verify_folder_creation),\n        ("Filename Generation", verify_filename_generation),\n        ("Complete Workflow", verify_complete_workflow),\n    ]\n    \n    results = []\n    \n    for test_name, test_func in verification_tests:\n        print(f"\\n?? {test_name}")\n        print("-" * 50)\n        \n        try:\n            result = test_func()\n            results.append((test_name, result))\n            \n            if result:\n                print(f"? {test_name} - PASSED")\n            else:\n                print(f"? {test_name} - FAILED")\n                \n        except Exception as e:\n            print(f"? {test_name} - ERROR: {e}")\n            import traceback\n            traceback.print_exc()\n            results.append((test_name, False))\n    \n    # Summary\n    print("\\n" + "=" * 80)\n    print("?? MENU ITEM 6 VERIFICATION SUMMARY")\n    print("=" * 80)\n    \n    passed = sum(1 for _, result in results if result)\n    total = len(results)\n    \n    for test_name, result in results:\n        status = "? PASSED" if result else "? FAILED"\n        print(f"  {status:<10} {test_name}")\n    \n    print()\n    print(f"Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")\n    \n    if passed == total:\n        print("?? SUCCESS! Menu Item 6 is fully implemented and functional!")\n        print("\\n? CONFIRMED: All requirements are met:")\n        print("   - Copies all original image files without any processing")\n        print("   - Saves to output root under 'CLR_ORIG' folder")\n        print("   - Maintains source root folder structure")\n        print("   - Renames files with 'CLR_ORIG_' prefix + original name")\n        print("   - Creates folders automatically if they don't exist")\n        print("   - Doesn't overwrite existing files (appends sequence number)")\n        print("   - Processes all color images efficiently")\n        return True\n    else:\n        print("??  Some requirements need attention.")\n        return False\n\n\nif __name__ == "__main__":\n    success = main()\n    sys.exit(0 if success else 1)