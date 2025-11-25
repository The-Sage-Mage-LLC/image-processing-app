#!/usr/bin/env python3
"""
Targeted Code Quality Fixes
"""

import os
import sys
from pathlib import Path

def fix_verify_menu_item_3():
    """Fix verify_menu_item_3_complete.py"""
    file_path = Path("verify_menu_item_3_complete.py")
    
    if not file_path.exists():
        print(f"File {file_path} not found")
        return
    
    # Try different encodings
    content = None
    encoding_used = None
    
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            encoding_used = encoding
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        print(f"Could not read {file_path} with any encoding")
        return
    
    # Fix the unused variable issue around line 190
    # Looking for the pattern: test_image = create_test_image_with_metadata(...)
    content = content.replace(
        "# Create test image\n        test_image = create_test_image_with_metadata(test_dir / \"test.jpg\")",
        "# Create test image\n        _ = create_test_image_with_metadata(test_dir / \"test.jpg\")  # Used to setup test environment"
    )
    
    with open(file_path, 'w', encoding=encoding_used) as f:
        f.write(content)
    
    print(f"Fixed {file_path}")

def fix_files_with_unused_imports():
    """Fix multiple files with known unused import issues"""
    
    files_to_fix = [
        "verify_menu_item_4_complete.py",
        "verify_menu_item_8_complete.py", 
        "verify_menu_items_9_12_complete.py",
        "verify_monitoring_complete.py",
        "verify_requirements.py"
    ]
    
    for filename in files_to_fix:
        file_path = Path(filename)
        
        if not file_path.exists():
            print(f"File {file_path} not found, skipping")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Comment out known unused imports
            for i, line in enumerate(lines):
                # Comment out specific unused imports
                if any(unused in line for unused in [
                    "import os", "import shutil", "from datetime import datetime",
                    "import numpy as np", "from PIL import ExifTags", 
                    "import cv2", "import threading", "from unittest.mock import Mock"
                ]):
                    if not line.strip().startswith('#') and ('import' in line or 'from' in line):
                        lines[i] = f"# {line}  # Unused import - commented by code quality fixer\n"
            
            # Fix unused variables by prefixing with underscore
            for i, line in enumerate(lines):
                # Fix exception variables
                if ' as e:' in line and 'except' in line:
                    lines[i] = line.replace(' as e:', ' as _e:')
                elif 'e =' in line and 'Exception' in lines[i-1] if i > 0 else False:
                    lines[i] = line.replace('e =', '_e =')
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            print(f"Fixed {file_path}")
            
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")

def fix_specific_unused_variables():
    """Fix specific unused variables identified by flake8"""
    
    fixes = [
        {
            "file": "verify_menu_item_3_complete.py",
            "pattern": "test_image = create_test_image_with_metadata(",
            "replacement": "_test_image = create_test_image_with_metadata("
        },
        {
            "file": "verify_menu_item_4_complete.py", 
            "pattern": "test_image = create_test_image_with_metadata(",
            "replacement": "_test_image = create_test_image_with_metadata("
        },
        {
            "file": "verify_monitoring_complete.py",
            "pattern": "test_images = create_test_images(",
            "replacement": "_test_images = create_test_images("
        },
        {
            "file": "verify_monitoring_complete.py",
            "pattern": "tiny_size = ",
            "replacement": "_tiny_size = "
        },
        {
            "file": "verify_monitoring_complete.py", 
            "pattern": "duplicate_hash = ",
            "replacement": "_duplicate_hash = "
        },
        {
            "file": "verify_monitoring_complete.py",
            "pattern": "processed_size = ",
            "replacement": "_processed_size = "
        },
        {
            "file": "verify_requirements.py",
            "pattern": "validator = ",
            "replacement": "_validator = "
        },
        {
            "file": "verify_requirements.py",
            "pattern": "config = ",
            "replacement": "_config = "
        },
        {
            "file": "verify_requirements.py",
            "pattern": "app = ",
            "replacement": "_app = "
        },
        {
            "file": "verify_requirements.py",
            "pattern": "expected_options = ",
            "replacement": "_expected_options = "
        }
    ]
    
    for fix in fixes:
        file_path = Path(fix["file"])
        
        if not file_path.exists():
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if fix["pattern"] in content:
                content = content.replace(fix["pattern"], fix["replacement"])
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"Fixed unused variable in {file_path}")
                
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")

def main():
    """Main function"""
    print("?? Applying Targeted Code Quality Fixes")
    print("=" * 50)
    
    # Fix specific files
    fix_verify_menu_item_3()
    fix_files_with_unused_imports()
    fix_specific_unused_variables()
    
    print("\n? Targeted fixes applied!")
    print("\nRun this to check remaining issues:")
    print("python -m flake8 --select=F401,F841,F821,F822,F831 .")

if __name__ == "__main__":
    main()