#!/usr/bin/env python3
"""
Quick Startup Verification Script
Project ID: Image Processing App 20251119
Created: 2025-01-19
Author: The-Sage-Mage

Run this before using the application to ensure everything is working.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version."""
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 11:
        print(f"? Python {major}.{minor} - OK")
        return True
    else:
        print(f"? Python {major}.{minor} - Requires Python 3.11+")
        return False

def check_dependencies():
    """Check core dependencies."""
    dependencies = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('sklearn', 'scikit-learn'),
        ('tomli', 'TOML parser'),
        ('tqdm', 'Progress bars'),
        ('click', 'CLI framework')
    ]
    
    all_good = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"? {name} - OK")
        except ImportError:
            print(f"? {name} - MISSING")
            all_good = False
    
    # Optional dependencies
    optional_deps = [
        ('PyQt6', 'PyQt6 (GUI support)'),
        ('torch', 'PyTorch (AI features)'),
        ('transformers', 'Transformers (AI models)')
    ]
    
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"? {name} - OK")
        except ImportError:
            print(f"?? {name} - Optional (some features may be limited)")
    
    return all_good

def check_config():
    """Check configuration file."""
    config_path = Path("config/config.toml")
    if config_path.exists():
        print("? Configuration file - OK")
        return True
    else:
        print("?? Configuration file - Missing (will use defaults)")
        return True

def check_directories():
    """Check directory structure."""
    required_dirs = [
        "src/cli",
        "src/core", 
        "src/transforms",
        "src/models",
        "src/utils",
        "src/gui"
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"? {dir_path} - OK")
        else:
            print(f"? {dir_path} - MISSING")
            all_good = False
    
    return all_good

def check_main_files():
    """Check main executable files."""
    main_files = [
        ("main.py", "Main launcher"),
        ("src/cli/main.py", "CLI interface"),
        ("src/gui/main_window.py", "GUI interface"),
        ("test_application.py", "Test suite")
    ]
    
    all_good = True
    
    for file_path, description in main_files:
        if Path(file_path).exists():
            print(f"? {description} - OK")
        else:
            print(f"? {description} - MISSING")
            all_good = False
    
    return all_good

def quick_import_test():
    """Quick import test of core modules."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test core imports
        from src.cli.main import CLIApp
        from src.core.file_manager import FileManager
        from src.core.image_processor import ImageProcessor
        
        print("? Core module imports - OK")
        return True
        
    except Exception as e:
        print(f"? Core module imports - FAILED: {e}")
        return False

def main():
    """Run all verification checks."""
    print("?? Image Processing Application - Startup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Configuration", check_config),
        ("Directory Structure", check_directories),
        ("Main Files", check_main_files),
        ("Module Imports", quick_import_test)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n?? Checking {check_name}:")
        result = check_func()
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("?? ALL CHECKS PASSED!")
        print("\n?? Ready to run:")
        print("   GUI: python main.py --gui")
        print("   CLI: python main.py --cli --source-paths \"C:\\Photos\" --output-path \"C:\\Output\" --admin-path \"C:\\Admin\" --menu-option 7")
        print("\n?? For full testing: python test_application.py")
    else:
        print("??  SOME CHECKS FAILED!")
        print("\n?? To fix issues:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Check file integrity")
        print("   3. Verify Python version")
    
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)