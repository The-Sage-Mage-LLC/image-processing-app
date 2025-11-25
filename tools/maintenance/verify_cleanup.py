#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Workspace Verification
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Verify the cleaned and organized workspace structure.
"""

import os
from pathlib import Path
from datetime import datetime


def verify_essential_files():
    """Verify all essential files are present in root."""
    essential_files = [
        "main.py",
        "gui_launcher.py",
        "launch_gui.bat", 
        "requirements.txt",
        "requirements_modern.txt",
        "pyproject.toml",
        "README.md",
        ".gitignore",
        ".pre-commit-config.yaml",
        "docker-compose.yml"
    ]
    
    print("VERIFYING ESSENTIAL FILES")
    print("-" * 40)
    
    missing_files = []
    for file_name in essential_files:
        if Path(file_name).exists():
            print(f"PASS {file_name}")
        else:
            print(f"FAIL {file_name} - MISSING")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\nMissing {len(missing_files)} essential files!")
        return False
    else:
        print(f"\nAll {len(essential_files)} essential files present")
        return True


def verify_directory_structure():
    """Verify the organized directory structure."""
    required_dirs = [
        "src",
        "config", 
        "docs",
        "tools",
        "scripts",
        "archive"
    ]
    
    print("\nVERIFYING DIRECTORY STRUCTURE")
    print("-" * 40)
    
    missing_dirs = []
    for dir_name in required_dirs:
        if Path(dir_name).is_dir():
            print(f"PASS {dir_name}/")
        else:
            print(f"FAIL {dir_name}/ - MISSING")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\nMissing {len(missing_dirs)} directories!")
        return False
    else:
        print(f"\nAll {len(required_dirs)} required directories present")
        return True


def verify_src_structure():
    """Verify source code structure."""
    src_modules = [
        "src/__init__.py",
        "src/cli",
        "src/core",
        "src/gui", 
        "src/models",
        "src/transforms",
        "src/utils",
        "src/web"
    ]
    
    print("\nVERIFYING SOURCE STRUCTURE")
    print("-" * 40)
    
    issues = []
    for module in src_modules:
        module_path = Path(module)
        if module.endswith('.py'):
            if module_path.exists():
                print(f"PASS {module}")
            else:
                print(f"FAIL {module} - MISSING")
                issues.append(module)
        else:
            if module_path.is_dir():
                init_file = module_path / "__init__.py"
                if init_file.exists():
                    print(f"PASS {module}/ (with __init__.py)")
                else:
                    print(f"WARN {module}/ (missing __init__.py)")
                    issues.append(f"{module}/__init__.py")
            else:
                print(f"FAIL {module}/ - MISSING")
                issues.append(module)
    
    if issues:
        print(f"\nFound {len(issues)} issues in source structure")
        return False
    else:
        print(f"\nSource structure is properly organized")
        return True


def count_root_files():
    """Count files remaining in root directory."""
    print("\nROOT DIRECTORY ANALYSIS")
    print("-" * 40)
    
    root_files = [f for f in Path(".").iterdir() if f.is_file() and not f.name.startswith('.git')]
    
    print(f"Total files in root: {len(root_files)}")
    print("Files:")
    for file_path in sorted(root_files):
        print(f"  - {file_path.name}")
    
    if len(root_files) <= 15:  # Reasonable number for a clean project
        print("\nRoot directory is clean and organized")
        return True
    else:
        print(f"\nRoot directory has {len(root_files)} files (consider further cleanup)")
        return False


def main():
    """Main verification function."""
    print("WORKSPACE CLEANUP VERIFICATION")
    print(f"Verification Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Project: Image Processing App 20251119")
    print("=" * 60)
    
    # Run verifications
    results = []
    results.append(("Essential Files", verify_essential_files()))
    results.append(("Directory Structure", verify_directory_structure())) 
    results.append(("Source Structure", verify_src_structure()))
    results.append(("Root Directory", count_root_files()))
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results:
        status = "PASS" if passed_test else "FAIL"
        print(f"{status} {test_name}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\nWORKSPACE CLEANUP SUCCESSFUL!")
        print("Your project is clean, organized, and production-ready!")
    elif passed >= total * 0.8:
        print("\nWORKSPACE CLEANUP MOSTLY SUCCESSFUL!")
        print("Minor issues detected but overall well-organized.")
    else:
        print("\nWORKSPACE CLEANUP NEEDS ATTENTION")
        print("Several issues detected that should be addressed.")
    
    return 0 if passed >= total * 0.8 else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)