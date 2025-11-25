#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Unicode Encoding Issue Detection and Fix Script
"""

import os
import sys
from pathlib import Path
import re

def fix_file_encoding_issues(file_path):
    """Fix encoding issues in a single file."""
    try:
        # Try reading with different encodings
        content = None
        encoding_used = None
        
        for encoding in ['utf-8', 'cp1252', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                encoding_used = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"Could not read {file_path}")
            return False
        
        original_content = content
        
        # Replace problematic characters
        replacements = {
            '\u2022': '-',    # bullet point
            '\u2019': "'",    # right single quote
            '\u201c': '"',    # left double quote
            '\u201d': '"',    # right double quote
            '\u2013': '-',    # en dash
            '\u2014': '--',   # em dash
            '\u2026': '...',  # ellipsis
        }
        
        for unicode_char, replacement in replacements.items():
            if unicode_char in content:
                content = content.replace(unicode_char, replacement)
                print(f"  Fixed Unicode character in {file_path.name}")
        
        # Fix line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Write back with UTF-8 encoding if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def scan_and_fix_project():
    """Scan and fix Unicode issues in the entire project."""
    project_root = Path(__file__).parent
    fixed_count = 0
    
    print("Scanning for Unicode encoding issues...")
    
    # Find all Python files
    python_files = []
    for py_file in project_root.rglob("*.py"):
        if not any(part.startswith(('.', '__pycache__', 'venv', 'env')) for part in py_file.parts):
            python_files.append(py_file)
    
    print(f"Found {len(python_files)} Python files to check")
    
    # Process each file
    for py_file in python_files:
        if fix_file_encoding_issues(py_file):
            fixed_count += 1
            print(f"Fixed: {py_file}")
    
    # Also check text files
    text_files = []
    for ext in ['.md', '.txt', '.toml', '.yml', '.yaml']:
        for text_file in project_root.rglob(f"*{ext}"):
            if not any(part.startswith(('.', '__pycache__', 'venv', 'env')) for part in text_file.parts):
                text_files.append(text_file)
    
    print(f"Found {len(text_files)} text files to check")
    
    for text_file in text_files:
        if fix_file_encoding_issues(text_file):
            fixed_count += 1
            print(f"Fixed: {text_file}")
    
    print(f"\nUnicode encoding fix complete: {fixed_count} files fixed")
    return fixed_count

def test_comprehensive_qaqc():
    """Test if the comprehensive QA/QC file can run now."""
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'comprehensive_qaqc_verification.py'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("SUCCESS: comprehensive_qaqc_verification.py runs without Unicode errors")
            return True
        else:
            print(f"comprehensive_qaqc_verification.py failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error testing QA/QC script: {e}")
        return False

def main():
    """Main function."""
    print("UNICODE ENCODING ISSUE REMEDIATION")
    print("=" * 50)
    
    # Fix Unicode issues
    fixed_count = scan_and_fix_project()
    
    # Test the QA/QC script
    print("\nTesting fixed files...")
    qaqc_success = test_comprehensive_qaqc()
    
    print("\nSUMMARY:")
    print(f"Files fixed: {fixed_count}")
    print(f"QA/QC script working: {qaqc_success}")
    
    if qaqc_success:
        print("\nAll Unicode encoding issues have been resolved!")
        return 0
    else:
        print("\nSome issues may remain.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)