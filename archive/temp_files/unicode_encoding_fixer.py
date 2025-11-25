#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unicode Encoding Issues Detection and Remediation Script
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

This script searches for and fixes Unicode encoding issues throughout the entire codebase.
Handles common Unicode problems including:
- Invalid UTF-8 byte sequences
- Mixed encoding issues
- Unicode characters in string literals that cause issues
- BOM (Byte Order Mark) problems
- Line ending inconsistencies
"""

import os
import sys
import re
import codecs
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import chardet


class UnicodeRemediator:
    """Comprehensive Unicode encoding issue detection and remediation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues_found = []
        self.files_fixed = []
        self.problematic_chars = {
            # Unicode box-drawing characters
            '-': '-',    # bullet point
            '?': '*',    # black circle
            '?': 'o',    # white circle
            '?': '*',    # black diamond
            '?': '*',    # white diamond
            '?': '*',    # black square
            '?': '*',    # white square
            '?': '*',    # black small square
            '?': '*',    # white small square
            '?': '>',    # black right triangle
            '?': '>',    # white right triangle
            '?': '^',    # black up triangle
            '?': '^',    # white up triangle
            '?': 'v',    # black down triangle
            '?': 'v',    # white down triangle
            '?': '<',    # black left triangle
            '?': '<',    # white left triangle
            
            # Smart quotes and dashes
            '"': '"',    # left double quotation mark
            '"': '"',    # right double quotation mark
            ''': "'",    # left single quotation mark  
            ''': "'",    # right single quotation mark
            '-': '-',    # en dash
            '--': '--',   # em dash
            '...': '...',  # horizontal ellipsis
            
            # Mathematical symbols
            '×': 'x',    # multiplication sign
            '÷': '/',    # division sign
            '±': '+/-',  # plus-minus sign
            '?': '<=',   # less than or equal
            '?': '>=',   # greater than or equal
            '?': '!=',   # not equal
            '?': '~=',   # approximately equal
            
            # Currency and misc
            '©': '(c)',  # copyright sign
            '®': '(R)',  # registered sign
            '™': '(TM)', # trademark sign
            '°': ' deg', # degree sign
            'µ': 'u',    # micro sign
            '?': 'alpha',
            '?': 'beta',
            '?': 'gamma',
            '?': 'delta',
            '?': 'epsilon',
            '?': 'pi',
            '?': 'sigma',
            '?': 'phi',
            '?': 'omega'
        }
    
    def detect_file_encoding(self, file_path: Path) -> Optional[str]:
        """Detect the encoding of a file."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            
            if not raw_data:
                return 'utf-8'
            
            # Check for BOM
            if raw_data.startswith(codecs.BOM_UTF8):
                return 'utf-8-sig'
            elif raw_data.startswith(codecs.BOM_UTF16_LE):
                return 'utf-16-le'
            elif raw_data.startswith(codecs.BOM_UTF16_BE):
                return 'utf-16-be'
            elif raw_data.startswith(codecs.BOM_UTF32_LE):
                return 'utf-32-le'
            elif raw_data.startswith(codecs.BOM_UTF32_BE):
                return 'utf-32-be'
            
            # Use chardet for detection
            result = chardet.detect(raw_data)
            if result and result['confidence'] > 0.7:
                return result['encoding']
            
            # Fallback encodings to try
            fallback_encodings = ['utf-8', 'cp1252', 'latin-1', 'ascii']
            
            for encoding in fallback_encodings:
                try:
                    raw_data.decode(encoding)
                    return encoding
                except UnicodeDecodeError:
                    continue
            
            return None
            
        except Exception as e:
            print(f"Error detecting encoding for {file_path}: {e}")
            return None
    
    def read_file_safely(self, file_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Read file content safely, trying different encodings."""
        detected_encoding = self.detect_file_encoding(file_path)
        
        if not detected_encoding:
            return None, None
        
        try:
            with open(file_path, 'r', encoding=detected_encoding, errors='replace') as f:
                content = f.read()
            return content, detected_encoding
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None, None
    
    def fix_unicode_characters(self, content: str) -> Tuple[str, List[str]]:
        """Replace problematic Unicode characters with ASCII equivalents."""
        fixed_content = content
        replacements_made = []
        
        for unicode_char, ascii_replacement in self.problematic_chars.items():
            if unicode_char in fixed_content:
                count = fixed_content.count(unicode_char)
                fixed_content = fixed_content.replace(unicode_char, ascii_replacement)
                replacements_made.append(f"Replaced {count} instances of '{unicode_char}' with '{ascii_replacement}'")
        
        return fixed_content, replacements_made
    
    def fix_line_endings(self, content: str) -> str:
        """Normalize line endings to LF (Unix style)."""
        # Replace Windows CRLF and old Mac CR with Unix LF
        content = content.replace('\r\n', '\n')  # Windows
        content = content.replace('\r', '\n')    # Old Mac
        return content
    
    def validate_utf8(self, content: str) -> List[str]:
        """Validate that content can be encoded as UTF-8."""
        issues = []
        
        try:
            content.encode('utf-8')
        except UnicodeEncodeError as e:
            issues.append(f"UTF-8 encoding error: {e}")
        
        # Check for null bytes
        if '\x00' in content:
            null_count = content.count('\x00')
            issues.append(f"Found {null_count} null bytes")
        
        # Check for other control characters
        control_chars = set()
        for char in content:
            if ord(char) < 32 and char not in '\t\n\r':
                control_chars.add(repr(char))
        
        if control_chars:
            issues.append(f"Found control characters: {', '.join(control_chars)}")
        
        return issues
    
    def fix_python_file(self, file_path: Path) -> Dict[str, any]:
        """Fix Unicode issues in a Python file."""
        result = {
            'file': str(file_path),
            'fixed': False,
            'issues': [],
            'encoding_detected': None,
            'replacements': [],
            'validation_errors': []
        }
        
        # Read file content
        content, encoding = self.read_file_safely(file_path)
        if content is None:
            result['issues'].append("Could not read file")
            return result
        
        result['encoding_detected'] = encoding
        original_content = content
        
        # Fix Unicode characters
        content, replacements = self.fix_unicode_characters(content)
        result['replacements'] = replacements
        
        # Fix line endings
        content = self.fix_line_endings(content)
        
        # Validate UTF-8
        validation_errors = self.validate_utf8(content)
        result['validation_errors'] = validation_errors
        
        # Check if changes were made
        if content != original_content:
            try:
                # Write fixed content back to file
                with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(content)
                
                result['fixed'] = True
                self.files_fixed.append(str(file_path))
                
            except Exception as e:
                result['issues'].append(f"Error writing file: {e}")
        
        return result
    
    def scan_python_files(self) -> List[Path]:
        """Scan for Python files in the project."""
        python_files = []
        
        for file_path in self.project_root.rglob("*.py"):
            # Skip virtual environments and cache directories
            if any(part.startswith(('.', '__pycache__', 'venv', 'env')) for part in file_path.parts):
                continue
            python_files.append(file_path)
        
        return python_files
    
    def scan_text_files(self) -> List[Path]:
        """Scan for other text files that might have Unicode issues."""
        text_extensions = {'.txt', '.md', '.rst', '.toml', '.yaml', '.yml', '.json', '.cfg', '.ini'}
        text_files = []
        
        for ext in text_extensions:
            for file_path in self.project_root.rglob(f"*{ext}"):
                if any(part.startswith(('.', '__pycache__', 'venv', 'env')) for part in file_path.parts):
                    continue
                text_files.append(file_path)
        
        return text_files
    
    def run_comprehensive_remediation(self) -> Dict[str, any]:
        """Run comprehensive Unicode issue remediation."""
        print("UNICODE ENCODING ISSUES DETECTION AND REMEDIATION")
        print("=" * 60)
        
        # Scan for files
        print("Scanning for files...")
        python_files = self.scan_python_files()
        text_files = self.scan_text_files()
        
        print(f"Found {len(python_files)} Python files")
        print(f"Found {len(text_files)} other text files")
        
        all_results = []
        files_with_issues = []
        files_fixed = []
        
        # Process Python files
        print("\nProcessing Python files...")
        for file_path in python_files:
            print(f"Checking: {file_path.name}")
            result = self.fix_python_file(file_path)
            all_results.append(result)
            
            if result['replacements'] or result['issues']:
                files_with_issues.append(str(file_path))
            
            if result['fixed']:
                files_fixed.append(str(file_path))
                print(f"  ? Fixed {file_path.name}")
                for replacement in result['replacements']:
                    print(f"    - {replacement}")
            elif result['issues']:
                print(f"  ?? Issues in {file_path.name}: {', '.join(result['issues'])}")
            else:
                print(f"  ? {file_path.name} - No issues")
        
        # Process text files
        print("\nProcessing other text files...")
        for file_path in text_files:
            print(f"Checking: {file_path.name}")
            # Use same logic as Python files
            result = self.fix_python_file(file_path)
            all_results.append(result)
            
            if result['replacements'] or result['issues']:
                files_with_issues.append(str(file_path))
            
            if result['fixed']:
                files_fixed.append(str(file_path))
                print(f"  ? Fixed {file_path.name}")
            else:
                print(f"  ? {file_path.name} - No issues")
        
        # Summary
        print(f"\nUNICODE REMEDIATION COMPLETE")
        print("=" * 60)
        print(f"Files processed: {len(all_results)}")
        print(f"Files with issues: {len(files_with_issues)}")
        print(f"Files fixed: {len(files_fixed)}")
        
        if files_fixed:
            print("\nFixed files:")
            for file_path in files_fixed:
                print(f"  ? {file_path}")
        
        # Specific issue summary
        encoding_issues = sum(1 for r in all_results if r['validation_errors'])
        replacement_count = sum(len(r['replacements']) for r in all_results)
        
        print(f"\nIssue summary:")
        print(f"  Unicode character replacements: {replacement_count}")
        print(f"  Encoding validation issues: {encoding_issues}")
        
        return {
            'total_files': len(all_results),
            'files_with_issues': len(files_with_issues),
            'files_fixed': len(files_fixed),
            'results': all_results
        }


def main():
    """Main entry point for Unicode remediation."""
    project_root = Path(__file__).parent
    remediator = UnicodeRemediator(project_root)
    
    try:
        results = remediator.run_comprehensive_remediation()
        
        if results['files_fixed'] > 0:
            print(f"\n? SUCCESS: Fixed Unicode issues in {results['files_fixed']} files")
        else:
            print(f"\n? No Unicode issues found to fix")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n?? Unicode remediation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n? Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)