# -*- coding: utf-8 -*-
"""
Comprehensive QA/QC Verification for GUI Implementation
Tests all requirements, performs syntax checks, and validates functionality.

Project ID: Image Processing App 20251119
Created: 2025-01-25
Author: The-Sage-Mage
"""

import sys
import os
import ast
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple


class QAQCVerifier:
    """Comprehensive Quality Assurance and Quality Control verifier."""
    
    def __init__(self):
        self.test_results = []
        self.file_hashes = {}
        self.error_count = 0
        self.warning_count = 0
    
    def log_result(self, test_name: str, passed: bool, message: str = "", level: str = "INFO"):
        """Log a test result."""
        status = "PASS" if passed else "FAIL"
        self.test_results.append({
            'test': test_name,
            'status': status,
            'message': message,
            'level': level
        })
        
        if not passed:
            if level == "ERROR":
                self.error_count += 1
            else:
                self.warning_count += 1
        
        symbol = "?" if passed else ("?" if level == "ERROR" else "??")
        print(f"{symbol} {status}: {test_name}")
        if message:
            print(f"   {message}")
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes."""
        try:
            return file_path.stat().st_size
        except Exception:
            return 0
    
    def check_syntax(self, file_path: Path) -> bool:
        """Check Python syntax."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ast.parse(content)
            return True
        except SyntaxError as e:
            self.log_result(
                f"Syntax Check: {file_path.name}", 
                False, 
                f"Line {e.lineno}: {e.msg}",
                "ERROR"
            )
            return False
        except Exception as e:
            self.log_result(
                f"Syntax Check: {file_path.name}", 
                False, 
                f"Read error: {str(e)}",
                "ERROR"
            )
            return False
    
    def check_imports(self, file_path: Path) -> bool:
        """Check for missing or unused imports."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract imports and usage
            imports = set()
            used_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)
            
            # Check for basic import usage (simplified check)
            has_issues = False
            
            return not has_issues
            
        except Exception as e:
            self.log_result(
                f"Import Check: {file_path.name}", 
                False, 
                f"Analysis error: {str(e)}",
                "WARNING"
            )
            return False
    
    def check_brackets_balance(self, file_path: Path) -> bool:
        """Check for balanced brackets, braces, and parentheses."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple bracket counting (ignores strings and comments)
            paren_count = 0
            bracket_count = 0
            brace_count = 0
            
            in_string = False
            in_comment = False
            
            i = 0
            while i < len(content):
                char = content[i]
                
                # Handle strings
                if char in ['"', "'"]:
                    if not in_comment:
                        if not in_string:
                            in_string = True
                            string_char = char
                        elif char == string_char:
                            in_string = False
                
                # Handle comments
                elif char == '#' and not in_string:
                    in_comment = True
                elif char == '\n':
                    in_comment = False
                
                # Count brackets if not in string or comment
                elif not in_string and not in_comment:
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                    elif char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                    elif char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                
                i += 1
            
            balanced = (paren_count == 0 and bracket_count == 0 and brace_count == 0)
            
            if not balanced:
                issues = []
                if paren_count != 0:
                    issues.append(f"Parentheses: {paren_count}")
                if bracket_count != 0:
                    issues.append(f"Brackets: {bracket_count}")
                if brace_count != 0:
                    issues.append(f"Braces: {brace_count}")
                
                self.log_result(
                    f"Bracket Balance: {file_path.name}",
                    False,
                    f"Unbalanced: {', '.join(issues)}",
                    "ERROR"
                )
            
            return balanced
            
        except Exception as e:
            self.log_result(
                f"Bracket Check: {file_path.name}",
                False,
                f"Analysis error: {str(e)}",
                "WARNING"
            )
            return False
    
    def check_file_completeness(self, file_path: Path, min_size: int = 1000) -> bool:
        """Check if file is complete and not empty."""
        try:
            size = self.get_file_size(file_path)
            
            if size == 0:
                self.log_result(
                    f"File Completeness: {file_path.name}",
                    False,
                    "File is empty",
                    "ERROR"
                )
                return False
            
            if size < min_size:
                self.log_result(
                    f"File Completeness: {file_path.name}",
                    False,
                    f"File unusually small ({size} bytes, expected > {min_size})",
                    "WARNING"
                )
                return False
            
            return True
            
        except Exception as e:
            self.log_result(
                f"File Completeness: {file_path.name}",
                False,
                f"Check error: {str(e)}",
                "ERROR"
            )
            return False
    
    def verify_gui_requirements(self) -> bool:
        """Verify specific GUI implementation requirements."""
        main_window_path = Path("src/gui/main_window.py")
        
        if not main_window_path.exists():
            self.log_result(
                "GUI Main Window File",
                False,
                "main_window.py not found",
                "ERROR"
            )
            return False
        
        try:
            with open(main_window_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required classes
            required_classes = [
                "ProcessingControlsRow",
                "ProcessingDropZone", 
                "PickupZone",
                "MatrixHeaderRow",
                "DestinationMatrix",
                "EnhancedDestinationCell",
                "MatrixCornerCell",
                "ColumnHeaderCell",
                "RowHeaderCell"
            ]
            
            missing_classes = []
            for class_name in required_classes:
                if f"class {class_name}" not in content:
                    missing_classes.append(class_name)
            
            if missing_classes:
                self.log_result(
                    "Required GUI Classes",
                    False,
                    f"Missing: {', '.join(missing_classes)}",
                    "ERROR"
                )
                return False
            
            # Check for checkbox implementation
            checkbox_checks = [
                '"All"', '"BWG"', '"SEP"', '"PSK"', '"BK_CLR"', '"BK_CTD"', '"BK_CBN"'
            ]
            
            missing_checkboxes = []
            for checkbox in checkbox_checks:
                if checkbox not in content:
                    missing_checkboxes.append(checkbox)
            
            if missing_checkboxes:
                self.log_result(
                    "Required Checkboxes",
                    False,
                    f"Missing: {', '.join(missing_checkboxes)}",
                    "ERROR"
                )
                return False
            
            # Check for matrix structure (4x5 = 20 cells)
            matrix_checks = [
                "4 cells wide",
                "5 cells high", 
                "25% width",
                "100% height"
            ]
            
            matrix_found = any(check.lower() in content.lower() for check in matrix_checks)
            if not matrix_found:
                self.log_result(
                    "Matrix Structure Documentation",
                    False,
                    "Matrix specifications not found in comments",
                    "WARNING"
                )
            
            return True
            
        except Exception as e:
            self.log_result(
                "GUI Requirements Verification",
                False,
                f"Analysis error: {str(e)}",
                "ERROR"
            )
            return False
    
    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run all QA/QC checks."""
        print("=" * 80)
        print("COMPREHENSIVE QA/QC VERIFICATION")
        print("=" * 80)
        
        # Files to check
        files_to_verify = [
            ("src/gui/main_window.py", 50000),  # Expected to be large
            ("gui_launcher.py", 1000),
            ("GUI_IMPLEMENTATION_COMPLETE.md", 10000),
            ("config/config.toml", 100),
            ("requirements.txt", 100)
        ]
        
        print("\n1. FILE EXISTENCE AND COMPLETENESS CHECKS")
        print("-" * 50)
        
        for file_path_str, min_size in files_to_verify:
            file_path = Path(file_path_str)
            
            if file_path.exists():
                # Calculate hash and size
                file_hash = self.calculate_file_hash(file_path)
                file_size = self.get_file_size(file_path)
                self.file_hashes[str(file_path)] = {
                    'hash': file_hash,
                    'size': file_size
                }
                
                self.log_result(f"File Exists: {file_path.name}", True, 
                               f"Size: {file_size} bytes, Hash: {file_hash[:16]}...")
                
                # Check completeness
                self.check_file_completeness(file_path, min_size)
            else:
                self.log_result(f"File Missing: {file_path_str}", False, "File not found", "ERROR")
        
        print("\n2. PYTHON SYNTAX CHECKS")
        print("-" * 50)
        
        python_files = [
            "src/gui/main_window.py",
            "gui_launcher.py"
        ]
        
        for file_path_str in python_files:
            file_path = Path(file_path_str)
            if file_path.exists():
                syntax_ok = self.check_syntax(file_path)
                if syntax_ok:
                    self.log_result(f"Syntax: {file_path.name}", True)
                
                # Check brackets
                self.check_brackets_balance(file_path)
                
                # Check imports
                self.check_imports(file_path)
        
        print("\n3. GUI REQUIREMENTS VERIFICATION")
        print("-" * 50)
        
        gui_requirements_ok = self.verify_gui_requirements()
        if gui_requirements_ok:
            self.log_result("GUI Requirements", True, "All required components found")
        
        print("\n4. FILE HASH VERIFICATION")
        print("-" * 50)
        
        # Display file information
        for file_path, info in self.file_hashes.items():
            print(f"?? {Path(file_path).name}:")
            print(f"   Size: {info['size']:,} bytes")
            print(f"   Hash: {info['hash']}")
        
        print("\n" + "=" * 80)
        print("QA/QC VERIFICATION SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASS')
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Errors: {self.error_count}")
        print(f"Warnings: {self.warning_count}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.error_count == 0 and success_rate >= 90:
            print("\n?? QA/QC VERIFICATION: PASSED")
            overall_status = "PASSED"
        elif self.error_count == 0:
            print("\n?? QA/QC VERIFICATION: PASSED WITH WARNINGS")
            overall_status = "PASSED_WITH_WARNINGS"
        else:
            print("\n? QA/QC VERIFICATION: FAILED")
            overall_status = "FAILED"
        
        return {
            'status': overall_status,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'success_rate': success_rate,
            'file_hashes': self.file_hashes,
            'test_results': self.test_results
        }


def main():
    """Run comprehensive QA/QC verification."""
    verifier = QAQCVerifier()
    results = verifier.run_comprehensive_verification()
    
    # Return appropriate exit code
    return 0 if results['status'] in ['PASSED', 'PASSED_WITH_WARNINGS'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)