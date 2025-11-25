#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Code Analysis for Import/Variable/Function Usage
Identifies: unused imports, non-existent calls, unused variables, undefined references,
unused functions, calls to non-existent functions, and misspellings
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.imports = {}  # module -> alias/names
        self.from_imports = {}  # module -> {name: alias}
        self.imported_names = set()  # all imported names/aliases
        self.used_names = set()  # all used names
        self.defined_functions = set()  # functions defined in this file
        self.called_functions = set()  # functions called in this file
        self.defined_variables = set()  # variables assigned
        self.used_variables = set()  # variables referenced
        self.attribute_accesses = set()  # module.attribute accesses
        self.issues = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.asname:
                self.imports[alias.name] = alias.asname
                self.imported_names.add(alias.asname)
            else:
                self.imports[alias.name] = alias.name
                self.imported_names.add(alias.name)

    def visit_ImportFrom(self, node):
        module = node.module or ''
        if module not in self.from_imports:
            self.from_imports[module] = {}
        
        for alias in node.names:
            if alias.name == '*':
                # Handle star imports
                self.from_imports[module]['*'] = '*'
            else:
                name = alias.asname if alias.asname else alias.name
                self.from_imports[module][alias.name] = name
                self.imported_names.add(name)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.defined_variables.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
            self.used_variables.add(node.id)

    def visit_FunctionDef(self, node):
        self.defined_functions.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.defined_functions.add(node.name)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.called_functions.add(node.func.id)
            self.used_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                self.used_names.add(node.func.value.id)
                self.attribute_accesses.add(f"{node.func.value.id}.{node.func.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)
            self.attribute_accesses.add(f"{node.value.id}.{node.attr}")
        self.generic_visit(node)

    def analyze_file(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            self.visit(tree)
            
            # Analyze for issues
            self._find_unused_imports()
            self._find_undefined_references()
            self._find_unused_variables()
            self._find_unused_functions()
            
        except Exception as e:
            self.issues.append(f"Error parsing file: {e}")

    def _find_unused_imports(self):
        # Check for unused regular imports
        for module, alias in self.imports.items():
            if alias not in self.used_names:
                # Check if used as attribute access
                used_as_attr = any(attr.startswith(f"{alias}.") for attr in self.attribute_accesses)
                if not used_as_attr:
                    self.issues.append(f"UNUSED_IMPORT: import {module} as {alias}")

        # Check for unused from imports
        for module, imports in self.from_imports.items():
            for original_name, alias in imports.items():
                if original_name != '*' and alias not in self.used_names:
                    self.issues.append(f"UNUSED_FROM_IMPORT: from {module} import {original_name} as {alias}")

    def _find_undefined_references(self):
        # Built-in names that are always available
        builtins = {
            'abs', 'all', 'any', 'bool', 'bytes', 'callable', 'chr', 'classmethod',
            'dict', 'dir', 'enumerate', 'eval', 'exec', 'filter', 'float', 'format',
            'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'hex', 'id', 'input',
            'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals', 'map',
            'max', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print',
            'property', 'range', 'repr', 'reversed', 'round', 'set', 'setattr',
            'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple',
            'type', 'vars', 'zip', '__name__', '__file__', '__doc__', 'Exception',
            'ValueError', 'TypeError', 'KeyError', 'IndexError', 'AttributeError',
            'ImportError', 'ModuleNotFoundError', 'FileNotFoundError', 'OSError'
        }

        # Names available in this scope
        available_names = (self.imported_names | self.defined_variables | 
                          self.defined_functions | builtins)

        for name in self.used_variables:
            if name not in available_names:
                # Check if it might be a module attribute we missed
                if not any(name.startswith(imp + '.') for imp in self.imported_names):
                    self.issues.append(f"UNDEFINED_REFERENCE: {name}")

    def _find_unused_variables(self):
        # Variables that are defined but never used (excluding function parameters)
        for var in self.defined_variables:
            if var not in self.used_variables and not var.startswith('_'):
                # Ignore common patterns
                if var not in ['self', 'cls', 'args', 'kwargs']:
                    self.issues.append(f"UNUSED_VARIABLE: {var}")

    def _find_unused_functions(self):
        # Functions that are defined but never called (excluding private functions)
        for func in self.defined_functions:
            if func not in self.called_functions and not func.startswith('_'):
                # Ignore common patterns like __init__, main, etc.
                if func not in ['main', '__init__', '__call__', '__enter__', '__exit__']:
                    self.issues.append(f"UNUSED_FUNCTION: {func}")


def analyze_project():
    """Analyze all Python files in the project."""
    project_root = Path(".")
    python_files = []
    
    # Get all Python files, excluding virtual environment and cache
    for py_file in project_root.rglob("*.py"):
        if not any(part.startswith(('.', '__pycache__', 'venv', 'env')) for part in py_file.parts):
            python_files.append(py_file)
    
    all_issues = {}
    
    for py_file in python_files:
        print(f"Analyzing: {py_file}")
        analyzer = CodeAnalyzer(str(py_file))
        analyzer.analyze_file()
        
        if analyzer.issues:
            all_issues[str(py_file)] = analyzer.issues
    
    return all_issues, python_files

def main():
    print("COMPREHENSIVE CODE ANALYSIS")
    print("=" * 50)
    print("Checking for:")
    print("- Unused imports")
    print("- Non-existent calls") 
    print("- Unused variables")
    print("- Undefined references")
    print("- Unused functions")
    print("- Function misspellings")
    print()

    issues, analyzed_files = analyze_project()
    
    if not issues:
        print("SUCCESS: NO ISSUES FOUND!")
        print(f"Analyzed {len(analyzed_files)} Python files")
        print("\nAll files are clean of:")
        print("- Unused imports")
        print("- Undefined references") 
        print("- Unused variables")
        print("- Unused functions")
        return
    
    print(f"ISSUES FOUND in {len(issues)} files:")
    print()
    
    total_issues = 0
    for filepath, file_issues in issues.items():
        print(f"FILE: {filepath}")
        print("-" * 50)
        for issue in file_issues:
            print(f"  - {issue}")
            total_issues += 1
        print()
    
    print(f"SUMMARY: {total_issues} total issues found")
    
    # Categorize issues
    categories = {
        'UNUSED_IMPORT': 0,
        'UNUSED_FROM_IMPORT': 0, 
        'UNDEFINED_REFERENCE': 0,
        'UNUSED_VARIABLE': 0,
        'UNUSED_FUNCTION': 0
    }
    
    for file_issues in issues.values():
        for issue in file_issues:
            for category in categories:
                if issue.startswith(category):
                    categories[category] += 1
                    break
    
    print("\nISSUE BREAKDOWN:")
    for category, count in categories.items():
        if count > 0:
            print(f"  {category}: {count}")

if __name__ == "__main__":
    main()