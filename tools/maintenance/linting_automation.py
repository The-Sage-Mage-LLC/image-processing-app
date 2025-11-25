#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linting Automation System
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive linting with flake8, pylint, and additional linters.
"""

import subprocess
import sys
import json
import configparser
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import argparse


class LintingSystem:
    """Advanced linting automation with multiple tools."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.lint_reports_dir = self.project_root / "lint_reports"
        self.lint_reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def install_linters(self) -> bool:
        """Install comprehensive linting tools."""
        print("?? Installing linting tools...")
        
        linters = [
            "flake8",
            "pylint", 
            "pycodestyle",
            "pydocstyle",
            "bandit",
            "vulture",  # Dead code finder
            "pyflakes",
            "mccabe",   # Complexity checker
            "flake8-docstrings",
            "flake8-import-order",
            "flake8-bugbear",
            "flake8-comprehensions",
            "flake8-simplify",
        ]
        
        for linter in linters:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", linter],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"   ? Installed {linter}")
            except subprocess.CalledProcessError as e:
                print(f"   ?? Failed to install {linter}: {e.stderr}")
        
        return True
    
    def setup_flake8_configuration(self) -> None:
        """Setup comprehensive flake8 configuration."""
        flake8_config = """[flake8]
max-line-length = 88
extend-ignore = 
    # E203: whitespace before ':' (conflicts with black)
    E203,
    # W503: line break before binary operator (conflicts with black)
    W503,
    # E501: line too long (handled by black)
    E501,
    # F401: module imported but unused (handled by isort/autoflake)
    F401,
    # E402: module level import not at top of file
    E402

select = 
    # pycodestyle errors
    E,
    # pycodestyle warnings  
    W,
    # pyflakes
    F,
    # mccabe complexity
    C90,
    # flake8-bugbear
    B,
    # flake8-comprehensions
    C4,
    # flake8-simplify
    SIM

exclude =
    .git,
    .mypy_cache,
    .pytest_cache,
    .tox,
    venv,
    venv_*,
    _build,
    build,
    dist,
    *.egg-info,
    __pycache__

# McCabe complexity
max-complexity = 10

# flake8-docstrings
docstring-convention = google

# Import order
import-order-style = google
application-import-names = src

# Per-file ignores
per-file-ignores =
    # Test files can be more relaxed
    tests/*: F401, F811, D, E501
    # Setup files
    setup.py: E402, F401
    # Configuration files
    */conftest.py: F401, E402
    # CLI scripts can import at module level
    */cli/*: E402
    # Main/launcher scripts
    *_launcher.py: E402
    main.py: E402

# Builtins (add common test fixtures)
builtins = 
    _,
    pytest,
    fixture,
    tmpdir,
    monkeypatch,
    capsys
"""
        
        flake8_config_file = self.project_root / ".flake8"
        with open(flake8_config_file, 'w') as f:
            f.write(flake8_config)
        
        print(f"? Created flake8 configuration: {flake8_config_file}")
    
    def setup_pylint_configuration(self) -> None:
        """Setup comprehensive pylint configuration."""
        pylint_config = """[MASTER]
# Use multiple processes to speed up Pylint
jobs = 0

# Pickle collected data for later comparisons
persistent = yes

# Load and enable all available extensions
load-plugins = 
    pylint.extensions.check_elif,
    pylint.extensions.bad_builtin,
    pylint.extensions.docparams,
    pylint.extensions.docstyle,
    pylint.extensions.overlapping_exceptions,
    pylint.extensions.redefined_variable_type,
    pylint.extensions.comparetozero,
    pylint.extensions.emptystring

# Ignore paths
ignore-paths = 
    ^venv.*,
    ^build.*,
    ^dist.*,
    ^\\..*

[MESSAGES CONTROL]
# Disable specific warnings that conflict with other tools or are overly strict
disable =
    # Import related (handled by isort)
    wrong-import-order,
    wrong-import-position,
    ungrouped-imports,
    
    # Formatting (handled by black)
    line-too-long,
    bad-continuation,
    
    # Documentation (can be overly strict)
    missing-module-docstring,
    missing-class-docstring,
    missing-function-docstring,
    
    # Type hints (handled by mypy)
    missing-type-doc,
    missing-return-type-doc,
    
    # Overly strict rules
    too-few-public-methods,
    too-many-arguments,
    too-many-locals,
    too-many-branches,
    too-many-statements,
    too-many-instance-attributes,
    
    # Personal preference
    invalid-name,
    
    # Test specific
    redefined-outer-name,
    unused-argument

# Enable specific useful warnings
enable = 
    use-symbolic-message-instead,
    useless-suppression

[REPORTS]
# Set the output format
output-format = colorized

# Include message's id in the output
include-ids = yes

# Template for messages
msg-template = {path}:{line}:{column}: {msg_id}:{symbol} {msg} ({category})

# Activate the evaluation score
score = yes

[REFACTORING]
# Maximum number of nested blocks
max-nested-blocks = 5

# Maximum number of arguments for function / method
max-args = 8

# Maximum number of attributes for a class
max-attributes = 10

# Maximum number of boolean expressions in an if statement
max-bool-expr = 5

# Maximum number of branch for function / method body
max-branches = 15

# Maximum number of locals for function / method body  
max-locals = 20

# Maximum number of parents for a class
max-parents = 7

# Maximum number of public methods for a class
max-public-methods = 25

# Maximum number of return / yield for function / method body
max-returns = 8

# Maximum number of statements in function / method body
max-statements = 60

# Minimum number of public methods for a class
min-public-methods = 1

[SIMILARITIES]
# Minimum lines of duplicated code
min-similarity-lines = 6

# Ignore comments when computing similarities
ignore-comments = yes

# Ignore docstrings when computing similarities
ignore-docstrings = yes

# Ignore imports when computing similarities
ignore-imports = yes

[VARIABLES]
# List of additional names supposed to be defined in builtins
additional-builtins = 

# Allow unused variables when pattern matches this regex
dummy-variables-rgx = _.*|dummy.*|unused.*

[FORMAT]
# Expected format of line ending
expected-line-ending-format = LF

# Maximum number of characters on a single line (handled by black)
max-line-length = 88

# Maximum number of lines in a module
max-module-lines = 2000

# Allow the body of a class to be on the same line
single-line-class-stmt = no

# Allow the body of an if to be on the same line
single-line-if-stmt = no

[DESIGN]
# Maximum number of parents for a class
max-parents = 7

# Maximum number of attributes for a class
max-attributes = 10

# Minimum number of public methods for a class
min-public-methods = 1

# Maximum number of public methods for a class
max-public-methods = 25

[IMPORTS]
# Create a graph of every import and their dependencies
import-graph = 

# Create a graph of external dependencies
ext-import-graph =

# Create a graph of internal dependencies
int-import-graph =

[CLASSES]
# List of method names used to declare (i.e. assign) instance attributes
defining-attr-methods = 
    __init__,
    __new__,
    setUp,
    __post_init__

# List of member names, which should be excluded from the protected access warning
exclude-protected = 
    _asdict,
    _fields,
    _replace,
    _source,
    _make

# List of valid names for the first argument in a class method
valid-classmethod-first-arg = cls

# List of valid names for the first argument in a metaclass class method  
valid-metaclass-classmethod-first-arg = mcs

[EXCEPTIONS]
# Exceptions that will emit a warning when caught
overgeneral-exceptions = 
    BaseException,
    Exception
"""
        
        pylint_config_file = self.project_root / ".pylintrc"
        with open(pylint_config_file, 'w') as f:
            f.write(pylint_config)
        
        print(f"? Created pylint configuration: {pylint_config_file}")
    
    def run_flake8_linting(self, paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run flake8 linting."""
        if paths is None:
            paths = ["src/", "tests/"]
        
        print("?? Running flake8 linting...")
        
        existing_paths = []
        for path in paths:
            path_obj = self.project_root / path
            if path_obj.exists():
                existing_paths.append(str(path_obj))
        
        if not existing_paths:
            return {"error": "No valid paths found"}
        
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "flake8",
                    "--format=json",
                    "--tee",
                    "--output-file", str(self.lint_reports_dir / f"flake8_report_{self.timestamp}.json"),
                ] + existing_paths,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Also run with standard output for immediate feedback
            result_display = subprocess.run(
                [sys.executable, "-m", "flake8"] + existing_paths,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Parse JSON output if available
            json_report_file = self.lint_reports_dir / f"flake8_report_{self.timestamp}.json"
            issues = []
            
            if json_report_file.exists():
                try:
                    with open(json_report_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            # flake8 JSON format is one object per line
                            for line in content.split('\n'):
                                if line.strip():
                                    issues.append(json.loads(line))
                except json.JSONDecodeError:
                    print("   ?? Could not parse flake8 JSON output")
            
            issue_count = len(issues)
            if issue_count == 0:
                print("   ? flake8: No issues found")
            else:
                print(f"   ?? flake8: {issue_count} issues found")
                
                # Show sample issues
                for issue in issues[:5]:
                    print(f"      {issue.get('filename', '')}:{issue.get('line_number', '')} "
                          f"{issue.get('code', '')} {issue.get('text', '')}")
            
            return {
                "tool": "flake8",
                "return_code": result.returncode,
                "issue_count": issue_count,
                "issues": issues,
                "stdout": result_display.stdout,
                "stderr": result_display.stderr,
                "timestamp": datetime.now().isoformat()
            }
            
        except FileNotFoundError:
            print("   ? flake8 not found. Install with: pip install flake8")
            return {"error": "flake8 not installed"}
    
    def run_pylint_analysis(self, paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run pylint analysis."""
        if paths is None:
            paths = ["src/"]  # pylint can be slow, focus on main code
        
        print("?? Running pylint analysis...")
        
        existing_paths = []
        for path in paths:
            path_obj = self.project_root / path
            if path_obj.exists():
                existing_paths.append(str(path_obj))
        
        if not existing_paths:
            return {"error": "No valid paths found"}
        
        try:
            # Run with JSON output
            result_json = subprocess.run(
                [
                    sys.executable, "-m", "pylint",
                    "--output-format=json",
                    "--reports=yes",
                ] + existing_paths,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Run with text output for display
            result_text = subprocess.run(
                [sys.executable, "-m", "pylint"] + existing_paths,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Parse JSON output
            issues = []
            score = 0.0
            
            try:
                if result_json.stdout.strip():
                    json_data = json.loads(result_json.stdout)
                    if isinstance(json_data, list):
                        issues = json_data
            except json.JSONDecodeError:
                print("   ?? Could not parse pylint JSON output")
            
            # Extract score from text output
            score_line = [line for line in result_text.stdout.split('\n') if 'Your code has been rated at' in line]
            if score_line:
                try:
                    score_str = score_line[0].split('rated at ')[1].split('/')[0]
                    score = float(score_str)
                except (IndexError, ValueError):
                    pass
            
            issue_count = len(issues)
            if issue_count == 0:
                print(f"   ? pylint: No issues found (Score: {score}/10)")
            else:
                print(f"   ?? pylint: {issue_count} issues found (Score: {score}/10)")
                
                # Show sample issues by severity
                for issue in issues[:5]:
                    print(f"      {issue.get('path', '')}:{issue.get('line', '')} "
                          f"{issue.get('type', '')} {issue.get('message', '')} "
                          f"({issue.get('symbol', '')})")
            
            # Save detailed report
            report_file = self.lint_reports_dir / f"pylint_report_{self.timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump({
                    "issues": issues,
                    "score": score,
                    "text_output": result_text.stdout
                }, f, indent=2)
            
            return {
                "tool": "pylint",
                "return_code": result_json.returncode,
                "issue_count": issue_count,
                "score": score,
                "issues": issues,
                "text_output": result_text.stdout,
                "timestamp": datetime.now().isoformat()
            }
            
        except FileNotFoundError:
            print("   ? pylint not found. Install with: pip install pylint")
            return {"error": "pylint not installed"}
    
    def run_additional_linters(self) -> Dict[str, Any]:
        """Run additional specialized linters."""
        print("?? Running additional linters...")
        
        results = {}
        
        # Vulture - dead code detection
        try:
            vulture_result = subprocess.run(
                [sys.executable, "-m", "vulture", "src/", "--min-confidence", "80"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            dead_code_lines = vulture_result.stdout.count('\n') if vulture_result.stdout else 0
            results["vulture"] = {
                "return_code": vulture_result.returncode,
                "dead_code_findings": dead_code_lines,
                "output": vulture_result.stdout
            }
            
            if dead_code_lines == 0:
                print("   ? vulture: No dead code found")
            else:
                print(f"   ?? vulture: {dead_code_lines} potential dead code findings")
                
        except FileNotFoundError:
            print("   ?? vulture not available")
            results["vulture"] = {"error": "not_installed"}
        
        # pydocstyle - docstring checker
        try:
            pydocstyle_result = subprocess.run(
                [sys.executable, "-m", "pydocstyle", "src/"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            docstring_issues = pydocstyle_result.stdout.count('\n') if pydocstyle_result.stdout else 0
            results["pydocstyle"] = {
                "return_code": pydocstyle_result.returncode,
                "docstring_issues": docstring_issues,
                "output": pydocstyle_result.stdout
            }
            
            if docstring_issues == 0:
                print("   ? pydocstyle: Docstrings are well formatted")
            else:
                print(f"   ?? pydocstyle: {docstring_issues} docstring issues")
                
        except FileNotFoundError:
            print("   ?? pydocstyle not available")
            results["pydocstyle"] = {"error": "not_installed"}
        
        return results
    
    def generate_comprehensive_lint_report(self, flake8_results: Dict, pylint_results: Dict, additional_results: Dict) -> None:
        """Generate comprehensive linting report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_path": str(self.project_root),
            "flake8": flake8_results,
            "pylint": pylint_results,
            "additional_linters": additional_results,
            "summary": {}
        }
        
        # Calculate summary metrics
        total_issues = 0
        total_issues += flake8_results.get("issue_count", 0)
        total_issues += pylint_results.get("issue_count", 0)
        
        pylint_score = pylint_results.get("score", 0)
        
        # Quality score calculation
        quality_score = 100
        quality_score -= min(total_issues * 2, 60)  # Max 60 points deduction for issues
        quality_score = max(quality_score, pylint_score * 10)  # Use pylint score as minimum
        
        report["summary"] = {
            "total_issues": total_issues,
            "pylint_score": pylint_score,
            "quality_score": round(quality_score, 1),
            "flake8_issues": flake8_results.get("issue_count", 0),
            "pylint_issues": pylint_results.get("issue_count", 0),
            "dead_code_findings": additional_results.get("vulture", {}).get("dead_code_findings", 0),
            "docstring_issues": additional_results.get("pydocstyle", {}).get("docstring_issues", 0)
        }
        
        # Save report
        report_file = self.lint_reports_dir / f"comprehensive_lint_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown summary
        self.generate_markdown_lint_report(report)
        
        print(f"\n?? LINTING SUMMARY")
        print(f"   Overall Quality Score: {quality_score:.1f}/100")
        print(f"   Total Issues: {total_issues}")
        print(f"   Pylint Score: {pylint_score:.1f}/10")
        print(f"   Report saved: {report_file}")
    
    def generate_markdown_lint_report(self, report: Dict) -> None:
        """Generate markdown linting report."""
        md_file = self.lint_reports_dir / f"lint_summary_{self.timestamp}.md"
        
        with open(md_file, 'w') as f:
            f.write("# Code Linting Report\n\n")
            f.write(f"**Generated:** {report['timestamp']}\n")
            f.write(f"**Quality Score:** {report['summary']['quality_score']}/100\n\n")
            
            # Executive Summary
            quality_score = report['summary']['quality_score']
            if quality_score >= 90:
                f.write("## ?? EXCELLENT - High code quality maintained\n\n")
            elif quality_score >= 80:
                f.write("## ?? GOOD - Minor issues detected\n\n")
            elif quality_score >= 70:
                f.write("## ?? FAIR - Several issues need attention\n\n")
            else:
                f.write("## ?? NEEDS IMPROVEMENT - Significant issues detected\n\n")
            
            # Detailed Results
            f.write("## Linting Results\n\n")
            
            # flake8
            f.write("### flake8 (Style & Error Checking)\n")
            flake8_issues = report['summary']['flake8_issues']
            f.write(f"- **Issues Found:** {flake8_issues}\n")
            if flake8_issues == 0:
                f.write("- **Status:** ? No style or error issues\n\n")
            else:
                f.write("- **Status:** ?? Style/error issues detected\n\n")
            
            # pylint
            f.write("### pylint (Code Analysis)\n")
            pylint_score = report['summary']['pylint_score']
            pylint_issues = report['summary']['pylint_issues']
            f.write(f"- **Score:** {pylint_score:.1f}/10\n")
            f.write(f"- **Issues Found:** {pylint_issues}\n")
            if pylint_score >= 9.0:
                f.write("- **Status:** ? Excellent code quality\n\n")
            elif pylint_score >= 8.0:
                f.write("- **Status:** ?? Good code quality\n\n")
            else:
                f.write("- **Status:** ?? Code quality needs improvement\n\n")
            
            # Additional linters
            f.write("### Additional Analysis\n")
            dead_code = report['summary']['dead_code_findings']
            docstring_issues = report['summary']['docstring_issues']
            
            f.write(f"- **Dead Code (vulture):** {dead_code} findings\n")
            f.write(f"- **Docstring Issues (pydocstyle):** {docstring_issues} issues\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            total_issues = report['summary']['total_issues']
            if total_issues == 0:
                f.write("? Code quality is excellent - no immediate actions needed.\n")
            else:
                f.write(f"1. **Address {flake8_issues} flake8 style issues**\n")
                f.write(f"2. **Resolve {pylint_issues} pylint code quality issues**\n")
                if dead_code > 0:
                    f.write(f"3. **Review {dead_code} potential dead code findings**\n")
                if docstring_issues > 0:
                    f.write(f"4. **Improve {docstring_issues} docstring formatting issues**\n")
        
        print(f"?? Markdown report saved: {md_file}")


def main():
    """Main entry point for linting automation."""
    parser = argparse.ArgumentParser(description="Comprehensive linting automation")
    parser.add_argument("--install", action="store_true", help="Install linters")
    parser.add_argument("--configure", action="store_true", help="Setup configurations") 
    parser.add_argument("--flake8", action="store_true", help="Run flake8")
    parser.add_argument("--pylint", action="store_true", help="Run pylint")
    parser.add_argument("--additional", action="store_true", help="Run additional linters")
    parser.add_argument("--all", action="store_true", help="Run all linters")
    parser.add_argument("--setup", action="store_true", help="Install and configure")
    
    args = parser.parse_args()
    
    linting_system = LintingSystem()
    
    if args.install or args.setup:
        linting_system.install_linters()
    
    if args.configure or args.setup:
        linting_system.setup_flake8_configuration()
        linting_system.setup_pylint_configuration()
    
    if args.all or len(sys.argv) == 1:
        # Run comprehensive linting
        print("?? Running comprehensive code linting...")
        
        flake8_results = linting_system.run_flake8_linting()
        pylint_results = linting_system.run_pylint_analysis()
        additional_results = linting_system.run_additional_linters()
        
        linting_system.generate_comprehensive_lint_report(
            flake8_results, pylint_results, additional_results
        )
        
    elif args.flake8:
        linting_system.run_flake8_linting()
    elif args.pylint:
        linting_system.run_pylint_analysis()
    elif args.additional:
        linting_system.run_additional_linters()


if __name__ == "__main__":
    main()