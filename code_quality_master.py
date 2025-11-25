#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Quality Automation Master Suite
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Complete code quality automation integrating all tools:
- Static type checking (mypy)
- Code formatting (black/prettier) 
- Linting (flake8, pylint)
- Complexity analysis
- Application metrics
"""

import subprocess
import sys
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import argparse


class CodeQualityMaster:
    """Master controller for all code quality automation tools."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.quality_reports_dir = self.project_root / "quality_reports"
        self.quality_reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Tool availability tracking
        self.tools_available = {}
        self.check_tool_availability()
        
    def check_tool_availability(self) -> None:
        """Check which quality tools are available."""
        tools = {
            "mypy": "mypy --version",
            "black": "black --version", 
            "isort": "isort --version",
            "flake8": "flake8 --version",
            "pylint": "pylint --version",
            "radon": "radon --version",
            "prettier": "prettier --version",
            "bandit": "bandit --version"
        }
        
        for tool, version_cmd in tools.items():
            try:
                result = subprocess.run(
                    version_cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                self.tools_available[tool] = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                self.tools_available[tool] = False
    
    def install_all_tools(self) -> bool:
        """Install all code quality tools."""
        print("?? Installing comprehensive code quality tools...")
        
        # Python tools via pip
        python_tools = [
            "mypy",
            "black",
            "isort", 
            "flake8",
            "pylint",
            "radon",
            "bandit",
            "vulture",
            "pydocstyle",
            "mccabe",
            "flake8-docstrings",
            "flake8-import-order",
            "flake8-bugbear",
            "flake8-comprehensions",
            "flake8-simplify",
            "cognitive-complexity",
            "types-requests",
            "types-PyYAML",
            "types-setuptools",
            "types-Pillow",
            "psutil",
            "prometheus-client",
        ]
        
        success_count = 0
        total_count = len(python_tools)
        
        for tool in python_tools:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", tool],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    success_count += 1
                    print(f"   ? Installed {tool}")
                else:
                    print(f"   ?? Failed to install {tool}: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"   ? Timeout installing {tool}")
            except Exception as e:
                print(f"   ? Error installing {tool}: {e}")
        
        # Try to install prettier via npm
        try:
            npm_result = subprocess.run(
                ["npm", "install", "-g", "prettier"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if npm_result.returncode == 0:
                print("   ? Installed prettier (npm)")
            else:
                print("   ?? Could not install prettier - npm may not be available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("   ?? npm not available for prettier installation")
        
        print(f"?? Installation summary: {success_count}/{total_count} Python tools installed")
        
        # Refresh tool availability
        self.check_tool_availability()
        
        return success_count > (total_count * 0.8)  # 80% success rate
    
    def setup_all_configurations(self) -> None:
        """Setup configurations for all tools."""
        print("?? Setting up tool configurations...")
        
        self.setup_mypy_config()
        self.setup_black_config() 
        self.setup_isort_config()
        self.setup_flake8_config()
        self.setup_pylint_config()
        self.setup_prettier_config()
        self.setup_vscode_integration()
        self.setup_git_hooks()
        
        print("? All configurations setup complete")
    
    def setup_mypy_config(self) -> None:
        """Setup mypy configuration."""
        mypy_config = """[mypy]
python_version = 3.11
warn_return_any = true
warn_unused_configs = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

# Gradually enable strict mode
disallow_untyped_defs = false
disallow_incomplete_defs = false  
check_untyped_defs = true
no_implicit_optional = true
strict_optional = true

# Import handling
ignore_missing_imports = true
follow_imports = silent
show_error_codes = true
color_output = true
pretty = true

# Exclusions
exclude = (?x)(
    ^venv_312_fixed/
    | ^build/
    | ^dist/
    | ^\.git/
    | ^\.mypy_cache/
    | ^__pycache__/
)

# Third-party libraries
[mypy-cv2.*]
ignore_missing_imports = true
[mypy-PIL.*]
ignore_missing_imports = true
[mypy-PyQt6.*]
ignore_missing_imports = true
[mypy-pytest.*]
ignore_missing_imports = true
"""
        
        with open(self.project_root / "mypy.ini", 'w') as f:
            f.write(mypy_config)
        print("   ? Created mypy.ini")
    
    def setup_black_config(self) -> None:
        """Setup Black configuration in pyproject.toml."""
        pyproject_file = self.project_root / "pyproject.toml"
        
        black_config = """
[tool.black]
line-length = 88
target-version = ['py311', 'py312']
include = '\\.pyi?$'
extend-exclude = '''
/(
    \\.git
    | \\.mypy_cache
    | \\.tox
    | \\.venv
    | build
    | dist
    | venv_312_fixed
)/
'''
"""
        
        if pyproject_file.exists():
            content = pyproject_file.read_text()
            if "[tool.black]" not in content:
                with open(pyproject_file, 'a') as f:
                    f.write(black_config)
                print("   ? Added Black config to pyproject.toml")
        else:
            with open(pyproject_file, 'w') as f:
                f.write(black_config.strip())
            print("   ? Created pyproject.toml with Black config")
    
    def setup_isort_config(self) -> None:
        """Setup isort configuration."""
        pyproject_file = self.project_root / "pyproject.toml"
        
        isort_config = """
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["src"]
sections = ["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
"""
        
        if pyproject_file.exists():
            content = pyproject_file.read_text()
            if "[tool.isort]" not in content:
                with open(pyproject_file, 'a') as f:
                    f.write(isort_config)
                print("   ? Added isort config to pyproject.toml")
    
    def setup_flake8_config(self) -> None:
        """Setup flake8 configuration."""
        flake8_config = """[flake8]
max-line-length = 88
extend-ignore = E203, W503, E501
select = E, W, F, C90, B, C4
exclude = .git, .mypy_cache, .pytest_cache, .tox, venv, build, dist, *.egg-info, __pycache__
max-complexity = 10

# Per-file ignores
per-file-ignores =
    tests/*: F401, F811, D
    setup.py: E402, F401
    */__init__.py: F401
"""
        
        with open(self.project_root / ".flake8", 'w') as f:
            f.write(flake8_config)
        print("   ? Created .flake8")
    
    def setup_pylint_config(self) -> None:
        """Setup pylint configuration."""
        pylint_config = """[MASTER]
jobs = 0
persistent = yes

[MESSAGES CONTROL]
disable = 
    wrong-import-order,
    wrong-import-position,
    line-too-long,
    missing-module-docstring,
    missing-class-docstring,
    missing-function-docstring,
    too-few-public-methods,
    too-many-arguments,
    too-many-locals,
    invalid-name,
    redefined-outer-name

[REPORTS]
output-format = colorized
score = yes

[FORMAT]
max-line-length = 88
"""
        
        with open(self.project_root / ".pylintrc", 'w') as f:
            f.write(pylint_config)
        print("   ? Created .pylintrc")
    
    def setup_prettier_config(self) -> None:
        """Setup Prettier configuration."""
        prettier_config = {
            "semi": True,
            "trailingComma": "es5",
            "singleQuote": False,
            "printWidth": 80,
            "tabWidth": 2,
            "overrides": [
                {
                    "files": "*.md",
                    "options": {"printWidth": 100, "proseWrap": "always"}
                },
                {
                    "files": ["*.yml", "*.yaml"],
                    "options": {"tabWidth": 2}
                }
            ]
        }
        
        with open(self.project_root / ".prettierrc.json", 'w') as f:
            json.dump(prettier_config, f, indent=2)
        print("   ? Created .prettierrc.json")
    
    def setup_vscode_integration(self) -> None:
        """Setup VS Code integration."""
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        settings = {
            "python.linting.enabled": True,
            "python.linting.mypyEnabled": True,
            "python.linting.flake8Enabled": True,
            "python.linting.pylintEnabled": True,
            "python.formatting.provider": "black",
            "editor.formatOnSave": True,
            "python.sortImports.args": ["--profile", "black"],
            "[python]": {
                "editor.codeActionsOnSave": {
                    "source.organizeImports": True
                }
            }
        }
        
        with open(vscode_dir / "settings.json", 'w') as f:
            json.dump(settings, f, indent=2)
        print("   ? Created .vscode/settings.json")
    
    def setup_git_hooks(self) -> None:
        """Setup Git pre-commit hooks."""
        pre_commit_config = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.4.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']

  - repo: https://github.com/prettier/prettier
    rev: v3.0.0
    hooks:
      - id: prettier
        types_or: [markdown, yaml, json]
"""
        
        with open(self.project_root / ".pre-commit-config.yaml", 'w') as f:
            f.write(pre_commit_config)
        print("   ? Created .pre-commit-config.yaml")
    
    def run_type_checking(self) -> Dict[str, Any]:
        """Run mypy type checking."""
        if not self.tools_available.get("mypy", False):
            return {"error": "mypy not available"}
        
        print("?? Running type checking with mypy...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mypy", "src/", "tests/"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            error_count = result.stdout.count("error:")
            
            return {
                "tool": "mypy",
                "return_code": result.returncode,
                "error_count": error_count,
                "output": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "mypy timeout"}
        except Exception as e:
            return {"error": str(e)}
    
    def run_formatting(self, check_only: bool = False) -> Dict[str, Any]:
        """Run code formatting with Black and isort."""
        results = {"black": {}, "isort": {}, "prettier": {}}
        
        # Black formatting
        if self.tools_available.get("black", False):
            print("?? Running Black formatting...")
            try:
                cmd = [sys.executable, "-m", "black"]
                if check_only:
                    cmd.append("--check")
                cmd.extend(["src/", "tests/", "*.py"])
                
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                results["black"] = {
                    "return_code": result.returncode,
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "stderr": result.stderr
                }
                
            except Exception as e:
                results["black"] = {"error": str(e)}
        
        # isort import sorting
        if self.tools_available.get("isort", False):
            print("?? Running isort...")
            try:
                cmd = [sys.executable, "-m", "isort"]
                if check_only:
                    cmd.append("--check-only")
                cmd.extend(["src/", "tests/", "*.py"])
                
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                results["isort"] = {
                    "return_code": result.returncode,
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "stderr": result.stderr
                }
                
            except Exception as e:
                results["isort"] = {"error": str(e)}
        
        # Prettier for config files
        if self.tools_available.get("prettier", False):
            print("?? Running Prettier...")
            try:
                config_files = []
                for pattern in ["*.md", "*.yml", "*.yaml", "*.json"]:
                    config_files.extend(self.project_root.glob(pattern))
                
                if config_files:
                    cmd = ["prettier"]
                    if check_only:
                        cmd.append("--check")
                    else:
                        cmd.append("--write")
                    cmd.extend([str(f) for f in config_files])
                    
                    result = subprocess.run(
                        cmd,
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    results["prettier"] = {
                        "return_code": result.returncode,
                        "success": result.returncode == 0,
                        "files_processed": len(config_files),
                        "output": result.stdout,
                        "stderr": result.stderr
                    }
                
            except Exception as e:
                results["prettier"] = {"error": str(e)}
        
        return results
    
    def run_linting(self) -> Dict[str, Any]:
        """Run comprehensive linting."""
        results = {"flake8": {}, "pylint": {}, "bandit": {}}
        
        # flake8 linting
        if self.tools_available.get("flake8", False):
            print("?? Running flake8...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "flake8", "src/", "tests/"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                issue_count = result.stdout.count('\n') if result.stdout else 0
                
                results["flake8"] = {
                    "return_code": result.returncode,
                    "success": result.returncode == 0,
                    "issue_count": issue_count,
                    "output": result.stdout,
                    "stderr": result.stderr
                }
                
            except Exception as e:
                results["flake8"] = {"error": str(e)}
        
        # pylint analysis
        if self.tools_available.get("pylint", False):
            print("?? Running pylint...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pylint", "src/"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                # Extract score
                score = 0.0
                for line in result.stdout.split('\n'):
                    if 'Your code has been rated at' in line:
                        try:
                            score_str = line.split('rated at ')[1].split('/')[0]
                            score = float(score_str)
                            break
                        except (IndexError, ValueError):
                            pass
                
                results["pylint"] = {
                    "return_code": result.returncode,
                    "score": score,
                    "output": result.stdout,
                    "stderr": result.stderr
                }
                
            except Exception as e:
                results["pylint"] = {"error": str(e)}
        
        # Bandit security analysis
        if self.tools_available.get("bandit", False):
            print("?? Running bandit security check...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "bandit", "-r", "src/", "-f", "json"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                issue_count = 0
                if result.stdout:
                    try:
                        bandit_data = json.loads(result.stdout)
                        issue_count = len(bandit_data.get("results", []))
                    except json.JSONDecodeError:
                        pass
                
                results["bandit"] = {
                    "return_code": result.returncode,
                    "issue_count": issue_count,
                    "output": result.stdout,
                    "stderr": result.stderr
                }
                
            except Exception as e:
                results["bandit"] = {"error": str(e)}
        
        return results
    
    def run_complexity_analysis(self) -> Dict[str, Any]:
        """Run complexity analysis."""
        if not self.tools_available.get("radon", False):
            return {"error": "radon not available"}
        
        print("?? Running complexity analysis...")
        
        results = {"cyclomatic": {}, "halstead": {}, "raw": {}}
        
        # Cyclomatic complexity
        try:
            result = subprocess.run(
                [sys.executable, "-m", "radon", "cc", "src/", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                try:
                    cc_data = json.loads(result.stdout)
                    
                    all_functions = []
                    for file_path, functions in cc_data.items():
                        all_functions.extend(functions)
                    
                    if all_functions:
                        complexities = [f.get('complexity', 0) for f in all_functions]
                        avg_complexity = sum(complexities) / len(complexities)
                        high_complexity = len([c for c in complexities if c > 10])
                        
                        results["cyclomatic"] = {
                            "total_functions": len(all_functions),
                            "average_complexity": round(avg_complexity, 2),
                            "max_complexity": max(complexities),
                            "high_complexity_count": high_complexity,
                            "data": cc_data
                        }
                    
                except json.JSONDecodeError:
                    results["cyclomatic"] = {"error": "JSON parsing failed"}
            
        except Exception as e:
            results["cyclomatic"] = {"error": str(e)}
        
        # Halstead metrics
        try:
            result = subprocess.run(
                [sys.executable, "-m", "radon", "hal", "src/", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                try:
                    hal_data = json.loads(result.stdout)
                    
                    total_volume = 0
                    total_difficulty = 0
                    total_bugs = 0
                    file_count = 0
                    
                    for file_path, metrics in hal_data.items():
                        if metrics:
                            total_volume += metrics.get('volume', 0)
                            total_difficulty += metrics.get('difficulty', 0)
                            total_bugs += metrics.get('bugs', 0)
                            file_count += 1
                    
                    if file_count > 0:
                        results["halstead"] = {
                            "total_files": file_count,
                            "average_volume": round(total_volume / file_count, 2),
                            "average_difficulty": round(total_difficulty / file_count, 2),
                            "total_estimated_bugs": round(total_bugs, 2),
                            "data": hal_data
                        }
                    
                except json.JSONDecodeError:
                    results["halstead"] = {"error": "JSON parsing failed"}
            
        except Exception as e:
            results["halstead"] = {"error": str(e)}
        
        return results
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run complete code quality analysis."""
        print("?? Running comprehensive code quality analysis...")
        
        start_time = datetime.now()
        
        # Run all analyses
        analysis_results = {
            "timestamp": start_time.isoformat(),
            "project_path": str(self.project_root),
            "tools_available": self.tools_available,
            "type_checking": self.run_type_checking(),
            "formatting": self.run_formatting(check_only=True),
            "linting": self.run_linting(),
            "complexity": self.run_complexity_analysis(),
            "summary": {}
        }
        
        end_time = datetime.now()
        analysis_time = (end_time - start_time).total_seconds()
        
        # Calculate overall quality score
        quality_score = self.calculate_quality_score(analysis_results)
        
        analysis_results["summary"] = {
            "analysis_time_seconds": round(analysis_time, 2),
            "overall_quality_score": quality_score["score"],
            "quality_grade": quality_score["grade"],
            "issues_found": quality_score["total_issues"],
            "recommendations": quality_score["recommendations"]
        }
        
        # Save comprehensive report
        report_file = self.quality_reports_dir / f"comprehensive_quality_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Generate markdown report
        self.generate_quality_report(analysis_results)
        
        print(f"\n?? COMPREHENSIVE CODE QUALITY ANALYSIS COMPLETE")
        print(f"   Quality Score: {quality_score['score']}/100")
        print(f"   Grade: {quality_score['grade']}")
        print(f"   Analysis Time: {analysis_time:.2f}s")
        print(f"   Report: {report_file}")
        
        return analysis_results
    
    def calculate_quality_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall code quality score."""
        score = 100
        issues = []
        total_issues = 0
        
        # Type checking penalties
        type_results = results.get("type_checking", {})
        if "error_count" in type_results:
            type_errors = type_results["error_count"]
            score -= min(type_errors * 2, 20)
            total_issues += type_errors
            if type_errors > 0:
                issues.append(f"{type_errors} type checking errors")
        
        # Formatting penalties
        format_results = results.get("formatting", {})
        format_issues = 0
        for tool, result in format_results.items():
            if isinstance(result, dict) and not result.get("success", True):
                format_issues += 1
        
        if format_issues > 0:
            score -= format_issues * 5
            issues.append(f"Code formatting issues in {format_issues} tools")
        
        # Linting penalties
        lint_results = results.get("linting", {})
        
        # flake8 issues
        if "flake8" in lint_results and "issue_count" in lint_results["flake8"]:
            flake8_issues = lint_results["flake8"]["issue_count"]
            score -= min(flake8_issues * 1, 15)
            total_issues += flake8_issues
            if flake8_issues > 0:
                issues.append(f"{flake8_issues} flake8 style issues")
        
        # pylint score
        if "pylint" in lint_results and "score" in lint_results["pylint"]:
            pylint_score = lint_results["pylint"]["score"]
            if pylint_score < 8.0:
                penalty = (8.0 - pylint_score) * 5
                score -= penalty
                issues.append(f"Low pylint score: {pylint_score}/10")
        
        # Bandit security issues
        if "bandit" in lint_results and "issue_count" in lint_results["bandit"]:
            bandit_issues = lint_results["bandit"]["issue_count"]
            score -= min(bandit_issues * 3, 15)
            total_issues += bandit_issues
            if bandit_issues > 0:
                issues.append(f"{bandit_issues} security issues")
        
        # Complexity penalties
        complexity_results = results.get("complexity", {})
        if "cyclomatic" in complexity_results:
            cc_data = complexity_results["cyclomatic"]
            if "high_complexity_count" in cc_data:
                high_complexity = cc_data["high_complexity_count"]
                score -= min(high_complexity * 3, 10)
                if high_complexity > 0:
                    issues.append(f"{high_complexity} high complexity functions")
        
        # Determine grade
        final_score = max(0, round(score, 1))
        
        if final_score >= 95:
            grade = "A+ (Excellent)"
        elif final_score >= 90:
            grade = "A (Excellent)"
        elif final_score >= 85:
            grade = "B+ (Very Good)"
        elif final_score >= 80:
            grade = "B (Good)"
        elif final_score >= 75:
            grade = "C+ (Fair)"
        elif final_score >= 70:
            grade = "C (Fair)"
        elif final_score >= 65:
            grade = "D+ (Poor)"
        elif final_score >= 60:
            grade = "D (Poor)"
        else:
            grade = "F (Failing)"
        
        # Generate recommendations
        recommendations = []
        if total_issues > 0:
            recommendations.append("Address code quality issues identified by linters")
        if format_issues > 0:
            recommendations.append("Fix code formatting inconsistencies")
        if any("complexity" in issue.lower() for issue in issues):
            recommendations.append("Refactor complex functions to improve maintainability")
        if any("security" in issue.lower() for issue in issues):
            recommendations.append("Review and fix security vulnerabilities")
        
        return {
            "score": final_score,
            "grade": grade,
            "total_issues": total_issues,
            "issues": issues,
            "recommendations": recommendations
        }
    
    def generate_quality_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive quality report in markdown."""
        md_file = self.quality_reports_dir / f"quality_report_{self.timestamp}.md"
        
        with open(md_file, 'w') as f:
            f.write("# Comprehensive Code Quality Report\n\n")
            f.write(f"**Generated:** {results['timestamp']}\n")
            f.write(f"**Analysis Time:** {results['summary']['analysis_time_seconds']}s\n")
            f.write(f"**Quality Score:** {results['summary']['overall_quality_score']}/100\n")
            f.write(f"**Grade:** {results['summary']['quality_grade']}\n\n")
            
            # Executive Summary
            score = results['summary']['overall_quality_score']
            if score >= 90:
                f.write("## ?? EXCELLENT - High code quality maintained\n\n")
            elif score >= 80:
                f.write("## ?? GOOD - Minor issues to address\n\n")
            elif score >= 70:
                f.write("## ?? FAIR - Several quality issues detected\n\n")
            else:
                f.write("## ?? NEEDS IMPROVEMENT - Significant quality issues\n\n")
            
            # Tool Availability
            f.write("## Tool Availability\n\n")
            for tool, available in results['tools_available'].items():
                status = "? Available" if available else "? Missing"
                f.write(f"- **{tool}:** {status}\n")
            f.write("\n")
            
            # Detailed Results
            f.write("## Analysis Results\n\n")
            
            # Type Checking
            type_results = results.get('type_checking', {})
            if 'error_count' in type_results:
                f.write("### Type Checking (mypy)\n")
                error_count = type_results['error_count']
                status = "? No issues" if error_count == 0 else f"? {error_count} errors"
                f.write(f"- **Status:** {status}\n\n")
            
            # Linting
            lint_results = results.get('linting', {})
            f.write("### Linting Results\n")
            
            if 'flake8' in lint_results:
                flake8 = lint_results['flake8']
                if 'issue_count' in flake8:
                    issues = flake8['issue_count']
                    status = "? No issues" if issues == 0 else f"? {issues} issues"
                    f.write(f"- **flake8:** {status}\n")
            
            if 'pylint' in lint_results:
                pylint = lint_results['pylint']
                if 'score' in pylint:
                    score = pylint['score']
                    f.write(f"- **pylint Score:** {score}/10\n")
            
            if 'bandit' in lint_results:
                bandit = lint_results['bandit']
                if 'issue_count' in bandit:
                    issues = bandit['issue_count']
                    status = "? No issues" if issues == 0 else f"? {issues} security issues"
                    f.write(f"- **bandit:** {status}\n")
            
            f.write("\n")
            
            # Complexity Analysis
            complexity_results = results.get('complexity', {})
            if complexity_results and 'error' not in complexity_results:
                f.write("### Complexity Analysis\n")
                
                if 'cyclomatic' in complexity_results:
                    cc = complexity_results['cyclomatic']
                    if 'average_complexity' in cc:
                        f.write(f"- **Average Cyclomatic Complexity:** {cc['average_complexity']}\n")
                        f.write(f"- **High Complexity Functions:** {cc.get('high_complexity_count', 0)}\n")
                
                if 'halstead' in complexity_results:
                    hal = complexity_results['halstead']
                    if 'total_estimated_bugs' in hal:
                        f.write(f"- **Estimated Bugs (Halstead):** {hal['total_estimated_bugs']}\n")
                
                f.write("\n")
            
            # Recommendations
            recommendations = results['summary']['recommendations']
            if recommendations:
                f.write("## Recommendations\n\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            else:
                f.write("## Recommendations\n\n")
                f.write("? Code quality is excellent - no specific recommendations.\n")
        
        print(f"?? Quality report generated: {md_file}")
        return str(md_file)


def main():
    """Main entry point for code quality automation."""
    parser = argparse.ArgumentParser(description="Comprehensive code quality automation")
    parser.add_argument("--install", action="store_true", help="Install all tools")
    parser.add_argument("--configure", action="store_true", help="Setup all configurations")
    parser.add_argument("--check-types", action="store_true", help="Run type checking only")
    parser.add_argument("--format", action="store_true", help="Format code")
    parser.add_argument("--format-check", action="store_true", help="Check formatting")
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument("--complexity", action="store_true", help="Run complexity analysis")
    parser.add_argument("--all", action="store_true", help="Run comprehensive analysis")
    parser.add_argument("--setup", action="store_true", help="Install and configure everything")
    
    args = parser.parse_args()
    
    master = CodeQualityMaster()
    
    if args.setup:
        print("?? Setting up complete code quality automation...")
        master.install_all_tools()
        master.setup_all_configurations()
        print("? Code quality automation setup complete!")
        
    elif args.install:
        master.install_all_tools()
        
    elif args.configure:
        master.setup_all_configurations()
        
    elif args.check_types:
        results = master.run_type_checking()
        print(json.dumps(results, indent=2))
        
    elif args.format:
        results = master.run_formatting(check_only=False)
        print(json.dumps(results, indent=2))
        
    elif args.format_check:
        results = master.run_formatting(check_only=True)
        print(json.dumps(results, indent=2))
        
    elif args.lint:
        results = master.run_linting()
        print(json.dumps(results, indent=2))
        
    elif args.complexity:
        results = master.run_complexity_analysis()
        print(json.dumps(results, indent=2))
        
    elif args.all or len(sys.argv) == 1:
        master.run_comprehensive_analysis()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()