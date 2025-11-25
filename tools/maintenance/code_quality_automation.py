#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static Type Checking Configuration and Automation
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive mypy configuration with enterprise-grade type checking.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import argparse


class TypeChecker:
    """Advanced static type checking automation with mypy."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.mypy_config_file = self.project_root / "mypy.ini"
        self.type_reports_dir = self.project_root / "type_reports"
        self.type_reports_dir.mkdir(exist_ok=True)
        
    def create_mypy_configuration(self) -> None:
        """Create comprehensive mypy configuration."""
        mypy_config = """[mypy]
python_version = 3.11
warn_return_any = true
warn_unused_configs = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
strict_concatenate = true

# Gradually enable strict mode
disallow_untyped_defs = false
disallow_incomplete_defs = false  
disallow_untyped_decorators = false
disallow_any_generics = false

# Enable these incrementally
check_untyped_defs = true
disallow_subclassing_any = false
disallow_untyped_calls = false
disallow_any_unimported = false
disallow_any_expr = false
disallow_any_decorated = false
disallow_any_explicit = false

# Error handling
no_implicit_optional = true
strict_optional = true

# Import handling
ignore_missing_imports = true
follow_imports = silent
show_error_codes = true
show_error_context = true
show_column_numbers = true
color_output = true
error_summary = true
pretty = true

# Performance
cache_dir = .mypy_cache
sqlite_cache = true
incremental = true

# Exclusions
exclude = (?x)(
    ^venv_312_fixed/
    | ^build/
    | ^dist/
    | ^\.eggs/
    | ^\.git/
    | ^\.mypy_cache/
    | ^\.pytest_cache/
    | ^\.tox/
    | ^__pycache__/
    | .*\.pyc$
    | .*\.pyo$
)

[mypy-tests.*]
disallow_untyped_defs = false
disallow_incomplete_defs = false
disallow_untyped_decorators = false
ignore_errors = false

[mypy-conftest]
ignore_errors = true

# Third-party libraries without stubs
[mypy-cv2.*]
ignore_missing_imports = true

[mypy-PIL.*]
ignore_missing_imports = true

[mypy-numpy.*]
ignore_missing_imports = true

[mypy-pandas.*]
ignore_missing_imports = true

[mypy-matplotlib.*]
ignore_missing_imports = true

[mypy-sklearn.*]
ignore_missing_imports = true

[mypy-PyQt6.*]
ignore_missing_imports = true

[mypy-reportlab.*]
ignore_missing_imports = true

[mypy-joblib.*]
ignore_missing_imports = true

[mypy-tqdm.*]
ignore_missing_imports = true

[mypy-pytest.*]
ignore_missing_imports = true

[mypy-setuptools.*]
ignore_missing_imports = true

[mypy-psutil.*]
ignore_missing_imports = true
"""
        
        with open(self.mypy_config_file, 'w') as f:
            f.write(mypy_config)
        
        print(f"? Created mypy configuration: {self.mypy_config_file}")
    
    def install_mypy_and_types(self) -> bool:
        """Install mypy and type stubs."""
        print("?? Installing mypy and type stubs...")
        
        packages = [
            "mypy",
            "types-requests",
            "types-PyYAML", 
            "types-setuptools",
            "types-toml",
            "types-colorama",
            "types-tqdm",
            "types-Pillow",
        ]
        
        for package in packages:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"   ? Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"   ?? Failed to install {package}: {e.stderr}")
        
        return True
    
    def run_type_check(self, paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run mypy type checking."""
        if paths is None:
            paths = ["src/", "tests/"]
        
        print("?? Running mypy type checking...")
        
        # Ensure mypy config exists
        if not self.mypy_config_file.exists():
            self.create_mypy_configuration()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for path in paths:
            path_obj = self.project_root / path
            if not path_obj.exists():
                print(f"   ?? Path does not exist: {path}")
                continue
            
            print(f"   ?? Checking {path}...")
            
            try:
                # Run mypy with JSON output
                result = subprocess.run(
                    [
                        sys.executable, "-m", "mypy",
                        str(path_obj),
                        "--config-file", str(self.mypy_config_file),
                        "--json-report", str(self.type_reports_dir / f"mypy_report_{timestamp}.json"),
                        "--html-report", str(self.type_reports_dir / f"mypy_html_{timestamp}"),
                        "--txt-report", str(self.type_reports_dir / f"mypy_text_{timestamp}.txt"),
                    ],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                # Parse results
                error_count = result.stdout.count("error:")
                warning_count = result.stdout.count("note:")
                
                if result.returncode == 0:
                    print(f"   ? {path}: No type errors found")
                else:
                    print(f"   ? {path}: {error_count} errors, {warning_count} warnings")
                    
                    # Show first few errors
                    lines = result.stdout.split('\n')
                    error_lines = [line for line in lines if "error:" in line][:5]
                    for error in error_lines:
                        print(f"      {error}")
                
                # Save detailed output
                output_file = self.type_reports_dir / f"mypy_output_{path.replace('/', '_')}_{timestamp}.txt"
                with open(output_file, 'w') as f:
                    f.write(f"MyPy Type Check Report\n")
                    f.write(f"Path: {path}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Return Code: {result.returncode}\n")
                    f.write(f"Errors: {error_count}\n")
                    f.write(f"Warnings: {warning_count}\n")
                    f.write("\n" + "="*50 + "\n")
                    f.write("STDOUT:\n")
                    f.write(result.stdout)
                    f.write("\n" + "="*50 + "\n")
                    f.write("STDERR:\n")
                    f.write(result.stderr)
                
            except subprocess.CalledProcessError as e:
                print(f"   ? MyPy execution failed for {path}: {e}")
            except FileNotFoundError:
                print("   ? MyPy not found. Install with: pip install mypy")
                return {"error": "MyPy not installed"}
        
        return self.generate_type_checking_report()
    
    def generate_type_checking_report(self) -> Dict[str, Any]:
        """Generate comprehensive type checking report."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "tool": "mypy",
            "configuration": str(self.mypy_config_file),
            "results": {}
        }
        
        # Find latest reports
        json_reports = list(self.type_reports_dir.glob("mypy_report_*.json"))
        if json_reports:
            latest_report = max(json_reports, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_report, 'r') as f:
                    mypy_data = json.load(f)
                    report_data["results"] = mypy_data
            except (json.JSONDecodeError, FileNotFoundError):
                print("   ?? Could not parse MyPy JSON report")
        
        # Save comprehensive report
        report_file = self.type_reports_dir / f"type_checking_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"?? Type checking report saved: {report_file}")
        return report_data
    
    def setup_vscode_integration(self) -> None:
        """Setup VS Code integration for mypy."""
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        settings = {
            "python.linting.enabled": True,
            "python.linting.mypyEnabled": True,
            "python.linting.mypyArgs": [
                "--config-file", "mypy.ini",
                "--show-column-numbers",
                "--show-error-codes"
            ],
            "python.analysis.typeCheckingMode": "basic",
            "python.analysis.autoImportCompletions": True,
            "mypy-type-checker.args": [
                "--config-file=mypy.ini"
            ]
        }
        
        settings_file = vscode_dir / "settings.json"
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        
        print(f"? VS Code integration configured: {settings_file}")


## 2. Code Formatting Automation (Black/Prettier)

<function_calls>
<invoke name="create_file">
<parameter name="content">#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Formatting Automation
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Automated code formatting with Black and Prettier.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse


class CodeFormatter:
    """Automated code formatting with multiple tools."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.format_reports_dir = self.project_root / "format_reports"
        self.format_reports_dir.mkdir(exist_ok=True)
    
    def install_formatters(self) -> bool:
        """Install code formatting tools."""
        print("?? Installing code formatters...")
        
        formatters = [
            "black",
            "isort", 
            "autopep8",
            "prettier",  # For non-Python files
        ]
        
        for formatter in formatters:
            try:
                if formatter == "prettier":
                    # Install prettier via npm if available
                    try:
                        subprocess.run(["npm", "install", "-g", "prettier"], check=True, capture_output=True)
                        print(f"   ? Installed {formatter} (npm)")
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        print(f"   ?? Could not install {formatter} - npm not available")
                else:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", formatter],
                        check=True,
                        capture_output=True
                    )
                    print(f"   ? Installed {formatter}")
            except subprocess.CalledProcessError as e:
                print(f"   ? Failed to install {formatter}")
        
        return True
    
    def setup_black_configuration(self) -> None:
        """Setup Black configuration."""
        black_config = """[tool.black]
line-length = 88
target-version = ['py311', 'py312', 'py313']
include = '\\.pyi?$'
extend-exclude = '''
/(
    \\.eggs
    | \\.git
    | \\.hg
    | \\.mypy_cache
    | \\.tox
    | \\.venv
    | _build
    | buck-out
    | build
    | dist
    | venv_312_fixed
)/
'''
"""
        
        # Add to pyproject.toml if it doesn't exist there
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            content = pyproject_file.read_text()
            if "[tool.black]" not in content:
                with open(pyproject_file, 'a') as f:
                    f.write("\n" + black_config)
                print("? Added Black configuration to pyproject.toml")
        else:
            # Create separate black config
            black_config_file = self.project_root / "black.toml"
            with open(black_config_file, 'w') as f:
                f.write(black_config)
            print(f"? Created Black configuration: {black_config_file}")
    
    def setup_isort_configuration(self) -> None:
        """Setup isort configuration."""
        isort_config = """[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["src"]
known_third_party = ["pytest", "numpy", "pandas", "PIL", "cv2", "PyQt6"]
sections = ["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]
force_single_line = false
lines_after_imports = 2
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
"""
        
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            content = pyproject_file.read_text()
            if "[tool.isort]" not in content:
                with open(pyproject_file, 'a') as f:
                    f.write("\n" + isort_config)
                print("? Added isort configuration to pyproject.toml")
    
    def setup_prettier_configuration(self) -> None:
        """Setup Prettier configuration for non-Python files."""
        prettier_config = {
            "semi": True,
            "trailingComma": "es5",
            "singleQuote": False,
            "printWidth": 80,
            "tabWidth": 2,
            "useTabs": False,
            "overrides": [
                {
                    "files": "*.md",
                    "options": {
                        "printWidth": 100,
                        "proseWrap": "always"
                    }
                },
                {
                    "files": ["*.yml", "*.yaml"],
                    "options": {
                        "tabWidth": 2
                    }
                },
                {
                    "files": "*.json",
                    "options": {
                        "tabWidth": 2
                    }
                }
            ]
        }
        
        prettier_config_file = self.project_root / ".prettierrc.json"
        with open(prettier_config_file, 'w') as f:
            json.dump(prettier_config, f, indent=2)
        
        # Create .prettierignore
        prettier_ignore = """# Dependencies
node_modules/
venv/
venv_*/
.venv/

# Build outputs
build/
dist/
*.egg-info/

# Cache
.mypy_cache/
.pytest_cache/
__pycache__/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Temporary
*.tmp
*.temp
"""
        
        prettier_ignore_file = self.project_root / ".prettierignore"
        with open(prettier_ignore_file, 'w') as f:
            f.write(prettier_ignore)
        
        print("? Configured Prettier for Markdown, YAML, and JSON")
    
    def format_python_code(self, check_only: bool = False) -> Dict[str, Any]:
        """Format Python code with Black and isort."""
        print("?? Formatting Python code...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "formatters": {},
            "summary": {}
        }
        
        python_paths = ["src/", "tests/", "*.py"]
        existing_paths = []
        
        for path in python_paths:
            if path.endswith("*.py"):
                # Find Python files in root
                existing_paths.extend(self.project_root.glob("*.py"))
            else:
                path_obj = self.project_root / path
                if path_obj.exists():
                    existing_paths.append(path_obj)
        
        # Format with Black
        try:
            black_cmd = [sys.executable, "-m", "black"]
            if check_only:
                black_cmd.append("--check")
            black_cmd.extend([str(p) for p in existing_paths])
            
            black_result = subprocess.run(
                black_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            results["formatters"]["black"] = {
                "return_code": black_result.returncode,
                "stdout": black_result.stdout,
                "stderr": black_result.stderr,
                "files_formatted": black_result.stdout.count("reformatted") if not check_only else 0,
                "files_checked": black_result.stdout.count("would reformat") if check_only else black_result.stdout.count("left unchanged")
            }
            
            if black_result.returncode == 0:
                action = "checked" if check_only else "formatted"
                print(f"   ? Black {action} successfully")
            else:
                print(f"   ?? Black formatting issues found")
                
        except FileNotFoundError:
            print("   ? Black not found. Install with: pip install black")
            results["formatters"]["black"] = {"error": "not_installed"}
        
        # Sort imports with isort
        try:
            isort_cmd = [sys.executable, "-m", "isort"]
            if check_only:
                isort_cmd.append("--check-only")
            isort_cmd.extend([str(p) for p in existing_paths])
            
            isort_result = subprocess.run(
                isort_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            results["formatters"]["isort"] = {
                "return_code": isort_result.returncode,
                "stdout": isort_result.stdout,
                "stderr": isort_result.stderr
            }
            
            if isort_result.returncode == 0:
                action = "checked" if check_only else "sorted"
                print(f"   ? isort imports {action} successfully")
            else:
                print(f"   ?? isort found import issues")
                
        except FileNotFoundError:
            print("   ? isort not found. Install with: pip install isort")
            results["formatters"]["isort"] = {"error": "not_installed"}
        
        return results
    
    def format_config_files(self, check_only: bool = False) -> Dict[str, Any]:
        """Format configuration files with Prettier."""
        print("?? Formatting configuration files...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "prettier": {}
        }
        
        # Find files to format
        config_patterns = ["*.md", "*.yml", "*.yaml", "*.json"]
        config_files = []
        
        for pattern in config_patterns:
            config_files.extend(self.project_root.glob(pattern))
            # Also check .github directory
            github_dir = self.project_root / ".github"
            if github_dir.exists():
                config_files.extend(github_dir.rglob(pattern))
        
        if not config_files:
            print("   ?? No configuration files found")
            return results
        
        try:
            prettier_cmd = ["prettier"]
            if check_only:
                prettier_cmd.append("--check")
            else:
                prettier_cmd.append("--write")
            
            prettier_cmd.extend([str(f) for f in config_files])
            
            prettier_result = subprocess.run(
                prettier_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            results["prettier"] = {
                "return_code": prettier_result.returncode,
                "stdout": prettier_result.stdout,
                "stderr": prettier_result.stderr,
                "files_processed": len(config_files)
            }
            
            if prettier_result.returncode == 0:
                action = "checked" if check_only else "formatted"
                print(f"   ? Prettier {action} {len(config_files)} files")
            else:
                print(f"   ?? Prettier found formatting issues")
                
        except FileNotFoundError:
            print("   ? Prettier not found. Install with: npm install -g prettier")
            results["prettier"] = {"error": "not_installed"}
        
        return results
    
    def setup_format_automation(self) -> None:
        """Setup automated formatting workflows."""
        print("?? Setting up format automation...")
        
        # Create format script
        format_script_content = '''#!/usr/bin/env python3
"""Automated code formatting script."""

import sys
import subprocess
from pathlib import Path

def main():
    """Run all formatters."""
    project_root = Path(__file__).parent
    
    print("?? Running automated code formatting...")
    
    # Run Black
    subprocess.run([
        sys.executable, "-m", "black",
        "src/", "tests/", "*.py"
    ], cwd=project_root)
    
    # Run isort
    subprocess.run([
        sys.executable, "-m", "isort", 
        "src/", "tests/", "*.py"
    ], cwd=project_root)
    
    # Run Prettier (if available)
    try:
        subprocess.run([
            "prettier", "--write",
            "*.md", "*.yml", "*.yaml", "*.json"
        ], cwd=project_root, check=False)
    except FileNotFoundError:
        print("?? Prettier not available")
    
    print("? Code formatting complete")

if __name__ == "__main__":
    main()
'''
        
        format_script = self.project_root / "format_code.py"
        with open(format_script, 'w') as f:
            f.write(format_script_content)
        
        print(f"? Created format automation script: {format_script}")
        
        # Make executable on Unix systems
        try:
            import stat
            format_script.chmod(format_script.stat().st_mode | stat.S_IEXEC)
        except:
            pass


def main():
    """Main entry point for type checking setup."""
    parser = argparse.ArgumentParser(description="Static type checking and formatting setup")
    parser.add_argument("--install", action="store_true", help="Install tools")
    parser.add_argument("--configure", action="store_true", help="Setup configurations")
    parser.add_argument("--check", action="store_true", help="Run type checking")
    parser.add_argument("--format", action="store_true", help="Format code")
    parser.add_argument("--format-check", action="store_true", help="Check formatting")
    parser.add_argument("--all", action="store_true", help="Setup everything")
    
    args = parser.parse_args()
    
    type_checker = TypeChecker()
    code_formatter = CodeFormatter()
    
    if args.all or args.install:
        type_checker.install_mypy_and_types()
        code_formatter.install_formatters()
    
    if args.all or args.configure:
        type_checker.create_mypy_configuration()
        type_checker.setup_vscode_integration()
        code_formatter.setup_black_configuration()
        code_formatter.setup_isort_configuration()
        code_formatter.setup_prettier_configuration()
        code_formatter.setup_format_automation()
    
    if args.check:
        type_checker.run_type_check()
    
    if args.format:
        code_formatter.format_python_code(check_only=False)
        code_formatter.format_config_files(check_only=False)
    
    if args.format_check:
        code_formatter.format_python_code(check_only=True)
        code_formatter.format_config_files(check_only=True)
    
    if len(sys.argv) == 1:
        # Default: setup everything
        print("?? Setting up comprehensive code quality automation...")
        type_checker.install_mypy_and_types()
        type_checker.create_mypy_configuration()
        type_checker.setup_vscode_integration()
        code_formatter.install_formatters()
        code_formatter.setup_black_configuration()
        code_formatter.setup_isort_configuration()
        code_formatter.setup_prettier_configuration()
        code_formatter.setup_format_automation()
        print("? Code quality automation setup complete!")


if __name__ == "__main__":
    main()