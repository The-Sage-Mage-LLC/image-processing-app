#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Workspace Cleanup - Phase 2
Continue organizing remaining files that need proper placement.
"""

import os
import shutil
from pathlib import Path
import glob


def organize_remaining_files():
    """Organize the remaining files that need proper placement."""
    print("?? FINAL CLEANUP - Organizing remaining files...")
    
    # Create additional directories
    additional_dirs = [
        "tools/analysis",
        "tools/deployment", 
        "tools/security",
        "scripts/setup",
        "scripts/demo"
    ]
    
    for dir_path in additional_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_path}")
    
    # File organization mapping
    file_moves = {
        # Analysis tools
        "tools/analysis/": [
            "code_analysis_comprehensive.py",
            "performance_benchmarks.py",
            "quality_status_report.py"
        ],
        
        # Deployment tools  
        "tools/deployment/": [
            "api_launcher.py",
            "Dockerfile"
        ],
        
        # Security tools
        "tools/security/": [
            "security_scanner.py"
        ],
        
        # Setup scripts
        "scripts/setup/": [
            "setup.py",
            "fix_numpy_compatibility.py"
        ],
        
        # Demo and validation
        "scripts/demo/": [
            "demo_modern_features.py",
            "nice_to_have_features.py"
        ],
        
        # Archive remaining verification files
        "archive/validation/": [
            "validate_enterprise_features.py",
            "validate_infrastructure.py", 
            "validation_report.py",
            "file_currency_assessment.py",
            "implementation_summary.py",
            "targeted_quality_fixes.py",
            "run_comprehensive_tests.py"
        ],
        
        # Config files should stay in root but organize them
        ".": [
            "main.py",
            "gui_launcher.py", 
            "launch_gui.bat",
            "requirements.txt",
            "requirements_modern.txt",
            "requirements - no versions.txt",
            "pyproject.toml",
            "poetry_config.toml",
            "README.md",
            ".gitignore",
            ".pre-commit-config.yaml",
            "docker-compose.yml",
            "LICENSE"
        ]
    }
    
    moved_count = 0
    
    for destination, files in file_moves.items():
        if destination != ".":  # Skip root directory files
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)
            
            for file_name in files:
                if os.path.isfile(file_name):
                    try:
                        target = dest_path / file_name
                        shutil.move(file_name, target)
                        print(f"  Moved: {file_name} -> {target}")
                        moved_count += 1
                    except Exception as e:
                        print(f"  Error moving {file_name}: {e}")
    
    print(f"Moved {moved_count} additional files")
    return moved_count


def organize_config_files():
    """Move config files to proper config directory.""" 
    print("?? Organizing configuration files...")
    
    # Config files that should be in config directory
    config_files = [
        ".env.example",
        ".flake8", 
        ".prettierrc.json",
        ".pylintrc",
        "mypy.ini"
    ]
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    moved_count = 0
    for config_file in config_files:
        if os.path.isfile(config_file):
            try:
                target = config_dir / config_file
                shutil.move(config_file, target)
                print(f"  Moved: {config_file} -> {target}")
                moved_count += 1
            except Exception as e:
                print(f"  Error moving {config_file}: {e}")
    
    return moved_count


def create_project_structure_summary():
    """Create final project structure documentation."""
    structure_content = """# Image Processing Application - Final Project Structure

## Root Directory (Essential Files Only)
```
image-processing-app/
??? main.py                     # Main application entry point
??? gui_launcher.py             # GUI application launcher
??? launch_gui.bat              # Windows batch launcher
??? requirements.txt            # Production dependencies  
??? requirements_modern.txt     # Modern development stack
??? pyproject.toml              # Modern Python packaging
??? poetry_config.toml          # Poetry configuration
??? README.md                   # Main project documentation
??? .gitignore                  # Git ignore patterns
??? .pre-commit-config.yaml     # Pre-commit hooks
??? docker-compose.yml          # Container orchestration
??? LICENSE                     # Project license
??? CLEANUP_SUMMARY.md          # Cleanup documentation
```

## Source Code Structure
```
src/
??? __init__.py                 # Package initialization
??? cli/                        # Command-line interface
??? core/                       # Core processing logic
??? gui/                        # GUI application
??? models/                     # AI/ML models
??? transforms/                 # Image transformations
??? utils/                      # Utility modules
??? web/                        # Web interface (optional)
??? access_control/             # Enterprise access control
```

## Configuration
```
config/
??? config.toml                 # Application configuration
??? .env.example                # Environment variables template
??? .flake8                     # Flake8 linting config
??? .prettierrc.json            # Prettier formatting config
??? .pylintrc                   # Pylint configuration
??? mypy.ini                    # MyPy type checking config
```

## Documentation
```
docs/
??? ENTERPRISE_FEATURES.md      # Enterprise features documentation
??? GUI_IMPLEMENTATION_COMPLETE.md # GUI implementation details
??? FINAL_QA_QC_VERIFICATION_REPORT.md # QA/QC verification
??? [Other specialized documentation]
```

## Tools and Scripts
```
tools/
??? maintenance/                # Code quality and maintenance
??? analysis/                   # Code analysis and performance
??? deployment/                 # Deployment tools
??? security/                   # Security scanning

scripts/
??? setup/                      # Setup and installation scripts
??? demo/                       # Demo and example scripts
```

## Archive
```
archive/
??? test_files/                 # Archived test files
??? temp_files/                 # Archived temporary files  
??? verification_files/         # Archived verification files
??? validation/                 # Archived validation scripts
```

## Production Deployment Files
- `main.py` - Application entry point
- `gui_launcher.py` - GUI launcher
- `requirements.txt` - Dependencies
- `config/config.toml` - Configuration
- `src/` - Source code

## Development Files
- `pyproject.toml` - Modern packaging
- `poetry_config.toml` - Poetry setup
- `docker-compose.yml` - Container setup
- `tools/` - Development tools
- `.pre-commit-config.yaml` - Git hooks

The workspace is now clean, organized, and production-ready! ?
"""
    
    with open("PROJECT_STRUCTURE.md", "w", encoding="utf-8") as f:
        f.write(structure_content)
    
    print("?? Created: PROJECT_STRUCTURE.md")
    return True


def main():
    """Main function for final cleanup."""
    print("=" * 80)
    print("FINAL WORKSPACE CLEANUP - PHASE 2")  
    print("=" * 80)
    
    # Organize remaining files
    moved_files = organize_remaining_files()
    print()
    
    # Organize config files
    moved_configs = organize_config_files() 
    print()
    
    # Create structure documentation
    create_project_structure_summary()
    print()
    
    print("=" * 80)
    print("? FINAL CLEANUP COMPLETE!")
    print("=" * 80)
    print(f"  ?? Additional files organized: {moved_files}")
    print(f"  ?? Config files organized: {moved_configs}")
    print(f"  ?? Project structure documented")
    print()
    print("?? Workspace is fully cleaned and production-ready!")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)