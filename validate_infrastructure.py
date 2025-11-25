#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Infrastructure Validation
Project ID: Image Processing App 20251119
Author: The-Sage-Mage
"""

import os
import sys
from pathlib import Path

def validate_files():
    """Validate that our new files were created correctly."""
    print("?? Validating Modern Infrastructure Files")
    print("=" * 50)
    
    required_files = {
        "pyproject.toml": "Modern Python packaging configuration",
        ".github/workflows/ci-cd.yml": "GitHub Actions CI/CD pipeline",
        ".github/dependabot.yml": "Automated dependency updates",
        ".pre-commit-config.yaml": "Pre-commit hooks configuration",
        "src/web/api_server.py": "FastAPI server with OpenAPI docs",
        "api_launcher.py": "API documentation generator",
        "setup_modern_infrastructure.py": "Infrastructure setup script",
        "setup_precommit.py": "Pre-commit setup script",
        "Dockerfile": "Docker containerization",
        "docker-compose.yml": "Docker orchestration",
        ".env.example": "Environment configuration template"
    }
    
    success_count = 0
    total_files = len(required_files)
    
    for file_path, description in required_files.items():
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"? {file_path} ({size:,} bytes)")
            print(f"   {description}")
            success_count += 1
        else:
            print(f"? {file_path} - MISSING")
            print(f"   {description}")
    
    print(f"\n?? Results: {success_count}/{total_files} files found")
    
    if success_count == total_files:
        print("?? ALL MODERN INFRASTRUCTURE FILES ARE PRESENT!")
        return True
    else:
        print("??  Some files are missing. Please check the setup.")
        return False

def validate_pyproject():
    """Validate pyproject.toml content."""
    print("\n?? Validating pyproject.toml Configuration")
    print("-" * 50)
    
    try:
        import tomli
        
        with open("pyproject.toml", "rb") as f:
            config = tomli.load(f)
        
        # Check key sections
        checks = [
            ("build-system", config.get("build-system")),
            ("project", config.get("project")),
            ("project.dependencies", config.get("project", {}).get("dependencies")),
            ("tool.black", config.get("tool", {}).get("black")),
            ("tool.ruff", config.get("tool", {}).get("ruff")),
            ("tool.pytest", config.get("tool", {}).get("pytest")),
            ("tool.mypy", config.get("tool", {}).get("mypy"))
        ]
        
        for section, value in checks:
            if value is not None:
                print(f"? {section} - configured")
            else:
                print(f"??  {section} - missing")
        
        # Check dependencies count
        deps = config.get("project", {}).get("dependencies", [])
        print(f"?? Dependencies: {len(deps)} packages configured")
        
        return True
        
    except Exception as e:
        print(f"? Error reading pyproject.toml: {e}")
        return False

def main():
    """Main validation function."""
    print("?? MODERN INFRASTRUCTURE VALIDATION")
    print("Project ID: Image Processing App 20251119")
    print("=" * 80)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Run validations
    files_ok = validate_files()
    config_ok = validate_pyproject()
    
    print("\n" + "=" * 80)
    print("?? VALIDATION SUMMARY")
    print("=" * 80)
    
    if files_ok and config_ok:
        print("? ALL VALIDATIONS PASSED!")
        print("\n?? Your modern infrastructure is ready:")
        print("   ?? Modern packaging with pyproject.toml")
        print("   ?? Automated CI/CD with GitHub Actions")
        print("   ?? Code quality with pre-commit hooks")
        print("   ?? API documentation with OpenAPI")
        print("   ?? Dependency updates with Dependabot")
        print("   ?? Containerization with Docker")
        
        print("\n?? Next steps:")
        print("   1. Install pre-commit: python setup_precommit.py --setup")
        print("   2. Test API docs: python api_launcher.py --generate-docs")
        print("   3. Build package: python -m build")
        print("   4. Run tests: pytest")
        
        return 0
    else:
        print("? SOME VALIDATIONS FAILED")
        print("Please check the setup and fix any issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())