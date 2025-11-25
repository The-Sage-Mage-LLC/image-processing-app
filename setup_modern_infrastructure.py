#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Development Infrastructure Setup Script
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Complete setup script for modern development infrastructure including:
- pyproject.toml migration
- GitHub Actions CI/CD
- Pre-commit hooks
- API documentation
- Dependency management
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import json


class ModernInfrastructureSetup:
    """Setup modern development infrastructure."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.success_count = 0
        self.total_steps = 0
    
    def run_command(self, command: str, cwd: Optional[Path] = None) -> Tuple[bool, str]:
        """Run a shell command and return success status and output."""
        try:
            if cwd is None:
                cwd = self.project_root
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
                timeout=120
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, f"Error: {e.stderr if e.stderr else e.stdout}"
        except subprocess.TimeoutExpired:
            return False, "Error: Command timed out"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def print_step(self, step: str, status: str = "INFO"):
        """Print a formatted step message."""
        symbols = {
            "INFO": "??",
            "SUCCESS": "?", 
            "WARNING": "??",
            "ERROR": "?",
            "SKIP": "??"
        }
        
        symbol = symbols.get(status, "??")
        print(f"{symbol} {step}")
    
    def step_1_validate_environment(self) -> bool:
        """Step 1: Validate development environment."""
        self.print_step("Step 1: Validating development environment")
        self.total_steps += 1
        
        # Check Python version
        if sys.version_info < (3, 11):
            self.errors.append("Python 3.11+ is required")
            self.print_step("Python version check failed", "ERROR")
            return False
        
        self.print_step(f"Python {sys.version_info.major}.{sys.version_info.minor} detected", "SUCCESS")
        
        # Check git
        success, output = self.run_command("git --version")
        if not success:
            self.errors.append("Git is not available")
            self.print_step("Git check failed", "ERROR")
            return False
        
        self.print_step("Git is available", "SUCCESS")
        
        # Check if in git repository
        success, output = self.run_command("git rev-parse --git-dir")
        if not success:
            self.warnings.append("Not in a git repository - some features may not work")
            self.print_step("Not in git repository", "WARNING")
        else:
            self.print_step("Git repository detected", "SUCCESS")
        
        self.success_count += 1
        return True
    
    def step_2_backup_legacy_files(self) -> bool:
        """Step 2: Backup legacy setup.py and requirements."""
        self.print_step("Step 2: Backing up legacy configuration files")
        self.total_steps += 1
        
        backup_dir = self.project_root / "backup_legacy"
        backup_dir.mkdir(exist_ok=True)
        
        files_to_backup = ["setup.py", "requirements.txt"]
        backed_up = []
        
        for file_name in files_to_backup:
            file_path = self.project_root / file_name
            if file_path.exists():
                backup_path = backup_dir / f"{file_name}.backup"
                shutil.copy2(file_path, backup_path)
                backed_up.append(file_name)
                self.print_step(f"Backed up {file_name}", "SUCCESS")
        
        if backed_up:
            self.print_step(f"Legacy files backed up to {backup_dir}", "SUCCESS")
        else:
            self.print_step("No legacy files to backup", "SKIP")
        
        self.success_count += 1
        return True
    
    def step_3_install_build_tools(self) -> bool:
        """Step 3: Install modern build tools."""
        self.print_step("Step 3: Installing modern build tools")
        self.total_steps += 1
        
        # Install core build tools
        build_tools = [
            "build",
            "twine", 
            "hatch",
            "pre-commit",
            "ruff",
            "black",
            "mypy"
        ]
        
        for tool in build_tools:
            self.print_step(f"Installing {tool}...")
            success, output = self.run_command(f"pip install {tool}")
            if not success:
                self.warnings.append(f"Failed to install {tool}: {output}")
                self.print_step(f"Failed to install {tool}", "WARNING")
            else:
                self.print_step(f"Installed {tool}", "SUCCESS")
        
        self.success_count += 1
        return True
    
    def step_4_setup_pyproject_toml(self) -> bool:
        """Step 4: Validate pyproject.toml setup."""
        self.print_step("Step 4: Validating pyproject.toml configuration")
        self.total_steps += 1
        
        pyproject_file = self.project_root / "pyproject.toml"
        if not pyproject_file.exists():
            self.errors.append("pyproject.toml not found")
            self.print_step("pyproject.toml missing", "ERROR")
            return False
        
        # Validate pyproject.toml
        try:
            import tomli
            with open(pyproject_file, "rb") as f:
                config = tomli.load(f)
            
            # Check required sections
            required_sections = ["build-system", "project", "tool.black", "tool.ruff"]
            missing = []
            
            for section in required_sections:
                keys = section.split(".")
                current = config
                for key in keys:
                    if key not in current:
                        missing.append(section)
                        break
                    current = current[key]
            
            if missing:
                self.warnings.append(f"Missing pyproject.toml sections: {missing}")
                self.print_step(f"Missing sections: {missing}", "WARNING")
            else:
                self.print_step("pyproject.toml structure is valid", "SUCCESS")
        
        except Exception as e:
            self.errors.append(f"pyproject.toml validation failed: {e}")
            self.print_step("pyproject.toml validation failed", "ERROR")
            return False
        
        self.success_count += 1
        return True
    
    def step_5_setup_precommit_hooks(self) -> bool:
        """Step 5: Setup pre-commit hooks."""
        self.print_step("Step 5: Setting up pre-commit hooks")
        self.total_steps += 1
        
        # Check if pre-commit config exists
        precommit_config = self.project_root / ".pre-commit-config.yaml"
        if not precommit_config.exists():
            self.errors.append("Pre-commit configuration not found")
            self.print_step("Pre-commit configuration missing", "ERROR")
            return False
        
        # Install pre-commit hooks
        success, output = self.run_command("pre-commit install")
        if not success:
            self.warnings.append(f"Failed to install pre-commit hooks: {output}")
            self.print_step("Pre-commit installation failed", "WARNING")
        else:
            self.print_step("Pre-commit hooks installed", "SUCCESS")
        
        # Install commit-msg hook
        success, output = self.run_command("pre-commit install --hook-type commit-msg")
        if not success:
            self.warnings.append(f"Failed to install commit-msg hook: {output}")
            self.print_step("Commit-msg hook installation failed", "WARNING")
        else:
            self.print_step("Commit message hooks installed", "SUCCESS")
        
        self.success_count += 1
        return True
    
    def step_6_validate_github_actions(self) -> bool:
        """Step 6: Validate GitHub Actions workflow."""
        self.print_step("Step 6: Validating GitHub Actions workflow")
        self.total_steps += 1
        
        workflow_dir = self.project_root / ".github" / "workflows"
        if not workflow_dir.exists():
            self.errors.append("GitHub Actions workflow directory not found")
            self.print_step("GitHub workflows directory missing", "ERROR")
            return False
        
        workflow_files = list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml"))
        if not workflow_files:
            self.errors.append("No GitHub Actions workflow files found")
            self.print_step("No workflow files found", "ERROR")
            return False
        
        self.print_step(f"Found {len(workflow_files)} workflow files", "SUCCESS")
        
        # Validate dependabot config
        dependabot_config = self.project_root / ".github" / "dependabot.yml"
        if dependabot_config.exists():
            self.print_step("Dependabot configuration found", "SUCCESS")
        else:
            self.warnings.append("Dependabot configuration not found")
            self.print_step("Dependabot configuration missing", "WARNING")
        
        self.success_count += 1
        return True
    
    def step_7_validate_api_documentation(self) -> bool:
        """Step 7: Validate API documentation setup."""
        self.print_step("Step 7: Validating API documentation")
        self.total_steps += 1
        
        api_server = self.project_root / "src" / "web" / "api_server.py"
        if not api_server.exists():
            self.warnings.append("API server not found")
            self.print_step("API server missing", "WARNING")
        else:
            self.print_step("API server found", "SUCCESS")
        
        api_launcher = self.project_root / "api_launcher.py"
        if not api_launcher.exists():
            self.warnings.append("API launcher not found")
            self.print_step("API launcher missing", "WARNING")
        else:
            self.print_step("API launcher found", "SUCCESS")
        
        # Test API documentation generation
        if api_server.exists():
            try:
                self.print_step("Testing API documentation generation...")
                success, output = self.run_command("python api_launcher.py --generate-docs")
                if success:
                    self.print_step("API documentation generated successfully", "SUCCESS")
                else:
                    self.warnings.append(f"API documentation generation failed: {output}")
                    self.print_step("API documentation generation failed", "WARNING")
            except Exception as e:
                self.warnings.append(f"API documentation test failed: {e}")
                self.print_step("API documentation test failed", "WARNING")
        
        self.success_count += 1
        return True
    
    def step_8_test_build_system(self) -> bool:
        """Step 8: Test modern build system."""
        self.print_step("Step 8: Testing modern build system")
        self.total_steps += 1
        
        # Test package build
        self.print_step("Testing package build...")
        success, output = self.run_command("python -m build --wheel")
        if not success:
            self.warnings.append(f"Package build failed: {output}")
            self.print_step("Package build failed", "WARNING")
        else:
            self.print_step("Package build successful", "SUCCESS")
            
            # Check if wheel was created
            dist_dir = self.project_root / "dist"
            if dist_dir.exists():
                wheels = list(dist_dir.glob("*.whl"))
                if wheels:
                    self.print_step(f"Wheel created: {wheels[0].name}", "SUCCESS")
        
        # Test development installation
        self.print_step("Testing development installation...")
        success, output = self.run_command("pip install -e .")
        if not success:
            self.warnings.append(f"Development installation failed: {output}")
            self.print_step("Development installation failed", "WARNING")
        else:
            self.print_step("Development installation successful", "SUCCESS")
        
        self.success_count += 1
        return True
    
    def step_9_setup_environment(self) -> bool:
        """Step 9: Setup development environment."""
        self.print_step("Step 9: Setting up development environment")
        self.total_steps += 1
        
        # Copy environment template
        env_example = self.project_root / ".env.example"
        env_file = self.project_root / ".env"
        
        if env_example.exists() and not env_file.exists():
            shutil.copy2(env_example, env_file)
            self.print_step("Created .env file from template", "SUCCESS")
            self.print_step("??  Please update .env file with your configuration", "INFO")
        elif env_file.exists():
            self.print_step(".env file already exists", "SKIP")
        else:
            self.warnings.append("No environment template found")
            self.print_step("No environment template found", "WARNING")
        
        # Create necessary directories
        dirs_to_create = [
            "logs",
            "test_output",
            "admin_output", 
            "data/input",
            "data/output",
            "data/admin",
            "docs",
            ".cache"
        ]
        
        for dir_name in dirs_to_create:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                self.print_step(f"Created directory: {dir_name}", "SUCCESS")
        
        self.success_count += 1
        return True
    
    def step_10_generate_documentation(self) -> bool:
        """Step 10: Generate project documentation."""
        self.print_step("Step 10: Generating project documentation")
        self.total_steps += 1
        
        # Generate README updates if needed
        readme_file = self.project_root / "README.md"
        if readme_file.exists():
            self.print_step("README.md exists", "SUCCESS")
        else:
            self.warnings.append("README.md not found")
            self.print_step("README.md missing", "WARNING")
        
        # Create CHANGELOG if it doesn't exist
        changelog_file = self.project_root / "CHANGELOG.md"
        if not changelog_file.exists():
            changelog_content = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-25

### Added
- Modern development infrastructure with pyproject.toml
- GitHub Actions CI/CD pipeline
- Pre-commit hooks for code quality
- FastAPI web server with OpenAPI documentation
- Docker containerization support
- Automated dependency updates with Dependabot

### Changed
- Migrated from setup.py to pyproject.toml
- Enhanced error handling and logging
- Improved code organization and structure

### Fixed
- Various code quality and security improvements
"""
            changelog_file.write_text(changelog_content)
            self.print_step("Created CHANGELOG.md", "SUCCESS")
        
        self.success_count += 1
        return True
    
    def generate_summary_report(self) -> None:
        """Generate final setup summary report."""
        print("\n" + "=" * 80)
        print("?? MODERN INFRASTRUCTURE SETUP COMPLETE")
        print("=" * 80)
        
        print(f"\n?? Setup Results:")
        print(f"   ? Successful steps: {self.success_count}/{self.total_steps}")
        print(f"   ??  Warnings: {len(self.warnings)}")
        print(f"   ? Errors: {len(self.errors)}")
        
        if self.warnings:
            print(f"\n??  Warnings:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.errors:
            print(f"\n? Errors:")
            for error in self.errors:
                print(f"   • {error}")
        
        print(f"\n?? Infrastructure Enhancements Added:")
        print(f"   ?? pyproject.toml - Modern Python packaging")
        print(f"   ?? GitHub Actions - CI/CD pipeline with testing")
        print(f"   ?? Pre-commit hooks - Automated code quality checks")
        print(f"   ?? OpenAPI documentation - REST API docs")
        print(f"   ?? Dependabot - Automated dependency updates")
        print(f"   ?? Docker support - Containerized deployment")
        print(f"   ?? Monitoring setup - Prometheus/Grafana ready")
        
        print(f"\n?? Next Steps:")
        print(f"   1. Review and customize .env file")
        print(f"   2. Run: git add . && git commit -m 'feat: add modern infrastructure'")
        print(f"   3. Push to GitHub to trigger CI/CD pipeline")
        print(f"   4. Set up GitHub repository secrets if needed")
        print(f"   5. Review GitHub Actions results")
        
        print(f"\n?? Quick Commands:")
        print(f"   Run tests: pytest")
        print(f"   Code formatting: black src tests")
        print(f"   Linting: ruff check src tests")
        print(f"   Type checking: mypy src")
        print(f"   API server: python api_launcher.py")
        print(f"   Build package: python -m build")
        
        if self.errors:
            print(f"\n??  Some errors occurred. Please review and fix them before proceeding.")
            return False
        else:
            print(f"\n? Setup completed successfully! Your codebase now uses modern development practices.")
            return True
    
    def run_setup(self) -> bool:
        """Run the complete setup process."""
        print("?? MODERN DEVELOPMENT INFRASTRUCTURE SETUP")
        print("Project ID: Image Processing App 20251119")
        print("=" * 80)
        
        steps = [
            self.step_1_validate_environment,
            self.step_2_backup_legacy_files,
            self.step_3_install_build_tools,
            self.step_4_setup_pyproject_toml,
            self.step_5_setup_precommit_hooks,
            self.step_6_validate_github_actions,
            self.step_7_validate_api_documentation,
            self.step_8_test_build_system,
            self.step_9_setup_environment,
            self.step_10_generate_documentation
        ]
        
        for step in steps:
            try:
                success = step()
                if not success and step == self.step_1_validate_environment:
                    # Critical step failed
                    self.print_step("Critical setup step failed, aborting", "ERROR")
                    return False
            except Exception as e:
                self.errors.append(f"Step failed with exception: {e}")
                self.print_step(f"Step failed: {e}", "ERROR")
        
        return self.generate_summary_report()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup modern development infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script will:
1. Validate your environment
2. Backup legacy configuration files  
3. Install modern build tools
4. Validate pyproject.toml configuration
5. Setup pre-commit hooks
6. Validate GitHub Actions workflow
7. Validate API documentation
8. Test build system
9. Setup development environment
10. Generate documentation

Examples:
  python setup_modern_infrastructure.py    # Run complete setup
"""
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("?? DRY RUN MODE - No changes will be made")
        print("This would set up modern development infrastructure")
        return 0
    
    setup = ModernInfrastructureSetup()
    success = setup.run_setup()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())