#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poetry Migration and Management Script
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive Poetry setup and migration from pip requirements.
"""

import subprocess
import sys
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse


class PoetryManager:
    """Poetry dependency management and migration tool."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.poetry_config_file = self.project_root / "poetry_config.toml"
        self.current_pyproject = self.project_root / "pyproject.toml"
        
    def run_command(self, command: str, capture_output: bool = True) -> Tuple[bool, str]:
        """Run a shell command and return success status and output."""
        try:
            if capture_output:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=300
                )
                return result.returncode == 0, result.stdout + result.stderr
            else:
                result = subprocess.run(command, shell=True, cwd=self.project_root)
                return result.returncode == 0, ""
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def check_poetry_installed(self) -> bool:
        """Check if Poetry is installed."""
        success, output = self.run_command("poetry --version")
        if success:
            version = output.strip()
            print(f"? Poetry is installed: {version}")
            return True
        else:
            print("? Poetry is not installed")
            return False
    
    def install_poetry(self) -> bool:
        """Install Poetry using the official installer."""
        print("?? Installing Poetry...")
        
        # Try different installation methods
        methods = [
            "pip install poetry",
            "pipx install poetry",
        ]
        
        for method in methods:
            print(f"   Trying: {method}")
            success, output = self.run_command(method)
            if success:
                print("? Poetry installed successfully")
                return True
            else:
                print(f"   Failed: {output}")
        
        print("? Failed to install Poetry with all methods")
        print("?? Please install Poetry manually:")
        print("   curl -sSL https://install.python-poetry.org | python3 -")
        print("   or visit: https://python-poetry.org/docs/#installation")
        return False
    
    def backup_current_config(self) -> bool:
        """Backup current configuration files."""
        print("?? Backing up current configuration...")
        
        backup_dir = self.project_root / "backup_poetry_migration"
        backup_dir.mkdir(exist_ok=True)
        
        files_to_backup = [
            "pyproject.toml",
            "requirements.txt",
            "requirements-dev.txt",
        ]
        
        for file_name in files_to_backup:
            file_path = self.project_root / file_name
            if file_path.exists():
                backup_path = backup_dir / f"{file_name}.backup"
                shutil.copy2(file_path, backup_path)
                print(f"   ? Backed up {file_name}")
        
        return True
    
    def init_poetry_project(self) -> bool:
        """Initialize Poetry project."""
        print("?? Initializing Poetry project...")
        
        # Check if poetry config exists
        if not self.poetry_config_file.exists():
            print("? Poetry configuration template not found")
            return False
        
        # Copy poetry config to pyproject.toml
        try:
            # Read the poetry configuration
            with open(self.poetry_config_file, 'r') as f:
                poetry_content = f.read()
            
            # Read existing pyproject.toml if it exists
            existing_content = ""
            if self.current_pyproject.exists():
                with open(self.current_pyproject, 'r') as f:
                    existing_content = f.read()
            
            # Merge configurations (keeping tool configurations from existing)
            print("   ?? Merging Poetry config with existing pyproject.toml...")
            
            # For now, save poetry config separately and let user choose
            poetry_pyproject = self.project_root / "pyproject_poetry.toml"
            with open(poetry_pyproject, 'w') as f:
                f.write(poetry_content)
                # Add tool configurations from existing pyproject.toml
                f.write("\n\n# Tool configurations from existing pyproject.toml\n")
                
                # Extract tool configurations from existing file
                if existing_content:
                    lines = existing_content.split('\n')
                    in_tool_section = False
                    for line in lines:
                        if line.startswith('[tool.') and not line.startswith('[tool.poetry'):
                            in_tool_section = True
                        elif line.startswith('[') and not line.startswith('[tool.'):
                            in_tool_section = False
                        
                        if in_tool_section:
                            f.write(line + '\n')
            
            print(f"   ? Poetry configuration saved to {poetry_pyproject}")
            print("   ?? Review the new configuration and replace pyproject.toml when ready")
            
            return True
            
        except Exception as e:
            print(f"? Failed to initialize Poetry project: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install dependencies using Poetry."""
        print("?? Installing dependencies with Poetry...")
        
        # First, ensure we have a poetry.lock file
        success, output = self.run_command("poetry lock")
        if not success:
            print(f"? Failed to generate poetry.lock: {output}")
            return False
        
        # Install all dependencies
        success, output = self.run_command("poetry install")
        if success:
            print("? Dependencies installed successfully")
            
            # Install optional groups
            optional_groups = ["ai", "web", "docs", "monitoring"]
            for group in optional_groups:
                print(f"   ?? Installing {group} group...")
                group_success, group_output = self.run_command(f"poetry install --extras {group}")
                if group_success:
                    print(f"   ? {group} group installed")
                else:
                    print(f"   ?? {group} group failed: {group_output}")
            
            return True
        else:
            print(f"? Failed to install dependencies: {output}")
            return False
    
    def migrate_from_requirements(self) -> bool:
        """Migrate from requirements.txt to Poetry."""
        print("?? Migrating from requirements.txt to Poetry...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("? requirements.txt not found")
            return False
        
        # Parse requirements.txt
        try:
            with open(requirements_file, 'r') as f:
                lines = f.readlines()
            
            dependencies = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    dependencies.append(line)
            
            print(f"   ?? Found {len(dependencies)} dependencies in requirements.txt")
            
            # Add each dependency to Poetry
            for dep in dependencies:
                if ';' in dep:  # Skip platform-specific dependencies for now
                    print(f"   ?? Skipping platform-specific: {dep}")
                    continue
                
                # Clean up the dependency name
                dep_clean = dep.split('>=')[0].split('==')[0].split('<')[0].split('>')[0]
                print(f"   ?? Adding {dep_clean}...")
                
                success, output = self.run_command(f"poetry add {dep}")
                if not success:
                    print(f"   ?? Failed to add {dep_clean}: {output}")
            
            print("? Migration from requirements.txt completed")
            return True
            
        except Exception as e:
            print(f"? Failed to migrate from requirements.txt: {e}")
            return False
    
    def show_dependency_info(self) -> None:
        """Show current dependency information."""
        print("?? Current Dependency Information")
        print("-" * 50)
        
        # Show dependency tree
        success, output = self.run_command("poetry show --tree")
        if success:
            print("?? Dependency Tree:")
            print(output[:2000] + "..." if len(output) > 2000 else output)
        
        # Show outdated packages
        print("\n?? Outdated Packages:")
        success, output = self.run_command("poetry show --outdated")
        if success:
            print(output[:1000] + "..." if len(output) > 1000 else output)
        else:
            print("All packages are up to date!")
    
    def export_requirements(self) -> bool:
        """Export Poetry dependencies to requirements.txt format."""
        print("?? Exporting Poetry dependencies to requirements format...")
        
        # Export main dependencies
        success, output = self.run_command("poetry export -f requirements.txt --output requirements_poetry.txt")
        if success:
            print("? Main dependencies exported to requirements_poetry.txt")
        
        # Export dev dependencies
        success, output = self.run_command("poetry export -f requirements.txt --dev --output requirements_poetry_dev.txt")
        if success:
            print("? Dev dependencies exported to requirements_poetry_dev.txt")
        
        return True
    
    def setup_virtual_environment(self) -> bool:
        """Setup and configure Poetry virtual environment."""
        print("?? Setting up Poetry virtual environment...")
        
        # Configure Poetry to create venv in project directory
        self.run_command("poetry config virtualenvs.in-project true")
        
        # Create virtual environment
        success, output = self.run_command("poetry env use python")
        if success:
            print("? Virtual environment created")
            
            # Show environment info
            success, output = self.run_command("poetry env info")
            if success:
                print("?? Environment Info:")
                print(output)
            
            return True
        else:
            print(f"? Failed to create virtual environment: {output}")
            return False
    
    def run_quality_checks(self) -> bool:
        """Run code quality checks using Poetry."""
        print("?? Running code quality checks...")
        
        checks = [
            ("poetry run black --check src tests", "Black formatting"),
            ("poetry run ruff check src tests", "Ruff linting"),
            ("poetry run mypy src", "MyPy type checking"),
            ("poetry run bandit -r src", "Bandit security scan"),
            ("poetry run safety check", "Safety vulnerability scan"),
        ]
        
        all_passed = True
        for command, description in checks:
            print(f"   ?? {description}...")
            success, output = self.run_command(command)
            if success:
                print(f"   ? {description} passed")
            else:
                print(f"   ? {description} failed")
                all_passed = False
        
        return all_passed
    
    def generate_poetry_migration_report(self) -> None:
        """Generate comprehensive migration report."""
        print("\n?? POETRY MIGRATION REPORT")
        print("=" * 60)
        
        # Check Poetry installation
        poetry_installed = self.check_poetry_installed()
        
        # Check if pyproject.toml exists
        has_pyproject = self.current_pyproject.exists()
        
        # Check if poetry.lock exists
        has_lock = (self.project_root / "poetry.lock").exists()
        
        # Check if virtual environment exists
        has_venv = (self.project_root / ".venv").exists()
        
        print(f"\n?? Migration Status:")
        print(f"   Poetry Installed: {'?' if poetry_installed else '?'}")
        print(f"   pyproject.toml: {'?' if has_pyproject else '?'}")
        print(f"   poetry.lock: {'?' if has_lock else '?'}")
        print(f"   Virtual Environment: {'?' if has_venv else '?'}")
        
        print(f"\n?? Recommended Actions:")
        if not poetry_installed:
            print("   1. Install Poetry: poetry_manager.py --install")
        if not has_pyproject:
            print("   2. Initialize project: poetry_manager.py --init")
        if not has_lock:
            print("   3. Lock dependencies: poetry lock")
        if not has_venv:
            print("   4. Create environment: poetry_manager.py --setup-env")
        
        print(f"\n?? Poetry Benefits:")
        print("   - Deterministic dependency resolution")
        print("   - Separate virtual environment management")
        print("   - Dependency groups for different environments")
        print("   - Built-in security scanning")
        print("   - Simplified packaging and publishing")
        print("   - Better dependency conflict resolution")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Poetry dependency management for Image Processing App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python poetry_manager.py --install          # Install Poetry
  python poetry_manager.py --init             # Initialize Poetry project
  python poetry_manager.py --migrate          # Migrate from requirements.txt
  python poetry_manager.py --setup-env        # Setup virtual environment
  python poetry_manager.py --install-deps     # Install all dependencies
  python poetry_manager.py --check            # Run quality checks
  python poetry_manager.py --export           # Export to requirements.txt
  python poetry_manager.py --report           # Show migration status
"""
    )
    
    parser.add_argument("--install", action="store_true", help="Install Poetry")
    parser.add_argument("--init", action="store_true", help="Initialize Poetry project")
    parser.add_argument("--migrate", action="store_true", help="Migrate from requirements.txt")
    parser.add_argument("--setup-env", action="store_true", help="Setup virtual environment")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--check", action="store_true", help="Run quality checks")
    parser.add_argument("--export", action="store_true", help="Export to requirements.txt")
    parser.add_argument("--info", action="store_true", help="Show dependency info")
    parser.add_argument("--report", action="store_true", help="Generate migration report")
    parser.add_argument("--full-setup", action="store_true", help="Complete Poetry setup")
    
    args = parser.parse_args()
    
    manager = PoetryManager()
    
    print("?? POETRY DEPENDENCY MANAGER")
    print("Project ID: Image Processing App 20251119")
    print("=" * 60)
    
    if args.install:
        manager.install_poetry()
    elif args.init:
        manager.init_poetry_project()
    elif args.migrate:
        manager.migrate_from_requirements()
    elif args.setup_env:
        manager.setup_virtual_environment()
    elif args.install_deps:
        manager.install_dependencies()
    elif args.check:
        manager.run_quality_checks()
    elif args.export:
        manager.export_requirements()
    elif args.info:
        manager.show_dependency_info()
    elif args.report:
        manager.generate_poetry_migration_report()
    elif args.full_setup:
        print("?? Running complete Poetry setup...")
        steps = [
            (manager.check_poetry_installed, "Check Poetry installation"),
            (manager.backup_current_config, "Backup current configuration"),
            (manager.init_poetry_project, "Initialize Poetry project"),
            (manager.setup_virtual_environment, "Setup virtual environment"),
            (manager.install_dependencies, "Install dependencies"),
        ]
        
        for step_func, step_name in steps:
            print(f"\n?? {step_name}...")
            success = step_func()
            if not success and step_name == "Check Poetry installation":
                if manager.install_poetry():
                    success = True
            
            if not success:
                print(f"? {step_name} failed. Stopping setup.")
                break
        
        if success:
            print("\n? Complete Poetry setup finished successfully!")
            manager.generate_poetry_migration_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()