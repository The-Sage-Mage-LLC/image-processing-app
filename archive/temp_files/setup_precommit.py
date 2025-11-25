#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-commit Setup and Management Script
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Script to set up and manage pre-commit hooks for code quality.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, cwd: Path = None) -> tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"


def check_pre_commit_installed() -> bool:
    """Check if pre-commit is installed."""
    success, output = run_command("pre-commit --version")
    if success:
        print(f"OK Pre-commit is installed: {output.strip()}")
        return True
    else:
        print("ERROR Pre-commit is not installed")
        return False


def install_pre_commit():
    """Install pre-commit."""
    print("INFO Installing pre-commit...")
    success, output = run_command("pip install pre-commit")
    
    if success:
        print("OK Pre-commit installed successfully")
        return True
    else:
        print(f"ERROR Failed to install pre-commit: {output}")
        return False


def install_hooks():
    """Install pre-commit hooks."""
    print("INFO Installing pre-commit hooks...")
    
    project_root = Path(__file__).parent
    success, output = run_command("pre-commit install", cwd=project_root)
    
    if success:
        print("OK Pre-commit hooks installed successfully")
        print("   Hooks will now run automatically on git commit")
        return True
    else:
        print(f"ERROR Failed to install hooks: {output}")
        return False


def install_commit_msg_hook():
    """Install commit message hook for conventional commits."""
    print("INFO Installing commit message hook...")
    
    project_root = Path(__file__).parent
    success, output = run_command("pre-commit install --hook-type commit-msg", cwd=project_root)
    
    if success:
        print("OK Commit message hook installed")
        print("   Commit messages will be validated against conventional commit format")
        return True
    else:
        print(f"ERROR Failed to install commit message hook: {output}")
        return False


def run_hooks():
    """Run pre-commit hooks on all files."""
    print("INFO Running pre-commit hooks on all files...")
    
    project_root = Path(__file__).parent
    success, output = run_command("pre-commit run --all-files", cwd=project_root)
    
    if success:
        print("OK All pre-commit hooks passed")
    else:
        print("WARNING Some pre-commit hooks failed or made changes:")
        print(output)
        print("\nINFO This is normal for the first run. Files have been formatted.")
        print("   Please review the changes and commit them.")
    
    return success


def create_secrets_baseline():
    """Create baseline for secrets detection."""
    print("INFO Creating secrets detection baseline...")
    
    project_root = Path(__file__).parent
    success, output = run_command("detect-secrets scan --baseline .secrets.baseline", cwd=project_root)
    
    if success:
        print("OK Secrets baseline created")
    else:
        print("WARNING Detect-secrets not available, skipping baseline creation")
        # Create empty baseline
        baseline_file = project_root / ".secrets.baseline"
        baseline_file.write_text('{\n  "version": "1.4.0",\n  "plugins_used": [],\n  "results": {}\n}\n')
        print("OK Empty secrets baseline created")


def show_status():
    """Show pre-commit installation status."""
    print("INFO Pre-commit Status Check")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    
    # Check if pre-commit is installed
    if not check_pre_commit_installed():
        print("ERROR Pre-commit not installed")
        return False
    
    # Check if hooks are installed
    git_hooks_dir = project_root / ".git" / "hooks"
    pre_commit_hook = git_hooks_dir / "pre-commit"
    
    if pre_commit_hook.exists():
        print("OK Pre-commit hooks are installed")
    else:
        print("ERROR Pre-commit hooks not installed")
        return False
    
    # Check hook configuration
    config_file = project_root / ".pre-commit-config.yaml"
    if config_file.exists():
        print("OK Pre-commit configuration found")
    else:
        print("ERROR Pre-commit configuration missing")
        return False
    
    # Show configured hooks
    success, output = run_command("pre-commit --version && pre-commit validate-config", cwd=project_root)
    if success:
        print("OK Configuration is valid")
    else:
        print(f"ERROR Configuration validation failed: {output}")
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pre-commit setup and management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_precommit.py --setup      # Complete setup
  python setup_precommit.py --run        # Run hooks on all files
  python setup_precommit.py --update     # Update hook versions
  python setup_precommit.py --status     # Show installation status
"""
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Complete pre-commit setup (install + configure)"
    )
    
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run pre-commit hooks on all files"
    )
    
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update pre-commit hooks to latest versions"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pre-commit installation status"
    )
    
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Only install pre-commit (don't configure)"
    )
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
        return
    
    if args.update:
        project_root = Path(__file__).parent
        success, output = run_command("pre-commit autoupdate", cwd=project_root)
        if success:
            print("OK Pre-commit hooks updated")
            print(output)
        else:
            print(f"ERROR Failed to update hooks: {output}")
        return
    
    if args.run:
        run_hooks()
        return
    
    if args.install_only:
        if not check_pre_commit_installed():
            install_pre_commit()
        return
    
    if args.setup or len(sys.argv) == 1:
        print("INFO Setting up pre-commit hooks for Image Processing App")
        print("=" * 60)
        
        # Step 1: Install pre-commit
        if not check_pre_commit_installed():
            if not install_pre_commit():
                print("ERROR Setup failed: Could not install pre-commit")
                sys.exit(1)
        
        # Step 2: Create secrets baseline
        create_secrets_baseline()
        
        # Step 3: Install hooks
        if not install_hooks():
            print("ERROR Setup failed: Could not install hooks")
            sys.exit(1)
        
        # Step 4: Install commit message hook
        if not install_commit_msg_hook():
            print("WARNING Could not install commit message hook")
        
        # Step 5: Run hooks once to format existing code
        print("\nINFO Running initial code formatting...")
        run_hooks()
        
        print("\n" + "=" * 60)
        print("OK PRE-COMMIT SETUP COMPLETE!")
        print("\nINFO What happens now:")
        print("  * Code will be automatically formatted on commit")
        print("  * Linting will run before each commit")
        print("  * Type checking will validate your code")
        print("  * Security checks will scan for vulnerabilities")
        print("  * Commit messages will be validated")
        
        print("\nINFO Useful commands:")
        print("  git commit -m 'feat: add new feature'     # Normal commit")
        print("  git commit --no-verify                   # Skip hooks")
        print("  pre-commit run --all-files               # Run hooks manually")
        print("  pre-commit autoupdate                    # Update hook versions")
        
        print("\nINFO Conventional Commit Format:")
        print("  feat: add new feature")
        print("  fix: resolve bug")
        print("  docs: update documentation")
        print("  style: formatting changes")
        print("  refactor: code refactoring")
        print("  test: add tests")
        print("  chore: maintenance tasks")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()