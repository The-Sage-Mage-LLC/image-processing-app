#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Enterprise Infrastructure Setup
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Complete setup for advanced enterprise features:
- Security scanning automation
- Poetry dependency management 
- Comprehensive test coverage
- GUI integration testing
- Performance benchmarking
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import argparse


class EnterpriseSetupManager:
    """Advanced enterprise infrastructure setup manager."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.setup_results = {
            "timestamp": datetime.now().isoformat(),
            "completed_steps": [],
            "failed_steps": [],
            "warnings": []
        }
    
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
    
    def print_step(self, step: str, status: str = "INFO"):
        """Print formatted step message."""
        symbols = {
            "INFO": "??",
            "SUCCESS": "?", 
            "WARNING": "??",
            "ERROR": "?",
            "SKIP": "??"
        }
        
        symbol = symbols.get(status, "??")
        print(f"{symbol} {step}")
        
        # Track in results
        if status == "SUCCESS":
            self.setup_results["completed_steps"].append(step)
        elif status == "ERROR":
            self.setup_results["failed_steps"].append(step)
        elif status == "WARNING":
            self.setup_results["warnings"].append(step)
    
    def setup_security_scanning(self) -> bool:
        """Setup automated security scanning infrastructure."""
        self.print_step("Setting up automated security scanning...")
        
        try:
            # Install security tools
            security_tools = [
                "safety",
                "bandit", 
                "pip-audit",
                "semgrep",
                "detect-secrets"
            ]
            
            for tool in security_tools:
                self.print_step(f"Installing {tool}...")
                success, output = self.run_command(f"pip install {tool}")
                if not success:
                    self.print_step(f"Failed to install {tool}: {output}", "WARNING")
                else:
                    self.print_step(f"Installed {tool}", "SUCCESS")
            
            # Setup secrets baseline
            self.print_step("Creating secrets detection baseline...")
            success, output = self.run_command("detect-secrets scan --baseline .secrets.baseline")
            if not success:
                # Create empty baseline
                baseline_content = {
                    "version": "1.4.0",
                    "plugins_used": [],
                    "results": {}
                }
                baseline_file = self.project_root / ".secrets.baseline"
                with open(baseline_file, 'w') as f:
                    json.dump(baseline_content, f, indent=2)
                self.print_step("Created empty secrets baseline", "SUCCESS")
            
            # Test security scanner
            scanner_script = self.project_root / "security_scanner.py"
            if scanner_script.exists():
                self.print_step("Testing security scanner...")
                success, output = self.run_command("python security_scanner.py --licenses")
                if success:
                    self.print_step("Security scanner working correctly", "SUCCESS")
                else:
                    self.print_step("Security scanner test failed", "WARNING")
            
            return True
            
        except Exception as e:
            self.print_step(f"Security scanning setup failed: {e}", "ERROR")
            return False
    
    def setup_poetry_dependency_management(self) -> bool:
        """Setup Poetry for modern dependency management."""
        self.print_step("Setting up Poetry dependency management...")
        
        try:
            # Check if Poetry is installed
            success, output = self.run_command("poetry --version")
            if not success:
                self.print_step("Installing Poetry...")
                
                # Try different installation methods
                install_methods = [
                    "pip install poetry",
                    "pipx install poetry"
                ]
                
                poetry_installed = False
                for method in install_methods:
                    success, output = self.run_command(method)
                    if success:
                        poetry_installed = True
                        self.print_step(f"Poetry installed via: {method}", "SUCCESS")
                        break
                
                if not poetry_installed:
                    self.print_step("Poetry installation failed. Please install manually.", "WARNING")
                    return False
            else:
                self.print_step("Poetry is already installed", "SUCCESS")
            
            # Run Poetry manager
            poetry_manager = self.project_root / "poetry_manager.py"
            if poetry_manager.exists():
                self.print_step("Running Poetry setup...")
                success, output = self.run_command("python poetry_manager.py --report")
                if success:
                    self.print_step("Poetry configuration verified", "SUCCESS")
                else:
                    self.print_step("Poetry configuration needs attention", "WARNING")
            
            return True
            
        except Exception as e:
            self.print_step(f"Poetry setup failed: {e}", "ERROR")
            return False
    
    def setup_comprehensive_testing(self) -> bool:
        """Setup comprehensive test infrastructure."""
        self.print_step("Setting up comprehensive test infrastructure...")
        
        try:
            # Install testing dependencies
            test_tools = [
                "pytest",
                "pytest-cov",
                "pytest-asyncio",
                "pytest-mock",
                "pytest-xvfb",
                "pytest-benchmark",
                "pytest-html",
                "pytest-sugar",
                "coverage"
            ]
            
            for tool in test_tools:
                success, output = self.run_command(f"pip install {tool}")
                if not success:
                    self.print_step(f"Failed to install {tool}", "WARNING")
                else:
                    self.print_step(f"Installed {tool}", "SUCCESS")
            
            # Create test structure using test suite manager
            test_manager = self.project_root / "test_suite_manager.py"
            if test_manager.exists():
                self.print_step("Creating test templates...")
                success, output = self.run_command("python test_suite_manager.py --create-templates")
                if success:
                    self.print_step("Test templates created", "SUCCESS")
                else:
                    self.print_step("Test template creation failed", "WARNING")
            
            # Validate test structure
            required_test_dirs = [
                "tests/unit",
                "tests/integration",
                "tests/gui", 
                "tests/performance",
                "tests/security",
                "tests/fixtures"
            ]
            
            for test_dir in required_test_dirs:
                dir_path = self.project_root / test_dir
                if dir_path.exists():
                    self.print_step(f"Test directory exists: {test_dir}", "SUCCESS")
                else:
                    self.print_step(f"Missing test directory: {test_dir}", "WARNING")
            
            return True
            
        except Exception as e:
            self.print_step(f"Test infrastructure setup failed: {e}", "ERROR")
            return False
    
    def setup_gui_integration_testing(self) -> bool:
        """Setup GUI integration testing."""
        self.print_step("Setting up GUI integration testing...")
        
        try:
            # Install GUI testing dependencies
            gui_test_tools = [
                "PyQt6",
                "pytest-qt"
            ]
            
            for tool in gui_test_tools:
                success, output = self.run_command(f"pip install {tool}")
                if not success:
                    self.print_step(f"Failed to install {tool}", "WARNING")
                else:
                    self.print_step(f"Installed {tool}", "SUCCESS")
            
            # Setup headless display for CI
            if sys.platform.startswith('linux'):
                self.print_step("Setting up headless display for Linux...")
                os.environ["QT_QPA_PLATFORM"] = "offscreen"
                self.print_step("Configured QT for headless testing", "SUCCESS")
            
            # Validate GUI test file
            gui_test_file = self.project_root / "tests" / "integration" / "test_gui_integration.py"
            if gui_test_file.exists():
                self.print_step("GUI integration tests found", "SUCCESS")
                
                # Test GUI test runner
                self.print_step("Testing GUI test infrastructure...")
                success, output = self.run_command("python -m pytest tests/integration/test_gui_integration.py::TestGUIIntegration::test_main_window_initialization -v")
                if success:
                    self.print_step("GUI testing infrastructure works", "SUCCESS")
                else:
                    self.print_step("GUI test infrastructure needs attention", "WARNING")
            else:
                self.print_step("GUI integration tests not found", "WARNING")
            
            return True
            
        except Exception as e:
            self.print_step(f"GUI testing setup failed: {e}", "ERROR")
            return False
    
    def setup_performance_benchmarking(self) -> bool:
        """Setup performance benchmarking infrastructure."""
        self.print_step("Setting up performance benchmarking...")
        
        try:
            # Install benchmarking dependencies
            benchmark_tools = [
                "pytest-benchmark",
                "psutil", 
                "memory-profiler",
                "py-spy",
                "line-profiler"
            ]
            
            for tool in benchmark_tools:
                success, output = self.run_command(f"pip install {tool}")
                if not success:
                    self.print_step(f"Failed to install {tool}", "WARNING")
                else:
                    self.print_step(f"Installed {tool}", "SUCCESS")
            
            # Test performance benchmarking
            benchmark_script = self.project_root / "performance_benchmarks.py"
            if benchmark_script.exists():
                self.print_step("Testing performance benchmarking...")
                success, output = self.run_command("python performance_benchmarks.py --run")
                if success:
                    self.print_step("Performance benchmarking working", "SUCCESS")
                    
                    # Check if baseline was created
                    baseline_file = self.project_root / "benchmark_reports" / "performance_baseline.json"
                    if baseline_file.exists():
                        self.print_step("Performance baseline established", "SUCCESS")
                else:
                    self.print_step("Performance benchmarking test failed", "WARNING")
            
            return True
            
        except Exception as e:
            self.print_step(f"Performance benchmarking setup failed: {e}", "ERROR")
            return False
    
    def setup_ci_cd_integration(self) -> bool:
        """Setup CI/CD pipeline integration."""
        self.print_step("Setting up CI/CD pipeline integration...")
        
        try:
            # Validate GitHub Actions workflows
            workflows_dir = self.project_root / ".github" / "workflows"
            if not workflows_dir.exists():
                self.print_step("GitHub Actions directory not found", "WARNING")
                return False
            
            required_workflows = [
                "ci-cd.yml",
                "security.yml"
            ]
            
            for workflow in required_workflows:
                workflow_file = workflows_dir / workflow
                if workflow_file.exists():
                    self.print_step(f"Found workflow: {workflow}", "SUCCESS")
                else:
                    self.print_step(f"Missing workflow: {workflow}", "WARNING")
            
            # Validate pre-commit configuration
            precommit_config = self.project_root / ".pre-commit-config.yaml"
            if precommit_config.exists():
                self.print_step("Pre-commit configuration found", "SUCCESS")
                
                # Test pre-commit installation
                success, output = self.run_command("pre-commit --version")
                if success:
                    self.print_step("Pre-commit is available", "SUCCESS")
                else:
                    self.print_step("Installing pre-commit...")
                    success, output = self.run_command("pip install pre-commit")
                    if success:
                        self.print_step("Pre-commit installed", "SUCCESS")
            else:
                self.print_step("Pre-commit configuration not found", "WARNING")
            
            return True
            
        except Exception as e:
            self.print_step(f"CI/CD integration setup failed: {e}", "ERROR")
            return False
    
    def validate_enterprise_setup(self) -> Dict[str, Any]:
        """Validate complete enterprise setup."""
        self.print_step("Validating enterprise setup...")
        
        validation_results = {
            "security_scanning": False,
            "dependency_management": False,
            "testing_infrastructure": False,
            "gui_testing": False,
            "performance_benchmarking": False,
            "ci_cd_integration": False
        }
        
        try:
            # Check security scanner
            security_script = self.project_root / "security_scanner.py"
            if security_script.exists():
                validation_results["security_scanning"] = True
                self.print_step("Security scanning: Available", "SUCCESS")
            else:
                self.print_step("Security scanning: Missing", "WARNING")
            
            # Check Poetry configuration
            poetry_config = self.project_root / "poetry_config.toml"
            if poetry_config.exists():
                validation_results["dependency_management"] = True
                self.print_step("Dependency management: Configured", "SUCCESS")
            else:
                self.print_step("Dependency management: Not configured", "WARNING")
            
            # Check test infrastructure
            test_manager = self.project_root / "test_suite_manager.py"
            tests_dir = self.project_root / "tests"
            if test_manager.exists() and tests_dir.exists():
                validation_results["testing_infrastructure"] = True
                self.print_step("Testing infrastructure: Available", "SUCCESS")
            else:
                self.print_step("Testing infrastructure: Incomplete", "WARNING")
            
            # Check GUI testing
            gui_tests = self.project_root / "tests" / "integration" / "test_gui_integration.py"
            if gui_tests.exists():
                validation_results["gui_testing"] = True
                self.print_step("GUI testing: Available", "SUCCESS")
            else:
                self.print_step("GUI testing: Missing", "WARNING")
            
            # Check performance benchmarking
            benchmark_script = self.project_root / "performance_benchmarks.py"
            if benchmark_script.exists():
                validation_results["performance_benchmarking"] = True
                self.print_step("Performance benchmarking: Available", "SUCCESS")
            else:
                self.print_step("Performance benchmarking: Missing", "WARNING")
            
            # Check CI/CD
            workflows_dir = self.project_root / ".github" / "workflows"
            if workflows_dir.exists() and list(workflows_dir.glob("*.yml")):
                validation_results["ci_cd_integration"] = True
                self.print_step("CI/CD integration: Configured", "SUCCESS")
            else:
                self.print_step("CI/CD integration: Not configured", "WARNING")
            
            return validation_results
            
        except Exception as e:
            self.print_step(f"Validation failed: {e}", "ERROR")
            return validation_results
    
    def generate_setup_report(self) -> None:
        """Generate comprehensive setup report."""
        self.print_step("Generating setup report...")
        
        validation_results = self.validate_enterprise_setup()
        
        # Calculate setup score
        total_features = len(validation_results)
        enabled_features = sum(validation_results.values())
        setup_score = (enabled_features / total_features) * 100
        
        report = {
            "setup_summary": self.setup_results,
            "validation_results": validation_results,
            "setup_score": setup_score,
            "recommendations": []
        }
        
        # Generate recommendations
        if not validation_results["security_scanning"]:
            report["recommendations"].append("Enable security scanning automation")
        if not validation_results["dependency_management"]:
            report["recommendations"].append("Configure Poetry dependency management")
        if not validation_results["testing_infrastructure"]:
            report["recommendations"].append("Complete test infrastructure setup")
        if not validation_results["gui_testing"]:
            report["recommendations"].append("Implement GUI integration testing")
        if not validation_results["performance_benchmarking"]:
            report["recommendations"].append("Set up performance benchmarking")
        if not validation_results["ci_cd_integration"]:
            report["recommendations"].append("Configure CI/CD pipelines")
        
        # Save report
        reports_dir = self.project_root / "setup_reports"
        reports_dir.mkdir(exist_ok=True)
        report_file = reports_dir / f"enterprise_setup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("?? ENTERPRISE SETUP COMPLETE")
        print("=" * 80)
        
        print(f"\n?? Setup Results:")
        print(f"   ? Completed steps: {len(self.setup_results['completed_steps'])}")
        print(f"   ? Failed steps: {len(self.setup_results['failed_steps'])}")
        print(f"   ??  Warnings: {len(self.setup_results['warnings'])}")
        print(f"   ?? Overall score: {setup_score:.1f}%")
        
        print(f"\n?? Enterprise Features:")
        for feature, enabled in validation_results.items():
            status = "? Enabled" if enabled else "? Disabled"
            print(f"   {status} {feature.replace('_', ' ').title()}")
        
        if report["recommendations"]:
            print(f"\n?? Recommendations:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")
        
        print(f"\n?? Detailed report: {report_file}")
        
        if setup_score >= 90:
            print("\n?? EXCELLENT - Enterprise setup is comprehensive!")
        elif setup_score >= 75:
            print("\n?? GOOD - Enterprise setup is mostly complete")
        elif setup_score >= 50:
            print("\n??  FAIR - Enterprise setup needs improvement")
        else:
            print("\n?? POOR - Significant setup issues need attention")
    
    def run_complete_setup(self) -> None:
        """Run complete enterprise setup."""
        print("?? ADVANCED ENTERPRISE INFRASTRUCTURE SETUP")
        print("Project ID: Image Processing App 20251119")
        print("=" * 80)
        
        setup_steps = [
            ("Security Scanning", self.setup_security_scanning),
            ("Poetry Dependency Management", self.setup_poetry_dependency_management),
            ("Comprehensive Testing", self.setup_comprehensive_testing),
            ("GUI Integration Testing", self.setup_gui_integration_testing),
            ("Performance Benchmarking", self.setup_performance_benchmarking),
            ("CI/CD Integration", self.setup_ci_cd_integration),
        ]
        
        for step_name, step_function in setup_steps:
            print(f"\n?? {step_name}...")
            try:
                success = step_function()
                if success:
                    self.print_step(f"{step_name} completed", "SUCCESS")
                else:
                    self.print_step(f"{step_name} failed", "ERROR")
            except Exception as e:
                self.print_step(f"{step_name} error: {e}", "ERROR")
        
        # Generate final report
        self.generate_setup_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced enterprise infrastructure setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enterprise_setup.py --all           # Complete enterprise setup
  python enterprise_setup.py --security      # Security scanning only
  python enterprise_setup.py --poetry        # Poetry setup only
  python enterprise_setup.py --testing       # Testing infrastructure only
  python enterprise_setup.py --gui-testing   # GUI testing only
  python enterprise_setup.py --benchmarking  # Performance benchmarking only
  python enterprise_setup.py --ci-cd         # CI/CD integration only
  python enterprise_setup.py --validate      # Validate setup only
"""
    )
    
    parser.add_argument("--all", action="store_true", help="Run complete enterprise setup")
    parser.add_argument("--security", action="store_true", help="Setup security scanning")
    parser.add_argument("--poetry", action="store_true", help="Setup Poetry dependency management")
    parser.add_argument("--testing", action="store_true", help="Setup testing infrastructure")
    parser.add_argument("--gui-testing", action="store_true", help="Setup GUI testing")
    parser.add_argument("--benchmarking", action="store_true", help="Setup performance benchmarking")
    parser.add_argument("--ci-cd", action="store_true", help="Setup CI/CD integration")
    parser.add_argument("--validate", action="store_true", help="Validate current setup")
    
    args = parser.parse_args()
    
    manager = EnterpriseSetupManager()
    
    if args.all or len(sys.argv) == 1:
        manager.run_complete_setup()
    elif args.security:
        manager.setup_security_scanning()
    elif args.poetry:
        manager.setup_poetry_dependency_management()
    elif args.testing:
        manager.setup_comprehensive_testing()
    elif args.gui_testing:
        manager.setup_gui_integration_testing()
    elif args.benchmarking:
        manager.setup_performance_benchmarking()
    elif args.ci_cd:
        manager.setup_ci_cd_integration()
    elif args.validate:
        manager.generate_setup_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()