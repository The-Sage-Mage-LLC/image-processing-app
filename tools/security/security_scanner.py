#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Scanning and Vulnerability Management
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive security scanning tool for dependencies, code, and containers.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import argparse


class SecurityScanner:
    """Comprehensive security scanner for the application."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.reports_dir = self.project_root / "security_reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.scan_results = {}
        
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
    
    def scan_dependencies_safety(self) -> Dict[str, Any]:
        """Scan dependencies for known vulnerabilities using Safety."""
        print("?? Scanning dependencies with Safety...")
        
        # Install Safety if not available
        self.run_command("pip install safety")
        
        # Run Safety scan
        success, output = self.run_command("safety check --json")
        
        report_file = self.reports_dir / f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if success:
            try:
                vulnerabilities = json.loads(output) if output.strip() else []
                result = {
                    "tool": "safety",
                    "status": "passed" if not vulnerabilities else "failed",
                    "vulnerabilities": vulnerabilities,
                    "total_vulnerabilities": len(vulnerabilities),
                    "report_file": str(report_file)
                }
                
                with open(report_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
                print(f"? Safety scan complete: {len(vulnerabilities)} vulnerabilities found")
                return result
                
            except json.JSONDecodeError:
                return {"tool": "safety", "status": "error", "error": "Invalid JSON output"}
        else:
            print(f"? Safety scan failed: {output}")
            return {"tool": "safety", "status": "error", "error": output}
    
    def scan_dependencies_pip_audit(self) -> Dict[str, Any]:
        """Scan dependencies using pip-audit."""
        print("?? Scanning dependencies with pip-audit...")
        
        # Install pip-audit if not available
        self.run_command("pip install pip-audit")
        
        # Run pip-audit scan
        success, output = self.run_command("pip-audit --format=json")
        
        report_file = self.reports_dir / f"pip_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if success:
            try:
                if output.strip():
                    audit_results = json.loads(output)
                    vulnerabilities = audit_results.get("vulnerabilities", [])
                else:
                    vulnerabilities = []
                
                result = {
                    "tool": "pip-audit",
                    "status": "passed" if not vulnerabilities else "failed",
                    "vulnerabilities": vulnerabilities,
                    "total_vulnerabilities": len(vulnerabilities),
                    "report_file": str(report_file)
                }
                
                with open(report_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
                print(f"? pip-audit scan complete: {len(vulnerabilities)} vulnerabilities found")
                return result
                
            except json.JSONDecodeError:
                return {"tool": "pip-audit", "status": "error", "error": "Invalid JSON output"}
        else:
            print(f"? pip-audit scan failed: {output}")
            return {"tool": "pip-audit", "status": "error", "error": output}
    
    def scan_code_bandit(self) -> Dict[str, Any]:
        """Scan code for security issues using Bandit."""
        print("?? Scanning code with Bandit...")
        
        # Install Bandit if not available
        self.run_command("pip install bandit")
        
        # Run Bandit scan
        success, output = self.run_command("bandit -r src/ -f json")
        
        report_file = self.reports_dir / f"bandit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            if output.strip():
                bandit_results = json.loads(output)
                issues = bandit_results.get("results", [])
            else:
                issues = []
            
            # Categorize issues by severity
            high_severity = [issue for issue in issues if issue.get("issue_severity") == "HIGH"]
            medium_severity = [issue for issue in issues if issue.get("issue_severity") == "MEDIUM"]
            low_severity = [issue for issue in issues if issue.get("issue_severity") == "LOW"]
            
            result = {
                "tool": "bandit",
                "status": "passed" if not high_severity else "failed",
                "high_severity": len(high_severity),
                "medium_severity": len(medium_severity),
                "low_severity": len(low_severity),
                "total_issues": len(issues),
                "issues": issues,
                "report_file": str(report_file)
            }
            
            with open(report_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            print(f"? Bandit scan complete: {len(issues)} issues found")
            print(f"   High: {len(high_severity)}, Medium: {len(medium_severity)}, Low: {len(low_severity)}")
            return result
            
        except json.JSONDecodeError:
            return {"tool": "bandit", "status": "error", "error": "Invalid JSON output"}
    
    def scan_secrets(self) -> Dict[str, Any]:
        """Scan for exposed secrets using detect-secrets."""
        print("?? Scanning for secrets...")
        
        # Install detect-secrets if not available
        self.run_command("pip install detect-secrets")
        
        # Initialize baseline if it doesn't exist
        baseline_file = self.project_root / ".secrets.baseline"
        if not baseline_file.exists():
            self.run_command("detect-secrets scan --baseline .secrets.baseline")
        
        # Run secrets scan
        success, output = self.run_command("detect-secrets scan --baseline .secrets.baseline")
        
        # Audit the baseline
        audit_success, audit_output = self.run_command("detect-secrets audit .secrets.baseline --report")
        
        report_file = self.reports_dir / f"secrets_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("Secrets Scan Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("Scan Output:\n")
            f.write(output)
            f.write("\n\nAudit Output:\n")
            f.write(audit_output)
        
        # Count potential secrets
        secret_count = output.count("Potential secrets") if "Potential secrets" in output else 0
        
        result = {
            "tool": "detect-secrets",
            "status": "passed" if secret_count == 0 else "warning",
            "potential_secrets": secret_count,
            "report_file": str(report_file)
        }
        
        print(f"? Secrets scan complete: {secret_count} potential secrets found")
        return result
    
    def scan_licenses(self) -> Dict[str, Any]:
        """Scan dependency licenses for compliance."""
        print("?? Scanning dependency licenses...")
        
        # Install license checker
        self.run_command("pip install pip-licenses")
        
        # Get license information
        success, output = self.run_command("pip-licenses --format=json")
        
        report_file = self.reports_dir / f"licenses_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if success and output.strip():
            try:
                licenses = json.loads(output)
                
                # Categorize licenses
                permissive = ["MIT", "BSD", "Apache", "ISC", "Unlicense"]
                copyleft = ["GPL", "LGPL", "AGPL", "MPL"]
                unknown = []
                
                license_categories = {
                    "permissive": [],
                    "copyleft": [],
                    "unknown": [],
                    "other": []
                }
                
                for package in licenses:
                    license_name = package.get("License", "Unknown")
                    if any(p in license_name.upper() for p in permissive):
                        license_categories["permissive"].append(package)
                    elif any(c in license_name.upper() for c in copyleft):
                        license_categories["copyleft"].append(package)
                    elif license_name in ["Unknown", "UNKNOWN", ""]:
                        license_categories["unknown"].append(package)
                    else:
                        license_categories["other"].append(package)
                
                result = {
                    "tool": "pip-licenses",
                    "status": "passed" if not license_categories["copyleft"] else "warning",
                    "total_packages": len(licenses),
                    "categories": {k: len(v) for k, v in license_categories.items()},
                    "licenses": license_categories,
                    "report_file": str(report_file)
                }
                
                with open(report_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"? License scan complete: {len(licenses)} packages analyzed")
                if license_categories["copyleft"]:
                    print(f"??  Warning: {len(license_categories['copyleft'])} copyleft licenses found")
                
                return result
                
            except json.JSONDecodeError:
                return {"tool": "pip-licenses", "status": "error", "error": "Invalid JSON output"}
        else:
            return {"tool": "pip-licenses", "status": "error", "error": "No output or command failed"}
    
    def generate_security_report(self) -> None:
        """Generate comprehensive security report."""
        print("\n?? Generating comprehensive security report...")
        
        # Run all scans
        results = {
            "timestamp": datetime.now().isoformat(),
            "scans": {
                "dependencies_safety": self.scan_dependencies_safety(),
                "dependencies_pip_audit": self.scan_dependencies_pip_audit(),
                "code_bandit": self.scan_code_bandit(),
                "secrets": self.scan_secrets(),
                "licenses": self.scan_licenses()
            }
        }
        
        # Calculate overall security score
        security_score = self.calculate_security_score(results["scans"])
        results["security_score"] = security_score
        
        # Save comprehensive report
        report_file = self.reports_dir / f"comprehensive_security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate markdown report
        self.generate_markdown_report(results, report_file)
        
        print(f"\n?? Security Report Summary:")
        print(f"   Overall Security Score: {security_score}/100")
        print(f"   Report saved to: {report_file}")
        
        return results
    
    def calculate_security_score(self, scans: Dict[str, Any]) -> int:
        """Calculate overall security score based on scan results."""
        score = 100
        
        # Dependency vulnerabilities
        safety_vulns = scans.get("dependencies_safety", {}).get("total_vulnerabilities", 0)
        audit_vulns = scans.get("dependencies_pip_audit", {}).get("total_vulnerabilities", 0)
        score -= min(safety_vulns + audit_vulns, 30)  # Max 30 point deduction
        
        # Code security issues
        bandit_high = scans.get("code_bandit", {}).get("high_severity", 0)
        bandit_medium = scans.get("code_bandit", {}).get("medium_severity", 0)
        score -= (bandit_high * 10 + bandit_medium * 5)  # 10 points per high, 5 per medium
        
        # Secrets
        secrets_count = scans.get("secrets", {}).get("potential_secrets", 0)
        score -= min(secrets_count * 5, 20)  # Max 20 point deduction
        
        # License issues
        copyleft_count = scans.get("licenses", {}).get("categories", {}).get("copyleft", 0)
        if copyleft_count > 0:
            score -= 10  # 10 points for license compatibility issues
        
        return max(score, 0)
    
    def generate_markdown_report(self, results: Dict[str, Any], json_report_path: Path) -> None:
        """Generate a markdown security report."""
        markdown_file = json_report_path.with_suffix('.md')
        
        with open(markdown_file, 'w') as f:
            f.write("# Security Scan Report\n\n")
            f.write(f"**Generated:** {results['timestamp']}\n")
            f.write(f"**Security Score:** {results['security_score']}/100\n\n")
            
            scans = results['scans']
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            total_vulns = (scans.get('dependencies_safety', {}).get('total_vulnerabilities', 0) + 
                          scans.get('dependencies_pip_audit', {}).get('total_vulnerabilities', 0))
            high_issues = scans.get('code_bandit', {}).get('high_severity', 0)
            secrets = scans.get('secrets', {}).get('potential_secrets', 0)
            
            if total_vulns == 0 and high_issues == 0 and secrets == 0:
                f.write("?? **EXCELLENT** - No critical security issues detected.\n\n")
            elif total_vulns <= 5 and high_issues == 0 and secrets == 0:
                f.write("?? **GOOD** - Minor vulnerabilities detected, review recommended.\n\n")
            else:
                f.write("?? **ACTION REQUIRED** - Critical security issues detected.\n\n")
            
            # Detailed Results
            f.write("## Scan Results\n\n")
            
            # Dependencies
            f.write("### Dependency Vulnerabilities\n\n")
            f.write(f"- **Safety:** {scans.get('dependencies_safety', {}).get('total_vulnerabilities', 0)} vulnerabilities\n")
            f.write(f"- **pip-audit:** {scans.get('dependencies_pip_audit', {}).get('total_vulnerabilities', 0)} vulnerabilities\n\n")
            
            # Code Security
            f.write("### Code Security (Bandit)\n\n")
            bandit = scans.get('code_bandit', {})
            f.write(f"- **High Severity:** {bandit.get('high_severity', 0)} issues\n")
            f.write(f"- **Medium Severity:** {bandit.get('medium_severity', 0)} issues\n")
            f.write(f"- **Low Severity:** {bandit.get('low_severity', 0)} issues\n\n")
            
            # Secrets
            f.write("### Secrets Detection\n\n")
            f.write(f"- **Potential Secrets:** {scans.get('secrets', {}).get('potential_secrets', 0)}\n\n")
            
            # Licenses
            f.write("### License Compliance\n\n")
            licenses = scans.get('licenses', {}).get('categories', {})
            f.write(f"- **Permissive:** {licenses.get('permissive', 0)} packages\n")
            f.write(f"- **Copyleft:** {licenses.get('copyleft', 0)} packages\n")
            f.write(f"- **Unknown:** {licenses.get('unknown', 0)} packages\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if total_vulns > 0:
                f.write("1. **Update vulnerable dependencies** immediately\n")
            if high_issues > 0:
                f.write("2. **Fix high severity code issues** before deployment\n")
            if secrets > 0:
                f.write("3. **Review and remove exposed secrets**\n")
            if licenses.get('copyleft', 0) > 0:
                f.write("4. **Review copyleft licenses** for compliance\n")
            
            f.write(f"\n**Full Report:** `{json_report_path.name}`\n")
        
        print(f"?? Markdown report saved to: {markdown_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive security scanning for Image Processing App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python security_scanner.py --all              # Run all scans
  python security_scanner.py --dependencies     # Scan only dependencies
  python security_scanner.py --code             # Scan only code
  python security_scanner.py --secrets          # Scan only secrets
  python security_scanner.py --licenses         # Check only licenses
"""
    )
    
    parser.add_argument("--all", action="store_true", help="Run all security scans")
    parser.add_argument("--dependencies", action="store_true", help="Scan dependencies only")
    parser.add_argument("--code", action="store_true", help="Scan code only")
    parser.add_argument("--secrets", action="store_true", help="Scan secrets only")
    parser.add_argument("--licenses", action="store_true", help="Check licenses only")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing scans")
    
    args = parser.parse_args()
    
    scanner = SecurityScanner()
    
    print("???  COMPREHENSIVE SECURITY SCANNER")
    print("Project ID: Image Processing App 20251119")
    print("=" * 60)
    
    if args.all or len(sys.argv) == 1:
        scanner.generate_security_report()
    elif args.dependencies:
        scanner.scan_dependencies_safety()
        scanner.scan_dependencies_pip_audit()
    elif args.code:
        scanner.scan_code_bandit()
    elif args.secrets:
        scanner.scan_secrets()
    elif args.licenses:
        scanner.scan_licenses()
    elif args.report_only:
        # Generate report from existing scan files
        print("?? Generating report from existing scans...")
        scanner.generate_security_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()