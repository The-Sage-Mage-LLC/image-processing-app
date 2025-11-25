#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Scanning Framework
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive security scanning framework with vulnerability detection,
dependency scanning, and security policy enforcement.

Features:
- Static code analysis for security vulnerabilities
- Dependency vulnerability scanning
- File system security assessment
- Network security validation
- Security policy compliance checking
- Automated security reporting and alerting
"""

import os
import re
import ast
import json
import hashlib
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import pkg_resources
import requests
import logging

# Import our structured logging system
from ..utils.structured_logging import CorrelationLogger, with_correlation


class VulnerabilitySeverity(Enum):
    """Security vulnerability severity levels."""
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityScanType(Enum):
    """Types of security scans."""
    STATIC_CODE = "static_code"
    DEPENDENCY = "dependency"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    CONFIGURATION = "configuration"


@dataclass
class SecurityVulnerability:
    """Individual security vulnerability finding."""
    id: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    scan_type: SecurityScanType
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    remediation: Optional[str] = None
    references: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)
    false_positive: bool = False


@dataclass
class SecurityScanReport:
    """Comprehensive security scan report."""
    scan_id: str
    scan_type: SecurityScanType
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    scanned_files: List[str] = field(default_factory=list)
    scan_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_vulnerability(self, vulnerability: SecurityVulnerability) -> None:
        """Add vulnerability to report."""
        self.vulnerabilities.append(vulnerability)
    
    def get_vulnerabilities_by_severity(self, severity: VulnerabilitySeverity) -> List[SecurityVulnerability]:
        """Get vulnerabilities by severity level."""
        return [v for v in self.vulnerabilities if v.severity == severity]
    
    def get_critical_count(self) -> int:
        """Get count of critical vulnerabilities."""
        return len(self.get_vulnerabilities_by_severity(VulnerabilitySeverity.CRITICAL))
    
    def get_high_count(self) -> int:
        """Get count of high severity vulnerabilities."""
        return len(self.get_vulnerabilities_by_severity(VulnerabilitySeverity.HIGH))


class StaticCodeScanner:
    """
    Static code analysis scanner for security vulnerabilities.
    
    Scans Python source code for common security anti-patterns,
    dangerous function usage, and potential security vulnerabilities.
    """
    
    def __init__(self):
        """Initialize static code scanner."""
        self.logger = CorrelationLogger(__name__)
        
        # Dangerous function patterns
        self.dangerous_functions = {
            'eval': VulnerabilitySeverity.CRITICAL,
            'exec': VulnerabilitySeverity.CRITICAL,
            'compile': VulnerabilitySeverity.HIGH,
            '__import__': VulnerabilitySeverity.HIGH,
            'getattr': VulnerabilitySeverity.MEDIUM,
            'setattr': VulnerabilitySeverity.MEDIUM,
            'delattr': VulnerabilitySeverity.MEDIUM,
            'input': VulnerabilitySeverity.LOW,  # Python 2 style
            'raw_input': VulnerabilitySeverity.LOW,
        }
        
        # Dangerous import patterns
        self.dangerous_imports = {
            'pickle': VulnerabilitySeverity.HIGH,
            'cPickle': VulnerabilitySeverity.HIGH,
            'subprocess': VulnerabilitySeverity.MEDIUM,
            'os': VulnerabilitySeverity.LOW,
            'tempfile': VulnerabilitySeverity.LOW,
        }
        
        # Security-sensitive patterns
        self.security_patterns = [
            {
                'pattern': re.compile(r'password\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
                'title': 'Hardcoded Password',
                'severity': VulnerabilitySeverity.HIGH,
                'description': 'Hardcoded password found in source code'
            },
            {
                'pattern': re.compile(r'api[_-]?key\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
                'title': 'Hardcoded API Key',
                'severity': VulnerabilitySeverity.HIGH,
                'description': 'Hardcoded API key found in source code'
            },
            {
                'pattern': re.compile(r'secret[_-]?key\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
                'title': 'Hardcoded Secret Key',
                'severity': VulnerabilitySeverity.HIGH,
                'description': 'Hardcoded secret key found in source code'
            },
            {
                'pattern': re.compile(r'(?:mysql|postgresql|mongodb)://[^"\'\\s]+', re.IGNORECASE),
                'title': 'Database Connection String',
                'severity': VulnerabilitySeverity.MEDIUM,
                'description': 'Database connection string with credentials found'
            },
            {
                'pattern': re.compile(r'shell=True', re.IGNORECASE),
                'title': 'Shell Injection Risk',
                'severity': VulnerabilitySeverity.HIGH,
                'description': 'subprocess call with shell=True poses injection risk'
            },
            {
                'pattern': re.compile(r'verify=False', re.IGNORECASE),
                'title': 'SSL Verification Disabled',
                'severity': VulnerabilitySeverity.MEDIUM,
                'description': 'SSL certificate verification disabled'
            }
        ]
    
    @with_correlation("security_scan.static_code")
    def scan_directory(self, directory: Path, exclude_patterns: Optional[List[str]] = None) -> SecurityScanReport:
        """
        Scan directory for Python files and analyze for security vulnerabilities.
        
        Args:
            directory: Directory to scan
            exclude_patterns: File patterns to exclude
            
        Returns:
            Security scan report
        """
        scan_id = f"static_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report = SecurityScanReport(
            scan_id=scan_id,
            scan_type=SecurityScanType.STATIC_CODE,
            started_at=datetime.now()
        )
        
        exclude_patterns = exclude_patterns or [
            '*/venv/*', '*/venv_*/*', '*/.git/*', '*/__pycache__/*',
            '*/node_modules/*', '*/build/*', '*/dist/*'
        ]
        
        try:
            # Find Python files
            python_files = self._find_python_files(directory, exclude_patterns)
            report.scanned_files = [str(f) for f in python_files]
            
            self.logger.info(f"Starting static code scan of {len(python_files)} files")
            
            # Scan each file
            for file_path in python_files:
                try:
                    vulnerabilities = self._scan_file(file_path)
                    for vuln in vulnerabilities:
                        report.add_vulnerability(vuln)
                        
                except Exception as e:
                    self.logger.error(f"Error scanning file {file_path}: {str(e)}")
            
            report.completed_at = datetime.now()
            report.duration_seconds = (report.completed_at - report.started_at).total_seconds()
            
            self.logger.info(
                f"Static code scan completed",
                vulnerabilities_found=len(report.vulnerabilities),
                files_scanned=len(report.scanned_files),
                duration_seconds=report.duration_seconds
            )
            
        except Exception as e:
            self.logger.error(f"Static code scan failed: {str(e)}")
            report.scan_metadata['error'] = str(e)
        
        return report
    
    def _find_python_files(self, directory: Path, exclude_patterns: List[str]) -> List[Path]:
        """Find Python files in directory, excluding specified patterns."""
        python_files = []
        
        for file_path in directory.rglob("*.py"):
            # Check exclude patterns
            excluded = False
            for pattern in exclude_patterns:
                if file_path.match(pattern) or pattern in str(file_path):
                    excluded = True
                    break
            
            if not excluded:
                python_files.append(file_path)
        
        return python_files
    
    def _scan_file(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan individual Python file for security vulnerabilities."""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Pattern-based scanning
            vulnerabilities.extend(self._scan_patterns(file_path, content, lines))
            
            # AST-based scanning
            try:
                tree = ast.parse(content, filename=str(file_path))
                vulnerabilities.extend(self._scan_ast(file_path, tree, lines))
            except SyntaxError as e:
                # File has syntax errors, skip AST analysis
                self.logger.debug(f"Syntax error in {file_path}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
        
        return vulnerabilities
    
    def _scan_patterns(self, file_path: Path, content: str, lines: List[str]) -> List[SecurityVulnerability]:
        """Scan file content using regex patterns."""
        vulnerabilities = []
        
        for pattern_info in self.security_patterns:
            for match in pattern_info['pattern'].finditer(content):
                # Find line number
                line_number = content[:match.start()].count('\n') + 1
                
                vulnerability = SecurityVulnerability(
                    id=f"pattern_{hashlib.md5(match.group().encode()).hexdigest()[:8]}",
                    title=pattern_info['title'],
                    description=pattern_info['description'],
                    severity=pattern_info['severity'],
                    scan_type=SecurityScanType.STATIC_CODE,
                    file_path=str(file_path),
                    line_number=line_number,
                    remediation=f"Review and secure the identified pattern: {match.group()}"
                )
                
                vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    def _scan_ast(self, file_path: Path, tree: ast.AST, lines: List[str]) -> List[SecurityVulnerability]:
        """Scan file using Abstract Syntax Tree analysis."""
        vulnerabilities = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, scanner):
                self.scanner = scanner
                self.vulnerabilities = []
            
            def visit_Call(self, node):
                # Check dangerous function calls
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.scanner.dangerous_functions:
                        severity = self.scanner.dangerous_functions[func_name]
                        
                        vulnerability = SecurityVulnerability(
                            id=f"func_{func_name}_{node.lineno}",
                            title=f"Dangerous Function Call: {func_name}",
                            description=f"Use of potentially dangerous function '{func_name}'",
                            severity=severity,
                            scan_type=SecurityScanType.STATIC_CODE,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            remediation=f"Review the use of '{func_name}' and consider safer alternatives"
                        )
                        
                        self.vulnerabilities.append(vulnerability)
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check dangerous imports
                for alias in node.names:
                    module_name = alias.name
                    if module_name in self.scanner.dangerous_imports:
                        severity = self.scanner.dangerous_imports[module_name]
                        
                        vulnerability = SecurityVulnerability(
                            id=f"import_{module_name}_{node.lineno}",
                            title=f"Security-Sensitive Import: {module_name}",
                            description=f"Import of security-sensitive module '{module_name}'",
                            severity=severity,
                            scan_type=SecurityScanType.STATIC_CODE,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            remediation=f"Review the use of '{module_name}' module for security implications"
                        )
                        
                        self.vulnerabilities.append(vulnerability)
                
                self.generic_visit(node)
        
        visitor = SecurityVisitor(self)
        visitor.visit(tree)
        vulnerabilities.extend(visitor.vulnerabilities)
        
        return vulnerabilities


class DependencyScanner:
    """
    Dependency vulnerability scanner.
    
    Scans project dependencies against known vulnerability databases
    and identifies outdated or vulnerable packages.
    """
    
    def __init__(self, vulnerability_db_url: str = "https://pyup.io/api/v1/vulns/"):
        """Initialize dependency scanner."""
        self.logger = CorrelationLogger(__name__)
        self.vulnerability_db_url = vulnerability_db_url
        self.cache_dir = Path.home() / ".security_scanner_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_expiry = timedelta(hours=24)
    
    @with_correlation("security_scan.dependencies")
    def scan_dependencies(self, requirements_file: Optional[Path] = None) -> SecurityScanReport:
        """
        Scan project dependencies for known vulnerabilities.
        
        Args:
            requirements_file: Optional requirements.txt file path
            
        Returns:
            Security scan report
        """
        scan_id = f"deps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report = SecurityScanReport(
            scan_id=scan_id,
            scan_type=SecurityScanType.DEPENDENCY,
            started_at=datetime.now()
        )
        
        try:
            # Get installed packages
            if requirements_file and requirements_file.exists():
                packages = self._parse_requirements_file(requirements_file)
            else:
                packages = self._get_installed_packages()
            
            self.logger.info(f"Scanning {len(packages)} dependencies for vulnerabilities")
            
            # Check each package
            for package_name, version in packages.items():
                try:
                    vulnerabilities = self._check_package_vulnerabilities(package_name, version)
                    for vuln in vulnerabilities:
                        report.add_vulnerability(vuln)
                        
                except Exception as e:
                    self.logger.error(f"Error checking package {package_name}: {str(e)}")
            
            report.completed_at = datetime.now()
            report.duration_seconds = (report.completed_at - report.started_at).total_seconds()
            report.scan_metadata['packages_scanned'] = len(packages)
            
            self.logger.info(
                f"Dependency scan completed",
                vulnerabilities_found=len(report.vulnerabilities),
                packages_scanned=len(packages),
                duration_seconds=report.duration_seconds
            )
            
        except Exception as e:
            self.logger.error(f"Dependency scan failed: {str(e)}")
            report.scan_metadata['error'] = str(e)
        
        return report
    
    def _parse_requirements_file(self, requirements_file: Path) -> Dict[str, str]:
        """Parse requirements.txt file for package names and versions."""
        packages = {}
        
        try:
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse package specification
                        if '==' in line:
                            name, version = line.split('==', 1)
                            packages[name.strip()] = version.strip()
                        elif '>=' in line:
                            name = line.split('>=')[0].strip()
                            packages[name] = 'unknown'
                        else:
                            packages[line] = 'unknown'
        except Exception as e:
            self.logger.error(f"Error parsing requirements file: {str(e)}")
        
        return packages
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """Get currently installed packages and their versions."""
        packages = {}
        
        try:
            for dist in pkg_resources.working_set:
                packages[dist.project_name] = dist.version
        except Exception as e:
            self.logger.error(f"Error getting installed packages: {str(e)}")
        
        return packages
    
    def _check_package_vulnerabilities(self, package_name: str, version: str) -> List[SecurityVulnerability]:
        """Check specific package for known vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Check cache first
            cache_file = self.cache_dir / f"{package_name}.json"
            vuln_data = self._get_cached_vulnerability_data(cache_file)
            
            if not vuln_data:
                # Fetch from vulnerability database
                vuln_data = self._fetch_vulnerability_data(package_name)
                self._cache_vulnerability_data(cache_file, vuln_data)
            
            # Check if current version is vulnerable
            for vuln_info in vuln_data:
                if self._is_version_vulnerable(version, vuln_info.get('affected_versions', [])):
                    vulnerability = SecurityVulnerability(
                        id=f"cve_{vuln_info.get('id', 'unknown')}",
                        title=f"Vulnerable Dependency: {package_name}",
                        description=vuln_info.get('advisory', f"Vulnerability in {package_name} {version}"),
                        severity=self._map_severity(vuln_info.get('cvss', 0)),
                        scan_type=SecurityScanType.DEPENDENCY,
                        cve_id=vuln_info.get('cve'),
                        cvss_score=vuln_info.get('cvss'),
                        remediation=f"Update {package_name} to version {vuln_info.get('fixed_in', 'latest')}",
                        references=[vuln_info.get('url', '')]
                    )
                    
                    vulnerabilities.append(vulnerability)
        
        except Exception as e:
            self.logger.error(f"Error checking vulnerabilities for {package_name}: {str(e)}")
        
        return vulnerabilities
    
    def _get_cached_vulnerability_data(self, cache_file: Path) -> Optional[List[Dict[str, Any]]]:
        """Get cached vulnerability data if not expired."""
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is expired
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age > self.cache_expiry:
                return None
            
            with open(cache_file, 'r') as f:
                return json.load(f)
                
        except Exception:
            return None
    
    def _fetch_vulnerability_data(self, package_name: str) -> List[Dict[str, Any]]:
        """Fetch vulnerability data from external database."""
        # This is a simplified implementation
        # In production, you would integrate with actual vulnerability databases
        # like PyUp.io, GitHub Security Advisory Database, or National Vulnerability Database
        
        # Mock vulnerability data for demonstration
        known_vulnerable_packages = {
            'requests': [
                {
                    'id': 'PYUP-2020-001',
                    'advisory': 'Requests package vulnerable to SSRF attacks',
                    'affected_versions': ['<2.20.0'],
                    'fixed_in': '2.20.0',
                    'cvss': 7.5,
                    'cve': 'CVE-2018-18074'
                }
            ],
            'pillow': [
                {
                    'id': 'PYUP-2021-002',
                    'advisory': 'Pillow vulnerable to buffer overflow',
                    'affected_versions': ['<8.3.2'],
                    'fixed_in': '8.3.2',
                    'cvss': 9.8,
                    'cve': 'CVE-2021-34552'
                }
            ]
        }
        
        return known_vulnerable_packages.get(package_name, [])
    
    def _cache_vulnerability_data(self, cache_file: Path, data: List[Dict[str, Any]]) -> None:
        """Cache vulnerability data to disk."""
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.error(f"Error caching vulnerability data: {str(e)}")
    
    def _is_version_vulnerable(self, version: str, affected_versions: List[str]) -> bool:
        """Check if version is in affected versions list."""
        # Simplified version comparison
        # In production, use proper semantic version comparison
        for affected in affected_versions:
            if affected.startswith('<'):
                # Version should be less than specified
                continue  # Simplified - implement proper comparison
            elif affected.startswith('>='):
                # Version should be greater than or equal
                continue  # Simplified - implement proper comparison
            elif version == affected:
                return True
        
        return False
    
    def _map_severity(self, cvss_score: float) -> VulnerabilitySeverity:
        """Map CVSS score to vulnerability severity."""
        if cvss_score >= 9.0:
            return VulnerabilitySeverity.CRITICAL
        elif cvss_score >= 7.0:
            return VulnerabilitySeverity.HIGH
        elif cvss_score >= 4.0:
            return VulnerabilitySeverity.MEDIUM
        elif cvss_score > 0.0:
            return VulnerabilitySeverity.LOW
        else:
            return VulnerabilitySeverity.INFORMATIONAL


class FileSystemScanner:
    """
    File system security scanner.
    
    Scans file system for security misconfigurations, dangerous permissions,
    and potential security risks in file structures.
    """
    
    def __init__(self):
        """Initialize filesystem scanner."""
        self.logger = CorrelationLogger(__name__)
        
        # Dangerous file extensions
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js',
            '.jar', '.class', '.sh', '.pl', '.py', '.rb', '.php'
        }
        
        # Sensitive file patterns
        self.sensitive_patterns = [
            re.compile(r'.*\.key$', re.IGNORECASE),
            re.compile(r'.*\.pem$', re.IGNORECASE),
            re.compile(r'.*\.crt$', re.IGNORECASE),
            re.compile(r'.*\.p12$', re.IGNORECASE),
            re.compile(r'.*\.jks$', re.IGNORECASE),
            re.compile(r'.*password.*', re.IGNORECASE),
            re.compile(r'.*secret.*', re.IGNORECASE),
            re.compile(r'.*\.env$', re.IGNORECASE),
            re.compile(r'.*config.*\.ini$', re.IGNORECASE),
        ]
    
    @with_correlation("security_scan.filesystem")
    def scan_directory(self, directory: Path, max_depth: int = 5) -> SecurityScanReport:
        """
        Scan directory for filesystem security issues.
        
        Args:
            directory: Directory to scan
            max_depth: Maximum directory depth to scan
            
        Returns:
            Security scan report
        """
        scan_id = f"fs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report = SecurityScanReport(
            scan_id=scan_id,
            scan_type=SecurityScanType.FILESYSTEM,
            started_at=datetime.now()
        )
        
        try:
            self.logger.info(f"Starting filesystem security scan of {directory}")
            
            self._scan_directory_recursive(directory, report, max_depth, 0)
            
            report.completed_at = datetime.now()
            report.duration_seconds = (report.completed_at - report.started_at).total_seconds()
            
            self.logger.info(
                f"Filesystem scan completed",
                vulnerabilities_found=len(report.vulnerabilities),
                files_scanned=len(report.scanned_files),
                duration_seconds=report.duration_seconds
            )
            
        except Exception as e:
            self.logger.error(f"Filesystem scan failed: {str(e)}")
            report.scan_metadata['error'] = str(e)
        
        return report
    
    def _scan_directory_recursive(self, directory: Path, report: SecurityScanReport, 
                                max_depth: int, current_depth: int) -> None:
        """Recursively scan directory structure."""
        if current_depth >= max_depth:
            return
        
        try:
            for item in directory.iterdir():
                if item.is_file():
                    report.scanned_files.append(str(item))
                    self._scan_file_security(item, report)
                elif item.is_dir() and not item.name.startswith('.'):
                    self._scan_directory_recursive(item, report, max_depth, current_depth + 1)
        
        except PermissionError:
            # Cannot access directory
            vulnerability = SecurityVulnerability(
                id=f"perm_{hashlib.md5(str(directory).encode()).hexdigest()[:8]}",
                title="Directory Access Denied",
                description=f"Cannot access directory: {directory}",
                severity=VulnerabilitySeverity.INFORMATIONAL,
                scan_type=SecurityScanType.FILESYSTEM,
                file_path=str(directory),
                remediation="Review directory permissions"
            )
            report.add_vulnerability(vulnerability)
    
    def _scan_file_security(self, file_path: Path, report: SecurityScanReport) -> None:
        """Scan individual file for security issues."""
        try:
            # Check file extension
            if file_path.suffix.lower() in self.dangerous_extensions:
                vulnerability = SecurityVulnerability(
                    id=f"ext_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                    title="Dangerous File Extension",
                    description=f"File with potentially dangerous extension: {file_path.suffix}",
                    severity=VulnerabilitySeverity.MEDIUM,
                    scan_type=SecurityScanType.FILESYSTEM,
                    file_path=str(file_path),
                    remediation="Review if this file type is necessary and secure"
                )
                report.add_vulnerability(vulnerability)
            
            # Check sensitive file patterns
            for pattern in self.sensitive_patterns:
                if pattern.match(file_path.name):
                    vulnerability = SecurityVulnerability(
                        id=f"sens_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                        title="Sensitive File Detected",
                        description=f"File with sensitive pattern detected: {file_path.name}",
                        severity=VulnerabilitySeverity.HIGH,
                        scan_type=SecurityScanType.FILESYSTEM,
                        file_path=str(file_path),
                        remediation="Ensure sensitive files are properly protected and not exposed"
                    )
                    report.add_vulnerability(vulnerability)
                    break
            
            # Check file permissions (Unix-like systems)
            if hasattr(os, 'stat') and file_path.exists():
                stat_info = file_path.stat()
                mode = stat_info.st_mode
                
                # Check for world-writable files
                if mode & 0o002:  # World writable
                    vulnerability = SecurityVulnerability(
                        id=f"perm_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                        title="World-Writable File",
                        description=f"File is world-writable: {file_path}",
                        severity=VulnerabilitySeverity.HIGH,
                        scan_type=SecurityScanType.FILESYSTEM,
                        file_path=str(file_path),
                        remediation="Remove world-write permissions"
                    )
                    report.add_vulnerability(vulnerability)
        
        except Exception as e:
            self.logger.debug(f"Error scanning file {file_path}: {str(e)}")


class SecurityScanner:
    """
    Main security scanning coordinator.
    
    Orchestrates different types of security scans and provides
    unified reporting and alerting capabilities.
    """
    
    def __init__(self, project_root: Path):
        """Initialize security scanner."""
        self.project_root = project_root
        self.logger = CorrelationLogger(__name__)
        
        # Initialize scanners
        self.static_scanner = StaticCodeScanner()
        self.dependency_scanner = DependencyScanner()
        self.filesystem_scanner = FileSystemScanner()
        
        # Scan reports storage
        self.scan_reports: List[SecurityScanReport] = []
    
    @with_correlation("security_scan.comprehensive")
    def run_comprehensive_scan(self, 
                             include_static: bool = True,
                             include_dependencies: bool = True,
                             include_filesystem: bool = True) -> Dict[str, SecurityScanReport]:
        """
        Run comprehensive security scan across all available scanners.
        
        Args:
            include_static: Whether to run static code analysis
            include_dependencies: Whether to run dependency scanning
            include_filesystem: Whether to run filesystem scanning
            
        Returns:
            Dictionary of scan reports by scan type
        """
        self.logger.info("Starting comprehensive security scan")
        
        reports = {}
        
        try:
            # Static code analysis
            if include_static:
                self.logger.info("Running static code analysis...")
                static_report = self.static_scanner.scan_directory(self.project_root)
                reports['static_code'] = static_report
                self.scan_reports.append(static_report)
            
            # Dependency scanning
            if include_dependencies:
                self.logger.info("Running dependency vulnerability scan...")
                requirements_file = self.project_root / "requirements.txt"
                dependency_report = self.dependency_scanner.scan_dependencies(requirements_file)
                reports['dependencies'] = dependency_report
                self.scan_reports.append(dependency_report)
            
            # Filesystem scanning
            if include_filesystem:
                self.logger.info("Running filesystem security scan...")
                filesystem_report = self.filesystem_scanner.scan_directory(self.project_root)
                reports['filesystem'] = filesystem_report
                self.scan_reports.append(filesystem_report)
            
            # Generate summary
            self._log_scan_summary(reports)
            
        except Exception as e:
            self.logger.error(f"Comprehensive security scan failed: {str(e)}")
        
        return reports
    
    def _log_scan_summary(self, reports: Dict[str, SecurityScanReport]) -> None:
        """Log summary of security scan results."""
        total_vulnerabilities = 0
        critical_count = 0
        high_count = 0
        
        for scan_type, report in reports.items():
            vuln_count = len(report.vulnerabilities)
            total_vulnerabilities += vuln_count
            critical_count += report.get_critical_count()
            high_count += report.get_high_count()
            
            self.logger.info(
                f"{scan_type} scan completed",
                vulnerabilities_found=vuln_count,
                critical_vulnerabilities=report.get_critical_count(),
                high_vulnerabilities=report.get_high_count()
            )
        
        self.logger.info(
            "Security scan summary",
            total_vulnerabilities=total_vulnerabilities,
            critical_vulnerabilities=critical_count,
            high_vulnerabilities=high_count,
            scans_completed=len(reports)
        )
        
        # Alert on critical findings
        if critical_count > 0:
            self.logger.critical(
                f"CRITICAL SECURITY VULNERABILITIES FOUND: {critical_count}",
                critical_count=critical_count,
                immediate_action_required=True
            )
    
    def generate_security_report(self, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        if not self.scan_reports:
            self.logger.warning("No scan reports available for report generation")
            return {}
        
        report_data = {
            "scan_summary": {
                "generated_at": datetime.now().isoformat(),
                "total_scans": len(self.scan_reports),
                "project_root": str(self.project_root)
            },
            "vulnerability_summary": {},
            "scan_reports": []
        }
        
        # Aggregate vulnerability statistics
        total_vulns = 0
        severity_counts = {severity.value: 0 for severity in VulnerabilitySeverity}
        
        for report in self.scan_reports:
            report_data["scan_reports"].append(asdict(report))
            
            total_vulns += len(report.vulnerabilities)
            for vuln in report.vulnerabilities:
                severity_counts[vuln.severity.value] += 1
        
        report_data["vulnerability_summary"] = {
            "total_vulnerabilities": total_vulns,
            "by_severity": severity_counts,
            "critical_vulnerabilities": severity_counts[VulnerabilitySeverity.CRITICAL.value],
            "high_vulnerabilities": severity_counts[VulnerabilitySeverity.HIGH.value]
        }
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                self.logger.info(f"Security report saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error saving security report: {str(e)}")
        
        return report_data


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize security scanner
    project_root = Path(".")
    scanner = SecurityScanner(project_root)
    
    # Run comprehensive security scan
    reports = scanner.run_comprehensive_scan()
    
    # Generate report
    report_file = Path("security_report.json")
    security_data = scanner.generate_security_report(report_file)
    
    print(f"Security scan completed!")
    print(f"Total vulnerabilities: {security_data['vulnerability_summary']['total_vulnerabilities']}")
    print(f"Critical: {security_data['vulnerability_summary']['critical_vulnerabilities']}")
    print(f"High: {security_data['vulnerability_summary']['high_vulnerabilities']}")
    print(f"Report saved to: {report_file}")