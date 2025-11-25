#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enterprise Features Verification Script
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Quick verification script to ensure all enterprise features are properly installed
and can be imported successfully.
"""

import sys
from pathlib import Path

def verify_enterprise_features():
    """Verify all enterprise features are properly installed."""
    print("?? Verifying Enterprise Features Installation...")
    print("=" * 60)
    
    features_status = {}
    
    # 1. Performance Monitoring Dashboard
    try:
        from src.monitoring.performance_dashboard import PerformanceDashboard, MetricType
        features_status["Performance Monitoring"] = "? AVAILABLE"
    except ImportError as e:
        features_status["Performance Monitoring"] = f"? MISSING: {e}"
    
    # 2. Input Validation Framework
    try:
        from src.validation.input_validator import InputValidator, create_image_processing_validator
        features_status["Input Validation"] = "? AVAILABLE"
    except ImportError as e:
        features_status["Input Validation"] = f"? MISSING: {e}"
    
    # 3. Security Scanning
    try:
        from src.security.security_scanner import SecurityScanner, VulnerabilitySeverity
        features_status["Security Scanning"] = "? AVAILABLE"
    except ImportError as e:
        features_status["Security Scanning"] = f"? MISSING: {e}"
    
    # 4. File Audit Trail
    try:
        from src.audit.file_audit import FileOperationAuditor, AuditConfiguration
        features_status["File Audit Trail"] = "? AVAILABLE"
    except ImportError as e:
        features_status["File Audit Trail"] = f"? MISSING: {e}"
    
    # 5. User Access Control
    try:
        from src.access_control.user_access import UserAccessControl, UserRole, Permission
        features_status["User Access Control"] = "? AVAILABLE"
    except ImportError as e:
        features_status["User Access Control"] = f"? MISSING: {e}"
    
    # 6. Structured Logging (existing)
    try:
        from src.utils.structured_logging import CorrelationLogger, CorrelationContext
        features_status["Structured Logging"] = "? AVAILABLE"
    except ImportError as e:
        features_status["Structured Logging"] = f"? MISSING: {e}"
    
    # 7. Health Checks (existing)
    try:
        from src.utils.health_checks import HealthCheckManager
        features_status["Health Monitoring"] = "? AVAILABLE"
    except ImportError as e:
        features_status["Health Monitoring"] = f"? MISSING: {e}"
    
    # Print results
    print("\n?? Enterprise Features Status:")
    print("-" * 60)
    for feature, status in features_status.items():
        print(f"{feature:<25} {status}")
    
    # Count available features
    available_count = sum(1 for status in features_status.values() if status.startswith("?"))
    total_count = len(features_status)
    
    print(f"\n?? Summary: {available_count}/{total_count} enterprise features available")
    
    # Check file structure
    print(f"\n?? File Structure Verification:")
    print("-" * 60)
    
    expected_files = [
        "src/monitoring/performance_dashboard.py",
        "src/validation/input_validator.py", 
        "src/security/security_scanner.py",
        "src/audit/file_audit.py",
        "src/access_control/user_access.py",
        "src/utils/structured_logging.py",
        "src/utils/health_checks.py",
        "test_enterprise_integration.py",
        "IMPLEMENTATION_SUMMARY.md"
    ]
    
    file_status = {}
    for file_path in expected_files:
        path = Path(file_path)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            file_status[file_path] = f"? EXISTS ({size_kb:.1f} KB)"
        else:
            file_status[file_path] = "? MISSING"
    
    for file_path, status in file_status.items():
        print(f"{file_path:<40} {status}")
    
    # Final assessment
    files_available = sum(1 for status in file_status.values() if status.startswith("?"))
    files_total = len(file_status)
    
    print(f"\n?? ENTERPRISE READINESS ASSESSMENT:")
    print("=" * 60)
    print(f"?? Features Available: {available_count}/{total_count} ({(available_count/total_count)*100:.1f}%)")
    print(f"?? Files Available: {files_available}/{files_total} ({(files_available/files_total)*100:.1f}%)")
    
    if available_count == total_count and files_available == files_total:
        print(f"\n?? CONGRATULATIONS!")
        print(f"? ALL ENTERPRISE FEATURES ARE SUCCESSFULLY INSTALLED!")
        print(f"?? Your Image Processing Application is now ENTERPRISE-READY!")
        print(f"\n?? Next Steps:")
        print(f"   • Run: python test_enterprise_integration.py")
        print(f"   • Review: IMPLEMENTATION_SUMMARY.md")
        print(f"   • Deploy: Follow deployment guide in documentation")
        return True
    else:
        print(f"\n??  SOME FEATURES ARE MISSING")
        print(f"? Please check the missing components above")
        print(f"?? Ensure all files are properly created and accessible")
        return False

if __name__ == "__main__":
    success = verify_enterprise_features()
    sys.exit(0 if success else 1)