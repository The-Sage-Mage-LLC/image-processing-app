#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enterprise Features Integration Test
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive integration test demonstrating all enterprise features
working together: performance monitoring, input validation, security scanning,
audit trails, and access controls.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import time
import json

# Import all our enterprise systems
from src.monitoring.performance_dashboard import PerformanceDashboard, MetricType
from src.validation.input_validator import InputValidator, StringValidator, NumberValidator, FileValidator
from src.security.security_scanner import SecurityScanner
from src.audit.file_audit import FileOperationAuditor, AuditConfiguration, ComplianceLevel
from src.access_control.user_access import (
    UserAccessControl, SecurityConfiguration, UserRole, Permission
)
from src.utils.structured_logging import CorrelationLogger, CorrelationContext


class EnterpriseIntegrationTest:
    """
    Integration test suite demonstrating enterprise features.
    
    This test suite shows how all enterprise components work together
    to provide a comprehensive security and monitoring solution.
    """
    
    def __init__(self):
        """Initialize integration test environment."""
        self.logger = CorrelationLogger(__name__)
        self.test_dir = Path(tempfile.mkdtemp(prefix="enterprise_test_"))
        
        # Initialize all enterprise components
        self._setup_components()
        
        self.logger.info("Enterprise integration test initialized", 
                        test_directory=str(self.test_dir))
    
    def _setup_components(self):
        """Setup all enterprise components."""
        
        # 1. Performance monitoring
        self.performance_dashboard = PerformanceDashboard(
            db_path=str(self.test_dir / "performance.db")
        )
        
        # 2. Input validation
        self.input_validator = self._create_validator()
        
        # 3. Security scanning
        self.security_scanner = SecurityScanner(self.test_dir)
        
        # 4. Audit trail
        audit_config = AuditConfiguration(
            compliance_level=ComplianceLevel.ENHANCED,
            audit_database_path=str(self.test_dir / "audit.db"),
            encrypt_audit_logs=True,
            monitor_read_operations=True
        )
        self.file_auditor = FileOperationAuditor(audit_config)
        
        # 5. Access control
        security_config = SecurityConfiguration(
            password_min_length=8,
            enforce_password_complexity=True,
            session_timeout_minutes=60
        )
        self.access_control = UserAccessControl(
            db_path=str(self.test_dir / "users.db"),
            config=security_config
        )
    
    def _create_validator(self) -> InputValidator:
        """Create comprehensive input validator."""
        validator = InputValidator()
        
        # File path validation
        validator.add_validator(
            "file_path",
            StringValidator(
                max_length=255,
                pattern=r'^[a-zA-Z0-9_\-./\\]+$',
                forbidden_chars='<>"|*?',
                check_xss=True
            )
        )
        
        # Image processing parameters
        validator.add_validator(
            "quality",
            NumberValidator(
                numeric_type=int,
                min_value=1,
                max_value=100
            )
        )
        
        validator.add_validator(
            "width",
            NumberValidator(
                numeric_type=int,
                min_value=1,
                max_value=8192
            )
        )
        
        validator.add_validator(
            "height",
            NumberValidator(
                numeric_type=int,
                min_value=1,
                max_value=8192
            )
        )
        
        return validator
    
    async def run_comprehensive_test(self):
        """Run comprehensive integration test."""
        print("\n?? Starting Enterprise Features Integration Test")
        print("=" * 60)
        
        try:
            # Start monitoring
            self.performance_dashboard.start()
            
            # Test 1: Access Control and Authentication
            await self._test_access_control()
            
            # Test 2: Input Validation
            await self._test_input_validation()
            
            # Test 3: File Operations with Audit Trail
            await self._test_audited_file_operations()
            
            # Test 4: Security Scanning
            await self._test_security_scanning()
            
            # Test 5: Performance Monitoring
            await self._test_performance_monitoring()
            
            # Test 6: Integration Workflow
            await self._test_integration_workflow()
            
            # Generate comprehensive report
            await self._generate_final_report()
            
        except Exception as e:
            self.logger.error(f"Integration test failed: {str(e)}")
            raise
        finally:
            self.performance_dashboard.stop()
            self._cleanup()
    
    async def _test_access_control(self):
        """Test access control system."""
        print("\n1??  Testing Access Control System...")
        
        # Authenticate admin
        admin_session = self.access_control.authenticate_user("admin", "Admin123!")
        assert admin_session is not None, "Admin authentication failed"
        
        print(f"   ? Admin authenticated: {admin_session.session_id[:8]}...")
        
        # Create test users
        test_user = self.access_control.create_user(
            username="operator",
            email="operator@test.com",
            full_name="Test Operator",
            password="TestPass123!",
            role=UserRole.OPERATOR,
            creator_session_id=admin_session.session_id
        )
        
        assert test_user is not None, "User creation failed"
        print(f"   ? Created operator user: {test_user.username}")
        
        # Test permissions
        user_session = self.access_control.authenticate_user("operator", "TestPass123!")
        assert user_session is not None, "User authentication failed"
        
        can_read = self.access_control.authorize_operation(
            user_session.session_id, Permission.FILE_READ
        )
        can_admin = self.access_control.authorize_operation(
            user_session.session_id, Permission.SYSTEM_ADMIN
        )
        
        assert can_read, "User should have read permission"
        assert not can_admin, "User should not have admin permission"
        
        print(f"   ? Permissions verified - Read: {can_read}, Admin: {can_admin}")
        
        # Store session for later use
        self.test_session = user_session
    
    async def _test_input_validation(self):
        """Test input validation framework."""
        print("\n2??  Testing Input Validation Framework...")
        
        # Test valid input
        valid_data = {
            "file_path": "test_image.jpg",
            "quality": 85,
            "width": 800,
            "height": 600
        }
        
        report = self.input_validator.validate_data(valid_data)
        assert report.is_valid, f"Valid data failed validation: {report.errors}"
        
        print(f"   ? Valid input passed: {len(report.sanitized_data)} fields")
        
        # Test invalid input
        invalid_data = {
            "file_path": "<script>alert('xss')</script>",
            "quality": 150,  # Too high
            "width": -100,   # Negative
            "height": "not_a_number"
        }
        
        report = self.input_validator.validate_data(invalid_data)
        assert not report.is_valid, "Invalid data should fail validation"
        assert len(report.errors) > 0, "Should have validation errors"
        
        print(f"   ? Invalid input rejected: {len(report.errors)} errors found")
        
        # Record validation metrics
        self.performance_dashboard.record_metric(
            "validation.success_rate",
            50.0,  # 1 of 2 tests passed
            MetricType.GAUGE
        )
    
    async def _test_audited_file_operations(self):
        """Test file operations with audit trail."""
        print("\n3??  Testing Audited File Operations...")
        
        # Create audited file operations
        from src.audit.file_audit import create_audited_file_operations, FileOperationType
        
        file_ops = create_audited_file_operations(self.file_auditor)
        
        # Perform file operations with audit context
        with CorrelationContext.context(
            user_id=self.test_session.user_id,
            session_id=self.test_session.session_id
        ):
            test_file = self.test_dir / "test_audit_file.txt"
            
            # Create file
            file_ops['create_file'](test_file, "This is test content.")
            print(f"   ? Created file with audit: {test_file.name}")
            
            # Read file
            content = file_ops['read_file'](test_file)
            assert "test content" in content, "File content mismatch"
            print(f"   ? Read file with audit: {len(content)} bytes")
            
            # Update file
            file_ops['update_file'](test_file, "Updated content.")
            print(f"   ? Updated file with audit")
            
            # Copy file
            copy_file = self.test_dir / "test_audit_copy.txt"
            file_ops['copy_file'](test_file, copy_file)
            print(f"   ? Copied file with audit: {copy_file.name}")
            
            # Get audit history
            history = self.file_auditor.get_file_audit_history(str(test_file))
            assert len(history) >= 3, f"Expected at least 3 audit events, got {len(history)}"
            
            print(f"   ? Audit trail verified: {len(history)} events recorded")
            
            # Clean up
            file_ops['delete_file'](test_file)
            file_ops['delete_file'](copy_file)
    
    async def _test_security_scanning(self):
        """Test security scanning framework."""
        print("\n4??  Testing Security Scanning Framework...")
        
        # Create test files for scanning
        test_python_file = self.test_dir / "test_security.py"
        test_python_file.write_text("""
# Test file with security issues
password = "hardcoded123"
api_key = "sk-1234567890abcdef"

import subprocess
import pickle

def dangerous_function():
    # This is a security risk
    subprocess.run("rm -rf /", shell=True)
    eval("print('dangerous')")
    
def pickle_load(data):
    return pickle.loads(data)
""")
        
        # Run security scan
        reports = self.security_scanner.run_comprehensive_scan(
            include_static=True,
            include_dependencies=True,
            include_filesystem=True
        )
        
        print(f"   ? Security scans completed: {len(reports)} scan types")
        
        # Verify findings
        total_vulnerabilities = 0
        for scan_type, report in reports.items():
            vuln_count = len(report.vulnerabilities)
            total_vulnerabilities += vuln_count
            print(f"   ?? {scan_type}: {vuln_count} vulnerabilities")
        
        assert total_vulnerabilities > 0, "Should find security vulnerabilities in test file"
        
        # Record security metrics
        self.performance_dashboard.record_metric(
            "security.vulnerabilities_found",
            total_vulnerabilities,
            MetricType.GAUGE
        )
    
    async def _test_performance_monitoring(self):
        """Test performance monitoring dashboard."""
        print("\n5??  Testing Performance Monitoring Dashboard...")
        
        # Record various metrics
        for i in range(10):
            # Simulate image processing operations
            self.performance_dashboard.record_timing("image_processing", 50 + i * 5)
            
            # Simulate API requests
            self.performance_dashboard.record_metric(
                "api.requests.total",
                i + 1,
                MetricType.COUNTER
            )
            
            # Simulate error rates
            self.performance_dashboard.record_metric(
                "api.error_rate",
                0.05 if i % 3 == 0 else 0.02,
                MetricType.GAUGE
            )
            
            await asyncio.sleep(0.1)  # Small delay
        
        # Get performance snapshot
        snapshot = self.performance_dashboard.get_performance_snapshot()
        
        print(f"   ? Performance metrics recorded")
        print(f"   ?? System metrics: {len(snapshot.system_metrics)}")
        print(f"   ?? Active alerts: {len(snapshot.active_alerts)}")
        
        # Get dashboard data
        dashboard_data = self.performance_dashboard.get_dashboard_data()
        
        print(f"   ? Dashboard data generated: {len(dashboard_data)} sections")
    
    async def _test_integration_workflow(self):
        """Test integrated workflow with all components."""
        print("\n6??  Testing Integrated Workflow...")
        
        start_time = time.time()
        
        # Simulate image processing workflow
        with CorrelationContext.context(
            user_id=self.test_session.user_id,
            session_id=self.test_session.session_id,
            correlation_id="workflow_test_001"
        ):
            # 1. Validate input parameters
            input_data = {
                "file_path": "input_image.jpg",
                "quality": 90,
                "width": 1024,
                "height": 768
            }
            
            validation_report = self.input_validator.validate_data(input_data)
            assert validation_report.is_valid, "Input validation failed"
            
            # 2. Check user permissions
            can_process = self.access_control.authorize_operation(
                self.test_session.session_id,
                Permission.IMAGE_PROCESS
            )
            assert can_process, "User should have image processing permission"
            
            # 3. Create and process file with auditing
            from src.audit.file_audit import create_audited_file_operations
            file_ops = create_audited_file_operations(self.file_auditor)
            
            input_file = self.test_dir / "input_image.jpg"
            output_file = self.test_dir / "output_image.jpg"
            
            # Simulate file creation and processing
            file_ops['create_file'](input_file, b"dummy image data")
            
            # Record processing time
            processing_time = (time.time() - start_time) * 1000
            self.performance_dashboard.record_timing("workflow.total_time", processing_time)
            
            file_ops['create_file'](output_file, b"processed image data")
            
            # 4. Get workflow audit trail
            audit_events = self.file_auditor.get_file_audit_history(str(input_file))
            
            print(f"   ? Integrated workflow completed")
            print(f"   ?? Processing time: {processing_time:.2f}ms")
            print(f"   ?? Audit events: {len(audit_events)}")
            
            # Clean up
            file_ops['delete_file'](input_file)
            file_ops['delete_file'](output_file)
    
    async def _generate_final_report(self):
        """Generate comprehensive final report."""
        print("\n?? Generating Comprehensive Enterprise Report...")
        
        # Collect data from all systems
        report_data = {
            "test_summary": {
                "timestamp": datetime.now().isoformat(),
                "test_duration_seconds": time.time() - getattr(self, 'start_time', time.time()),
                "components_tested": [
                    "Performance Monitoring",
                    "Input Validation",
                    "Security Scanning", 
                    "Audit Trail",
                    "Access Control"
                ]
            },
            "performance_data": self.performance_dashboard.get_dashboard_data(),
            "security_report": self.security_scanner.generate_security_report(),
            "user_activity": self.file_auditor.get_user_activity_report(
                self.test_session.user_id, days=1
            ),
            "access_control_summary": {
                "active_sessions": len(self.access_control.session_manager.get_user_sessions(
                    self.test_session.user_id
                )),
                "user_permissions": len(self.test_session.permissions)
            }
        }
        
        # Save report
        report_file = self.test_dir / "enterprise_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"   ? Comprehensive report saved: {report_file}")
        print(f"   ?? Report size: {report_file.stat().st_size} bytes")
        
        # Print summary
        perf_data = report_data["performance_data"]
        security_data = report_data["security_report"]
        
        print(f"\n?? Enterprise Features Summary:")
        print(f"   ?? Security vulnerabilities found: {security_data.get('vulnerability_summary', {}).get('total_vulnerabilities', 0)}")
        print(f"   ?? Performance metrics collected: {len(perf_data.get('metrics_summary', {}))}")
        print(f"   ?? File operations audited: {report_data['user_activity']['total_operations']}")
        print(f"   ?? Active user sessions: {report_data['access_control_summary']['active_sessions']}")
        print(f"   ??? User permissions: {report_data['access_control_summary']['user_permissions']}")
    
    def _cleanup(self):
        """Clean up test environment."""
        try:
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
            print(f"\n?? Test environment cleaned up: {self.test_dir}")
        except Exception as e:
            self.logger.error(f"Error cleaning up test environment: {str(e)}")


async def main():
    """Run enterprise integration test."""
    print("?? ENTERPRISE IMAGE PROCESSING APPLICATION")
    print("Integration Test for All Enterprise Features")
    print("=" * 60)
    
    test = EnterpriseIntegrationTest()
    test.start_time = time.time()
    
    try:
        await test.run_comprehensive_test()
        
        print("\n" + "=" * 60)
        print("? ALL ENTERPRISE FEATURES INTEGRATION TEST PASSED!")
        print("?? Your Image Processing Application is now enterprise-ready!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n? Integration test failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())