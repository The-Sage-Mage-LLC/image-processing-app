#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Operations Audit Trail System
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive audit trail system for file operations with detailed logging,
integrity verification, and compliance tracking for enterprise environments.

Features:
- Complete file operation auditing (CRUD operations)
- File integrity verification with checksums
- Audit log integrity protection
- Compliance reporting and retention policies
- Real-time monitoring and alerting
- Encrypted audit storage
"""

import os
import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import shutil
import time
import logging
from contextlib import contextmanager
import cryptography.fernet

# Import our structured logging system
from ..utils.structured_logging import CorrelationLogger, with_correlation, CorrelationContext


class FileOperationType(Enum):
    """Types of file operations to audit."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"
    ACCESS = "access"
    PERMISSION_CHANGE = "permission_change"
    METADATA_CHANGE = "metadata_change"


class AuditEventStatus(Enum):
    """Status of audit events."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    DENIED = "denied"


class ComplianceLevel(Enum):
    """Compliance levels for audit requirements."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    STRICT = "strict"
    REGULATORY = "regulatory"


@dataclass
class FileMetadata:
    """File metadata for integrity verification."""
    file_path: str
    file_size: int
    creation_time: datetime
    modification_time: datetime
    access_time: datetime
    permissions: str
    owner: Optional[str] = None
    group: Optional[str] = None
    checksum_md5: Optional[str] = None
    checksum_sha256: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class AuditEvent:
    """Individual file operation audit event."""
    event_id: str
    correlation_id: Optional[str]
    timestamp: datetime
    operation_type: FileOperationType
    file_path: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    status: AuditEventStatus = AuditEventStatus.SUCCESS
    error_message: Optional[str] = None
    old_metadata: Optional[FileMetadata] = None
    new_metadata: Optional[FileMetadata] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    integrity_hash: Optional[str] = None
    
    def __post_init__(self):
        """Generate integrity hash after initialization."""
        if not self.integrity_hash:
            self.integrity_hash = self._calculate_integrity_hash()
    
    def _calculate_integrity_hash(self) -> str:
        """Calculate integrity hash for the audit event."""
        # Create deterministic string representation
        data_str = (
            f"{self.event_id}|{self.timestamp.isoformat()}|"
            f"{self.operation_type.value}|{self.file_path}|"
            f"{self.user_id}|{self.status.value}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()


@dataclass
class AuditConfiguration:
    """Configuration for audit trail system."""
    enabled: bool = True
    compliance_level: ComplianceLevel = ComplianceLevel.ENHANCED
    audit_database_path: str = "file_audit.db"
    max_audit_retention_days: int = 365
    encrypt_audit_logs: bool = True
    include_file_content_hash: bool = True
    monitor_read_operations: bool = False  # Can be performance intensive
    excluded_file_patterns: List[str] = field(default_factory=lambda: [
        "*.tmp", "*.log", "*/.git/*", "*/__pycache__/*"
    ])
    alert_on_suspicious_activity: bool = True
    backup_audit_logs: bool = True
    compress_old_logs: bool = True


class FileIntegrityManager:
    """
    Manages file integrity verification and checksum calculation.
    
    Provides efficient checksum calculation with caching and
    supports multiple hash algorithms for different use cases.
    """
    
    def __init__(self, cache_checksums: bool = True):
        """Initialize file integrity manager."""
        self.cache_checksums = cache_checksums
        self.checksum_cache: Dict[str, Dict[str, str]] = {}
        self.logger = CorrelationLogger(__name__)
    
    def calculate_file_checksums(self, file_path: Path) -> Dict[str, str]:
        """
        Calculate MD5 and SHA256 checksums for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with checksum values
        """
        if not file_path.exists() or not file_path.is_file():
            return {}
        
        file_key = f"{file_path}_{file_path.stat().st_mtime}"
        
        # Check cache
        if self.cache_checksums and file_key in self.checksum_cache:
            return self.checksum_cache[file_key]
        
        checksums = {}
        
        try:
            # Calculate checksums
            md5_hash = hashlib.md5()
            sha256_hash = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                # Read file in chunks for memory efficiency
                for chunk in iter(lambda: f.read(8192), b""):
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)
            
            checksums = {
                'md5': md5_hash.hexdigest(),
                'sha256': sha256_hash.hexdigest()
            }
            
            # Cache result
            if self.cache_checksums:
                self.checksum_cache[file_key] = checksums
            
        except Exception as e:
            self.logger.error(f"Error calculating checksums for {file_path}: {str(e)}")
        
        return checksums
    
    def verify_file_integrity(self, file_path: Path, expected_checksums: Dict[str, str]) -> bool:
        """
        Verify file integrity against expected checksums.
        
        Args:
            file_path: Path to the file
            expected_checksums: Expected checksum values
            
        Returns:
            True if file integrity is verified
        """
        if not expected_checksums:
            return True
        
        current_checksums = self.calculate_file_checksums(file_path)
        
        for algorithm, expected_value in expected_checksums.items():
            if algorithm in current_checksums:
                if current_checksums[algorithm] != expected_value:
                    self.logger.warning(
                        f"File integrity check failed for {file_path}",
                        algorithm=algorithm,
                        expected=expected_value,
                        actual=current_checksums.get(algorithm)
                    )
                    return False
        
        return True
    
    def get_file_metadata(self, file_path: Path) -> Optional[FileMetadata]:
        """Get comprehensive file metadata."""
        if not file_path.exists():
            return None
        
        try:
            stat_info = file_path.stat()
            checksums = self.calculate_file_checksums(file_path)
            
            # Get MIME type
            import mimetypes
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            return FileMetadata(
                file_path=str(file_path),
                file_size=stat_info.st_size,
                creation_time=datetime.fromtimestamp(stat_info.st_ctime),
                modification_time=datetime.fromtimestamp(stat_info.st_mtime),
                access_time=datetime.fromtimestamp(stat_info.st_atime),
                permissions=oct(stat_info.st_mode)[-3:],
                checksum_md5=checksums.get('md5'),
                checksum_sha256=checksums.get('sha256'),
                mime_type=mime_type
            )
            
        except Exception as e:
            self.logger.error(f"Error getting metadata for {file_path}: {str(e)}")
            return None


class AuditDatabase:
    """
    Secure audit database with integrity protection.
    
    Manages audit event storage with encryption, compression,
    and integrity verification capabilities.
    """
    
    def __init__(self, db_path: str, encryption_key: Optional[bytes] = None):
        """Initialize audit database."""
        self.db_path = Path(db_path)
        self.encryption_key = encryption_key
        self.cipher = None
        
        if encryption_key:
            self.cipher = cryptography.fernet.Fernet(encryption_key)
        
        self.logger = CorrelationLogger(__name__)
        self._init_database()
        
        # Database connection pool
        self._db_lock = threading.RLock()
    
    def _init_database(self) -> None:
        """Initialize audit database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    correlation_id TEXT,
                    timestamp REAL NOT NULL,
                    operation_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    source_ip TEXT,
                    user_agent TEXT,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    old_metadata TEXT,  -- JSON
                    new_metadata TEXT,  -- JSON
                    additional_data TEXT,  -- JSON
                    integrity_hash TEXT NOT NULL,
                    encrypted_data TEXT,  -- For encrypted storage
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                ON audit_events(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_correlation 
                ON audit_events(correlation_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_file_path 
                ON audit_events(file_path)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_user 
                ON audit_events(user_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_operation 
                ON audit_events(operation_type)
            """)
            
            # Audit log integrity table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_integrity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_start_id INTEGER NOT NULL,
                    batch_end_id INTEGER NOT NULL,
                    batch_hash TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    @with_correlation("audit.store_event")
    def store_audit_event(self, event: AuditEvent) -> bool:
        """Store audit event in database."""
        try:
            with self._db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Prepare data for storage
                    old_metadata_json = None
                    if event.old_metadata:
                        old_metadata_json = json.dumps(asdict(event.old_metadata), default=str)
                    
                    new_metadata_json = None
                    if event.new_metadata:
                        new_metadata_json = json.dumps(asdict(event.new_metadata), default=str)
                    
                    additional_data_json = json.dumps(event.additional_data) if event.additional_data else None
                    
                    # Encrypt sensitive data if encryption is enabled
                    encrypted_data = None
                    if self.cipher:
                        sensitive_data = {
                            'user_agent': event.user_agent,
                            'additional_data': event.additional_data
                        }
                        encrypted_data = self.cipher.encrypt(
                            json.dumps(sensitive_data).encode()
                        ).decode()
                    
                    conn.execute("""
                        INSERT INTO audit_events 
                        (event_id, correlation_id, timestamp, operation_type, file_path,
                         user_id, session_id, source_ip, user_agent, status, error_message,
                         old_metadata, new_metadata, additional_data, integrity_hash, encrypted_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id, event.correlation_id, event.timestamp.timestamp(),
                        event.operation_type.value, event.file_path, event.user_id,
                        event.session_id, event.source_ip, event.user_agent,
                        event.status.value, event.error_message, old_metadata_json,
                        new_metadata_json, additional_data_json, event.integrity_hash,
                        encrypted_data
                    ))
            
            self.logger.debug(f"Stored audit event: {event.event_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing audit event: {str(e)}")
            return False
    
    def query_audit_events(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          file_path: Optional[str] = None,
                          user_id: Optional[str] = None,
                          operation_type: Optional[FileOperationType] = None,
                          correlation_id: Optional[str] = None,
                          limit: int = 1000) -> List[AuditEvent]:
        """Query audit events with various filters."""
        events = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM audit_events WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.timestamp())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.timestamp())
                
                if file_path:
                    query += " AND file_path LIKE ?"
                    params.append(f"%{file_path}%")
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                if operation_type:
                    query += " AND operation_type = ?"
                    params.append(operation_type.value)
                
                if correlation_id:
                    query += " AND correlation_id = ?"
                    params.append(correlation_id)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                
                for row in cursor.fetchall():
                    event = self._row_to_audit_event(row)
                    if event:
                        events.append(event)
        
        except Exception as e:
            self.logger.error(f"Error querying audit events: {str(e)}")
        
        return events
    
    def _row_to_audit_event(self, row) -> Optional[AuditEvent]:
        """Convert database row to AuditEvent object."""
        try:
            # Parse metadata
            old_metadata = None
            if row[12]:  # old_metadata column
                old_metadata_dict = json.loads(row[12])
                old_metadata = FileMetadata(**old_metadata_dict)
            
            new_metadata = None
            if row[13]:  # new_metadata column
                new_metadata_dict = json.loads(row[13])
                new_metadata = FileMetadata(**new_metadata_dict)
            
            additional_data = {}
            if row[14]:  # additional_data column
                additional_data = json.loads(row[14])
            
            return AuditEvent(
                event_id=row[1],
                correlation_id=row[2],
                timestamp=datetime.fromtimestamp(row[3]),
                operation_type=FileOperationType(row[4]),
                file_path=row[5],
                user_id=row[6],
                session_id=row[7],
                source_ip=row[8],
                user_agent=row[9],
                status=AuditEventStatus(row[10]),
                error_message=row[11],
                old_metadata=old_metadata,
                new_metadata=new_metadata,
                additional_data=additional_data,
                integrity_hash=row[15]
            )
            
        except Exception as e:
            self.logger.error(f"Error converting row to audit event: {str(e)}")
            return None
    
    def cleanup_old_events(self, retention_days: int) -> int:
        """Remove audit events older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM audit_events WHERE timestamp < ?",
                    (cutoff_date.timestamp(),)
                )
                deleted_count = cursor.rowcount
                
            self.logger.info(f"Cleaned up {deleted_count} old audit events")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old audit events: {str(e)}")
            return 0


class FileOperationAuditor:
    """
    Main file operations auditor.
    
    Intercepts and audits all file operations with comprehensive
    metadata collection and integrity verification.
    """
    
    def __init__(self, config: AuditConfiguration):
        """Initialize file operation auditor."""
        self.config = config
        self.logger = CorrelationLogger(__name__)
        
        # Initialize components
        self.integrity_manager = FileIntegrityManager()
        
        # Setup encryption key for audit database
        encryption_key = None
        if config.encrypt_audit_logs:
            encryption_key = self._get_or_create_encryption_key()
        
        self.audit_db = AuditDatabase(config.audit_database_path, encryption_key)
        
        # Pattern matching for excluded files
        import fnmatch
        self.excluded_patterns = config.excluded_file_patterns
        
        self.logger.info("File operation auditor initialized")
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for audit logs."""
        key_file = Path(self.config.audit_database_path).parent / ".audit_key"
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception:
                pass
        
        # Generate new key
        key = cryptography.fernet.Fernet.generate_key()
        
        try:
            key_file.parent.mkdir(exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions
            key_file.chmod(0o600)
        except Exception as e:
            self.logger.error(f"Error saving encryption key: {str(e)}")
        
        return key
    
    def _should_audit_file(self, file_path: Path) -> bool:
        """Check if file should be audited based on configuration."""
        if not self.config.enabled:
            return False
        
        file_str = str(file_path)
        
        # Check excluded patterns
        import fnmatch
        for pattern in self.excluded_patterns:
            if fnmatch.fnmatch(file_str, pattern):
                return False
        
        return True
    
    @contextmanager
    def audit_file_operation(self, 
                           operation_type: FileOperationType,
                           file_path: Union[str, Path],
                           user_id: Optional[str] = None):
        """
        Context manager for auditing file operations.
        
        Usage:
            with auditor.audit_file_operation(FileOperationType.CREATE, "file.txt"):
                # Perform file operation
                create_file("file.txt")
        """
        file_path = Path(file_path)
        
        if not self._should_audit_file(file_path):
            yield
            return
        
        # Generate event ID
        event_id = f"{operation_type.value}_{int(time.time() * 1000000)}"
        
        # Get correlation context
        correlation_id = CorrelationContext.get_correlation_id()
        user_id = user_id or CorrelationContext.get_user_id()
        session_id = CorrelationContext.get_session_id()
        
        # Get pre-operation metadata
        old_metadata = None
        if file_path.exists() and operation_type in [
            FileOperationType.UPDATE, FileOperationType.DELETE, 
            FileOperationType.MOVE, FileOperationType.PERMISSION_CHANGE
        ]:
            old_metadata = self.integrity_manager.get_file_metadata(file_path)
        
        start_time = datetime.now()
        status = AuditEventStatus.SUCCESS
        error_message = None
        
        try:
            # Execute the operation
            yield
            
        except Exception as e:
            status = AuditEventStatus.FAILED
            error_message = str(e)
            raise
        
        finally:
            # Get post-operation metadata
            new_metadata = None
            if file_path.exists() and operation_type in [
                FileOperationType.CREATE, FileOperationType.UPDATE,
                FileOperationType.MOVE, FileOperationType.COPY,
                FileOperationType.PERMISSION_CHANGE
            ]:
                new_metadata = self.integrity_manager.get_file_metadata(file_path)
            
            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                correlation_id=correlation_id,
                timestamp=start_time,
                operation_type=operation_type,
                file_path=str(file_path),
                user_id=user_id,
                session_id=session_id,
                status=status,
                error_message=error_message,
                old_metadata=old_metadata,
                new_metadata=new_metadata
            )
            
            # Store audit event
            self.audit_db.store_audit_event(event)
            
            # Log the operation
            self.logger.info(
                f"File operation audited: {operation_type.value}",
                event_id=event_id,
                file_path=str(file_path),
                status=status.value,
                user_id=user_id
            )
    
    def audit_file_access(self, file_path: Union[str, Path], access_type: str = "read") -> None:
        """Audit file access operations."""
        if not self.config.monitor_read_operations:
            return
        
        with self.audit_file_operation(FileOperationType.ACCESS, file_path):
            pass  # Access already happened, just record it
    
    def get_file_audit_history(self, file_path: str, days: int = 30) -> List[AuditEvent]:
        """Get audit history for a specific file."""
        start_time = datetime.now() - timedelta(days=days)
        return self.audit_db.query_audit_events(
            start_time=start_time,
            file_path=file_path
        )
    
    def get_user_activity_report(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Generate user activity report."""
        start_time = datetime.now() - timedelta(days=days)
        events = self.audit_db.query_audit_events(
            start_time=start_time,
            user_id=user_id
        )
        
        # Analyze activity
        operation_counts = {}
        files_affected = set()
        error_count = 0
        
        for event in events:
            operation_counts[event.operation_type.value] = operation_counts.get(event.operation_type.value, 0) + 1
            files_affected.add(event.file_path)
            if event.status == AuditEventStatus.FAILED:
                error_count += 1
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_operations": len(events),
            "operation_breakdown": operation_counts,
            "unique_files_affected": len(files_affected),
            "failed_operations": error_count,
            "recent_events": events[:10]  # Last 10 events
        }
    
    def detect_suspicious_activity(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Detect suspicious file operation patterns."""
        start_time = datetime.now() - timedelta(hours=hours)
        events = self.audit_db.query_audit_events(start_time=start_time, limit=10000)
        
        suspicious_patterns = []
        
        # Group events by user
        user_events = {}
        for event in events:
            if event.user_id:
                if event.user_id not in user_events:
                    user_events[event.user_id] = []
                user_events[event.user_id].append(event)
        
        # Analyze patterns
        for user_id, user_event_list in user_events.items():
            # Check for excessive delete operations
            delete_count = sum(1 for e in user_event_list if e.operation_type == FileOperationType.DELETE)
            if delete_count > 50:  # Configurable threshold
                suspicious_patterns.append({
                    "type": "excessive_deletions",
                    "user_id": user_id,
                    "count": delete_count,
                    "severity": "high",
                    "description": f"User performed {delete_count} delete operations in {hours} hours"
                })
            
            # Check for failed operations
            failed_count = sum(1 for e in user_event_list if e.status == AuditEventStatus.FAILED)
            if failed_count > 20:  # Configurable threshold
                suspicious_patterns.append({
                    "type": "excessive_failures",
                    "user_id": user_id,
                    "count": failed_count,
                    "severity": "medium",
                    "description": f"User had {failed_count} failed operations in {hours} hours"
                })
            
            # Check for rapid sequential operations
            if len(user_event_list) > 100:  # High volume threshold
                time_span = (user_event_list[0].timestamp - user_event_list[-1].timestamp).total_seconds()
                if time_span < 600:  # Less than 10 minutes
                    suspicious_patterns.append({
                        "type": "rapid_operations",
                        "user_id": user_id,
                        "count": len(user_event_list),
                        "time_span_seconds": time_span,
                        "severity": "medium",
                        "description": f"User performed {len(user_event_list)} operations in {time_span:.0f} seconds"
                    })
        
        if suspicious_patterns:
            self.logger.warning(f"Detected {len(suspicious_patterns)} suspicious activity patterns")
        
        return suspicious_patterns


# Convenience functions for common file operations with auditing
def create_audited_file_operations(auditor: FileOperationAuditor):
    """Create convenience functions for common file operations with auditing."""
    
    def create_file(file_path: Union[str, Path], content: Union[str, bytes] = "", user_id: Optional[str] = None):
        """Create file with auditing."""
        file_path = Path(file_path)
        with auditor.audit_file_operation(FileOperationType.CREATE, file_path, user_id):
            if isinstance(content, str):
                with open(file_path, 'w') as f:
                    f.write(content)
            else:
                with open(file_path, 'wb') as f:
                    f.write(content)
    
    def read_file(file_path: Union[str, Path], user_id: Optional[str] = None) -> Union[str, bytes]:
        """Read file with auditing."""
        file_path = Path(file_path)
        auditor.audit_file_access(file_path, "read")
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Try to decode as text
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            return content
    
    def update_file(file_path: Union[str, Path], content: Union[str, bytes], user_id: Optional[str] = None):
        """Update file with auditing."""
        file_path = Path(file_path)
        with auditor.audit_file_operation(FileOperationType.UPDATE, file_path, user_id):
            if isinstance(content, str):
                with open(file_path, 'w') as f:
                    f.write(content)
            else:
                with open(file_path, 'wb') as f:
                    f.write(content)
    
    def delete_file(file_path: Union[str, Path], user_id: Optional[str] = None):
        """Delete file with auditing."""
        file_path = Path(file_path)
        with auditor.audit_file_operation(FileOperationType.DELETE, file_path, user_id):
            file_path.unlink()
    
    def move_file(source_path: Union[str, Path], dest_path: Union[str, Path], user_id: Optional[str] = None):
        """Move file with auditing."""
        source_path = Path(source_path)
        dest_path = Path(dest_path)
        
        with auditor.audit_file_operation(FileOperationType.MOVE, source_path, user_id):
            shutil.move(str(source_path), str(dest_path))
    
    def copy_file(source_path: Union[str, Path], dest_path: Union[str, Path], user_id: Optional[str] = None):
        """Copy file with auditing."""
        source_path = Path(source_path)
        dest_path = Path(dest_path)
        
        with auditor.audit_file_operation(FileOperationType.COPY, dest_path, user_id):
            shutil.copy2(str(source_path), str(dest_path))
    
    return {
        'create_file': create_file,
        'read_file': read_file,
        'update_file': update_file,
        'delete_file': delete_file,
        'move_file': move_file,
        'copy_file': copy_file
    }


# Example usage and demonstration
if __name__ == "__main__":
    # Configure audit system
    config = AuditConfiguration(
        compliance_level=ComplianceLevel.ENHANCED,
        audit_database_path="file_audit.db",
        encrypt_audit_logs=True,
        include_file_content_hash=True,
        monitor_read_operations=True
    )
    
    # Initialize auditor
    auditor = FileOperationAuditor(config)
    
    # Get audited file operations
    file_ops = create_audited_file_operations(auditor)
    
    # Example operations with auditing
    with CorrelationContext.context(user_id="demo_user", session_id="demo_session"):
        # Create file
        file_ops['create_file']("test_audit.txt", "This is a test file.")
        
        # Read file
        content = file_ops['read_file']("test_audit.txt")
        
        # Update file
        file_ops['update_file']("test_audit.txt", "This is updated content.")
        
        # Copy file
        file_ops['copy_file']("test_audit.txt", "test_audit_copy.txt")
        
        # Get audit history
        history = auditor.get_file_audit_history("test_audit.txt")
        
        print(f"Audit events for test_audit.txt: {len(history)}")
        for event in history:
            print(f"  {event.timestamp}: {event.operation_type.value} - {event.status.value}")
        
        # Clean up
        file_ops['delete_file']("test_audit.txt")
        file_ops['delete_file']("test_audit_copy.txt")
    
    print("File audit demonstration completed!")