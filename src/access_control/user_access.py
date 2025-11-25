#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Access Control System
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive role-based access control (RBAC) system with authentication,
authorization, session management, and security policy enforcement.

Features:
- Role-based access control with hierarchical permissions
- Multi-factor authentication support
- Session management with security controls
- Policy-based authorization engine
- User activity monitoring and audit integration
- Password security and compliance enforcement
"""

import uuid
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import bcrypt
from pathlib import Path
import time

# Import our structured logging and audit systems
from ..utils.structured_logging import CorrelationLogger, with_correlation, CorrelationContext


class UserRole(Enum):
    """Predefined user roles with hierarchical structure."""
    GUEST = "guest"                    # View-only access
    USER = "user"                      # Basic user operations
    OPERATOR = "operator"              # Advanced operations
    ADMINISTRATOR = "administrator"    # System administration
    SUPER_ADMIN = "super_admin"       # Full system control


class Permission(Enum):
    """System permissions for granular access control."""
    # File operations
    FILE_READ = "file.read"
    FILE_WRITE = "file.write"
    FILE_DELETE = "file.delete"
    FILE_EXECUTE = "file.execute"
    
    # Image processing
    IMAGE_PROCESS = "image.process"
    IMAGE_BATCH = "image.batch"
    IMAGE_ADVANCED = "image.advanced"
    
    # System operations
    SYSTEM_MONITOR = "system.monitor"
    SYSTEM_CONFIG = "system.config"
    SYSTEM_ADMIN = "system.admin"
    
    # User management
    USER_VIEW = "user.view"
    USER_CREATE = "user.create"
    USER_MODIFY = "user.modify"
    USER_DELETE = "user.delete"
    
    # Security
    SECURITY_AUDIT = "security.audit"
    SECURITY_CONFIG = "security.config"
    
    # API access
    API_READ = "api.read"
    API_WRITE = "api.write"
    API_ADMIN = "api.admin"


class AuthenticationMethod(Enum):
    """Supported authentication methods."""
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"


class SessionStatus(Enum):
    """User session status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    LOCKED = "locked"


@dataclass
class User:
    """User account representation."""
    user_id: str
    username: str
    email: str
    full_name: str
    role: UserRole
    password_hash: str
    salt: str
    is_active: bool = True
    is_locked: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    password_changed_at: datetime = field(default_factory=datetime.now)
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    api_key: Optional[str] = None
    custom_permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserSession:
    """User session representation."""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    auth_methods: List[AuthenticationMethod] = field(default_factory=list)
    permissions: Set[Permission] = field(default_factory=set)


@dataclass
class AccessPolicy:
    """Access control policy definition."""
    policy_id: str
    name: str
    description: str
    conditions: Dict[str, Any]  # JSON-serializable conditions
    permissions: Set[Permission]
    priority: int = 100
    is_active: bool = True


@dataclass
class SecurityConfiguration:
    """Security configuration for access control."""
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_digits: bool = True
    password_require_special: bool = True
    password_max_age_days: int = 90
    password_history_count: int = 5
    max_failed_login_attempts: int = 5
    account_lockout_duration_minutes: int = 30
    session_timeout_minutes: int = 480  # 8 hours
    session_absolute_timeout_hours: int = 24
    require_mfa_for_admin: bool = True
    api_key_expiration_days: int = 365
    enforce_password_complexity: bool = True
    log_all_access_attempts: bool = True


class RolePermissionManager:
    """
    Manages role-based permissions with hierarchical inheritance.
    
    Provides a flexible permission system where roles can inherit
    permissions from parent roles and have custom permissions added.
    """
    
    def __init__(self):
        """Initialize role permission manager."""
        self.role_hierarchy = {
            UserRole.SUPER_ADMIN: None,  # Top level
            UserRole.ADMINISTRATOR: UserRole.SUPER_ADMIN,
            UserRole.OPERATOR: UserRole.ADMINISTRATOR,
            UserRole.USER: UserRole.OPERATOR,
            UserRole.GUEST: UserRole.USER
        }
        
        # Define base permissions for each role
        self.role_permissions = {
            UserRole.GUEST: {
                Permission.FILE_READ,
                Permission.IMAGE_PROCESS,
                Permission.API_READ
            },
            UserRole.USER: {
                Permission.FILE_WRITE,
                Permission.IMAGE_BATCH,
                Permission.API_WRITE
            },
            UserRole.OPERATOR: {
                Permission.FILE_DELETE,
                Permission.IMAGE_ADVANCED,
                Permission.SYSTEM_MONITOR,
                Permission.USER_VIEW
            },
            UserRole.ADMINISTRATOR: {
                Permission.SYSTEM_CONFIG,
                Permission.USER_CREATE,
                Permission.USER_MODIFY,
                Permission.SECURITY_AUDIT,
                Permission.API_ADMIN
            },
            UserRole.SUPER_ADMIN: {
                Permission.SYSTEM_ADMIN,
                Permission.USER_DELETE,
                Permission.SECURITY_CONFIG,
                Permission.FILE_EXECUTE
            }
        }
    
    def get_role_permissions(self, role: UserRole, include_inherited: bool = True) -> Set[Permission]:
        """Get all permissions for a role, including inherited permissions."""
        permissions = set(self.role_permissions.get(role, set()))
        
        if include_inherited:
            # Add inherited permissions from parent roles
            current_role = role
            while current_role in self.role_hierarchy:
                parent_role = self.role_hierarchy[current_role]
                if parent_role:
                    permissions.update(self.role_permissions.get(parent_role, set()))
                    current_role = parent_role
                else:
                    break
        
        return permissions
    
    def has_permission(self, role: UserRole, permission: Permission, 
                      custom_permissions: Optional[Set[Permission]] = None) -> bool:
        """Check if role has specific permission."""
        role_permissions = self.get_role_permissions(role)
        
        if permission in role_permissions:
            return True
        
        if custom_permissions and permission in custom_permissions:
            return True
        
        return False
    
    def is_role_higher_or_equal(self, role1: UserRole, role2: UserRole) -> bool:
        """Check if role1 is higher than or equal to role2 in hierarchy."""
        role1_permissions = self.get_role_permissions(role1)
        role2_permissions = self.get_role_permissions(role2)
        
        return role2_permissions.issubset(role1_permissions)


class PasswordManager:
    """
    Secure password management with complexity validation.
    
    Implements secure password hashing, complexity validation,
    and history tracking to prevent password reuse.
    """
    
    def __init__(self, config: SecurityConfiguration):
        """Initialize password manager."""
        self.config = config
        self.logger = CorrelationLogger(__name__)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        """
        Hash password using bcrypt with salt.
        
        Args:
            password: Plain text password
            salt: Optional salt (generates new if not provided)
            
        Returns:
            Tuple of (password_hash, salt_hex)
        """
        if salt is None:
            salt = bcrypt.gensalt()
        
        password_hash = bcrypt.hashpw(password.encode(), salt)
        
        return password_hash.hex(), salt.hex()
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against stored hash and salt."""
        try:
            stored_hash = bytes.fromhex(password_hash)
            # Salt bytes not needed for bcrypt.checkpw with stored hash
            
            return bcrypt.checkpw(password.encode(), stored_hash)
        except Exception as e:
            self.logger.error(f"Error verifying password: {str(e)}")
            return False
    
    def validate_password_complexity(self, password: str) -> tuple[bool, List[str]]:
        """
        Validate password complexity against security policy.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Length check
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        if not self.config.enforce_password_complexity:
            return len(errors) == 0, errors
        
        # Character composition checks
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)
        
        if self.config.password_require_uppercase and not has_upper:
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_lowercase and not has_lower:
            errors.append("Password must contain at least one lowercase letter")
        
        if self.config.password_require_digits and not has_digit:
            errors.append("Password must contain at least one digit")
        
        if self.config.password_require_special and not has_special:
            errors.append("Password must contain at least one special character")
        
        # Additional security checks
        if password.lower() in ['password', '123456', 'admin', 'user']:
            errors.append("Password cannot be a common weak password")
        
        return len(errors) == 0, errors
    
    def is_password_expired(self, password_changed_at: datetime) -> bool:
        """Check if password has expired based on security policy."""
        if self.config.password_max_age_days <= 0:
            return False
        
        expiry_date = password_changed_at + timedelta(days=self.config.password_max_age_days)
        return datetime.now() > expiry_date


class SessionManager:
    """
    Manages user sessions with security controls.
    
    Provides secure session creation, validation, and cleanup
    with configurable timeout and security policies.
    """
    
    def __init__(self, config: SecurityConfiguration):
        """Initialize session manager."""
        self.config = config
        self.sessions: Dict[str, UserSession] = {}
        self.session_lock = threading.RLock()
        self.logger = CorrelationLogger(__name__)
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self.cleanup_thread.start()
    
    def create_session(self, user: User, ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None,
                      auth_methods: List[AuthenticationMethod] = None) -> UserSession:
        """Create new user session."""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Calculate expiration times
        session_timeout = timedelta(minutes=self.config.session_timeout_minutes)
        absolute_timeout = timedelta(hours=self.config.session_absolute_timeout_hours)
        
        expires_at = min(
            now + session_timeout,
            now + absolute_timeout
        )
        
        # Get user permissions
        role_manager = RolePermissionManager()
        permissions = role_manager.get_role_permissions(user.role)
        permissions.update(user.custom_permissions)
        
        session = UserSession(
            session_id=session_id,
            user_id=user.user_id,
            created_at=now,
            last_activity=now,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            auth_methods=auth_methods or [AuthenticationMethod.PASSWORD],
            permissions=permissions
        )
        
        with self.session_lock:
            self.sessions[session_id] = session
        
        self.logger.info(f"Created session for user {user.username}",
                        session_id=session_id, user_id=user.user_id)
        
        return session
    
    def validate_session(self, session_id: str) -> Optional[UserSession]:
        """Validate and refresh session if valid."""
        with self.session_lock:
            session = self.sessions.get(session_id)
            
            def _is_session_expired(session: UserSession, now: datetime) -> bool:
                # Check expiration
                if session.expires_at <= now:
                    session.status = SessionStatus.EXPIRED
                    return True
                
                # Check status
                if session.status != SessionStatus.ACTIVE:
                    return True
                
                return False
            
            if not session or _is_session_expired(session, datetime.now()):
                return None
            
            # Update last activity and extend session
            session.last_activity = datetime.now()
            
            # Extend session timeout (sliding window)
            session_timeout = timedelta(minutes=self.config.session_timeout_minutes)
            new_expires_at = session.last_activity + session_timeout
            
            # Don't extend beyond absolute timeout
            absolute_limit = session.created_at + timedelta(hours=self.config.session_absolute_timeout_hours)
            session.expires_at = min(new_expires_at, absolute_limit)
            
            return session
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate specific session."""
        with self.session_lock:
            session = self.sessions.get(session_id)
            if session:
                session.status = SessionStatus.TERMINATED
                self.logger.info(f"Terminated session {session_id}")
                return True
            return False
    
    def terminate_user_sessions(self, user_id: str, exclude_session: Optional[str] = None) -> int:
        """Terminate all sessions for a user."""
        terminated_count = 0
        
        with self.session_lock:
            for session_id, session in self.sessions.items():
                if (session.user_id == user_id and 
                    session_id != exclude_session and
                    session.status == SessionStatus.ACTIVE):
                    session.status = SessionStatus.TERMINATED
                    terminated_count += 1
        
        if terminated_count > 0:
            self.logger.info(f"Terminated {terminated_count} sessions for user {user_id}")
        
        return terminated_count
    
    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """Get active sessions for a user."""
        with self.session_lock:
            return [
                session for session in self.sessions.values()
                if session.user_id == user_id and session.status == SessionStatus.ACTIVE
            ]
    
    def _cleanup_expired_sessions(self) -> None:
        """Background thread to clean up expired sessions."""
        while True:
            try:
                now = datetime.now()
                expired_sessions = []
                
                with self.session_lock:
                    for session_id, session in self.sessions.items():
                        if session.expires_at <= now and session.status == SessionStatus.ACTIVE:
                            session.status = SessionStatus.EXPIRED
                            expired_sessions.append(session_id)
                
                if expired_sessions:
                    self.logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                # Sleep for 5 minutes before next cleanup
                time.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {str(e)}")
                time.sleep(60)  # Shorter sleep on error


class UserAccessControl:
    """
    Main user access control system.
    
    Provides comprehensive user authentication, authorization,
    and access control with enterprise-grade security features.
    """
    
    def __init__(self, db_path: str = "users.db", config: Optional[SecurityConfiguration] = None):
        """Initialize user access control system."""
        self.db_path = Path(db_path)
        self.config = config or SecurityConfiguration()
        self.logger = CorrelationLogger(__name__)
        
        # Initialize components
        self.password_manager = PasswordManager(self.config)
        self.session_manager = SessionManager(self.config)
        self.role_manager = RolePermissionManager()
        
        # Database lock
        self._db_lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Create default admin user if no users exist
        self._create_default_admin()
        
        self.logger.info("User access control system initialized")
    
    def _init_database(self) -> None:
        """Initialize user database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    is_locked BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME,
                    failed_login_attempts INTEGER DEFAULT 0,
                    password_changed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    mfa_enabled BOOLEAN DEFAULT 0,
                    mfa_secret TEXT,
                    api_key TEXT,
                    custom_permissions TEXT,  -- JSON
                    metadata TEXT  -- JSON
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS password_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_policies (
                    policy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    conditions TEXT,  -- JSON
                    permissions TEXT,  -- JSON
                    priority INTEGER DEFAULT 100,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _create_default_admin(self) -> None:
        """Create default admin user if no users exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                
                if user_count == 0:
                    # Create default admin
                    default_password = "Admin123!"
                    password_hash, salt = self.password_manager.hash_password(default_password)
                    
                    admin_user = User(
                        user_id=str(uuid.uuid4()),
                        username="admin",
                        email="admin@imageprocessing.local",
                        full_name="System Administrator",
                        role=UserRole.SUPER_ADMIN,
                        password_hash=password_hash,
                        salt=salt
                    )
                    
                    self._store_user(admin_user)
                    
                    self.logger.warning(
                        "Created default admin user",
                        username="admin",
                        password="Admin123!",
                        message="CHANGE DEFAULT PASSWORD IMMEDIATELY"
                    )
        
        except Exception as e:
            self.logger.error(f"Error creating default admin user: {str(e)}")
    
    @with_correlation("auth.authenticate_user")
    def authenticate_user(self, username: str, password: str, 
                         ip_address: Optional[str] = None,
                         user_agent: Optional[str] = None) -> Optional[UserSession]:
        """
        Authenticate user and create session.
        
        Args:
            username: Username or email
            password: User password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            User session if authentication successful, None otherwise
        """
        user = self.get_user_by_username(username)
        if not user:
            user = self.get_user_by_email(username)
        
        if not user:
            self.logger.warning(f"Authentication failed: user not found", username=username)
            return None
        
        # Check if account is active
        if not user.is_active:
            self.logger.warning(f"Authentication failed: account disabled", username=username)
            return None
        
        # Check if account is locked
        if user.is_locked:
            self.logger.warning(f"Authentication failed: account locked", username=username)
            return None
        
        # Check failed attempts lockout
        if user.failed_login_attempts >= self.config.max_failed_login_attempts:
            # Check if lockout period has expired
            lockout_duration = timedelta(minutes=self.config.account_lockout_duration_minutes)
            if user.last_login and (datetime.now() - user.last_login) < lockout_duration:
                self.logger.warning(f"Authentication failed: account temporarily locked", username=username)
                return None
            else:
                # Reset failed attempts after lockout period
                user.failed_login_attempts = 0
                self._store_user(user)
        
        # Verify password
        if not self.password_manager.verify_password(password, user.password_hash, user.salt):
            # Increment failed login attempts
            user.failed_login_attempts += 1
            self._store_user(user)
            
            self.logger.warning(f"Authentication failed: invalid password", 
                               username=username,
                               failed_attempts=user.failed_login_attempts)
            return None
        
        # Check password expiration
        if self.password_manager.is_password_expired(user.password_changed_at):
            self.logger.warning(f"Authentication failed: password expired", username=username)
            return None
        
        # Reset failed login attempts on successful authentication
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        self._store_user(user)
        
        # Create session
        session = self.session_manager.create_session(
            user, ip_address, user_agent, [AuthenticationMethod.PASSWORD]
        )
        
        self.logger.info(f"User authenticated successfully", 
                        username=username, 
                        session_id=session.session_id)
        
        return session
    
    def authorize_operation(self, session_id: str, required_permission: Permission,
                          resource_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Authorize user operation based on permissions.
        
        Args:
            session_id: User session ID
            required_permission: Required permission
            resource_context: Optional context about the resource
            
        Returns:
            True if operation is authorized
        """
        session = self.session_manager.validate_session(session_id)
        if not session:
            self.logger.warning(f"Authorization failed: invalid session", session_id=session_id)
            return False
        
        # Check if user has required permission
        if required_permission in session.permissions:
            self.logger.debug(f"Operation authorized", 
                             user_id=session.user_id,
                             permission=required_permission.value)
            return True
        
        self.logger.warning(f"Authorization failed: insufficient permissions",
                           user_id=session.user_id,
                           required_permission=required_permission.value)
        return False
    
    def create_user(self, username: str, email: str, full_name: str,
                   password: str, role: UserRole,
                   creator_session_id: str) -> Optional[User]:
        """Create new user account."""
        # Authorize creation
        if not self.authorize_operation(creator_session_id, Permission.USER_CREATE):
            return None
        
        # Validate password
        is_valid, errors = self.password_manager.validate_password_complexity(password)
        if not is_valid:
            self.logger.warning(f"User creation failed: password validation", errors=errors)
            return None
        
        # Check if username/email already exists
        if self.get_user_by_username(username) or self.get_user_by_email(email):
            self.logger.warning(f"User creation failed: username or email already exists")
            return None
        
        # Create user
        password_hash, salt = self.password_manager.hash_password(password)
        
        user = User(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            full_name=full_name,
            role=role,
            password_hash=password_hash,
            salt=salt
        )
        
        if self._store_user(user):
            self.logger.info(f"User created successfully", username=username, role=role.value)
            return user
        
        return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            )
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            )
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None
    
    def _store_user(self, user: User) -> bool:
        """Store user in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO users 
                    (user_id, username, email, full_name, role, password_hash, salt,
                     is_active, is_locked, created_at, last_login, failed_login_attempts,
                     password_changed_at, mfa_enabled, mfa_secret, api_key, 
                     custom_permissions, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user.user_id, user.username, user.email, user.full_name,
                    user.role.value, user.password_hash, user.salt, user.is_active,
                    user.is_locked, user.created_at, user.last_login,
                    user.failed_login_attempts, user.password_changed_at,
                    user.mfa_enabled, user.mfa_secret, user.api_key,
                    json.dumps([p.value for p in user.custom_permissions]),
                    json.dumps(user.metadata)
                ))
            return True
        except Exception as e:
            self.logger.error(f"Error storing user: {str(e)}")
            return False
    
    def _row_to_user(self, row) -> User:
        """Convert database row to User object."""
        custom_permissions = set()
        if row[16]:  # custom_permissions
            perm_list = json.loads(row[16])
            custom_permissions = {Permission(p) for p in perm_list}
        
        metadata = {}
        if row[17]:  # metadata
            metadata = json.loads(row[17])
        
        return User(
            user_id=row[0],
            username=row[1],
            email=row[2],
            full_name=row[3],
            role=UserRole(row[4]),
            password_hash=row[5],
            salt=row[6],
            is_active=bool(row[7]),
            is_locked=bool(row[8]),
            created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.now(),
            last_login=datetime.fromisoformat(row[10]) if row[10] else None,
            failed_login_attempts=row[11],
            password_changed_at=datetime.fromisoformat(row[12]) if row[12] else datetime.now(),
            mfa_enabled=bool(row[13]),
            mfa_secret=row[14],
            api_key=row[15],
            custom_permissions=custom_permissions,
            metadata=metadata
        )


# Decorator for access control
def require_permission(permission: Permission):
    """Decorator to require specific permission for function access."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get session ID from correlation context or kwargs
            session_id = kwargs.get('session_id') or CorrelationContext.get_session_id()
            
            if not session_id:
                raise PermissionError("No active session found")
            
            # Get access control instance (assuming it's available globally or in context)
            # In a real implementation, this would be dependency injected
            access_control = getattr(wrapper, '_access_control', None)
            if not access_control:
                raise RuntimeError("Access control not configured")
            
            if not access_control.authorize_operation(session_id, permission):
                raise PermissionError(f"Insufficient permissions: {permission.value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize access control system
    config = SecurityConfiguration(
        password_min_length=8,  # Relaxed for demo
        enforce_password_complexity=True,
        max_failed_login_attempts=3,
        session_timeout_minutes=60
    )
    
    access_control = UserAccessControl("demo_users.db", config)
    
    # Create test user
    admin_session = access_control.authenticate_user("admin", "Admin123!")
    if admin_session:
        print(f"Admin authenticated: {admin_session.session_id}")
        
        # Create test user
        test_user = access_control.create_user(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            password="TestPass123!",
            role=UserRole.USER,
            creator_session_id=admin_session.session_id
        )
        
        if test_user:
            print(f"Test user created: {test_user.username}")
            
            # Authenticate test user
            user_session = access_control.authenticate_user("testuser", "TestPass123!")
            if user_session:
                print(f"Test user authenticated: {user_session.session_id}")
                
                # Test authorization
                can_read = access_control.authorize_operation(user_session.session_id, Permission.FILE_READ)
                can_admin = access_control.authorize_operation(user_session.session_id, Permission.SYSTEM_ADMIN)
                
                print(f"Can read files: {can_read}")
                print(f"Can admin system: {can_admin}")
    
    print("Access control demonstration completed!")    print("Access control demonstration completed!")