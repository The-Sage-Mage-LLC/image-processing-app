#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Configuration Management with Pydantic
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Modern configuration management using Pydantic for type-safe settings
with environment variable support and validation.

Features:
- Type-safe configuration with Pydantic
- Environment variable support with .env files
- Nested configuration models
- Validation and error handling
- Runtime configuration changes
- Configuration export and import
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from functools import lru_cache

from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.types import SecretStr, DirectoryPath, FilePath
import pydantic


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class ImageFormat(str, Enum):
    """Supported image formats."""
    JPEG = "JPEG"
    PNG = "PNG"
    WEBP = "WEBP"
    TIFF = "TIFF"
    BMP = "BMP"


class ProcessingMode(str, Enum):
    """Image processing modes."""
    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"
    REALTIME = "realtime"


class SecuritySettings(BaseSettings):
    """Security-related configuration."""
    
    # Authentication
    secret_key: SecretStr = Field(
        default="change-me-in-production",
        description="Secret key for JWT tokens and encryption"
    )
    
    # Session management
    session_timeout_minutes: int = Field(
        default=480,  # 8 hours
        ge=1,
        le=43200,  # 30 days max
        description="Session timeout in minutes"
    )
    
    session_absolute_timeout_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # 7 days max
        description="Absolute session timeout in hours"
    )
    
    # Password policies
    password_min_length: int = Field(default=12, ge=8, le=128)
    password_require_uppercase: bool = Field(default=True)
    password_require_lowercase: bool = Field(default=True)
    password_require_digits: bool = Field(default=True)
    password_require_special: bool = Field(default=True)
    password_max_age_days: int = Field(default=90, ge=0, le=365)
    
    # Account lockout
    max_failed_login_attempts: int = Field(default=5, ge=1, le=50)
    account_lockout_duration_minutes: int = Field(default=30, ge=1, le=1440)
    
    # Encryption
    encrypt_audit_logs: bool = Field(default=True)
    encrypt_sensitive_data: bool = Field(default=True)
    
    class Config:
        env_prefix = "SECURITY_"


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    # Database connection
    type: DatabaseType = Field(default=DatabaseType.SQLITE)
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    username: str = Field(default="")
    password: SecretStr = Field(default="")
    database_name: str = Field(default="image_processing")
    
    # Connection pool
    pool_size: int = Field(default=5, ge=1, le=100)
    max_overflow: int = Field(default=10, ge=0, le=100)
    pool_timeout: int = Field(default=30, ge=1, le=300)
    pool_recycle: int = Field(default=3600, ge=60, le=86400)
    
    # SQLite specific
    sqlite_path: Optional[Path] = Field(default=Path("data/app.db"))
    sqlite_timeout: float = Field(default=20.0, ge=1.0, le=300.0)
    
    # Performance
    echo: bool = Field(default=False, description="Enable SQL query logging")
    query_cache_size: int = Field(default=100, ge=0, le=10000)
    
    @validator('sqlite_path', pre=True)
    def create_sqlite_directory(cls, v):
        if v:
            path = Path(v)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return v
    
    @property
    def connection_url(self) -> str:
        """Generate database connection URL."""
        if self.type == DatabaseType.SQLITE:
            return f"sqlite:///{self.sqlite_path}"
        elif self.type == DatabaseType.POSTGRESQL:
            password = self.password.get_secret_value() if self.password else ""
            auth = f"{self.username}:{password}@" if self.username else ""
            return f"postgresql://{auth}{self.host}:{self.port}/{self.database_name}"
        elif self.type == DatabaseType.MYSQL:
            password = self.password.get_secret_value() if self.password else ""
            auth = f"{self.username}:{password}@" if self.username else ""
            return f"mysql+pymysql://{auth}{self.host}:{self.port}/{self.database_name}"
        else:
            raise ValueError(f"Unsupported database type: {self.type}")
    
    class Config:
        env_prefix = "DB_"


class ProcessingSettings(BaseSettings):
    """Image processing configuration."""
    
    # Processing modes
    default_mode: ProcessingMode = Field(default=ProcessingMode.SINGLE)
    max_concurrent_jobs: int = Field(default=4, ge=1, le=64)
    batch_size: int = Field(default=100, ge=1, le=10000)
    
    # Image limits
    max_image_size_mb: int = Field(default=100, ge=1, le=1000)
    max_image_width: int = Field(default=8192, ge=64, le=32768)
    max_image_height: int = Field(default=8192, ge=64, le=32768)
    
    # Output settings
    default_output_format: ImageFormat = Field(default=ImageFormat.JPEG)
    default_quality: int = Field(default=85, ge=1, le=100)
    preserve_metadata: bool = Field(default=True)
    
    # Allowed formats
    input_formats: List[ImageFormat] = Field(
        default=[ImageFormat.JPEG, ImageFormat.PNG, ImageFormat.WEBP, ImageFormat.TIFF, ImageFormat.BMP]
    )
    output_formats: List[ImageFormat] = Field(
        default=[ImageFormat.JPEG, ImageFormat.PNG, ImageFormat.WEBP]
    )
    
    # Performance
    use_gpu: bool = Field(default=False)
    gpu_memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0)
    threading_enabled: bool = Field(default=True)
    
    # AI/ML settings
    enable_ai_analysis: bool = Field(default=True)
    ai_model_cache_size: int = Field(default=3, ge=1, le=10)
    ai_confidence_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
    
    class Config:
        env_prefix = "PROCESSING_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    # Metrics collection
    metrics_enabled: bool = Field(default=True)
    metrics_collection_interval: int = Field(default=10, ge=1, le=300)
    metrics_retention_days: int = Field(default=30, ge=1, le=365)
    
    # Performance monitoring
    performance_monitoring: bool = Field(default=True)
    alert_on_high_cpu: bool = Field(default=True)
    cpu_alert_threshold: float = Field(default=80.0, ge=10.0, le=100.0)
    memory_alert_threshold: float = Field(default=85.0, ge=10.0, le=100.0)
    disk_alert_threshold: float = Field(default=90.0, ge=10.0, le=100.0)
    
    # Health checks
    health_check_enabled: bool = Field(default=True)
    health_check_interval: int = Field(default=30, ge=5, le=300)
    health_check_timeout: int = Field(default=10, ge=1, le=60)
    
    # Audit trail
    audit_enabled: bool = Field(default=True)
    audit_retention_days: int = Field(default=365, ge=30, le=2555)  # 7 years max
    monitor_read_operations: bool = Field(default=False)
    
    class Config:
        env_prefix = "MONITORING_"


class APISettings(BaseSettings):
    """API server configuration."""
    
    # Server settings
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1, le=16)
    
    # CORS settings
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://127.0.0.1:3000"])
    cors_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    cors_headers: List[str] = Field(default=["*"])
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100, ge=1, le=10000)
    rate_limit_window: int = Field(default=3600, ge=60, le=86400)  # 1 hour default
    
    # API keys
    api_keys_enabled: bool = Field(default=True)
    api_key_length: int = Field(default=32, ge=16, le=128)
    api_key_expiration_days: int = Field(default=365, ge=1, le=3650)
    
    # Documentation
    docs_enabled: bool = Field(default=True)
    docs_url: str = Field(default="/docs")
    redoc_url: str = Field(default="/redoc")
    
    class Config:
        env_prefix = "API_"


class GUISettings(BaseSettings):
    """GUI application configuration."""
    
    # Window settings
    default_width: int = Field(default=1200, ge=800, le=3840)
    default_height: int = Field(default=800, ge=600, le=2160)
    remember_window_state: bool = Field(default=True)
    
    # Theme settings
    theme: str = Field(default="default", regex=r"^[a-z_]+$")
    dark_mode: bool = Field(default=False)
    language: str = Field(default="en", regex=r"^[a-z]{2}$")
    
    # Performance
    preview_cache_size: int = Field(default=50, ge=10, le=500)
    thumbnail_size: int = Field(default=128, ge=64, le=512)
    max_recent_files: int = Field(default=20, ge=5, le=100)
    
    # Features
    auto_save: bool = Field(default=True)
    auto_backup: bool = Field(default=True)
    backup_interval_minutes: int = Field(default=5, ge=1, le=60)
    
    class Config:
        env_prefix = "GUI_"


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    # Basic settings
    level: LogLevel = Field(default=LogLevel.INFO)
    structured_logging: bool = Field(default=True)
    correlation_enabled: bool = Field(default=True)
    
    # File logging
    file_logging: bool = Field(default=True)
    log_file: Path = Field(default=Path("logs/app.log"))
    max_file_size: int = Field(default=10_485_760, ge=1024)  # 10MB default
    backup_count: int = Field(default=5, ge=1, le=100)
    
    # Console logging
    console_logging: bool = Field(default=True)
    console_format: str = Field(default="pretty")
    
    # Log rotation
    rotation_enabled: bool = Field(default=True)
    rotation_interval: str = Field(default="midnight", regex=r"^(midnight|W[0-6]|H|M|S)$")
    
    # Performance logging
    performance_logging: bool = Field(default=True)
    slow_query_threshold: float = Field(default=1.0, ge=0.1, le=10.0)
    
    @validator('log_file', pre=True)
    def create_log_directory(cls, v):
        if v:
            path = Path(v)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return v
    
    class Config:
        env_prefix = "LOG_"


class StorageSettings(BaseSettings):
    """Storage and file management configuration."""
    
    # Directories
    input_directory: Path = Field(default=Path("input"))
    output_directory: Path = Field(default=Path("output"))
    temp_directory: Path = Field(default=Path("temp"))
    cache_directory: Path = Field(default=Path("cache"))
    
    # File management
    auto_cleanup_temp: bool = Field(default=True)
    temp_file_max_age_hours: int = Field(default=24, ge=1, le=168)
    max_disk_usage_percent: float = Field(default=90.0, ge=50.0, le=99.0)
    
    # Backup settings
    backup_enabled: bool = Field(default=True)
    backup_directory: Path = Field(default=Path("backups"))
    backup_retention_days: int = Field(default=30, ge=1, le=365)
    
    # Cache settings
    cache_enabled: bool = Field(default=True)
    cache_max_size_mb: int = Field(default=1000, ge=100, le=10000)
    cache_expiry_hours: int = Field(default=24, ge=1, le=168)
    
    @root_validator
    def create_directories(cls, values):
        """Create all configured directories."""
        for key, value in values.items():
            if key.endswith('_directory') and isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)
        return values
    
    class Config:
        env_prefix = "STORAGE_"


class AppSettings(BaseSettings):
    """Main application configuration."""
    
    # Application metadata
    app_name: str = Field(default="Image Processing App")
    app_version: str = Field(default="1.0.0")
    app_description: str = Field(default="Enterprise-grade image processing application")
    
    # Environment
    environment: str = Field(default="development", regex=r"^(development|testing|staging|production)$")
    debug: bool = Field(default=True)
    testing: bool = Field(default=False)
    
    # Nested configurations
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    api: APISettings = Field(default_factory=APISettings)
    gui: GUISettings = Field(default_factory=GUISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    
    @validator('debug')
    def debug_only_in_dev(cls, v, values):
        """Disable debug in production."""
        env = values.get('environment')
        if env == 'production' and v:
            raise ValueError("Debug mode cannot be enabled in production")
        return v
    
    @validator('security', pre=False, always=True)
    def validate_production_security(cls, v, values):
        """Validate security settings for production."""
        env = values.get('environment')
        if env == 'production':
            if v.secret_key.get_secret_value() == "change-me-in-production":
                raise ValueError("Secret key must be changed in production")
        return v
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == "testing" or self.testing
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
        extra = "forbid"


# Global settings instance
_settings: Optional[AppSettings] = None


@lru_cache(maxsize=1)
def get_settings(env_file: Optional[str] = None) -> AppSettings:
    """
    Get application settings singleton.
    
    Args:
        env_file: Optional path to environment file
        
    Returns:
        Application settings instance
    """
    global _settings
    
    if _settings is None:
        env_file_path = env_file or ".env"
        if os.path.exists(env_file_path):
            _settings = AppSettings(_env_file=env_file_path)
        else:
            _settings = AppSettings()
    
    return _settings


def reload_settings(env_file: Optional[str] = None) -> AppSettings:
    """
    Reload settings from environment.
    
    Args:
        env_file: Optional path to environment file
        
    Returns:
        Reloaded settings instance
    """
    global _settings
    
    # Clear the cache
    get_settings.cache_clear()
    _settings = None
    
    return get_settings(env_file)


def export_settings(settings: AppSettings, file_path: Path) -> None:
    """
    Export settings to JSON file.
    
    Args:
        settings: Settings instance to export
        file_path: Output file path
    """
    import json
    
    # Create export data (excluding secrets)
    export_data = settings.dict()
    
    # Remove secret values
    def remove_secrets(obj):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if isinstance(value, dict):
                    result[key] = remove_secrets(value)
                elif key.endswith('password') or key.endswith('secret_key'):
                    result[key] = "***HIDDEN***"
                else:
                    result[key] = value
            return result
        return obj
    
    safe_data = remove_secrets(export_data)
    
    with open(file_path, 'w') as f:
        json.dump(safe_data, f, indent=2, default=str)


def create_env_template(file_path: Path) -> None:
    """
    Create a template .env file with all available settings.
    
    Args:
        file_path: Output file path for template
    """
    template_content = """# Image Processing App Configuration
# Copy this file to .env and customize the values

# Application Settings
APP_NAME="Image Processing App"
ENVIRONMENT=development
DEBUG=true

# Security Settings
SECURITY_SECRET_KEY=change-me-in-production
SECURITY_SESSION_TIMEOUT_MINUTES=480
SECURITY_PASSWORD_MIN_LENGTH=12
SECURITY_MAX_FAILED_LOGIN_ATTEMPTS=5

# Database Settings
DB_TYPE=sqlite
DB_SQLITE_PATH=data/app.db
# DB_HOST=localhost
# DB_PORT=5432
# DB_USERNAME=
# DB_PASSWORD=
# DB_DATABASE_NAME=image_processing

# Processing Settings
PROCESSING_MAX_CONCURRENT_JOBS=4
PROCESSING_MAX_IMAGE_SIZE_MB=100
PROCESSING_DEFAULT_QUALITY=85
PROCESSING_ENABLE_AI_ANALYSIS=true

# Monitoring Settings
MONITORING_METRICS_ENABLED=true
MONITORING_CPU_ALERT_THRESHOLD=80.0
MONITORING_MEMORY_ALERT_THRESHOLD=85.0
MONITORING_AUDIT_ENABLED=true

# API Settings
API_HOST=127.0.0.1
API_PORT=8000
API_RATE_LIMIT_ENABLED=true
API_RATE_LIMIT_REQUESTS=100

# GUI Settings
GUI_DEFAULT_WIDTH=1200
GUI_DEFAULT_HEIGHT=800
GUI_DARK_MODE=false
GUI_AUTO_SAVE=true

# Logging Settings
LOG_LEVEL=INFO
LOG_STRUCTURED_LOGGING=true
LOG_FILE_LOGGING=true
LOG_CONSOLE_LOGGING=true

# Storage Settings
STORAGE_INPUT_DIRECTORY=input
STORAGE_OUTPUT_DIRECTORY=output
STORAGE_BACKUP_ENABLED=true
STORAGE_CACHE_ENABLED=true
"""
    
    with open(file_path, 'w') as f:
        f.write(template_content)


# Usage examples and utilities
def validate_configuration() -> bool:
    """
    Validate current configuration.
    
    Returns:
        True if configuration is valid
    """
    try:
        settings = get_settings()
        
        # Additional validation logic
        if settings.is_production():
            # Production-specific validations
            if not settings.security.encrypt_audit_logs:
                raise ValueError("Audit log encryption must be enabled in production")
            
            if settings.debug:
                raise ValueError("Debug mode must be disabled in production")
        
        return True
        
    except pydantic.ValidationError as e:
        print(f"Configuration validation failed: {e}")
        return False
    except Exception as e:
        print(f"Configuration error: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("?? Modern Configuration Management Demo")
    print("=" * 50)
    
    # Load settings
    settings = get_settings()
    
    print(f"App Name: {settings.app_name}")
    print(f"Environment: {settings.environment}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Database Type: {settings.database.type}")
    print(f"Max Image Size: {settings.processing.max_image_size_mb}MB")
    print(f"API Host: {settings.api.host}:{settings.api.port}")
    
    # Validate configuration
    is_valid = validate_configuration()
    print(f"Configuration Valid: {is_valid}")
    
    # Create .env template
    template_path = Path(".env.template")
    create_env_template(template_path)
    print(f"Template created: {template_path}")
    
    # Export current settings (without secrets)
    export_path = Path("current_settings.json")
    export_settings(settings, export_path)
    print(f"Settings exported: {export_path}")