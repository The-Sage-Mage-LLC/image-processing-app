#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured Logging System with Correlation IDs
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive structured logging framework with correlation ID tracking
for enterprise-grade debugging, monitoring, and audit trail capabilities.

Features:
- Correlation ID generation and propagation
- Thread-safe context management
- JSON structured logging format
- Performance metrics integration
- Audit trail capabilities
- Multi-level logging configuration
- Automatic correlation across async operations
"""

import uuid
import json
import logging
import logging.handlers
import contextvars
import threading
import time
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import traceback
import functools
from contextlib import contextmanager


class LogLevel(Enum):
    """Standard logging levels with enterprise extensions."""
    TRACE = 5        # Very detailed diagnostic information
    DEBUG = 10       # Detailed information for diagnosing problems
    INFO = 20        # General information about program execution
    WARNING = 30     # Something unexpected happened, but still working
    ERROR = 40       # Serious problem occurred
    CRITICAL = 50    # Very serious error occurred
    AUDIT = 60       # Security and compliance audit events


class CorrelationContext:
    """
    Thread-safe correlation context manager using contextvars.
    
    This class manages correlation IDs across the entire application lifecycle,
    ensuring that all related operations can be traced through log entries.
    
    Features:
    - Automatic correlation ID generation
    - Thread-safe context propagation
    - Nested operation support with parent-child relationships
    - Context inheritance across async operations
    - Performance optimized with minimal overhead
    
    Technical Implementation:
    - Uses Python's contextvars for thread-safe storage
    - Supports context copying for async/threading scenarios
    - Minimal memory footprint with automatic cleanup
    - Compatible with asyncio and threading patterns
    """
    
    # Context variables for thread-safe storage
    _correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
        'correlation_id', default=None
    )
    
    _user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
        'user_id', default=None
    )
    
    _session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
        'session_id', default=None
    )
    
    _operation_stack: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar(
        'operation_stack', default=None
    )
    
    _parent_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
        'parent_correlation_id', default=None
    )
    
    _start_time: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
        'start_time', default=None
    )
    
    @classmethod
    def generate_correlation_id(cls) -> str:
        """
        Generate a new correlation ID using UUID4.
        
        Format: 8-4-4-4-12 hexadecimal digits
        Example: 550e8400-e29b-41d4-a716-446655440000
        
        Returns:
            Unique correlation ID string
        """
        return str(uuid.uuid4())
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set the current correlation ID."""
        cls._correlation_id.set(correlation_id)
        if cls._start_time.get() is None:
            cls._start_time.set(time.time())
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get the current correlation ID."""
        return cls._correlation_id.get()
    
    @classmethod
    def set_user_id(cls, user_id: str) -> None:
        """Set the current user ID for audit purposes."""
        cls._user_id.set(user_id)
    
    @classmethod
    def get_user_id(cls) -> Optional[str]:
        """Get the current user ID."""
        return cls._user_id.get()
    
    @classmethod
    def set_session_id(cls, session_id: str) -> None:
        """Set the current session ID."""
        cls._session_id.set(session_id)
    
    @classmethod
    def get_session_id(cls) -> Optional[str]:
        """Get the current session ID."""
        return cls._session_id.get()
    
    @classmethod
    def push_operation(cls, operation_name: str) -> None:
        """
        Push an operation onto the stack for nested operation tracking.
        
        Args:
            operation_name: Name of the operation (e.g., 'image_processing.resize')
        """
        current_stack = cls._operation_stack.get()
        if current_stack is None:
            current_stack = []
        new_stack = current_stack.copy()
        new_stack.append(operation_name)
        cls._operation_stack.set(new_stack)
    
    @classmethod
    def pop_operation(cls) -> Optional[str]:
        """
        Pop the most recent operation from the stack.
        
        Returns:
            The popped operation name, or None if stack is empty
        """
        current_stack = cls._operation_stack.get()
        if not current_stack:
            return None
        
        new_stack = current_stack.copy()
        operation = new_stack.pop()
        cls._operation_stack.set(new_stack)
        return operation
    
    @classmethod
    def get_operation_stack(cls) -> List[str]:
        """Get the current operation stack."""
        current_stack = cls._operation_stack.get()
        return current_stack or []
    
    @classmethod
    def get_current_operation(cls) -> Optional[str]:
        """Get the current (top) operation name."""
        stack = cls.get_operation_stack()
        return stack[-1] if stack else None
    
    @classmethod
    def set_parent_correlation_id(cls, parent_id: str) -> None:
        """Set parent correlation ID for hierarchical tracing."""
        cls._parent_correlation_id.set(parent_id)
    
    @classmethod
    def get_parent_correlation_id(cls) -> Optional[str]:
        """Get parent correlation ID."""
        return cls._parent_correlation_id.get()
    
    @classmethod
    def get_operation_duration_ms(cls) -> Optional[float]:
        """Get duration since correlation context started (in milliseconds)."""
        start_time = cls._start_time.get()
        if start_time is None:
            return None
        return (time.time() - start_time) * 1000
    
    @classmethod
    def clear_context(cls) -> None:
        """Clear all correlation context variables."""
        cls._correlation_id.set(None)
        cls._user_id.set(None)
        cls._session_id.set(None)
        cls._operation_stack.set(None)
        cls._parent_correlation_id.set(None)
        cls._start_time.set(None)
    
    @classmethod
    @contextmanager
    def context(cls, correlation_id: Optional[str] = None, 
               user_id: Optional[str] = None, 
               session_id: Optional[str] = None):
        """
        Context manager for correlation ID management.
        
        Usage:
            with CorrelationContext.context(correlation_id="abc-123"):
                logger.info("This will include correlation ID")
        
        Args:
            correlation_id: Explicit correlation ID (generates if None)
            user_id: User ID for audit purposes
            session_id: Session ID for user session tracking
        """
        # Generate correlation ID if not provided
        if correlation_id is None:
            correlation_id = cls.generate_correlation_id()
        
        # Store previous context for restoration
        prev_correlation = cls.get_correlation_id()
        prev_user = cls.get_user_id()
        prev_session = cls.get_session_id()
        prev_stack = cls.get_operation_stack().copy()
        prev_parent = cls.get_parent_correlation_id()
        prev_start = cls._start_time.get()
        
        try:
            # Set new context
            cls.set_correlation_id(correlation_id)
            if user_id:
                cls.set_user_id(user_id)
            if session_id:
                cls.set_session_id(session_id)
            
            yield correlation_id
            
        finally:
            # Restore previous context
            cls._correlation_id.set(prev_correlation)
            cls._user_id.set(prev_user) 
            cls._session_id.set(prev_session)
            cls._operation_stack.set(prev_stack)
            cls._parent_correlation_id.set(prev_parent)
            cls._start_time.set(prev_start)
    
    @classmethod
    @contextmanager
    def child_context(cls, child_correlation_id: Optional[str] = None):
        """
        Create a child correlation context that inherits from parent.
        
        Args:
            child_correlation_id: Child correlation ID (generates if None)
        """
        parent_id = cls.get_correlation_id()
        
        if child_correlation_id is None:
            child_correlation_id = cls.generate_correlation_id()
        
        with cls.context(correlation_id=child_correlation_id):
            if parent_id:
                cls.set_parent_correlation_id(parent_id)
            yield child_correlation_id


@dataclass
class LogEntry:
    """
    Structured log entry with comprehensive metadata.
    
    This class defines the standard structure for all log entries,
    ensuring consistency across the application and enabling
    effective log analysis and monitoring.
    """
    timestamp: str
    level: str
    message: str
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    operation_stack: Optional[List[str]] = None
    parent_correlation_id: Optional[str] = None
    duration_ms: Optional[float] = None
    thread_id: Optional[str] = None
    process_id: Optional[int] = None
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    exception: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None
    environment: Optional[str] = None
    version: Optional[str] = None


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging with correlation context.
    
    This formatter converts Python log records into structured JSON format
    with automatic correlation context injection and comprehensive metadata.
    
    Features:
    - JSON structured output for machine parsing
    - Automatic correlation context injection
    - Exception stack trace formatting
    - Performance metrics integration
    - Configurable field inclusion/exclusion
    - ISO 8601 timestamp formatting
    """
    
    def __init__(self, 
                 include_process_info: bool = True,
                 include_thread_info: bool = True,
                 include_code_location: bool = True,
                 environment: Optional[str] = None,
                 version: Optional[str] = None):
        """
        Initialize the structured formatter.
        
        Args:
            include_process_info: Include process ID in log entries
            include_thread_info: Include thread information
            include_code_location: Include file/function/line info
            environment: Environment name (dev, staging, prod)
            version: Application version string
        """
        super().__init__()
        self.include_process_info = include_process_info
        self.include_thread_info = include_thread_info
        self.include_code_location = include_code_location
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.version = version or os.getenv('APP_VERSION', '1.0.0')
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as structured JSON.
        
        Args:
            record: Python logging.LogRecord instance
            
        Returns:
            JSON formatted log string
        """
        # Create base log entry
        log_entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=record.levelname,
            message=record.getMessage(),
            environment=self.environment,
            version=self.version
        )
        
        # Add correlation context
        log_entry.correlation_id = CorrelationContext.get_correlation_id()
        log_entry.user_id = CorrelationContext.get_user_id()
        log_entry.session_id = CorrelationContext.get_session_id()
        log_entry.operation = CorrelationContext.get_current_operation()
        log_entry.operation_stack = CorrelationContext.get_operation_stack()
        log_entry.parent_correlation_id = CorrelationContext.get_parent_correlation_id()
        log_entry.duration_ms = CorrelationContext.get_operation_duration_ms()
        
        # Add process information
        if self.include_process_info:
            log_entry.process_id = os.getpid()
        
        # Add thread information
        if self.include_thread_info:
            log_entry.thread_id = str(threading.get_ident())
        
        # Add code location information
        if self.include_code_location:
            log_entry.module = record.module
            log_entry.function = record.funcName
            log_entry.line_number = record.lineno
        
        # Add exception information if present
        if record.exc_info:
            log_entry.exception = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add custom fields from record
        if hasattr(record, 'metrics'):
            log_entry.metrics = record.metrics
        
        if hasattr(record, 'tags'):
            log_entry.tags = record.tags
        
        # Convert to dictionary and filter None values
        log_dict = asdict(log_entry)
        log_dict = {k: v for k, v in log_dict.items() if v is not None}
        
        # Return JSON string
        try:
            return json.dumps(log_dict, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            # Fallback to basic logging if JSON serialization fails
            return f"{{\"timestamp\":\"{log_entry.timestamp}\",\"level\":\"{log_entry.level}\",\"message\":\"JSON serialization failed: {str(e)}\",\"correlation_id\":\"{log_entry.correlation_id}\"}}"


class CorrelationLogger:
    """
    Enhanced logger with automatic correlation context and structured logging.
    
    This class provides a high-level interface for structured logging with
    automatic correlation ID injection, performance metrics, and audit capabilities.
    
    Features:
    - Automatic correlation context injection
    - Structured JSON logging format
    - Performance timing integration
    - Audit logging capabilities
    - Multiple output destinations
    - Configurable log levels and filters
    - Thread-safe operation
    
    Usage:
        logger = CorrelationLogger("my_module")
        logger.info("Operation started", user_id="user123")
        
        with logger.operation("image_processing"):
            logger.info("Processing image", file_path="image.jpg")
    """
    
    def __init__(self, 
                 name: str,
                 level: Union[int, str] = logging.INFO,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 log_directory: Optional[Path] = None):
        """
        Initialize correlation logger.
        
        Args:
            name: Logger name (typically __name__)
            level: Logging level
            enable_console: Enable console output
            enable_file: Enable file output
            log_directory: Directory for log files
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(enable_console, enable_file, log_directory)
    
    def _setup_handlers(self, 
                       enable_console: bool,
                       enable_file: bool,
                       log_directory: Optional[Path]) -> None:
        """Setup logging handlers with structured formatting."""
        formatter = StructuredFormatter()
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if enable_file:
            if log_directory is None:
                log_directory = Path("logs")
            
            log_directory.mkdir(exist_ok=True)
            
            # Main application log
            file_handler = logging.handlers.RotatingFileHandler(
                log_directory / "application.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=10
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Separate correlation log for tracing
            correlation_handler = logging.handlers.RotatingFileHandler(
                log_directory / "correlation.log",
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=20
            )
            correlation_handler.setFormatter(formatter)
            correlation_handler.addFilter(lambda record: hasattr(record, 'correlation_id'))
            self.logger.addHandler(correlation_handler)
            
            # Audit log with longer retention
            audit_handler = logging.handlers.RotatingFileHandler(
                log_directory / "audit.log",
                maxBytes=100 * 1024 * 1024,  # 100MB
                backupCount=50
            )
            audit_handler.setFormatter(formatter)
            audit_handler.addFilter(lambda record: record.levelno >= LogLevel.AUDIT.value)
            self.logger.addHandler(audit_handler)
    
    def _log(self, 
             level: int, 
             message: str, 
             exc_info: Optional[bool] = None,
             **kwargs) -> None:
        """
        Internal logging method with correlation context.
        
        Args:
            level: Logging level
            message: Log message
            exc_info: Include exception information
            **kwargs: Additional fields to include in log entry
        """
        # Create log record
        record = self.logger.makeRecord(
            self.name,
            level,
            __file__,
            sys._getframe(2).f_lineno,
            message,
            (),
            exc_info
        )
        
        # Add custom fields
        for key, value in kwargs.items():
            setattr(record, key, value)
        
        # Add correlation context as custom attribute
        setattr(record, 'correlation_id', CorrelationContext.get_correlation_id())
        
        # Handle the record
        self.logger.handle(record)
    
    def trace(self, message: str, **kwargs) -> None:
        """Log trace level message."""
        self._log(LogLevel.TRACE.value, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug level message."""
        self._log(LogLevel.DEBUG.value, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info level message."""
        self._log(LogLevel.INFO.value, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning level message."""
        self._log(LogLevel.WARNING.value, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = True, **kwargs) -> None:
        """Log error level message."""
        self._log(LogLevel.ERROR.value, message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs) -> None:
        """Log critical level message."""
        self._log(LogLevel.CRITICAL.value, message, exc_info=exc_info, **kwargs)
    
    def audit(self, message: str, **kwargs) -> None:
        """Log audit event with security/compliance context."""
        self._log(LogLevel.AUDIT.value, message, **kwargs)
    
    @contextmanager
    def operation(self, operation_name: str):
        """
        Context manager for operation tracking with automatic timing.
        
        Usage:
            with logger.operation("image_processing.resize"):
                # All logs here will include operation context
                logger.info("Resizing image")
        
        Args:
            operation_name: Name of the operation being performed
        """
        start_time = time.perf_counter()
        
        # Push operation onto stack
        CorrelationContext.push_operation(operation_name)
        
        try:
            self.info(f"Operation started: {operation_name}")
            yield
            
            duration = (time.perf_counter() - start_time) * 1000
            self.info(f"Operation completed: {operation_name}", 
                     metrics={"duration_ms": duration, "status": "success"})
            
        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            self.error(f"Operation failed: {operation_name}", 
                      metrics={"duration_ms": duration, "status": "error", "error": str(e)})
            raise
            
        finally:
            # Pop operation from stack
            CorrelationContext.pop_operation()


def with_correlation(operation_name: Optional[str] = None,
                    generate_id: bool = True,
                    log_performance: bool = True):
    """
    Decorator for automatic correlation tracking and performance logging.
    
    This decorator automatically:
    - Generates correlation ID if not present
    - Tracks operation in correlation context
    - Logs operation start/completion with timing
    - Handles exceptions with proper correlation
    
    Args:
        operation_name: Name of operation (defaults to module.function)
        generate_id: Generate new correlation ID if not present
        log_performance: Log performance metrics
    
    Usage:
        @with_correlation("image_processing.resize")
        def resize_image(image, size):
            # Function implementation
            pass
        
        @with_correlation()  # Uses automatic naming
        def process_batch(files):
            # Function implementation  
            pass
    """
    def decorator(func: Callable) -> Callable:
        # Determine operation name
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create correlation ID
            correlation_id = CorrelationContext.get_correlation_id()
            if not correlation_id and generate_id:
                correlation_id = CorrelationContext.generate_correlation_id()
                CorrelationContext.set_correlation_id(correlation_id)
            
            # Create logger for this function
            logger = CorrelationLogger(func.__module__)
            
            # Use operation context for tracking
            with logger.operation(op_name):
                if log_performance:
                    start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    
                    if log_performance:
                        duration = (time.perf_counter() - start_time) * 1000
                        logger.info(f"Function executed successfully: {op_name}",
                                   metrics={"execution_time_ms": duration})
                    
                    return result
                    
                except Exception as e:
                    if log_performance:
                        duration = (time.perf_counter() - start_time) * 1000
                        logger.error(f"Function failed: {op_name}",
                                    metrics={"execution_time_ms": duration},
                                    error_type=type(e).__name__)
                    raise
        
        return wrapper
    return decorator


class LoggingConfiguration:
    """
    Centralized logging configuration management.
    
    This class provides a single point for configuring the entire
    logging system across the application.
    """
    
    @staticmethod
    def setup_application_logging(
        app_name: str = "ImageProcessingApp",
        log_level: Union[int, str] = logging.INFO,
        log_directory: Optional[Path] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_correlation: bool = True,
        environment: Optional[str] = None,
        version: Optional[str] = None
    ) -> None:
        """
        Setup comprehensive application logging configuration.
        
        Args:
            app_name: Application name for logging context
            log_level: Global logging level
            log_directory: Directory for log files
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_correlation: Enable correlation ID tracking
            environment: Environment name (dev, staging, prod)
            version: Application version
        """
        # Clear any existing configuration
        logging.getLogger().handlers.clear()
        
        # Set root logger level
        logging.getLogger().setLevel(log_level)
        
        # Add custom log level for audit
        logging.addLevelName(LogLevel.AUDIT.value, 'AUDIT')
        logging.addLevelName(LogLevel.TRACE.value, 'TRACE')
        
        # Setup structured formatter
        formatter = StructuredFormatter(
            environment=environment,
            version=version
        )
        
        if log_directory is None:
            log_directory = Path("logs")
        
        log_directory.mkdir(exist_ok=True)
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            logging.getLogger().addHandler(console_handler)
        
        # File handlers with different purposes
        if enable_file:
            # Main application log
            app_handler = logging.handlers.RotatingFileHandler(
                log_directory / f"{app_name.lower()}.log",
                maxBytes=20 * 1024 * 1024,  # 20MB
                backupCount=15
            )
            app_handler.setFormatter(formatter)
            app_handler.setLevel(log_level)
            logging.getLogger().addHandler(app_handler)
            
            # Error log for critical issues
            error_handler = logging.handlers.RotatingFileHandler(
                log_directory / f"{app_name.lower()}_errors.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=20
            )
            error_handler.setFormatter(formatter)
            error_handler.setLevel(logging.ERROR)
            logging.getLogger().addHandler(error_handler)
            
            # Performance log
            perf_handler = logging.handlers.RotatingFileHandler(
                log_directory / f"{app_name.lower()}_performance.log",
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=10
            )
            perf_handler.setFormatter(formatter)
            perf_handler.addFilter(lambda record: hasattr(record, 'metrics'))
            logging.getLogger().addHandler(perf_handler)
        
        # Log configuration completion
        logger = CorrelationLogger(__name__)
        logger.info(f"Logging configuration completed for {app_name}",
                   tags={
                       "app_name": app_name,
                       "log_level": str(log_level),
                       "console_enabled": enable_console,
                       "file_enabled": enable_file,
                       "correlation_enabled": enable_correlation
                   })


# Example usage and testing functions
if __name__ == "__main__":
    # Setup logging
    LoggingConfiguration.setup_application_logging(
        app_name="ImageProcessingApp",
        log_level=logging.DEBUG,
        environment="development",
        version="1.0.0"
    )
    
    # Example usage
    logger = CorrelationLogger(__name__)
    
    # Basic logging with automatic correlation
    with CorrelationContext.context(user_id="user123", session_id="session456"):
        logger.info("User logged in", action="login")
        
        # Nested operation tracking
        with logger.operation("image_processing"):
            logger.info("Starting image processing", file_count=10)
            
            # Simulate some processing
            import time
            time.sleep(0.1)
            
            logger.info("Image processing completed", 
                       metrics={"images_processed": 10, "success_rate": 1.0})
        
        # Demonstrate error logging
        try:
            raise ValueError("Example error")
        except ValueError:
            logger.error("Processing failed", 
                        tags={"component": "image_processor", "operation": "resize"})
        
        # Audit logging
        logger.audit("User accessed sensitive data", 
                    resource="/admin/users",
                    action="view")
    
    print("Logging demonstration completed. Check logs directory for output.")