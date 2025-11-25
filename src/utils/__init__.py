"""
Utils Module - Core Utilities Package
Project ID: Image Processing App 20251119
"""

# Core utilities - always available
try:
    from .logger import setup_logging
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False

try:
    from .database import DatabaseManager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Array safety - optional
try:
    from .array_safety import (
        safe_array_conversion,
        enable_array_safety,
        ArraySafetyError
    )
    ARRAY_SAFETY_AVAILABLE = True
except ImportError:
    ARRAY_SAFETY_AVAILABLE = False

# Monitoring - optional
try:
    from .monitoring import EnhancedProcessingMonitor, HeartbeatLogger
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Quality control - optional
try:
    from .image_quality_manager import ImageQualityManager
    QUALITY_CONTROL_AVAILABLE = True
except ImportError:
    QUALITY_CONTROL_AVAILABLE = False

# Export what's available
__all__ = []

if LOGGER_AVAILABLE:
    __all__.extend(['setup_logging'])

if DATABASE_AVAILABLE:
    __all__.extend(['DatabaseManager'])

if ARRAY_SAFETY_AVAILABLE:
    __all__.extend(['safe_array_conversion', 'enable_array_safety', 'ArraySafetyError'])

if MONITORING_AVAILABLE:
    __all__.extend(['EnhancedProcessingMonitor', 'HeartbeatLogger'])

if QUALITY_CONTROL_AVAILABLE:
    __all__.extend(['ImageQualityManager'])
