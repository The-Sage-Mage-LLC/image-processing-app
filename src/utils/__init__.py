"""Utilities Module"""

# Import array safety utilities first
try:
    from .array_safety import (
        safe_array_conversion,
        safe_tensor_to_numpy,
        safe_opencv_to_numpy,
        enable_array_safety,
        safe_array_operation,
        ArraySafetyError
    )
    print("? Array safety enabled - NumPy access violation protection active")
except Exception as e:
    print(f"? Warning: Array safety initialization failed: {e}")

# Import monitoring utilities
try:
    from .monitoring import EnhancedProcessingMonitor, ProcessingMonitor, HeartbeatLogger
    print("? Enhanced monitoring system loaded successfully")
except Exception as e:
    print(f"? Warning: Enhanced monitoring initialization failed: {e}")
    # Fallback to basic monitoring
    try:
        from .monitoring import ProcessingMonitor, HeartbeatLogger
        print("? Basic monitoring system loaded as fallback")
    except Exception as e2:
        print(f"? Error: All monitoring systems failed: {e2}")
        raise

# Enable array safety by default with error handling
try:
    enable_array_safety(patch_numpy=True, log_level="WARNING")
    print("? Array safety protections activated")
except Exception as e:
    print(f"? Warning: Array safety activation failed: {e}")

__all__ = [
    'safe_array_conversion',
    'safe_tensor_to_numpy', 
    'safe_opencv_to_numpy',
    'enable_array_safety',
    'safe_array_operation',
    'ArraySafetyError',
    'EnhancedProcessingMonitor',
    'ProcessingMonitor',
    'HeartbeatLogger'
]