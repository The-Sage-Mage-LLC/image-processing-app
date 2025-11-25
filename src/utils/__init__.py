"""Utilities Module"""

# Import array safety utilities with error handling
try:
    from .array_safety import (
        safe_array_conversion,
        safe_tensor_to_numpy,
        safe_opencv_to_numpy,
        enable_array_safety,
        safe_array_operation,
        ArraySafetyError
    )
except ImportError:
    # Array safety not available - create dummy functions
    def safe_array_conversion(arr, **kwargs):
        return arr
    def safe_tensor_to_numpy(tensor, **kwargs):
        return tensor
    def safe_opencv_to_numpy(arr, **kwargs):
        return arr
    def enable_array_safety(**kwargs):
        pass
    def safe_array_operation(func):
        return func
    class ArraySafetyError(Exception):
        pass

# Import monitoring utilities with fallback
try:
    from .monitoring import EnhancedProcessingMonitor, ProcessingMonitor, HeartbeatLogger
except ImportError:
    # Create dummy monitoring classes if not available
    class ProcessingMonitor:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    class HeartbeatLogger:
        def __init__(self, *args, **kwargs):
            pass
    
    EnhancedProcessingMonitor = ProcessingMonitor

# DO NOT auto-enable array safety to prevent OpenCV conflicts
# Array safety can be manually enabled if needed

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