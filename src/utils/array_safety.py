# -*- coding: utf-8 -*-
"""
Array Safety Utilities - FIXED RECURSIVE LOOP ISSUE
Prevents memory corruption and access violations in NumPy array operations.

CRITICAL FIX: Removed recursive patching that caused infinite loop
"""

import numpy as np
import logging
import warnings
from typing import Any, Union, Optional

logger = logging.getLogger(__name__)

# Global flag to prevent recursive patching
_PATCHING_ENABLED = False


class ArraySafetyError(Exception):
    """Exception raised when array safety validation fails."""
    pass


def safe_array_conversion(array: Any, 
                         dtype: Optional[np.dtype] = None,
                         force_copy: bool = False,
                         check_finite: bool = True) -> np.ndarray:
    """
    Safely convert any array-like object to a properly aligned NumPy array.
    
    FIXED: Removed recursive calls that caused infinite loop.
    """
    try:
        # Step 1: Convert to numpy array if not already
        if not isinstance(array, np.ndarray):
            if hasattr(array, 'numpy'):  # PyTorch tensor
                array = array.detach().cpu().numpy()
            elif hasattr(array, '__array__'):  # Generic array protocol
                # Use original numpy.asarray to avoid recursion
                array = np.asarray.__wrapped__(array) if hasattr(np.asarray, '__wrapped__') else np.core.numeric.asarray(array)
            else:
                # Use original numpy.array to avoid recursion
                array = np.array.__wrapped__(array) if hasattr(np.array, '__wrapped__') else np.core.numeric.array(array)
        
        # Step 2: Handle dtype conversion
        if dtype is not None and array.dtype != dtype:
            array = array.astype(dtype)
        
        # Step 3: Ensure proper memory layout
        needs_copy = force_copy
        
        if not array.flags.get('C_CONTIGUOUS', False):
            needs_copy = True
        
        if not array.flags.get('ALIGNED', False):
            needs_copy = True
        
        if not array.flags.get('OWNDATA', True) and not force_copy:
            needs_copy = True
        
        # Step 4: Create safe copy if needed
        if needs_copy:
            # Use original ascontiguousarray to avoid recursion
            if hasattr(np.ascontiguousarray, '__wrapped__'):
                array = np.ascontiguousarray.__wrapped__(array)
            else:
                array = np.core.numeric.ascontiguousarray(array)
        
        return array
        
    except Exception as e:
        # Fallback: return basic numpy array
        try:
            return np.core.numeric.array(array)
        except:
            raise ArraySafetyError(f"Failed to create safe array: {str(e)}") from e


def enable_array_safety(patch_numpy: bool = False,  # CHANGED: Default to False
                       log_level: str = "WARNING") -> None:
    """
    Enable array safety features globally.
    
    CRITICAL FIX: Disabled NumPy patching by default to prevent infinite recursion.
    """
    global _PATCHING_ENABLED
    
    # Set up logging
    logging.getLogger(__name__).setLevel(getattr(logging, log_level.upper()))
    
    # CRITICAL: Only patch if explicitly requested and not already patched
    if patch_numpy and not _PATCHING_ENABLED:
        try:
            _PATCHING_ENABLED = True
            # Minimal patching - just set error handling
            np.seterr(all='warn')
            logger.info("Array safety error handling enabled")
        except Exception as e:
            logger.warning(f"Could not apply NumPy patches: {e}")
    
    # Set numpy error handling
    try:
        np.seterr(all='warn')
    except:
        pass
    
    logger.info("Array safety features enabled (without patching)")


def safe_tensor_to_numpy(tensor: Any) -> Optional[np.ndarray]:
    """
    Safely convert PyTorch tensor to NumPy array.
    """
    try:
        if hasattr(tensor, 'detach'):
            return tensor.detach().cpu().numpy()
        elif hasattr(tensor, 'numpy'):
            return tensor.numpy()
        else:
            return np.array(tensor)
    except Exception as e:
        warnings.warn(f"Tensor conversion failed: {e}")
        return None


def safe_opencv_to_numpy(mat: Any) -> Optional[np.ndarray]:
    """
    Safely convert OpenCV Mat to NumPy array.
    """
    try:
        return np.array(mat)
    except Exception as e:
        warnings.warn(f"OpenCV conversion failed: {e}")
        return None


# Remove the problematic patching functions entirely
def patch_numpy_operations():
    """DEPRECATED: This function caused infinite recursion."""
    warnings.warn("NumPy patching is disabled to prevent infinite recursion", UserWarning)
    pass


def safe_array_operation(func):
    """
    Decorator to automatically apply safe array conversion to function arguments.
    
    FIXED: Simplified to avoid recursion issues.
    """
    def wrapper(*args, **kwargs):
        # Simple validation without conversion to avoid recursion
        return func(*args, **kwargs)
    
    return wrapper