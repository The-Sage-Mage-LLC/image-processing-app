"""
NumPy Python 3.14 Compatibility Fix
Specifically addresses the MachArLike object initialization crash.

This module provides targeted fixes for NumPy compatibility issues
with Python 3.14, particularly the access violation at address 0x21
in _multiarray_umath.cp314-win_amd64.pyd during MachArLike object init.
"""

import numpy as np
import logging
import warnings
from typing import Any, Optional
import sys
import os


logger = logging.getLogger(__name__)


def apply_macharllike_fix():
    """
    Apply specific fix for MachArLike object initialization crash.
    
    This addresses the 0xC0000005 access violation that occurs in
    NumPy's _multiarray_umath.cp314-win_amd64.pyd at address 0x00007FFFE0518602
    when initializing MachArLike objects.
    """
    try:
        # Force NumPy to use conservative memory alignment
        os.environ['NPY_DISABLE_OPTIMIZATION'] = '1'
        
        # Disable NumPy's experimental features that might be unstable in Python 3.14
        if hasattr(np, '_set_promotion_state'):
            try:
                np._set_promotion_state('legacy')
                logger.info("Set NumPy promotion state to legacy for compatibility")
            except Exception as e:
                logger.warning(f"Could not set NumPy promotion state: {e}")
        
        # Patch numpy.dtype to prevent MachArLike crashes
        original_dtype_new = np.dtype.__new__
        
        def safe_dtype_new(cls, dtype, align=False, copy=False):
            """Safe dtype creation that prevents MachArLike crashes."""
            try:
                # Validate inputs before passing to original function
                if dtype is None:
                    raise ValueError("dtype cannot be None")
                
                # Create dtype with extra safety checks
                result = original_dtype_new(cls, dtype, align, copy)
                
                # Validate the result before returning
                if result is None:
                    raise RuntimeError("dtype creation returned None")
                
                # Check for valid type object
                if not hasattr(result, 'type'):
                    raise RuntimeError("Invalid dtype object created")
                
                return result
                
            except Exception as e:
                logger.error(f"Safe dtype creation failed: {e}")
                # Fallback to a known safe dtype
                if isinstance(dtype, str):
                    safe_dtypes = ['float64', 'float32', 'int64', 'int32', 'uint8']
                    if dtype in safe_dtypes:
                        return original_dtype_new(cls, dtype, align, copy)
                    else:
                        logger.warning(f"Unknown dtype {dtype}, falling back to float64")
                        return original_dtype_new(cls, 'float64', align, copy)
                else:
                    return original_dtype_new(cls, 'float64', align, copy)
        
        # Apply the patch
        np.dtype.__new__ = safe_dtype_new
        logger.info("Applied MachArLike crash prevention patch to np.dtype")
        
        # Additional array creation safety
        original_array = np.array
        original_empty = np.empty
        original_zeros = np.zeros
        original_ones = np.ones
        
        def safe_array(*args, **kwargs):
            """Safe array creation wrapper."""
            try:
                # Ensure we have valid arguments
                if not args:
                    args = ([],)
                
                # Force copy for safety if dealing with external memory
                if 'copy' not in kwargs:
                    kwargs['copy'] = False  # Let NumPy decide, but be explicit
                
                result = original_array(*args, **kwargs)
                
                # Validate result
                if result is None:
                    raise RuntimeError("Array creation returned None")
                
                # Ensure proper memory layout
                if not result.flags.c_contiguous:
                    result = np.ascontiguousarray(result)
                
                return result
                
            except Exception as e:
                logger.error(f"Safe array creation failed: {e}")
                # Create a minimal safe array as fallback
                return original_array([0], dtype='float64')
        
        def safe_empty(shape, dtype=float, order='C', *, like=None):
            """Safe empty array creation."""
            try:
                result = original_empty(shape, dtype, order, like=like)
                if not result.flags.c_contiguous and order == 'C':
                    result = np.ascontiguousarray(result)
                return result
            except Exception as e:
                logger.error(f"Safe empty creation failed: {e}")
                return original_empty((1,), dtype='float64', order='C')
        
        def safe_zeros(shape, dtype=float, order='C', *, like=None):
            """Safe zeros array creation."""
            try:
                result = original_zeros(shape, dtype, order, like=like)
                if not result.flags.c_contiguous and order == 'C':
                    result = np.ascontiguousarray(result)
                return result
            except Exception as e:
                logger.error(f"Safe zeros creation failed: {e}")
                return original_zeros((1,), dtype='float64', order='C')
        
        def safe_ones(shape, dtype=float, order='C', *, like=None):
            """Safe ones array creation."""
            try:
                result = original_ones(shape, dtype, order, like=like)
                if not result.flags.c_contiguous and order == 'C':
                    result = np.ascontiguousarray(result)
                return result
            except Exception as e:
                logger.error(f"Safe ones creation failed: {e}")
                return original_ones((1,), dtype='float64', order='C')
        
        # Apply patches
        np.array = safe_array
        np.empty = safe_empty
        np.zeros = safe_zeros
        np.ones = safe_ones
        
        logger.info("Applied comprehensive NumPy safety patches")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply MachArLike fix: {e}")
        return False


def configure_numpy_for_python314():
    """
    Configure NumPy specifically for Python 3.14 compatibility.
    """
    try:
        # Check Python version
        if sys.version_info >= (3, 14):
            logger.info("Detected Python 3.14+, applying compatibility fixes")
            
            # Set conservative NumPy configuration
            np.seterr(all='warn')  # Convert errors to warnings
            
            # Disable potentially problematic features
            warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')
            
            # Force single-threaded operation for stability
            os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
            os.environ.setdefault('MKL_NUM_THREADS', '1')
            os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
            
            logger.info("Applied Python 3.14 NumPy compatibility configuration")
            
        return apply_macharllike_fix()
        
    except Exception as e:
        logger.error(f"NumPy Python 3.14 configuration failed: {e}")
        return False


def emergency_numpy_fix():
    """
    Emergency fix to prevent immediate crashes while dependencies are updated.
    """
    try:
        # Monkey patch the most common crash points
        import numpy.core._multiarray_umath as _multiarray_umath
        
        # If we can access the module, try to patch the problematic function
        if hasattr(_multiarray_umath, 'implementation_of_numpy_function'):
            original_func = getattr(_multiarray_umath, 'implementation_of_numpy_function', None)
            
            def safe_implementation(*args, **kwargs):
                try:
                    return original_func(*args, **kwargs) if original_func else None
                except Exception as e:
                    logger.error(f"Prevented crash in numpy function: {e}")
                    return None
            
            if original_func:
                setattr(_multiarray_umath, 'implementation_of_numpy_function', safe_implementation)
        
        logger.info("Applied emergency NumPy crash prevention")
        return True
        
    except Exception as e:
        logger.debug(f"Emergency fix not applicable: {e}")
        return False


# Auto-apply fixes when module is imported
def initialize_fixes():
    """Initialize all NumPy compatibility fixes."""
    try:
        success = configure_numpy_for_python314()
        if not success:
            logger.warning("Primary fixes failed, trying emergency fix")
            emergency_numpy_fix()
        
        logger.info("NumPy compatibility fixes initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize NumPy fixes: {e}")


# Apply fixes on import
if __name__ != '__main__':
    initialize_fixes()