"""
Array Safety Utilities
Prevents memory corruption and access violations in NumPy array operations.

This module provides defensive array handling to prevent crashes caused by
incompatible memory layouts between different libraries (OpenCV, PyTorch, etc.)
"""

import numpy as np
import logging
from typing import Any, Union, Optional
import warnings


logger = logging.getLogger(__name__)


class ArraySafetyError(Exception):
    """Exception raised when array safety validation fails."""
    pass


def safe_array_conversion(array: Any, 
                         dtype: Optional[np.dtype] = None,
                         force_copy: bool = False,
                         check_finite: bool = True) -> np.ndarray:
    """
    Safely convert any array-like object to a properly aligned NumPy array.
    
    This function prevents the memory corruption issue that causes access violations
    in NumPy's MachArLike object initialization.
    
    Args:
        array: Input array-like object (numpy array, torch tensor, opencv mat, etc.)
        dtype: Target data type. If None, preserves original dtype
        force_copy: If True, always creates a copy even if array is already safe
        check_finite: If True, checks for NaN/inf values
        
    Returns:
        numpy.ndarray: Safe, properly aligned NumPy array
        
    Raises:
        ArraySafetyError: If array conversion or validation fails
        
    Example:
        >>> import torch
        >>> tensor = torch.randn(100, 100)
        >>> safe_array = safe_array_conversion(tensor)
        >>> # safe_array can now be used safely with any NumPy operation
    """
    try:
        # Step 1: Convert to numpy array if not already
        if not isinstance(array, np.ndarray):
            if hasattr(array, 'numpy'):  # PyTorch tensor
                logger.debug("Converting PyTorch tensor to numpy")
                array = array.detach().cpu().numpy()
            elif hasattr(array, '__array__'):  # Generic array protocol
                logger.debug("Converting array-like object via __array__ protocol")
                array = np.asarray(array)
            else:
                logger.debug("Converting object to numpy array")
                array = np.array(array)
        
        # Step 2: Handle dtype conversion
        if dtype is not None and array.dtype != dtype:
            logger.debug(f"Converting dtype from {array.dtype} to {dtype}")
            array = array.astype(dtype)
        
        # Step 3: Ensure proper memory layout
        needs_copy = force_copy
        
        if not array.flags['C_CONTIGUOUS']:
            logger.debug("Array is not C-contiguous, creating contiguous copy")
            needs_copy = True
        
        if not array.flags['ALIGNED']:
            logger.debug("Array is not properly aligned, creating aligned copy")
            needs_copy = True
        
        if not array.flags['OWNDATA'] and not force_copy:
            logger.debug("Array does not own its data, creating copy for safety")
            needs_copy = True
        
        # Step 4: Create safe copy if needed
        if needs_copy:
            array = np.ascontiguousarray(array)
        
        # Step 5: Validate array properties
        if array.size == 0:
            logger.warning("Array is empty")
        
        if check_finite and np.issubdtype(array.dtype, np.floating):
            if not np.all(np.isfinite(array)):
                warnings.warn("Array contains NaN or infinite values", UserWarning)
        
        # Step 6: Final safety check
        if not _validate_array_safety(array):
            raise ArraySafetyError("Array failed final safety validation")
        
        logger.debug(f"Successfully created safe array: shape={array.shape}, "
                    f"dtype={array.dtype}, C_CONTIGUOUS={array.flags['C_CONTIGUOUS']}")
        
        return array
        
    except Exception as e:
        logger.error(f"Array conversion failed: {str(e)}")
        raise ArraySafetyError(f"Failed to create safe array: {str(e)}") from e


def _validate_array_safety(array: np.ndarray) -> bool:
    """
    Validate that an array is safe for NumPy operations.
    
    Args:
        array: NumPy array to validate
        
    Returns:
        bool: True if array is safe, False otherwise
    """
    try:
        # Check basic properties
        if not isinstance(array, np.ndarray):
            return False
        
        # Check memory layout flags
        required_flags = ['C_CONTIGUOUS', 'ALIGNED', 'OWNDATA']
        for flag in required_flags:
            if not array.flags[flag]:
                logger.warning(f"Array missing required flag: {flag}")
                return False
        
        # Check for valid shape and strides
        if array.ndim < 0:
            return False
        
        if any(dim < 0 for dim in array.shape):
            return False
        
        if array.strides is None:
            return False
        
        # Check stride calculations don't overflow
        try:
            # Simulate the stride calculation that was failing
            for i, (dim, stride) in enumerate(zip(array.shape, array.strides)):
                if dim > 0:
                    max_offset = (dim - 1) * stride
                    # Check for integer overflow that could cause the 0x9 address issue
                    if max_offset < 0 or max_offset > array.nbytes:
                        logger.error(f"Invalid stride calculation at dimension {i}: "
                                   f"dim={dim}, stride={stride}, max_offset={max_offset}")
                        return False
        except (OverflowError, ValueError) as e:
            logger.error(f"Stride validation failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Array safety validation failed: {e}")
        return False


def safe_tensor_to_numpy(tensor: Any) -> np.ndarray:
    """
    Safely convert PyTorch tensor to NumPy array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        numpy.ndarray: Safe NumPy array
    """
    return safe_array_conversion(tensor)


def safe_opencv_to_numpy(mat: Any) -> np.ndarray:
    """
    Safely convert OpenCV Mat to NumPy array.
    
    Args:
        mat: OpenCV Mat object
        
    Returns:
        numpy.ndarray: Safe NumPy array
    """
    return safe_array_conversion(mat)


def patch_numpy_operations():
    """
    Apply patches to common NumPy operations to prevent memory corruption.
    
    This function monkey-patches numpy functions that commonly cause issues
    with arrays from other libraries.
    """
    import numpy as np
    
    # Store original functions
    _original_array = np.array
    _original_asarray = np.asarray
    _original_ascontiguousarray = np.ascontiguousarray
    
    def patched_array(*args, **kwargs):
        """Patched np.array with safety validation."""
        result = _original_array(*args, **kwargs)
        return safe_array_conversion(result, force_copy=False)
    
    def patched_asarray(*args, **kwargs):
        """Patched np.asarray with safety validation."""
        result = _original_asarray(*args, **kwargs)
        return safe_array_conversion(result, force_copy=False)
    
    def patched_ascontiguousarray(*args, **kwargs):
        """Patched np.ascontiguousarray with additional validation."""
        result = _original_ascontiguousarray(*args, **kwargs)
        if not _validate_array_safety(result):
            logger.warning("ascontiguousarray produced unsafe array, applying additional fixes")
            return safe_array_conversion(result, force_copy=True)
        return result
    
    # Apply patches
    np.array = patched_array
    np.asarray = patched_asarray
    np.ascontiguousarray = patched_ascontiguousarray
    
    logger.info("NumPy operations patched for array safety")


def enable_array_safety(patch_numpy: bool = True, 
                       log_level: str = "WARNING") -> None:
    """
    Enable array safety features globally.
    
    Args:
        patch_numpy: If True, patches numpy functions for automatic safety
        log_level: Logging level for array safety messages
    """
    # Set up logging
    logging.getLogger(__name__).setLevel(getattr(logging, log_level.upper()))
    
    if patch_numpy:
        patch_numpy_operations()
    
    # Set numpy error handling
    np.seterr(all='warn')
    
    logger.info("Array safety features enabled")


# Convenience decorators
def safe_array_operation(func):
    """
    Decorator to automatically apply safe array conversion to function arguments.
    
    Example:
        @safe_array_operation
        def process_image(image_array):
            # image_array is now guaranteed to be safe
            return some_operation(image_array)
    """
    def wrapper(*args, **kwargs):
        # Convert array-like arguments
        safe_args = []
        for arg in args:
            if hasattr(arg, '__array__') or isinstance(arg, np.ndarray):
                safe_args.append(safe_array_conversion(arg))
            else:
                safe_args.append(arg)
        
        safe_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(value, '__array__') or isinstance(value, np.ndarray):
                safe_kwargs[key] = safe_array_conversion(value)
            else:
                safe_kwargs[key] = value
        
        return func(*safe_args, **safe_kwargs)
    
    return wrapper