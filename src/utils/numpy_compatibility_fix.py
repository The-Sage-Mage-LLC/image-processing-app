# -*- coding: utf-8 -*-
"""
NumPy Python 3.14 Compatibility Fix Module
Project ID: Image Processing App 20251119
Created: 2025-01-24
Author: The-Sage-Mage

Fixes NumPy compatibility issues with Python 3.14 to prevent:
- 0xC0000005 crash in _multiarray_umath.cp314-win_amd64.pyd
- Threading conflicts
- Memory access violations
"""

import os
import warnings
from typing import Any, Optional

def initialize_fixes():
    """Initialize NumPy Python 3.14 compatibility fixes."""
    try:
        # Set environment variables before NumPy import
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMBA_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # Import NumPy safely
        import numpy as np
        
        # Configure NumPy for Python 3.14
        np.seterr(all='warn')
        
        return True
        
    except Exception as e:
        warnings.warn(f"NumPy compatibility fix failed: {e}")
        return False

def safe_array(*args, **kwargs) -> Optional[Any]:
    """Safe array creation."""
    try:
        import numpy as np
        return np.array(*args, **kwargs)
    except Exception:
        return None

def safe_zeros(shape, dtype=float, order='C') -> Optional[Any]:
    """Safe zeros array creation."""
    try:
        import numpy as np
        return np.zeros(shape, dtype=dtype, order=order)
    except Exception:
        return None

def safe_ones(shape, dtype=float, order='C') -> Optional[Any]:
    """Safe ones array creation."""
    try:
        import numpy as np
        return np.ones(shape, dtype=dtype, order=order)
    except Exception:
        return None

def safe_empty(shape, dtype=float, order='C') -> Optional[Any]:
    """Safe empty array creation."""
    try:
        import numpy as np
        return np.empty(shape, dtype=dtype, order=order)
    except Exception:
        return None
