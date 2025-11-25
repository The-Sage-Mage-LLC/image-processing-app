"""
Image Processing Application Package
Project ID: Image Processing App 20251119
Created: 2025-11-19 06:52:45 UTC
Author: The-Sage-Mage
"""

__version__ = "1.0.0"
__project_id__ = "Image Processing App 20251119"
__author__ = "The-Sage-Mage"

# CRITICAL: Apply NumPy Python 3.14 compatibility fixes FIRST
# This prevents the 0xC0000005 crash in _multiarray_umath.cp314-win_amd64.pyd
try:
    from .utils.numpy_compatibility_fix import initialize_fixes
    initialize_fixes()
    print("? NumPy Python 3.14 compatibility fixes applied")
except ImportError as e:
    print(f"? Warning: Could not apply NumPy compatibility fixes: {e}")
except Exception as e:
    print(f"? Warning: NumPy compatibility fix failed: {e}")

# CRITICAL: Enable array safety to prevent NumPy access violations
# This fixes the 0xC0000005 crash in _multiarray_umath.cp314-win_amd64.pyd
try:
    from .utils.array_safety import enable_array_safety
    enable_array_safety(patch_numpy=True, log_level="WARNING")
    print("? Array safety enabled - NumPy access violation protection active")
except ImportError as e:
    print(f"? Warning: Could not enable array safety: {e}")
except Exception as e:
    print(f"? Warning: Array safety initialization failed: {e}")