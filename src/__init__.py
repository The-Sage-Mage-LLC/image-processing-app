#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Processing Application
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Enterprise-grade image processing application with AI-powered analysis.
"""

__version__ = "1.0.0"
__title__ = "Image Processing App"
__description__ = "Enterprise image processing application with AI-powered analysis and batch transformations"
__author__ = "The-Sage-Mage"
__email__ = "contact@thesagemage.com"
__license__ = "MIT"
__url__ = "https://github.com/The-Sage-Mage-LLC/image-processing-app"

# Project metadata
PROJECT_ID = "Image Processing App 20251119"
CREATION_DATE = "2025-01-19"

# Module version info
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

def get_version():
    """Get version string."""
    return __version__

def get_version_info():
    """Get detailed version information."""
    return VERSION_INFO.copy()

# Package exports
__all__ = [
    "__version__",
    "__title__", 
    "__description__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
    "PROJECT_ID",
    "CREATION_DATE",
    "get_version",
    "get_version_info"
]

# CRITICAL: Apply NumPy Python 3.14 compatibility fixes FIRST
# This prevents the 0xC0000005 crash in _multiarray_umath.cp314-win_amd64.pyd
try:
    from .utils.numpy_compatibility_fix import initialize_fixes
    initialize_fixes()
    print("+ NumPy Python 3.14 compatibility fixes applied")
except ImportError as e:
    print(f"! Warning: Could not apply NumPy compatibility fixes: {e}")
except Exception as e:
    print(f"! Warning: NumPy compatibility fix failed: {e}")

# CRITICAL: Enable array safety to prevent NumPy access violations
# FIXED: Disabled patching to prevent infinite recursion loop
try:
    from .utils.array_safety import enable_array_safety
    enable_array_safety(patch_numpy=False)  # CRITICAL: patch_numpy=False to prevent recursion
    print("+ Array safety enabled - NumPy access violation protection active")
except ImportError as e:
    print(f"! Warning: Could not enable array safety: {e}")
except Exception as e:
    print(f"! Warning: Array safety initialization failed: {e}")
