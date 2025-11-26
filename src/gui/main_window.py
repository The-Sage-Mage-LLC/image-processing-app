"""
Enhanced GUI Main Window Implementation - Optimized for 32" Samsung Monitor
Project ID: Image Processing App 20251119
Created: 2025-01-19 07:21:15 UTC
Enhanced: 2025-01-25 for 1920x1080 32" Samsung Smart Monitor
Fixed: 2025-01-19 - Resolved PyQt6 recursion issues
Author: The-Sage-Mage / GitHub Copilot

FIXED ISSUES:
- PyQt6 "Failed to create safe array" recursion errors
- Complex signal chain infinite loops
- Memory allocation issues in widget hierarchies
"""

# Import the simplified version to fix recursion issues
from .main_window_simple import SimplifiedImageProcessingGUI as MaximizedImageProcessingGUI
from .main_window_simple import main

# Keep backward compatibility
ImageProcessingGUI = MaximizedImageProcessingGUI

# Export main function
__all__ = ['MaximizedImageProcessingGUI', 'ImageProcessingGUI', 'main']