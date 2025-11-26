"""
Enhanced GUI Main Window Implementation - Complete Requirements Implementation
Project ID: Image Processing App 20251119
Created: 2025-01-19 07:21:15 UTC
Enhanced: 2025-01-25 for 1920x1080 32" Samsung Smart Monitor
Fixed: 2025-01-19 - Resolved PyQt6 recursion issues
Complete: 2025-01-19 - Full requirements implementation
Enhanced: 2025-01-19 - Exact checkbox behavior implementation
Author: The-Sage-Mage / GitHub Copilot

COMPLETE IMPLEMENTATION:
- Frame A: Windows Explorer functionality with metadata evaluation
- Frame B: Exact row structure (7%, 14%, 14%, 7%, 56%)  
- Metadata evaluation: Green checkmark/red X indicators for 25+ fields
- Seven checkboxes with EXACT behavior as specified in part two requirements
- Complete drag and drop functionality
- All sizing and layout requirements met
"""

# Import the complete clean version with exact checkbox behavior
from .main_window_complete_clean import CompleteImageProcessingGUI as MaximizedImageProcessingGUI
from .main_window_complete_clean import main

# Keep backward compatibility
ImageProcessingGUI = MaximizedImageProcessingGUI

# Export main function
__all__ = ['MaximizedImageProcessingGUI', 'ImageProcessingGUI', 'main']