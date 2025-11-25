#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple GUI Import Test
Tests that all GUI components can be imported successfully.
"""

import sys
from pathlib import Path

def test_gui_imports():
    """Test that GUI components can be imported."""
    try:
        print("Testing GUI component imports...")
        
        # Test PyQt6 imports
        from PyQt6.QtWidgets import QApplication, QMainWindow
        print("? PyQt6 widgets imported successfully")
        
        from PyQt6.QtCore import Qt, pyqtSignal
        print("? PyQt6 core imported successfully")
        
        from PyQt6.QtGui import QDragEnterEvent, QDropEvent
        print("? PyQt6 GUI imported successfully")
        
        # Test main GUI class import
        sys.path.insert(0, str(Path(__file__).parent))
        from src.gui.main_window import MaximizedImageProcessingGUI
        print("? Main GUI class imported successfully")
        
        # Test individual component imports
        from src.gui.main_window import (
            ProcessingControlsRow,
            ProcessingDropZone,
            PickupZone,
            MatrixHeaderRow,
            DestinationMatrix,
            EnhancedDestinationCell
        )
        print("? All GUI component classes imported successfully")
        
        print("\n?? ALL IMPORT TESTS PASSED!")
        print("?? GUI is ready to launch!")
        
        return True
        
    except ImportError as e:
        print(f"? Import Error: {e}")
        return False
    except Exception as e:
        print(f"? Error: {e}")
        return False

if __name__ == "__main__":
    success = test_gui_imports()
    sys.exit(0 if success else 1)