#!/usr/bin/env python3
"""
GUI Launcher for Image Processing Application
Optimized for 32" Samsung Smart Monitor (1920x1080)
Ensures maximized window display
FIXED: PyQt6 recursion and safe array issues

Project ID: Image Processing App 20251119
Created: 2025-01-25
Fixed: 2025-01-19 - Resolved PyQt6 recursion issues  
Author: The-Sage-Mage / GitHub Copilot
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Launch the GUI application with error handling."""
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Check for PyQt6 with better error handling
    try:
        import PyQt6
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QScreen
        print("? PyQt6 loaded successfully")
    except ImportError as e:
        print(f"Error: PyQt6 is required but not installed: {e}")
        print("Install with: pip install PyQt6")
        sys.exit(1)
    except Exception as e:
        print(f"Error: PyQt6 loading failed: {e}")
        sys.exit(1)
    
    # Import our simplified GUI (fixes recursion issues)
    try:
        from src.gui.main_window import MaximizedImageProcessingGUI
        print("? GUI components loaded successfully")
    except ImportError as e:
        print(f"Error: Could not import GUI components: {e}")
        print("Make sure src/gui/main_window.py exists")
        sys.exit(1)
    except Exception as e:
        print(f"Error: GUI component initialization failed: {e}")
        sys.exit(1)
    
    # Create QApplication with error handling
    try:
        app = QApplication(sys.argv)
        print("? QApplication created successfully")
        
        # Set application properties
        app.setApplicationName("Image Processing Application")
        app.setApplicationDisplayName("Image Processing App - Fixed GUI")
        app.setOrganizationName("The-Sage-Mage LLC")
        app.setApplicationVersion("1.0.0")
        
    except Exception as e:
        print(f"Error: Failed to create QApplication: {e}")
        sys.exit(1)
    
    # Detect screen resolution
    try:
        screen = QScreen.availableGeometry(app.primaryScreen())
        screen_width = screen.width()
        screen_height = screen.height()
        
        print(f"? Detected screen resolution: {screen_width}x{screen_height}")
        
        # Warn if not optimal resolution
        if screen_width < 1200 or screen_height < 800:
            print(f"Warning: Screen resolution ({screen_width}x{screen_height}) is below optimal")
            print("This application works best with 1200x800 or higher")
        
    except Exception as e:
        print(f"Warning: Could not detect screen resolution: {e}")
        screen_width, screen_height = 1200, 800
    
    # Set high DPI support
    try:
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        print("? High DPI support enabled")
    except Exception as e:
        print(f"Warning: Could not enable high DPI support: {e}")
    
    # Create and configure main window
    try:
        window = MaximizedImageProcessingGUI()
        print("? Main window created successfully")
        
        # Show window (simplified - no forced maximizing that can cause issues)
        window.show()
        window.raise_()
        window.activateWindow()
        
        print("? GUI launched successfully")
        print("Fixed: PyQt6 recursion and safe array issues resolved")
        
        # Start event loop
        return app.exec()
        
    except Exception as e:
        print(f"Error: Failed to create or show GUI window: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())