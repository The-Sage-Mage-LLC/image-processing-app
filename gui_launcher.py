#!/usr/bin/env python3
"""
GUI Launcher for Image Processing Application
Optimized for 32" Samsung Smart Monitor (1920x1080)
Ensures maximized window display

Project ID: Image Processing App 20251119
Created: 2025-01-25
Author: The-Sage-Mage
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Launch the maximized GUI application."""
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Check for PyQt6
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QScreen
    except ImportError:
        print("Error: PyQt6 is required but not installed")
        print("Install with: pip install PyQt6")
        sys.exit(1)
    
    # Import our GUI
    try:
        from src.gui.main_window import MaximizedImageProcessingGUI
    except ImportError as e:
        print(f"Error: Could not import GUI components: {e}")
        sys.exit(1)
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Image Processing Application")
    app.setApplicationDisplayName("Image Processing App - 32\" Monitor Optimized")
    app.setOrganizationName("The-Sage-Mage LLC")
    app.setApplicationVersion("1.0.0")
    
    # Detect screen resolution and validate
    screen = QScreen.availableGeometry(app.primaryScreen())
    screen_width = screen.width()
    screen_height = screen.height()
    
    print(f"Detected screen resolution: {screen_width}x{screen_height}")
    
    # Warn if not optimal resolution
    if screen_width < 1920 or screen_height < 1080:
        print(f"Warning: Screen resolution ({screen_width}x{screen_height}) is below optimal")
        print("This application is optimized for 1920x1080 or higher")
    
    # Set high DPI support for large monitors
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    # Create and configure main window
    try:
        window = MaximizedImageProcessingGUI()
        
        # Force maximized state (critical requirement)
        window.setWindowState(Qt.WindowState.WindowMaximized)
        window.showMaximized()
        
        # Ensure window is focused and on top
        window.raise_()
        window.activateWindow()
        
        print("GUI launched successfully in maximized mode")
        print("Optimized for 32\" Samsung Smart Monitor (LS32CM502EKXKR)")
        
        # Start event loop
        return app.exec()
        
    except Exception as e:
        print(f"Error: Failed to create GUI window: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())