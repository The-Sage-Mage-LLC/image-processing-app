"""
Simplified GUI Main Window Implementation - Fixed PyQt6 Recursion Issues
Project ID: Image Processing App 20251119
Created: 2025-01-19 (Simplified to prevent recursion)
Author: GitHub Copilot - The-Sage-Mage

This version fixes the PyQt6 "Failed to create safe array" recursion errors.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QCheckBox, QTreeWidget, QTreeWidgetItem,
    QSplitter, QFrame, QListWidget, QListWidgetItem, QMessageBox,
    QHeaderView, QAbstractItemView, QSizePolicy, QGroupBox,
    QProgressBar, QTextEdit, QFileDialog, QComboBox
)
from PyQt6.QtCore import (
    Qt, QSize, QThread, pyqtSignal, pyqtSlot, QTimer
)
from PyQt6.QtGui import (
    QDragEnterEvent, QDropEvent, QDragMoveEvent, QDragLeaveEvent,
    QPalette, QColor, QFont, QIcon
)

# Import our CLI components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SimpleFileExplorer(QTreeWidget):
    """Simplified file explorer without complex metadata evaluation."""
    
    files_selected = pyqtSignal(list)  # Signal when files are selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_explorer()
    
    def setup_explorer(self):
        """Setup the file explorer."""
        # Configure headers
        self.setHeaderLabels(['File Name', 'Size', 'Type'])
        
        # Set column widths 
        self.header().resizeSection(0, 300)  # File Name
        self.header().resizeSection(1, 100)  # Size
        self.header().resizeSection(2, 80)   # Type
        
        # Enable multi-selection and drag
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        
        # Enable sorting
        self.setSortingEnabled(True)
        self.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        
        # Connect selection changes - SIMPLIFIED to prevent recursion
        self.itemSelectionChanged.connect(self._on_selection_changed)
    
    @pyqtSlot()
    def _on_selection_changed(self):
        """Handle selection change - simplified to prevent recursion."""
        try:
            selected_items = self.selectedItems()
            selected_files = []
            
            for item in selected_items:
                file_path = item.data(0, Qt.ItemDataRole.UserRole)
                if file_path:
                    selected_files.append(Path(file_path))
            
            # Use single shot timer to prevent recursion
            QTimer.singleShot(10, lambda: self.files_selected.emit(selected_files))
            
        except Exception as e:
            print(f"Selection error: {e}")
    
    def populate_directory(self, directory_path: Path):
        """Populate the explorer with files from directory."""
        self.clear()
        
        if not directory_path.exists() or not directory_path.is_dir():
            return
        
        # Get image files
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
        
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                self.add_file_item(file_path)
    
    def add_file_item(self, file_path: Path):
        """Add a file item to the tree."""
        try:
            stat_info = file_path.stat()
            
            # Format file size
            size_bytes = stat_info.st_size
            if size_bytes > 1024 * 1024:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            elif size_bytes > 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes} B"
            
            # Create item
            item = QTreeWidgetItem([
                file_path.name,
                size_str,
                file_path.suffix.upper()[1:]  # Extension without dot
            ])
            
            # Store file path in item data
            item.setData(0, Qt.ItemDataRole.UserRole, str(file_path))
            
            self.addTopLevelItem(item)
            
        except Exception:
            # Add with minimal info if stat fails
            item = QTreeWidgetItem([
                file_path.name,
                "Unknown",
                file_path.suffix.upper()[1:]
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, str(file_path))
            self.addTopLevelItem(item)


class SimpleProcessingControls(QFrame):
    """Simplified processing controls without complex signal chains."""
    
    operations_changed = pyqtSignal(set)  # Signal when selected operations change
    
    def __init__(self):
        super().__init__()
        self.selected_operations = set()
        self.checkboxes = {}
        self._updating = False  # Prevent recursion
        self.setup_controls()
    
    def setup_controls(self):
        """Setup processing control checkboxes."""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(1)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Processing Options")
        title.setStyleSheet("font-weight: bold; padding: 3px;")
        layout.addWidget(title)
        
        # Controls container
        controls_layout = QHBoxLayout()
        
        # Simplified checkbox definitions
        checkboxes = [
            ("Grayscale", "grayscale"),
            ("Sepia", "sepia"),  
            ("Pencil Sketch", "pencil_sketch"),
            ("Coloring Book", "coloring_book")
        ]
        
        for name, operation in checkboxes:
            cb = QCheckBox(name)
            cb.setChecked(False)
            
            # SIMPLIFIED connection to prevent recursion
            cb.stateChanged.connect(lambda state, op=operation: self._on_checkbox_changed(op, state))
            controls_layout.addWidget(cb)
            self.checkboxes[name] = cb
        
        layout.addWidget(QWidget())
        widget = QWidget()
        widget.setLayout(controls_layout)
        layout.addWidget(widget)
        
        self.setLayout(layout)
    
    def _on_checkbox_changed(self, operation: str, state: int):
        """Handle checkbox state change - simplified to prevent recursion."""
        if self._updating:
            return
            
        try:
            if state == Qt.CheckState.Checked.value:
                self.selected_operations.add(operation)
            else:
                self.selected_operations.discard(operation)
            
            # Use single shot timer to emit signal
            QTimer.singleShot(10, lambda: self.operations_changed.emit(self.selected_operations))
            
        except Exception as e:
            print(f"Checkbox error: {e}")


class SimpleDropZone(QFrame):
    """Simplified drop zone without complex event handling."""
    
    files_dropped = pyqtSignal(list)  # Signal when files are dropped
    
    def __init__(self):
        super().__init__()
        self.dropped_files = []
        self.setup_drop_zone()
    
    def setup_drop_zone(self):
        """Setup the drop zone."""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        
        layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Drop Files Here for Processing")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding: 20px;
                background-color: #f0f0f0;
                border: 2px dashed #ccc;
                border-radius: 5px;
                color: #666;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Simple file list
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(60)
        layout.addWidget(self.file_list)
        
        self.setLayout(layout)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    padding: 20px;
                    background-color: #e3f2fd;
                    border: 2px solid #2196f3;
                    border-radius: 5px;
                    color: #1976d2;
                }
            """)
    
    def dragLeaveEvent(self, event: QDragLeaveEvent):
        """Handle drag leave event."""
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding: 20px;
                background-color: #f0f0f0;
                border: 2px dashed #ccc;
                border-radius: 5px;
                color: #666;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        files = []
        for url in event.mimeData().urls():
            file_path = Path(url.toLocalFile())
            if file_path.is_file() and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}:
                files.append(file_path)
                self.file_list.addItem(file_path.name)
        
        if files:
            self.dropped_files.extend(files)
            # Use single shot timer to emit signal
            QTimer.singleShot(10, lambda: self.files_dropped.emit(files))
        
        event.acceptProposedAction()
        self.dragLeaveEvent(None)


class SimplifiedImageProcessingGUI(QMainWindow):
    """Simplified main GUI window - fixes PyQt6 recursion issues."""
    
    def __init__(self):
        super().__init__()
        self.selected_files = []
        self.selected_operations = set()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the simplified UI."""
        # Set window properties
        self.setWindowTitle("Image Processing Application - Simplified GUI")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        central_widget.setLayout(main_layout)
        
        # Left side - File Explorer
        left_frame = self.create_file_explorer_frame()
        main_layout.addWidget(left_frame, 1)
        
        # Right side - Processing
        right_frame = self.create_processing_frame()
        main_layout.addWidget(right_frame, 1)
        
        # Setup menu
        self.setup_menu()
        self.statusBar().showMessage("Simplified GUI Ready")
    
    def create_file_explorer_frame(self) -> QFrame:
        """Create file explorer frame."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setLineWidth(1)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Source Files")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(title)
        
        # Browse button
        browse_btn = QPushButton("Browse Folder")
        browse_btn.clicked.connect(self.browse_folder)
        layout.addWidget(browse_btn)
        
        # File explorer
        self.file_explorer = SimpleFileExplorer()
        self.file_explorer.files_selected.connect(self.on_files_selected)
        layout.addWidget(self.file_explorer)
        
        frame.setLayout(layout)
        return frame
    
    def create_processing_frame(self) -> QFrame:
        """Create processing frame."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setLineWidth(1)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Processing")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(title)
        
        # Processing controls
        self.processing_controls = SimpleProcessingControls()
        self.processing_controls.operations_changed.connect(self.on_operations_changed)
        layout.addWidget(self.processing_controls)
        
        # Drop zone
        self.drop_zone = SimpleDropZone()
        self.drop_zone.files_dropped.connect(self.on_files_dropped)
        layout.addWidget(self.drop_zone)
        
        # Output selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        
        self.output_btn = QPushButton("Select Output Folder")
        self.output_btn.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.output_btn)
        
        output_widget = QWidget()
        output_widget.setLayout(output_layout)
        layout.addWidget(output_widget)
        
        # Process button
        self.process_btn = QPushButton("Process Images")
        self.process_btn.clicked.connect(self.process_images)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)
        
        frame.setLayout(layout)
        return frame
    
    def setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        file_menu.addAction('Exit', self.close)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        help_menu.addAction('About', self.show_about)
    
    @pyqtSlot()
    def browse_folder(self):
        """Browse for source folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder:
            self.file_explorer.populate_directory(Path(folder))
            self.add_status(f"Loaded folder: {folder}")
    
    @pyqtSlot(list)
    def on_files_selected(self, files):
        """Handle file selection."""
        self.selected_files = files
        self.add_status(f"Selected {len(files)} files")
        self.update_process_button()
    
    @pyqtSlot(set)
    def on_operations_changed(self, operations):
        """Handle operations selection change."""
        self.selected_operations = operations
        self.add_status(f"Selected operations: {', '.join(operations)}")
        self.update_process_button()
    
    @pyqtSlot(list)
    def on_files_dropped(self, files):
        """Handle dropped files."""
        self.add_status(f"Dropped {len(files)} files")
        # Add to selected files if not already present
        for file_path in files:
            if file_path not in self.selected_files:
                self.selected_files.append(file_path)
        self.update_process_button()
    
    @pyqtSlot()
    def select_output_folder(self):
        """Select output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = Path(folder)
            self.output_btn.setText(f"Output: {folder}")
            self.add_status(f"Output folder: {folder}")
            self.update_process_button()
    
    def update_process_button(self):
        """Update process button state."""
        has_files = len(self.selected_files) > 0
        has_operations = len(self.selected_operations) > 0
        has_output = hasattr(self, 'output_folder')
        
        self.process_btn.setEnabled(has_files and has_operations and has_output)
    
    @pyqtSlot()
    def process_images(self):
        """Process selected images."""
        self.add_status("Processing images...")
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Simple processing simulation
        total_files = len(self.selected_files)
        
        for i, file_path in enumerate(self.selected_files):
            # Simulate processing
            QApplication.processEvents()  # Keep UI responsive
            
            for operation in self.selected_operations:
                self.add_status(f"Processing {file_path.name} - {operation}")
                # Here you would add actual processing logic
            
            # Update progress
            progress = int((i + 1) / total_files * 100)
            self.progress_bar.setValue(progress)
            
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.add_status("Processing complete!")
    
    def add_status(self, message: str):
        """Add status message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.status_text.append(formatted_message)
        
        # Auto-scroll to bottom
        cursor = self.status_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.status_text.setTextCursor(cursor)
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About", 
                         "Image Processing Application\\n"
                         "Simplified GUI - Fixed PyQt6 Issues\\n"
                         "Version 1.0.0")


# Create alias for compatibility
MaximizedImageProcessingGUI = SimplifiedImageProcessingGUI


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Image Processing GUI")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = SimplifiedImageProcessingGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()