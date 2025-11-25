"""
Enhanced GUI Main Window Implementation - Optimized for 32" Samsung Monitor
Project ID: Image Processing App 20251119
Created: 2025-01-19 07:21:15 UTC
Enhanced: 2025-01-25 for 1920x1080 32" Samsung Smart Monitor
Author: The-Sage-Mage
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import json
import threading
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QCheckBox, QTreeWidget, QTreeWidgetItem,
    QSplitter, QFrame, QListWidget, QListWidgetItem, QMessageBox,
    QHeaderView, QAbstractItemView, QSizePolicy, QGroupBox,
    QTableWidget, QTableWidgetItem, QToolTip, QMenu, QLineEdit,
    QFileDialog, QComboBox, QProgressBar, QTextEdit, QScrollArea,
    QSpinBox, QSlider
)
from PyQt6.QtCore import (
    Qt, QSize, QPoint, QRect, QMimeData, QUrl, QTimer,
    QThread, pyqtSignal, QModelIndex, QDir, QFileInfo,
    QStandardPaths, QSettings, pyqtSlot
)
from PyQt6.QtGui import (
    QDragEnterEvent, QDropEvent, QDragMoveEvent, QDragLeaveEvent,
    QPalette, QColor, QFont, QIcon, QPixmap, QPainter,
    QBrush, QPen, QAction, QKeySequence, QFontMetrics
)

# Import our CLI components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.file_manager import FileManager
from src.core.image_processor import ImageProcessor
from src.core.metadata_handler import MetadataHandler
from src.utils.logger import setup_logging


class MetadataEvaluationThread(QThread):
    """Thread for evaluating metadata in image files."""
    
    file_evaluated = pyqtSignal(str, bool, int)  # file_path, has_rich_metadata, field_count
    evaluation_complete = pyqtSignal()
    
    def __init__(self, file_paths: List[Path], metadata_handler: MetadataHandler):
        super().__init__()
        self.file_paths = file_paths
        self.metadata_handler = metadata_handler
        self._stop_requested = False
    
    def run(self):
        """Evaluate metadata for each file."""
        for file_path in self.file_paths:
            if self._stop_requested:
                break
                
            try:
                # Extract metadata
                metadata = self.metadata_handler.extract_all_metadata(file_path)
                
                # Count non-empty fields (excluding computed/derived fields)
                field_count = 0
                for key, value in metadata.items():
                    if value and str(value).strip() and key not in [
                        'primary_key', 'file_path', 'file_name', 'file_extension',
                        'data_acquisition_timestamp', 'extraction_error'
                    ]:
                        field_count += 1
                
                # Check if file has rich metadata (25+ fields)
                has_rich_metadata = field_count >= 25
                
                self.file_evaluated.emit(str(file_path), has_rich_metadata, field_count)
                
            except Exception as e:
                # Assume poor metadata if evaluation fails
                self.file_evaluated.emit(str(file_path), False, 0)
        
        self.evaluation_complete.emit()
    
    def stop_evaluation(self):
        """Stop the evaluation process."""
        self._stop_requested = True


class ProcessingThread(QThread):
    """Thread for running image processing tasks."""
    
    progress = pyqtSignal(str)  # Progress message
    finished = pyqtSignal(list)  # List of processed files
    error = pyqtSignal(str)  # Error message
    
    def __init__(self, processor, files, operations):
        super().__init__()
        self.processor = processor
        self.files = files
        self.operations = operations
        
    def run(self):
        """Run the processing operations."""
        try:
            processed_files = []
            
            for file_path in self.files:
                self.progress.emit(f"Processing: {file_path.name}")
                
                # Process based on selected operations
                for operation in self.operations:
                    if operation == 'BWG':
                        # Process grayscale - processor handles this
                        pass
                    elif operation == 'SEP':
                        # Process sepia
                        pass
                    elif operation == 'PSK':
                        # Process pencil sketch
                        pass
                    elif operation == 'BK_CLR':
                        # Process coloring book
                        pass
                    elif operation == 'BK_CTD':
                        # Process connect-the-dots
                        pass
                    elif operation == 'BK_CBN':
                        # Process color-by-numbers
                        pass
                
                processed_files.append(file_path)
            
            self.finished.emit(processed_files)
            
        except Exception as e:
            self.error.emit(str(e))


class EnhancedFileExplorer(QTreeWidget):
    """Enhanced file explorer with metadata evaluation and visual indicators."""
    
    files_selected = pyqtSignal(list)  # Signal when files are selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metadata_handler = None
        self.evaluation_thread = None
        self.file_metadata_cache = {}
        
        self.setup_explorer()
    
    def setup_explorer(self):
        """Setup the file explorer."""
        # Configure headers
        self.setHeaderLabels(['File Name', 'Created Date', 'Size', 'Type'])
        
        # Set column widths for optimal display on 32" monitor
        self.header().resizeSection(0, 400)  # File Name (wider for metadata indicators)
        self.header().resizeSection(1, 150)  # Created Date
        self.header().resizeSection(2, 100)  # Size
        self.header().resizeSection(3, 80)   # Type
        
        # Enable multi-selection and drag
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        
        # Enable sorting
        self.setSortingEnabled(True)
        self.sortByColumn(0, Qt.SortOrder.AscendingOrder)  # Default: Name ASC
        
        # Connect selection changes
        self.itemSelectionChanged.connect(self.on_selection_changed)
        
        # Style for metadata indicators
        self.setStyleSheet("""
            QTreeWidget::item {
                padding: 3px;
                border-bottom: 1px solid #e0e0e0;
            }
            QTreeWidget::item:selected {
                background-color: #3daee9;
                color: white;
            }
            .metadata-indicator {
                font-weight: bold;
                padding: 2px 6px;
                border-radius: 3px;
                margin-right: 5px;
            }
            .metadata-rich {
                background-color: #27ae60;
                color: white;
            }
            .metadata-poor {
                background-color: #e74c3c;
                color: white;
            }
        """)
    
    def set_metadata_handler(self, metadata_handler: MetadataHandler):
        """Set the metadata handler for evaluation."""
        self.metadata_handler = metadata_handler
    
    def populate_directory(self, directory_path: Path):
        """Populate the explorer with files from directory."""
        self.clear()
        self.file_metadata_cache.clear()
        
        if not directory_path.exists() or not directory_path.is_dir():
            return
        
        # Get image files
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
        image_files = []
        
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                image_files.append(file_path)
        
        # Add files to tree
        for file_path in image_files:
            self.add_file_item(file_path)
        
        # Start metadata evaluation
        if self.metadata_handler and image_files:
            self.start_metadata_evaluation(image_files)
    
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
            
            # Format creation date
            created_date = datetime.fromtimestamp(stat_info.st_ctime)
            date_str = created_date.strftime('%Y-%m-%d %H:%M')
            
            # Create item with placeholder for metadata indicator
            item_text = f"⏳ {file_path.name}"  # Placeholder while evaluating
            
            item = QTreeWidgetItem([
                item_text,
                date_str,
                size_str,
                file_path.suffix.upper()[1:]  # Extension without dot
            ])
            
            # Store file path in item data
            item.setData(0, Qt.ItemDataRole.UserRole, str(file_path))
            
            self.addTopLevelItem(item)
            
        except Exception as e:
            # Add with minimal info if stat fails
            item = QTreeWidgetItem([
                f"❓ {file_path.name}",
                "Unknown",
                "Unknown",
                file_path.suffix.upper()[1:]
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, str(file_path))
            self.addTopLevelItem(item)
    
    def start_metadata_evaluation(self, file_paths: List[Path]):
        """Start metadata evaluation in background thread."""
        if self.evaluation_thread and self.evaluation_thread.isRunning():
            self.evaluation_thread.stop_evaluation()
            self.evaluation_thread.wait()
        
        self.evaluation_thread = MetadataEvaluationThread(file_paths, self.metadata_handler)
        self.evaluation_thread.file_evaluated.connect(self.on_file_evaluated)
        self.evaluation_thread.evaluation_complete.connect(self.on_evaluation_complete)
        self.evaluation_thread.start()
    
    @pyqtSlot(str, bool, int)
    def on_file_evaluated(self, file_path: str, has_rich_metadata: bool, field_count: int):
        """Handle when a file's metadata has been evaluated."""
        # Store in cache
        self.file_metadata_cache[file_path] = {
            'has_rich_metadata': has_rich_metadata,
            'field_count': field_count
        }
        
        # Find and update the corresponding item
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            item_file_path = item.data(0, Qt.ItemDataRole.UserRole)
            
            if item_file_path == file_path:
                # Update the item text with metadata indicator
                file_name = Path(file_path).name
                
                if has_rich_metadata:
                    # Green checkmark for rich metadata (25+ fields)
                    indicator = "✓"
                    item_text = f"{indicator} {file_name}"
                    item.setForeground(0, QColor("#27ae60"))  # Green text
                else:
                    # Red X for poor metadata (<25 fields)
                    indicator = "✗"
                    item_text = f"{indicator} {file_name}"
                    item.setForeground(0, QColor("#e74c3c"))  # Red text
                
                item.setText(0, item_text)
                item.setToolTip(0, f"Metadata fields: {field_count}")
                break
    
    @pyqtSlot()
    def on_evaluation_complete(self):
        """Handle when metadata evaluation is complete."""
        # Optional: Show completion status
        pass
    
    def on_selection_changed(self):
        """Handle selection change."""
        selected_items = self.selectedItems()
        selected_files = []
        
        for item in selected_items:
            file_path = item.data(0, Qt.ItemDataRole.UserRole)
            if file_path:
                selected_files.append(Path(file_path))
        
        self.files_selected.emit(selected_files)
    
    def sort_by_name_asc(self):
        """Sort files by name ascending."""
        self.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    
    def sort_by_date_desc(self):
        """Sort files by creation date descending."""
        self.sortByColumn(1, Qt.SortOrder.DescendingOrder)
    
    def get_file_statistics(self):
        """Get file statistics for display."""
        total_files = self.topLevelItemCount()
        jpg_count = 0
        png_count = 0
        
        for i in range(total_files):
            item = self.topLevelItem(i)
            file_type = item.text(3).lower()
            
            if file_type in ['jpg', 'jpeg']:
                jpg_count += 1
            elif file_type == 'png':
                png_count += 1
        
        return {
            'total': total_files,
            'jpg': jpg_count,
            'png': png_count
        }
    
    def startDrag(self, supportedActions):
        """Start drag operation."""
        selected_items = self.selectedItems()
        if not selected_items:
            return
        
        # Create mime data with file URLs
        mime_data = QMimeData()
        urls = []
        
        for item in selected_items:
            file_path = item.data(0, Qt.ItemDataRole.UserRole)
            if file_path:
                urls.append(QUrl.fromLocalFile(file_path))
        
        mime_data.setUrls(urls)
        
        # Start drag
        drag = self.startDrag(supportedActions)
        if drag:
            drag.setMimeData(mime_data)
            drag.exec(supportedActions)


class RowFrame(QFrame):
    """Base frame for Frame B rows with proper sizing."""
    
    def __init__(self, height_percentage: float, title: str = ""):
        super().__init__()
        self.height_percentage = height_percentage
        self.title = title
        self.setup_frame()
    
    def setup_frame(self):
        """Setup the frame."""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(1)
        
        # Set minimum height based on percentage of 1080p (approximate)
        # 1080p height ≈ 1020 usable pixels (accounting for title bar, taskbar)
        usable_height = 1020
        min_height = int(usable_height * self.height_percentage / 100)
        self.setMinimumHeight(min_height)
        
        if self.title:
            layout = QVBoxLayout()
            title_label = QLabel(self.title)
            title_label.setStyleSheet("font-weight: bold; padding: 3px;")
            layout.addWidget(title_label)
            self.setLayout(layout)


class ProcessingControlsRow(RowFrame):
    """Row 1: Processing operation checkboxes (7% height)."""
    
    operations_changed = pyqtSignal(set)  # Signal when selected operations change
    
    def __init__(self):
        super().__init__(7, "Processing Options")
        self.selected_operations = set()
        self.checkboxes = {}
        self._updating_programmatically = False  # Prevent recursive updates
        self.setup_controls()
    
    def setup_controls(self):
        """Setup processing control checkboxes."""
        # Get existing layout or create new one
        layout = self.layout()
        
        # Controls container
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        # Checkbox definitions - EXACTLY as specified in requirements
        checkboxes = [
            ("All", "All six menu items", "all"),
            ("BWG", "Black and White (grayscale)", "grayscale"),
            ("SEP", "Sepia-toned", "sepia"),  
            ("PSK", "Pencil Sketch", "pencil_sketch"),
            ("BK_CLR", "Coloring book", "coloring_book"),
            ("BK_CTD", "Connect-the-dots activity book", "connect_dots"),
            ("BK_CBN", "Color-by-numbers activity book", "color_by_numbers")
        ]
        
        for name, tooltip, operation in checkboxes:
            cb = QCheckBox(name)
            cb.setToolTip(tooltip)
            
            # Set initial state to unchecked (requirement: defaults unchecked/off)
            cb.setChecked(False)
            
            cb.stateChanged.connect(lambda state, op=operation, n=name: self.on_checkbox_changed(n, op, state))
            controls_layout.addWidget(cb)
            self.checkboxes[name] = cb
        
        controls_layout.addStretch()
        controls_widget.setLayout(controls_layout)
        layout.addWidget(controls_widget)
        
        # Initialize selected operations to empty (defaults unchecked)
        self.selected_operations.clear()
    
    def reset_to_defaults(self):
        """Reset all checkboxes to default (unchecked) state."""
        self._updating_programmatically = True
        
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)
        
        self.selected_operations.clear()
        self._updating_programmatically = False
        self.operations_changed.emit(self.selected_operations)
    
    def on_checkbox_changed(self, name: str, operation: str, state: int):
        """
        Handle checkbox state change according to exact requirements.
        
        Requirements implemented:
        - Checkbox 1 ("All"): Controls all other 6 checkboxes
        - Auto-check "All" when all 6 individual operations are selected
        - Auto-uncheck "All" when any individual operation is unselected
        - Auto-uncheck all when "All" is unchecked
        - Auto-check all when "All" is checked
        """
        # Prevent recursive updates during programmatic changes
        if self._updating_programmatically:
            return
        
        self._updating_programmatically = True
        
        try:
            if name == "All":
                # Checkbox 1: "All" checkbox behavior
                if state == Qt.CheckState.Checked.value:
                    # Check all other 6 checkboxes
                    for cb_name, cb in self.checkboxes.items():
                        if cb_name != "All":
                            cb.setChecked(True)
                    # Set all 6 operations as selected
                    self.selected_operations = {"grayscale", "sepia", "pencil_sketch", "coloring_book", "connect_dots", "color_by_numbers"}
                else:
                    # Uncheck all other 6 checkboxes
                    for cb_name, cb in self.checkboxes.items():
                        if cb_name != "All":
                            cb.setChecked(False)
                    # Clear all selected operations
                    self.selected_operations.clear()
            else:
                # Individual checkbox behavior (checkboxes 2-7)
                if state == Qt.CheckState.Checked.value:
                    # Add this operation to selected set
                    self.selected_operations.add(operation)
                    
                    # Check if all 6 operations are now selected
                    if len(self.selected_operations) == 6:
                        # Auto-check "All" checkbox
                        self.checkboxes["All"].setChecked(True)
                else:
                    # Remove this operation from selected set
                    self.selected_operations.discard(operation)
                    
                    # Auto-uncheck "All" if any operation is unchecked
                    if self.checkboxes["All"].isChecked():
                        self.checkboxes["All"].setChecked(False)
        
        finally:
            self._updating_programmatically = False
        
        # Emit the operations change signal
        self.operations_changed.emit(self.selected_operations)


class ProcessingDropZone(RowFrame):
    """
    Row 2: Processing drop zone (14% height).
    Windows Explorer-style bare bones box/frame for temporary file drops.
    """
    
    files_dropped = pyqtSignal(list)  # Signal when files are dropped
    
    def __init__(self):
        super().__init__(14, "Drop Files Here for Processing")
        self.dropped_files = []
        self.processing_status = "ready"  # ready, processing, readonly
        self.is_processing = False
        self.setup_drop_zone()
    
    def setup_drop_zone(self):
        """Setup the drop zone with all requirements."""
        layout = self.layout()
        
        # Main container
        container = QWidget()
        container_layout = QVBoxLayout()
        
        # Status indicator with busy icon support
        self.status_label = QLabel("⏸ Ready - Drop image files here to process")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding: 10px;
                background-color: #f0f0f0;
                border: 2px dashed #ccc;
                border-radius: 5px;
                color: #666;
            }
        """)
        container_layout.addWidget(self.status_label)
        
        # File list - Windows Explorer style, bare bones
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(80)
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #ddd;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 2px 5px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
        """)
        container_layout.addWidget(self.file_list)
        
        container.setLayout(container_layout)
        layout.addWidget(container)
        
        # Enable drag and drop initially
        self.setAcceptDrops(True)
        
        # Default state: empty/blank/null as required
        self.clear_files()
    
    def set_readonly_state(self, readonly: bool):
        """Set the zone to readonly state when no checkboxes selected or during processing."""
        if readonly:
            self.setAcceptDrops(False)
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    padding: 10px;
                    background-color: #f5f5f5;
                    border: 2px dashed #bbb;
                    border-radius: 5px;
                    color: #999;
                }
            """)
            if not self.is_processing:
                self.status_label.setText("⚠ Inactive - Select processing options first")
        else:
            self.setAcceptDrops(True)
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    padding: 10px;
                    background-color: #f0f0f0;
                    border: 2px dashed #ccc;
                    border-radius: 5px;
                    color: #666;
                }
            """)
            if not self.is_processing:
                self.status_label.setText("⏸ Ready - Drop image files here to process")
    
    def set_processing_state(self, processing: bool):
        """Set processing state with busy indicator."""
        self.is_processing = processing
        
        if processing:
            # Show busy state with cursor and visual indicator
            self.setCursor(Qt.CursorShape.BusyCursor)
            self.status_label.setText("⏳ PROCESSING - Please wait...")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    padding: 10px;
                    background-color: #fff3cd;
                    border: 2px solid #ffc107;
                    border-radius: 5px;
                    color: #856404;
                    font-weight: bold;
                }
            """)
            self.setAcceptDrops(False)
        else:
            # Reset to normal cursor and state
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.set_readonly_state(False)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event with validation."""
        if not self.acceptDrops():
            event.ignore()
            return
            
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    padding: 10px;
                    background-color: #e3f2fd;
                    border: 2px solid #2196f3;
                    border-radius: 5px;
                    color: #1976d2;
                }
            """)
            self.status_label.setText("📥 Drop files to add to processing queue")
    
    def dragLeaveEvent(self, event: QDragLeaveEvent):
        """Handle drag leave event."""
        if not self.is_processing:
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    padding: 10px;
                    background-color: #f0f0f0;
                    border: 2px dashed #ccc;
                    border-radius: 5px;
                    color: #666;
                }
            """)
            self.status_label.setText("⏸ Ready - Drop image files here to process")
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event with validation."""
        # Validate supported file types
        files = []
        for url in event.mimeData().urls():
            file_path = Path(url.toLocalFile())
            if file_path.is_file() and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}:
                files.append(file_path)
                # Add to Windows Explorer style list
                self.file_list.addItem(f"📄 {file_path.name}")
        
        if files:
            self.dropped_files.extend(files)
            self.files_dropped.emit(files)
            self.update_status(f"Ready - {len(self.dropped_files)} file(s) queued for processing")
        
        event.acceptProposedAction()
        self.dragLeaveEvent(None)  # Reset style
    
    def update_status(self, message: str):
        """Update the status message."""
        if not self.is_processing:
            self.status_label.setText(message)
    
    def clear_files(self):
        """Clear dropped files and reset to default empty state."""
        self.dropped_files.clear()
        self.file_list.clear()
        self.is_processing = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # REQUIREMENT: Default value is blank/empty/null
        self.update_status("⏸ Ready - Drop image files here to process")
    
    def get_dropped_files(self) -> List[Path]:
        """Get list of currently dropped files."""
        return self.dropped_files.copy()


class PickupZone(RowFrame):
    """
    Row 3: Pickup zone for processed files (14% height).
    Windows Explorer-style bare bones box/frame for temporary file pickup.
    """
    
    def __init__(self):
        super().__init__(14, "Processed Files - Drag to Destination")
        self.processed_files = []
        self.setup_pickup_zone()
    
    def setup_pickup_zone(self):
        """Setup the pickup zone with all requirements."""
        layout = self.layout()
        
        # Instructions label
        instructions = QLabel("📤 Processed files will appear here - drag them to destination folders")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #666;
                padding: 3px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
            }
        """)
        layout.addWidget(instructions)
        
        # File list - Windows Explorer style with drag enabled
        self.file_list = QListWidget()
        self.file_list.setDragEnabled(True)
        self.file_list.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #dee2e6;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                alternate-background-color: #f8f9fa;
            }
            QListWidget::item {
                padding: 3px 8px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e3f2fd;
            }
        """)
        layout.addWidget(self.file_list)
        
        # Empty state label
        self.empty_label = QLabel("📁 No processed files yet")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("""
            QLabel {
                color: #999;
                font-style: italic;
                font-size: 12px;
                margin: 20px;
            }
        """)
        layout.addWidget(self.empty_label)
        
        # REQUIREMENT: Default state is blank/empty/null
        self.clear_files()
        
        # Set frame style for Windows Explorer look
        self.setStyleSheet("""
            PickupZone {
                background-color: #fafafa;
                border: 1px solid #ddd;
            }
        """)
    
    def add_processed_files(self, files: List[Path]):
        """
        Add processed files to pickup zone.
        Files remain here and accumulate until user drags them out.
        """
        for file_path in files:
            # Add with processed icon and timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            display_name = f"✅ {file_path.name} (completed at {timestamp})"
            
            item = QListWidgetItem(display_name)
            # Store original file path in item data for drag operations
            item.setData(Qt.ItemDataRole.UserRole, str(file_path))
            self.file_list.addItem(item)
            
            self.processed_files.append(file_path)
        
        # Hide empty label when files are present
        self.empty_label.setVisible(False)
        
        # Update instructions
        file_count = len(self.processed_files)
        instructions_text = f"📤 {file_count} processed file(s) - drag to destination folders or matrix cells"
        self.layout().itemAt(0).widget().setText(instructions_text)
    
    def remove_file(self, file_path: Path):
        """Remove a specific file from the pickup zone (when dragged out)."""
        try:
            self.processed_files.remove(file_path)
            
            # Remove from list widget
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item and Path(item.data(Qt.ItemDataRole.UserRole)) == file_path:
                    self.file_list.takeItem(i)
                    break
            
            # Update display
            if len(self.processed_files) == 0:
                self.clear_files()
            else:
                file_count = len(self.processed_files)
                instructions_text = f"📤 {file_count} processed file(s) - drag to destination folders or matrix cells"
                self.layout().itemAt(0).widget().setText(instructions_text)
                
        except ValueError:
            pass  # File not in list
    
    def clear_files(self):
        """
        Clear all files and reset to default empty state.
        REQUIREMENT: Upon application startup, ensure container is empty.
        Default value is blank/empty/null.
        """
        self.file_list.clear()
        self.processed_files.clear()
        
        # Show empty state
        self.empty_label.setVisible(True)
        
        # Reset instructions
        instructions_text = "📤 Processed files will appear here - drag them to destination folders"
        self.layout().itemAt(0).widget().setText(instructions_text)
    
    def startDrag(self, supportedActions):
        """Handle drag start for processed files."""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        # Create mime data with selected files
        mime_data = QMimeData()
        urls = []
        
        for item in selected_items:
            file_path_str = item.data(Qt.ItemDataRole.UserRole)
            if file_path_str:
                urls.append(QUrl.fromLocalFile(file_path_str))
        
        if urls:
            mime_data.setUrls(urls)
            
            # Start drag operation
            drag = self.file_list.startDrag(supportedActions)
            if drag:
                drag.setMimeData(mime_data)
                drag.exec(supportedActions)
    
    def get_processed_files(self) -> List[Path]:
        """Get list of all processed files currently in pickup zone."""
        return self.processed_files.copy()


class MatrixHeaderRow(QFrame):
    """Row 4: Destination matrix headers (7% height)."""
    
    files_dropped = pyqtSignal(list)  # Signal when files are dropped
    
    def __init__(self):
        super().__init__(7, "Destination Matrix")
        self.setup_headers()
    
    def setup_headers(self):
        """
        Setup matrix column headers.
        
        REQUIREMENTS:
        - 4 cells (columns) wide
        - Corner placeholder cell (top-left)
        - 3 column header drop zones
        - Minimalistic placeholders
        """
        layout = QHBoxLayout()
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)
        
        # Corner placeholder cell (top-left matrix corner)
        self.corner_cell = MatrixCornerCell()
        self.corner_cell.files_dropped.connect(lambda files: self.files_dropped.emit(files, "all"))
        layout.addWidget(self.corner_cell)
        
        # Column header cells (3 columns)
        self.column_headers = []
        for i in range(3):
            col_header = ColumnHeaderCell(i)
            col_header.files_dropped.connect(lambda files, col=i: self.files_dropped.emit(files, f"column_{col}"))
            layout.addWidget(col_header)
            self.column_headers.append(col_header)
        
        self.setLayout(layout)
        

class MatrixCornerCell(QFrame):
    """Top-left corner placeholder cell for dropping to ALL 12 primary cells."""
    
    files_dropped = pyqtSignal(list)  # Signal when files dropped
    
    def __init__(self):
        super().__init__()
        self.setup_cell()
    
    def setup_cell(self):
        """Setup the corner cell as minimalistic placeholder."""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setAcceptDrops(True)
        self.setFixedWidth(100)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(3, 3, 3, 3)
        
        # Placeholder icon
        icon = QLabel("📋")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 18px;")
        layout.addWidget(icon)
        
        # Label
        label = QLabel("All")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 8px; color: #666;")
        layout.addWidget(label)
        
        self.setLayout(layout)
        
        # Style
        self.setStyleSheet("""
            MatrixCornerCell {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            MatrixCornerCell:hover {
                background-color: #e8f4fd;
                border-color: #0078d4;
            }
        """)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                MatrixCornerCell {
                    background-color: #fff3cd;
                    border: 2px solid #ffc107;
                    border-radius: 3px;
                }
            """)
    
    def dragLeaveEvent(self, event: QDragLeaveEvent):
        """Handle drag leave event."""
        self.setStyleSheet("""
            MatrixCornerCell {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event - files go to ALL 12 primary cells."""
        files = []
        for url in event.mimeData().urls():
            file_path = Path(url.toLocalFile())
            if file_path.is_file():
                files.append(file_path)
        
        if files:
            self.files_dropped.emit(files)
            event.acceptProposedAction()
        
        self.dragLeaveEvent(None)


class ColumnHeaderCell(QFrame):
    """Column header placeholder cell for dropping to all cells in that column."""
    
    files_dropped = pyqtSignal(list)  # Signal when files dropped
    
    def __init__(self, column_index: int):
        super().__init__()
        self.column_index = column_index
        self.setup_cell()
    
    def setup_cell(self):
        """Setup the column header as minimalistic placeholder."""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setAcceptDrops(True)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(3, 3, 3, 3)
        
        # Column icon
        icon = QLabel("📂")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 16px;")
        layout.addWidget(icon)
        
        # Column label
        label = QLabel(f"Col {self.column_index + 1}")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 9px; color: #666; font-weight: bold;")
        layout.addWidget(label)
        
        self.setLayout(layout)
        
        # Style
        self.setStyleSheet("""
            ColumnHeaderCell {
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
                border-radius: 3px;
            }
            ColumnHeaderCell:hover {
                background-color: #bbdefb;
                border-color: #2196f3;
            }
        """)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                ColumnHeaderCell {
                    background-color: #c8e6c9;
                    border: 2px solid #4caf50;
                    border-radius: 3px;
                }
            """)
    
    def dragLeaveEvent(self, event: QDragLeaveEvent):
        """Handle drag leave event."""
        self.setStyleSheet("""
            ColumnHeaderCell {
                background-color: #e3f2fd;
                border: 1px solid #90caf9;
                border-radius: 3px;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event - files go to all cells in this column."""
        files = []
        for url in event.mimeData().urls():
            file_path = Path(url.toLocalFile())
            if file_path.is_file():
                files.append(file_path)
        
        if files:
            self.files_dropped.emit(files)
            event.acceptProposedAction()
        
        self.dragLeaveEvent(None)


class RowHeaderCell(QFrame):
    """Row header placeholder cell for dropping to all cells in that row."""
    
    files_dropped = pyqtSignal(list)  # Signal when files dropped
    
    def __init__(self, row_index: int):
        super().__init__()
        self.row_index = row_index
        self.setup_cell()
    
    def setup_cell(self):
        """Setup the row header as minimalistic placeholder."""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setAcceptDrops(True)
        self.setFixedWidth(100)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(3, 3, 3, 3)
        
        # Row icon
        icon = QLabel("📁")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 14px;")
        layout.addWidget(icon)
        
        # Row label
        label = QLabel(f"Row {self.row_index + 1}")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 8px; color: #666; font-weight: bold;")
        layout.addWidget(label)
        
        self.setLayout(layout)
        
        # Style
        self.setStyleSheet("""
            RowHeaderCell {
                background-color: #fff3e0;
                border: 1px solid #ffcc02;
                border-radius: 3px;
            }
            RowHeaderCell:hover {
                background-color: #ffe0b2;
                border-color: #ff9800;
            }
        """)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                RowHeaderCell {
                    background-color: #f3e5f5;
                    border: 2px solid #9c27b0;
                    border-radius: 3px;
                }
            """)
    
    def dragLeaveEvent(self, event: QDragLeaveEvent):
        """Handle drag leave event."""
        self.setStyleSheet("""
            RowHeaderCell {
                background-color: #fff3e0;
                border: 1px solid #ffcc02;
                border-radius: 3px;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event - files go to all cells in this row."""
        files = []
        for url in event.mimeData().urls():
            file_path = Path(url.toLocalFile())
            if file_path.is_file():
                files.append(file_path)
        
        if files:
            self.files_dropped.emit(files)
            event.acceptProposedAction()
        
        self.dragLeaveEvent(None)


class EnhancedDestinationCell(QFrame):
    """
    Enhanced destination cell with Windows Explorer-style interface.
    
    REQUIREMENTS:
    - Windows Explorer-style frame/container
    - Path display with precedence on tail end
    - File counts: Total, JPG/JPEG, PNG
    - Name column only with alphabetical sorting
    - Browse/navigate/paste path functionality
    - Copy from Frame A vs Move from Frame B Row 3
    """
    
    files_dropped = pyqtSignal(list, int, int, str)  # files, row, col, source_type
    
    def __init__(self, row: int, col: int):
        super().__init__()
        self.row = row
        self.col = col
        self.destination_path = None
        self.current_files = []
        self.sort_ascending = True
        self.setup_cell()
    
    def setup_cell(self):
        """Setup the enhanced destination cell."""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)
        
        # Header with path and controls
        header_widget = self.create_header()
        layout.addWidget(header_widget)
        
        # File statistics
        self.stats_label = QLabel("Total: 0 | JPG: 0 | PNG: 0")
        self.stats_label.setStyleSheet("font-size: 8px; color: #666; background-color: #f8f9fa; padding: 2px;")
        layout.addWidget(self.stats_label)
        
        # File list (Name column only)
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(60)
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #dee2e6;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 9px;
            }
            QListWidget::item {
                padding: 1px 3px;
                border-bottom: 1px solid #f0f0f0;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
        """)
        layout.addWidget(self.file_list)
        
        self.setLayout(layout)
        
        # Style
        self.setStyleSheet("""
            EnhancedDestinationCell {
                background-color: #fafafa;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            EnhancedDestinationCell:hover {
                border-color: #aaa;
            }
        """)
    
    def create_header(self) -> QWidget:
        """Create the header with path display and controls."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        
        # Controls row
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(2)
        
        # Browse button
        self.browse_btn = QPushButton("📂")
        self.browse_btn.setMaximumSize(16, 16)
        self.browse_btn.setToolTip("Browse for folder")
        self.browse_btn.clicked.connect(self.browse_destination)
        controls_layout.addWidget(self.browse_btn)
        
        # Sort toggle button
        self.sort_btn = QPushButton("↑")
        self.sort_btn.setMaximumSize(16, 16)
        self.sort_btn.setToolTip("Toggle sort order")
        self.sort_btn.clicked.connect(self.toggle_sort)
        controls_layout.addWidget(self.sort_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Path display with scrolling capability
        self.path_display = QLineEdit()
        self.path_display.setPlaceholderText("No destination set")
        self.path_display.setStyleSheet("""
            QLineEdit {
                font-size: 8px;
                padding: 1px 2px;
                border: 1px solid #ccc;
                background-color: #f8f9fa;
            }
        """)
        self.path_display.textChanged.connect(self.on_path_changed)
        layout.addWidget(self.path_display)
        
        widget.setLayout(layout)
        return widget
    
    def browse_destination(self):
        """Browse for destination folder."""
        folder = QFileDialog.getExistingDirectory(
            self, f"Select Destination for Cell [{self.row+1}, {self.col+1}]", 
            str(self.destination_path) if self.destination_path else "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.set_destination_path(Path(folder))
    
    def set_destination_path(self, path: Path):
        """Set the destination path and update display."""
        self.destination_path = path
        
        # Update path display with precedence on tail end (right-hand portion)
        path_str = str(path)
        self.path_display.setText(path_str)
        
        # Scroll to show the right end of the path
        self.path_display.setCursorPosition(len(path_str))
        
        # Update file list and statistics
        self.refresh_file_list()
    
    def on_path_changed(self):
        """Handle manual path input."""
        path_text = self.path_display.text().strip()
        if path_text and Path(path_text).exists() and Path(path_text).is_dir():
            self.destination_path = Path(path_text)
            self.refresh_file_list()
    
    def refresh_file_list(self):
        """Refresh the file list and update statistics."""
        self.file_list.clear()
        self.current_files.clear()
        
        if not self.destination_path or not self.destination_path.exists():
            self.stats_label.setText("Total: 0 | JPG: 0 | PNG: 0")
            return
        
        # Get image files
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
        image_files = []
        
        try:
            for file_path in self.destination_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    image_files.append(file_path)
        except Exception:
            self.stats_label.setText("Total: ? | JPG: ? | PNG: ?")
            return
        
        # Sort by name (default: alphabetical ascending)
        image_files.sort(key=lambda f: f.name.lower(), reverse=not self.sort_ascending)
        self.current_files = image_files
        
        # Update file list (Name column only)
        for file_path in image_files:
            self.file_list.addItem(file_path.name)
        
        # Update statistics
        total_count = len(image_files)
        jpg_count = sum(1 for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg'})
        png_count = sum(1 for f in image_files if f.suffix.lower() == '.png')
        
        self.stats_label.setText(f"Total: {total_count} | JPG: {jpg_count} | PNG: {png_count}")
    
    def toggle_sort(self):
        """Toggle sort order between ascending and descending."""
        self.sort_ascending = not self.sort_ascending
        self.sort_btn.setText("↑" if self.sort_ascending else "↓")
        self.sort_btn.setToolTip("Sort ascending" if self.sort_ascending else "Sort descending")
        self.refresh_file_list()
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                EnhancedDestinationCell {
                    background-color: #e8f5e8;
                    border: 2px solid #4caf50;
                    border-radius: 3px;
                }
            """)
    
    def dragLeaveEvent(self, event: QDragLeaveEvent):
        """Handle drag leave event."""
        self.setStyleSheet("""
            EnhancedDestinationCell {
                background-color: #fafafa;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        """
        Handle drop event.
        
        REQUIREMENTS:
        - Copy from Frame A (file copy)
        - Move from Frame B Row 3 (file move from pickup zone)
        """
        if not self.destination_path:
            QMessageBox.warning(self, "No Destination", 
                              f"Please set a destination folder for cell [{self.row+1}, {self.col+1}] first.")
            return
        
        files = []
        for url in event.mimeData().urls():
            file_path = Path(url.toLocalFile())
            if file_path.is_file():
                files.append(file_path)
        
        if files:
            # Determine source type based on sender
            source_type = "frame_a"  # Default to copy
            
            # Check if drag came from pickup zone (Frame B Row 3)
            source_widget = event.source()
            if source_widget and hasattr(source_widget, 'parent'):
                parent_widget = source_widget.parent()
                while parent_widget:
                    if isinstance(parent_widget, PickupZone):
                        source_type = "pickup_zone"
                        break
                    parent_widget = parent_widget.parent()
            
            self.files_dropped.emit(files, self.row, self.col, source_type)
            event.acceptProposedAction()
        
        self.dragLeaveEvent(None)


class DestinationMatrix(RowFrame):
    """
    Rows 5-8: Complete 4x5 destination matrix (56% height total).
    
    REQUIREMENTS:
    - 4 rows of data cells (handled here in Rows 5-8)
    - Each row: 1 row header + 3 primary destination cells  
    - Each cell: 25% width, 100% row height
    - Total: 12 primary destination cells (3×4 matrix)
    """
    
    files_copied = pyqtSignal(list, int, int, str)  # files, row, col, operation_type
    header_drop = pyqtSignal(list, str)  # files, header_type
    
    def __init__(self):
        super().__init__(56, "")  # 56% for all 4 rows combined
        self.destination_cells = {}  # (row, col) -> EnhancedDestinationCell
        self.row_headers = {}  # row -> RowHeaderCell
        self.setup_matrix()
    
    def setup_matrix(self):
        """
        Setup the complete 4×5 destination matrix.
        
        REQUIREMENTS:
        - 4 rows of data cells (handled here in Rows 5-8)
        - Each row: 1 row header + 3 primary destination cells  
        - Each cell: 25% width, 100% row height
        - Total: 12 primary destination cells (3×4 matrix)
        """
        layout = self.layout()
        
        # Matrix container
        matrix_widget = QWidget()
        matrix_layout = QGridLayout()
        matrix_layout.setSpacing(2)
        matrix_layout.setContentsMargins(3, 3, 3, 3)
        
        # Create 4 data rows
        for row in range(4):
            # Row header (25% width, 100% height)
            row_header = RowHeaderCell(row)
            row_header.files_dropped.connect(lambda files, r=row: self.on_row_header_drop(files, r))
            matrix_layout.addWidget(row_header, row, 0)
            self.row_headers[row] = row_header
            
            # Primary destination cells for this row (3 cells, each 25% width, 100% height)
            for col in range(3):
                cell = EnhancedDestinationCell(row, col)
                cell.files_dropped.connect(self.on_files_dropped_to_cell)
                matrix_layout.addWidget(cell, row, col + 1)
                self.destination_cells[(row, col)] = cell
        
        # Set column stretch factors for proper 25% width allocation
        for col in range(4):  # 4 columns: 1 header + 3 cells
            matrix_layout.setColumnStretch(col, 1)  # Equal distribution = 25% each
        
        # Set row stretch factors for proper height allocation  
        for row in range(4):  # 4 rows
            matrix_layout.setRowStretch(row, 1)  # Equal distribution = 25% each of 56% total
        
        matrix_widget.setLayout(matrix_layout)
        layout.addWidget(matrix_widget)
    
    def on_files_dropped_to_cell(self, files: List[Path], row: int, col: int, source_type: str):
        """Handle files dropped to a specific destination cell."""
        cell = self.destination_cells.get((row, col))
        if not cell or not cell.destination_path:
            return
        
        import shutil
        success_count = 0
        
        for file_path in files:
            try:
                dest_file_path = cell.destination_path / file_path.name
                
                if source_type == "pickup_zone":
                    # Move operation
                    shutil.move(str(file_path), str(dest_file_path))
                else:
                    # Copy operation  
                    shutil.copy2(str(file_path), str(dest_file_path))
                
                success_count += 1
            except Exception:
                pass
        
        # Update cell display
        cell.refresh_file_list()
        self.files_copied.emit(files, row, col, source_type)
    
    def on_row_header_drop(self, files: List[Path], row: int):
        """Handle files dropped on row header - distribute to all cells in row."""
        for col in range(3):
            cell = self.destination_cells.get((row, col))
            if cell and cell.destination_path:
                self.on_files_dropped_to_cell(files, row, col, "frame_a")


class MaximizedImageProcessingGUI(QMainWindow):
    """Main GUI window optimized for 32" Samsung monitor (1920x1080) - Always Maximized."""
    
    def __init__(self):
        super().__init__()
        
        # Force maximized state for optimal 32" monitor usage
        self.setWindowState(Qt.WindowState.WindowMaximized)
        
        # Initialize components
        self.config = self.load_config()
        self.logger = None
        self.processor = None
        self.metadata_handler = None
        self.processing_thread = None
        self.selected_files = []
        
        self.setup_ui()
        self.setup_processor()
    
    def load_config(self) -> dict:
        """Load configuration from file."""
        config_path = Path("config/config.toml")
        if config_path.exists():
            try:
                import tomli
                with open(config_path, "rb") as f:
                    return tomli.load(f)
            except:
                pass
        return {}
    
    def setup_processor(self):
        """Setup the image processor and metadata handler."""
        try:
            # Setup logging
            admin_path = Path("./admin_output")
            admin_path.mkdir(exist_ok=True)
            self.logger = setup_logging(admin_path, self.config)
            
            # Setup metadata handler
            self.metadata_handler = MetadataHandler(self.config, self.logger)
            
            # Set metadata handler in file explorer
            if hasattr(self, 'file_explorer'):
                self.file_explorer.set_metadata_handler(self.metadata_handler)
            
        except Exception as e:
            QMessageBox.critical(self, "Setup Error", f"Failed to initialize processor: {e}")
    
    def setup_ui(self):
        """Setup the main UI with Frame A and Frame B layout."""
        # Set window properties for 32" monitor optimization
        self.setWindowTitle("Image Processing Application - Optimized for 32\" Samsung Monitor")
        self.setMinimumSize(1920, 1080)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout: Frame A (50%) | Frame B (50%)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        central_widget.setLayout(main_layout)
        
        # Create Frame A (left side - 50% width)
        frame_a = self.create_frame_a()
        main_layout.addWidget(frame_a, 1)  # Stretch factor 1 = 50%
        
        # Create Frame B (right side - 50% width) 
        frame_b = self.create_frame_b()
        main_layout.addWidget(frame_b, 1)  # Stretch factor 1 = 50%
        
        # Setup menu and status bar
        self.setup_menu_bar()
        self.statusBar().showMessage("Ready - Optimized for 32\" Monitor @ 1920x1080")
    
    def create_frame_a(self) -> QFrame:
        """Create Frame A - Windows Explorer-like file browser."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setLineWidth(2)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("Frame A - Source Files Explorer")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px; background-color: #e3f2fd;")
        layout.addWidget(title)
        
        # Enhanced file explorer
        self.file_explorer = EnhancedFileExplorer()
        layout.addWidget(self.file_explorer)
        
        frame.setLayout(layout)
        return frame
    
    def create_frame_b(self) -> QFrame:
        """Create Frame B - Processing and destinations."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setLineWidth(2)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # Title
        title = QLabel("Frame B - Processing & Destinations")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px; background-color: #fff3e0;")
        layout.addWidget(title)
        
        # Row 1: Processing controls (7% height)
        self.processing_controls = ProcessingControlsRow()
        layout.addWidget(self.processing_controls, 1)
        
        # Row 2: Processing drop zone (14% height)
        self.processing_zone = ProcessingDropZone()
        layout.addWidget(self.processing_zone, 2)
        
        # Row 3: Pickup zone (14% height)
        self.pickup_zone = PickupZone()
        layout.addWidget(self.pickup_zone, 2)
        
        # Row 4: Matrix header (7% height)
        self.matrix_header = MatrixHeaderRow()
        layout.addWidget(self.matrix_header, 1)
        
        # Rows 5-8: Destination matrix (56% height total)
        self.destination_matrix = DestinationMatrix()
        layout.addWidget(self.destination_matrix, 8)
        
        frame.setLayout(layout)
        return frame
    
    def setup_menu_bar(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        file_menu.addAction('Exit', self.close)


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Image Processing GUI")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = MaximizedImageProcessingGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()