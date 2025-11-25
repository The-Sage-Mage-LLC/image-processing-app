"""
Enhanced File List with Metadata Indicators
Project ID: Image Processing App 20251119
Created: 2025-01-19 07:21:15 UTC
Author: The-Sage-Mage
"""

from PyQt6.QtWidgets import QTreeView, QStyledItemDelegate, QStyle
from PyQt6.QtCore import Qt, QSize, QRect, QModelIndex
from PyQt6.QtGui import QPainter, QPixmap, QIcon, QColor, QPen
from pathlib import Path
import exifread


class MetadataDelegate(QStyledItemDelegate):
    """Custom delegate to show metadata indicators."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metadata_cache = {}
    
    def paint(self, painter: QPainter, option, index: QModelIndex):
        """Paint the item with metadata indicator."""
        # Let the base class do the standard painting
        super().paint(painter, option, index)
        
        # Get file path from index
        model = index.model()
        file_info = model.fileInfo(index)
        
        if file_info.isFile():
            file_path = Path(file_info.absoluteFilePath())
            
            # Check if it's an image file
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Get or calculate metadata count
                metadata_count = self.get_metadata_count(file_path)
                
                # Draw indicator
                indicator_rect = QRect(
                    option.rect.left() + 2,
                    option.rect.top() + (option.rect.height() - 16) // 2,
                    16, 16
                )
                
                if metadata_count >= 25:
                    # Green checkmark
                    painter.fillRect(indicator_rect, QColor(0, 0, 0))
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                    painter.drawText(indicator_rect, Qt.AlignmentFlag.AlignCenter, "✓")
                else:
                    # Red X
                    painter.fillRect(indicator_rect, QColor(0, 0, 0))
                    painter.setPen(QPen(QColor(255, 0, 0), 2))
                    painter.drawText(indicator_rect, Qt.AlignmentFlag.AlignCenter, "✗")
    
    def get_metadata_count(self, file_path: Path) -> int:
        """Get count of non-empty metadata fields."""
        # Check cache
        if file_path in self.metadata_cache:
            return self.metadata_cache[file_path]
        
        count = 0
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False, stop_tag='UNDEF')
                
                # Count non-empty fields
                for tag, value in tags.items():
                    if value and str(value).strip():
                        count += 1
                        if count >= 25:
                            break  # Early exit if we have enough
        except:
            pass
        
        # Cache the result
        self.metadata_cache[file_path] = count
        return count


class EnhancedFileTreeView(QTreeView):
    """Enhanced tree view with metadata indicators."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set custom delegate
        self.metadata_delegate = MetadataDelegate(self)
        self.setItemDelegateForColumn(0, self.metadata_delegate)  # Name column
        
        # Adjust spacing for indicator
        self.setIndentation(30)