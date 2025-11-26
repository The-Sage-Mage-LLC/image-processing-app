"""
Enhanced Processing Controls with Exact Checkbox Requirements
Project ID: Image Processing App 20251119
Created: 2025-01-19 - Full Checkbox Behavior Implementation
Author: GitHub Copilot - The-Sage-Mage

This implementation meets ALL specified checkbox requirements from part two:
- Checkbox 1 ("All") controls all 6 remaining checkboxes exactly as specified
- Each individual checkbox has exact behavior with the "All" checkbox
- All tooltip requirements met
- All auto-check/uncheck behaviors implemented
- Corresponds to command-line menu items 7-12
"""

import sys
from pathlib import Path
from typing import Set, List
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

class EnhancedProcessingControlsRow(QFrame):
    """
    Row 1: Seven checkboxes with EXACT requirements implementation (7% height).
    
    Implements all specified checkbox behaviors:
    - Checkbox 1 ("All") controls all 6 remaining checkboxes
    - Auto-check "All" when all 6 individual operations are manually selected
    - Auto-uncheck "All" when any individual operation is unchecked
    - Each checkbox corresponds to specific command-line menu items (7-12)
    - Exact tooltips as specified
    - Defaults unchecked/off on startup
    """
    
    operations_changed = pyqtSignal(set)  # Signal when selected operations change
    
    def __init__(self):
        super().__init__()
        self.selected_operations = set()
        self.checkboxes = {}
        self._updating = False  # Prevent recursion during programmatic updates
        self.setup_controls()
    
    def setup_controls(self):
        """Setup seven checkboxes with EXACT requirements."""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setMinimumHeight(int(1080 * 0.07))  # 7% of 1080p height
        
        layout = QVBoxLayout()
        layout.setContentsMargins(3, 3, 3, 3)
        
        # Title
        title = QLabel("Processing Options")
        title.setStyleSheet("font-weight: bold; padding: 3px; font-size: 12px;")
        layout.addWidget(title)
        
        # Controls container
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(5, 2, 5, 2)
        controls_layout.setSpacing(8)
        
        # EXACT REQUIREMENT: Seven checkboxes from left to right with exact labels and tooltips
        checkbox_specs = [
            # (label, tooltip, operation_key)
            ("All", "All six menu items", "all"),
            ("BWG", "Black and White (grayscale)", "grayscale"),
            ("SEP", "Sepia-toned", "sepia"),
            ("PSK", "Pencil Sketch", "pencil_sketch"),
            ("BK_CLR", "Coloring book", "coloring_book"),
            ("BK_CTD", "Connect-the-dots activity book", "connect_dots"),
            ("BK_CBN", "Color-by-numbers activity book", "color_by_numbers")
        ]
        
        for label, tooltip, operation_key in checkbox_specs:
            cb = QCheckBox(label)
            cb.setToolTip(tooltip)
            cb.setChecked(False)  # REQUIREMENT: defaults unchecked/off
            
            # Store operation key for easy mapping
            cb.setProperty("operation_key", operation_key)
            
            # Connect to enhanced handler
            cb.stateChanged.connect(lambda state, lbl=label: self._on_checkbox_changed(lbl, state))
            controls_layout.addWidget(cb)
            self.checkboxes[label] = cb
        
        controls_layout.addStretch()
        controls_widget.setLayout(controls_layout)
        layout.addWidget(controls_widget)
        
        self.setLayout(layout)
    
    def _on_checkbox_changed(self, checkbox_name: str, state: int):
        """
        Handle checkbox state change with EXACT requirements implementation.
        
        EXACT REQUIREMENTS IMPLEMENTED:
        
        Checkbox 1 ("All"):
        - Upon checking: automatically checks all 6 remaining checkboxes
        - Upon unchecking: automatically clears all 6 remaining checkboxes
        - If user unchecks any of the other 6 after "All" was checked: auto-clears "All"
        - If user manually checks all 6 remaining: auto-checks "All"
        
        Checkboxes 2-7 (BWG, SEP, PSK, BK_CLR, BK_CTD, BK_CBN):
        - If checking results in all 6 being checked: auto-checks "All"
        - If unchecking and "All" was previously selected: auto-unselects "All"
        """
        if self._updating:
            return
            
        self._updating = True
        
        try:
            is_checked = (state == Qt.CheckState.Checked.value)
            
            if checkbox_name == "All":
                # CHECKBOX 1 ("All") BEHAVIOR
                if is_checked:
                    # REQUIREMENT: "Upon the user checking this checkbox, 
                    # the application will automatically check all six (6) remaining checkboxes"
                    for cb_name, cb in self.checkboxes.items():
                        if cb_name != "All":
                            cb.setChecked(True)
                    
                    # Set all 6 operations as selected (menu items 7-12)
                    self.selected_operations = {
                        "grayscale",      # Menu item 7
                        "sepia",          # Menu item 8 
                        "pencil_sketch",  # Menu item 9
                        "coloring_book",  # Menu item 10
                        "connect_dots",   # Menu item 11
                        "color_by_numbers"# Menu item 12
                    }
                else:
                    # REQUIREMENT: "Upon the user unchecking this checkbox,
                    # the application will automatically clear all six (6) remaining checkboxes"
                    for cb_name, cb in self.checkboxes.items():
                        if cb_name != "All":
                            cb.setChecked(False)
                    
                    # Clear all operations
                    self.selected_operations.clear()
            
            else:
                # CHECKBOXES 2-7 INDIVIDUAL BEHAVIOR
                operation_key = self.checkboxes[checkbox_name].property("operation_key")
                
                if is_checked:
                    # Add this operation to selected set
                    self.selected_operations.add(operation_key)
                    
                    # REQUIREMENT: "If, after the user checks this box, this results in 
                    # the system showing all six (6) checkboxes as checked (not the 'All' checkbox), 
                    # then the application will automatically checkmark the 'All' checkbox"
                    if len(self.selected_operations) == 6:
                        self.checkboxes["All"].setChecked(True)
                
                else:
                    # Remove this operation from selected set
                    self.selected_operations.discard(operation_key)
                    
                    # REQUIREMENT: "If the user unchecks this box and the user previously 
                    # selected the 'All' checkbox, then the application will automatically 
                    # unselect the 'All' checkbox"
                    if self.checkboxes["All"].isChecked():
                        self.checkboxes["All"].setChecked(False)
        
        finally:
            self._updating = False
        
        # Emit signal after a brief delay to prevent recursion
        QTimer.singleShot(10, lambda: self.operations_changed.emit(self.selected_operations.copy()))
    
    def reset_to_defaults(self):
        """
        Reset all checkboxes to default (unchecked) state.
        
        REQUIREMENT: "Upon the user newly opening/executing this application, 
        the application sets/resets all checkboxes to the defaults"
        """
        self._updating = True
        
        try:
            for checkbox in self.checkboxes.values():
                checkbox.setChecked(False)
            
            self.selected_operations.clear()
        finally:
            self._updating = False
        
        self.operations_changed.emit(self.selected_operations.copy())
    
    def get_selected_operations(self) -> Set[str]:
        """Get currently selected operations corresponding to menu items 7-12."""
        return self.selected_operations.copy()
    
    def get_menu_item_mapping(self) -> dict:
        """
        Get mapping of operations to command-line menu items.
        
        Returns mapping of selected operations to their corresponding 
        command-line Part 1 menu items as specified in requirements.
        """
        return {
            "grayscale": 7,       # Menu item 7: Black and White (grayscale)
            "sepia": 8,           # Menu item 8: Sepia-toned
            "pencil_sketch": 9,   # Menu item 9: Pencil Sketch (broad tip, large radius, high clarity, low blur, medium strength)
            "coloring_book": 10,  # Menu item 10: Strong outline for coloring book
            "connect_dots": 11,   # Menu item 11: Connect-the-dots style image
            "color_by_numbers": 12 # Menu item 12: Color-by-numbers style image
        }
    
    def get_selected_menu_items(self) -> List[int]:
        """Get list of command-line menu items corresponding to selected operations."""
        mapping = self.get_menu_item_mapping()
        return [mapping[op] for op in self.selected_operations if op in mapping]


def test_checkbox_behavior():
    """
    Test function to verify exact checkbox behavior requirements.
    This can be called to verify all requirements are met.
    """
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create the enhanced controls
    controls = EnhancedProcessingControlsRow()
    
    print("Testing Enhanced Checkbox Behavior:")
    print("="*50)
    
    # Test 1: Default state
    print("Test 1: Default state (all unchecked)")
    controls.reset_to_defaults()
    selected = controls.get_selected_operations()
    assert len(selected) == 0, f"Expected 0 operations, got {len(selected)}"
    print("? PASS: All checkboxes default to unchecked")
    
    # Test 2: Check "All" checkbox
    print("\nTest 2: Check 'All' checkbox behavior")
    controls.checkboxes["All"].setChecked(True)
    selected = controls.get_selected_operations()
    assert len(selected) == 6, f"Expected 6 operations when 'All' checked, got {len(selected)}"
    
    # Verify all individual checkboxes are checked
    individual_boxes = ["BWG", "SEP", "PSK", "BK_CLR", "BK_CTD", "BK_CBN"]
    for box_name in individual_boxes:
        assert controls.checkboxes[box_name].isChecked(), f"Expected {box_name} to be checked"
    print("? PASS: 'All' checkbox checks all 6 individual boxes")
    
    # Test 3: Uncheck one individual box
    print("\nTest 3: Uncheck individual box when 'All' was checked")
    controls.checkboxes["BWG"].setChecked(False)
    assert not controls.checkboxes["All"].isChecked(), "Expected 'All' to be unchecked"
    selected = controls.get_selected_operations()
    assert "grayscale" not in selected, "Expected grayscale operation to be removed"
    print("? PASS: Unchecking individual box auto-unchecks 'All'")
    
    # Test 4: Manually check all 6 individual boxes
    print("\nTest 4: Manually check all 6 individual boxes")
    controls.reset_to_defaults()
    for box_name in individual_boxes:
        controls.checkboxes[box_name].setChecked(True)
    
    assert controls.checkboxes["All"].isChecked(), "Expected 'All' to be auto-checked"
    selected = controls.get_selected_operations()
    assert len(selected) == 6, f"Expected 6 operations, got {len(selected)}"
    print("? PASS: Manually checking all 6 boxes auto-checks 'All'")
    
    # Test 5: Menu item mapping
    print("\nTest 5: Menu item mapping")
    mapping = controls.get_menu_item_mapping()
    expected_mapping = {
        "grayscale": 7,
        "sepia": 8, 
        "pencil_sketch": 9,
        "coloring_book": 10,
        "connect_dots": 11,
        "color_by_numbers": 12
    }
    assert mapping == expected_mapping, f"Menu mapping mismatch"
    print("? PASS: Menu item mapping correct (operations 7-12)")
    
    print("\n" + "="*50)
    print("ALL CHECKBOX BEHAVIOR TESTS PASSED!")
    print("Requirements fully implemented and verified.")
    
    return True


if __name__ == "__main__":
    # Run the test if this file is executed directly
    test_checkbox_behavior()