#!/usr/bin/env python3
"""
Comprehensive Checkbox Behavior Test Script
Project ID: Image Processing App 20251119
Created: 2025-01-19
Author: GitHub Copilot - The-Sage-Mage

This script tests that ALL checkbox requirements from part two are properly implemented.
"""

import sys
from pathlib import Path
import traceback

def test_checkbox_requirements():
    """Test all checkbox requirements from part two specifications."""
    print("="*80)
    print("COMPREHENSIVE CHECKBOX REQUIREMENTS TEST")
    print("Testing Part Two Requirements - Seven Checkboxes")
    print("="*80)
    
    all_passed = True
    
    # Test 1: Import and basic setup
    try:
        from src.gui.enhanced_processing_controls import EnhancedProcessingControlsRow
        print("? Enhanced checkbox controls import successfully")
    except Exception as e:
        print(f"? Enhanced checkbox import failed: {e}")
        return False
    
    # Test 2: Checkbox labels and tooltips
    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        controls = EnhancedProcessingControlsRow()
        
        # Test checkbox labels and tooltips match exact specifications
        expected_specs = [
            ("All", "All six menu items"),
            ("BWG", "Black and White (grayscale)"),
            ("SEP", "Sepia-toned"),
            ("PSK", "Pencil Sketch"),
            ("BK_CLR", "Coloring book"),
            ("BK_CTD", "Connect-the-dots activity book"),
            ("BK_CBN", "Color-by-numbers activity book")
        ]
        
        for label, tooltip in expected_specs:
            if label in controls.checkboxes:
                checkbox = controls.checkboxes[label]
                actual_tooltip = checkbox.toolTip()
                if actual_tooltip == tooltip:
                    print(f"? Checkbox '{label}' has correct tooltip: '{tooltip}'")
                else:
                    print(f"? Checkbox '{label}' tooltip mismatch. Expected: '{tooltip}', Got: '{actual_tooltip}'")
                    all_passed = False
            else:
                print(f"? Checkbox '{label}' not found")
                all_passed = False
        
    except Exception as e:
        print(f"? Label and tooltip test failed: {e}")
        all_passed = False
    
    # Test 3: Default state (all unchecked)
    try:
        controls.reset_to_defaults()
        for label, checkbox in controls.checkboxes.items():
            if checkbox.isChecked():
                print(f"? Checkbox '{label}' should default to unchecked")
                all_passed = False
        print("? All checkboxes default to unchecked state")
    except Exception as e:
        print(f"? Default state test failed: {e}")
        all_passed = False
    
    # Test 4: "All" checkbox behavior - checking
    try:
        controls.reset_to_defaults()
        controls.checkboxes["All"].setChecked(True)
        
        # Verify all 6 remaining checkboxes are checked
        individual_boxes = ["BWG", "SEP", "PSK", "BK_CLR", "BK_CTD", "BK_CBN"]
        for box_name in individual_boxes:
            if not controls.checkboxes[box_name].isChecked():
                print(f"? Checkbox '{box_name}' should be auto-checked when 'All' is checked")
                all_passed = False
        print("? Checking 'All' auto-checks all 6 remaining checkboxes")
        
        # Verify operations are set correctly
        expected_ops = {"grayscale", "sepia", "pencil_sketch", "coloring_book", "connect_dots", "color_by_numbers"}
        actual_ops = controls.get_selected_operations()
        if actual_ops == expected_ops:
            print("? Checking 'All' sets all 6 operations correctly")
        else:
            print(f"? Operations mismatch. Expected: {expected_ops}, Got: {actual_ops}")
            all_passed = False
            
    except Exception as e:
        print(f"? 'All' checkbox checking test failed: {e}")
        all_passed = False
    
    # Test 5: "All" checkbox behavior - unchecking
    try:
        # Start with all checked
        controls.checkboxes["All"].setChecked(True)
        # Then uncheck "All"
        controls.checkboxes["All"].setChecked(False)
        
        # Verify all 6 remaining checkboxes are unchecked
        individual_boxes = ["BWG", "SEP", "PSK", "BK_CLR", "BK_CTD", "BK_CBN"]
        for box_name in individual_boxes:
            if controls.checkboxes[box_name].isChecked():
                print(f"? Checkbox '{box_name}' should be auto-unchecked when 'All' is unchecked")
                all_passed = False
        print("? Unchecking 'All' auto-unchecks all 6 remaining checkboxes")
        
        # Verify no operations are selected
        actual_ops = controls.get_selected_operations()
        if len(actual_ops) == 0:
            print("? Unchecking 'All' clears all operations correctly")
        else:
            print(f"? Operations should be empty. Got: {actual_ops}")
            all_passed = False
            
    except Exception as e:
        print(f"? 'All' checkbox unchecking test failed: {e}")
        all_passed = False
    
    # Test 6: Individual checkbox auto-unchecks "All"
    try:
        # Start with all checked via "All"
        controls.reset_to_defaults()
        controls.checkboxes["All"].setChecked(True)
        
        # Uncheck one individual checkbox
        controls.checkboxes["BWG"].setChecked(False)
        
        # Verify "All" is auto-unchecked
        if controls.checkboxes["All"].isChecked():
            print("? 'All' checkbox should be auto-unchecked when individual box is unchecked")
            all_passed = False
        else:
            print("? Unchecking individual checkbox auto-unchecks 'All'")
        
        # Verify operation is removed
        actual_ops = controls.get_selected_operations()
        if "grayscale" in actual_ops:
            print("? Grayscale operation should be removed when BWG is unchecked")
            all_passed = False
        else:
            print("? Unchecking BWG removes grayscale operation")
            
    except Exception as e:
        print(f"? Individual checkbox unchecking test failed: {e}")
        all_passed = False
    
    # Test 7: Manually checking all 6 auto-checks "All"
    try:
        controls.reset_to_defaults()
        
        # Manually check all 6 individual checkboxes
        individual_boxes = ["BWG", "SEP", "PSK", "BK_CLR", "BK_CTD", "BK_CBN"]
        for box_name in individual_boxes:
            controls.checkboxes[box_name].setChecked(True)
        
        # Verify "All" is auto-checked
        if not controls.checkboxes["All"].isChecked():
            print("? 'All' checkbox should be auto-checked when all 6 individual boxes are manually checked")
            all_passed = False
        else:
            print("? Manually checking all 6 individual boxes auto-checks 'All'")
        
        # Verify all operations are selected
        actual_ops = controls.get_selected_operations()
        if len(actual_ops) == 6:
            print("? Manually checking all 6 boxes selects all 6 operations")
        else:
            print(f"? Should have 6 operations selected. Got: {len(actual_ops)}")
            all_passed = False
            
    except Exception as e:
        print(f"? Manual checking test failed: {e}")
        all_passed = False
    
    # Test 8: Menu item mapping
    try:
        controls.reset_to_defaults()
        controls.checkboxes["All"].setChecked(True)
        
        menu_items = controls.get_selected_menu_items()
        expected_items = [7, 8, 9, 10, 11, 12]  # Command-line menu items
        
        if sorted(menu_items) == expected_items:
            print("? Menu item mapping correct (CLI items 7-12)")
        else:
            print(f"? Menu item mapping incorrect. Expected: {expected_items}, Got: {sorted(menu_items)}")
            all_passed = False
            
    except Exception as e:
        print(f"? Menu item mapping test failed: {e}")
        all_passed = False
    
    # Test 9: Command-line correspondence
    try:
        mapping = controls.get_menu_item_mapping()
        expected_mapping = {
            "grayscale": 7,      # BWG -> Menu item 7
            "sepia": 8,          # SEP -> Menu item 8  
            "pencil_sketch": 9,  # PSK -> Menu item 9
            "coloring_book": 10, # BK_CLR -> Menu item 10
            "connect_dots": 11,  # BK_CTD -> Menu item 11
            "color_by_numbers": 12 # BK_CBN -> Menu item 12
        }
        
        if mapping == expected_mapping:
            print("? Command-line correspondence correct (operations map to menu items 7-12)")
        else:
            print(f"? Command-line correspondence incorrect")
            all_passed = False
            
    except Exception as e:
        print(f"? Command-line correspondence test failed: {e}")
        all_passed = False
    
    # Final Results
    print("\n" + "="*80)
    if all_passed:
        print("?? ALL CHECKBOX REQUIREMENTS TESTS PASSED!")
        print("\nImplemented Requirements Summary:")
        print("? Checkbox 1 ('All') controls all 6 remaining checkboxes")
        print("? Auto-check 'All' when all 6 manually selected")
        print("? Auto-uncheck 'All' when any individual unchecked")
        print("? Auto-check all 6 when 'All' checked")
        print("? Auto-uncheck all 6 when 'All' unchecked")
        print("? Exact labels: All, BWG, SEP, PSK, BK_CLR, BK_CTD, BK_CBN")
        print("? Exact tooltips as specified")
        print("? Defaults unchecked/off")
        print("? Corresponds to command-line menu items 7-12")
        print("? Individual checkbox behavior with 'All' checkbox")
    else:
        print("? SOME CHECKBOX REQUIREMENTS TESTS FAILED")
        print("Review the failures above")
    
    print("="*80)
    return all_passed

def main():
    """Run comprehensive checkbox requirements test."""
    try:
        result = test_checkbox_requirements()
        if result:
            print("\n? SUCCESS: ALL CHECKBOX REQUIREMENTS FROM PART TWO ARE FULLY IMPLEMENTED!")
        else:
            print("\n? FAILURE: Some checkbox requirements not met")
        return 0 if result else 1
    except Exception as e:
        print(f"\n?? TEST ERROR: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())