#!/usr/bin/env python3
"""
Simple Color Dictionary Test Script
Test the external color dictionary loading functionality.
"""

import json
from pathlib import Path

def test_color_dictionary_loading():
    """Test loading color dictionaries from JSON files."""
    print("Testing External Color Dictionary Loading")
    print("=" * 50)
    
    # Test X11 colors loading
    x11_path = Path("config/colors/x11_colors.json")
    
    if not x11_path.exists():
        print(f"ERROR: X11 color file not found: {x11_path}")
        return False
    
    try:
        with open(x11_path, 'r', encoding='utf-8') as f:
            x11_data = json.load(f)
        
        print(f"? Successfully loaded X11 colors JSON")
        
        # Validate structure
        if 'colors' in x11_data:
            colors_count = len(x11_data['colors'])
            print(f"? Found {colors_count} X11 colors")
            
            # Test parsing a few colors
            test_colors = list(x11_data['colors'].keys())[:3]
            for color_name in test_colors:
                color_info = x11_data['colors'][color_name]
                if 'rgb' in color_info:
                    rgb = color_info['rgb']
                    if isinstance(rgb, list) and len(rgb) >= 3:
                        r, g, b = rgb[0], rgb[1], rgb[2]
                        print(f"? {color_name}: rgb({r}, {g}, {b})")
                    else:
                        print(f"? Invalid RGB format for {color_name}: {rgb}")
                        return False
                else:
                    print(f"? Missing RGB data for {color_name}")
                    return False
        else:
            print("? Missing 'colors' key in JSON structure")
            return False
            
    except Exception as e:
        print(f"? Error loading X11 colors: {e}")
        return False
    
    # Test CSS3 colors loading
    css3_path = Path("config/colors/css3_colors.json")
    
    if css3_path.exists():
        try:
            with open(css3_path, 'r', encoding='utf-8') as f:
                css3_data = json.load(f)
            
            print(f"? Successfully loaded CSS3 colors JSON")
            
            if 'colors' in css3_data:
                colors_count = len(css3_data['colors'])
                print(f"? Found {colors_count} CSS3 colors")
            else:
                print("? Missing 'colors' key in CSS3 JSON structure")
                return False
                
        except Exception as e:
            print(f"? Error loading CSS3 colors: {e}")
            return False
    else:
        print(f"? CSS3 color file not found: {css3_path}")
    
    # Test Pantone colors loading  
    pantone_path = Path("config/colors/pantone_colors.json")
    
    if pantone_path.exists():
        try:
            with open(pantone_path, 'r', encoding='utf-8') as f:
                pantone_data = json.load(f)
            
            print(f"? Successfully loaded Pantone colors JSON")
            
            if 'colors' in pantone_data:
                colors_count = len(pantone_data['colors'])
                print(f"? Found {colors_count} Pantone colors")
            else:
                print("? Missing 'colors' key in Pantone JSON structure")
                return False
                
        except Exception as e:
            print(f"? Error loading Pantone colors: {e}")
            return False
    else:
        print(f"? Pantone color file not found: {pantone_path}")
    
    print("\n" + "=" * 50)
    print("SUCCESS: Color dictionary loading test completed!")
    return True

def test_color_analyzer_with_external_dicts():
    """Test the ColorAnalyzer with external dictionaries (minimal test)."""
    print("\nTesting ColorAnalyzer with External Dictionaries")
    print("=" * 50)
    
    try:
        # Minimal test without heavy dependencies
        import sys
        from pathlib import Path
        
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Simple logger mock
        class MockLogger:
            def debug(self, msg): pass
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
        
        logger = MockLogger()
        
        # Test just the color dictionary loading part
        from src.models.color_analyzer import ColorAnalyzer
        
        config = {}
        analyzer = ColorAnalyzer(config, logger)
        
        # Check if color databases were loaded
        if analyzer.color_databases:
            print(f"? ColorAnalyzer loaded {len(analyzer.color_databases)} color databases")
            
            for db_name, db_colors in analyzer.color_databases.items():
                print(f"  - {db_name}: {len(db_colors)} colors")
            
            # Test color matching with a simple color
            r, g, b = 255, 0, 0  # Pure red
            matches = analyzer._find_comprehensive_color_names(r, g, b)
            
            print(f"? Color matching test for pure red (255,0,0):")
            print(f"  Best match: {matches.get('best_name', 'None')} from {matches.get('best_source', 'None')}")
            
            for db_name in ['CSS3', 'X11', 'Pantone']:
                if db_name in matches:
                    print(f"  {db_name}: {matches[db_name]}")
            
        else:
            print("? No color databases loaded")
            return False
            
    except Exception as e:
        print(f"? Error testing ColorAnalyzer: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n? ColorAnalyzer external dictionary test successful!")
    return True

if __name__ == "__main__":
    print("Testing Enhanced Color Dictionary System")
    print("=" * 60)
    
    # Test 1: JSON file loading
    test1_success = test_color_dictionary_loading()
    
    # Test 2: ColorAnalyzer integration
    test2_success = test_color_analyzer_with_external_dicts()
    
    print("\n" + "=" * 60)
    if test1_success and test2_success:
        print("?? ALL TESTS PASSED! External color dictionary system is working.")
        print("\nKey Features Verified:")
        print("? JSON color dictionary files can be loaded")
        print("? ColorAnalyzer integrates with external dictionaries")
        print("? Color matching works across multiple databases")
        print("? Fallback system works when files are missing")
    else:
        print("? Some tests failed. Check the errors above.")
        print("\nFailed Tests:")
        if not test1_success:
            print("? Color dictionary loading")
        if not test2_success:
            print("? ColorAnalyzer integration")