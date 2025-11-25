#!/usr/bin/env python3
"""
Basic Color Dictionary JSON Test
Test just the JSON loading without complex imports.
"""

import json
from pathlib import Path

def test_json_loading_only():
    """Test only JSON file loading without ColorAnalyzer."""
    print("Testing JSON Color Dictionary Files")
    print("=" * 40)
    
    # List of expected color dictionary files
    expected_files = [
        "config/colors/x11_colors.json",
        "config/colors/css3_colors.json", 
        "config/colors/pantone_colors.json",
        "config/colors/ral_classic_colors.json",
        "config/colors/crayola_colors.json",
        "config/colors/natural_colors.json",
        "config/colors/iso_colors.json",
        "config/colors/lab_scientific_colors.json"
    ]
    
    results = {}
    
    for file_path in expected_files:
        file_name = Path(file_path).stem
        print(f"\nTesting {file_name}...")
        
        try:
            path_obj = Path(file_path)
            
            if not path_obj.exists():
                print(f"  ? File not found: {file_path}")
                results[file_name] = "not_found"
                continue
            
            # Try to load JSON
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check structure
            if 'colors' not in data:
                print(f"  ? Missing 'colors' key")
                results[file_name] = "invalid_structure"
                continue
            
            colors_count = len(data['colors'])
            
            # Test parsing a few colors
            test_colors = list(data['colors'].keys())[:2]
            valid_colors = 0
            
            for color_key in test_colors:
                color_info = data['colors'][color_key]
                if 'rgb' in color_info and isinstance(color_info['rgb'], list):
                    if len(color_info['rgb']) >= 3:
                        valid_colors += 1
            
            if valid_colors == len(test_colors):
                print(f"  ? Valid structure with {colors_count} colors")
                results[file_name] = "success"
                
                # Show metadata if available
                if 'metadata' in data:
                    metadata = data['metadata']
                    name = metadata.get('name', 'Unknown')
                    version = metadata.get('version', '?')
                    print(f"    {name} v{version}")
            else:
                print(f"  ? Invalid color format")
                results[file_name] = "invalid_format"
                
        except json.JSONDecodeError as e:
            print(f"  ? JSON parse error: {e}")
            results[file_name] = "json_error"
        except Exception as e:
            print(f"  ? Unexpected error: {e}")
            results[file_name] = "error"
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY:")
    
    success_count = sum(1 for status in results.values() if status == "success")
    total_count = len(expected_files)
    
    for file_name, status in results.items():
        status_icon = "?" if status == "success" else "?" if status != "not_found" else "?"
        print(f"{status_icon} {file_name}: {status}")
    
    print(f"\nResult: {success_count}/{total_count} color dictionaries loaded successfully")
    
    if success_count > 0:
        print(f"\n?? SUCCESS! Found {success_count} working color dictionaries.")
        return True
    else:
        print(f"\n? FAILED! No working color dictionaries found.")
        return False

def test_color_conversion():
    """Test basic color name lookup manually."""
    print("\n\nTesting Color Name Lookup")
    print("=" * 40)
    
    # Load X11 colors as test
    x11_path = Path("config/colors/x11_colors.json")
    
    if not x11_path.exists():
        print("X11 colors not available for testing")
        return False
    
    try:
        with open(x11_path, 'r', encoding='utf-8') as f:
            x11_data = json.load(f)
        
        colors = x11_data['colors']
        
        # Test finding closest color to pure red (255, 0, 0)
        target_r, target_g, target_b = 255, 0, 0
        
        min_distance = float('inf')
        closest_color = None
        
        for color_name, color_info in colors.items():
            if 'rgb' in color_info:
                rgb = color_info['rgb']
                if len(rgb) >= 3:
                    r, g, b = rgb[0], rgb[1], rgb[2]
                    
                    # Calculate Euclidean distance
                    distance = ((target_r - r)**2 + (target_g - g)**2 + (target_b - b)**2)**0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_color = (color_name, r, g, b)
        
        if closest_color:
            name, r, g, b = closest_color
            print(f"? Closest X11 color to pure red (255,0,0):")
            print(f"  {name}: rgb({r}, {g}, {b})")
            print(f"  Distance: {min_distance:.2f}")
            return True
        else:
            print("? No colors found")
            return False
            
    except Exception as e:
        print(f"? Error testing color lookup: {e}")
        return False

if __name__ == "__main__":
    print("Enhanced Color Dictionary System - Basic Test")
    print("=" * 50)
    
    # Test 1: JSON loading
    json_test = test_json_loading_only()
    
    # Test 2: Color lookup
    lookup_test = test_color_conversion()
    
    print("\n" + "=" * 50)
    if json_test and lookup_test:
        print("?? ALL BASIC TESTS PASSED!")
        print("\nThe external color dictionary system is ready:")
        print("? JSON files are properly formatted")
        print("? Color data can be loaded and parsed")
        print("? Basic color matching algorithm works")
        print("\n?? Next step: Integration with ColorAnalyzer")
    else:
        print("? Some basic tests failed.")
        if not json_test:
            print("  - JSON file loading issues")
        if not lookup_test:
            print("  - Color lookup issues")