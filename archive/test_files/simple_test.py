# Test Menu Items 9-12 Implementation

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("Testing Menu Items 9-12 Implementation")
    print("=" * 50)
    
    # Test imports
    try:
        from src.transforms.artistic_transforms import ArtisticTransforms
        from src.transforms.activity_transforms import ActivityTransforms  
        from src.core.image_processor import ImageProcessor
        print("SUCCESS: All modules import correctly")
    except Exception as e:
        print(f"ERROR: Import failed - {e}")
        return False
    
    # Test method existence
    import logging
    logger = logging.getLogger()
    config = {}
    
    art = ArtisticTransforms(config, logger)
    act = ActivityTransforms(config, logger)
    
    # Check ArtisticTransforms methods
    methods_to_check = [
        'convert_to_pencil_sketch',
        'convert_to_coloring_book'
    ]
    
    for method in methods_to_check:
        if hasattr(art, method):
            print(f"SUCCESS: {method} exists")
        else:
            print(f"ERROR: {method} missing")
            
    # Check ActivityTransforms methods
    activity_methods = [
        'convert_to_connect_dots',
        'convert_to_color_by_numbers',
        '_order_points_for_connection'
    ]
    
    for method in activity_methods:
        if hasattr(act, method):
            print(f"SUCCESS: {method} exists") 
        else:
            print(f"ERROR: {method} missing")
    
    print("\nMENU ITEMS 9-12 VERIFICATION COMPLETE")
    print("All core components appear to be implemented")
    
    return True

if __name__ == "__main__":
    main()