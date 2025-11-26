#!/usr/bin/env python3
"""
FINAL FIX VERIFICATION TEST
Tests all the fixes we just implemented:
1. CaptionGenerator.generate_caption exists
2. BasicTransforms.analyze_colors exists  
3. EnhancedProcessingMonitor QA checks typo fixed
4. Hash duplicate detection context-aware for copy operations
"""

import sys
from pathlib import Path

sys.path.insert(0, 'src')

def test_all_fixes():
    """Test all fixes."""
    print("="*80)
    print("FINAL FIX VERIFICATION TEST")
    print("="*80)
    
    import logging
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger()
    config = {}
    
    # Test 1: CaptionGenerator.generate_caption
    print("\n1. Testing CaptionGenerator.generate_caption...")
    from models.caption_generator import CaptionGenerator
    cg = CaptionGenerator(config, logger)
    assert hasattr(cg, 'generate_caption'), "? generate_caption missing"
    assert hasattr(cg, 'generate_captions'), "? generate_captions missing"
    print("   ? CaptionGenerator has both generate_caption and generate_captions")
    
    # Test 2: BasicTransforms.analyze_colors
    print("\n2. Testing BasicTransforms.analyze_colors...")
    from transforms.basic_transforms import BasicTransforms
    bt = BasicTransforms(config, logger)
    assert hasattr(bt, 'analyze_colors'), "? analyze_colors missing"
    print("   ? BasicTransforms has analyze_colors method")
    
    # Test 3: EnhancedProcessingMonitor QA checks
    print("\n3. Testing EnhancedProcessingMonitor QA checks initialization...")
    from utils.monitoring import EnhancedProcessingMonitor
    mon = EnhancedProcessingMonitor(logger, config)
    
    # Start an operation to test initialization
    mon.start_operation("Test Operation", 10)
    assert hasattr(mon, 'qa_checks'), "? qa_checks attribute missing"
    assert all(v == 0 for v in mon.qa_checks.values()), "? QA checks not initialized to 0"
    print(f"   ? QA checks properly initialized: {len(mon.qa_checks)} checks at 0")
    
    # Test 4: Context-aware hash duplicate detection
    print("\n4. Testing context-aware hash duplicate detection...")
    import tempfile
    import shutil
    from PIL import Image
    import numpy as np
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test image
        test_img_path = temp_path / "test.jpg"
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(test_img_path)
        
        # Copy image
        copy_path = temp_path / "copy.jpg"
        shutil.copy2(test_img_path, copy_path)
        
        # Test with "Copying" operation (should NOT flag as issue)
        mon_copy = EnhancedProcessingMonitor(logger, config)
        mon_copy.start_operation("Copying Color Images", 1)
        
        original_hash = mon_copy._calculate_file_hash(test_img_path)
        mon_copy.record_processing_result(
            test_img_path,
            test_img_path.stat().st_size,
            copy_path,
            0.1,
            True,
            original_hash
        )
        
        # Should NOT have flagged as duplicate for copy operation
        assert mon_copy.qa_checks['hash_duplicates'] == 0, "? Copy operation incorrectly flagged as duplicate"
        print("   ? Copy operation: Identical hash NOT flagged as issue (correct)")
        
        # Test with transformation operation (SHOULD flag as issue)
        mon_transform = EnhancedProcessingMonitor(logger, config)
        mon_transform.start_operation("Converting to Grayscale", 1)
        
        mon_transform.record_processing_result(
            test_img_path,
            test_img_path.stat().st_size,
            copy_path,
            0.1,
            True,
            original_hash
        )
        
        # SHOULD have flagged as duplicate for non-copy operation
        assert mon_transform.qa_checks['hash_duplicates'] == 1, "? Transform operation did not flag duplicate"
        print("   ? Transform operation: Identical hash correctly flagged as issue")
    
    print("\n" + "="*80)
    print("??? ALL FIXES VERIFIED SUCCESSFULLY!")
    print("="*80)
    print("\nFix Summary:")
    print("  1. ? CaptionGenerator.generate_caption - FIXED")
    print("  2. ? BasicTransforms.analyze_colors - FIXED")
    print("  3. ? EnhancedProcessingMonitor typo - FIXED")
    print("  4. ? Context-aware hash detection - FIXED")
    print("\n? All components are now fully functional!")
    
    return True

if __name__ == "__main__":
    try:
        success = test_all_fixes()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n? TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
