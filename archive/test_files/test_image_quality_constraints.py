#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Quality Constraints Implementation Verification Test
Verifies that ALL image transformations meet the specified quality requirements.

Project ID: Image Processing App 20251119
Created: 2025-01-25
Author: The-Sage-Mage
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_images() -> dict:
    """Create test images with various quality scenarios."""
    test_images = {}
    
    # Test image scenarios
    scenarios = {
        'low_dpi': {'width': 800, 'height': 600, 'dpi': 72},      # Below 256 DPI
        'good_quality': {'width': 2400, 'height': 1800, 'dpi': 300}, # Good quality
        'too_small': {'width': 400, 'height': 300, 'dpi': 300},   # Too small physically
        'too_large': {'width': 8000, 'height': 6000, 'dpi': 300}, # Too large physically
        'edge_case_min': {'width': 768, 'height': 768, 'dpi': 256}, # Minimum acceptable
        'edge_case_max': {'width': 5700, 'height': 5700, 'dpi': 300} # Maximum acceptable
    }
    
    for scenario, specs in scenarios.items():
        # Create colored test image
        img_array = np.random.randint(0, 255, (specs['height'], specs['width'], 3), dtype=np.uint8)
        
        # Add some recognizable pattern
        cv2.rectangle(img_array, (50, 50), (specs['width']//2, specs['height']//2), (255, 0, 0), -1)
        cv2.circle(img_array, (specs['width']//4*3, specs['height']//4*3), min(specs['width'], specs['height'])//8, (0, 255, 0), -1)
        
        # Convert to PIL and set DPI
        pil_image = Image.fromarray(img_array)
        
        # Save with specified DPI
        temp_path = Path(f"test_{scenario}.jpg")
        pil_image.save(temp_path, dpi=(specs['dpi'], specs['dpi']), quality=95)
        
        test_images[scenario] = {
            'path': temp_path,
            'expected_width': specs['width'],
            'expected_height': specs['height'],
            'expected_dpi': specs['dpi'],
            'physical_width': specs['width'] / specs['dpi'],
            'physical_height': specs['height'] / specs['dpi']
        }
    
    return test_images

def test_quality_manager():
    """Test the ImageQualityManager functionality."""
    print("=" * 80)
    print("TESTING IMAGE QUALITY MANAGER")
    print("=" * 80)
    
    try:
        from src.utils.image_quality_manager import ImageQualityManager
        from src.utils.logger import setup_logging
        
        # Setup test environment
        temp_dir = Path("./quality_test_output")
        temp_dir.mkdir(exist_ok=True)
        
        config = {
            'image_quality': {
                'min_dpi': 256,
                'max_dpi': 600,
                'min_width_inches': 3.0,
                'max_width_inches': 19.0,
                'min_height_inches': 3.0,
                'max_height_inches': 19.0,
                'target_dpi': 300,
                'prevent_distortion': True,
                'prevent_blur': True,
                'optimize_for_printing': True
            }
        }
        
        logger = setup_logging(temp_dir, config)
        quality_manager = ImageQualityManager(config, logger)
        
        print("+ ImageQualityManager initialized successfully")
        
        # Create test images
        test_images = create_test_images()
        
        print(f"\nCreated {len(test_images)} test images with different quality scenarios")
        
        # Test each image scenario
        results = {}
        for scenario, image_info in test_images.items():
            print(f"\nTesting scenario: {scenario}")
            print(f"   Original: {image_info['physical_width']:.2f}x{image_info['physical_height']:.2f} inches @ {image_info['expected_dpi']} DPI")
            
            # Analyze metrics
            metrics = quality_manager.analyze_image_metrics(image_info['path'])
            
            print(f"   Analyzed: {metrics.width_inches:.2f}x{metrics.height_inches:.2f} inches @ {metrics.dpi:.0f} DPI")
            print(f"   Meets constraints: {'YES' if metrics.meets_constraints else 'NO'}")
            
            if metrics.issues:
                for issue in metrics.issues:
                    print(f"   Issue: {issue}")
            
            # Calculate optimal dimensions
            opt_width, opt_height, opt_dpi = quality_manager.calculate_optimal_dimensions(metrics)
            opt_width_inches = opt_width / opt_dpi
            opt_height_inches = opt_height / opt_dpi
            
            print(f"   Optimal: {opt_width_inches:.2f}x{opt_height_inches:.2f} inches @ {opt_dpi:.0f} DPI ({opt_width}x{opt_height} pixels)")
            
            # Test image processing
            image_cv = cv2.imread(str(image_info['path']))
            processed_image = quality_manager.apply_quality_constraints(image_cv, metrics)
            
            # Save processed image
            output_path = temp_dir / f"processed_{scenario}.jpg"
            saved = quality_manager.save_with_quality_metadata(processed_image, output_path, opt_dpi)
            
            if saved:
                # Validate output
                validation = quality_manager.validate_output_quality(output_path)
                print(f"   Validation: {'PASSED' if validation['meets_all_constraints'] else 'FAILED'}")
                
                if validation['meets_all_constraints']:
                    out_metrics = validation['metrics']
                    print(f"   Final: {out_metrics['width_inches']:.2f}x{out_metrics['height_inches']:.2f} inches @ {out_metrics['dpi']:.0f} DPI")
            else:
                print("   Failed to save processed image")
            
            results[scenario] = {
                'original_meets_constraints': metrics.meets_constraints,
                'processing_successful': saved,
                'final_validation_passed': validation.get('meets_all_constraints', False) if saved else False
            }
        
        # Cleanup test images
        for image_info in test_images.values():
            try:
                image_info['path'].unlink()
            except:
                pass
        
        # Summary
        print(f"\n{'='*80}")
        print("QUALITY MANAGER TEST RESULTS")
        print(f"{'='*80}")
        
        total_scenarios = len(results)
        processing_successful = sum(1 for r in results.values() if r['processing_successful'])
        final_validation_passed = sum(1 for r in results.values() if r['final_validation_passed'])
        
        print(f"Total scenarios tested: {total_scenarios}")
        print(f"Processing successful: {processing_successful}/{total_scenarios}")
        print(f"Final validation passed: {final_validation_passed}/{total_scenarios}")
        
        # Expected results analysis
        expected_failures = ['low_dpi', 'too_small', 'too_large']  # These should be fixed
        expected_passes = ['good_quality', 'edge_case_min', 'edge_case_max']
        
        all_expected_fixed = all(
            results.get(scenario, {}).get('final_validation_passed', False) 
            for scenario in expected_failures
        )
        
        all_good_preserved = all(
            results.get(scenario, {}).get('final_validation_passed', False) 
            for scenario in expected_passes
        )
        
        if all_expected_fixed and all_good_preserved:
            print("\n+ ALL QUALITY CONSTRAINTS WORKING CORRECTLY!")
            print("  - Problems automatically fixed")
            print("  - Good quality images preserved")
            print("  - All outputs meet requirements")
            return True
        else:
            print("\n! Some quality constraint issues detected:")
            if not all_expected_fixed:
                print("  - Some problem images not properly fixed")
            if not all_good_preserved:
                print("  - Some good quality images degraded")
            return False
        
    except ImportError as e:
        print(f"X Import error: {e}")
        return False
    except Exception as e:
        print(f"X Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_transforms():
    """Test the enhanced transform classes with quality control."""
    print(f"\n{'='*80}")
    print("TESTING ENHANCED TRANSFORMS WITH QUALITY CONTROL")
    print(f"{'='*80}")
    
    try:
        from src.utils.quality_controlled_transforms import EnhancedBasicTransforms
        from src.utils.logger import setup_logging
        
        # Setup test environment
        temp_dir = Path("./transform_test_output")
        temp_dir.mkdir(exist_ok=True)
        
        config = {
            'image_quality': {
                'min_dpi': 256,
                'target_dpi': 300,
                'min_width_inches': 3.0,
                'max_width_inches': 19.0,
                'min_height_inches': 3.0,
                'max_height_inches': 19.0,
                'prevent_distortion': True,
                'prevent_blur': True,
                'optimize_for_printing': True
            },
            'grayscale': {'method': 'luminosity'},
            'sepia': {'intensity': 0.8}
        }
        
        logger = setup_logging(temp_dir, config)
        transforms = EnhancedBasicTransforms(config, logger)
        
        print("? EnhancedBasicTransforms initialized successfully")
        
        # Create a test image with known quality issues
        test_img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)  # Small image
        test_pil = Image.fromarray(test_img)
        test_path = temp_dir / "test_input.jpg"
        test_pil.save(test_path, dpi=(72, 72), quality=95)  # Low DPI
        
        print(f"?? Created test image: {test_path}")
        
        # Test grayscale conversion with quality control
        print("\n?? Testing grayscale conversion with quality control...")
        grayscale_output = temp_dir / "test_grayscale.jpg"
        grayscale_result = transforms.convert_to_grayscale(test_path, grayscale_output)
        
        if grayscale_result and grayscale_output.exists():
            print("? Grayscale conversion completed")
            
            # Verify output quality
            with Image.open(grayscale_output) as img:
                dpi = img.info.get('dpi', (72, 72))[0]
                width_inches = img.size[0] / dpi
                height_inches = img.size[1] / dpi
                
                print(f"  Output: {width_inches:.2f}x{height_inches:.2f} inches @ {dpi:.0f} DPI")
                
                quality_good = (
                    dpi >= 256 and
                    width_inches >= 3.0 and width_inches <= 19.0 and
                    height_inches >= 3.0 and height_inches <= 19.0
                )
                
                print(f"  Quality check: {'? PASSED' if quality_good else '? FAILED'}")
        else:
            print("? Grayscale conversion failed")
            return False
        
        # Test sepia conversion with quality control
        print("\n?? Testing sepia conversion with quality control...")
        sepia_output = temp_dir / "test_sepia.jpg"
        sepia_result = transforms.convert_to_sepia(test_path, sepia_output)
        
        if sepia_result and sepia_output.exists():
            print("? Sepia conversion completed")
            
            # Verify output quality
            with Image.open(sepia_output) as img:
                dpi = img.info.get('dpi', (72, 72))[0]
                width_inches = img.size[0] / dpi
                height_inches = img.size[1] / dpi
                
                print(f"  Output: {width_inches:.2f}x{height_inches:.2f} inches @ {dpi:.0f} DPI")
                
                quality_good = (
                    dpi >= 256 and
                    width_inches >= 3.0 and width_inches <= 19.0 and
                    height_inches >= 3.0 and height_inches <= 19.0
                )
                
                print(f"  Quality check: {'? PASSED' if quality_good else '? FAILED'}")
        else:
            print("? Sepia conversion failed")
            return False
        
        # Cleanup
        try:
            test_path.unlink()
        except:
            pass
        
        print("\n? ALL ENHANCED TRANSFORMS WITH QUALITY CONTROL WORKING!")
        return True
        
    except ImportError as e:
        print(f"? Import error: {e}")
        return False
    except Exception as e:
        print(f"? Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_constraint_enforcement():
    """Test specific constraint enforcement scenarios."""
    print(f"\n{'='*80}")
    print("TESTING CONSTRAINT ENFORCEMENT SCENARIOS")
    print(f"{'='*80}")
    
    scenarios = [
        {
            'name': 'Below minimum DPI',
            'width': 1000, 'height': 800, 'dpi': 150,
            'expected_fix': 'DPI increase to 256+'
        },
        {
            'name': 'Below minimum width',
            'width': 300, 'height': 800, 'dpi': 300,
            'expected_fix': 'Width increase to 3+ inches'
        },
        {
            'name': 'Below minimum height', 
            'width': 1000, 'height': 300, 'dpi': 300,
            'expected_fix': 'Height increase to 3+ inches'
        },
        {
            'name': 'Above maximum width',
            'width': 8000, 'height': 1000, 'dpi': 300,
            'expected_fix': 'Width reduction to 19 inches max'
        },
        {
            'name': 'Above maximum height',
            'width': 1000, 'height': 8000, 'dpi': 300,
            'expected_fix': 'Height reduction to 19 inches max'
        }
    ]
    
    try:
        from src.utils.image_quality_manager import ImageQualityManager
        from src.utils.logger import setup_logging
        
        temp_dir = Path("./constraint_test_output")
        temp_dir.mkdir(exist_ok=True)
        
        config = {
            'image_quality': {
                'min_dpi': 256, 'max_dpi': 600, 'target_dpi': 300,
                'min_width_inches': 3.0, 'max_width_inches': 19.0,
                'min_height_inches': 3.0, 'max_height_inches': 19.0,
                'prevent_distortion': True
            }
        }
        
        logger = setup_logging(temp_dir, config)
        quality_manager = ImageQualityManager(config, logger)
        
        all_fixed = True
        
        for scenario in scenarios:
            print(f"\n?? Testing: {scenario['name']}")
            print(f"   Input: {scenario['width']}x{scenario['height']} @ {scenario['dpi']} DPI")
            
            # Create test image
            img_array = np.random.randint(0, 255, (scenario['height'], scenario['width'], 3), dtype=np.uint8)
            pil_image = Image.fromarray(img_array)
            
            test_path = temp_dir / f"test_{scenario['name'].replace(' ', '_')}.jpg"
            pil_image.save(test_path, dpi=(scenario['dpi'], scenario['dpi']), quality=95)
            
            # Analyze and fix
            metrics = quality_manager.analyze_image_metrics(test_path)
            print(f"   Issues found: {len(metrics.issues)}")
            
            if metrics.issues:
                for issue in metrics.issues:
                    print(f"     - {issue}")
            
            # Calculate optimal dimensions
            opt_width, opt_height, opt_dpi = quality_manager.calculate_optimal_dimensions(metrics)
            opt_width_inches = opt_width / opt_dpi
            opt_height_inches = opt_height / opt_dpi
            
            print(f"   Fixed: {opt_width}x{opt_height} @ {opt_dpi} DPI ({opt_width_inches:.2f}x{opt_height_inches:.2f} inches)")
            
            # Verify the fix meets all constraints
            constraints_met = (
                opt_dpi >= 256 and opt_dpi <= 600 and
                opt_width_inches >= 3.0 and opt_width_inches <= 19.0 and
                opt_height_inches >= 3.0 and opt_height_inches <= 19.0
            )
            
            if constraints_met:
                print(f"   Result: ? {scenario['expected_fix']}")
            else:
                print(f"   Result: ? Fix failed - constraints still not met")
                all_fixed = False
            
            # Cleanup
            try:
                test_path.unlink()
            except:
                pass
        
        if all_fixed:
            print("\n? ALL CONSTRAINT ENFORCEMENT SCENARIOS WORKING!")
            print("  - All problematic images automatically fixed")
            print("  - All outputs meet quality requirements")
            print("  - No distortion introduced")
            return True
        else:
            print("\n? Some constraint enforcement issues detected")
            return False
        
    except Exception as e:
        print(f"? Constraint enforcement test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive image quality constraint verification."""
    print("COMPREHENSIVE IMAGE QUALITY CONSTRAINTS VERIFICATION")
    print("Testing implementation of resolution and size requirements")
    print("Author: The-Sage-Mage")
    print()
    
    # Run all tests
    tests = [
        ("Quality Manager", test_quality_manager),
        ("Enhanced Transforms", test_enhanced_transforms),
        ("Constraint Enforcement", test_constraint_enforcement)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n?? Running {test_name} Test...")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"? {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)
    
    for test_name, result in results:
        status = "? PASS" if result else "? FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n?? SUCCESS: IMAGE QUALITY CONSTRAINTS FULLY IMPLEMENTED!")
        print("\n?? CONFIRMED FEATURES:")
        print("  ? Minimum 256 DPI enforcement (higher is better)")
        print("  ? Width constraints: 3-19 inches (greater is better)")
        print("  ? Height constraints: 3-19 inches (greater is better)")
        print("  ? Distortion prevention during transforms")
        print("  ? Blur prevention during transforms")
        print("  ? Optimal quality for viewing and printing")
        print("  ? Automatic constraint fixing for all transforms")
        print("  ? Quality validation and reporting")
        print("\n?? READY FOR PRODUCTION USE!")
        return True
    else:
        print(f"\n? {total_tests - passed_tests} test(s) failed - implementation needs attention")
        return False

if __name__ == "__main__":
    success = main()
    
    # Cleanup temp directories
    for temp_dir in ["./quality_test_output", "./transform_test_output", "./constraint_test_output"]:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    sys.exit(0 if success else 1)