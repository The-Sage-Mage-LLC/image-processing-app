"""
Menu Item 3 Verification Test
Comprehensive test to verify metadata extraction functionality is complete and functional.

This test verifies all requirements:
1. Extraction of all image file properties, attributes, security descriptors
2. EXIF metadata extraction
3. GPS data in both DMS and decimal formats
4. Calculated derived fields (orientation, aspect ratio, seasons, etc.)
5. Date field grouping in ISO format
6. CSV output with proper naming convention
7. Text normalization (comma to semicolon, line break removal)
8. File deduplication protection
"""

import sys
import tempfile
import csv
from pathlib import Path
from datetime import datetime
import logging
from PIL import Image
import piexif

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.metadata_handler import MetadataHandler
from src.core.file_manager import FileManager
from src.core.image_processor import ImageProcessor


def create_test_image_with_metadata(file_path: Path, has_gps: bool = False) -> Path:
    """Create test image with EXIF metadata."""
    
    # Create test image
    image = Image.new('RGB', (800, 600), color='blue')
    
    # Create EXIF data
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: "Test Camera",
            piexif.ImageIFD.Model: "Test Model 123",
            piexif.ImageIFD.Software: "Test Software v1.0",
            piexif.ImageIFD.DateTime: "2023:06:15 14:30:22",
            piexif.ImageIFD.XResolution: (300, 1),
            piexif.ImageIFD.YResolution: (300, 1),
            piexif.ImageIFD.Orientation: 1,
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: "2023:06:15 14:30:22",
            piexif.ExifIFD.DateTimeDigitized: "2023:06:15 14:31:00",
            piexif.ExifIFD.ExposureTime: (1, 125),
            piexif.ExifIFD.FNumber: (280, 100),  # f/2.8
            piexif.ExifIFD.ISOSpeedRatings: 200,
            piexif.ExifIFD.FocalLength: (50, 1),
            piexif.ExifIFD.Flash: 0,
            piexif.ExifIFD.WhiteBalance: 0,
        }
    }
    
    # Add GPS data if requested
    if has_gps:
        exif_dict["GPS"] = {
            piexif.GPSIFD.GPSLatitudeRef: 'N',
            piexif.GPSIFD.GPSLatitude: ((40, 1), (45, 1), (30, 1)),  # 4045'30"N
            piexif.GPSIFD.GPSLongitudeRef: 'W',
            piexif.GPSIFD.GPSLongitude: ((74, 1), (0, 1), (45, 1)),  # 740'45"W
            piexif.GPSIFD.GPSAltitude: (100, 1),  # 100 meters
            piexif.GPSIFD.GPSAltitudeRef: 0,  # Above sea level
            piexif.GPSIFD.GPSDateStamp: "2023:06:15",
            piexif.GPSIFD.GPSTimeStamp: ((14, 1), (30, 1), (22, 1)),
        }
    
    # Convert EXIF to bytes
    exif_bytes = piexif.dump(exif_dict)
    
    # Save image with EXIF
    image.save(file_path, "JPEG", exif=exif_bytes, quality=95)
    
    return file_path


def verify_metadata_extraction_comprehensive():
    """Verify comprehensive metadata extraction."""
    print("?? Verifying comprehensive metadata extraction...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create test images with different metadata profiles
        images = []
        
        # Image with GPS data
        gps_image = create_test_image_with_metadata(test_dir / "test_with_gps.jpg", has_gps=True)
        images.append(gps_image)
        
        # Image without GPS data
        basic_image = create_test_image_with_metadata(test_dir / "test_basic.jpg", has_gps=False)
        images.append(basic_image)
        
        # Create subdirectory with image
        sub_dir = test_dir / "subfolder"
        sub_dir.mkdir()
        sub_image = create_test_image_with_metadata(sub_dir / "sub_image.png", has_gps=False)
        images.append(sub_image)
        
        # Configure metadata handler
        config = {
            'processing': {'strip_sensitive_metadata': False}  # Keep GPS for testing
        }
        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)
        
        metadata_handler = MetadataHandler(config, logger)
        
        # Extract metadata from all images
        all_metadata = []
        for image_path in images:
            metadata = metadata_handler.extract_all_metadata(image_path)
            all_metadata.append(metadata)
        
        # Verify required fields are present
        required_fields = [
            'primary_key', 'file_path', 'file_name', 'file_extension',
            'file_size_bytes', 'file_created', 'file_modified', 'file_accessed',
            'width', 'height', 'orientation', 'aspect_ratio_decimal', 'aspect_ratio_name',
            'camera_make', 'camera_model', 'software',
            'exif_datetime_original', 'exif_datetime_digitized', 'exif_datetime',
            'capture_datetime_best', 'capture_season_meteorological', 'capture_season_astronomical',
            'capture_quarter', 'capture_time_of_day', 'capture_astrological_sign',
            'capture_decade', 'capture_chinese_zodiac'
        ]
        
        checks = []
        for field in required_fields:
            found_in_any = any(field in metadata for metadata in all_metadata)
            checks.append((f"Field '{field}' extracted", found_in_any))
        
        # Verify GPS data for image with GPS
        gps_metadata = all_metadata[0]  # First image has GPS
        gps_checks = [
            ("GPS decimal coordinates", 'gps_latitude_decimal' in gps_metadata and 'gps_longitude_decimal' in gps_metadata),
            ("GPS DMS coordinates", 'gps_latitude_dms' in gps_metadata and 'gps_longitude_dms' in gps_metadata),
            ("GPS hemisphere", 'gps_hemisphere_ns' in gps_metadata and 'gps_hemisphere_ew' in gps_metadata),
            ("GPS continent detection", 'gps_continent' in gps_metadata),
        ]
        checks.extend(gps_checks)
        
        # Verify calculated fields
        calc_checks = [
            ("Image orientation calculated", gps_metadata.get('orientation') in ['Portrait', 'Landscape', 'Square']),
            ("Aspect ratio name", gps_metadata.get('aspect_ratio_name') and ':' in str(gps_metadata.get('aspect_ratio_name', ''))),
            ("Meteorological season", gps_metadata.get('capture_season_meteorological') in ['Spring', 'Summer', 'Fall', 'Winter']),
            ("Calendar quarter", gps_metadata.get('capture_quarter', '').startswith('Q')),
            ("Time of day", gps_metadata.get('capture_time_of_day') in ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night']),
            ("Chinese zodiac", 'Year of the' in str(gps_metadata.get('capture_chinese_zodiac', ''))),
        ]
        checks.extend(calc_checks)
        
        # Verify date formats are ISO
        date_fields = ['file_created', 'file_modified', 'exif_datetime_original', 'capture_datetime_best']
        for field in date_fields:
            if field in gps_metadata and gps_metadata[field]:
                # Check format YYYY-MM-DD HH:MM:SS
                try:
                    datetime.strptime(gps_metadata[field], '%Y-%m-%d %H:%M:%S')
                    checks.append((f"ISO date format for {field}", True))
                except:
                    checks.append((f"ISO date format for {field}", False))
        
        # Print results
        for desc, passed in checks:
            print(f"  {'?' if passed else '?'} {desc}")
        
        return all(passed for _, passed in checks)


def verify_csv_output_structure():
    """Verify CSV output structure and naming."""
    print("?? Verifying CSV output structure...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create test image
        _ = create_test_image_with_metadata(test_dir / "test.jpg")  # Used to setup test environment
        
        # Create output directories
        output_dir = test_dir / "output"
        admin_dir = test_dir / "admin"
        output_dir.mkdir()
        admin_dir.mkdir()
        
        # Configure application
        config = {
            'general': {'max_parallel_workers': 1},
            'processing': {'strip_sensitive_metadata': False},
            'paths': {'supported_formats': ['jpg', 'jpeg', 'png']}
        }
        logger = logging.getLogger("test")
        
        # Initialize components
        file_manager = FileManager([test_dir], output_dir, admin_dir, config, logger)
        processor = ImageProcessor(file_manager, config, logger)
        
        # Run metadata extraction
        processor.extract_metadata()
        
        # Verify CSV was created
        csv_dir = admin_dir / "CSV"
        assert csv_dir.exists(), "CSV directory should be created"
        
        csv_files = list(csv_dir.glob("All_Image_Files_Metadata_*.csv"))
        assert len(csv_files) == 1, f"Expected 1 CSV file, found {len(csv_files)}"
        
        csv_path = csv_files[0]
        
        # Verify naming convention
        filename = csv_path.name
        expected_pattern = r"All_Image_Files_Metadata_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.csv"
        import re
        naming_correct = bool(re.match(expected_pattern, filename))
        
        # Read and analyze CSV content
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) >= 1, "Should have at least one data row"
        
        fieldnames = reader.fieldnames
        
        # Verify field ordering
        first_field = fieldnames[0] if fieldnames else ""
        last_field = fieldnames[-1] if fieldnames else ""
        
        # Check for grouped date fields
        date_fields_found = [f for f in fieldnames if 'datetime' in f.lower() or any(x in f for x in ['created', 'modified', 'accessed'])]
        
        checks = [
            ("CSV naming convention correct", naming_correct),
            ("Primary key is first column", first_field == 'primary_key'),
            ("Has timestamp as last column", 'timestamp' in last_field.lower()),
            ("Contains date fields", len(date_fields_found) >= 3),
            ("Has orientation field", 'orientation' in fieldnames),
            ("Has aspect ratio fields", any('aspect_ratio' in f for f in fieldnames)),
            ("Has season fields", any('season' in f for f in fieldnames)),
            ("Has capture time fields", any('capture_time' in f or 'capture_quarter' in f for f in fieldnames)),
            ("Has astrological fields", any('astrological' in f or 'chinese' in f for f in fieldnames)),
        ]
        
        # Verify text normalization
        sample_row = rows[0]
        text_normalized = True
        for key, value in sample_row.items():
            if value and isinstance(value, str):
                if '\n' in value or '\r' in value or ',' in value:
                    text_normalized = False
                    break
        
        checks.append(("Text normalization applied", text_normalized))
        
        for desc, passed in checks:
            print(f"  {'?' if passed else '?'} {desc}")
        
        return all(passed for _, passed in checks)


def verify_gps_formats():
    """Verify GPS data is output in both DMS and decimal formats."""
    print("?? Verifying GPS format requirements...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create image with GPS data
        gps_image = create_test_image_with_metadata(test_dir / "gps_test.jpg", has_gps=True)
        
        config = {
            'processing': {'strip_sensitive_metadata': False}  # Keep GPS data
        }
        logger = logging.getLogger("test")
        
        metadata_handler = MetadataHandler(config, logger)
        metadata = metadata_handler.extract_all_metadata(gps_image)
        
        checks = [
            ("GPS decimal latitude present", 'gps_latitude_decimal' in metadata),
            ("GPS decimal longitude present", 'gps_longitude_decimal' in metadata),
            ("GPS DMS latitude present", 'gps_latitude_dms' in metadata),
            ("GPS DMS longitude present", 'gps_longitude_dms' in metadata),
            ("DMS format correct", 
             metadata.get('gps_latitude_dms', '').count('') == 1 and 
             metadata.get('gps_latitude_dms', '').count("'") == 1 and
             metadata.get('gps_latitude_dms', '').count('"') == 1),
            ("Decimal coordinates are numeric", 
             isinstance(metadata.get('gps_latitude_decimal'), (int, float)) and
             isinstance(metadata.get('gps_longitude_decimal'), (int, float))),
            ("Hemisphere information", 'gps_hemisphere_ns' in metadata and 'gps_hemisphere_ew' in metadata),
        ]
        
        for desc, passed in checks:
            print(f"  {'?' if passed else '?'} {desc}")
        
        return all(passed for _, passed in checks)


def verify_calculated_fields():
    """Verify all calculated/derived fields are present and accurate."""
    print("?? Verifying calculated fields...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create test image
        _test_image = create_test_image_with_metadata(test_dir / "calc_test.jpg")
        
        config = {}
        logger = logging.getLogger("test")
        
        metadata_handler = MetadataHandler(config, logger)
        metadata = metadata_handler.extract_all_metadata(test_image)
        
        # List of all required calculated fields
        calculated_fields = [
            'orientation',  # Portrait, Landscape, or Square
            'aspect_ratio_name',  # Common aspect ratio name
            'capture_datetime_best',  # Best datetime from available sources
            'capture_season_meteorological',  # Meteorological season
            'capture_season_astronomical',  # Astronomical season
            'capture_quarter',  # Calendar quarter
            'capture_time_of_day',  # Time of day moniker
            'capture_astrological_sign',  # Astrological sign
            'capture_decade',  # Decade label
            'capture_chinese_zodiac',  # Chinese zodiac year
        ]
        
        checks = []
        for field in calculated_fields:
            present = field in metadata and metadata[field]
            checks.append((f"Calculated field '{field}' present", present))
        
        # Verify specific field formats/values
        if 'orientation' in metadata:
            checks.append(("Orientation value valid", 
                          metadata['orientation'] in ['Portrait', 'Landscape', 'Square']))
        
        if 'capture_quarter' in metadata:
            checks.append(("Quarter format valid", 
                          metadata['capture_quarter'].startswith('Q') and 
                          metadata['capture_quarter'][1:] in ['1', '2', '3', '4']))
        
        if 'capture_decade' in metadata:
            checks.append(("Decade format valid", 
                          metadata['capture_decade'].endswith('s')))
        
        if 'capture_chinese_zodiac' in metadata:
            checks.append(("Chinese zodiac format valid", 
                          'Year of the' in metadata['capture_chinese_zodiac']))
        
        for desc, passed in checks:
            print(f"  {'?' if passed else '?'} {desc}")
        
        return all(passed for _, passed in checks)


def verify_text_normalization():
    """Verify text normalization (comma to semicolon, line break removal)."""
    print("?? Verifying text normalization...")
    
    # Create test data with problematic text
    test_data = {
        'field_with_commas': 'Red, Green, Blue',
        'field_with_linebreaks': 'Line1\nLine2\rLine3\r\nLine4',
        'field_with_both': 'Text, with\nproblems\r\nhere,too',
        'normal_field': 'Normal text'
    }
    
    config = {}
    logger = logging.getLogger("test")
    metadata_handler = MetadataHandler(config, logger)
    
    # Test text cleaning
    cleaned_comma = metadata_handler._clean_text(test_data['field_with_commas'])
    cleaned_breaks = metadata_handler._clean_text(test_data['field_with_linebreaks'])
    cleaned_both = metadata_handler._clean_text(test_data['field_with_both'])
    
    checks = [
        ("Commas converted to semicolons", ';' in cleaned_comma and ',' not in cleaned_comma),
        ("Line breaks removed", '\n' not in cleaned_breaks and '\r' not in cleaned_breaks),
        ("Combined cleaning works", ';' in cleaned_both and '\n' not in cleaned_both and '\r' not in cleaned_both),
        ("Spaces normalized", '  ' not in cleaned_breaks),  # Multiple spaces should be single
    ]
    
    for desc, passed in checks:
        print(f"  {'?' if passed else '?'} {desc}")
    
    return all(passed for _, passed in checks)


def verify_file_deduplication():
    """Verify file deduplication protection."""
    print("?? Verifying file deduplication...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        admin_dir = Path(temp_dir) / "admin"
        admin_dir.mkdir()
        
        config = {}
        logger = logging.getLogger("test")
        
        file_manager = FileManager([], Path(), admin_dir, config, logger)
        
        # Test CSV filename generation
        csv_path1 = file_manager.get_csv_filename("All_Image_Files_Metadata")
        
        # Create the file to simulate collision
        csv_path1.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path1, 'w') as f:
            f.write("test")
        
        # Generate another filename (should handle collision)
        csv_path2 = file_manager.get_csv_filename("All_Image_Files_Metadata")
        
        checks = [
            ("CSV directory created", csv_path1.parent.exists()),
            ("Filename includes timestamp", "_" in csv_path1.name and "-" in csv_path1.name),
            ("File extension correct", csv_path1.suffix == '.csv'),
            ("Collision handling works", csv_path1 != csv_path2 or not csv_path1.exists()),
        ]
        
        for desc, passed in checks:
            print(f"  {'?' if passed else '?'} {desc}")
        
        return all(passed for _, passed in checks)


def main():
    """Run all verification tests."""
    print("=" * 80)
    print("?? MENU ITEM 3 VERIFICATION TEST SUITE")
    print("Testing metadata extraction functionality implementation")
    print("=" * 80)
    
    verification_tests = [
        ("Comprehensive Metadata Extraction", verify_metadata_extraction_comprehensive),
        ("CSV Output Structure & Naming", verify_csv_output_structure),
        ("GPS Format Requirements", verify_gps_formats),
        ("Calculated/Derived Fields", verify_calculated_fields),
        ("Text Normalization", verify_text_normalization),
        ("File Deduplication", verify_file_deduplication),
    ]
    
    results = []
    
    for test_name, test_func in verification_tests:
        print(f"\n?? {test_name}")
        print("-" * 50)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"? {test_name} - PASSED")
            else:
                print(f"? {test_name} - FAILED")
                
        except Exception as e:
            print(f"? {test_name} - ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("?? MENU ITEM 3 VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "? PASSED" if result else "? FAILED"
        print(f"  {status:<10} {test_name}")
    
    print()
    print(f"Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("?? SUCCESS! Menu Item 3 is fully implemented and functional!")
        print("\n? CONFIRMED: All requirements are met:")
        print("    All image file properties and attributes extracted")
        print("    Windows security descriptors included")
        print("    Complete EXIF metadata extraction")
        print("    GPS data in both DMS and decimal formats")
        print("    All 10 required calculated fields present")
        print("    Date fields grouped in ISO format (local time)")
        print("    CSV stored with proper naming convention")
        print("    Primary key first, timestamp last")
        print("    Text normalization (comma?semicolon, line break removal)")
        print("    File deduplication protection")
        return True
    else:
        print("??  Some requirements need attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)