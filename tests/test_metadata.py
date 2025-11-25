"""
Test script for metadata extraction
Project ID: Image Processing App 20251119
Created: 2025-01-19 07:03:16 UTC
Author: The-Sage-Mage
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.metadata_handler import MetadataHandler
from src.utils.logger import setup_logging
import json


def test_metadata_extraction():
    """Test metadata extraction on sample images."""
    
    # Setup
    config = {
        'processing': {
            'strip_sensitive_metadata': False  # Keep GPS for testing
        },
        'logging': {
            'log_to_console': True,
            'log_to_file': False
        }
    }
    
    logger = setup_logging(Path('./test_output'), config)
    handler = MetadataHandler(config, logger)
    
    # Test with sample image
    test_image = Path('./sample_images/test.jpg')  # Update with your test image path
    
    if test_image.exists():
        print(f"Testing metadata extraction on: {test_image}")
        
        # Extract metadata
        metadata = handler.extract_all_metadata(test_image)
        
        # Print results
        print("\n" + "="*60)
        print("EXTRACTED METADATA")
        print("="*60)
        
        for key, value in metadata.items():
            if value and value != '':
                print(f"{key:30s}: {str(value)[:50]}")
        
        # Test CSV generation
        csv_path = Path('./test_output/test_metadata.csv')
        handler.save_metadata_to_csv([metadata], csv_path)
        print(f"\nCSV saved to: {csv_path}")
        
    else:
        print(f"Test image not found: {test_image}")
        print("Please place a test image at the specified path")


def test_gps_conversion():
    """Test GPS coordinate conversion."""
    config = {}
    logger = setup_logging(Path('./test_output'), config)
    handler = MetadataHandler(config, logger)
    
    # Test decimal to DMS conversion
    test_coords = [
        (40.7128, -74.0060),  # New York
        (51.5074, -0.1278),   # London
        (35.6762, 139.6503),  # Tokyo
        (-33.8688, 151.2093), # Sydney
    ]
    
    print("\n" + "="*60)
    print("GPS COORDINATE CONVERSION TEST")
    print("="*60)
    
    for lat, lon in test_coords:
        lat_dms = handler._decimal_to_dms(lat, ['N', 'S'])
        lon_dms = handler._decimal_to_dms(lon, ['E', 'W'])
        
        print(f"Decimal: ({lat:.4f}, {lon:.4f})")
        print(f"DMS:     {lat_dms}, {lon_dms}")
        print()


def test_datetime_calculations():
    """Test datetime-based calculations."""
    from datetime import datetime
    
    config = {}
    logger = setup_logging(Path('./test_output'), config)
    handler = MetadataHandler(config, logger)
    
    # Test dates
    test_dates = [
        datetime(2025, 1, 15, 14, 30),  # Winter, Afternoon, Capricorn
        datetime(2025, 4, 22, 7, 15),   # Spring, Morning, Taurus
        datetime(2025, 7, 4, 20, 45),   # Summer, Night, Cancer
        datetime(2025, 10, 31, 18, 0),  # Fall, Evening, Scorpio
    ]
    
    print("\n" + "="*60)
    print("DATETIME CALCULATIONS TEST")
    print("="*60)
    
    for dt in test_dates:
        print(f"\nDate/Time: {dt.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Meteorological Season: {handler._get_meteorological_season(dt)}")
        print(f"  Astronomical Season: {handler._get_astronomical_season(dt)}")
        print(f"  Quarter: Q{(dt.month - 1) // 3 + 1}")
        print(f"  Time of Day: {handler._get_time_of_day(dt)}")
        print(f"  Astrological Sign: {handler._get_astrological_sign(dt)}")
        print(f"  Decade: {(dt.year // 10) * 10}s")
        print(f"  Chinese Zodiac: {handler._get_chinese_zodiac(dt.year)}")


if __name__ == "__main__":
    print("="*60)
    print("METADATA EXTRACTION TEST SUITE")
    print("="*60)
    
    # Run tests
    test_datetime_calculations()
    test_gps_conversion()
    
    # Uncomment when you have a test image
    # test_metadata_extraction()
    
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)