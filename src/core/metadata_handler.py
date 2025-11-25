"""
Metadata Extraction and Processing Module
Project ID: Image Processing App 20251119
Created: 2025-01-19 07:03:16 UTC
Author: The-Sage-Mage
"""

import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
import logging
from collections import OrderedDict

import exifread
import piexif
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import win32api
import win32con
import win32security
import os
import stat


class MetadataHandler:
    """Handles extraction and processing of image metadata."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.strip_sensitive = config.get('processing', {}).get('strip_sensitive_metadata', True)
        self._primary_key_counter = 1
        
        # Season definitions
        self.meteorological_seasons = {
            'Spring': [(3, 1), (5, 31)],
            'Summer': [(6, 1), (8, 31)],
            'Fall': [(9, 1), (11, 30)],
            'Winter': [(12, 1), (2, 28)]  # Simplified, not handling leap years
        }
        
        self.astronomical_seasons = {
            'Spring': [(3, 20), (6, 20)],
            'Summer': [(6, 21), (9, 22)],
            'Fall': [(9, 23), (12, 20)],
            'Winter': [(12, 21), (3, 19)]
        }
        
        # Time of day definitions
        self.time_of_day = {
            'Early Morning': (5, 7),
            'Morning': (7, 12),
            'Afternoon': (12, 17),
            'Evening': (17, 20),
            'Night': (20, 24),
            'Late Night': (0, 5)
        }
        
        # Astrological signs
        self.astrological_signs = [
            ('Capricorn', (12, 22), (1, 19)),
            ('Aquarius', (1, 20), (2, 18)),
            ('Pisces', (2, 19), (3, 20)),
            ('Aries', (3, 21), (4, 19)),
            ('Taurus', (4, 20), (5, 20)),
            ('Gemini', (5, 21), (6, 20)),
            ('Cancer', (6, 21), (7, 22)),
            ('Leo', (7, 23), (8, 22)),
            ('Virgo', (8, 23), (9, 22)),
            ('Libra', (9, 23), (10, 22)),
            ('Scorpio', (10, 23), (11, 21)),
            ('Sagittarius', (11, 22), (12, 21))
        ]
        
        # Chinese zodiac
        self.chinese_zodiac = [
            'Rat', 'Ox', 'Tiger', 'Rabbit', 'Dragon', 'Snake',
            'Horse', 'Goat', 'Monkey', 'Rooster', 'Dog', 'Pig'
        ]
    
    def extract_all_metadata(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing all extracted metadata
        """
        metadata = OrderedDict()
        
        # Initialize with primary key
        metadata['primary_key'] = self._primary_key_counter
        self._primary_key_counter += 1
        
        # Basic file information
        metadata['file_path'] = str(image_path)
        metadata['file_name'] = image_path.name
        metadata['file_extension'] = image_path.suffix.lower()
        
        try:
            # File system metadata
            self._extract_file_metadata(image_path, metadata)
            
            # Windows-specific metadata
            self._extract_windows_metadata(image_path, metadata)
            
            # EXIF metadata
            self._extract_exif_metadata(image_path, metadata)
            
            # Image properties
            self._extract_image_properties(image_path, metadata)
            
            # GPS data
            self._extract_gps_data(image_path, metadata)
            
            # Calculate derived fields
            self._calculate_derived_fields(metadata)
            
            # Add timestamp
            metadata['data_acquisition_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {image_path}: {e}")
            metadata['extraction_error'] = str(e)
        
        return metadata
    
    def _extract_file_metadata(self, path: Path, metadata: Dict[str, Any]):
        """Extract file system metadata."""
        try:
            stat_info = path.stat()
            
            metadata['file_size_bytes'] = stat_info.st_size
            metadata['file_size_mb'] = round(stat_info.st_size / (1024 * 1024), 2)
            
            # Dates in ISO format
            metadata['file_created'] = datetime.fromtimestamp(stat_info.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            metadata['file_modified'] = datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            metadata['file_accessed'] = datetime.fromtimestamp(stat_info.st_atime).strftime('%Y-%m-%d %H:%M:%S')
            
            # File permissions
            metadata['file_mode'] = oct(stat_info.st_mode)
            metadata['is_readonly'] = not os.access(path, os.W_OK)
            
        except Exception as e:
            self.logger.debug(f"Error extracting file metadata: {e}")
    
    def _extract_windows_metadata(self, path: Path, metadata: Dict[str, Any]):
        """Extract Windows-specific metadata and security descriptors."""
        try:
            # Get file attributes
            attributes = win32api.GetFileAttributes(str(path))
            
            attr_list = []
            if attributes & win32con.FILE_ATTRIBUTE_ARCHIVE:
                attr_list.append('Archive')
            if attributes & win32con.FILE_ATTRIBUTE_HIDDEN:
                attr_list.append('Hidden')
            if attributes & win32con.FILE_ATTRIBUTE_READONLY:
                attr_list.append('ReadOnly')
            if attributes & win32con.FILE_ATTRIBUTE_SYSTEM:
                attr_list.append('System')
            if attributes & win32con.FILE_ATTRIBUTE_COMPRESSED:
                attr_list.append('Compressed')
            if attributes & win32con.FILE_ATTRIBUTE_ENCRYPTED:
                attr_list.append('Encrypted')
            
            metadata['windows_attributes'] = '; '.join(attr_list) if attr_list else 'Normal'
            
            # Get owner information
            try:
                security_descriptor = win32security.GetFileSecurity(
                    str(path),
                    win32security.OWNER_SECURITY_INFORMATION
                )
                owner_sid = security_descriptor.GetSecurityDescriptorOwner()
                owner_name, domain, _ = win32security.LookupAccountSid(None, owner_sid)
                metadata['file_owner'] = f"{domain}\\{owner_name}" if domain else owner_name
            except:
                metadata['file_owner'] = 'Unknown'
                
        except Exception as e:
            self.logger.debug(f"Error extracting Windows metadata: {e}")
    
    def _extract_exif_metadata(self, path: Path, metadata: Dict[str, Any]):
        """Extract EXIF metadata from image."""
        try:
            # Method 1: Using PIL
            with Image.open(path) as img:
                exif_data = img.getexif()
                
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        # Skip sensitive data if configured
                        if self.strip_sensitive and tag in ['GPSInfo', 'MakerNote', 'UserComment']:
                            continue
                        
                        # Convert bytes to string if needed
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', errors='ignore')
                            except:
                                value = str(value)
                        
                        # Handle specific tags
                        if tag == 'DateTime':
                            metadata['exif_datetime'] = self._parse_exif_datetime(value)
                        elif tag == 'DateTimeOriginal':
                            metadata['exif_datetime_original'] = self._parse_exif_datetime(value)
                        elif tag == 'DateTimeDigitized':
                            metadata['exif_datetime_digitized'] = self._parse_exif_datetime(value)
                        elif tag == 'Make':
                            metadata['camera_make'] = self._clean_text(value)
                        elif tag == 'Model':
                            metadata['camera_model'] = self._clean_text(value)
                        elif tag == 'Software':
                            metadata['software'] = self._clean_text(value)
                        elif tag == 'Orientation':
                            metadata['exif_orientation'] = value
                        elif tag == 'XResolution':
                            metadata['x_resolution'] = value
                        elif tag == 'YResolution':
                            metadata['y_resolution'] = value
                        elif tag == 'ExposureTime':
                            metadata['exposure_time'] = str(value)
                        elif tag == 'FNumber':
                            metadata['f_number'] = float(value) if value else None
                        elif tag == 'ISO':
                            metadata['iso'] = value
                        elif tag == 'FocalLength':
                            metadata['focal_length'] = float(value) if value else None
                        elif tag == 'Flash':
                            metadata['flash'] = value
                        elif tag == 'WhiteBalance':
                            metadata['white_balance'] = value
                        else:
                            # Store other tags with normalized names
                            key = f"exif_{tag.lower().replace(' ', '_')}"
                            metadata[key] = value
            
            # Method 2: Using exifread for additional data
            with open(path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                
                for tag, value in tags.items():
                    # Skip thumbnails and already processed tags
                    if 'Thumbnail' in tag or 'JPEGThumbnail' in tag:
                        continue
                    
                    # Normalize tag name
                    tag_name = tag.replace(' ', '_').lower()
                    
                    # Convert value to string
                    value_str = str(value)
                    
                    # Store if not already present
                    if f"exif_{tag_name}" not in metadata:
                        metadata[f"exif_{tag_name}"] = value_str
                        
        except Exception as e:
            self.logger.debug(f"Error extracting EXIF metadata: {e}")
    
    def _extract_image_properties(self, path: Path, metadata: Dict[str, Any]):
        """Extract image properties."""
        try:
            with Image.open(path) as img:
                # Basic properties
                metadata['width'] = img.width
                metadata['height'] = img.height
                metadata['mode'] = img.mode
                metadata['format'] = img.format
                
                # Calculate orientation
                if img.width > img.height:
                    metadata['orientation'] = 'Landscape'
                elif img.width < img.height:
                    metadata['orientation'] = 'Portrait'
                else:
                    metadata['orientation'] = 'Square'
                
                # Calculate aspect ratio
                aspect_ratio = img.width / img.height if img.height > 0 else 0
                metadata['aspect_ratio_decimal'] = round(aspect_ratio, 3)
                metadata['aspect_ratio_name'] = self._get_aspect_ratio_name(aspect_ratio)
                
                # Color information
                if hasattr(img, 'getbands'):
                    metadata['color_bands'] = ''.join(img.getbands())
                
                # Get palette info if present
                if img.mode == 'P':
                    palette = img.getpalette()
                    if palette:
                        metadata['has_palette'] = True
                        metadata['palette_colors'] = len(palette) // 3
                
                # Check for transparency
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    metadata['has_transparency'] = True
                else:
                    metadata['has_transparency'] = False
                    
        except Exception as e:
            self.logger.debug(f"Error extracting image properties: {e}")
    
    def _extract_gps_data(self, path: Path, metadata: Dict[str, Any]):
        """Extract and process GPS data."""
        try:
            with Image.open(path) as img:
                exif_data = img.getexif()
                
                if exif_data:
                    # Get GPS IFD
                    gps_ifd = exif_data.get_ifd(0x8825)
                    
                    if gps_ifd and not self.strip_sensitive:
                        gps_data = {}
                        
                        for tag_id, value in gps_ifd.items():
                            tag = GPSTAGS.get(tag_id, tag_id)
                            gps_data[tag] = value
                        
                        # Extract coordinates
                        lat = self._convert_gps_coordinate(
                            gps_data.get('GPSLatitude'),
                            gps_data.get('GPSLatitudeRef')
                        )
                        lon = self._convert_gps_coordinate(
                            gps_data.get('GPSLongitude'),
                            gps_data.get('GPSLongitudeRef')
                        )
                        
                        if lat is not None and lon is not None:
                            # Decimal format
                            metadata['gps_latitude_decimal'] = lat
                            metadata['gps_longitude_decimal'] = lon
                            
                            # DMS format
                            metadata['gps_latitude_dms'] = self._decimal_to_dms(lat, ['N', 'S'])
                            metadata['gps_longitude_dms'] = self._decimal_to_dms(lon, ['E', 'W'])
                            
                            # Hemisphere
                            metadata['gps_hemisphere_ns'] = 'North' if lat >= 0 else 'South'
                            metadata['gps_hemisphere_ew'] = 'East' if lon >= 0 else 'West'
                            
                            # Get location details
                            location = self._get_location_from_coordinates(lat, lon)
                            metadata.update(location)
                        
                        # Altitude
                        if 'GPSAltitude' in gps_data:
                            altitude = float(gps_data['GPSAltitude'])
                            if 'GPSAltitudeRef' in gps_data and gps_data['GPSAltitudeRef'] == 1:
                                altitude = -altitude
                            metadata['gps_altitude_meters'] = altitude
                        
                        # GPS timestamp
                        if 'GPSDateStamp' in gps_data:
                            metadata['gps_datestamp'] = str(gps_data['GPSDateStamp'])
                        if 'GPSTimeStamp' in gps_data:
                            metadata['gps_timestamp'] = str(gps_data['GPSTimeStamp'])
                            
        except Exception as e:
            self.logger.debug(f"Error extracting GPS data: {e}")
    
    def _convert_gps_coordinate(self, coord_data: Any, ref: str) -> Optional[float]:
        """Convert GPS coordinate to decimal degrees."""
        if not coord_data or not ref:
            return None
        
        try:
            # coord_data is typically a tuple of (degrees, minutes, seconds)
            if len(coord_data) == 3:
                degrees = float(coord_data[0])
                minutes = float(coord_data[1])
                seconds = float(coord_data[2])
                
                decimal = degrees + (minutes / 60) + (seconds / 3600)
                
                if ref in ['S', 'W']:
                    decimal = -decimal
                
                return round(decimal, 6)
        except:
            return None
    
    def _decimal_to_dms(self, decimal: float, hemispheres: List[str]) -> str:
        """Convert decimal degrees to DMS format."""
        hemisphere = hemispheres[0] if decimal >= 0 else hemispheres[1]
        decimal = abs(decimal)
        
        degrees = int(decimal)
        minutes_decimal = (decimal - degrees) * 60
        minutes = int(minutes_decimal)
        seconds = (minutes_decimal - minutes) * 60
        
        return f"{degrees}°{minutes}'{seconds:.2f}\"{hemisphere}"
    
    def _get_location_from_coordinates(self, lat: float, lon: float) -> Dict[str, str]:
        """Get human-readable location from GPS coordinates."""
        location = {}
        
        # Determine continent (simplified)
        if -35 <= lat <= 37 and -17 <= lon <= 52:
            location['gps_continent'] = 'Africa'
        elif -55 <= lat <= 83 and -170 <= lon <= 190:
            if lat > 35 and lon > -10:
                location['gps_continent'] = 'Europe'
            elif lat > 8 and lon > 25:
                location['gps_continent'] = 'Asia'
            else:
                location['gps_continent'] = 'Asia'
        elif 15 <= lat <= 72 and -165 <= lon <= -50:
            location['gps_continent'] = 'North America'
        elif -55 <= lat <= 15 and -82 <= lon <= -34:
            location['gps_continent'] = 'South America'
        elif -47 <= lat <= -10 and 112 <= lon <= 179:
            location['gps_continent'] = 'Australia/Oceania'
        else:
            location['gps_continent'] = 'Unknown'
        
        # Note: Full reverse geocoding would require external API
        # This is a placeholder for the structure
        location['gps_country'] = 'Reverse geocoding required'
        location['gps_region'] = ''
        location['gps_state_province'] = ''
        location['gps_city_district'] = ''
        location['gps_local_area'] = ''
        location['gps_readable_location'] = f"{lat:.4f}, {lon:.4f}"
        
        return location
    
    def _calculate_derived_fields(self, metadata: Dict[str, Any]):
        """Calculate additional derived fields."""
        # Find best datetime
        datetime_fields = [
            'exif_datetime_original',
            'exif_datetime_digitized',
            'exif_datetime',
            'file_created',
            'file_modified'
        ]
        
        best_datetime = None
        for field in datetime_fields:
            if field in metadata and metadata[field]:
                try:
                    if isinstance(metadata[field], str):
                        # Parse the datetime string
                        dt = datetime.strptime(metadata[field], '%Y-%m-%d %H:%M:%S')
                        best_datetime = dt
                        break
                except:
                    continue
        
        if best_datetime:
            metadata['capture_datetime_best'] = best_datetime.strftime('%Y-%m-%d %H:%M:%S')
            
            # Meteorological season
            metadata['capture_season_meteorological'] = self._get_meteorological_season(best_datetime)
            
            # Astronomical season
            metadata['capture_season_astronomical'] = self._get_astronomical_season(best_datetime)
            
            # Calendar quarter
            metadata['capture_quarter'] = f"Q{(best_datetime.month - 1) // 3 + 1}"
            
            # Time of day
            metadata['capture_time_of_day'] = self._get_time_of_day(best_datetime)
            
            # Astrological sign
            metadata['capture_astrological_sign'] = self._get_astrological_sign(best_datetime)
            
            # Decade
            metadata['capture_decade'] = f"{(best_datetime.year // 10) * 10}s"
            
            # Chinese zodiac
            metadata['capture_chinese_zodiac'] = self._get_chinese_zodiac(best_datetime.year)
    
    def _get_meteorological_season(self, dt: datetime) -> str:
        """Get meteorological season for date."""
        month = dt.month
        day = dt.day
        
        for season, (start, end) in self.meteorological_seasons.items():
            start_month, start_day = start
            end_month, end_day = end
            
            if start_month <= month <= end_month:
                if month == start_month and day < start_day:
                    continue
                if month == end_month and day > end_day:
                    continue
                return season
        
        return 'Winter'  # Default
    
    def _get_astronomical_season(self, dt: datetime) -> str:
        """Get astronomical season for date."""
        month = dt.month
        day = dt.day
        
        for season, (start, end) in self.astronomical_seasons.items():
            start_month, start_day = start
            end_month, end_day = end
            
            if start_month <= month <= end_month:
                if month == start_month and day < start_day:
                    continue
                if month == end_month and day > end_day:
                    continue
                return season
        
        return 'Winter'  # Default
    
    def _get_time_of_day(self, dt: datetime) -> str:
        """Get time of day moniker."""
        hour = dt.hour
        
        for period, (start, end) in self.time_of_day.items():
            if start <= hour < end:
                return period
        
        return 'Night'
    
    def _get_astrological_sign(self, dt: datetime) -> str:
        """Get astrological sign for date."""
        month = dt.month
        day = dt.day
        
        for sign, start, end in self.astrological_signs:
            start_month, start_day = start
            end_month, end_day = end
            
            if month == start_month and day >= start_day:
                return sign
            elif month == end_month and day <= end_day:
                return sign
        
        return 'Unknown'
    
    def _get_chinese_zodiac(self, year: int) -> str:
        """Get Chinese zodiac animal for year."""
        # Chinese zodiac is based on a 12-year cycle
        # Year 1900 was the Year of the Rat
        base_year = 1900
        index = (year - base_year) % 12
        animal = self.chinese_zodiac[index]
        return f"Year of the {animal}"
    
    def _get_aspect_ratio_name(self, ratio: float) -> str:
        """Get common aspect ratio name."""
        common_ratios = {
            (1.0, 0.02): '1:1 (Square)',
            (1.33, 0.02): '4:3 (Standard)',
            (1.5, 0.02): '3:2 (Classic)',
            (1.6, 0.02): '16:10 (Wide)',
            (1.77, 0.02): '16:9 (Widescreen)',
            (1.85, 0.02): '1.85:1 (Cinema)',
            (2.35, 0.05): '2.35:1 (Cinemascope)',
            (2.39, 0.05): '2.39:1 (Panavision)'
        }
        
        for (target_ratio, tolerance), name in common_ratios.items():
            if abs(ratio - target_ratio) <= tolerance:
                return name
        
        # Return decimal ratio if no match
        return f"{ratio:.2f}:1 (Custom)"
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not text:
            return ''
        
        # Remove null bytes and control characters
        text = text.replace('\x00', '').strip()
        
        # Replace line breaks with spaces
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Replace multiple spaces with single space
        text = ' '.join(text.split())
        
        # Replace commas with semicolons for CSV
        text = text.replace(',', ';')
        
        return text
    
    def _parse_exif_datetime(self, datetime_str: str) -> Optional[str]:
        """Parse EXIF datetime string to ISO format."""
        if not datetime_str:
            return None
        
        try:
            # EXIF datetime format: 'YYYY:MM:DD HH:MM:SS'
            dt = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return datetime_str
    
    def save_metadata_to_csv(self, metadata_list: List[Dict[str, Any]], output_path: Path):
        """
        Save metadata to CSV file with enhanced text processing.
        
        Args:
            metadata_list: List of metadata dictionaries
            output_path: Path to CSV file
        """
        if not metadata_list:
            self.logger.warning("No metadata to save")
            return
        
        # Apply text processing to all metadata
        from ..utils.text_processor import text_processor
        
        processed_metadata = []
        for metadata in metadata_list:
            processed = text_processor.process_all_text_fields(metadata)
            processed_metadata.append(processed)
        
        # Get all unique keys across all metadata
        all_keys = set()
        for metadata in processed_metadata:
            all_keys.update(metadata.keys())
        
        # Group and order fields
        ordered_keys = self._order_csv_fields(all_keys)
        
        # Write CSV
        try:
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=ordered_keys)
                
                # Write header
                writer.writeheader()
                
                # Write data rows
                for metadata in processed_metadata:
                    # Ensure all fields are present and clean data for CSV
                    row = {}
                    for key in ordered_keys:
                        value = metadata.get(key, '')
                        if value:
                            # Convert commas to semicolons for CSV compatibility
                            value = str(value).replace(',', ';')
                            # Remove line breaks and CRLFs
                            value = value.replace('\n', ' ').replace('\r', ' ').replace('\r\n', ' ')
                            # Clean multiple spaces
                            value = ' '.join(value.split())
                        row[key] = value
                    writer.writerow(row)
            
            self.logger.info(f"Metadata saved to {output_path}")
            self.logger.info(f"Total rows: {len(processed_metadata)}, Total columns: {len(ordered_keys)}")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata to CSV: {e}")
    
    def _order_csv_fields(self, fields: set) -> List[str]:
        """Order CSV fields logically."""
        # Define field groups in order
        field_order = [
            # Primary key (first)
            ['primary_key'],
            
            # File information
            ['file_path', 'file_name', 'file_extension', 'file_size_bytes', 'file_size_mb'],
            
            # File dates (grouped)
            ['file_created', 'file_modified', 'file_accessed'],
            
            # EXIF dates (grouped)
            ['exif_datetime_original', 'exif_datetime_digitized', 'exif_datetime',
             'capture_datetime_best'],
            
            # Image properties
            ['width', 'height', 'orientation', 'aspect_ratio_decimal', 'aspect_ratio_name'],
            
            # Camera information
            ['camera_make', 'camera_model', 'software'],
            
            # Camera settings
            ['exposure_time', 'f_number', 'iso', 'focal_length', 'flash', 'white_balance'],
            
            # GPS data (DMS format)
            ['gps_latitude_dms', 'gps_longitude_dms'],
            
            # GPS data (decimal format)
            ['gps_latitude_decimal', 'gps_longitude_decimal'],
            
            # GPS location (readable)
            ['gps_hemisphere_ns', 'gps_hemisphere_ew', 'gps_continent', 'gps_country',
             'gps_region', 'gps_state_province', 'gps_city_district', 'gps_local_area',
             'gps_readable_location'],
            
            # Calculated seasonal data
            ['capture_season_meteorological', 'capture_season_astronomical', 'capture_quarter'],
            
            # Calculated time data
            ['capture_time_of_day', 'capture_astrological_sign', 'capture_decade',
             'capture_chinese_zodiac'],
            
            # Timestamp (last)
            ['data_acquisition_timestamp']
        ]
        
        # Flatten the order list
        ordered = []
        for group in field_order:
            for field in group:
                if field in fields:
                    ordered.append(field)
        
        # Add any remaining fields not in the order list
        remaining = sorted(fields - set(ordered))
        ordered.extend(remaining)
        
        return ordered