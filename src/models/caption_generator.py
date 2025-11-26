"""
Caption and Description Generation Module
Project ID: Image Processing App 20251119
Created: 2025-11-19 07:08:42 UTC
Author: The-Sage-Mage
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from PIL import Image
import json


class CaptionGenerator:
    """Generates captions, descriptions, keywords, and alt text for images."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize AI models if available
        self.use_ai = False
        self.clip_model = None
        self.yolo_model = None
        
        try:
            self._initialize_ai_models()
        except Exception as e:
            self.logger.warning(f"AI models not available, using fallback methods: {e}")
            self.use_ai = False
    
    def _initialize_ai_models(self):
        """Initialize AI models for caption generation."""
        try:
            # Try to import and initialize CLIP for captions
            # Note: Full implementation would require transformers library
            self.logger.info("AI caption models initialization skipped (Phase 3 simplified)")
            self.use_ai = False  # Set to False for Phase 3 simplified version
            
        except ImportError as e:
            self.logger.warning(f"AI libraries not available: {e}")
            self.use_ai = False
    
    def generate_captions(self, image_path: Path) -> Dict[str, Any]:
        """
        Generate captions, descriptions, keywords, and alt text for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing generated text content
        """
        results = {
            'file_path': str(image_path),
            'file_name': image_path.name,
            'generation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to read image: {image_path}")
                results['error'] = 'Failed to read image'
                return results
            
            # Convert to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # First extract metadata for GPS integration
            metadata = self._extract_basic_metadata(image_path)
            
            if self.use_ai:
                # Use AI models for generation (placeholder for full implementation)
                results.update(self._generate_with_ai(pil_image, image_path, metadata))
            else:
                # Use computer vision techniques for basic generation
                results.update(self._generate_with_cv(image, image_path, metadata))
            
            # Ensure all required fields are present
            self._ensure_required_fields(results)
            
            # Apply grammar and spell checking
            self._apply_text_corrections(results)
            
        except Exception as e:
            self.logger.error(f"Error generating captions for {image_path}: {e}")
            results['error'] = str(e)
            self._ensure_required_fields(results)
        
        return results
    
    def _extract_basic_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract basic metadata including GPS for caption enhancement."""
        metadata = {}
        
        try:
            from src.core.metadata_handler import MetadataHandler
            metadata_handler = MetadataHandler(self.config, self.logger)
            full_metadata = metadata_handler.extract_all_metadata(image_path)
            
            # Extract relevant fields for caption generation
            metadata['gps_latitude'] = full_metadata.get('gps_latitude_decimal')
            metadata['gps_longitude'] = full_metadata.get('gps_longitude_decimal')
            
            # Compile comprehensive GPS location information
            location_components = []
            
            # 1) N-S hemisphere
            ns_hemisphere = full_metadata.get('gps_hemisphere_ns', '')
            if ns_hemisphere:
                location_components.append(f"{ns_hemisphere} hemisphere")
                metadata['gps_hemisphere_ns'] = ns_hemisphere
            
            # 2) E-W hemisphere  
            ew_hemisphere = full_metadata.get('gps_hemisphere_ew', '')
            if ew_hemisphere:
                location_components.append(f"{ew_hemisphere} longitude")
                metadata['gps_hemisphere_ew'] = ew_hemisphere
            
            # 3) Continent
            continent = full_metadata.get('gps_continent', '')
            if continent and continent != 'Unknown':
                location_components.append(continent)
                metadata['gps_continent'] = continent
            
            # 4) Country
            country = full_metadata.get('gps_country', '')
            if country and 'geocoding' not in country.lower():
                location_components.append(country)
                metadata['gps_country'] = country
            
            # 5) Region
            region = full_metadata.get('gps_region', '')
            if region:
                location_components.append(f"{region} region")
                metadata['gps_region'] = region
            
            # 6) State or Province
            state = full_metadata.get('gps_state_province', '')
            if state:
                location_components.append(state)
                metadata['gps_state_province'] = state
            
            # 7) City or District
            city = full_metadata.get('gps_city_district', '')
            if city:
                location_components.append(city)
                metadata['gps_city_district'] = city
            
            # 8) Local Region, District, or Neighborhood
            local_area = full_metadata.get('gps_local_area', '')
            if local_area:
                location_components.append(f"{local_area} area")
                metadata['gps_local_area'] = local_area
            
            # 9) ISO tags/designators (if available)
            iso_tags = []
            if metadata.get('gps_country'):
                # Add country codes if available (simplified implementation)
                country_codes = {
                    'United States': 'US-USA',
                    'Canada': 'CA-CAN', 
                    'United Kingdom': 'GB-GBR',
                    'France': 'FR-FRA',
                    'Germany': 'DE-DEU',
                    'Japan': 'JP-JPN',
                    'Australia': 'AU-AUS'
                }
                country_name = metadata['gps_country']
                if country_name in country_codes:
                    iso_tags.append(country_codes[country_name])
            
            if iso_tags:
                metadata['gps_iso_tags'] = '; '.join(iso_tags)
                location_components.append(f"ISO: {'; '.join(iso_tags)}")
            
            # Compile human-readable location
            if location_components:
                metadata['gps_location'] = '; '.join(location_components)
            else:
                # Fallback to coordinates if no other location data
                lat = metadata.get('gps_latitude')
                lon = metadata.get('gps_longitude') 
                if lat is not None and lon is not None:
                    metadata['gps_location'] = f"Coordinates: {lat:.4f}; {lon:.4f}"
                else:
                    metadata['gps_location'] = 'Location data not available'
            
            # Other metadata for context
            metadata['camera_make'] = full_metadata.get('camera_make', '')
            metadata['camera_model'] = full_metadata.get('camera_model', '')
            metadata['datetime_original'] = full_metadata.get('exif_datetime_original', '')
            metadata['capture_season'] = full_metadata.get('capture_season_meteorological', '')
            metadata['capture_time'] = full_metadata.get('capture_time_of_day', '')
            
        except Exception as e:
            self.logger.debug(f"Could not extract metadata for caption generation: {e}")
            metadata['gps_location'] = 'Location data not available'
        
        return metadata
    
    def generate_caption(self, image_path: Path) -> Optional[str]:
        """
        Generate caption for image (simplified interface).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Primary caption string or None if error
        """
        try:
            results = self.generate_captions(image_path)
            return results.get('primary_caption', None)
        except Exception as e:
            self.logger.error(f"Error generating caption for {image_path}: {e}")
            return None
    
    def _generate_with_cv(self, image: np.ndarray, image_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content using computer vision techniques with enhanced object detection.
        
        Args:
            image: OpenCV image array
            image_path: Path to image file
            metadata: Extracted metadata
            
        Returns:
            Dictionary with generated content
        """
        results = {}
        
        # Analyze image properties
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Detect dominant colors
        colors = self._detect_dominant_colors(image)
        
        # Enhanced edge and shape detection
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # Detect basic shapes and patterns
        shapes_detected = self._detect_basic_shapes(image)
        
        # Analyze texture and patterns
        texture_analysis = self._analyze_texture(image)
        
        # Detect brightness and contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Enhanced image type determination
        image_context = self._determine_image_context(
            aspect_ratio, edge_density, brightness, contrast, 
            shapes_detected, texture_analysis, metadata
        )
        
        # Generate enhanced primary caption
        primary_caption = self._generate_enhanced_caption(
            image_context, colors, brightness, width, height, metadata
        )
        results['primary_caption'] = primary_caption
        
        # Generate alternate captions with more variety
        results['alternate_caption_1'] = self._generate_alternate_caption(
            image_context, colors, shapes_detected, 1, metadata
        )
        results['alternate_caption_2'] = self._generate_alternate_caption(
            image_context, colors, shapes_detected, 2, metadata
        )
        
        # Generate enhanced descriptions
        results['primary_description'] = self._generate_enhanced_description(
            image_context, colors, brightness, width, height, edge_density, 
            texture_analysis, metadata, verbose=True
        )
        results['alternate_description_1'] = self._generate_enhanced_description(
            image_context, colors, brightness, width, height, edge_density, 
            texture_analysis, metadata, verbose=True, variant=1
        )
        results['alternate_description_2'] = self._generate_enhanced_description(
            image_context, colors, brightness, width, height, edge_density, 
            texture_analysis, metadata, verbose=True, variant=2
        )
        
        # Generate enhanced keywords
        keywords = self._generate_enhanced_keywords(
            image_context, colors, shapes_detected, texture_analysis, metadata
        )
        results['keywords'] = '; '.join(keywords[:25])
        results['keyword_count'] = len(keywords[:25])
        
        # Generate enhanced tags
        tags = self._generate_enhanced_tags(keywords, image_context, metadata)
        results['tags'] = '; '.join(tags[:25])
        results['tag_count'] = len(tags[:25])
        
        # Generate enhanced alt text
        results['alt_text'] = self._generate_enhanced_alt_text(
            image_context, colors, primary_caption, metadata
        )
        
        # Include GPS location from metadata
        results['gps_location'] = metadata.get('gps_location', 'Location not available')
        
        return results
    
    def _detect_basic_shapes(self, image: np.ndarray) -> List[str]:
        """Detect basic shapes and objects in the image."""
        shapes = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, 
                                  param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None and len(circles[0]) > 2:
            shapes.append("circular objects")
        
        # Detect lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None and len(lines) > 10:
            shapes.append("linear structures")
        
        # Detect contours for shape analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Only consider significant shapes
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:
                    rectangular_count += 1
        
        if rectangular_count > 3:
            shapes.append("rectangular shapes")
        
        return shapes
    
    def _analyze_texture(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze texture patterns in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture metrics
        # Variance of Laplacian for texture smoothness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Local Binary Pattern approximation
        rows, cols = gray.shape
        lbp_variance = np.var([
            gray[1:rows-1, 1:cols-1],
            gray[0:rows-2, 0:cols-2],
            gray[2:rows, 2:cols]
        ])
        
        texture_info = {
            'smoothness': laplacian_var,
            'pattern_complexity': lbp_variance,
            'is_smooth': laplacian_var < 500,
            'is_textured': laplacian_var > 1000,
            'has_patterns': lbp_variance > 200
        }
        
        return texture_info
    
    def _determine_image_context(self, aspect_ratio: float, edge_density: float, 
                                brightness: float, contrast: float, shapes: List[str],
                                texture: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Determine enhanced image context with object detection hints."""
        context_parts = []
        
        # Orientation
        if aspect_ratio > 1.5:
            context_parts.append("landscape")
        elif aspect_ratio < 0.67:
            context_parts.append("portrait")
        else:
            context_parts.append("square")
        
        # Scene complexity
        if edge_density > 0.15 and len(shapes) > 2:
            context_parts.append("complex architectural")
        elif edge_density > 0.1:
            context_parts.append("detailed")
        elif edge_density < 0.03 and texture['is_smooth']:
            context_parts.append("minimalist")
        else:
            context_parts.append("moderate")
        
        # Content type inference
        if "circular objects" in shapes and "rectangular shapes" in shapes:
            context_parts.append("urban scene")
        elif texture['is_smooth'] and brightness > 180:
            context_parts.append("studio photograph")
        elif texture['has_patterns'] and edge_density > 0.08:
            context_parts.append("natural scene")
        elif contrast < 30:
            context_parts.append("atmospheric")
        
        # Time and season context from metadata
        if metadata.get('capture_time'):
            context_parts.append(f"{metadata['capture_time'].lower()}")
        
        if metadata.get('capture_season'):
            context_parts.append(f"{metadata['capture_season'].lower()}")
        
        return " ".join(context_parts)
    
    def _generate_enhanced_caption(self, context: str, colors: List[str], 
                                  brightness: float, width: int, height: int, 
                                  metadata: Dict[str, Any]) -> str:
        """Generate an enhanced caption with more specificity."""
        # Start with context-aware description
        if "architectural" in context or "urban" in context:
            base = f"An architectural composition featuring {colors[0] if colors else 'neutral'} tones"
        elif "natural" in context:
            base = f"A natural scene with {colors[0] if colors else 'earth'} coloration"
        elif "studio" in context:
            base = f"A studio photograph with {colors[0] if colors else 'controlled'} lighting"
        elif "minimalist" in context:
            base = f"A minimalist image emphasizing {colors[0] if colors else 'simple'} elements"
        else:
            base = f"A {context.split()[0] if context else 'standard'} image with {colors[0] if colors else 'varied'} hues"
        
        # Add specific details
        details = []
        
        if brightness > 200:
            details.append("bright illumination")
        elif brightness < 60:
            details.append("dramatic shadows")
        
        if width > 3000 or height > 3000:
            details.append("high resolution")
        
        # Add location if available
        location = metadata.get('gps_location', '')
        if location and location != 'Location not available' and 'not available' not in location.lower():
            details.append(f"captured at {location}")
        
        # Add camera info if available
        camera = metadata.get('camera_make', '')
        if camera and camera.strip():
            details.append(f"photographed with {camera}")
        
        # Combine elements
        if details:
            caption = f"{base}, featuring {', '.join(details[:2])}"
        else:
            caption = base
        
        return f"{caption}."
    
    def _generate_enhanced_description(self, context: str, colors: List[str], brightness: float,
                                     width: int, height: int, edge_density: float, 
                                     texture: Dict[str, Any], metadata: Dict[str, Any],
                                     verbose: bool = True, variant: int = 0) -> str:
        """Generate enhanced verbose descriptions with context awareness."""
        parts = []
        
        if variant == 0:
            parts.append(f"This {context} image measures {width} by {height} pixels")
            
            # Describe visual complexity
            if texture['is_textured']:
                parts.append("displaying rich textural details")
            elif texture['is_smooth']:
                parts.append("characterized by smooth gradients")
            
            # Color analysis
            if len(colors) >= 3:
                parts.append(f"The color palette primarily consists of {', '.join(colors[:3])}")
            elif colors:
                parts.append(f"Dominated by {colors[0]} tones")
            
            # Technical details
            parts.append(f"The exposure produces an average brightness of {brightness:.0f}")
            
            # Location context
            location = metadata.get('gps_location', '')
            if location and 'not available' not in location.lower():
                parts.append(f"Geographic location: {location}")
            
        elif variant == 1:
            # Focus on artistic elements
            parts.append(f"An artistic {context} composition")
            
            if edge_density > 0.1:
                parts.append("with strong definition and contrast")
            else:
                parts.append("emphasizing subtle tonal variations")
            
            # Time context
            time_info = metadata.get('capture_time', '')
            season_info = metadata.get('capture_season', '')
            if time_info and season_info:
                parts.append(f"captured during {season_info} {time_info}")
            
            # Camera details
            camera_make = metadata.get('camera_make', '')
            camera_model = metadata.get('camera_model', '')
            if camera_make and camera_model:
                parts.append(f"photographed using {camera_make} {camera_model}")
            
        else:  # variant == 2
            # Focus on technical aspects
            aspect = width / height
            parts.append(f"Technical specifications: {width}x{height} pixels (aspect ratio {aspect:.2f}:1)")
            
            if texture['smoothness'] > 1000:
                parts.append("High detail retention with complex texture patterns")
            else:
                parts.append("Smooth surface rendering with minimal noise")
            
            # Date context
            date_info = metadata.get('datetime_original', '')
            if date_info:
                parts.append(f"Captured on {date_info}")
        
        description = ". ".join(parts) + "."
        return description
    
    def _generate_enhanced_keywords(self, context: str, colors: List[str], shapes: List[str],
                                   texture: Dict[str, Any], metadata: Dict[str, Any]) -> List[str]:
        """Generate enhanced keywords with context awareness."""
        keywords = []
        
        # Context-based keywords
        keywords.extend(context.split())
        
        # Color keywords
        keywords.extend(colors[:5])  # Top 5 colors
        
        # Shape-based keywords
        if "circular objects" in shapes:
            keywords.extend(['circles', 'round', 'curved'])
        if "rectangular shapes" in shapes:
            keywords.extend(['geometric', 'angular', 'structured'])
        if "linear structures" in shapes:
            keywords.extend(['lines', 'linear', 'directional'])
        
        # Texture keywords
        if texture['is_textured']:
            keywords.extend(['textured', 'detailed', 'complex'])
        if texture['is_smooth']:
            keywords.extend(['smooth', 'clean', 'minimal'])
        if texture['has_patterns']:
            keywords.extend(['patterned', 'repetitive', 'decorative'])
        
        # Location-based keywords
        location = metadata.get('gps_location', '')
        if location and 'not available' not in location.lower():
            # Extract location keywords (simplified)
            location_parts = location.replace(',', ' ').split()
            keywords.extend([part.lower() for part in location_parts if len(part) > 2])
        
        # Time-based keywords
        season = metadata.get('capture_season', '')
        time_of_day = metadata.get('capture_time', '')
        if season:
            keywords.append(season.lower())
        if time_of_day:
            keywords.append(time_of_day.lower().replace(' ', '_'))
        
        # Equipment keywords
        camera = metadata.get('camera_make', '')
        if camera:
            keywords.append(camera.lower())
        
        # Photography keywords
        keywords.extend(['photography', 'digital', 'image', 'visual', 'composition'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword and keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword.lower())
        
        return unique_keywords
    
    def _generate_enhanced_alt_text(self, context: str, colors: List[str], 
                                   caption: str, metadata: Dict[str, Any]) -> str:
        """Generate enhanced alt text with accessibility focus."""
        # Start with context
        alt_parts = []
        
        if "architectural" in context:
            alt_parts.append("Architectural photograph")
        elif "natural" in context:
            alt_parts.append("Natural scene photograph")
        elif "studio" in context:
            alt_parts.append("Studio photograph")
        else:
            alt_parts.append("Photograph")
        
        # Add primary color
        if colors:
            alt_parts.append(f"featuring {colors[0]} tones")
        
        # Add location if significant
        location = metadata.get('gps_location', '')
        if location and 'not available' not in location.lower() and len(location) < 30:
            alt_parts.append(f"taken at {location}")
        
        alt_text = " ".join(alt_parts)
        
        # Ensure accessibility compliance (max 125 characters)
        if len(alt_text) > 125:
            alt_text = alt_text[:122] + "..."
        
        return alt_text + "."
    
    def _ensure_required_fields(self, results: Dict[str, Any]):
        """Ensure all required fields are present in results."""
        required_fields = [
            'primary_caption',
            'alternate_caption_1',
            'alternate_caption_2',
            'primary_description',
            'alternate_description_1',
            'alternate_description_2',
            'keywords',
            'tags',
            'alt_text'
        ]
        
        for field in required_fields:
            if field not in results:
                results[field] = 'Not available'
    
    def _apply_text_corrections(self, results: Dict[str, Any]):
        """Apply comprehensive grammar and spell checking to all text fields."""
        from ..utils.text_processor import text_processor
        
        # Process all text fields using the enhanced text processor
        results.update(text_processor.process_all_text_fields(results))
    
    def _detect_dominant_colors(self, image: np.ndarray) -> List[str]:
        """Detect dominant colors in the image."""
        # Resize for faster processing
        small = cv2.resize(image, (150, 150))
        
        # Convert to RGB and reshape
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pixels = rgb.reshape(-1, 3)
        
        # Use k-means to find dominant colors
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get color names
        colors = []
        for color in kmeans.cluster_centers_:
            color_name = self._rgb_to_name(color)
            colors.append(color_name)
        
        return colors
    
    def _rgb_to_name(self, rgb: np.ndarray) -> str:
        """Convert RGB color to descriptive name."""
        # Placeholder for actual color naming logic
        # Assuming rgb is a 1x3 array-like [R, G, B]
        r, g, b = rgb.astype(int)
        return f"rgb({r},{g},{b})"
    
    def _generate_enhanced_tags(self, keywords: List[str], image_context: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate enhanced hashtags/tags based on keywords and context."""
        tags = []
        
        # Convert keywords to hashtag format
        for keyword in keywords[:15]:  # Limit base keywords for tag conversion
            if keyword and len(keyword) > 2:
                # Clean keyword for hashtag format
                tag = keyword.replace(' ', '').replace('-', '').lower()
                if tag.isalnum() and len(tag) > 2:
                    tags.append(f"#{tag}")
        
        # Add context-based tags
        context_tags = []
        if "landscape" in image_context:
            context_tags.extend(["#landscape", "#scenery", "#nature"])
        if "portrait" in image_context:
            context_tags.extend(["#portrait", "#people", "#photography"])
        if "architectural" in image_context or "urban" in image_context:
            context_tags.extend(["#architecture", "#urban", "#cityscape"])
        if "studio" in image_context:
            context_tags.extend(["#studio", "#professional", "#photoshoot"])
        if "minimalist" in image_context:
            context_tags.extend(["#minimal", "#clean", "#simple"])
        if "natural" in image_context:
            context_tags.extend(["#nature", "#outdoor", "#natural"])
        
        # Add unique context tags
        for tag in context_tags:
            if tag not in tags:
                tags.append(tag)
        
        # Add time-based tags
        season = metadata.get('capture_season', '')
        time_of_day = metadata.get('capture_time', '')
        if season:
            tags.append(f"#{season.lower()}")
        if time_of_day and ' ' in time_of_day:
            time_tag = time_of_day.lower().replace(' ', '')
            tags.append(f"#{time_tag}")
        
        # Add location-based tags (if available)
        location = metadata.get('gps_location', '')
        if location and 'not available' not in location.lower():
            # Extract meaningful location terms for hashtags
            location_parts = location.lower().replace(',', ' ').replace(';', ' ').split()
            location_tags = []
            for part in location_parts:
                if len(part) > 3 and part.isalpha():
                    location_tags.append(f"#{part}")
            
            # Add top location tags
            tags.extend(location_tags[:3])
        
        # Add photography technique tags
        photo_tags = [
            "#photography", "#digital", "#image", "#photo", "#picture",
            "#capture", "#shot", "#frame", "#composition", "#visual"
        ]
        
        # Add some photo tags that aren't already present
        for tag in photo_tags:
            if tag not in tags and len(tags) < 20:
                tags.append(tag)
        
        # Add camera-based tags
        camera_make = metadata.get('camera_make', '')
        if camera_make and len(tags) < 22:
            camera_tag = f"#{camera_make.lower().replace(' ', '')}"
            if camera_tag not in tags:
                tags.append(camera_tag)
        
        # Ensure we have at least 1 and at most 25 tags
        if not tags:
            tags = ["#photography", "#image"]
        
        return tags[:25]  # Maximum 25 tags as required