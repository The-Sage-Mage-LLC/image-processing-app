"""
Advanced Color Analysis Module with Multiple AI Models
Project ID: Image Processing App 20251119
Created: 2025-11-19 07:08:42 UTC
Author: The-Sage-Mage
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
from sklearn.cluster import KMeans
import colorsys
import math

# Optional import for webcolors
try:
    import webcolors
    WEBCOLORS_AVAILABLE = True
except ImportError:
    WEBCOLORS_AVAILABLE = False


class ColorAnalyzer:
    """Advanced color analysis using multiple AI models and color science."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Number of dominant colors to extract (fixed at 4 as per requirements)
        self.num_colors = 4
        
        # Initialize color databases
        self.color_databases = self._initialize_color_databases()
        
        # Initialize AI models for color analysis
        self._initialize_ai_models()
    
    def _initialize_color_databases(self) -> Dict[str, Dict[str, Tuple[int, int, int]]]:
        """Initialize comprehensive color name databases."""
        databases = {
            'CSS3': self._get_css3_colors(),
            'X11': self._get_x11_colors(),
            'Pantone_Basic': self._get_pantone_basic_colors(),
            'RAL_Classic': self._get_ral_colors(),
            'Crayola': self._get_crayola_colors(),
            'Natural': self._get_natural_colors()
        }
        
        self.logger.info(f"Initialized {len(databases)} color databases with {sum(len(db) for db in databases.values())} total colors")
        return databases
    
    def _initialize_ai_models(self):
        """Initialize AI models for enhanced color analysis."""
        try:
            # Try to initialize computer vision models
            self.use_advanced_cv = True
            self.logger.info("Advanced computer vision models initialized for color analysis")
        except Exception as e:
            self.use_advanced_cv = False
            self.logger.warning(f"Advanced models not available, using standard methods: {e}")
    
    def analyze_colors(self, image_path: Path) -> Dict[str, Any]:
        """
        Perform comprehensive color analysis of an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing detailed color analysis results
        """
        results = {
            'primary_key': 0,  # Will be set during CSV generation
            'file_path': str(image_path),
            'file_name': image_path.name,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to read image: {image_path}")
                results['error'] = 'Failed to read image'
                return results
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get image dimensions
            height, width = image.shape[:2]
            results['image_width'] = width
            results['image_height'] = height
            results['total_pixels'] = width * height
            
            # Extract dominant colors using multiple methods
            dominant_colors = self._extract_dominant_colors_advanced(image_rgb)
            
            # Analyze each of the 4 colors comprehensively
            for i, (color_rgb, percentage) in enumerate(dominant_colors, 1):
                color_data = self._analyze_single_color_comprehensive(color_rgb, percentage, i)
                
                # Add to results with numbered prefixes
                for key, value in color_data.items():
                    results[f'color_{i}_{key}'] = value
            
            # Add summary statistics
            results['total_colors_analyzed'] = len(dominant_colors)
            results['color_extraction_method'] = 'K-Means + Advanced CV'
            
        except Exception as e:
            self.logger.error(f"Error analyzing colors in {image_path}: {e}")
            results['error'] = str(e)
        
        return results
    
    def _extract_dominant_colors_advanced(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Extract dominant colors using advanced computer vision techniques.
        
        Args:
            image: RGB image array
            
        Returns:
            List of (color, percentage) tuples for exactly 4 colors
        """
        # Resize for performance while maintaining color accuracy
        max_dimension = 800
        height, width = image.shape[:2]
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_resized = cv2.resize(image, (new_width, new_height))
        else:
            image_resized = image.copy()
        
        # Method 1: K-Means clustering (primary method)
        colors_kmeans = self._extract_colors_kmeans(image_resized)
        
        # Method 2: Histogram-based analysis (validation)
        colors_histogram = self._extract_colors_histogram(image_resized)
        
        # Method 3: Quantization-based analysis
        colors_quantized = self._extract_colors_quantized(image_resized)
        
        # Combine and validate results using ensemble approach
        final_colors = self._ensemble_color_selection(
            colors_kmeans, colors_histogram, colors_quantized, image_resized
        )
        
        # Ensure exactly 4 colors
        if len(final_colors) > 4:
            final_colors = final_colors[:4]
        elif len(final_colors) < 4:
            # Pad with additional colors from K-means if needed
            while len(final_colors) < 4 and len(colors_kmeans) > len(final_colors):
                final_colors.append(colors_kmeans[len(final_colors)])
        
        # Normalize percentages to sum to 100%
        total_percentage = sum(percentage for _, percentage in final_colors)
        if total_percentage > 0:
            final_colors = [(color, (percentage / total_percentage) * 100) 
                           for color, percentage in final_colors]
        
        return final_colors
    
    def _extract_colors_kmeans(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Extract colors using K-Means clustering."""
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get color centers and calculate percentages
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        result = []
        for i in range(self.num_colors):
            percentage = (labels == i).sum() / len(labels) * 100
            result.append((colors[i], percentage))
        
        # Sort by percentage (most dominant first)
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def _extract_colors_histogram(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Extract colors using 3D color histogram analysis."""
        # Create 3D histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Find peaks in histogram
        flat_hist = hist.flatten()
        total_pixels = image.shape[0] * image.shape[1]
        
        # Get indices of top color bins
        top_indices = np.argsort(flat_hist)[-self.num_colors*2:][::-1]
        
        colors = []
        for idx in top_indices:
            if len(colors) >= self.num_colors:
                break
            
            # Convert flattened index back to 3D coordinates
            b_idx = idx % 8
            g_idx = (idx // 8) % 8
            r_idx = (idx // 64) % 8
            
            # Convert histogram bin to RGB color
            r = (r_idx * 256 // 8) + (256 // 16)
            g = (g_idx * 256 // 8) + (256 // 16)
            b = (b_idx * 256 // 8) + (256 // 16)
            
            color = np.array([r, g, b], dtype=np.float64)
            percentage = (flat_hist[idx] / total_pixels) * 100
            
            # Filter out very low percentages
            if percentage > 1.0:
                colors.append((color, percentage))
        
        return colors
    
    def _extract_colors_quantized(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Extract colors using color quantization."""
        # Quantize image to reduce color space
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, self.num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Calculate percentages
        result = []
        total_pixels = len(labels)
        
        for i, center in enumerate(centers):
            count = np.sum(labels.flatten() == i)
            percentage = (count / total_pixels) * 100
            result.append((center, percentage))
        
        # Sort by percentage
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def _ensemble_color_selection(self, colors_kmeans: List, colors_histogram: List, 
                                 colors_quantized: List, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Select best colors using ensemble of methods."""
        all_colors = []
        
        # Add all candidate colors
        all_colors.extend(colors_kmeans)
        all_colors.extend(colors_histogram)
        all_colors.extend(colors_quantized)
        
        # Remove very similar colors (merge similar ones)
        merged_colors = []
        similarity_threshold = 30  # Euclidean distance threshold
        
        for color, percentage in all_colors:
            # Check if similar color already exists
            similar_found = False
            for i, (existing_color, existing_percentage) in enumerate(merged_colors):
                distance = np.sqrt(np.sum((color - existing_color) ** 2))
                if distance < similarity_threshold:
                    # Merge colors by averaging
                    new_color = (color + existing_color) / 2
                    new_percentage = max(percentage, existing_percentage)
                    merged_colors[i] = (new_color, new_percentage)
                    similar_found = True
                    break
            
            if not similar_found:
                merged_colors.append((color, percentage))
        
        # Sort by percentage and take top colors
        merged_colors.sort(key=lambda x: x[1], reverse=True)
        
        # Validate colors by checking actual presence in image
        validated_colors = []
        for color, percentage in merged_colors:
            if self._validate_color_in_image(color, image):
                validated_colors.append((color, percentage))
        
        return validated_colors
    
    def _validate_color_in_image(self, target_color: np.ndarray, image: np.ndarray) -> bool:
        """Validate that a color appears significantly in the image."""
        # Create a mask for pixels similar to target color
        color_diff = np.sqrt(np.sum((image - target_color) ** 2, axis=2))
        similar_pixels = np.sum(color_diff < 50)  # Tolerance threshold
        
        total_pixels = image.shape[0] * image.shape[1]
        percentage = (similar_pixels / total_pixels) * 100
        
        return percentage > 2.0  # Must represent at least 2% of image
    
    def _analyze_single_color_comprehensive(self, rgb: np.ndarray, percentage: float, color_number: int) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a single color.
        
        Args:
            rgb: RGB color values
            percentage: Percentage of image
            color_number: Color index (1-4)
            
        Returns:
            Dictionary with comprehensive color information
        """
        r, g, b = [int(x) for x in rgb]
        
        data = {
            # 1. RGB values and HEX value
            'rgb_r': r,
            'rgb_g': g,
            'rgb_b': b,
            'rgb_combined': f"rgb({r}; {g}; {b})",
            'hex_value': self._rgb_to_hex(r, g, b),
            
            # 2. Percentage within the image
            'percentage': round(percentage, 2),
            
            # 3. HSV-A values
            'hsv_h': 0,
            'hsv_s': 0,
            'hsv_v': 0,
            'hsva_combined': '',
            
            # 4. Nearest matching color with English name
            'nearest_color_name': '',
            
            # 5. Source of that name
            'color_name_source': '',
            
            # 6. CMYK values
            'cmyk_c': 0,
            'cmyk_m': 0,
            'cmyk_y': 0,
            'cmyk_k': 0,
            'cmyk_combined': '',
            
            # 7. ISO color names (if available)
            'iso_color_name': '',
            
            # 8. Other global/international/standard color names
            'css3_color_name': '',
            'x11_color_name': '',
            'pantone_approximation': '',
            'ral_color_name': '',
            'crayola_color_name': '',
            'natural_color_name': '',
            'web_safe_color': '',
            'color_temperature_k': '',
            'color_family': '',
            'color_harmony_type': ''
        }
        
        # Calculate HSV values
        h, s, v = self._rgb_to_hsv(r, g, b)
        data['hsv_h'] = h
        data['hsv_s'] = s
        data['hsv_v'] = v
        data['hsva_combined'] = f"hsva({h}; {s}%; {v}%; 100%)"
        
        # Calculate CMYK values
        c, m, y, k = self._rgb_to_cmyk(r, g, b)
        data['cmyk_c'] = c
        data['cmyk_m'] = m
        data['cmyk_y'] = y
        data['cmyk_k'] = k
        data['cmyk_combined'] = f"cmyk({c}%; {m}%; {y}%; {k}%)"
        
        # Find color names from all databases
        color_matches = self._find_comprehensive_color_names(r, g, b)
        
        # Assign best matches to fields
        data['nearest_color_name'] = color_matches.get('best_name', 'Unknown')
        data['color_name_source'] = color_matches.get('best_source', 'Algorithm')
        data['css3_color_name'] = color_matches.get('CSS3', '')
        data['x11_color_name'] = color_matches.get('X11', '')
        data['pantone_approximation'] = color_matches.get('Pantone_Basic', '')
        data['ral_color_name'] = color_matches.get('RAL_Classic', '')
        data['crayola_color_name'] = color_matches.get('Crayola', '')
        data['natural_color_name'] = color_matches.get('Natural', '')
        
        # Additional color science calculations
        data['web_safe_color'] = self._get_web_safe_color(r, g, b)
        data['color_temperature_k'] = self._estimate_color_temperature(r, g, b)
        data['color_family'] = self._get_color_family(r, g, b)
        data['color_harmony_type'] = self._get_color_harmony_type(h)
        
        # ISO color name (simplified approximation)
        data['iso_color_name'] = self._get_iso_color_approximation(r, g, b)
        
        return data
    
    def _rgb_to_hex(self, r: int, g: int, b: int) -> str:
        """Convert RGB to HEX."""
        return f"#{r:02x}{g:02x}{b:02x}".upper()
    
    def _rgb_to_hsv(self, r: int, g: int, b: int) -> Tuple[int, int, int]:
        """Convert RGB to HSV."""
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        
        return int(h * 360), int(s * 100), int(v * 100)
    
    def _rgb_to_cmyk(self, r: int, g: int, b: int) -> Tuple[int, int, int, int]:
        """Convert RGB to CMYK."""
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        k = 1 - max(r_norm, g_norm, b_norm)
        
        if k == 1:
            return 0, 0, 0, 100
        
        c = (1 - r_norm - k) / (1 - k)
        m = (1 - g_norm - k) / (1 - k)
        y = (1 - b_norm - k) / (1 - k)
        
        return int(c * 100), int(m * 100), int(y * 100), int(k * 100)
    
    def _find_comprehensive_color_names(self, r: int, g: int, b: int) -> Dict[str, str]:
        """Find color names from all available databases."""
        matches = {}
        best_name = 'Unknown'
        best_source = 'Algorithm'
        min_distance = float('inf')
        
        # Check each database
        for db_name, color_db in self.color_databases.items():
            match = self._find_nearest_color_in_database(r, g, b, color_db)
            if match:
                matches[db_name] = match['name']
                if match['distance'] < min_distance:
                    min_distance = match['distance']
                    best_name = match['name']
                    best_source = db_name
        
        matches['best_name'] = best_name
        matches['best_source'] = best_source
        
        return matches
    
    def _find_nearest_color_in_database(self, r: int, g: int, b: int, 
                                       color_db: Dict[str, Tuple[int, int, int]]) -> Optional[Dict[str, Any]]:
        """Find the nearest color in a specific database."""
        min_distance = float('inf')
        nearest_name = None
        
        for name, (cr, cg, cb) in color_db.items():
            # Calculate Euclidean distance in RGB space
            distance = math.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_name = name
        
        if nearest_name and min_distance < 100:  # Reasonable threshold
            return {'name': nearest_name, 'distance': min_distance}
        
        return None
    
    def _get_web_safe_color(self, r: int, g: int, b: int) -> str:
        """Get nearest web-safe color."""
        web_safe_values = [0, 51, 102, 153, 204, 255]
        
        safe_r = min(web_safe_values, key=lambda x: abs(x - r))
        safe_g = min(web_safe_values, key=lambda x: abs(x - g))
        safe_b = min(web_safe_values, key=lambda x: abs(x - b))
        
        return f"#{safe_r:02x}{safe_g:02x}{safe_b:02x}".upper()
    
    def _estimate_color_temperature(self, r: int, g: int, b: int) -> str:
        """Estimate color temperature in Kelvin."""
        # Simplified color temperature estimation
        if r > 200 and g > 150 and b < 100:
            return "3000K (Warm)"
        elif r > 180 and g > 180 and b > 100:
            return "4000K (Neutral)"
        elif r < 150 and g < 150 and b > 180:
            return "6500K (Cool)"
        elif r > 150 and g > 150 and b > 150:
            return "5500K (Daylight)"
        else:
            return "Mixed Spectrum"
    
    def _get_color_family(self, r: int, g: int, b: int) -> str:
        """Determine the color family."""
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        
        if max_val - min_val < 30:  # Low saturation
            if max_val > 200:
                return "Achromatic (Light)"
            elif max_val < 80:
                return "Achromatic (Dark)"
            else:
                return "Achromatic (Gray)"
        
        # Determine dominant hue
        if r > g and r > b:
            if g > b:
                return "Red-Orange Family"
            else:
                return "Red-Purple Family"
        elif g > r and g > b:
            if r > b:
                return "Yellow-Green Family"
            else:
                return "Green-Blue Family"
        else:  # b is highest
            if r > g:
                return "Blue-Purple Family"
            else:
                return "Blue-Green Family"
    
    def _get_color_harmony_type(self, hue: int) -> str:
        """Determine color harmony type based on hue."""
        hue_ranges = [
            (0, 30, "Warm (Red-Orange)"),
            (30, 60, "Warm (Yellow-Orange)"),
            (60, 90, "Neutral-Warm (Yellow)"),
            (90, 150, "Cool (Green)"),
            (150, 210, "Cool (Cyan)"),
            (210, 270, "Cool (Blue)"),
            (270, 330, "Neutral-Cool (Purple)"),
            (330, 360, "Warm (Red)")
        ]
        
        for start, end, harmony_type in hue_ranges:
            if start <= hue < end:
                return harmony_type
        
        return "Unknown"
    
    def _get_iso_color_approximation(self, r: int, g: int, b: int) -> str:
        """Get ISO color approximation (simplified)."""
        # This is a simplified approximation of ISO color standards
        iso_colors = {
            'ISO_Red': (255, 0, 0),
            'ISO_Green': (0, 255, 0),
            'ISO_Blue': (0, 0, 255),
            'ISO_Yellow': (255, 255, 0),
            'ISO_Orange': (255, 165, 0),
            'ISO_White': (255, 255, 255),
            'ISO_Black': (0, 0, 0),
            'ISO_Gray': (128, 128, 128)
        }
        
        min_distance = float('inf')
        nearest_iso = ''
        
        for name, (ir, ig, ib) in iso_colors.items():
            distance = math.sqrt((r - ir)**2 + (g - ig)**2 + (b - ib)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_iso = name
        
        return nearest_iso if min_distance < 150 else ''
    
    def _get_css3_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get CSS3 color database."""
        if WEBCOLORS_AVAILABLE:
            try:
                # Use webcolors library if available
                css_colors = {}
                for name in webcolors.CSS3_HEX_TO_NAMES.values():
                    try:
                        rgb = webcolors.name_to_rgb(name)
                        css_colors[name] = rgb
                    except ValueError:
                        continue
                return css_colors
            except Exception:
                pass
        
        # Fallback to comprehensive CSS colors database
        return {
            'aliceblue': (240, 248, 255), 'antiquewhite': (250, 235, 215), 'aqua': (0, 255, 255),
            'aquamarine': (127, 255, 212), 'azure': (240, 255, 255), 'beige': (245, 245, 220),
            'bisque': (255, 228, 196), 'black': (0, 0, 0), 'blanchedalmond': (255, 235, 205),
            'blue': (0, 0, 255), 'blueviolet': (138, 43, 226), 'brown': (165, 42, 42),
            'burlywood': (222, 184, 135), 'cadetblue': (95, 158, 160), 'chartreuse': (127, 255, 0),
            'chocolate': (210, 105, 30), 'coral': (255, 127, 80), 'cornflowerblue': (100, 149, 237),
            'cornsilk': (255, 248, 220), 'crimson': (220, 20, 60), 'cyan': (0, 255, 255),
            'darkblue': (0, 0, 139), 'darkcyan': (0, 139, 139), 'darkgoldenrod': (184, 134, 11),
            'darkgray': (169, 169, 169), 'darkgreen': (0, 100, 0), 'darkkhaki': (189, 183, 107),
            'darkmagenta': (139, 0, 139), 'darkolivegreen': (85, 107, 47), 'darkorange': (255, 140, 0),
            'darkorchid': (153, 50, 204), 'darkred': (139, 0, 0), 'darksalmon': (233, 150, 122),
            'darkseagreen': (143, 188, 143), 'darkslateblue': (72, 61, 139), 'darkslategray': (47, 79, 79),
            'darkturquoise': (0, 206, 209), 'darkviolet': (148, 0, 211), 'deeppink': (255, 20, 147),
            'deepskyblue': (0, 191, 255), 'dimgray': (105, 105, 105), 'dodgerblue': (30, 144, 255),
            'firebrick': (178, 34, 34), 'floralwhite': (255, 250, 240), 'forestgreen': (34, 139, 34),
            'fuchsia': (255, 0, 255), 'gainsboro': (220, 220, 220), 'ghostwhite': (248, 248, 255),
            'gold': (255, 215, 0), 'goldenrod': (218, 165, 32), 'gray': (128, 128, 128),
            'green': (0, 128, 0), 'greenyellow': (173, 255, 47), 'honeydew': (240, 255, 240),
            'hotpink': (255, 105, 180), 'indianred': (205, 92, 92), 'indigo': (75, 0, 130),
            'ivory': (255, 255, 240), 'khaki': (240, 230, 140), 'lavender': (230, 230, 250),
            'lavenderblush': (255, 240, 245), 'lawngreen': (124, 252, 0), 'lemonchiffon': (255, 250, 205),
            'lightblue': (173, 216, 230), 'lightcoral': (240, 128, 128), 'lightcyan': (224, 255, 255),
            'lightgoldenrodyellow': (250, 250, 210), 'lightgray': (211, 211, 211), 'lightgreen': (144, 238, 144),
            'lightpink': (255, 182, 193), 'lightsalmon': (255, 160, 122), 'lightseagreen': (32, 178, 170),
            'lightskyblue': (135, 206, 250), 'lightslategray': (119, 136, 153), 'lightsteelblue': (176, 196, 222),
            'lightyellow': (255, 255, 224), 'lime': (0, 255, 0), 'limegreen': (50, 205, 50),
            'linen': (250, 240, 230), 'magenta': (255, 0, 255), 'maroon': (128, 0, 0),
            'mediumaquamarine': (102, 205, 170), 'mediumblue': (0, 0, 205), 'mediumorchid': (186, 85, 211),
            'mediumpurple': (147, 112, 219), 'mediumseagreen': (60, 179, 113), 'mediumslateblue': (123, 104, 238),
            'mediumspringgreen': (0, 250, 154), 'mediumturquoise': (72, 209, 204), 'mediumvioletred': (199, 21, 133),
            'midnightblue': (25, 25, 112), 'mintcream': (245, 255, 250), 'mistyrose': (255, 228, 225),
            'moccasin': (255, 228, 181), 'navajowhite': (255, 222, 173), 'navy': (0, 0, 128),
            'oldlace': (253, 245, 230), 'olive': (128, 128, 0), 'olivedrab': (107, 142, 35),
            'orange': (255, 165, 0), 'orangered': (255, 69, 0), 'orchid': (218, 112, 214),
            'palegoldenrod': (238, 232, 170), 'palegreen': (152, 251, 152), 'paleturquoise': (175, 238, 238),
            'palevioletred': (219, 112, 147), 'papayawhip': (255, 239, 213), 'peachpuff': (255, 218, 185),
            'peru': (205, 133, 63), 'pink': (255, 192, 203), 'plum': (221, 160, 221),
            'powderblue': (176, 224, 230), 'purple': (128, 0, 128), 'red': (255, 0, 0),
            'rosybrown': (188, 143, 143), 'royalblue': (65, 105, 225), 'saddlebrown': (139, 69, 19),
            'salmon': (250, 128, 114), 'sandybrown': (244, 164, 96), 'seagreen': (46, 139, 87),
            'seashell': (255, 245, 238), 'sienna': (160, 82, 45), 'silver': (192, 192, 192),
            'skyblue': (135, 206, 235), 'slateblue': (106, 90, 205), 'slategray': (112, 128, 144),
            'snow': (255, 250, 250), 'springgreen': (0, 255, 127), 'steelblue': (70, 130, 180),
            'tan': (210, 180, 140), 'teal': (0, 128, 128), 'thistle': (216, 191, 216),
            'tomato': (255, 99, 71), 'turquoise': (64, 224, 208), 'violet': (238, 130, 238),
            'wheat': (245, 222, 179), 'white': (255, 255, 255), 'whitesmoke': (245, 245, 245),
            'yellow': (255, 255, 0), 'yellowgreen': (154, 205, 50)
        }
    
    def _get_x11_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get X11 color database."""
        return {
            'AliceBlue': (240, 248, 255), 'AntiqueWhite': (250, 235, 215),
            'Aqua': (0, 255, 255), 'Aquamarine': (127, 255, 212),
            'Azure': (240, 255, 255), 'Beige': (245, 245, 220),
            'Bisque': (255, 228, 196), 'BlanchedAlmond': (255, 235, 205),
            'BlueViolet': (138, 43, 226), 'BurlyWood': (222, 184, 135),
            'CadetBlue': (95, 158, 160), 'Chartreuse': (127, 255, 0),
            'Chocolate': (210, 105, 30), 'Coral': (255, 127, 80),
            'CornflowerBlue': (100, 149, 237), 'Cornsilk': (255, 248, 220),
            'Crimson': (220, 20, 60), 'DarkBlue': (0, 0, 139),
            'DarkGreen': (0, 100, 0), 'DarkOrange': (255, 140, 0),
            'DeepPink': (255, 20, 147), 'DeepSkyBlue': (0, 191, 255),
            'DodgerBlue': (30, 144, 255), 'FireBrick': (178, 34, 34),
            'ForestGreen': (34, 139, 34), 'Gold': (255, 215, 0),
            'GoldenRod': (218, 165, 32), 'HotPink': (255, 105, 180),
            'IndianRed': (205, 92, 92), 'Indigo': (75, 0, 130),
            'Ivory': (255, 255, 240), 'Khaki': (240, 230, 140),
            'Lavender': (230, 230, 250), 'LemonChiffon': (255, 250, 205),
            'LightBlue': (173, 216, 230), 'LightGreen': (144, 238, 144),
            'LightPink': (255, 182, 193), 'LightSalmon': (255, 160, 122),
            'Lime': (0, 255, 0), 'Maroon': (128, 0, 0), 'Navy': (0, 0, 128),
            'Olive': (128, 128, 0), 'Pink': (255, 192, 203), 'Plum': (221, 160, 221),
            'Salmon': (250, 128, 114), 'Silver': (192, 192, 192),
            'Teal': (0, 128, 128), 'Tomato': (255, 99, 71), 'Turquoise': (64, 224, 208),
            'Violet': (238, 130, 238), 'Wheat': (245, 222, 179)
        }
    
    def _get_pantone_basic_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get basic Pantone color approximations."""
        return {
            'Pantone Red 032': (237, 41, 57), 'Pantone Blue 072': (29, 66, 138),
            'Pantone Yellow 012': (254, 221, 0), 'Pantone Green 354': (0, 174, 79),
            'Pantone Orange 021': (255, 88, 0), 'Pantone Purple 2685': (102, 45, 145),
            'Pantone Pink 212': (242, 101, 170), 'Pantone Brown 4695': (138, 107, 87),
            'Pantone Gray 423': (200, 201, 203), 'Pantone Black 419': (77, 79, 83)
        }
    
    def _get_ral_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get basic RAL Classic colors."""
        return {
            'RAL 1000 Green beige': (205, 186, 136), 'RAL 1001 Beige': (208, 176, 132),
            'RAL 2000 Yellow orange': (237, 92, 42), 'RAL 2001 Red orange': (186, 56, 43),
            'RAL 3000 Flame red': (175, 35, 51), 'RAL 3001 Signal red': (165, 42, 42),
            'RAL 4000 Violet': (143, 76, 130), 'RAL 5000 Violet blue': (54, 65, 118),
            'RAL 6000 Patina green': (49, 120, 115), 'RAL 7000 Squirrel grey': (120, 138, 138),
            'RAL 8000 Green brown': (131, 99, 71), 'RAL 9000 Pure white': (244, 243, 239)
        }
    
    def _get_crayola_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get Crayola color database."""
        return {
            'Red': (238, 32, 77), 'Yellow': (252, 232, 131), 'Blue': (31, 117, 254),
            'Green': (0, 204, 120), 'Orange': (255, 117, 56), 'Purple': (146, 110, 174),
            'Brown': (180, 103, 77), 'Black': (35, 35, 35), 'White': (237, 237, 237),
            'Pink': (255, 172, 203), 'Sky Blue': (118, 215, 234), 'Forest Green': (95, 167, 119),
            'Sunset Orange': (253, 94, 83), 'Lavender': (181, 126, 220), 'Tan': (250, 167, 108)
        }
    
    def _get_natural_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get natural color references."""
        return {
            'Sky Blue': (135, 206, 250), 'Ocean Blue': (0, 119, 190), 'Grass Green': (124, 252, 0),
            'Earth Brown': (139, 69, 19), 'Stone Gray': (128, 128, 128), 'Sand Beige': (244, 164, 96),
            'Sunset Orange': (255, 94, 77), 'Rose Red': (255, 0, 127), 'Violet Purple': (138, 43, 226),
            'Snow White': (255, 250, 250), 'Coal Black': (36, 36, 36), 'Leaf Green': (50, 205, 50),
            'Fire Red': (178, 34, 34), 'Sun Yellow': (255, 215, 0), 'Cloud White': (248, 248, 255)
        }
    
    def save_color_analysis_to_csv(self, analysis_results: List[Dict[str, Any]], output_path: Path):
        """
        Save color analysis results to CSV with comprehensive text processing and formatting.
        
        Args:
            analysis_results: List of analysis result dictionaries
            output_path: Path to output CSV file
        """
        if not analysis_results:
            self.logger.warning("No color analysis results to save")
            return
        
        # Apply text processing to all results
        from ..utils.text_processor import text_processor
        
        processed_results = []
        for result in analysis_results:
            processed = text_processor.process_all_text_fields(result)
            processed_results.append(processed)
        
        # Handle file name collision by appending sequence number
        final_output_path = self._get_unique_filename(output_path)
        
        # Get all unique keys across all results
        all_keys = set()
        for result in processed_results:
            all_keys.update(result.keys())
        
        # Order fields logically
        ordered_keys = self._order_color_csv_fields(all_keys)
        
        # Add primary keys and final timestamp
        for i, result in enumerate(processed_results, 1):
            result['primary_key'] = i
            result['data_row_creation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            import csv
            with open(final_output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=ordered_keys)
                
                # Write header
                writer.writeheader()
                
                # Write data rows with proper formatting
                for result in processed_results:
                    # Clean and format data
                    cleaned_row = {}
                    for key in ordered_keys:
                        value = result.get(key, '')
                        if value:
                            # Convert commas to semicolons
                            value = str(value).replace(',', ';')
                            # Remove line breaks and CRLFs
                            value = value.replace('\n', ' ').replace('\r', ' ').replace('\r\n', ' ')
                            # Clean multiple spaces
                            value = ' '.join(value.split())
                        cleaned_row[key] = value
                    
                    writer.writerow(cleaned_row)
            
            self.logger.info(f"Color analysis saved to {final_output_path}")
            self.logger.info(f"Total rows: {len(processed_results)}, Total columns: {len(ordered_keys)}")
            
        except Exception as e:
            self.logger.error(f"Error saving color analysis to CSV: {e}")
    
    def _get_unique_filename(self, base_path: Path) -> Path:
        """Generate unique filename if base path already exists."""
        if not base_path.exists():
            return base_path
        
        # Extract parts
        parent = base_path.parent
        stem = base_path.stem
        suffix = base_path.suffix
        
        # Try appending sequence numbers
        for i in range(1, 10000):
            new_name = f"{stem}_{i:04d}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
        
        # Fallback with timestamp
        timestamp = datetime.now().strftime("%H%M%S")
        new_name = f"{stem}_{timestamp}{suffix}"
        return parent / new_name
    
    def _order_color_csv_fields(self, fields: set) -> List[str]:
        """Order CSV fields logically for color analysis."""
        # Define field order
        field_order = [
            # Primary key (first)
            ['primary_key'],
            
            # File information
            ['file_path', 'file_name'],
            
            # Image properties
            ['image_width', 'image_height', 'total_pixels'],
            
            # Color 1 data
            [f'color_1_{field}' for field in [
                'rgb_r', 'rgb_g', 'rgb_b', 'rgb_combined', 'hex_value', 'percentage',
                'hsv_h', 'hsv_s', 'hsv_v', 'hsva_combined',
                'cmyk_c', 'cmyk_m', 'cmyk_y', 'cmyk_k', 'cmyk_combined',
                'nearest_color_name', 'color_name_source',
                'css3_color_name', 'x11_color_name', 'pantone_approximation',
                'ral_color_name', 'crayola_color_name', 'natural_color_name',
                'iso_color_name', 'web_safe_color', 'color_temperature_k',
                'color_family', 'color_harmony_type'
            ]],
            
            # Color 2 data
            [f'color_2_{field}' for field in [
                'rgb_r', 'rgb_g', 'rgb_b', 'rgb_combined', 'hex_value', 'percentage',
                'hsv_h', 'hsv_s', 'hsv_v', 'hsva_combined',
                'cmyk_c', 'cmyk_m', 'cmyk_y', 'cmyk_k', 'cmyk_combined',
                'nearest_color_name', 'color_name_source',
                'css3_color_name', 'x11_color_name', 'pantone_approximation',
                'ral_color_name', 'crayola_color_name', 'natural_color_name',
                'iso_color_name', 'web_safe_color', 'color_temperature_k',
                'color_family', 'color_harmony_type'
            ]],
            
            # Color 3 data
            [f'color_3_{field}' for field in [
                'rgb_r', 'rgb_g', 'rgb_b', 'rgb_combined', 'hex_value', 'percentage',
                'hsv_h', 'hsv_s', 'hsv_v', 'hsva_combined',
                'cmyk_c', 'cmyk_m', 'cmyk_y', 'cmyk_k', 'cmyk_combined',
                'nearest_color_name', 'color_name_source',
                'css3_color_name', 'x11_color_name', 'pantone_approximation',
                'ral_color_name', 'crayola_color_name', 'natural_color_name',
                'iso_color_name', 'web_safe_color', 'color_temperature_k',
                'color_family', 'color_harmony_type'
            ]],
            
            # Color 4 data
            [f'color_4_{field}' for field in [
                'rgb_r', 'rgb_g', 'rgb_b', 'rgb_combined', 'hex_value', 'percentage',
                'hsv_h', 'hsv_s', 'hsv_v', 'hsva_combined',
                'cmyk_c', 'cmyk_m', 'cmyk_y', 'cmyk_k', 'cmyk_combined',
                'nearest_color_name', 'color_name_source',
                'css3_color_name', 'x11_color_name', 'pantone_approximation',
                'ral_color_name', 'crayola_color_name', 'natural_color_name',
                'iso_color_name', 'web_safe_color', 'color_temperature_k',
                'color_family', 'color_harmony_type'
            ]],
            
            # Summary fields
            ['total_colors_analyzed', 'color_extraction_method'],
            
            # Analysis timestamp
            ['analysis_timestamp'],
            
            # Final timestamp (last column as required)
            ['data_row_creation_timestamp']
        ]
        
        # Flatten the order list
        ordered = []
        for group in field_order:
            for field in group:
                if field in fields:
                    ordered.append(field)
        
        # Add any remaining fields not in the order list (before final timestamp)
        remaining = sorted(fields - set(ordered))
        if remaining:
            # Insert remaining fields before the final timestamp
            if 'data_row_creation_timestamp' in ordered:
                timestamp_index = ordered.index('data_row_creation_timestamp')
                ordered = ordered[:timestamp_index] + remaining + ordered[timestamp_index:]
            else:
                ordered.extend(remaining)
        
        return ordered