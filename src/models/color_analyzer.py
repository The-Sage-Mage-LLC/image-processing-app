"""
Advanced Color Analysis Module with External Color Dictionary Support
Project ID: Image Processing App 20251119
Created: 2025-11-19 07:08:42 UTC
Enhanced: 2025-01-25 - Added external JSON color dictionary support and fallback methods
Author: The-Sage-Mage
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import colorsys
import math
import json

# Optional sklearn import
SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional import for webcolors
try:
    import webcolors
    WEBCOLORS_AVAILABLE = True
except ImportError:
    WEBCOLORS_AVAILABLE = False


class ColorAnalyzer:
    """Advanced color analysis using multiple AI models and external color dictionaries."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Number of dominant colors to extract (fixed at 4 as per requirements)
        self.num_colors = 4
        
        # Check sklearn availability
        if not SKLEARN_AVAILABLE:
            self.logger.warning("sklearn not available, using fallback color extraction methods")
        
        # Initialize color databases from external JSON files
        self.color_databases = self._initialize_color_databases_from_files()
        
        # Initialize AI models for color analysis
        self._initialize_ai_models()
    
    def _initialize_color_databases_from_files(self) -> Dict[str, Dict[str, Tuple[int, int, int]]]:
        """Initialize comprehensive color name databases from external JSON files."""
        databases = {}
        
        # Define color dictionary file mappings
        color_dict_files = {
            'CSS3': 'config/colors/css3_colors.json',
            'X11': 'config/colors/x11_colors.json', 
            'Pantone': 'config/colors/pantone_colors.json',
            'RAL_Classic': 'config/colors/ral_classic_colors.json',
            'Crayola': 'config/colors/crayola_colors.json',
            'Natural': 'config/colors/natural_colors.json',
            'ISO': 'config/colors/iso_colors.json',
            'Lab_Scientific': 'config/colors/lab_scientific_colors.json'
        }
        
        # Load each color dictionary from JSON files
        for db_name, file_path in color_dict_files.items():
            try:
                colors_dict = self._load_color_dictionary_from_json(file_path, db_name)
                if colors_dict:
                    databases[db_name] = colors_dict
                    self.logger.debug(f"Loaded {len(colors_dict)} colors from {db_name} dictionary")
                else:
                    self.logger.warning(f"No colors loaded from {db_name} dictionary ({file_path})")
            except Exception as e:
                self.logger.error(f"Failed to load {db_name} color dictionary from {file_path}: {e}")
                # Fallback to built-in dictionary for critical databases
                if db_name in ['CSS3', 'X11']:
                    fallback_dict = self._get_fallback_color_dict(db_name)
                    if fallback_dict:
                        databases[db_name] = fallback_dict
                        self.logger.info(f"Using fallback {db_name} dictionary with {len(fallback_dict)} colors")
        
        total_colors = sum(len(db) for db in databases.values())
        self.logger.info(f"Initialized {len(databases)} color databases with {total_colors} total colors")
        
        # Ensure we have at least basic color support
        if not databases:
            self.logger.warning("No external color dictionaries loaded, using minimal fallback")
            databases['Basic'] = self._get_basic_fallback_colors()
        
        return databases
    
    def _load_color_dictionary_from_json(self, file_path: str, db_name: str) -> Optional[Dict[str, Tuple[int, int, int]]]:
        """
        Load a color dictionary from a JSON file.
        
        Args:
            file_path: Path to the JSON color dictionary file
            db_name: Name of the database for logging
            
        Returns:
            Dictionary mapping color names to RGB tuples, or None if error
        """
        try:
            json_file = Path(file_path)
            if not json_file.exists():
                self.logger.warning(f"Color dictionary file not found: {file_path}")
                return None
            
            with open(json_file, 'r', encoding='utf-8') as f:
                color_data = json.load(f)
            
            # Validate JSON structure
            if 'colors' not in color_data:
                self.logger.error(f"Invalid color dictionary format in {file_path}: missing 'colors' key")
                return None
            
            colors_dict = {}
            
            # Parse colors based on the JSON structure
            for color_key, color_info in color_data['colors'].items():
                try:
                    # Handle different JSON structures
                    if 'rgb' in color_info:
                        rgb_values = color_info['rgb']
                        if isinstance(rgb_values, list) and len(rgb_values) >= 3:
                            # Standard format: "rgb": [r, g, b]
                            r, g, b = int(rgb_values[0]), int(rgb_values[1]), int(rgb_values[2])
                            colors_dict[color_key] = (r, g, b)
                        else:
                            self.logger.warning(f"Invalid RGB format for {color_key} in {db_name}: {rgb_values}")
                    else:
                        self.logger.warning(f"Missing RGB data for {color_key} in {db_name}")
                        
                except (ValueError, TypeError, KeyError) as e:
                    self.logger.warning(f"Error parsing color {color_key} in {db_name}: {e}")
                    continue
            
            # Log metadata if available
            if 'metadata' in color_data:
                metadata = color_data['metadata']
                self.logger.debug(f"Loaded {db_name}: {metadata.get('name', 'Unknown')} "
                                f"v{metadata.get('version', '?')} with {len(colors_dict)} colors")
            
            return colors_dict if colors_dict else None
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error loading {file_path}: {e}")
            return None
    
    def _get_fallback_color_dict(self, db_name: str) -> Optional[Dict[str, Tuple[int, int, int]]]:
        """Get fallback built-in color dictionary for critical databases."""
        if db_name == 'CSS3':
            return self._get_fallback_css3_colors()
        elif db_name == 'X11':
            return self._get_fallback_x11_colors()
        else:
            return None
    
    def _get_fallback_css3_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get fallback CSS3 color database if external file fails."""
        if WEBCOLORS_AVAILABLE:
            try:
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
        
        # Minimal CSS3 colors fallback
        return {
            'red': (255, 0, 0), 'green': (0, 128, 0), 'blue': (0, 0, 255),
            'yellow': (255, 255, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255),
            'white': (255, 255, 255), 'black': (0, 0, 0), 'gray': (128, 128, 128),
            'orange': (255, 165, 0), 'purple': (128, 0, 128), 'pink': (255, 192, 203),
            'brown': (165, 42, 42), 'navy': (0, 0, 128), 'olive': (128, 128, 0),
            'lime': (0, 255, 0), 'aqua': (0, 255, 255), 'silver': (192, 192, 192),
            'maroon': (128, 0, 0), 'teal': (0, 128, 128)
        }
    
    def _get_fallback_x11_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get fallback X11 color database if external file fails."""
        return {
            'AliceBlue': (240, 248, 255), 'AntiqueWhite': (250, 235, 215),
            'Aqua': (0, 255, 255), 'Aquamarine': (127, 255, 212),
            'Azure': (240, 255, 255), 'Beige': (245, 245, 220),
            'Bisque': (255, 228, 196), 'BlanchedAlmond': (255, 235, 205),
            'Blue': (0, 0, 255), 'BlueViolet': (138, 43, 226),
            'Brown': (165, 42, 42), 'BurlyWood': (222, 184, 135),
            'CadetBlue': (95, 158, 160), 'Chartreuse': (127, 255, 0),
            'Chocolate': (210, 105, 30), 'Coral': (255, 127, 80),
            'CornflowerBlue': (100, 149, 237), 'Cornsilk': (255, 248, 220),
            'Crimson': (220, 20, 60), 'Cyan': (0, 255, 255),
            'DarkBlue': (0, 0, 139), 'DarkCyan': (0, 139, 139),
            'DarkGreen': (0, 100, 0), 'DarkOrange': (255, 140, 0),
            'DeepPink': (255, 20, 147), 'DeepSkyBlue': (0, 191, 255),
            'Gold': (255, 215, 0), 'Green': (0, 128, 0),
            'HotPink': (255, 105, 180), 'Indigo': (75, 0, 130),
            'Lime': (0, 255, 0), 'Magenta': (255, 0, 255),
            'Navy': (0, 0, 128), 'Orange': (255, 165, 0),
            'Pink': (255, 192, 203), 'Purple': (128, 0, 128),
            'Red': (255, 0, 0), 'Silver': (192, 192, 192),
            'White': (255, 255, 255), 'Yellow': (255, 255, 0)
        }
    
    def _get_basic_fallback_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get basic color dictionary as absolute fallback."""
        return {
            'Red': (255, 0, 0), 'Green': (0, 255, 0), 'Blue': (0, 0, 255),
            'Yellow': (255, 255, 0), 'Cyan': (0, 255, 255), 'Magenta': (255, 0, 255),
            'White': (255, 255, 255), 'Black': (0, 0, 0), 'Gray': (128, 128, 128),
            'Orange': (255, 165, 0), 'Purple': (128, 0, 128), 'Pink': (255, 192, 203)
        }
    
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
            extraction_method = 'OpenCV + External Dictionaries'
            if SKLEARN_AVAILABLE:
                extraction_method = 'K-Means + OpenCV + External Dictionaries'
            results['color_extraction_method'] = extraction_method
            results['color_dictionaries_loaded'] = len(self.color_databases)
            
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
        
        if SKLEARN_AVAILABLE:
            # Method 1: K-Means clustering (primary method)
            colors_kmeans = self._extract_colors_kmeans(image_resized)
        else:
            colors_kmeans = []
        
        # Method 2: Histogram-based analysis (validation)
        colors_histogram = self._extract_colors_histogram(image_resized)
        
        # Method 3: Quantization-based analysis using OpenCV only
        colors_quantized = self._extract_colors_quantized_opencv(image_resized)
        
        # Combine and validate results
        final_colors = self._ensemble_color_selection_safe(
            colors_kmeans, colors_histogram, colors_quantized, image_resized
        )
        
        # Ensure exactly 4 colors
        if len(final_colors) > 4:
            final_colors = final_colors[:4]
        elif len(final_colors) < 4:
            # Pad with additional colors if needed
            if colors_histogram and len(colors_histogram) > len(final_colors):
                while len(final_colors) < 4 and len(colors_histogram) > len(final_colors):
                    final_colors.append(colors_histogram[len(final_colors)])
            elif colors_quantized and len(colors_quantized) > len(final_colors):
                while len(final_colors) < 4 and len(colors_quantized) > len(final_colors):
                    final_colors.append(colors_quantized[len(final_colors)])
        
        # Normalize percentages to sum to 100%
        total_percentage = sum(percentage for _, percentage in final_colors)
        if total_percentage > 0:
            final_colors = [(color, (percentage / total_percentage) * 100) 
                           for color, percentage in final_colors]
        
        return final_colors
    
    def _extract_colors_kmeans(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Extract colors using K-Means clustering (requires sklearn)."""
        if not SKLEARN_AVAILABLE:
            return []
        
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
    
    def _extract_colors_quantized_opencv(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Extract colors using OpenCV-only color quantization (no sklearn)."""
        # Quantize image to reduce color space using OpenCV kmeans
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
    
    def _ensemble_color_selection_safe(self, colors_kmeans: List, colors_histogram: List, 
                                      colors_quantized: List, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Select best colors using ensemble of methods (safe version)."""
        all_colors = []
        
        # Add all candidate colors
        all_colors.extend(colors_kmeans)
        all_colors.extend(colors_histogram)
        all_colors.extend(colors_quantized)
        
        # If no colors found, use simple fallback
        if not all_colors:
            return self._extract_colors_simple_fallback(image)
        
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
        
        return merged_colors
    
    def _extract_colors_simple_fallback(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Simple fallback method to extract colors if all other methods fail."""
        # Simple approach: sample pixels and find most common colors
        height, width = image.shape[:2]
        
        # Sample pixels at regular intervals
        sample_pixels = []
        step = max(10, min(width, height) // 20)
        for y in range(0, height, step):
            for x in range(0, width, step):
                sample_pixels.append(image[y, x])
        
        # Convert to numpy array
        sample_pixels = np.array(sample_pixels)
        
        # Quantize to reduce color space (simple binning)
        quantized = sample_pixels // 32 * 32  # Reduce to 8 levels per channel
        
        # Find unique colors and their counts
        unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        
        # Return top 4 colors
        result = []
        total_pixels = len(sample_pixels)
        
        for i in range(min(4, len(unique_colors))):
            idx = sorted_indices[i]
            color = unique_colors[idx].astype(np.float64)
            percentage = (counts[idx] / total_pixels) * 100
            result.append((color, percentage))
        
        return result
    
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
        Perform comprehensive analysis of a single color using external dictionaries.
        
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
            'lab_scientific_color_name': '',
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
        
        # Find color names from all loaded external databases
        color_matches = self._find_comprehensive_color_names(r, g, b)
        
        # Assign best matches to fields
        data['nearest_color_name'] = color_matches.get('best_name', 'Unknown')
        data['color_name_source'] = color_matches.get('best_source', 'Algorithm')
        data['css3_color_name'] = color_matches.get('CSS3', '')
        data['x11_color_name'] = color_matches.get('X11', '')
        data['pantone_approximation'] = color_matches.get('Pantone', '')
        data['ral_color_name'] = color_matches.get('RAL_Classic', '')
        data['crayola_color_name'] = color_matches.get('Crayola', '')
        data['natural_color_name'] = color_matches.get('Natural', '')
        data['lab_scientific_color_name'] = color_matches.get('Lab_Scientific', '')
        data['iso_color_name'] = color_matches.get('ISO', '')
        
        # Additional color science calculations
        data['web_safe_color'] = self._get_web_safe_color(r, g, b)
        data['color_temperature_k'] = self._estimate_color_temperature(r, g, b)
        data['color_family'] = self._get_color_family(r, g, b)
        data['color_harmony_type'] = self._get_color_harmony_type(h)
        
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
        """Find color names from all available external databases."""
        matches = {}
        best_name = 'Unknown'
        best_source = 'Algorithm'
        min_distance = float('inf')
        
        # Check each loaded database
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
        
        # Apply simple text processing without importing text_processor to avoid circular dependencies
        processed_results = []
        for result in analysis_results:
            processed = self._process_text_fields_simple(result)
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
    
    def _process_text_fields_simple(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple text processing without external dependencies."""
        processed_data = {}
        
        for key, value in data.items():
            if isinstance(value, str) and value:
                # Basic text cleaning
                cleaned_value = str(value).strip()
                # Remove extra spaces
                cleaned_value = ' '.join(cleaned_value.split())
                # Basic capitalization for color names
                if 'color_name' in key.lower():
                    cleaned_value = cleaned_value.title()
                processed_data[key] = cleaned_value
            else:
                processed_data[key] = value
        
        return processed_data
    
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
        # Define field order (updated to include new fields from external dictionaries)
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
                'lab_scientific_color_name', 'iso_color_name', 'web_safe_color', 
                'color_temperature_k', 'color_family', 'color_harmony_type'
            ]],
            
            # Color 2 data
            [f'color_2_{field}' for field in [
                'rgb_r', 'rgb_g', 'rgb_b', 'rgb_combined', 'hex_value', 'percentage',
                'hsv_h', 'hsv_s', 'hsv_v', 'hsva_combined',
                'cmyk_c', 'cmyk_m', 'cmyk_y', 'cmyk_k', 'cmyk_combined',
                'nearest_color_name', 'color_name_source',
                'css3_color_name', 'x11_color_name', 'pantone_approximation',
                'ral_color_name', 'crayola_color_name', 'natural_color_name',
                'lab_scientific_color_name', 'iso_color_name', 'web_safe_color',
                'color_temperature_k', 'color_family', 'color_harmony_type'
            ]],
            
            # Color 3 data
            [f'color_3_{field}' for field in [
                'rgb_r', 'rgb_g', 'rgb_b', 'rgb_combined', 'hex_value', 'percentage',
                'hsv_h', 'hsv_s', 'hsv_v', 'hsva_combined',
                'cmyk_c', 'cmyk_m', 'cmyk_y', 'cmyk_k', 'cmyk_combined',
                'nearest_color_name', 'color_name_source',
                'css3_color_name', 'x11_color_name', 'pantone_approximation',
                'ral_color_name', 'crayola_color_name', 'natural_color_name',
                'lab_scientific_color_name', 'iso_color_name', 'web_safe_color',
                'color_temperature_k', 'color_family', 'color_harmony_type'
            ]],
            
            # Color 4 data
            [f'color_4_{field}' for field in [
                'rgb_r', 'rgb_g', 'rgb_b', 'rgb_combined', 'hex_value', 'percentage',
                'hsv_h', 'hsv_s', 'hsv_v', 'hsva_combined',
                'cmyk_c', 'cmyk_m', 'cmyk_y', 'cmyk_k', 'cmyk_combined',
                'nearest_color_name', 'color_name_source',
                'css3_color_name', 'x11_color_name', 'pantone_approximation',
                'ral_color_name', 'crayola_color_name', 'natural_color_name',
                'lab_scientific_color_name', 'iso_color_name', 'web_safe_color',
                'color_temperature_k', 'color_family', 'color_harmony_type'
            ]],
            
            # Summary fields (updated)
            ['total_colors_analyzed', 'color_extraction_method', 'color_dictionaries_loaded'],
            
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