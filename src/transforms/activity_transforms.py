"""
Activity Book Transformations Module
Project ID: Image Processing App 20251119
Created: 2025-11-19 07:15:45 UTC
Author: The-Sage-Mage
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import logging
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from scipy.spatial import distance
from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage.measure import label, regionprops
import random


class ActivityTransforms:
    """Complex transformations for activity books."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Load configuration settings
        self.dots_config = config.get('connect_the_dots', {})
        self.color_numbers_config = config.get('color_by_numbers', {})
        
        # Connect-the-dots settings
        self.max_dots = self.dots_config.get('max_dots_per_image', 200)
        self.min_dots = self.dots_config.get('min_dots_per_image', 20)
        self.min_dot_distance = self.dots_config.get('min_distance_between_dots', 10)
        self.max_dot_distance = self.dots_config.get('max_distance_between_dots', 50)
        self.dot_size = self.dots_config.get('dot_size', 5)
        self.number_font_size = self.dots_config.get('number_font_size', 10)
        self.edge_sensitivity = self.dots_config.get('edge_detection_sensitivity', 0.7)
        
        # Color-by-numbers settings
        self.max_colors = self.color_numbers_config.get('max_distinct_colors', 20)
        self.min_colors = self.color_numbers_config.get('min_distinct_colors', 5)
        self.min_area_size = self.color_numbers_config.get('min_area_size', 100)
        self.max_area_size = self.color_numbers_config.get('max_area_size', 10000)
        self.smoothing_kernel = self.color_numbers_config.get('smoothing_kernel_size', 5)
        self.color_similarity_threshold = self.color_numbers_config.get('color_similarity_threshold', 30)
        self.cbn_font_size = self.color_numbers_config.get('number_font_size', 12)
        self.border_thickness = self.color_numbers_config.get('border_thickness', 1)
    
    def convert_to_connect_dots(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Convert image to connect-the-dots activity with improved parameters.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Processed image as numpy array or None if error/unsuitable
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to read image: {image_path}")
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Check image complexity before processing
            height, width = gray.shape
            image_area = height * width
            
            # Adjusted parameters for fewer, larger dots
            self.max_dots = min(50, self.dots_config.get('max_dots_per_image', 50))  # Reduced from 200
            self.min_dots = max(8, self.dots_config.get('min_dots_per_image', 8))   # Increased from 20
            self.dot_size = max(8, self.dots_config.get('dot_size', 8))             # Increased from 5
            self.min_dot_distance = max(25, self.dots_config.get('min_distance_between_dots', 25))  # Increased from 10
            
            # Early rejection for overly complex images
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / image_area
            
            if edge_density > 0.2:  # Too complex
                self.logger.warning(f"Image too complex for connect-the-dots: {image_path.name} (edge density: {edge_density:.3f})")
                return None
            
            if image_area > 1000000:  # Image too large
                self.logger.warning(f"Image too large for connect-the-dots: {image_path.name} ({width}x{height})")
                return None
            
            # Step 1: Enhanced edge detection with simplification
            edges = self._detect_edges_for_dots(gray)
            
            # Step 2: Find key points along edges
            key_points = self._find_key_points(edges, gray.shape)
            
            # Early check for too many potential points
            if len(key_points) > self.max_dots * 3:
                self.logger.warning(f"Too many potential dots for connect-the-dots: {image_path.name} ({len(key_points)} points)")
                return None
            
            # Step 3: Filter and optimize points more aggressively
            optimized_points = self._optimize_dot_placement(key_points)
            
            # Final validation
            if len(optimized_points) < self.min_dots:
                self.logger.warning(f"Too few valid dots for connect-the-dots: {image_path.name} ({len(optimized_points)} dots)")
                return None
            
            if len(optimized_points) > self.max_dots:
                self.logger.warning(f"Still too many dots after optimization: {image_path.name} ({len(optimized_points)} dots)")
                return None
            
            # Step 4: Order points for logical connection
            ordered_points = self._order_points_for_connection(optimized_points)
            
            # Step 5: Create the connect-the-dots image
            result = self._create_connect_dots_image(gray.shape, ordered_points)
            
            self.logger.debug(f"Created connect-the-dots with {len(ordered_points)} dots for {image_path.name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating connect-the-dots for {image_path}: {e}")
            return None
    
    def _detect_edges_for_dots(self, gray: np.ndarray) -> np.ndarray:
        """Detect edges suitable for dot placement with better filtering."""
        # Apply stronger Gaussian blur to reduce noise and complexity
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Use single-scale edge detection with higher thresholds
        edges = cv2.Canny(blurred, 80, 160)
        
        # Apply morphological operations to reduce small details
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Remove small edge fragments
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_edges = np.zeros_like(edges)
        
        min_contour_length = 50  # Increased minimum contour length
        for contour in contours:
            if cv2.arcLength(contour, False) > min_contour_length:
                cv2.drawContours(filtered_edges, [contour], -1, 255, 2)
        
        return filtered_edges
    
    def _find_key_points(self, edges: np.ndarray, shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find key points along edges with better spacing control."""
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        key_points = []
        h, w = shape
        
        # Process only the largest contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        max_contours = min(3, len(contours))  # Process only top 3 contours
        
        for contour in contours[:max_contours]:
            # Skip very small contours
            if cv2.contourArea(contour) < 500:  # Increased minimum area
                continue
            
            # Use larger approximation epsilon to reduce points
            epsilon = 0.05 * cv2.arcLength(contour, True)  # Increased from 0.02
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Add corner points (reduced)
            for i, point in enumerate(approx[::2]):  # Take every other point
                x, y = point[0]
                # Ensure points are not too close to edges
                if 20 < x < w-20 and 20 < y < h-20:
                    key_points.append((x, y))
            
            # Add fewer points along the contour
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # Much fewer points based on perimeter
                num_points = max(2, min(int(perimeter / 100), 8))  # Reduced significantly
                
                for i in range(num_points):
                    t = i / num_points
                    idx = int(t * (len(contour) - 1))
                    x, y = contour[idx][0]
                    if 20 < x < w-20 and 20 < y < h-20:
                        key_points.append((x, y))
        
        # Add fewer boundary points
        margin = 30
        boundary_points = [
            (margin, margin),
            (w - margin, margin),
            (w - margin, h - margin),
            (margin, h - margin)
        ]
        key_points.extend(boundary_points)
        
        return key_points
    
    def _optimize_dot_placement(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Optimize dot placement with stricter distance requirements."""
        if len(points) <= self.min_dots:
            return points[:self.max_dots]
        
        # Convert to numpy array
        points_array = np.array(points)
        
        # Remove exact duplicates
        unique_points = np.unique(points_array, axis=0)
        
        # If we have too many points, use spatial clustering to reduce
        if len(unique_points) > self.max_dots:
            from sklearn.cluster import KMeans
            n_clusters = min(self.max_dots, max(self.min_dots, len(unique_points) // 4))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(unique_points)
            centers = kmeans.cluster_centers_.astype(int)
            unique_points = centers
        
        # Enforce minimum distance with stricter requirements
        optimized = []
        for point in unique_points:
            if not optimized:
                optimized.append(tuple(point))
            else:
                # Check distance to all existing points
                distances = [np.sqrt((point[0] - opt[0])**2 + (point[1] - opt[1])**2) 
                           for opt in optimized]
                if min(distances) >= self.min_dot_distance:
                    optimized.append(tuple(point))
        
        # Ensure we have enough points but not too many
        if len(optimized) < self.min_dots:
            # If we don't have enough after distance filtering, relax distance requirement
            relaxed_distance = self.min_dot_distance * 0.7
            optimized = [tuple(unique_points[0])]  # Start over with first point
            
            for point in unique_points[1:]:
                distances = [np.sqrt((point[0] - opt[0])**2 + (point[1] - opt[1])**2) 
                           for opt in optimized]
                if min(distances) >= relaxed_distance:
                    optimized.append(tuple(point))
                
                if len(optimized) >= self.max_dots:
                    break
        
        return optimized[:self.max_dots]
    
    def _create_connect_dots_image(self, shape: Tuple[int, int], 
                                   points: List[Tuple[int, int]]) -> np.ndarray:
        """Create the final connect-the-dots image with larger dots and numbers."""
        # Create white background
        h, w = shape
        result = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(result)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a larger font
        try:
            font = ImageFont.truetype("arial.ttf", max(14, self.number_font_size))
        except:
            font = ImageFont.load_default()
        
        # Draw larger dots and numbers
        for i, (x, y) in enumerate(points, 1):
            # Draw larger dot
            cv2.circle(result, (x, y), self.dot_size, (0, 0, 0), -1)
            
            # Draw number with better positioning
            text = str(i)
            text_offset = self.dot_size + 8  # Increased offset
            
            # Smart positioning to avoid edge overflow
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position text to avoid overlapping with dots
            if x + text_offset + text_width < w and y - text_offset - text_height > 0:
                text_x = x + text_offset
                text_y = y - text_offset - text_height
            elif x - text_offset - text_width > 0 and y + text_offset + text_height < h:
                text_x = x - text_offset - text_width
                text_y = y + text_offset
            else:
                text_x = min(max(x - text_width // 2, 0), w - text_width)
                text_y = min(max(y - text_height // 2, 0), h - text_height)
            
            # Draw white background for number for better visibility
            padding = 2
            draw.rectangle([
                text_x - padding, 
                text_y - padding,
                text_x + text_width + padding,
                text_y + text_height + padding
            ], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
            
            draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
        
        # Convert back to numpy array
        result = np.array(pil_image)
        
        # Don't draw guide lines for cleaner appearance
        
        return result
    
    def convert_to_color_by_numbers(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Convert image to color-by-numbers activity with improved color management.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Processed image as numpy array or None if error/unsuitable
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to read image: {image_path}")
                return None
            
            # Check image complexity before processing
            height, width = image.shape[:2]
            image_area = height * width
            
            # Adjust parameters for fewer colors
            self.max_colors = min(8, self.color_numbers_config.get('max_distinct_colors', 8))  # Reduced from 20
            self.min_colors = max(3, self.color_numbers_config.get('min_distinct_colors', 3))  # Reduced from 5
            self.min_area_size = max(500, self.color_numbers_config.get('min_area_size', 500))  # Increased from 100
            
            # Early rejection for overly complex images
            if image_area > 800000:  # Image too large
                self.logger.warning(f"Image too large for color-by-numbers: {image_path.name} ({width}x{height})")
                return None
            
            # Step 1: Reduce colors using k-means with stricter limits
            reduced_colors, color_map, color_names = self._reduce_colors_enhanced(image)
            
            # Validate color count
            if len(color_map) < self.min_colors:
                self.logger.warning(f"Too few distinct colors for color-by-numbers: {image_path.name} ({len(color_map)} colors)")
                return None
            
            if len(color_map) > self.max_colors:
                self.logger.warning(f"Too many colors after reduction for color-by-numbers: {image_path.name} ({len(color_map)} colors)")
                return None
            
            # Step 2: Create regions using watershed segmentation
            regions = self._segment_regions(reduced_colors)
            
            # Step 3: Merge small regions more aggressively
            merged_regions = self._merge_small_regions_enhanced(regions, reduced_colors)
            
            # Step 4: Assign numbers to regions with color names
            numbered_image, color_legend = self._create_numbered_image_enhanced(
                merged_regions, reduced_colors, color_map, color_names
            )
            
            if numbered_image is None:
                return None
            
            # Step 5: Add enhanced color legend with names
            result = self._add_enhanced_color_legend(numbered_image, color_legend)
            
            self.logger.debug(f"Created color-by-numbers with {len(color_legend)} colors for {image_path.name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating color-by-numbers for {image_path}: {e}")
            return None
    
    def _reduce_colors_enhanced(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int]], Dict[int, str]]:
        """Reduce image colors using k-means clustering with color naming."""
        # Resize for faster processing and reduce detail
        h, w = image.shape[:2]
        if w > 400 or h > 400:  # More aggressive resizing
            scale = 400 / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            small_image = cv2.resize(image, (new_w, new_h))
        else:
            small_image = image.copy()
        
        # Apply stronger bilateral filter to smooth while preserving edges
        smoothed = cv2.bilateralFilter(small_image, 15, 100, 100)  # Stronger smoothing
        
        # Additional blur to reduce color variations
        smoothed = cv2.GaussianBlur(smoothed, (5, 5), 1)
        
        # Reshape for k-means
        pixels = smoothed.reshape(-1, 3)
        
        # More aggressive color reduction
        n_colors = min(self.max_colors, max(self.min_colors, len(np.unique(pixels, axis=0)) // 500))  # Reduced divisor
        
        # Apply k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (palette colors)
        palette = kmeans.cluster_centers_.astype(np.uint8)
        
        # Create color map and names
        color_map = {}
        color_names = {}
        
        for i, color in enumerate(palette):
            color_rgb = tuple(color.tolist())
            color_map[i] = color_rgb
            color_names[i] = self._get_color_name(color_rgb)
        
        # Replace pixels with cluster centers
        labels = kmeans.labels_
        quantized = palette[labels].reshape(small_image.shape)
        
        # Resize back to original size
        if w != small_image.shape[1] or h != small_image.shape[0]:
            quantized = cv2.resize(quantized, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return quantized, color_map, color_names
    
    def _get_color_name(self, rgb_color: Tuple[int, int, int]) -> str:
        """Get a human-readable name for an RGB color."""
        r, g, b = rgb_color
        
        # Define color ranges with names
        color_definitions = [
            # Format: (name, (r_min, r_max), (g_min, g_max), (b_min, b_max))
            ("White", (200, 255), (200, 255), (200, 255)),
            ("Black", (0, 50), (0, 50), (0, 50)),
            ("Red", (150, 255), (0, 100), (0, 100)),
            ("Green", (0, 100), (150, 255), (0, 100)),
            ("Blue", (0, 100), (0, 100), (150, 255)),
            ("Yellow", (200, 255), (200, 255), (0, 150)),
            ("Orange", (200, 255), (100, 200), (0, 100)),
            ("Purple", (100, 200), (0, 100), (150, 255)),
            ("Pink", (200, 255), (150, 220), (150, 220)),
            ("Brown", (80, 150), (40, 100), (20, 80)),
            ("Gray", (80, 180), (80, 180), (80, 180)),
            ("Light Blue", (100, 200), (150, 255), (200, 255)),
            ("Dark Green", (0, 80), (80, 150), (0, 80)),
            ("Beige", (200, 255), (180, 230), (150, 200)),
            ("Maroon", (80, 150), (0, 50), (0, 50)),
            ("Navy", (0, 50), (0, 50), (80, 150)),
        ]
        
        # Find best matching color name
        for name, (r_min, r_max), (g_min, g_max), (b_min, b_max) in color_definitions:
            if r_min <= r <= r_max and g_min <= g <= g_max and b_min <= b <= b_max:
                return name
        
        # If no exact match, classify by dominant channel
        max_val = max(r, g, b)
        if max_val == r and r > g + 30 and r > b + 30:
            return "Red Tone"
        elif max_val == g and g > r + 30 and g > b + 30:
            return "Green Tone"
        elif max_val == b and b > r + 30 and b > g + 30:
            return "Blue Tone"
        elif abs(r - g) < 30 and abs(r - b) < 30:
            if max_val > 150:
                return "Light Gray"
            else:
                return "Dark Gray"
        else:
            return "Mixed Color"
    
    def _merge_small_regions_enhanced(self, regions: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Merge small regions more aggressively."""
        try:
            from skimage.measure import label, regionprops
        except ImportError:
            # Fallback to OpenCV-based region analysis
            return self._merge_small_regions_opencv(regions, image)
        
        # Get region properties
        labeled = label(regions)
        props = regionprops(labeled)
        
        # More aggressive merging
        merge_map = {}
        
        for prop in props:
            if prop.area < self.min_area_size:  # Using increased minimum area
                # Find neighboring region to merge with
                y, x = prop.centroid
                y, x = int(y), int(x)
                
                # Look for nearby regions with larger search window
                window_size = 20  # Increased from 10
                y_min = max(0, y - window_size)
                y_max = min(regions.shape[0], y + window_size)
                x_min = max(0, x - window_size)
                x_max = min(regions.shape[1], x + window_size)
                
                neighbor_regions = regions[y_min:y_max, x_min:x_max]
                unique_neighbors = np.unique(neighbor_regions)
                unique_neighbors = unique_neighbors[unique_neighbors != prop.label]
                unique_neighbors = unique_neighbors[unique_neighbors != 0]
                
                if len(unique_neighbors) > 0:
                    # Merge with the most common neighbor
                    neighbor_counts = [(n, np.sum(neighbor_regions == n)) 
                                     for n in unique_neighbors]
                    neighbor_counts.sort(key=lambda x: x[1], reverse=True)
                    merge_map[prop.label] = neighbor_counts[0][0]
        
        # Apply merging
        merged = regions.copy()
        for old_label, new_label in merge_map.items():
            merged[merged == old_label] = new_label
        
        # Renumber regions consecutively
        unique_labels = np.unique(merged)
        unique_labels = unique_labels[unique_labels != 0]
        
        renumbered = np.zeros_like(merged)
        for i, label_val in enumerate(unique_labels, 1):
            renumbered[merged == label_val] = i
        
        return renumbered
    
    def _merge_small_regions_opencv(self, regions: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Fallback region merging using OpenCV only."""
        # Simple region merging based on size
        unique_labels = np.unique(regions)
        unique_labels = unique_labels[unique_labels != 0]
        
        merged = regions.copy()
        
        for label in unique_labels:
            region_mask = regions == label
            area = np.sum(region_mask)
            
            if area < self.min_area_size:
                # Find neighbors
                dilated = cv2.dilate(region_mask.astype(np.uint8), np.ones((5, 5)), iterations=1)
                neighbors = merged[dilated > 0]
                neighbors = neighbors[neighbors != label]
                neighbors = neighbors[neighbors != 0]
                
                if len(neighbors) > 0:
                    # Merge with most common neighbor
                    unique_neighbors, counts = np.unique(neighbors, return_counts=True)
                    best_neighbor = unique_neighbors[np.argmax(counts)]
                    merged[merged == label] = best_neighbor
        
        return merged
    
    def _create_numbered_image_enhanced(self, regions: np.ndarray, image: np.ndarray, 
                                      color_map: Dict[int, Tuple[int, int, int]],
                                      color_names: Dict[int, str]) -> Tuple[Optional[np.ndarray], Dict[int, Dict[str, Any]]]:
        """Create the numbered image with regions and enhanced color information."""
        # Create white background
        h, w = regions.shape
        result = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Get unique regions
        unique_regions = np.unique(regions)
        unique_regions = unique_regions[unique_regions != 0]
        
        # Validate region count
        if len(unique_regions) > self.max_colors:
            self.logger.warning(f"Too many regions after processing: {len(unique_regions)}")
            return None, {}
        
        if len(unique_regions) < self.min_colors:
            self.logger.warning(f"Too few regions after processing: {len(unique_regions)}")
            return None, {}
        
        # Create enhanced color legend mapping
        color_legend = {}
        
        # Map each region to a color number with enhanced information
        for i, region_id in enumerate(unique_regions, 1):
            # Find the most common color in this region
            region_mask = regions == region_id
            region_pixels = image[region_mask]
            
            if len(region_pixels) > 0:
                # Find most common color
                unique_colors, counts = np.unique(
                    region_pixels.reshape(-1, 3), 
                    axis=0, 
                    return_counts=True
                )
                dominant_color = unique_colors[np.argmax(counts)]
                dominant_color_tuple = tuple(dominant_color.tolist())
                
                # Find matching color in our palette
                best_match_idx = 0
                min_distance = float('inf')
                for idx, palette_color in color_map.items():
                    distance = sum((a - b) ** 2 for a, b in zip(dominant_color_tuple, palette_color))
                    if distance < min_distance:
                        min_distance = distance
                        best_match_idx = idx
                
                color_legend[i] = {
                    'rgb_color': color_map[best_match_idx],
                    'color_name': color_names[best_match_idx],
                    'hex_color': '#{:02x}{:02x}{:02x}'.format(*color_map[best_match_idx])
                }
        
        # Draw region boundaries with thicker lines
        boundaries = cv2.Canny(regions.astype(np.uint8), 1, 1)
        boundary_kernel = np.ones((3, 3), np.uint8)
        boundaries = cv2.dilate(boundaries, boundary_kernel, iterations=1)
        result[boundaries > 0] = [0, 0, 0]
        
        # Convert to PIL for text rendering
        pil_image = Image.fromarray(result)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a larger font
        try:
            font = ImageFont.truetype("arial.ttf", max(16, self.cbn_font_size))
        except:
            font = ImageFont.load_default()
        
        # Add numbers to regions with better visibility
        for i, region_id in enumerate(unique_regions, 1):
            # Find centroid of region
            region_mask = regions == region_id
            props = regionprops(region_mask.astype(int))
            
            if props:
                centroid = props[0].centroid
                cy, cx = int(centroid[0]), int(centroid[1])
                
                # Draw number at centroid
                text = str(i)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Center the text
                text_x = cx - text_width // 2
                text_y = cy - text_height // 2
                
                # Draw white background with black border for better visibility
                padding = 4
                draw.rectangle([
                    text_x - padding, 
                    text_y - padding,
                    text_x + text_width + padding,
                    text_y + text_height + padding
                ], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
                
                # Draw text in black
                draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
        
        # Convert back to numpy array
        result = np.array(pil_image)
        
        return result
    
    def _add_enhanced_color_legend(self, image: np.ndarray, 
                                  color_legend: Dict[int, Dict[str, Any]]) -> np.ndarray:
        """Add an enhanced color legend with color names to the side of the image."""
        h, w = image.shape[:2]
        
        # Calculate legend dimensions
        legend_width = 200  # Increased from 150
        color_box_size = 25  # Increased from 20
        text_height = 35    # Increased from 25
        legend_height = max(h, len(color_legend) * text_height + 80)
        
        # Create extended canvas
        extended_width = w + legend_width
        extended_height = max(h, legend_height)
        result = np.ones((extended_height, extended_width, 3), dtype=np.uint8) * 255
        
        # Copy original image
        result[:h, :w] = image
        
        # Draw vertical separator
        cv2.line(result, (w, 0), (w, extended_height), (0, 0, 0), 3)
        
        # Convert to PIL for text rendering
        pil_image = Image.fromarray(result)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 16)
            name_font = ImageFont.truetype("arial.ttf", 12)
            number_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            name_font = title_font
            number_font = title_font
        
        # Add title
        draw.text((w + 15, 15), "Color Guide", fill=(0, 0, 0), font=title_font)
        
        # Add color entries with names
        y_offset = 50
        for number, color_info in sorted(color_legend.items()):
            color = color_info['rgb_color']
            color_name = color_info['color_name']
            
            # Draw color box
            box_x = w + 15
            box_y = y_offset
            
            # Draw black border
            draw.rectangle([
                box_x, box_y,
                box_x + color_box_size, box_y + color_box_size
            ], outline=(0, 0, 0), width=2)
            
            # Fill with color (convert RGB to BGR for OpenCV)
            bgr_color = (color[2], color[1], color[0])
            cv2.rectangle(result, 
                        (box_x + 2, box_y + 2),
                        (box_x + color_box_size - 2, box_y + color_box_size - 2),
                        bgr_color, -1)
            
            # Add number text
            number_text = f"{number}"
            draw.text((box_x + color_box_size + 10, box_y), 
                     number_text, fill=(0, 0, 0), font=number_font)
            
            # Add color name text
            name_text = color_name
            draw.text((box_x + color_box_size + 10, box_y + 15), 
                     name_text, fill=(0, 0, 0), font=name_font)
            
            y_offset += text_height
        
        # Convert back to numpy array
        result = np.array(pil_image)
        
        return result
    
    def _segment_regions(self, image: np.ndarray) -> np.ndarray:
        """Segment image into regions using watershed algorithm."""
        try:
            from skimage.segmentation import watershed
            from skimage.filters import sobel
            from skimage.measure import label
        except ImportError:
            # Fallback to simpler OpenCV-based segmentation
            return self._segment_regions_opencv(image)
        
        # Convert to grayscale for segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        gradient = sobel(gray)
        
        # Apply threshold to create markers
        _, markers = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(markers, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening.astype(np.uint8), cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg.astype(np.uint8), sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Create regions mask
        regions = markers.copy()
        regions[regions == -1] = 0  # Boundaries
        
        return regions
    
    def _segment_regions_opencv(self, image: np.ndarray) -> np.ndarray:
        """Fallback segmentation using OpenCV only."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        return markers
    
    def _order_points_for_connection(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Order points for logical connection in connect-the-dots activity.
        Uses a simple nearest-neighbor approach starting from top-left.
        
        Args:
            points: List of (x, y) coordinate tuples
            
        Returns:
            Ordered list of points for logical connection
        """
        if len(points) <= 1:
            return points
        
        # Convert to numpy array for easier computation
        points_array = np.array(points)
        
        # Start from the top-leftmost point
        distances_to_origin = np.sum(points_array, axis=1)
        start_idx = np.argmin(distances_to_origin)
        
        ordered_points = [points[start_idx]]
        remaining_points = [p for i, p in enumerate(points) if i != start_idx]
        
        # Use nearest neighbor to order remaining points
        current_point = points[start_idx]
        
        while remaining_points:
            # Find the nearest remaining point
            distances = []
            for point in remaining_points:
                dist = np.sqrt((current_point[0] - point[0])**2 + (current_point[1] - point[1])**2)
                distances.append(dist)
            
            nearest_idx = np.argmin(distances)
            nearest_point = remaining_points[nearest_idx]
            
            ordered_points.append(nearest_point)
            current_point = nearest_point
            remaining_points.pop(nearest_idx)
       