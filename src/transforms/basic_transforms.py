"""
Basic Image Transformations
Project ID: Image Processing App 20251119
Created: 2025-11-19 06:52:45 UTC
Author: The-Sage-Mage
"""

from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
from typing import Optional
import logging


class BasicTransforms:
    """Basic image transformation operations with thread safety."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Load configuration settings
        self.grayscale_config = config.get('grayscale', {})
        self.sepia_config = config.get('sepia', {})
        self.preserve_metadata = config.get('basic_transforms', {}).get('preserve_metadata', False)
        self.strip_sensitive = config.get('processing', {}).get('strip_sensitive_metadata', True)
        
        self.logger.debug("BasicTransforms initialized with thread safety")
    
    def convert_to_grayscale(self, image_path: Path) -> Optional[Image.Image]:
        """
        Convert image to grayscale using configured method with thread safety.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Processed PIL Image or None if error
        """
        try:
            # Open image with validation
            image = Image.open(image_path)
            if not image:
                self.logger.error(f"Failed to open image: {image_path}")
                return None
            
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Get conversion method
            method = self.grayscale_config.get('method', 'luminosity')
            
            if method == 'average':
                # Simple average method
                grayscale = image.convert('L')
            
            elif method == 'luminosity':
                # Luminosity method (weighted average)
                if image.mode == 'RGB':
                    # Use PIL's built-in conversion with proper weights
                    grayscale = image.convert('L')
                else:
                    grayscale = image
            
            elif method == 'desaturation':
                # Desaturation method (max + min) / 2
                if image.mode == 'RGB':
                    try:
                        # Thread-safe numpy operations
                        np_image = np.array(image)
                        if np_image.size == 0:
                            self.logger.error(f"Empty image array for: {image_path}")
                            return None
                            
                        max_rgb = np.max(np_image, axis=2)
                        min_rgb = np.min(np_image, axis=2)
                        grayscale_array = ((max_rgb + min_rgb) / 2).astype(np.uint8)
                        grayscale = Image.fromarray(grayscale_array, mode='L')
                    except Exception as e:
                        self.logger.warning(f"NumPy desaturation failed for {image_path}, falling back to luminosity: {e}")
                        grayscale = image.convert('L')
                else:
                    grayscale = image
            
            else:
                # Default to luminosity
                grayscale = image.convert('L')
            
            # Validate result
            if not grayscale:
                self.logger.error(f"Grayscale conversion failed for: {image_path}")
                return None
            
            # Handle metadata preservation
            if self.preserve_metadata and not self.strip_sensitive:
                try:
                    # Preserve EXIF data
                    if hasattr(image, 'info') and image.info:
                        grayscale.info = image.info.copy()
                    if hasattr(image, '_getexif'):
                        try:
                            exif = image._getexif()
                            if exif:
                                grayscale.info['exif'] = exif
                        except:
                            # Silently ignore EXIF errors
                            pass
                except Exception as e:
                    self.logger.warning(f"Failed to preserve metadata for {image_path}: {e}")
            
            return grayscale
            
        except Exception as e:
            self.logger.error(f"Error converting {image_path} to grayscale: {e}")
            return None
    
    def convert_to_sepia(self, image_path: Path) -> Optional[Image.Image]:
        """
        Convert image to sepia tone with thread safety.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Processed PIL Image or None if error
        """
        try:
            # Open image with validation
            image = Image.open(image_path)
            if not image:
                self.logger.error(f"Failed to open image: {image_path}")
                return None
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get sepia configuration
            intensity = self.sepia_config.get('intensity', 0.8)
            # Clamp intensity to valid range
            intensity = max(0.0, min(1.0, intensity))
            
            # Convert to numpy array with validation
            try:
                pixels = np.array(image, dtype=np.float32)
                if pixels.size == 0:
                    self.logger.error(f"Empty pixel array for: {image_path}")
                    return None
                
                # Apply sepia matrix transformation
                sepia_matrix = np.array([
                    [0.393, 0.769, 0.189],  # Red channel
                    [0.349, 0.686, 0.168],  # Green channel  
                    [0.272, 0.534, 0.131]   # Blue channel
                ], dtype=np.float32)
                
                # Apply transformation (thread-safe matrix multiplication)
                sepia_pixels = np.dot(pixels, sepia_matrix.T)
                
                # Blend with original based on intensity
                result_pixels = (1 - intensity) * pixels + intensity * sepia_pixels
                
                # Clip values to valid range and convert to uint8
                result_pixels = np.clip(result_pixels, 0, 255).astype(np.uint8)
                
                # Convert back to PIL Image
                sepia_image = Image.fromarray(result_pixels, mode='RGB')
                
            except Exception as numpy_error:
                self.logger.error(f"NumPy sepia processing failed for {image_path}: {numpy_error}")
                # Fallback to simple sepia using PIL
                try:
                    # Simple sepia using ImageOps
                    sepia_image = ImageOps.colorize(
                        image.convert('L'),
                        black='#704214',
                        white='#C8B99C'
                    )
                except Exception as fallback_error:
                    self.logger.error(f"Fallback sepia failed for {image_path}: {fallback_error}")
                    return None
            
            # Validate result
            if not sepia_image:
                self.logger.error(f"Sepia conversion failed for: {image_path}")
                return None
            
            # Handle metadata preservation
            if self.preserve_metadata and not self.strip_sensitive:
                try:
                    if hasattr(image, 'info') and image.info:
                        sepia_image.info = image.info.copy()
                except Exception as e:
                    self.logger.warning(f"Failed to preserve metadata for {image_path}: {e}")
            
            return sepia_image
            
        except Exception as e:
            self.logger.error(f"Error converting {image_path} to sepia: {e}")
            return None
    
    def apply_watermark(self, image: Image.Image, watermark_text: str) -> Image.Image:
        """
        Apply watermark to image with thread safety.
        
        Args:
            image: PIL Image to watermark
            watermark_text: Text to use as watermark
            
        Returns:
            Watermarked image
        """
        try:
            # Validate input
            if not image:
                self.logger.error("Invalid image provided for watermarking")
                return image
            
            # Get watermark configuration
            watermark_config = self.config.get('watermark', {})
            
            if not watermark_config.get('enabled', False):
                return image
            
            # Create a copy to avoid modifying original
            watermarked = image.copy()
            
            try:
                from PIL import ImageDraw, ImageFont
                
                # Create drawing context
                draw = ImageDraw.Draw(watermarked)
                
                # Try to load a font
                font_size = watermark_config.get('font_size', 20)
                font = None
                try:
                    # Try to use a TrueType font
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    try:
                        # Try alternative system fonts
                        for font_name in ["calibri.ttf", "tahoma.ttf", "segoeui.ttf"]:
                            try:
                                font = ImageFont.truetype(font_name, font_size)
                                break
                            except:
                                continue
                    except:
                        pass
                
                if not font:
                    # Fall back to default font
                    font = ImageFont.load_default()
                
                # Get text dimensions
                text = watermark_text or watermark_config.get('text', '© 2025')
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except:
                    # Fallback for older PIL versions
                    text_width, text_height = draw.textsize(text, font=font)
                
                # Calculate position
                position = watermark_config.get('position', 'bottom-right')
                margin = 10
                
                if position == 'top-left':
                    x, y = margin, margin
                elif position == 'top-right':
                    x, y = watermarked.width - text_width - margin, margin
                elif position == 'bottom-left':
                    x, y = margin, watermarked.height - text_height - margin
                elif position == 'bottom-right':
                    x, y = watermarked.width - text_width - margin, watermarked.height - text_height - margin
                elif position == 'center':
                    x, y = (watermarked.width - text_width) // 2, (watermarked.height - text_height) // 2
                else:
                    x, y = watermarked.width - text_width - margin, watermarked.height - text_height - margin
                
                # Get text color
                font_color = tuple(watermark_config.get('font_color', [255, 255, 255]))
                opacity = int(watermark_config.get('opacity', 0.5) * 255)
                
                # Add alpha channel for transparency
                if len(font_color) == 3:
                    font_color = (*font_color, opacity)
                
                # Draw watermark
                draw.text((x, y), text, font=font, fill=font_color)
                
            except ImportError:
                self.logger.warning("PIL ImageDraw/ImageFont not available for watermarking")
                return image
            except Exception as draw_error:
                self.logger.error(f"Watermark drawing failed: {draw_error}")
                return image
            
            return watermarked
            
        except Exception as e:
            self.logger.error(f"Error applying watermark: {e}")
            return image