"""
Text Processing and Grammar Enhancement Utilities
Project ID: Image Processing App 20251119
Created: 2025-01-19
Author: The-Sage-Mage
"""

import re
from typing import Dict, Any, List, Optional


class TextProcessor:
    """Handles text processing, grammar checking, and formatting for all output data."""
    
    def __init__(self):
        # Common grammar fixes
        self.grammar_fixes = {
            # Double words
            r'\b(\w+)\s+\1\b': r'\1',
            
            # Spacing issues
            r'\s+': ' ',  # Multiple spaces to single space
            r'\s+([.!?])': r'\1',  # Space before punctuation
            r'([.!?])([A-Z])': r'\1 \2',  # Missing space after sentence end
            
            # Common typos
            r'\bthe\s+the\b': 'the',
            r'\band\s+and\b': 'and',
            r'\bwith\s+with\b': 'with',
            r'\bin\s+in\b': 'in',
            r'\bof\s+of\b': 'of',
            r'\bto\s+to\b': 'to',
            
            # Punctuation fixes
            r'\.+': '.',  # Multiple periods
            r',+': ',',   # Multiple commas
            r'\s*,\s*': ', ',  # Comma spacing
            r'\s*;\s*': '; ',  # Semicolon spacing
            
            # Capitalization issues
            r'\bi\b': 'I',  # Lowercase i as pronoun
        }
        
        # Articles and prepositions (should not be capitalized in titles)
        self.lowercase_words = {
            'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'if', 'in', 
            'nor', 'of', 'on', 'or', 'so', 'the', 'to', 'up', 'yet', 'is',
            'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'with', 'into', 'onto', 'from', 'about'
        }
        
        # Words that should always be capitalized
        self.always_capitalize = {
            'i', 'rgb', 'cmyk', 'hsv', 'gps', 'exif', 'jpeg', 'png', 'gif', 'bmp',
            'tiff', 'raw', 'dng', 'cr2', 'nef', 'arw', 'orf', 'pef', 'rw2',
            'dxo', 'iso', 'cpu', 'gpu', 'ai', 'ml', 'cv', 'api', 'url', 'http',
            'https', 'ftp', 'smtp', 'tcp', 'udp', 'ip', 'dns', 'sql', 'html',
            'css', 'js', 'xml', 'json', 'csv', 'pdf', 'doc', 'docx', 'xls',
            'xlsx', 'ppt', 'pptx', 'mp3', 'mp4', 'avi', 'mov', 'wmv', 'flv'
        }
    
    def process_all_text_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process all text fields in a data dictionary."""
        processed_data = {}
        
        for key, value in data.items():
            if isinstance(value, str) and value and value != 'Not available':
                # Determine field type and apply appropriate processing
                if self._is_title_field(key):
                    processed_data[key] = self.format_as_title(value)
                elif self._is_sentence_field(key):
                    processed_data[key] = self.format_as_sentence(value)
                elif self._is_keyword_field(key):
                    processed_data[key] = self.format_as_keywords(value)
                elif self._is_technical_field(key):
                    processed_data[key] = self.format_as_technical(value)
                else:
                    processed_data[key] = self.clean_text(value)
            else:
                processed_data[key] = value
        
        return processed_data
    
    def format_as_sentence(self, text: str) -> str:
        """Format text as a proper English sentence."""
        if not text:
            return text
        
        # Clean the text first
        text = self.clean_text(text)
        
        # Apply grammar fixes
        text = self.apply_grammar_fixes(text)
        
        # Ensure proper sentence structure
        text = text.strip()
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Ensure proper ending punctuation
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = re.sub(r'\s+([.!?])', r'\1', text)  # Space before punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Missing space after sentence
        
        return text.strip()
    
    def format_as_title(self, text: str) -> str:
        """Format text as a proper title case."""
        if not text:
            return text
        
        # Clean the text first
        text = self.clean_text(text)
        
        # Split into words
        words = text.split()
        
        # Process each word
        formatted_words = []
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:"\'()[]{}')
            
            # Always capitalize first and last word
            if i == 0 or i == len(words) - 1:
                formatted_words.append(self.capitalize_word(word))
            # Check if word should always be capitalized
            elif word_lower in self.always_capitalize:
                formatted_words.append(word.upper())
            # Check if word should remain lowercase
            elif word_lower in self.lowercase_words:
                formatted_words.append(word.lower())
            # Default: capitalize first letter
            else:
                formatted_words.append(self.capitalize_word(word))
        
        return ' '.join(formatted_words)
    
    def format_as_keywords(self, text: str) -> str:
        """Format text as properly formatted keywords."""
        if not text:
            return text
        
        # Split by semicolon or comma
        keywords = re.split(r'[;,]', text)
        
        # Process each keyword
        formatted_keywords = []
        for keyword in keywords:
            keyword = keyword.strip()
            if keyword:
                # Clean and format
                keyword = self.clean_text(keyword)
                keyword = keyword.lower()
                
                # Capitalize if it's a technical term
                if keyword in self.always_capitalize:
                    keyword = keyword.upper()
                else:
                    # Capitalize first letter only
                    keyword = keyword.capitalize()
                
                formatted_keywords.append(keyword)
        
        return '; '.join(formatted_keywords)
    
    def format_as_technical(self, text: str) -> str:
        """Format technical data with proper capitalization."""
        if not text:
            return text
        
        # Clean the text
        text = self.clean_text(text)
        
        # Handle technical abbreviations
        for abbrev in self.always_capitalize:
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            replacement = abbrev.upper()
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and fixing basic issues."""
        if not text:
            return text
        
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize line breaks to spaces
        text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def apply_grammar_fixes(self, text: str) -> str:
        """Apply common grammar fixes."""
        for pattern, replacement in self.grammar_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def capitalize_word(self, word: str) -> str:
        """Properly capitalize a word while preserving punctuation."""
        if not word:
            return word
        
        # Find the first letter
        for i, char in enumerate(word):
            if char.isalpha():
                return word[:i] + char.upper() + word[i+1:]
        
        return word
    
    def ensure_proper_punctuation(self, text: str, is_sentence: bool = True) -> str:
        """Ensure text has proper punctuation."""
        if not text:
            return text
        
        text = text.strip()
        
        if is_sentence and text:
            # Check if it already ends with proper punctuation
            if not text.endswith(('.', '!', '?', ':', ';')):
                text += '.'
        
        return text
    
    def _is_title_field(self, field_name: str) -> bool:
        """Check if field should be formatted as a title."""
        title_indicators = [
            'name', 'title', 'caption', 'heading', 'subject', 'description_title',
            'color_name', 'category', 'type', 'model', 'make', 'software'
        ]
        
        field_lower = field_name.lower()
        return any(indicator in field_lower for indicator in title_indicators)
    
    def _is_sentence_field(self, field_name: str) -> bool:
        """Check if field should be formatted as a sentence."""
        sentence_indicators = [
            'description', 'caption', 'alt_text', 'comment', 'note', 'summary',
            'explanation', 'details', 'analysis'
        ]
        
        field_lower = field_name.lower()
        return any(indicator in field_lower for indicator in sentence_indicators)
    
    def _is_keyword_field(self, field_name: str) -> bool:
        """Check if field contains keywords."""
        keyword_indicators = ['keyword', 'tag', 'label', 'category']
        
        field_lower = field_name.lower()
        return any(indicator in field_lower for indicator in keyword_indicators)
    
    def _is_technical_field(self, field_name: str) -> bool:
        """Check if field contains technical data."""
        technical_indicators = [
            'exif', 'gps', 'rgb', 'cmyk', 'hsv', 'iso', 'resolution', 'format',
            'mode', 'codec', 'compression', 'algorithm', 'model', 'version'
        ]
        
        field_lower = field_name.lower()
        return any(indicator in field_lower for indicator in technical_indicators)


# Global text processor instance
text_processor = TextProcessor()