"""
File Management Core Module
Project ID: Image Processing App 20251119
Created: 2025-11-19 06:52:45 UTC
Author: The-Sage-Mage
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Generator, Tuple
from datetime import datetime
import logging


class FileManager:
    """Manages file operations, scanning, and organization."""
    
    def __init__(self, source_paths: List[Path], output_path: Path, 
                 admin_path: Path, config: dict, logger: logging.Logger):
        self.source_paths = source_paths
        self.output_path = output_path
        self.admin_path = admin_path
        self.config = config
        self.logger = logger
        self.supported_formats = [f.lower() for f in config.get('paths', {}).get('supported_formats', ['jpg', 'jpeg', 'png'])]
        self._total_files_found = 0
        self._current_file_index = 0
        
    def scan_for_images(self, source_path: Optional[Path] = None) -> Generator[Path, None, None]:
        """
        Recursively scan directories for image files.
        
        Args:
            source_path: Specific path to scan, or None for all source paths
            
        Yields:
            Path objects for found image files
        """
        paths_to_scan = [source_path] if source_path else self.source_paths
        
        for root_path in paths_to_scan:
            self.logger.info(f"Scanning directory: {root_path}")
            
            for root, dirs, files in os.walk(root_path):
                # Skip hidden and system directories
                if self.config.get('processing', {}).get('skip_hidden_files', True):
                    dirs[:] = [d for d in dirs if not d.startswith('.') and not self._is_system_dir(d)]
                
                for file in files:
                    # Skip hidden and system files
                    if self.config.get('processing', {}).get('skip_hidden_files', True):
                        if file.startswith('.') or self._is_system_file(file):
                            continue
                    
                    # Check file extension
                    if any(file.lower().endswith(f'.{fmt}') for fmt in self.supported_formats):
                        file_path = Path(root) / file
                        
                        # Validate path length (Windows limitation)
                        if len(str(file_path)) > self.config.get('processing', {}).get('max_path_length', 260):
                            self.logger.warning(f"Path too long, skipping: {file_path}")
                            continue
                        
                        self._total_files_found += 1
                        yield file_path
    
    def _is_system_dir(self, dirname: str) -> bool:
        """Check if directory is a system directory."""
        system_dirs = ['$RECYCLE.BIN', 'System Volume Information', 'Windows', 'Program Files']
        return dirname in system_dirs
    
    def _is_system_file(self, filename: str) -> bool:
        """Check if file is a system file."""
        system_prefixes = ['~$', 'desktop.ini', 'Thumbs.db']
        return any(filename.startswith(prefix) or filename == prefix for prefix in system_prefixes)
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """
        Calculate hash of a file.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm to use
            
        Returns:
            Hexadecimal hash string
        """
        hash_func = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def create_output_directory(self, folder_name: str, maintain_structure: bool = True,
                              source_path: Optional[Path] = None, relative_path: Optional[Path] = None) -> Path:
        """
        Create output directory with optional source structure preservation.
        
        Args:
            folder_name: Top-level folder name (e.g., 'CLR_ORIG')
            maintain_structure: Whether to maintain source directory structure
            source_path: Source root path for structure calculation
            relative_path: Relative path from source root
            
        Returns:
            Created directory path
        """
        output_dir = self.output_path / folder_name
        
        if maintain_structure and source_path and relative_path:
            # Calculate relative path from source root
            try:
                rel_path = relative_path.relative_to(source_path)
                output_dir = output_dir / rel_path.parent
            except ValueError:
                # If relative path calculation fails, use flat structure
                self.logger.warning(f"Could not calculate relative path for {relative_path}")
        
        # Create directory
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        except Exception as e:
            self.logger.error(f"Error creating directory {output_dir}: {e}")
            raise
    
    def generate_output_filename(self, original_filename: str, prefix: str,
                                suffix: str = "", sequence: Optional[int] = None) -> str:
        """
        Generate output filename with prefix and optional suffix.
        
        Args:
            original_filename: Original file name
            prefix: Prefix to add (e.g., 'CLR_ORIG_')
            suffix: Optional suffix before extension
            sequence: Optional sequence number for duplicates
            
        Returns:
            New filename
        """
        path = Path(original_filename)
        name_without_ext = path.stem
        extension = path.suffix
        
        new_name = f"{prefix}{name_without_ext}"
        
        if suffix:
            new_name += f"_{suffix}"
        
        if sequence is not None:
            new_name += f"_{sequence:04d}"
        
        return f"{new_name}{extension}"
    
    def save_file_with_dedup(self, source_path: Path, dest_dir: Path,
                            new_filename: str, copy: bool = True) -> Optional[Path]:
        """
        Save file with automatic deduplication.
        
        Args:
            source_path: Source file path
            dest_dir: Destination directory
            new_filename: New filename
            copy: If True, copy file; if False, move file
            
        Returns:
            Path to saved file or None if error
        """
        dest_path = dest_dir / new_filename
        sequence = 1
        
        # Check for existing file and deduplicate
        while dest_path.exists():
            # Compare hashes to check if truly duplicate
            source_hash = self.calculate_file_hash(source_path)
            dest_hash = self.calculate_file_hash(dest_path)
            
            if source_hash == dest_hash:
                self.logger.info(f"Duplicate file detected (same hash), skipping: {new_filename}")
                return dest_path
            
            # Generate new filename with sequence
            name_parts = Path(new_filename).stem.split('_')
            # Check if last part is already a sequence number
            if name_parts[-1].isdigit() and len(name_parts[-1]) == 4:
                name_parts[-1] = f"{sequence:04d}"
            else:
                name_parts.append(f"{sequence:04d}")
            
            new_stem = '_'.join(name_parts)
            new_filename = f"{new_stem}{Path(new_filename).suffix}"
            dest_path = dest_dir / new_filename
            sequence += 1
            
            if sequence > 9999:
                self.logger.error(f"Too many duplicates for file: {source_path}")
                return None
        
        # Perform file operation
        try:
            if copy:
                shutil.copy2(source_path, dest_path)
                self.logger.debug(f"Copied: {source_path} -> {dest_path}")
            else:
                shutil.move(str(source_path), str(dest_path))
                self.logger.debug(f"Moved: {source_path} -> {dest_path}")
            
            return dest_path
            
        except Exception as e:
            self.logger.error(f"Error {'copying' if copy else 'moving'} file {source_path}: {e}")
            return None
    
    def move_blurry_images(self, blurry_files: List[Tuple[Path, float]]) -> int:
        """
        Move blurry images to separate folder.
        
        Args:
            blurry_files: List of (file_path, blur_score) tuples
            
        Returns:
            Number of files successfully moved
        """
        moved_count = 0
        
        for file_path, blur_score in blurry_files:
            try:
                # Find which source root this file belongs to
                source_root = None
                for root in self.source_paths:
                    try:
                        file_path.relative_to(root)
                        source_root = root
                        break
                    except ValueError:
                        continue
                
                if not source_root:
                    self.logger.warning(f"Could not determine source root for {file_path}")
                    continue
                
                # Create IMGOrig-Blurry folder at same level as source root
                blurry_root = source_root.parent / "IMGOrig-Blurry"
                
                # Create output directory maintaining structure
                output_dir = self.create_output_directory(
                    "",  # No additional subfolder
                    maintain_structure=True,
                    source_path=source_root,
                    relative_path=file_path
                )
                
                # Update output_dir to be under blurry_root
                rel_path = file_path.relative_to(source_root)
                output_dir = blurry_root / rel_path.parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate new filename
                new_filename = self.generate_output_filename(
                    file_path.name,
                    "BLUR_ORIG_"
                )
                
                # Move file
                if self.save_file_with_dedup(file_path, output_dir, new_filename, copy=False):
                    moved_count += 1
                    self.logger.info(f"Moved blurry image (score: {blur_score:.2f}): {file_path.name}")
                    
            except Exception as e:
                self.logger.error(f"Error moving blurry image {file_path}: {e}")
        
        return moved_count
    
    def get_csv_filename(self, base_name: str) -> Path:
        """
        Generate CSV filename with timestamp.
        
        Args:
            base_name: Base name for CSV file (e.g., 'All_Image_Files_Focus')
            
        Returns:
            Full path to CSV file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{base_name}_{timestamp}.csv"
        
        # Create CSV directory if needed
        csv_dir = self.admin_path / "CSV"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        return csv_dir / filename
    
    def cleanup_handles(self):
        """Clean up file handles and release resources."""
        import gc
        gc.collect()
        self.logger.debug("File handles cleaned up")