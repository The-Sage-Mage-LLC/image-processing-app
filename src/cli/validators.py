"""
Path and Input Validators
Project ID: Image Processing App 20251119
Created: 2025-01-19 06:46:25 UTC
"""

import os
from pathlib import Path
from typing import List, Optional
import logging


class PathValidator:
    """Validates and processes path inputs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_source_paths(self, paths_string: str, config: dict) -> List[Path]:
        """
        Validate source paths from comma-delimited string.
        
        Args:
            paths_string: Comma-delimited string of paths
            config: Configuration dictionary
            
        Returns:
            List of validated Path objects
        """
        max_paths = config.get('validation', {}).get('max_source_paths', 10)
        min_paths = config.get('validation', {}).get('min_source_paths', 1)
        
        # Parse paths
        paths = [p.strip() for p in paths_string.split(',') if p.strip()]
        
        # Check count
        if len(paths) < min_paths:
            self.logger.error(f"Too few source paths provided. Minimum: {min_paths}")
            return []
        
        if len(paths) > max_paths:
            self.logger.warning(f"Too many source paths provided. Maximum: {max_paths}. Using first {max_paths}.")
            paths = paths[:max_paths]
        
        # Validate each path
        validated_paths = []
        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                self.logger.warning(f"Source path does not exist, skipping: {path}")
                continue
            if not path.is_dir():
                self.logger.warning(f"Source path is not a directory, skipping: {path}")
                continue
            validated_paths.append(path.resolve())
            self.logger.info(f"Validated source path: {path.resolve()}")
        
        return validated_paths
    
    def validate_output_path(self, path_string: str, config: dict) -> Optional[Path]:
        """
        Validate output path from string.
        
        Args:
            path_string: Path string
            config: Configuration dictionary
            
        Returns:
            Validated Path object or None
        """
        paths = [p.strip() for p in path_string.split(',') if p.strip()]
        
        if len(paths) != 1:
            self.logger.error(f"Exactly one output path required. Received: {len(paths)}")
            return None
        
        path = Path(paths[0])
        
        # Create if doesn't exist
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created output path: {path.resolve()}")
            except Exception as e:
                self.logger.error(f"Failed to create output path: {e}")
                return None
        
        if not path.is_dir():
            self.logger.error(f"Output path is not a directory: {path}")
            return None
        
        return path.resolve()
    
    def validate_admin_path(self, path_string: str, config: dict) -> Optional[Path]:
        """
        Validate admin path from string.
        
        Args:
            path_string: Path string
            config: Configuration dictionary
            
        Returns:
            Validated Path object or None
        """
        paths = [p.strip() for p in path_string.split(',') if p.strip()]
        
        if len(paths) != 1:
            self.logger.error(f"Exactly one admin path required. Received: {len(paths)}")
            return None
        
        path = Path(paths[0])
        
        # Create if doesn't exist
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created admin path: {path.resolve()}")
            except Exception as e:
                self.logger.error(f"Failed to create admin path: {e}")
                return None
        
        if not path.is_dir():
            self.logger.error(f"Admin path is not a directory: {path}")
            return None
        
        return path.resolve()