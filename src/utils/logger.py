"""
Logging Configuration and Utilities
Project ID: Image Processing App 20251119
Created: 2025-11-19 06:52:45 UTC
Author: The-Sage-Mage
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(admin_path: Path, config: dict) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        admin_path: Path to admin directory for log files
        config: Configuration dictionary
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_dir = admin_path / "Logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get logging configuration
    log_config = config.get('logging', {})
    log_level = config.get('general', {}).get('log_level', 'INFO')
    
    # Create logger
    logger = logging.getLogger('ImageProcessingApp')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    if log_config.get('log_to_console', True):
        console_handler = logging.StreamHandler()
        console_format = log_config.get(
            'console_format',
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(logging.Formatter(console_format))
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_config.get('log_to_file', True):
        log_filename = log_dir / f"image_processing_{datetime.now().strftime('%Y-%m-%d')}.log"
        
        # Rotating file handler
        max_bytes = log_config.get('max_log_size_mb', 100) * 1024 * 1024
        backup_count = log_config.get('backup_count', 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        file_format = log_config.get(
            'file_format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(logging.Formatter(file_format))
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        logger.addHandler(file_handler)
    
    # Log initialization
    logger.info("="*60)
    logger.info("Logging system initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {log_dir}")
    logger.info("="*60)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'ImageProcessingApp.{name}')