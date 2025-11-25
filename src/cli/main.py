#!/usr/bin/env python3
"""
Image Processing Application - CLI Main Entry Point
Project ID: Image Processing App 20251119
Created: 2025-01-19 06:46:25 UTC
Author: The-Sage-Mage
"""

import click
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import tomli
from colorama import init, Fore, Style

from ..utils.logger import setup_logging
from ..core.file_manager import FileManager
from ..core.image_processor import ImageProcessor
from .validators import PathValidator

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Version and project information
VERSION = "1.0.0"
PROJECT_ID = "Image Processing App 20251119"


class CLIApp:
    """Main CLI application class."""
    
    def __init__(self):
        self.logger = None
        self.config = None
        self.file_manager = None
        self.processor = None
        self.validator = PathValidator()
        
    def load_config(self, config_path: Path) -> dict:
        """Load configuration from TOML file."""
        try:
            with open(config_path, "rb") as f:
                return tomli.load(f)
        except FileNotFoundError:
            print(f"{Fore.YELLOW}Warning: Configuration file not found. Using defaults.{Style.RESET_ALL}")
            return self._get_default_config()
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Error loading configuration: {e}{Style.RESET_ALL}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Return default configuration."""
        return {
            "general": {
                "max_parallel_workers": 4,
                "log_level": "INFO",
                "enable_gpu": False
            },
            "validation": {
                "max_source_paths": 10,
                "min_source_paths": 1,
                "max_output_paths": 1,
                "max_admin_paths": 1
            }
        }
    
    def initialize(self, source_paths: str, output_path: str, admin_path: str, 
                  menu_option: int, config_path: Optional[str] = None):
        """Initialize the application with validated parameters."""
        
        # Load configuration
        config_file = Path(config_path) if config_path else Path("config/config.toml")
        self.config = self.load_config(config_file)
        
        # Setup logging
        admin_dir = Path(admin_path.split(',')[0].strip())
        self.logger = setup_logging(admin_dir, self.config)
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Image Processing Application v{VERSION}")
        self.logger.info(f"Project ID: {PROJECT_ID}")
        self.logger.info(f"Execution started: {datetime.now().isoformat()}")
        self.logger.info(f"{'='*60}")
        
        # Validate and parse paths
        validated_sources = self.validator.validate_source_paths(source_paths, self.config)
        validated_output = self.validator.validate_output_path(output_path, self.config)
        validated_admin = self.validator.validate_admin_path(admin_path, self.config)
        
        if not validated_sources:
            self.logger.error("No valid source paths provided. Exiting.")
            sys.exit(1)
        
        # Initialize components
        self.file_manager = FileManager(
            source_paths=validated_sources,
            output_path=validated_output,
            admin_path=validated_admin,
            config=self.config,
            logger=self.logger
        )
        
        self.processor = ImageProcessor(
            file_manager=self.file_manager,
            config=self.config,
            logger=self.logger
        )
        
        # Log initialization summary
        self.logger.info(f"Source paths: {', '.join([str(p) for p in validated_sources])}")
        self.logger.info(f"Output path: {validated_output}")
        self.logger.info(f"Admin path: {validated_admin}")
        self.logger.info(f"Menu option selected: {menu_option}")
        
        return validated_sources, validated_output, validated_admin
    
    def execute_menu_option(self, menu_option: int):
        """Execute the selected menu option."""
        
        menu_map = {
            1: self.execute_all,
            2: self.execute_blur_detection,
            3: self.execute_metadata_extraction,
            4: self.execute_caption_generation,
            5: self.execute_color_analysis,
            6: self.execute_color_copy,
            7: self.execute_grayscale,
            8: self.execute_sepia,
            9: self.execute_pencil_sketch,
            10: self.execute_coloring_book,
            11: self.execute_connect_dots,
            12: self.execute_color_by_numbers
        }
        
        if menu_option not in menu_map:
            self.logger.error(f"Invalid menu option: {menu_option}")
            print(f"{Fore.RED}Error: Invalid menu option {menu_option}. Valid options are 1-12.{Style.RESET_ALL}")
            sys.exit(1)
        
        try:
            self.logger.info(f"Executing menu option {menu_option}")
            menu_map[menu_option]()
            self.logger.info(f"Menu option {menu_option} completed successfully")
        except Exception as e:
            self.logger.error(f"Error executing menu option {menu_option}: {e}", exc_info=True)
            print(f"{Fore.RED}Error: Failed to execute menu option {menu_option}: {e}{Style.RESET_ALL}")
            sys.exit(1)
    
    def execute_all(self):
        """Execute all processing functions (Menu Item 1)."""
        self.logger.info("Executing all processing functions")
        for i in range(2, 13):
            self.execute_menu_option(i)
    
    def execute_blur_detection(self):
        """Execute blur detection (Menu Item 2)."""
        self.processor.detect_blur()
    
    def execute_metadata_extraction(self):
        """Execute metadata extraction (Menu Item 3)."""
        self.processor.extract_metadata()
    
    def execute_caption_generation(self):
        """Execute caption generation (Menu Item 4)."""
        self.processor.generate_captions()
    
    def execute_color_analysis(self):
        """Execute comprehensive color analysis (Menu Item 5)."""
        self.logger.info("Executing menu option 5")
        self.logger.info("Starting color analysis (Menu Item 5)")
        
        try:
            # Call the color analysis method from the processor
            self.processor.execute_menu_option_5()
            
        except Exception as e:
            self.logger.error(f"Error during color analysis execution: {e}")
            raise
    
    def execute_color_copy(self):
        """Execute color copy (Menu Item 6)."""
        self.processor.copy_color_images()
    
    def execute_grayscale(self):
        """Execute grayscale conversion (Menu Item 7)."""
        self.processor.convert_grayscale()
    
    def execute_sepia(self):
        """Execute sepia conversion (Menu Item 8)."""
        self.processor.convert_sepia()
    
    def execute_pencil_sketch(self):
        """Execute pencil sketch conversion (Menu Item 9)."""
        self.processor.convert_pencil_sketch()
    
    def execute_coloring_book(self):
        """Execute coloring book conversion (Menu Item 10)."""
        self.processor.convert_coloring_book()
    
    def execute_connect_dots(self):
        """Execute connect-the-dots conversion (Menu Item 11)."""
        self.processor.convert_connect_dots()
    
    def execute_color_by_numbers(self):
        """Execute color-by-numbers conversion (Menu Item 12)."""
        self.processor.convert_color_by_numbers()


@click.command()
@click.option(
    '--source-paths',
    required=True,
    help='Comma-delimited string of Windows-style source root paths (required)'
)
@click.option(
    '--output-path',
    required=True,
    help='Windows-style output root path (required)'
)
@click.option(
    '--admin-path',
    required=True,
    help='Windows-style admin output path for CSV and log files (required)'
)
@click.option(
    '--menu-option',
    default=1,
    type=click.IntRange(1, 12),
    help='Processing functionality option (1-12, default=1)'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    help='Path to configuration file (optional)'
)
@click.version_option(version=VERSION)
def main(source_paths: str, output_path: str, admin_path: str, 
         menu_option: int, config: Optional[str]):
    """
    Image Processing Application - Command Line Interface
    
    Process image files with various transformations and analyses.
    
    Menu Options:
    \b
    1  - Execute all functionality
    2  - Blur detection
    3  - Metadata extraction
    4  - Caption generation
    5  - Color analysis
    6  - Color copy (original)
    7  - Grayscale conversion
    8  - Sepia conversion
    9  - Pencil sketch conversion
    10 - Coloring book conversion
    11 - Connect-the-dots conversion
    12 - Color-by-numbers conversion
    """
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Image Processing Application v{VERSION}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Project ID: {PROJECT_ID}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    # Create and initialize application
    app = CLIApp()
    
    try:
        # Initialize with validated parameters
        app.initialize(source_paths, output_path, admin_path, menu_option, config)
        
        # Execute selected menu option
        app.execute_menu_option(menu_option)
        
        print(f"\n{Fore.GREEN}✓ Processing completed successfully!{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠ Processing interrupted by user.{Style.RESET_ALL}")
        if app.logger:
            app.logger.warning("Processing interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n{Fore.RED}✗ Error: {e}{Style.RESET_ALL}")
        if app.logger:
            app.logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()