"""
Main Application Launcher
Project ID: Image Processing App 20251119
Created: 2025-01-19 07:21:15 UTC
Author: The-Sage-Mage
"""

import sys
import os
from pathlib import Path
import argparse


def print_banner():
    """Print application banner."""
    print("="*60)
    print(" " * 10 + "IMAGE PROCESSING APPLICATION v1.0.0")
    print(" " * 10 + "Project ID: Image Processing App 20251119")
    print(" " * 15 + "Author: The-Sage-Mage")
    print("="*60)


def launch_cli(args):
    """Launch CLI interface."""
    print("\nLaunching CLI interface...")
    from src.cli.main import main
    
    # Build CLI arguments
    cli_args = [
        '--source-paths', args.source_paths,
        '--output-path', args.output_path,
        '--admin-path', args.admin_path,
        '--menu-option', str(args.menu_option)
    ]
    
    if args.config:
        cli_args.extend(['--config', args.config])
    
    # Modify sys.argv for Click
    sys.argv = ['cli'] + cli_args
    main()


def launch_gui():
    """Launch GUI interface."""
    print("\nLaunching GUI interface...")
    from src.gui.main_window import main
    main()


def main():
    """Main entry point."""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='Image Processing Application - Main Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Launch GUI:
    python main.py --gui
    
  Launch CLI:
    python main.py --cli --source-paths "C:\\Photos" --output-path "C:\\Output" --admin-path "C:\\Admin" --menu-option 1
    
  Process with specific operation:
    python main.py --cli --source-paths "D:\\Images" --output-path "D:\\Processed" --admin-path "D:\\Logs" --menu-option 7
        """
    )
    
    # Interface selection
    interface_group = parser.add_mutually_exclusive_group(required=True)
    interface_group.add_argument('--gui', action='store_true', 
                                help='Launch GUI interface')
    interface_group.add_argument('--cli', action='store_true',
                                help='Launch CLI interface')
    
    # CLI arguments
    parser.add_argument('--source-paths', type=str,
                       help='Comma-delimited source paths (CLI only)')
    parser.add_argument('--output-path', type=str,
                       help='Output path (CLI only)')
    parser.add_argument('--admin-path', type=str,
                       help='Admin output path (CLI only)')
    parser.add_argument('--menu-option', type=int, default=1,
                       choices=range(1, 13),
                       help='Menu option 1-12 (CLI only)')
    parser.add_argument('--config', type=str,
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Validate CLI arguments if CLI mode
    if args.cli:
        if not all([args.source_paths, args.output_path, args.admin_path]):
            parser.error("CLI mode requires --source-paths, --output-path, and --admin-path")
        launch_cli(args)
    else:
        launch_gui()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)