#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sphinx Documentation Builder
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Automated Sphinx documentation generation and building.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse


class SphinxDocumentationBuilder:
    """Automated Sphinx documentation builder and manager."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.docs_dir = self.project_root / "docs"
        self.source_dir = self.docs_dir
        self.build_dir = self.docs_dir / "_build"
        self.static_dir = self.docs_dir / "_static"
        self.templates_dir = self.docs_dir / "_templates"
        
        # Ensure directories exist
        for directory in [self.docs_dir, self.static_dir, self.templates_dir]:
            directory.mkdir(exist_ok=True)
    
    def install_sphinx_dependencies(self) -> bool:
        """Install Sphinx and related documentation dependencies."""
        print("?? Installing Sphinx documentation dependencies...")
        
        dependencies = [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinxcontrib-mermaid>=0.9.2",
            "sphinx-autodoc-typehints>=1.24.0",
            "sphinx-copybutton>=0.5.2",
            "myst-parser>=2.0.0",      # Markdown support
            "sphinx-tabs>=3.4.1",      # Tabbed content
            "sphinx-design>=0.5.0",    # Design elements
        ]
        
        success_count = 0
        for dep in dependencies:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", dep],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    success_count += 1
                    print(f"   ? Installed {dep.split('>=')[0]}")
                else:
                    print(f"   ?? Failed to install {dep}")
            except subprocess.TimeoutExpired:
                print(f"   ? Timeout installing {dep}")
            except Exception as e:
                print(f"   ? Error installing {dep}: {e}")
        
        print(f"?? Installation summary: {success_count}/{len(dependencies)} packages installed")
        return success_count > len(dependencies) * 0.8
    
    def create_api_documentation_structure(self) -> None:
        """Create comprehensive API documentation structure."""
        print("?? Creating API documentation structure...")
        
        # Create API directory structure
        api_dir = self.docs_dir / "api"
        api_dir.mkdir(exist_ok=True)
        
        # Main API index
        api_index_content = """# API Reference

Complete API documentation for the Image Processing Application.

## Core Components

```{toctree}
:maxdepth: 2

core/index
gui/index
utils/index
```

## Quick Reference

### Core Processing Classes

- {class}`~src.core.image_processor.ImageProcessor` - Main image processing engine
- {class}`~src.core.batch_processor.BatchProcessor` - Batch processing controller
- {class}`~src.core.filters.FilterLibrary` - Image filtering operations

### GUI Components

- {class}`~src.gui.main_window.MainWindow` - Primary application window
- {class}`~src.gui.widgets.ImageViewWidget` - Image display widget
- {class}`~src.gui.dialogs.ProcessingDialog` - Processing configuration dialog

### Utility Modules

- {mod}`~src.utils.file_manager` - File and directory operations
- {mod}`~src.utils.config_manager` - Configuration management
- {mod}`~src.utils.logger` - Structured logging utilities

## Usage Examples

### Basic Image Processing

```python
from src.core.image_processor import ImageProcessor
from src.utils.config_manager import ConfigManager

# Initialize processor
config = ConfigManager()
processor = ImageProcessor(config)

# Process an image
result = processor.process_image(
    input_path="input.jpg",
    output_path="output.jpg",
    operations=['resize', 'enhance']
)
```

### Batch Processing

```python
from src.core.batch_processor import BatchProcessor

# Initialize batch processor
batch_processor = BatchProcessor()

# Process multiple images
results = batch_processor.process_directory(
    input_dir="./images/",
    output_dir="./processed/",
    operations=['resize', 'sharpen']
)
```

### GUI Integration

```python
from src.gui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication

# Create application
app = QApplication([])

# Initialize main window
window = MainWindow()
window.show()

# Start event loop
app.exec()
```
"""
        
        with open(api_dir / "index.md", 'w') as f:
            f.write(api_index_content)
        
        # Create core API documentation
        core_api_dir = api_dir / "core"
        core_api_dir.mkdir(exist_ok=True)
        
        core_index_content = """# Core API

Core image processing components and algorithms.

```{toctree}
:maxdepth: 2

image_processor
batch_processor
filters
algorithms
```

## Overview

The core API provides the fundamental image processing capabilities:

- **Image Processing**: Primary processing engine with support for various operations
- **Batch Processing**: Efficient processing of multiple images
- **Filters**: Comprehensive library of image filters and enhancements
- **Algorithms**: Advanced image processing algorithms

## Architecture

```{mermaid}
graph TB
    A[ImageProcessor] --> B[FilterLibrary]
    A --> C[AlgorithmEngine]
    D[BatchProcessor] --> A
    E[ConfigManager] --> A
    F[FileManager] --> D
```
"""
        
        with open(core_api_dir / "index.md", 'w') as f:
            f.write(core_index_content)
        
        # Create individual module documentation files
        self._create_module_documentation_files(api_dir)
        
        print("   ? API documentation structure created")
    
    def _create_module_documentation_files(self, api_dir: Path) -> None:
        """Create individual module documentation files."""
        
        # Core module docs
        core_modules = [
            ("image_processor", "Image Processor", "Main image processing engine with comprehensive operation support."),
            ("batch_processor", "Batch Processor", "Efficient batch processing of multiple images with progress tracking."),
            ("filters", "Filter Library", "Comprehensive collection of image filters and enhancement operations."),
            ("algorithms", "Algorithms", "Advanced image processing algorithms and mathematical operations."),
        ]
        
        for module_name, title, description in core_modules:
            content = f"""# {title}

{description}

```{{eval-rst}}
.. automodule:: src.core.{module_name}
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Usage Examples

### Basic Usage

```python
from src.core.{module_name} import *

# Example usage will be added based on actual implementation
```

## Class Hierarchy

```{{mermaid}}
classDiagram
    class {title.replace(' ', '')} {{
        +process()
        +configure()
        +validate()
    }}
```
"""
            
            file_path = api_dir / "core" / f"{module_name}.md"
            with open(file_path, 'w') as f:
                f.write(content)
        
        # GUI module docs
        gui_api_dir = api_dir / "gui"
        gui_api_dir.mkdir(exist_ok=True)
        
        gui_index_content = """# GUI API

Graphical user interface components and widgets.

```{toctree}
:maxdepth: 2

main_window
widgets
dialogs
```

## Overview

The GUI API provides a modern, intuitive interface built with PyQt6:

- **Main Window**: Primary application interface
- **Widgets**: Specialized UI components for image processing
- **Dialogs**: Configuration and processing dialogs

## Widget Hierarchy

```{mermaid}
graph TB
    A[MainWindow] --> B[ImageViewWidget]
    A --> C[ControlPanelWidget]
    A --> D[StatusWidget]
    E[ProcessingDialog] --> F[ParameterWidget]
    E --> G[ProgressWidget]
```
"""
        
        with open(gui_api_dir / "index.md", 'w') as f:
            f.write(gui_index_content)
        
        # Utils module docs
        utils_api_dir = api_dir / "utils"
        utils_api_dir.mkdir(exist_ok=True)
        
        utils_index_content = """# Utilities API

Utility modules for configuration, logging, and file management.

```{toctree}
:maxdepth: 2

config_manager
logger
file_manager
performance
```

## Overview

Utility modules provide supporting functionality:

- **Configuration Management**: Application settings and preferences
- **Logging**: Structured logging with correlation IDs
- **File Management**: File operations and path handling
- **Performance**: Performance monitoring and optimization
"""
        
        with open(utils_api_dir / "index.md", 'w') as f:
            f.write(utils_index_content)
    
    def create_custom_css(self) -> None:
        """Create custom CSS for enhanced documentation styling."""
        custom_css_content = """/* Custom CSS for Image Processing App Documentation */

/* Enhanced code blocks */
.highlight {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 0.375rem;
    padding: 1rem;
    margin: 1rem 0;
}

/* Improved admonitions */
.admonition {
    margin: 1.5rem 0;
    padding: 1rem;
    border-radius: 0.375rem;
    border-left: 4px solid #007bff;
}

.admonition.note {
    background-color: #e7f3ff;
    border-color: #007bff;
}

.admonition.warning {
    background-color: #fff3cd;
    border-color: #ffc107;
}

.admonition.danger {
    background-color: #f8d7da;
    border-color: #dc3545;
}

.admonition.tip {
    background-color: #d1ecf1;
    border-color: #17a2b8;
}

/* API documentation enhancements */
.py.class > dt {
    background-color: #f1f3f4;
    padding: 0.5rem;
    border-radius: 0.25rem;
    font-weight: 600;
}

.py.method > dt {
    background-color: #f8f9fa;
    padding: 0.25rem 0.5rem;
    border-left: 3px solid #007bff;
    margin-top: 1rem;
}

/* Enhanced navigation */
.wy-nav-content-wrap {
    margin-left: 300px;
}

.wy-nav-side {
    width: 300px;
}

/* Code copy button styling */
.highlight .copybutton {
    background-color: #6c757d;
    color: white;
    border: none;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
    cursor: pointer;
    opacity: 0.7;
    transition: opacity 0.2s;
}

.highlight .copybutton:hover {
    opacity: 1;
}

/* Mermaid diagram styling */
.mermaid {
    text-align: center;
    margin: 2rem 0;
}

/* Table styling */
table.docutils {
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
}

table.docutils th,
table.docutils td {
    border: 1px solid #dee2e6;
    padding: 0.75rem;
    text-align: left;
}

table.docutils th {
    background-color: #f8f9fa;
    font-weight: 600;
}

/* Responsive design improvements */
@media (max-width: 768px) {
    .wy-nav-content-wrap {
        margin-left: 0;
    }
    
    .wy-nav-side {
        width: 100%;
    }
}

/* Enhanced search results */
.search-summary {
    font-style: italic;
    color: #6c757d;
    margin-bottom: 1rem;
}

/* Version badge */
.version-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    background-color: #28a745;
    color: white;
    border-radius: 0.25rem;
    font-size: 0.875rem;
    font-weight: 500;
}

/* Documentation metadata */
.doc-metadata {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.375rem;
    margin-bottom: 2rem;
    border: 1px solid #e9ecef;
}

.doc-metadata h4 {
    margin-top: 0;
    color: #495057;
}

/* Performance indicators */
.performance-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    background-color: #e7f3ff;
    border: 1px solid #007bff;
    border-radius: 0.25rem;
    font-size: 0.875rem;
}

.performance-indicator.fast {
    background-color: #d4edda;
    border-color: #28a745;
    color: #155724;
}

.performance-indicator.slow {
    background-color: #f8d7da;
    border-color: #dc3545;
    color: #721c24;
}

/* API status indicators */
.api-status {
    display: inline-block;
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
}

.api-status.stable {
    background-color: #d4edda;
    color: #155724;
}

.api-status.experimental {
    background-color: #fff3cd;
    color: #856404;
}

.api-status.deprecated {
    background-color: #f8d7da;
    color: #721c24;
}
"""
        
        with open(self.static_dir / "custom.css", 'w') as f:
            f.write(custom_css_content)
        
        print("   ? Custom CSS created")
    
    def build_documentation(self, builder: str = "html", clean: bool = False) -> bool:
        """Build Sphinx documentation."""
        print(f"?? Building documentation with {builder} builder...")
        
        if clean and self.build_dir.exists():
            print("   ?? Cleaning previous build...")
            shutil.rmtree(self.build_dir)
        
        try:
            # Build documentation
            result = subprocess.run([
                sys.executable, "-m", "sphinx",
                "-b", builder,
                "-W",  # Treat warnings as errors
                "-T",  # Show full traceback
                str(self.source_dir),
                str(self.build_dir / builder)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"   ? Documentation built successfully")
                print(f"   ?? Output: {self.build_dir / builder}")
                return True
            else:
                print(f"   ? Documentation build failed")
                print(f"   Error: {result.stderr}")
                print(f"   Output: {result.stdout}")
                return False
        
        except subprocess.TimeoutExpired:
            print("   ? Documentation build timeout")
            return False
        except Exception as e:
            print(f"   ? Error building documentation: {e}")
            return False
    
    def generate_api_docs(self) -> bool:
        """Generate API documentation from source code."""
        print("?? Generating API documentation from source code...")
        
        try:
            # Use sphinx-apidoc to generate API documentation
            result = subprocess.run([
                sys.executable, "-m", "sphinx.ext.apidoc",
                "-f",  # Force overwrite
                "-e",  # Put each module on separate page
                "-T",  # Generate table of contents
                "-M",  # Put module documentation before submodule
                "-o", str(self.docs_dir / "api" / "generated"),
                str(self.project_root / "src"),
                str(self.project_root / "src" / "tests*"),  # Exclude tests
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("   ? API documentation generated")
                return True
            else:
                print(f"   ?? API generation had issues: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("   ? API generation timeout")
            return False
        except Exception as e:
            print(f"   ? Error generating API docs: {e}")
            return False
    
    def setup_documentation_automation(self) -> None:
        """Setup automated documentation building."""
        print("?? Setting up documentation automation...")
        
        # Create documentation build script
        build_script_content = '''#!/usr/bin/env python3
"""Automated documentation build script."""

import sys
import subprocess
from pathlib import Path

def main():
    """Build documentation automatically."""
    docs_dir = Path(__file__).parent
    
    print("?? Building Image Processing App Documentation...")
    
    # Install dependencies
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "sphinx", "sphinx-rtd-theme", "myst-parser"
    ], check=True)
    
    # Generate API docs
    subprocess.run([
        sys.executable, "-m", "sphinx.ext.apidoc",
        "-f", "-e", "-T", "-M",
        "-o", "api/generated",
        "../src",
        "../src/tests*"
    ], cwd=docs_dir)
    
    # Build HTML documentation
    subprocess.run([
        sys.executable, "-m", "sphinx",
        "-b", "html",
        ".", "_build/html"
    ], cwd=docs_dir, check=True)
    
    print("? Documentation build complete!")
    print(f"?? Open: {docs_dir / '_build' / 'html' / 'index.html'}")

if __name__ == "__main__":
    main()
'''
        
        build_script_path = self.docs_dir / "build.py"
        with open(build_script_path, 'w') as f:
            f.write(build_script_content)
        
        # Make executable on Unix systems
        try:
            import stat
            build_script_path.chmod(build_script_path.stat().st_mode | stat.S_IEXEC)
        except:
            pass
        
        # Create GitHub Actions workflow for documentation
        github_dir = self.project_root / ".github" / "workflows"
        github_dir.mkdir(parents=True, exist_ok=True)
        
        docs_workflow_content = '''name: Build and Deploy Documentation

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'docs/**'
      - 'src/**'
      - '.github/workflows/docs.yml'
  pull_request:
    branches: [ main ]
    paths:
      - 'docs/**'
      - 'src/**'

jobs:
  build-docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install sphinx sphinx-rtd-theme myst-parser sphinxcontrib-mermaid
        
    - name: Build documentation
      run: |
        cd docs
        python build.py
        
    - name: Upload documentation artifacts
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: docs/_build/html/
        
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: docs/_build/html/
'''
        
        with open(github_dir / "docs.yml", 'w') as f:
            f.write(docs_workflow_content)
        
        print("   ? Documentation automation setup complete")
    
    def setup_complete_documentation_system(self) -> bool:
        """Setup the complete Sphinx documentation system."""
        print("?? Setting up complete Sphinx documentation system...")
        
        # Install dependencies
        if not self.install_sphinx_dependencies():
            print("? Failed to install Sphinx dependencies")
            return False
        
        # Create documentation structure
        self.create_api_documentation_structure()
        
        # Create custom styling
        self.create_custom_css()
        
        # Generate API documentation
        self.generate_api_docs()
        
        # Setup automation
        self.setup_documentation_automation()
        
        # Build initial documentation
        if self.build_documentation(clean=True):
            print("? Complete Sphinx documentation system setup successfully!")
            print(f"?? Documentation available at: {self.build_dir / 'html' / 'index.html'}")
            return True
        else:
            print("?? Documentation system setup with build warnings")
            return False


def main():
    """Main entry point for Sphinx documentation setup."""
    parser = argparse.ArgumentParser(description="Sphinx documentation builder")
    parser.add_argument("--install", action="store_true", help="Install Sphinx dependencies")
    parser.add_argument("--build", action="store_true", help="Build documentation")
    parser.add_argument("--clean", action="store_true", help="Clean build before building")
    parser.add_argument("--api", action="store_true", help="Generate API documentation")
    parser.add_argument("--setup", action="store_true", help="Setup complete documentation system")
    parser.add_argument("--builder", default="html", help="Sphinx builder to use")
    
    args = parser.parse_args()
    
    builder = SphinxDocumentationBuilder()
    
    if args.setup:
        builder.setup_complete_documentation_system()
    elif args.install:
        builder.install_sphinx_dependencies()
    elif args.api:
        builder.generate_api_docs()
    elif args.build:
        builder.build_documentation(args.builder, args.clean)
    elif len(sys.argv) == 1:
        # Default: setup complete system
        builder.setup_complete_documentation_system()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()