# Installation Guide

## System Requirements

### Python Requirements
- **Python Version**: 3.11 or higher
- **Package Manager**: pip or conda

### Operating System Support
- **Windows**: Windows 10 (version 1903+) or Windows 11
- **macOS**: macOS 10.15 (Catalina) or later
- **Linux**: Ubuntu 20.04+, CentOS 8+, or equivalent distributions

### Hardware Requirements

#### Minimum Requirements
- **CPU**: Dual-core processor (2.0 GHz or faster)
- **RAM**: 4 GB system memory
- **Storage**: 2 GB available disk space
- **GPU**: Optional, but recommended for performance acceleration

#### Recommended Requirements
- **CPU**: Quad-core processor (3.0 GHz or faster)
- **RAM**: 8 GB system memory (16 GB+ for large image processing)
- **Storage**: 10 GB available disk space (SSD recommended)
- **GPU**: NVIDIA GPU with CUDA support or AMD GPU with OpenCL support

## Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/The-Sage-Mage-LLC/image-processing-app.git
cd image-processing-app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the application
pip install -e .
```

### Method 2: Development Installation

```bash
# Clone the repository
git clone https://github.com/The-Sage-Mage-LLC/image-processing-app.git
cd image-processing-app

# Create development environment
python -m venv venv_dev
source venv_dev/bin/activate  # On Windows: venv_dev\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### Method 3: Docker Installation

```bash
# Pull the Docker image
docker pull ghcr.io/the-sage-mage-llc/image-processing-app:latest

# Run the application
docker run -it --rm \
  -v $(pwd)/images:/app/images \
  -v $(pwd)/output:/app/output \
  ghcr.io/the-sage-mage-llc/image-processing-app:latest
```

## Dependencies

### Core Dependencies

The application requires the following core packages:

```text
# Image Processing Libraries
Pillow>=10.0.0           # PIL fork for image processing
opencv-python>=4.8.0     # Computer vision library
numpy>=1.24.0            # Numerical computing
scipy>=1.11.0            # Scientific computing

# GUI Framework
PyQt6>=6.5.0            # Modern GUI framework
PyQt6-tools>=6.5.0      # Additional Qt tools

# Configuration and Utilities
PyYAML>=6.0              # YAML configuration files
toml>=0.10.2             # TOML configuration support
click>=8.1.0             # Command-line interface
tqdm>=4.65.0             # Progress bars

# Performance and Monitoring
psutil>=5.9.0            # System monitoring
joblib>=1.3.0            # Parallel processing
```

### Development Dependencies

Additional packages for development:

```text
# Testing Framework
pytest>=7.4.0           # Testing framework
pytest-cov>=4.1.0       # Coverage reporting
pytest-qt>=4.2.0        # Qt testing support
pytest-mock>=3.11.0     # Mocking utilities

# Code Quality
black>=23.7.0            # Code formatting
isort>=5.12.0           # Import sorting
flake8>=6.0.0           # Style checking
pylint>=2.17.0          # Code analysis
mypy>=1.5.0             # Static type checking

# Documentation
sphinx>=7.1.0           # Documentation generator
sphinx-rtd-theme>=1.3.0 # Read the Docs theme
sphinxcontrib-mermaid>=0.9.2  # Mermaid diagrams
```

## Platform-Specific Instructions

### Windows Installation

1. **Install Python 3.11+**:
   - Download from [python.org](https://python.org)
   - Ensure "Add Python to PATH" is checked during installation

2. **Install Git** (if not already installed):
   - Download from [git-scm.com](https://git-scm.com)

3. **Install Visual C++ Build Tools** (for some dependencies):
   - Download "Microsoft C++ Build Tools" from Microsoft

4. **Install the application**:
   ```cmd
   git clone https://github.com/The-Sage-Mage-LLC/image-processing-app.git
   cd image-processing-app
   python -m venv venv
   venv\Scripts\activate
   pip install -e .
   ```

### macOS Installation

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python and Git**:
   ```bash
   brew install python@3.11 git
   ```

3. **Install the application**:
   ```bash
   git clone https://github.com/The-Sage-Mage-LLC/image-processing-app.git
   cd image-processing-app
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

### Linux Installation

#### Ubuntu/Debian:
```bash
# Install system dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev git build-essential

# Install the application
git clone https://github.com/The-Sage-Mage-LLC/image-processing-app.git
cd image-processing-app
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

#### CentOS/RHEL/Fedora:
```bash
# Install system dependencies
sudo dnf install python3.11 python3.11-devel git gcc gcc-c++ make

# Install the application
git clone https://github.com/The-Sage-Mage-LLC/image-processing-app.git
cd image-processing-app
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

## Verification

### Test the Installation

1. **Verify Python packages**:
   ```bash
   python -c "import src.core.image_processor; print('Core modules imported successfully')"
   python -c "import PyQt6; print('GUI framework available')"
   ```

2. **Run the test suite**:
   ```bash
   pytest tests/ -v
   ```

3. **Check version information**:
   ```bash
   python src/main.py --version
   ```

4. **Test GUI application**:
   ```bash
   python src/gui/main_window.py
   ```

### Performance Testing

Run a quick performance test to ensure everything is working:

```bash
# Create test directory with sample images
mkdir -p test_images
python -c "
from PIL import Image
import numpy as np
for i in range(5):
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    img.save(f'test_images/test_{i}.jpg')
"

# Test batch processing
python src/main.py --input test_images/ --output test_output/ --operation resize --size 200,200
```

## Troubleshooting

### Common Installation Issues

1. **PyQt6 Installation Problems**:
   ```bash
   # On Linux, install Qt development packages
   sudo apt install qt6-base-dev qt6-tools-dev-tools
   
   # Reinstall PyQt6
   pip uninstall PyQt6
   pip install PyQt6
   ```

2. **OpenCV Installation Issues**:
   ```bash
   # Use opencv-python-headless for server environments
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```

3. **Permission Errors**:
   ```bash
   # Use user installation
   pip install --user -e .
   ```

4. **Memory Issues During Installation**:
   ```bash
   # Increase pip cache and use system memory efficiently
   pip install --no-cache-dir -e .
   ```

### Environment Issues

1. **Virtual Environment Problems**:
   ```bash
   # Remove and recreate virtual environment
   rm -rf venv
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install --upgrade pip setuptools wheel
   pip install -e .
   ```

2. **Path Issues**:
   ```bash
   # Ensure Python and pip are in PATH
   which python
   which pip
   python -m pip --version
   ```

## Next Steps

After successful installation:

1. **Read the Quick Start Guide**: Learn basic usage patterns
2. **Explore Examples**: Check out the examples directory
3. **API Documentation**: Familiarize yourself with the API
4. **Development Setup**: Set up development environment if contributing

## Support

If you encounter installation issues:

1. **Check the FAQ**: Common issues and solutions
2. **GitHub Issues**: Report bugs or request help
3. **Documentation**: Comprehensive guides and API reference
4. **Community**: Join our community forums