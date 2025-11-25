"""
Setup script for Image Processing Application
Project ID: Image Processing App 20251119
Created: 2025-11-19 06:52:45 UTC
Author: The-Sage-Mage
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith('#')]

setup(
    name="image-processing-app",
    version="1.0.0",
    author="The-Sage-Mage",
    author_email="",
    description="Comprehensive image processing application with CLI and GUI interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/The-Sage-Mage-LLC/the-sage-mage-llc-image-file-processing-app-20251119",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: Microsoft :: Windows :: Windows 11",
    ],
    python_requires=">=3.11,<3.14",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "imgproc=src.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.toml"],
    },
)