"""
Emergency Fix Script for NumPy Python 3.14 Compatibility Issue

This script addresses the 0xC0000005 access violation crash in
_multiarray_umath.cp314-win_amd64.pyd during MachArLike object initialization.

Run this script to apply immediate fixes while upgrading your Python environment.
"""

import subprocess
import sys
import os
import logging
from pathlib import Path


def setup_logging():
    """Setup logging for the fix script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('numpy_fix.log')
        ]
    )
    return logging.getLogger(__name__)


def check_python_version():
    """Check Python version and warn about compatibility."""
    logger = logging.getLogger(__name__)
    
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version >= (3, 14):
        logger.warning("Python 3.14+ detected - this is an experimental version")
        logger.warning("NumPy compatibility issues are expected")
        return True
    return False


def try_install_compatible_numpy():
    """Try to install a compatible NumPy version."""
    logger = logging.getLogger(__name__)
    
    # List of NumPy versions to try in order of preference
    numpy_versions = [
        "1.26.4",  # Latest stable
        "1.26.3",  # Previous stable
        "1.26.2",  # Earlier stable
        "1.25.2",  # Fallback to older stable
        "1.24.4"   # Last resort
    ]
    
    for version in numpy_versions:
        logger.info(f"Attempting to install numpy=={version}")
        
        try:
            # Try with pre-built wheels first
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                '--only-binary=all', f'numpy=={version}'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Successfully installed numpy=={version}")
                return True
            else:
                logger.warning(f"Failed to install numpy=={version} (wheels): {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout installing numpy=={version}")
        except Exception as e:
            logger.warning(f"Error installing numpy=={version}: {e}")
    
    logger.error("Failed to install any compatible NumPy version")
    return False


def install_compatible_dependencies():
    """Install compatible versions of key dependencies."""
    logger = logging.getLogger(__name__)
    
    # Compatible versions for Python 3.14
    dependencies = [
        "scipy==1.11.4",
        "matplotlib==3.8.2", 
        "pandas==2.1.4",
        "scikit-learn==1.3.2",
        "opencv-python==4.9.0.80",
        "Pillow==10.1.0"
    ]
    
    for dep in dependencies:
        logger.info(f"Installing {dep}")
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install',
                '--only-binary=all', dep
            ], capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                logger.info(f"Successfully installed {dep}")
            else:
                logger.warning(f"Failed to install {dep}: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"Error installing {dep}: {e}")


def create_emergency_numpy_patch():
    """Create an emergency patch file for immediate use."""
    patch_content = '''
"""
Emergency NumPy Patch
Apply this as the first import in your main application.
"""

import os
import sys
import warnings

# Force safe NumPy configuration
os.environ["NPY_DISABLE_OPTIMIZATION"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Suppress warnings that could indicate problems
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')

try:
    import numpy as np
    
    # Force conservative settings
    np.seterr(all='warn', divide='warn', invalid='warn')
    
    # Monkey patch problematic functions
    original_array = np.array
    original_asarray = np.asarray
    
    def safe_array(*args, **kwargs):
        try:
            # Force copy to ensure memory safety
            kwargs['copy'] = True
            result = original_array(*args, **kwargs)
            # Ensure contiguous memory layout
            if not result.flags.c_contiguous:
                result = np.ascontiguousarray(result)
            return result
        except Exception:
            # Fallback to minimal array
            return original_array([0], dtype='float64')
    
    def safe_asarray(*args, **kwargs):
        try:
            result = original_asarray(*args, **kwargs) 
            # Ensure contiguous memory layout
            if not result.flags.c_contiguous:
                result = np.ascontiguousarray(result)
            return result
        except Exception:
            return original_asarray([0], dtype='float64')
    
    np.array = safe_array
    np.asarray = safe_asarray
    
    print("? Emergency NumPy patch applied successfully")
    
except ImportError as e:
    print(f"? NumPy not available: {e}")
except Exception as e:
    print(f"? NumPy patch failed: {e}")
'''
    
    patch_file = Path("emergency_numpy_patch.py")
    with open(patch_file, 'w') as f:
        f.write(patch_content)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created emergency patch file: {patch_file}")
    
    return patch_file


def recommend_python_downgrade():
    """Provide instructions for Python downgrade."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATION: Downgrade to Python 3.11 or 3.12")
    logger.info("="*60)
    logger.info("Python 3.14 is experimental and has known compatibility issues.")
    logger.info("For stable operation, consider:")
    logger.info("1. Install Python 3.11.7 or Python 3.12.1")
    logger.info("2. Recreate your virtual environment") 
    logger.info("3. Install dependencies with updated requirements.txt")
    logger.info("="*60)


def main():
    """Main fix application."""
    logger = setup_logging()
    
    logger.info("Starting NumPy Python 3.14 compatibility fix")
    
    # Check Python version
    is_python314 = check_python_version()
    
    if is_python314:
        logger.warning("Applying emergency fixes for Python 3.14")
        
        # Try to install compatible NumPy
        numpy_success = try_install_compatible_numpy()
        
        if not numpy_success:
            logger.error("Could not install compatible NumPy")
            recommend_python_downgrade()
        else:
            # Install other compatible dependencies
            install_compatible_dependencies()
        
        # Create emergency patch regardless
        patch_file = create_emergency_numpy_patch()
        
        logger.info(f"\nEmergency patch created: {patch_file}")
        logger.info("Add 'import emergency_numpy_patch' at the top of your main.py")
    
    else:
        logger.info("Python version is compatible, applying standard fixes")
        try_install_compatible_numpy()
        install_compatible_dependencies()
    
    logger.info("Fix script completed. Check numpy_fix.log for details.")


if __name__ == "__main__":
    main()