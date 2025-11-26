"""
Force clear Python bytecode cache to ensure latest code runs
"""
import os
import shutil
from pathlib import Path

def clear_pycache(directory):
    """Recursively clear __pycache__ directories."""
    cleared = 0
    for root, dirs, files in os.walk(directory):
        if '__pycache__' in dirs:
            pycache_path = Path(root) / '__pycache__'
            print(f"Removing: {pycache_path}")
            shutil.rmtree(pycache_path)
            cleared += 1
    return cleared

if __name__ == "__main__":
    # Get the repository root
    repo_root = Path(__file__).parent
    
    print("=" * 60)
    print("CLEARING PYTHON BYTECODE CACHE")
    print("=" * 60)
    
    # Clear src directory
    src_dir = repo_root / "src"
    if src_dir.exists():
        print(f"\nClearing cache in: {src_dir}")
        count = clear_pycache(src_dir)
        print(f"Removed {count} __pycache__ directories")
    
    # Clear tests directory
    tests_dir = repo_root / "tests"
    if tests_dir.exists():
        print(f"\nClearing cache in: {tests_dir}")
        count = clear_pycache(tests_dir)
        print(f"Removed {count} __pycache__ directories")
    
    print("\n" + "=" * 60)
    print("CACHE CLEARED - Please re-run your application now")
    print("=" * 60)
