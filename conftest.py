# conftest.py — pytest configuration for NPU Fit Checker
import sys
import pathlib

# Ensure the repo root is on the Python path so `import fitchecker` works
# without installing the package.
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))
