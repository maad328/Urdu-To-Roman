"""
Backend package for Urdu to Roman-Urdu translation system
"""
import sys
from pathlib import Path

# Add backend to Python path for proper imports
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

__version__ = "1.0.0"