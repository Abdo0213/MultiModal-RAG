"""
Convenience entry point for scripts/prepare_data.py.
"""
import sys
from pathlib import Path

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.prepare_data import main

if __name__ == "__main__":
    main()
