"""Shared test configuration for tests/ directory.

Ensures the project root is on sys.path so that ``import ptbr`` works
in test modules without requiring a full package install.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'ptbr' is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
