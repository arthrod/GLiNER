"""Pytest configuration for GLiNER tests.

Provides a shim for importing gliner.config without requiring torch/onnxruntime
(pulled in transitively by gliner/__init__.py -> gliner/model.py).
"""

import sys
import types

# Shim the top-level gliner package so that ``import gliner.config`` does NOT
# trigger ``gliner/__init__.py`` (which imports torch via model.py).  The shim
# only needs to set __path__ so that sub-module resolution works.
if "gliner" not in sys.modules:
    _pkg = types.ModuleType("gliner")
    _pkg.__path__ = ["gliner"]
    sys.modules["gliner"] = _pkg
