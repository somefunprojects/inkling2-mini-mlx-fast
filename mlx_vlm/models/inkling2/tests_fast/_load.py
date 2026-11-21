"""Import inkling2 modules without running mlx_vlm's heavy package __init__s.

Usage (from the repo root): python mlx_vlm/models/inkling2/tests_fast/test_x.py
"""
import importlib, os, sys, types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _stub(name, path):
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = [path]
        sys.modules[name] = m


_stub("mlx_vlm", ROOT)
_stub("mlx_vlm.models", os.path.join(ROOT, "models"))
_stub("mlx_vlm.models.inkling2", os.path.join(ROOT, "models", "inkling2"))
_stub("mlx_vlm.models.inkling", os.path.join(ROOT, "models", "inkling"))
_stub("mlx_vlm.speculative", os.path.join(ROOT, "speculative"))


def load(name):
    return importlib.import_module(name)
