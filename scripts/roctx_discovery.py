"""Source profiler adapter; do not bind hipengine before child setup.

Load the shared implementation without importing its package. Profiling wrappers
must be able to choose the child's checkout and runtime before hipengine loads.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_spec = spec_from_file_location(
    __name__, Path(__file__).resolve().parents[1] / "hipengine/benchmark/roctx_discovery.py"
)
_module = module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
