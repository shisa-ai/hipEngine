"""Campaign bench with a live counter proving the sgemm tier actually ran.

§4.4: a win production never selects is not a win.  This wraps
``get_rocblas`` in the router module, counts ``sgemm_rowmajor_nt`` calls,
and prints the count at exit so the artifact run carries route evidence.
"""

import atexit
import sys

import hipengine.kernels.hip_gfx1100.gemma4.gemma4_router as _router

_real_get = _router.get_rocblas
_counter = {"n": 0}


class _Counting:
    def sgemm_rowmajor_nt(self, *args, **kwargs):
        _counter["n"] += 1
        return _real_get().sgemm_rowmajor_nt(*args, **kwargs)


_router.get_rocblas = lambda: _Counting()
atexit.register(lambda: print(f"P8_ROUTE_SGEMM_CALLS={_counter['n']}", flush=True))

from scripts.gemma4_campaign_bench import main as _bench_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_bench_main(sys.argv[1:]))
