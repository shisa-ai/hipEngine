"""Process-local K320 comparison control for single-threaded diagnostic arms."""
from contextlib import contextmanager
from hipengine.runtime import qwen4_exp_runner as runtime


@contextmanager
def k320_selection(enabled: bool):
    """Restore the live selector even if a diagnostic arm raises."""
    original = runtime._qwen4_exp_gr_k320_full_rows_key
    if not enabled:
        runtime._qwen4_exp_gr_k320_full_rows_key = lambda key, **kwargs: key
    try:
        yield
    finally:
        runtime._qwen4_exp_gr_k320_full_rows_key = original
