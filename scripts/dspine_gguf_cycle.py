"""Thin re-export of the shared DSpine cycle helpers (campaign baseline harness).

The canonical implementation lives in hipengine.speculative.dspine_cycle so the
serving provider and the baseline harness share one cycle.
"""
from hipengine.speculative.dspine_cycle import (  # noqa: F401
    _capture_prompt_taps,
    load_dspine_weights,
    load_target_arrays,
    run_dspine_cycle,
)