#!/usr/bin/env python3
"""Execution-local producer-mirror control for Qwen4Exp pair diagnostics.

The GR mirror serves both consumers, so pair diagnostics disable publication
inside each measured execution scope to exercise staging instead. The scope
restores its prior value without changing process environment or peer contexts.
"""
from __future__ import annotations

from hipengine.runtime.qwen4_exp_runner import gr_f16_mirror


def pair_staging_without_producer_mirror():
    """Run the enclosed arms without GR producer mirror publication."""
    return gr_f16_mirror(False)


def producer_mirror_record(prior: bool) -> dict[str, object]:
    """Report the scoped control and the decision it replaced."""
    return dict(
        control="gr_f16_mirror",
        scope="execution-context",
        value=False,
        prior=prior,
        reason="the GR read's F16 mirror serves both consumers, so the pair "
               "would stage no conversion; publication is disabled in this "
               "execution context for every arm",
    )
