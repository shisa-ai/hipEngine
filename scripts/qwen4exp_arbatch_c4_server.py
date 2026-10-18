#!/usr/bin/env python3
"""Qualification launcher: serve Qwen4Exp AR batch at capacity 4.

The Qwen4Exp plugin declares ``server_plain_ar_max_active_requests = 2`` and
rejects a larger resident capacity. This launcher raises that class attribute
before the server builds its generator so the width-4 physical batch path can
be exercised on hardware. It is a test harness: nothing in the package imports
it, no serving default changes, and the caller still passes
``--max-active-requests 4``.

Usage matches ``python -m hipengine.server`` with the extra requirement that
``HIPENGINE_QWEN4_EXP_AR_BATCH=1`` and ``HIPENGINE_QWEN4_EXP_AR_BATCH_SLOTS=4``
are set (or the slot count defaults to the serving capacity).
"""
from __future__ import annotations

from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator

CAPACITY = 4

Qwen4ExpGGUFTextGenerator.server_plain_ar_max_active_requests = CAPACITY
Qwen4ExpGGUFTextGenerator.server_plain_ar_max_active_requests_by_max_sequence_length = {
    1024: CAPACITY,
}

from hipengine.server.__main__ import main  # noqa: E402  (after the cap override)

if __name__ == "__main__":
    raise SystemExit(main())
