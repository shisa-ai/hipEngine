from __future__ import annotations

import numpy as np

from scripts.qwen35_native_prefill_fullattn_stage_probe import _diff_payload


def test_qwen35_fullattn_stage_probe_diff_payload_reports_top_indices() -> None:
    serial = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    native = np.asarray([1.25, -2.0, -0.5], dtype=np.float32)

    diff = _diff_payload(serial, native)

    assert diff["elements"] == 3
    assert diff["max_abs"] == 1.0
    assert diff["top_abs_indices"][0]["index"] == 2
    assert diff["top_abs_indices"][0]["serial"] == 0.5
    assert diff["top_abs_indices"][0]["native"] == -0.5
