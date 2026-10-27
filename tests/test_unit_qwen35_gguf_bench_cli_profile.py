"""Common defaults must survive CLI parsing and reach both execution modes."""
from __future__ import annotations

import json
import sys

import pytest

from scripts import qwen35_gguf_bench as bench


@pytest.mark.parametrize("persistent", (False, True))
@pytest.mark.parametrize(
    "flags, expected, applied, legacy",
    (
        ([], (True, True), True, None),
        (["--public-ar-profile"], (True, True), True, True),
        (["--no-use-wmma-prefill"], (False, True), True, None),
        (["--no-use-gemv-decode"], (True, False), True, None),
        (["--no-public-ar-profile"], (None, None), False, False),
    ),
)
def test_cli_routes_session_selectors_and_reports_effective_mismatch(
    monkeypatch, capsys, persistent, flags, expected, applied, legacy
):
    monkeypatch.delenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", raising=False)
    argv = ["qwen35_gguf_bench.py", "--warmup-runs", "0", "--measured-runs", "1", *flags]
    if persistent:
        argv.append("--persistent-session")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(bench, "scan_gguf", lambda path: object())
    monkeypatch.setattr(bench, "_gguf_tensor_inventory_summary", lambda info: {"tensor_inventory_hash": "fixture"})
    monkeypatch.setattr(bench, "_collect_benchmark_provenance", lambda **kwargs: {})
    monkeypatch.setattr(bench, "_summary", lambda runs: {})

    def run(**kwargs):
        assert (kwargs["use_wmma_prefill"], kwargs["use_gemv_decode"]) == expected
        return {
            "measured": True, "run_index": 1,
            "resolved_backend": "fixture", "target_arch": "fixture",
            "throughput": {"prefill_tok_s": 1.0, "decode_tok_s": 1.0},
            "memory": {"tracked_peak_allocated_gib": 0.0},
            "effective_use_wmma_prefill": expected[0],
            "effective_use_gemv_decode": expected[1],
        }

    monkeypatch.setattr(bench, "_run_once", run)
    monkeypatch.setattr(bench, "_run_persistent_session", lambda **kwargs: ([run(**kwargs)], 0.0, {}))
    assert bench.main() == 0
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["public_ar_profile"] is applied
    assert payload["public_ar_profile_applied"] is applied
    assert payload["requested_public_ar_profile"] is legacy
    mismatch = False in expected
    assert payload["shipping_ar_route_mismatch"] is mismatch
    assert ("WARNING:" in output.err) is mismatch
