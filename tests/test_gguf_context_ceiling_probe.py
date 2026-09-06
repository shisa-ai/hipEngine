"""Unit gates for the context-ceiling probe's pure helpers.

These cover the parsing and classification the capacity campaign depends on and
need no GPU; the probe's server path is exercised on hardware.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "gguf_context_ceiling_probe.py"
    spec = importlib.util.spec_from_file_location("gguf_context_ceiling_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SMI = """
GPU[0]		: VRAM Total Memory (B): 48301604864
GPU[0]		: VRAM Total Used Memory (B): 27959296
GPU[1]		: VRAM Total Memory (B): 25753026560
GPU[1]		: VRAM Total Used Memory (B): 23481671680
"""


def test_parse_vram_selects_the_requested_gpu() -> None:
    module = _load()
    used, total = module.parse_vram(_SMI, 1)
    assert (used, total) == (23481671680, 25753026560)
    used0, total0 = module.parse_vram(_SMI, 0)
    assert (used0, total0) == (27959296, 48301604864)


def test_parse_vram_returns_none_for_an_absent_gpu() -> None:
    module = _load()
    assert module.parse_vram(_SMI, 7) == (None, None)


def test_classify_requires_a_completed_request() -> None:
    module = _load()
    # Allocating is not passing: a dead server never completed a request.
    assert module.classify(False, False, "")[0] == "server_died"
    assert module.classify(True, True, '{"text": "hi"}')[0] == "ok"


def test_classify_recognizes_an_out_of_memory_body() -> None:
    module = _load()
    status, _reason = module.classify(True, False, "HipError: HIP error 2: out of memory")
    assert status == "oom"


def test_classify_preserves_an_unrecognized_failure_body() -> None:
    module = _load()
    status, reason = module.classify(True, False, "some other server error")
    assert status == "request_failed"
    assert "some other server error" in reason
