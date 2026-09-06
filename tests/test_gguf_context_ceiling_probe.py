"""Unit gates for the context-ceiling probe's pure helpers.

These cover the parsing, classification, prompt fitting, and response
validation the capacity campaign depends on and need no GPU; the probe's
server path is exercised on hardware.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "gguf_context_ceiling_probe.py"
    spec = importlib.util.spec_from_file_location("gguf_context_ceiling_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass resolution can find the module.
    import sys

    sys.modules["gguf_context_ceiling_probe"] = module
    spec.loader.exec_module(module)
    return module


def test_fit_prompt_tokens_hits_the_target_exactly() -> None:
    module = _load()
    unit = [1, 2, 3, 4, 5]

    def detokenize(ids):
        return "".join(chr(ord("a") + (i % 26)) for i in ids)

    def count(text: str) -> int:
        # Additive fake: each source char is one token; the "\n\n" suffix
        # contributes 2.
        return len(text)

    prompt, counted = module.fit_prompt_tokens(
        unit_token_ids=unit,
        suffix="\n\n",
        target_tokens=107,
        count_tokens=count,
        detokenize=detokenize,
    )
    assert counted == 107


def test_fit_prompt_tokens_corrects_retokenization_drift() -> None:
    module = _load()
    unit = [1, 2, 3, 4]

    def detokenize(ids):
        return "".join(chr(ord("a") + (i % 26)) for i in ids)

    calls = {"n": 0}

    def count(text: str) -> int:
        calls["n"] += 1
        base = len(text)
        # The first probe of the concatenated prompt merges one seam token.
        return base - 1 if calls["n"] > 1 and base > 8 else base

    prompt, counted = module.fit_prompt_tokens(
        unit_token_ids=unit,
        suffix="\n\n",
        target_tokens=50,
        count_tokens=count,
        detokenize=detokenize,
        tolerance=0,
    )
    assert counted == 50


def test_fit_prompt_tokens_accepts_within_tolerance() -> None:
    module = _load()
    unit = [1, 2, 3]

    def detokenize(ids):
        return "m" * len(ids)

    def count(text: str) -> int:
        # Always lands three under the token arithmetic: only tolerance saves it.
        return len(text) - 3

    prompt, counted = module.fit_prompt_tokens(
        unit_token_ids=unit,
        suffix="\n\n",
        target_tokens=20,
        count_tokens=count,
        detokenize=detokenize,
        tolerance=3,
    )
    assert 17 <= counted <= 20


def test_fit_prompt_tokens_rejects_an_unreachable_target() -> None:
    module = _load()

    def detokenize(ids):
        return "m" * len(ids)

    def count(text: str) -> int:
        # Size-dependent drift that alternates parity: with tolerance 0 no
        # filler size lands exactly on the target.
        return len(text) + (7 if len(text) % 2 == 0 else 8)

    with pytest.raises(ValueError, match="converge"):
        module.fit_prompt_tokens(
            unit_token_ids=[1, 2, 3, 4, 5],
            suffix="\n\n",
            target_tokens=40,
            count_tokens=count,
            detokenize=detokenize,
            tolerance=0,
        )


def test_fit_prompt_tokens_rejects_an_oversized_suffix() -> None:
    module = _load()

    def count(text: str) -> int:
        return len(text)

    with pytest.raises(ValueError, match="above target"):
        module.fit_prompt_tokens(
            unit_token_ids=[1, 2, 3, 4, 5],
            suffix="s" * 40,
            target_tokens=10,
            count_tokens=count,
            detokenize=lambda ids: "m" * len(ids),
        )


def test_classify_declares_oom_only_from_matching_evidence() -> None:
    module = _load()
    oom_log = "RuntimeError: HipError: HIP error 2: out of memory on device"
    status, reason = module.classify_failure(stage="startup", log_text=oom_log, body="")
    assert status == "oom_startup"
    assert "out-of-memory evidence" in reason

    # A different HIP error must not be labeled OOM.
    other = "HipError: HIP error 101: invalid device function"
    status, reason = module.classify_failure(stage="request", log_text=other, body="")
    assert status == "hip_error_101_request"

    # No HIP evidence at all: keep the raw failure visible.
    status, _ = module.classify_failure(stage="startup", log_text="segmentation fault", body="")
    assert status == "server_died_startup"
    status, _ = module.classify_failure(stage="request", log_text="", body="HTTP 500 boom")
    assert status == "request_failed"


def test_classify_matches_out_of_memory_without_the_word_hip() -> None:
    module = _load()
    status, _ = module.classify_failure(
        stage="request", log_text="", body="CUDA error: out of memory"
    )
    assert status == "oom_request"


def _completion_payload(
    *,
    prompt_tokens: int,
    generated: list[int],
    finish_reason: str = "length",
) -> dict:
    return {
        "choices": [
            {"text": "ledger", "finish_reason": finish_reason},
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(generated)},
        "hipengine": {
            "token_accounting": {"choice_generated_token_ids": [generated]},
            "speculative_mtp": {"serving_route": "off"},
        },
    }


def test_validate_completion_accepts_exact_accounting() -> None:
    module = _load()
    info, error = module.validate_completion(
        _completion_payload(prompt_tokens=3072, generated=[7, 8, 9, 10]),
        expected_prompt_tokens=3072,
        expected_completion_tokens=4,
    )
    assert error is None
    assert info is not None
    assert info["prompt_tokens"] == 3072
    assert info["finish_reason"] == "length"
    assert info["generated_token_ids"] == [7, 8, 9, 10]


def test_validate_completion_rejects_drifted_prompt_accounting() -> None:
    module = _load()
    info, error = module.validate_completion(
        _completion_payload(prompt_tokens=3000, generated=[7, 8, 9, 10]),
        expected_prompt_tokens=3072,
        expected_completion_tokens=4,
    )
    assert info is None
    assert "prompt_tokens" in error


def test_validate_completion_rejects_a_short_horizon() -> None:
    module = _load()
    info, error = module.validate_completion(
        _completion_payload(prompt_tokens=64, generated=[7, 8]),
        expected_prompt_tokens=64,
        expected_completion_tokens=4,
    )
    assert info is None
    assert "horizon" in error


def test_validate_completion_labels_an_early_stop() -> None:
    module = _load()
    info, error = module.validate_completion(
        _completion_payload(prompt_tokens=64, generated=[7, 8, 9, 10], finish_reason="stop"),
        expected_prompt_tokens=64,
        expected_completion_tokens=4,
    )
    assert info is None
    assert "finish_reason" in error


def test_validate_completion_rejects_missing_authoritative_ids() -> None:
    module = _load()
    payload = _completion_payload(prompt_tokens=64, generated=[7, 8])
    del payload["hipengine"]["token_accounting"]
    info, error = module.validate_completion(
        payload,
        expected_prompt_tokens=64,
        expected_completion_tokens=4,
    )
    assert info is None
    assert "authoritative" in error


def test_token_ids_sha256_is_stable() -> None:
    module = _load()
    first = module.token_ids_sha256([1, 2, 3])
    second = module.token_ids_sha256([1, 2, 3])
    other = module.token_ids_sha256([1, 2, 4])
    assert first == second
    assert first != other


def test_port_in_use_reports_a_bound_listener() -> None:
    module = _load()
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert module.port_in_use("127.0.0.1", port) is True
    assert module.port_in_use("127.0.0.1", port) is False


def test_parse_vram_selects_the_requested_gpu() -> None:
    module = _load()
    smi = """
GPU[0]\t\t: VRAM Total Memory (B): 48301604864
GPU[0]\t\t: VRAM Total Used Memory (B): 27959296
GPU[1]\t\t: VRAM Total Memory (B): 25753026560
GPU[1]\t\t: VRAM Total Used Memory (B): 23481671680
"""
    used, total = module.parse_vram(smi, 1)
    assert (used, total) == (23481671680, 25753026560)
    used0, total0 = module.parse_vram(smi, 0)
    assert (used0, total0) == (27959296, 48301604864)
    assert module.parse_vram(smi, 7) == (None, None)


def test_stage_peaks_windows_cover_startup_and_request() -> None:
    module = _load()
    samples = [
        (0.0, 100),
        (1.0, 500),   # startup window
        (2.0, 900),   # ready at 2.0
        (3.0, 2000),  # request window
        (4.0, 1500),  # request end at 4.0
        (5.0, 800),   # teardown
    ]
    marks = module.StageMarks(
        server_start=0.0, ready=2.0, request_start=2.5, request_end=4.0
    )
    peaks = module._stage_peaks(samples, marks)
    assert peaks["startup_peak_bytes"] == 900
    assert peaks["request_peak_bytes"] == 2000


def test_expected_workspace_lease_matches_server_formula() -> None:
    module = _load()
    # Capacity-honest lease: N=1 serves one slot x max(ceil(2048/256)=8, 4) = 8.
    assert module.expected_workspace_lease_pages(1, 2048) == 8
    # N=8 resident slots at 4K context: 8 x 16 = 128.
    assert module.expected_workspace_lease_pages(8, 4096) == 128
    # Wide context dominates the 1024-token minimum; N=2 leases two slots:
    # 2 x ceil(16384/256)=64 -> 128.
    assert module.expected_workspace_lease_pages(2, 16384) == 128


def test_kv_fallback_is_detected_from_capability() -> None:
    module = _load()
    ready = {
        "model": {
            "kv_capability": {
                "effective_kv_storage": "bf16",
                "runtime_action": "fallback_bf16",
            }
        },
        "kv_capacity": {"storage": "bf16"},
    }
    assert module._kv_fell_back(ready, "int8_per_token_head") is True
    engaged = {
        "model": {
            "kv_capability": {
                "effective_kv_storage": "int8_per_token_head",
                "runtime_action": "use_requested",
            }
        },
        "kv_capacity": {"storage": "int8_per_token_head"},
    }
    assert module._kv_fell_back(engaged, "int8_per_token_head") is False
    assert module._kv_fell_back({"kv_capacity": {"storage": "bf16"}}, "bf16") is False
