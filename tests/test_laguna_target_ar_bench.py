from __future__ import annotations

import pytest

import scripts.laguna_target_ar_bench as benchmark
from scripts.laguna_target_ar_bench import (
    _aggregate,
    _laguna_f16_prefill_configuration,
    _paired_correctness,
    _promotion_gate,
)


def _row(
    *,
    prompt_id: str,
    category: str,
    mode: str,
    repetition: int,
    prompt_tokens: int,
    prefill_seconds: float,
    decode_seconds: float,
    generated: tuple[int, ...],
) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "category": category,
        "prompt_tokens": prompt_tokens,
        "prompt_token_ids_sha256": f"prompt-{prompt_id}",
        "mode": mode,
        "repetition": repetition,
        "prefill_seconds": prefill_seconds,
        "ttft_seconds": prefill_seconds,
        "prefill_tok_s": prompt_tokens / prefill_seconds,
        "checkpoints": {
            "4": {
                "output_tokens": 4,
                "generated_token_ids": list(generated[:4]),
                "generated_ids_sha256": f"{mode}-{prompt_id}-{repetition}-{generated[:4]}",
                "decode_forward_calls": 3,
                "decode_seconds": decode_seconds,
                "decode_tok_s": 3 / decode_seconds,
                "total_seconds": prefill_seconds + decode_seconds,
                "e2e_output_tok_s": 4 / (prefill_seconds + decode_seconds),
            }
        },
    }


def test_aggregate_uses_weighted_complete_timing_scopes() -> None:
    rows = [
        _row(
            prompt_id="a",
            category="code",
            mode="serial",
            repetition=0,
            prompt_tokens=10,
            prefill_seconds=2.0,
            decode_seconds=3.0,
            generated=(1, 2, 3, 4),
        ),
        _row(
            prompt_id="b",
            category="general_en",
            mode="serial",
            repetition=0,
            prompt_tokens=20,
            prefill_seconds=3.0,
            decode_seconds=3.0,
            generated=(5, 6, 7, 8),
        ),
        _row(
            prompt_id="a",
            category="code",
            mode="bulk",
            repetition=0,
            prompt_tokens=10,
            prefill_seconds=1.0,
            decode_seconds=3.0,
            generated=(1, 2, 3, 4),
        ),
        _row(
            prompt_id="b",
            category="general_en",
            mode="bulk",
            repetition=0,
            prompt_tokens=20,
            prefill_seconds=2.0,
            decode_seconds=3.0,
            generated=(5, 6, 7, 8),
        ),
    ]

    aggregate = _aggregate(rows, (4,))

    assert aggregate["serial"]["prefill_tok_s"] == pytest.approx(6.0)
    assert aggregate["bulk"]["prefill_tok_s"] == pytest.approx(10.0)
    assert aggregate["serial"]["horizons"]["4"]["decode_tok_s"] == pytest.approx(1.0)
    assert aggregate["bulk_vs_serial"]["4"]["prefill_speedup"] == pytest.approx(5 / 3)
    assert aggregate["bulk_vs_serial"]["4"]["decode_speedup"] == pytest.approx(1.0)
    assert aggregate["bulk_vs_serial"]["4"]["e2e_speedup"] > 1.0


def test_paired_correctness_requires_serial_bulk_and_repeat_determinism() -> None:
    rows = []
    for repetition in (0, 1):
        for mode in ("serial", "bulk"):
            rows.append(
                _row(
                    prompt_id="a",
                    category="code",
                    mode=mode,
                    repetition=repetition,
                    prompt_tokens=10,
                    prefill_seconds=1.0,
                    decode_seconds=1.0,
                    generated=(1, 2, 3, 4),
                )
            )
            rows[-1]["checkpoints"]["4"]["generated_ids_sha256"] = "stable"

    assert _paired_correctness(rows, (4,))["pass"] is True

    rows[-1]["checkpoints"]["4"]["generated_token_ids"] = [1, 2, 3, 9]
    rows[-1]["checkpoints"]["4"]["generated_ids_sha256"] = "changed"
    failed = _paired_correctness(rows, (4,))
    assert failed["pass"] is False
    assert failed["same_mode_repeat_deterministic"] is False


def test_f16_prefill_configuration_records_requested_and_resolved_strategy(
    monkeypatch,
) -> None:
    capabilities = {
        "LAGUNA_F16_PREFILL_STRATEGY": "tiled",
        "LAGUNA_F16_PREFILL_MIN_ROWS": 8,
    }
    monkeypatch.setattr(
        benchmark,
        "backend_package_capability",
        lambda _backend, name, default=None: capabilities.get(name, default),
    )

    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_PREFILL", "auto")
    automatic = _laguna_f16_prefill_configuration("hip_gfx1151")
    assert automatic == {
        "requested": "auto",
        "backend_strategy": "tiled",
        "backend_min_rows": 8,
        "effective_strategy": "tiled",
        "effective_min_rows": 8,
        "rows_one_always_gemv": True,
    }

    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_PREFILL", "tiled")
    forced = _laguna_f16_prefill_configuration("hip_gfx1151")
    assert forced["effective_strategy"] == "tiled"
    assert forced["effective_min_rows"] == 2

    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_PREFILL", "wmma")
    matrix = _laguna_f16_prefill_configuration("hip_gfx1151")
    assert matrix["effective_strategy"] == "wmma"
    assert matrix["effective_min_rows"] == 16

    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_PREFILL", "gemv")
    disabled = _laguna_f16_prefill_configuration("hip_gfx1151")
    assert disabled["effective_strategy"] == "gemv"
    assert disabled["effective_min_rows"] is None


def test_promotion_gate_is_fail_closed_per_category() -> None:
    aggregate = {
        "bulk_vs_serial": {
            "16": {
                "prefill_speedup": 1.2,
                "decode_speedup": 1.0,
                "e2e_speedup": 1.1,
            },
            "32": {
                "prefill_speedup": 1.2,
                "decode_speedup": 0.99,
                "e2e_speedup": 1.1,
            },
        }
    }
    categories = {
        category: {
            "bulk_vs_serial": {
                "16": {
                    "prefill_speedup": 1.1,
                    "decode_speedup": 0.99,
                    "e2e_speedup": 1.05,
                },
                "32": {
                    "prefill_speedup": 1.1,
                    "decode_speedup": 1.01,
                    "e2e_speedup": 1.05,
                },
            }
        }
        for category in ("code", "general_en", "general_ja", "mixed_ja_en")
    }

    accepted = _promotion_gate(aggregate, categories, (16, 32))
    assert accepted["pass"] is True

    categories["general_ja"]["bulk_vs_serial"]["32"]["e2e_speedup"] = 0.97
    rejected = _promotion_gate(aggregate, categories, (16, 32))
    assert rejected["pass"] is False
    assert "general_ja:h32:e2e_speedup" in rejected["failed_checks"]
