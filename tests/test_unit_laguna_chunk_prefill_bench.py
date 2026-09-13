from __future__ import annotations

from copy import deepcopy

from scripts.laguna_chunk_prefill_bench import (
    MODES,
    _aggregate,
    _mode_order,
    _paired_correctness,
    _promotion_gate,
)

HORIZONS = (16, 32)
CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")


def _rows():
    rows = []
    for prompt_index, category in enumerate(CATEGORIES):
        prompt_id = f"prompt_{prompt_index}"
        for repetition in range(2):
            generated = list(range(32))
            for mode in MODES:
                prefill = 2.0 if mode == "chunk_64" else 1.0
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "category": category,
                        "prompt_tokens": 96,
                        "mode": mode,
                        "repetition": repetition,
                        "prefill_seconds": prefill,
                        "ttft_seconds": prefill,
                        "checkpoints": {
                            str(horizon): {
                                "generated_token_ids": generated[:horizon],
                                "generated_ids_sha256": f"{prompt_id}-{horizon}",
                                "decode_forward_calls": horizon - 1,
                                "decode_seconds": float(horizon - 1),
                                "output_tokens": horizon,
                                "total_seconds": prefill + horizon - 1,
                                "decode_tok_s": 1.0,
                                "e2e_output_tok_s": horizon / (prefill + horizon - 1),
                            }
                            for horizon in HORIZONS
                        },
                    }
                )
    return rows


def test_lpf4_mode_order_balances_prompts_and_repetitions() -> None:
    for index in range(10):
        assert _mode_order(index, 1) == tuple(reversed(_mode_order(index, 0)))


def test_lpf4_chunk128_aggregate_passes_exact_category_gate() -> None:
    rows = _rows()
    correctness = _paired_correctness(rows, HORIZONS)
    aggregate = _aggregate(rows, HORIZONS)
    promotion = _promotion_gate(
        aggregate,
        correctness,
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
    )

    assert correctness["pass"] is True
    assert aggregate["chunk128_vs_chunk64"]["prefill_speedup"] == 2.0
    assert all(
        value["prefill_speedup"] == 2.0
        for value in aggregate["chunk128_vs_chunk64"]["categories"].values()
    )
    assert promotion["pass"] is True
    assert promotion["failed_checks"] == []


def test_lpf4_gate_fails_on_output_mismatch_or_decode_regression() -> None:
    rows = _rows()
    mismatch_rows = deepcopy(rows)
    candidate = next(row for row in mismatch_rows if row["mode"] == "chunk_128")
    candidate["checkpoints"]["32"]["generated_token_ids"][-1] = 999
    mismatch = _paired_correctness(mismatch_rows, HORIZONS)
    mismatch_gate = _promotion_gate(
        _aggregate(mismatch_rows, HORIZONS),
        mismatch,
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
    )
    assert mismatch["pass"] is False
    assert "chunk_outputs_not_exact" in mismatch_gate["failed_checks"]

    decode_rows = deepcopy(rows)
    for row in decode_rows:
        if row["mode"] != "chunk_128":
            continue
        for horizon in HORIZONS:
            checkpoint = row["checkpoints"][str(horizon)]
            checkpoint["decode_seconds"] *= 1.1
            checkpoint["total_seconds"] = row["prefill_seconds"] + checkpoint["decode_seconds"]
            checkpoint["decode_tok_s"] = checkpoint["decode_forward_calls"] / checkpoint[
                "decode_seconds"
            ]
            checkpoint["e2e_output_tok_s"] = checkpoint["output_tokens"] / checkpoint[
                "total_seconds"
            ]
    decode_gate = _promotion_gate(
        _aggregate(decode_rows, HORIZONS),
        _paired_correctness(decode_rows, HORIZONS),
        {"pass": True},
        horizons=HORIZONS,
        recovered=True,
    )
    assert decode_gate["pass"] is False
    assert "h32_decode_outside_2pct" in decode_gate["failed_checks"]
