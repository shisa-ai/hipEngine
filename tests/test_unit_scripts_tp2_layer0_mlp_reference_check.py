"""CPU tests for the offline layer-0 MLP FP32 reference check.

No GPU, no model, no capture: the test synthesizes the ``.npz`` the GPU script
writes (same key naming, same raw-byte encoding) from the numpy reference in
``scripts/tp2_layer0_mlp_capture.py`` and drives ``main`` end to end. The
down-boundary block is the part that crashed on the real capture, so it is
covered by a real run rather than by a unit-level shape assertion.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.tp2_layer0_mlp_reference_check as refcheck
from scripts.tp2_layer0_mlp_capture import (
    BF16,
    bf16_round,
    reference_full_width_mlp,
    reference_sharded_mlp,
    shard_slices,
)

ROWS = 3
HIDDEN = 4
FFN = 8
RANKS = 2


def _bf16_bytes(values: np.ndarray) -> np.ndarray:
    """bf16 payload as raw little-endian bytes, exactly as the capture stores it."""

    bits = bf16_round(values).view(np.uint32) >> np.uint32(16)
    return bits.astype(np.uint16).view(np.uint8)


def _f32_bytes(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint8)


def _weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(23)
    return {
        "gate": (rng.standard_normal((FFN, HIDDEN)) * 0.05).astype(np.float32),
        "up": (rng.standard_normal((FFN, HIDDEN)) * 0.05).astype(np.float32),
        "down": (rng.standard_normal((HIDDEN, FFN)) * 0.05).astype(np.float32),
    }


def _synthetic_capture(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A capture in which both routes are *exactly* their own reference schedule."""

    rng = np.random.default_rng(29)
    post_norm = (rng.standard_normal((ROWS, HIDDEN)) * 1.5).astype(np.float32)
    residual = (rng.standard_normal((ROWS, HIDDEN)) * 3.0).astype(np.float32)

    full = reference_full_width_mlp(
        post_norm, weights["gate"], weights["up"], weights["down"]
    )
    sharded = reference_sharded_mlp(
        post_norm,
        weights["gate"],
        weights["up"],
        weights["down"],
        ranks=RANKS,
        partial_dtype=BF16,
    )

    payload: dict[str, np.ndarray] = {}
    for name, value in (
        ("post_norm", post_norm),
        ("residual", residual),
        ("ffn_intermediate", full["intermediate"]),
        ("ffn_down", full["down"]),
        ("out", bf16_round((residual + full["down"]).astype(np.float32))),
    ):
        payload[f"teacher.mlp.{name}"] = _bf16_bytes(np.asarray(value))

    reduced = np.asarray(sharded["reduced_f32"], dtype=np.float32)
    cast = bf16_round(reduced)
    for index, (start, stop) in enumerate(shard_slices(FFN, ranks=RANKS, what="ffn")):
        tag = f"candidate_{index}"
        rank = shard_slices(FFN, ranks=RANKS, what="ffn")[index]
        for name, value in (
            ("post_norm", post_norm),
            ("residual", residual),
            ("act", sharded["intermediates"][index]),
            ("down_partial", sharded["partials"][index]),
            ("out", bf16_round((residual + cast).astype(np.float32))),
        ):
            payload[f"{tag}.mlp.{name}"] = _bf16_bytes(np.asarray(value))
        payload[f"{tag}.mlp.gate"] = _bf16_bytes(
            bf16_round(post_norm @ weights["gate"][start:stop].T)
        )
        payload[f"{tag}.mlp.up"] = _bf16_bytes(
            bf16_round(post_norm @ weights["up"][start:stop].T)
        )
        assert rank == (start, stop)
    payload["candidate_0.mlp.reduced"] = _f32_bytes(reduced)
    payload["candidate_1.mlp.reduced"] = _f32_bytes(reduced)
    return payload


@pytest.fixture()
def synthetic_capture(tmp_path, monkeypatch):
    weights = _weights()
    capture = tmp_path / "synthetic.layer0.npz"
    np.savez(capture, **_synthetic_capture(weights))
    monkeypatch.setattr(
        refcheck, "_dequantized_weights", lambda model: dict(weights)
    )
    return capture, weights


def test_reference_check_computes_the_down_boundary_from_the_shard_activation(
    synthetic_capture, tmp_path, capsys
) -> None:
    """End-to-end: the per-rank exact partial is keyed on the *shard* activation.

    Regression for the captured defect: contracting the full-width activation
    against a shard-width down slice does not have the right inner dimension
    (``ffn`` vs ``ffn / ranks``) and raised ``ValueError`` before the
    down-boundary block was ever written.
    """

    capture, _weights_used = synthetic_capture
    out = tmp_path / "report.json"
    assert refcheck.main(["--capture", str(capture), "--json", str(out)]) == 0
    capsys.readouterr()

    result = json.loads(out.read_text())
    assert result["rows"] == ROWS
    assert result["hidden"] == HIDDEN
    assert result["ffn"] == FFN
    assert result["per_rank_ffn"] == FFN // RANKS

    down = result["down_boundary"]
    for index in range(RANKS):
        stat = down[f"candidate_{index}_partial_vs_exact_slice"]
        assert stat["shape"] == [ROWS, HIDDEN]
        assert stat["rel"] is not None
        # The captured partial is the bf16 rounding of *its own* exact slice, so
        # no cell may deviate from that slice by more than the local half ULP.
        # A partial keyed on the full-width activation breaks this bound.
        rounding = down[f"candidate_{index}_partial_rounding"]
        assert rounding["cells"] == ROWS * HIDDEN
        assert rounding["nonzero_cells"] > 0
        assert rounding["over_half_ulp_cells"] == 0
    assert down["teacher_ffn_down_vs_reference_bf16"]["bit_equal"] is True


def test_reference_check_separates_partial_rounding_from_reassociation(
    synthetic_capture, tmp_path, capsys
) -> None:
    """The reference isolates the bf16 partial boundary, not the schedule as a whole."""

    capture, _weights_used = synthetic_capture
    out = tmp_path / "report.json"
    assert refcheck.main(["--capture", str(capture), "--json", str(out)]) == 0
    capsys.readouterr()

    result = json.loads(out.read_text())
    boundary = result["partial_boundary"]
    assert boundary["partial_dtype"] == "bf16"
    # f32 partials differ from the full-width projection only by reassociation.
    assert boundary["f32_partials_vs_full_f32"]["rel"] < 1e-5
    # Rounding the same partials to bf16 is the larger, separate term.
    assert (
        boundary["bf16_partials_vs_f32_partials"]["max_abs"]
        > boundary["f32_partials_vs_full_f32"]["max_abs"]
    )
    assert boundary["bf16_partials_vs_f32_partials_rounding"]["cells"] == ROWS * HIDDEN


def test_reference_check_fails_closed_without_a_teacher_or_rank(tmp_path, capsys) -> None:
    weights = _weights()
    capture = tmp_path / "empty.layer0.npz"
    np.savez(capture, **{"teacher.mlp.post_norm": _bf16_bytes(np.zeros((ROWS, HIDDEN)))})
    with pytest.raises(SystemExit, match="no teacher/rank MLP fields"):
        refcheck.main(["--capture", str(capture), "--json", str(tmp_path / "r.json")])
    capsys.readouterr()
    assert weights  # weights are never reached on this path
