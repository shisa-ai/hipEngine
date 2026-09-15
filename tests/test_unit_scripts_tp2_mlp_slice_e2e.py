"""Tests for scripts/tp2_mlp_slice_e2e.py (TP2-A).

The host oracle is checked without any GPU: the bf16 contract emulation must
reproduce the float64 truth through the split, which is the property the device
run is measured against. The end-to-end run needs two gfx1100 devices and the
real Q4_K_M artifact, and is skipped elsewhere.
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tp2_mlp_slice_e2e.py"
GGUF_PATH = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location("tp2_mlp_slice_e2e", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    try:
        from hipengine.core.hip import get_hip_runtime  # noqa: PLC0415

        return get_hip_runtime().device_count() >= 2
    except Exception:  # noqa: BLE001 - no ROCm runtime in this environment
        return False


def _gguf_available() -> bool:
    return Path(GGUF_PATH).is_file()


# -- host oracle ---------------------------------------------------------------


def test_silu_matches_the_stable_definition(mod) -> None:
    x = np.array([0.0, 1.0, -1.0, 20.0, -20.0], dtype=np.float32)
    expected = x / (1.0 + np.exp(-x))
    got = mod.silu_f32(x)
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7)
    # Large negative inputs saturate towards zero, not NaN; the residual is
    # f32 underflow of the sigmoid, far below any bf16 representation.
    assert got[-1] == expected[-1]
    assert abs(float(got[-1])) < 1e-7


def test_bf16_round_is_round_to_nearest_even(mod) -> None:
    # 1.0 is exact; the value just above 1.0's bf16 grid rounds down; a tie
    # rounds to even.
    values = np.array([1.0, 1.00390625, 1.005859375, 1.0078125], dtype=np.float32)
    got = mod.bf16_round(values)
    assert got[0] == np.float32(1.0)
    assert got[1] == np.float32(1.0)
    assert got[3] == np.float32(1.0078125)
    # Every output is exactly representable in bf16.
    bits = mod.f32_to_bf16_bits(got)
    back = mod.bf16_to_float32(bits)
    assert np.array_equal(got, back)


def test_the_split_contract_reproduces_the_truth(mod) -> None:
    """The oracle's own consistency: summed rank partials equal the full chain.

    This is the property that makes the device comparison meaningful: the
    contract emulation and the truth chain are two different computations of the
    same math, and their disagreement bounds what the device can be blamed for.
    """

    rng = np.random.default_rng(7)
    hidden, ffn, world_size = 32, 48, 2
    gate = rng.standard_normal((ffn, hidden), dtype=np.float32) * 0.1
    up = rng.standard_normal((ffn, hidden), dtype=np.float32) * 0.1
    down = rng.standard_normal((hidden, ffn), dtype=np.float32) * 0.1
    x_f32 = rng.standard_normal(hidden, dtype=np.float32)
    x_bf16 = mod.bf16_round(x_f32)

    truth = mod.full_truth(gate, up, down, x_bf16)
    partials = [
        mod.contract_rank_partials(gate, up, down, x_bf16, rank, world_size)
        for rank in range(world_size)
    ]
    summed = np.sum([p["down_partial"] for p in partials], axis=0, dtype=np.float32)

    metrics = mod._relative_errors(summed, truth)
    # Both sides round to bf16 at the same boundaries, so the mean agrees well
    # inside a bf16 ulp; the max sits on one small-magnitude element where the
    # relative floor dominates, which is why the mean is the tight gate here.
    # The 32-wide fixture has a small output dynamic range, so each bf16
    # rounding of the intermediates is a visible fraction of a partial; two
    # bf16 epsilons is the honest ceiling for this fixture's mean.
    assert metrics["mean_rel_err"] < 2 * 2**-8 * 2
    assert metrics["max_rel_err"] < 5e-2


def test_the_contract_partial_widths_follow_the_split(mod) -> None:
    rng = np.random.default_rng(11)
    hidden, ffn, world_size = 16, 32, 2
    gate = rng.standard_normal((ffn, hidden), dtype=np.float32)
    up = rng.standard_normal((ffn, hidden), dtype=np.float32)
    down = rng.standard_normal((hidden, ffn), dtype=np.float32)
    x = mod.bf16_round(rng.standard_normal(hidden, dtype=np.float32))
    for rank in range(world_size):
        partials = mod.contract_rank_partials(gate, up, down, x, rank, world_size)
        assert partials["gate"].shape == (ffn // world_size,)
        assert partials["down_partial"].shape == (hidden,), (
            "a down partial is full-width: the reduction is a plain sum"
        )


def test_f32_partials_skip_the_final_bf16_rounding(mod) -> None:
    """The reduction dtype decision: an f32 partial is the unrounded accumulator."""

    rng = np.random.default_rng(5)
    hidden, ffn, world_size = 16, 32, 2
    gate = rng.standard_normal((ffn, hidden), dtype=np.float32)
    up = rng.standard_normal((ffn, hidden), dtype=np.float32)
    down = rng.standard_normal((hidden, ffn), dtype=np.float32)
    x = mod.bf16_round(rng.standard_normal(hidden, dtype=np.float32))
    bf16_partial = mod.contract_rank_partials(
        gate, up, down, x, 0, world_size, down_output_dtype="bf16"
    )["down_partial"]
    f32_partial = mod.contract_rank_partials(
        gate, up, down, x, 0, world_size, down_output_dtype="f32"
    )["down_partial"]
    assert not np.array_equal(bf16_partial, f32_partial), (
        "the two dtypes are different contracts; a test that conflates them checks nothing"
    )
    # The f32 partial is the exact f32 matmul: no representable value was lost.
    exact = (down[:, : ffn // world_size] @ mod.contract_rank_partials(
        gate, up, down, x, 0, world_size, down_output_dtype="f32"
    )["activated"]).astype(np.float32)
    assert np.array_equal(f32_partial, exact)
    # The bf16 partial is that value rounded to bf16.
    assert np.array_equal(bf16_partial, mod.bf16_round(exact))


def test_the_boundary_adds_the_residual_and_normalizes(mod) -> None:
    """next = residual + mlp_out in f32, then the next block's input RMSNorm."""

    from hipengine.kernels.cpu_reference.ops import rmsnorm

    rng = np.random.default_rng(13)
    n = 64
    residual = rng.standard_normal(n, dtype=np.float32) * 0.1
    mlp_out = rng.standard_normal(n, dtype=np.float32) * 0.1
    weight = rng.standard_normal(n, dtype=np.float32) * 0.1 + 1.0
    out = mod.residual_boundary(residual, mlp_out, weight)
    expected_next = (residual + mlp_out).astype(np.float32)
    assert np.array_equal(out["next_hidden"], expected_next)
    assert np.allclose(out["next_norm"], rmsnorm(expected_next, weight), rtol=1e-6)


def test_an_uneven_ffn_is_refused(mod) -> None:
    rng = np.random.default_rng(3)
    gate = rng.standard_normal((7, 4), dtype=np.float32)
    up = rng.standard_normal((7, 4), dtype=np.float32)
    down = rng.standard_normal((4, 7), dtype=np.float32)
    x = mod.bf16_round(rng.standard_normal(4, dtype=np.float32))
    with pytest.raises(ValueError, match="does not split"):
        mod.contract_rank_partials(gate, up, down, x, 0, 2)


def test_relative_errors_floor_near_zero_values(mod) -> None:
    actual = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    expected = np.array([1e-6, 1.0, 1.0], dtype=np.float32)
    metrics = mod._relative_errors(actual, expected)
    # The near-zero element's absolute difference is floored, not exploded.
    assert metrics["max_rel_err"] == pytest.approx(1.0)
    assert metrics["max_abs_err"] == pytest.approx(1.0)


# -- end to end (two gfx1100 devices + the real artifact) ----------------------


@pytest.mark.skipif(
    not _hip_available() or not _gguf_available(),
    reason="two gfx1100 devices or the Q4_K_M artifact not available",
)
def test_the_two_gpu_slice_matches_the_oracle_and_the_teacher(mod) -> None:
    report = mod.run(model=Path(GGUF_PATH), layer=0, world_size=2)

    assert report["per_rank_ffn"] == report["feed_forward_length"] // 2
    assert len(report["per_rank"]) == 2
    assert [r["device"] for r in report["per_rank"]] == [0, 1], (
        "each rank must run on its own device"
    )

    # Partials are f32 and land on the f32 contract emulation; the activated
    # intermediate is bf16 and lands on its bf16 contract.
    assert report["reduction"]["down_output_dtype"] == "f32"
    for rank in report["per_rank"]:
        assert rank["down_partial_vs_contract"]["max_abs_err"] < 1e-6, rank
        assert rank["activated_vs_contract"]["max_rel_err"] < 4e-3, rank

    stages = report["stages"]
    # The remaining distance to the f64 truth is the bf16 activation contract,
    # which the TP1 teacher carries identically.
    assert stages["tp2_sum_vs_truth"]["mean_rel_err"] < 5e-3
    assert stages["tp2_sum_vs_truth"]["max_rel_err"] < 3e-2
    # With f32 partials, TP2 and TP1 agree to f32 accumulation noise.
    assert stages["tp2_sum_vs_tp1_teacher"]["max_abs_err"] < 1e-6
    assert stages["tp2_sum_vs_tp1_teacher"]["mean_rel_err"] < 1e-6
    # The exchange delivered the same reduced vector to every rank.
    for verified in report["reduction"]["exchange"]["verified_ranks"]:
        assert verified["h2d_roundtrip_max_abs"] == 0.0
    # The measured walls and the exchange profile are present with the fields
    # the transport decision needs; no timing gate here, walls vary by run.
    walls = report["segment_walls"]
    assert walls["tp1"]["step_us_p50"] > 0
    assert walls["tp2_unfused"]["step_us_p50"] > 0
    assert walls["tp2_fused"] is not None and walls["tp2_fused"]["step_us_p50"] > 0
    profile = report["reduction"]["exchange_profile"]
    assert profile["gather_p50_us"] > 0
    assert profile["full_p50_us"] > profile["gather_p50_us"], (
        "the return path must cost something; a profile showing otherwise is mislabelled"
    )
    assert profile["return_path_us_p50"] == (
        profile["full_p50_us"] - profile["gather_p50_us"]
    )
    # The residual/next-consumer boundary matches TP1 at f32 noise and stays
    # inside the bf16 activation envelope against the f64 truth.
    assert stages["boundary_next_hidden_tp2_vs_tp1"]["max_abs_err"] < 1e-7
    assert stages["boundary_next_norm_tp2_vs_tp1"]["mean_rel_err"] < 1e-6
    assert stages["boundary_next_norm_tp2_vs_truth"]["mean_rel_err"] < 2e-3

    # The fused candidate runs at the shard shape and is bit-exact with the
    # unfused baseline on both ranks; it stays a candidate - the policy table
    # is not touched by this run.
    fused = report["fused_candidate"]
    assert fused["policy_admitted"] is False
    assert fused["shape"] == [1, report["hidden_size"], report["per_rank_ffn"]]
    assert fused["sum_vs_unfused_sum"]["max_abs_err"] == 0.0
    for rank in fused["per_rank"]:
        assert rank["activated_vs_unfused"]["max_abs_err"] == 0.0
        assert rank["down_partial_vs_unfused"]["max_abs_err"] == 0.0
