"""Scope, fallback and ordered-tree checks for the GR DPP candidate."""

from pathlib import Path
import re

import numpy as np
import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner

from scripts.qwen4exp_halo_box_campaign_ab import (
    _apply_mode, build_parser, gr_dpp_expected_calls,
)
from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode

PARENT = "coltile2_branch4_rowbatch4_wave_scale_f32_exact"
DPP = "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact"
K320_FULL = "coltile2_branch4_rowbatch4_wave_scale_dpp_k320_full_rows_f32_exact"


def test_dpp_route_is_opt_in_and_keeps_scope(monkeypatch):
    base = KernelKey("hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
                     "coltile2_branch4_rowbatch4_f32_exact")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE", "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION", "0")
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    assert runner._qwen4_exp_gr_wave_scale_key(base, rows=512, branches=4).variant == PARENT
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION", "1")
    assert runner._qwen4_exp_gr_wave_scale_key(base, rows=257, branches=4).variant == DPP
    assert runner._qwen4_exp_gr_wave_scale_key(base, rows=256, branches=4) == base
    assert runner._qwen4_exp_gr_wave_scale_key(base, rows=512, branches=2) == base
    monkeypatch.setattr(runner, "is_registered", lambda key: key.variant != DPP)
    assert runner._qwen4_exp_gr_wave_scale_key(base, rows=512, branches=4).variant == PARENT
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert runner._qwen4_exp_gr_wave_scale_key(base, rows=512, branches=4) == base
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS", "0")
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    assert runner._qwen4_exp_gr_k320_full_rows_key(
        KernelKey(base.backend, base.layer, base.quant, DPP),
        rows=512, branches=4, low_rank=320,
    ).variant == DPP
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS", "1")
    assert runner._qwen4_exp_gr_k320_full_rows_key(
        KernelKey(base.backend, base.layer, base.quant, DPP),
        rows=512, branches=4, low_rank=320
    ).variant == K320_FULL
    assert runner._qwen4_exp_gr_k320_full_rows_key(
        KernelKey(base.backend, base.layer, base.quant, DPP),
        rows=1024, branches=4, low_rank=320
    ).variant == K320_FULL
    assert runner._qwen4_exp_gr_k320_full_rows_key(
        KernelKey(base.backend, base.layer, base.quant, DPP),
        rows=4096, branches=4, low_rank=320
    ).variant == K320_FULL
    for rows, branches, low_rank in ((257, 4, 320), (512, 2, 320), (512, 4, 321)):
        assert runner._qwen4_exp_gr_k320_full_rows_key(
            KernelKey(base.backend, base.layer, base.quant, DPP),
            rows=rows, branches=branches, low_rank=low_rank
        ) == KernelKey(base.backend, base.layer, base.quant, DPP)
    monkeypatch.setattr(runner, "is_registered", lambda key: key.variant != K320_FULL)
    assert runner._qwen4_exp_gr_k320_full_rows_key(
        KernelKey(base.backend, base.layer, base.quant, DPP),
        rows=512, branches=4, low_rank=320
    ).variant == DPP


def test_k320_route_is_default_off(monkeypatch):
    base = KernelKey("hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
                     "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS", "0")
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    assert runner._qwen4_exp_gr_k320_full_rows_key(
        base, rows=512, branches=4, low_rank=320
    ).variant == base.variant


@pytest.mark.parametrize("mode,value", [("before", "0"), ("after", "1")])
def test_benchmark_changes_only_dpp_flag(mode, value):
    env = {"HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE": "1", "other": "untouched"}
    _apply_mode(mode, route_package="gr-dpp-reduction", environment=env)
    assert env == {"HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE": "1", "other": "untouched",
                   "HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION": value}
    apply_state_gate_mode(None, "gr-dpp-reduction", value,
                          "HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION", environment=env)
    assert env["HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION"] == value
    parsed = build_parser().parse_args([
        "--route-package", "gr-dpp-reduction", "--model-root", "/unused",
        "--output", "/unused.json", "--prefill-chunk-size", "4096",
    ])
    assert parsed.route_package == "gr-dpp-reduction"


@pytest.mark.parametrize("tokens,chunk,expected", [
    (512, 512, 96), (1024, 512, 192), (4096, 512, 768),
    (4096, 1024, 384), (4096, 4096, 96), (512, 4096, 96),
    (4097, 4096, 96), (4353, 4096, 192), (512, 256, 0),
])
def test_gr_dpp_engagement_counts_scope(tokens, chunk, expected):
    assert gr_dpp_expected_calls(tokens, chunk) == expected


def test_helper_publishes_the_same_ordered_lane_zero_tree():
    """Check actual helper controls, not just an independently invented tree."""
    source = (Path(__file__).parents[1] / "hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip").read_text()
    start = source.index("__device__ inline void h7c_reduce_wave_accumulators_dpp(",
                         source.index("__device__ inline float h7c_dpp_add_row_shl1_f32"))
    body = source[start:source.index("template <typename out_t", start)]
    controls = [int(c, 16) for c in re.findall(r"h7c_dpp_move_f32<(0x[0-9a-fA-F]+)>", body)]
    assert controls == [0x108, 0x104, 0x102]  # row_shl, not row_shr.
    assert "h7c_permlanex16_f32" in body
    assert "h7c_dpp_add_row_shl1_f32" in body
    a = list(range(32))
    for offset in (16, 8, 4, 2, 1):
        old = a
        a = [(v, old[i + offset] if i + offset < 32 else v) for i, v in enumerate(old)]
    b = [(i, i ^ 16) for i in range(32)]
    for offset in [c - 0x100 for c in controls] + [1]:
        old = b
        b = [(v, old[i + offset] if (i % 16) + offset < 16 else 0)
             for i, v in enumerate(old)]
    assert a[0] == b[0]


@pytest.mark.parametrize("seed", [11, 20261005])
def test_ordered_dpp_lane_zero_bits(seed):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((1000, 32)) * np.exp2(
        rng.integers(-15, 15, (1000, 32)))).astype(np.float32)
    x[0] = 0
    x[1] = np.resize(np.array([1e20, 1, -1e20, -1], np.float32), 32)
    shuffle = x.copy()
    for offset in (16, 8, 4, 2, 1):
        old = shuffle.copy()
        shuffle[:, :32 - offset] += old[:, offset:]
    dpp = x + x[:, np.arange(32) ^ 16]
    for offset in (8, 4, 2, 1):
        old = dpp.copy()
        lanes = np.arange(32)
        live = (lanes % 16) + offset < 16
        dpp[:, live] += old[:, lanes[live] + offset]
    np.testing.assert_array_equal(shuffle[:, 0].view(np.uint32), dpp[:, 0].view(np.uint32))
