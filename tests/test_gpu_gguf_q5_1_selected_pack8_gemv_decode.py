"""Correctness fixtures for the selected GGUF Q5_1 pack8 GEMV decode.

Mirrors ``test_gpu_gguf_k_selected_pack8_gemv_decode.py`` for raw Q5_1
selected (down) experts — the quant the Gemma 4 26B-A4B MoE down projection
uses (``ffn_down_exps``, GGUF type Q5_1, 32-element / 24-byte blocks).
Gemma 4's FFN width is 704 — a multiple of 32 but not of 256 — so this
kernel and its wrapper deliberately accept ``in_features % 32 == 0``.

Three surfaces:

* no-GPU: registry keys, build plan, wrapper validation;
* GPU compact vs CPU oracle (``gguf_quant_gemv`` with ``Q5_1``);
* GPU compact vs the retained legacy per-row selected GEMV
  (``qwen4_exp_q5_1_selected_gemv_bf16_bf16_out``) on identical inputs —
  both dequantize ``w = d*q5 + m``; only summation order differs.

RED first: none of these existed before the Q5_1 compact kernel was ported.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q5_1_selected_pack8_gemv import (
    build_gguf_q5_1_selected_pack8_gemv,
    gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out,
    plan_gguf_q5_1_selected_pack8_gemv_build,
    register_gguf_q5_1_selected_pack8_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
    build_qwen4_exp_q5_1,
    qwen4_exp_q5_1_selected_gemv_bf16_bf16_out,
)
from hipengine.kernels.registry import resolve
from hipengine.loading.gguf_selected_contract import selected_allocation
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.runtime.qwen35_gguf_runner import _COMPACT_MOE_DOWN_GEMV_KEYS
from tests._gguf_synthetic_weights import make_q5_1_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def q5_1_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q5_1_selected_pack8_gemv(load=True)


@pytest.fixture(scope="module")
def legacy_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_qwen4_exp_q5_1(load=True)


# ---------------------------------------------------------------------------
# No-GPU surface.
# ---------------------------------------------------------------------------


def test_q5_1_compact_registry_keys_resolve() -> None:
    register_gguf_q5_1_selected_pack8_gemv_kernels()
    for variant in (
        "selected_pack8_gemv_decode_compact_bf16_bf16_out",
        "selected_pack8_gemv_decode_bf16_bf16_out",
    ):
        fn = resolve(backend="hip_gfx1100", layer="moe_linear", quant="gguf_q5_1", variant=variant)
        assert fn is not None, f"missing registry entry: gguf_q5_1 / {variant}"


def test_q5_1_compact_plan_key_present() -> None:
    key = _COMPACT_MOE_DOWN_GEMV_KEYS.get("gguf_q5_1")
    assert key is not None, "gguf_q5_1 missing from _COMPACT_MOE_DOWN_GEMV_KEYS"
    assert key.quant == "gguf_q5_1"
    assert key.variant == "selected_pack8_gemv_decode_compact_bf16_bf16_out"


def test_q5_1_selected_allocation_is_raw() -> None:
    # The compact plan resolves down_allocation through this table; a miss
    # raises ValueError at first MoE layer, so it is a launch contract.
    assert selected_allocation("gguf_q5_1") == "raw"


def test_q5_1_build_plan_is_dry_run_safe() -> None:
    plan = plan_gguf_q5_1_selected_pack8_gemv_build()
    assert plan.output_path.name == "gguf_q5_1_selected_pack8_gemv.so"


def test_q5_1_wrapper_validates_args() -> None:
    with pytest.raises(ValueError, match="compact_rows must be positive"):
        gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out(0, 0, 0, 0, 0, 704, 8, 1)
    with pytest.raises(ValueError, match="in_features must be divisible by GGUF Q5_1 block size 32"):
        gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out(0, 0, 0, 0, 1, 703, 8, 1)
    with pytest.raises(ValueError, match=r"out_features must be a multiple of 8 \(pack8 lane\)"):
        gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out(0, 0, 0, 0, 1, 704, 9, 1)


# ---------------------------------------------------------------------------
# Correctness vs CPU oracle.
# ---------------------------------------------------------------------------


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    lsb = (u32 >> 16) & 1
    rounded = ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)
    rounded[nan_mask] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _stack_experts(out_features: int, in_features: int, num_experts: int, seed: int) -> np.ndarray:
    base = make_q5_1_weight(out_features, in_features)
    return np.stack([np.roll(base, shift=e + seed, axis=0) for e in range(num_experts)], axis=0)


def _expected_single(
    x_ref: np.ndarray,
    expert_start: np.ndarray,
    qw: np.ndarray,
    out_features: int,
) -> np.ndarray:
    compact_rows = int(expert_start[-1])
    out = np.zeros((compact_rows, out_features), dtype=np.float32)
    for e in range(len(expert_start) - 1):
        s, sl = int(expert_start[e]), int(expert_start[e + 1])
        if sl == s:
            continue
        out[s:sl] = gguf_quant_gemv(x_ref[s:sl], qw[e], GGMLQuantizationType.Q5_1)
    return out


def _alloc(arr: np.ndarray):
    buf = malloc(arr.nbytes)
    copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)
    return buf


def _run_compact(
    x_bf16: np.ndarray,
    expert_start: np.ndarray,
    qw: np.ndarray,
    out_features: int,
    library,
) -> np.ndarray:
    compact_rows = int(expert_start[-1])
    in_features = x_bf16.shape[1]
    x_buf = _alloc(x_bf16)
    es_buf = _alloc(expert_start)
    w_buf = _alloc(qw)
    out_arr = np.zeros((compact_rows, out_features), dtype=np.uint16)
    out_buf = malloc(out_arr.nbytes)
    try:
        gguf_q5_1_selected_pack8_gemv_decode_compact_bf16_bf16_out(
            x_buf.ptr, es_buf.ptr, w_buf.ptr, out_buf.ptr,
            compact_rows, in_features, out_features, qw.shape[0],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (x_buf, es_buf, w_buf, out_buf):
            free(b)


def _run_legacy(
    x_bf16: np.ndarray,
    expert_start: np.ndarray,
    qw: np.ndarray,
    out_features: int,
    library,
) -> np.ndarray:
    """Run the retained legacy per-row selected GEMV on identical inputs."""
    compact_rows = int(expert_start[-1])
    in_features = x_bf16.shape[1]
    selected = np.concatenate(
        [
            np.full(int(expert_start[e + 1]) - int(expert_start[e]), e, dtype=np.int64)
            for e in range(len(expert_start) - 1)
        ]
    )
    x_buf = _alloc(x_bf16)
    sel_buf = _alloc(selected)
    w_buf = _alloc(qw)
    out_arr = np.zeros((compact_rows, out_features), dtype=np.uint16)
    out_buf = malloc(out_arr.nbytes)
    try:
        qwen4_exp_q5_1_selected_gemv_bf16_bf16_out(
            x_buf.ptr, sel_buf.ptr, w_buf.ptr, out_buf.ptr,
            compact_rows, compact_rows, qw.shape[0], in_features, out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (x_buf, sel_buf, w_buf, out_buf):
            free(b)


_TOL = dict(atol=1.0e-3, rtol=1.0e-2)

_EXPERT_LAYOUTS = [
    pytest.param([8], id="single-expert"),
    pytest.param([1], id="single-row"),
    pytest.param([3, 5], id="two-uneven"),
    pytest.param([0, 8], id="empty-start"),
    pytest.param([4, 0, 4], id="empty-middle"),
    pytest.param([8, 0], id="empty-tail"),
]

# 704 is the Gemma 4 MoE FFN width (in_features of ffn_down_exps); 768 and
# 1024 are qwen35-shaped controls whose widths are multiples of 256.
_SHAPES = [
    pytest.param(704, 64, id="gemma4-ffn-704"),
    pytest.param(704, 2816, id="gemma4-down-full"),
    pytest.param(768, 64, id="control-768"),
    pytest.param(1024, 8, id="control-1024x8"),
]


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("counts", _EXPERT_LAYOUTS)
@pytest.mark.parametrize("in_features,out_features", _SHAPES)
def test_q5_1_compact_matches_cpu_oracle(
    counts: list[int],
    in_features: int,
    out_features: int,
    q5_1_library,
) -> None:
    num_experts = len(counts)
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    compact_rows = int(expert_start[-1])
    rng = np.random.default_rng(num_experts * 401 + in_features + out_features)
    qw = _stack_experts(out_features, in_features, num_experts, seed=7)
    x = rng.normal(0.0, 0.3, size=(compact_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    actual = _run_compact(x_bf16, expert_start, qw, out_features, q5_1_library)
    actual_f32 = _bf16_u16_to_f32(actual)
    expected = _expected_single(x_ref, expert_start, qw, out_features)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(actual_f32, expected_bf16, **_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("counts", _EXPERT_LAYOUTS[:4])
@pytest.mark.parametrize(
    "in_features,out_features",
    [
        pytest.param(704, 64, id="gemma4-ffn-704"),
        pytest.param(768, 512, id="control-768x512"),
    ],
)
def test_q5_1_compact_matches_legacy_selected(
    counts: list[int],
    in_features: int,
    out_features: int,
    q5_1_library,
    legacy_library,
) -> None:
    """Compact pack8 vs the retained legacy per-row selected GEMV, bf16-bf16.

    Both dequantize Q5_1 identically (``w = d*q5 + m``); only summation order
    and the reduction tree differ, so bf16 outputs must agree within the same
    tolerance as the CPU-oracle comparison.
    """
    num_experts = len(counts)
    expert_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    compact_rows = int(expert_start[-1])
    rng = np.random.default_rng(num_experts * 911 + in_features * 3 + out_features)
    qw = _stack_experts(out_features, in_features, num_experts, seed=13)
    x = rng.normal(0.0, 0.3, size=(compact_rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    compact = _run_compact(x_bf16, expert_start, qw, out_features, q5_1_library)
    legacy = _run_legacy(x_bf16, expert_start, qw, out_features, legacy_library)
    np.testing.assert_allclose(
        _bf16_u16_to_f32(compact), _bf16_u16_to_f32(legacy), **_TOL
    )