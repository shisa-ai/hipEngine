"""PF-1 fork (b) RED gates: grouped selected Q8_0 down prefill candidate.

Campaign context: docs/QWEN3.8-FLASH-NEXT-HALO-BOX-CAMPAIGN.md section 6.3
fork (b) — a bit-exact (T0) faster dense kernel for the coltile/selected-served
shapes. The declared candidate (worklog entry
``20260903T234843.026834Z-lhl-pf-1-forkb-declaration-a7057c``) is a grouped
selected down that serves every lane of one ``(expert, out_col)`` pair in a
single block (weight row read once per pair instead of once per lane) while
keeping, per output, the exact incumbent arithmetic:

- per-thread ``k = tid; k += blockDim.x`` strided ordered-``fmaf`` accumulation
- the wave32 ``__shfl_down`` reduce of ``reduce_block_sum``
- the serial ``wave_sums[0..waves-1]`` publication by thread 0

The registered strict fallback is ``selected_gemv_bf16_bf16_out``
(``gguf_k_selected_prefill_out_kernel<unsigned short, unsigned short, 8>``,
block-per-output). Any candidate must be bit-identical to it on every tested
shape before a runner wiring or whole-model A/B is admitted.
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
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q8_0_selected_gemv_bf16_bf16_out,
)
from hipengine.core.hip import get_hip_runtime


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    # Round-to-nearest-even BF16, matching scalar_to_float/round_to_bf16_float
    # round trips used by the production chain fixtures.
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded.astype(np.uint16)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def make_q8_0_weight_large(out_features: int, in_features: int) -> np.ndarray:
    """Vectorized Q8_0 weight fixture for large ``out_features``."""
    block = 32
    blocks = in_features // block
    rng = np.random.default_rng(2026_09_04 + out_features)
    scales = (rng.random((out_features, blocks)).astype(np.float32) * 0.05 + 0.01).astype(
        np.float16
    )
    qs = rng.integers(-128, 128, size=(out_features, blocks, block), dtype=np.int8)
    scales_u16 = scales.view(np.uint16)
    scale_bytes = scales_u16.view(np.uint8).reshape(out_features, blocks, 2)
    layout = np.zeros((out_features, blocks, 34), dtype=np.uint8)
    layout[:, :, 0] = scale_bytes[:, :, 0]
    layout[:, :, 1] = scale_bytes[:, :, 1]
    layout[:, :, 2:34] = qs.view(np.uint8)
    return np.ascontiguousarray(layout.reshape(out_features, blocks * 34))


def _build_group_map(
    selected: np.ndarray, num_experts: int
) -> tuple[np.ndarray, np.ndarray]:
    """Host-side per-expert group map: exclusive starts + sorted lane->row."""
    counts = np.bincount(selected, minlength=num_experts)
    starts = np.zeros(num_experts + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    order = np.argsort(selected, kind="stable").astype(np.int64)
    return starts, order


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_pf1_forkb_grouped_selected_down_imports() -> None:
    """RED on the unmodified path: the candidate wrapper does not exist yet."""

    from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as gemv_module

    wrapper = getattr(gemv_module, "gguf_q8_0_selected_grouped_gemv_bf16_bf16_out")
    assert callable(wrapper)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "x_rows,top_k,num_experts,in_features,out_features",
    [
        (8, 10, 16, 512, 1024),  # PF-1b selected-fixture shape
        (64, 10, 512, 640, 2560),  # production down shape (ffn->hidden, top-10)
    ],
)
def test_pf1_forkb_grouped_selected_down_bit_parity(
    x_rows: int, top_k: int, num_experts: int, in_features: int, out_features: int
) -> None:
    """The grouped candidate is bit-identical to the block-per-output owner."""

    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        gguf_q8_0_selected_grouped_gemv_bf16_bf16_out,
    )

    rows = x_rows * top_k
    rng = np.random.default_rng(2026_09_04)
    qweight = np.ascontiguousarray(
        make_q8_0_weight_large(num_experts * out_features, in_features),
        dtype=np.uint8,
    )
    x_float = (rng.standard_normal((x_rows, in_features)) * 0.5).astype(np.float32)
    x_bf16 = _f32_to_bf16_bits(x_float)
    selected = rng.integers(0, num_experts, size=rows).astype(np.int64)
    # Guarantee every expert in range is exercised unevenly (boundary case):
    # force the first num_experts lanes to cover all experts exactly once.
    selected[:num_experts] = np.arange(num_experts, dtype=np.int64)
    expert_start, lane_to_row = _build_group_map(selected, num_experts)

    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(load=True)
    bufs: list = []

    def alloc(nbytes: int):
        buf = malloc(nbytes, runtime=runtime)
        bufs.append(buf)
        return buf

    try:
        x_dev = alloc(x_bf16.nbytes)
        selected_dev = alloc(selected.nbytes)
        starts_dev = alloc(expert_start.nbytes)
        lane_to_row_dev = alloc(lane_to_row.nbytes)
        weight_dev = alloc(qweight.nbytes)
        out_owner_dev = alloc(rows * out_features * 2)
        out_grouped_dev = alloc(rows * out_features * 2)
        copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
        copy_host_to_device(selected_dev, host_array_ptr(selected), runtime=runtime)
        copy_host_to_device(starts_dev, host_array_ptr(expert_start), runtime=runtime)
        copy_host_to_device(
            lane_to_row_dev, host_array_ptr(lane_to_row), runtime=runtime
        )
        copy_host_to_device(weight_dev, host_array_ptr(qweight), runtime=runtime)

        def run_owner() -> np.ndarray:
            gguf_q8_0_selected_gemv_bf16_bf16_out(
                x_dev.ptr,
                selected_dev.ptr,
                weight_dev.ptr,
                out_owner_dev.ptr,
                x_rows,
                rows,
                num_experts,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            out = np.empty((rows, out_features), dtype=np.uint16)
            copy_device_to_host(host_array_ptr(out), out_owner_dev, runtime=runtime)
            return out

        def run_grouped() -> np.ndarray:
            gguf_q8_0_selected_grouped_gemv_bf16_bf16_out(
                x_dev.ptr,
                starts_dev.ptr,
                lane_to_row_dev.ptr,
                weight_dev.ptr,
                out_grouped_dev.ptr,
                x_rows,
                rows,
                num_experts,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            out = np.empty((rows, out_features), dtype=np.uint16)
            copy_device_to_host(host_array_ptr(out), out_grouped_dev, runtime=runtime)
            return out

        owner = run_owner()
        grouped_first = run_grouped()
        grouped_second = run_grouped()
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_array_equal(
        grouped_first, grouped_second, err_msg="grouped run-to-run determinism"
    )
    np.testing.assert_array_equal(
        grouped_first, owner, err_msg="grouped vs block-per-output owner bits"
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_pf1_forkb_runner_default_matches_strict_flag_off(monkeypatch) -> None:
    """Runner wiring gate: the fork-b grouped down default is bit-identical to
    the incumbent strict selected gemv (flag off) at the whole-MoE level.

    The fork-b kernel is bit-exact per output and everything downstream of
    ``expert_down`` in ``run_qwen4_exp_moe`` is deterministic given the same
    input bits, so the full MoE output bits must match exactly between the
    default (fork-b) and ``HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN=0`` runs.
    """

    from hipengine.core.hip import get_hip_runtime
    from tests._gguf_synthetic_weights import make_q4_k_weight, make_q8_0_weight
    from tests.test_qwen4_exp_runner_moe import (
        _dense_f32_weight,
        _download,
        _q4_weight,
        _q8_0_weight,
        _upload,
    )
    from hipengine.core.memory import free
    from hipengine.runtime.qwen4_exp_runner import (
        Qwen4ExpMoEScratch,
        run_qwen4_exp_moe,
    )

    runtime = get_hip_runtime()
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL", "1")
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_0_GROUPED", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA", raising=False)

    rng = np.random.default_rng(2026_09_04)
    hidden, ffn, experts, top_k = 256, 256, 4, 2
    rows = 16
    mixed = rng.normal(0.0, 0.1, size=(rows, hidden)).astype(np.float32)
    router = rng.normal(0.0, 0.1, size=(experts, hidden)).astype(np.float32)
    gate_raw = np.stack([make_q4_k_weight(ffn, hidden) for _ in range(experts)])
    up_raw = np.stack([make_q4_k_weight(ffn, hidden) for _ in range(experts)])
    down_raw = np.stack([make_q8_0_weight(hidden, ffn) for _ in range(experts)])
    shared_gate = rng.normal(0.0, 0.1, size=(ffn, hidden)).astype(np.float32)
    shared_up = rng.normal(0.0, 0.1, size=(ffn, hidden)).astype(np.float32)
    shared_down = rng.normal(0.0, 0.1, size=(hidden, ffn)).astype(np.float32)
    shared_scalar = rng.normal(0.0, 0.1, size=(hidden,)).astype(np.float32)

    def run(flag_value: str) -> np.ndarray:
        monkeypatch.setenv("HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN", flag_value)
        allocations = []
        scratch = None
        try:
            d_mixed = _upload(mixed, runtime, allocations)
            weights = {
                "router": _dense_f32_weight("router", router, runtime, allocations),
                "expert_gate": _q4_weight("expert_gate", gate_raw, runtime, allocations),
                "expert_up": _q4_weight("expert_up", up_raw, runtime, allocations),
                "expert_down": _q8_0_weight("expert_down", down_raw, runtime, allocations),
                "shared_gate": _dense_f32_weight("shared_gate", shared_gate, runtime, allocations),
                "shared_up": _dense_f32_weight("shared_up", shared_up, runtime, allocations),
                "shared_down": _dense_f32_weight("shared_down", shared_down, runtime, allocations),
                "shared_gate_weight": _dense_f32_weight(
                    "shared_gate_weight", shared_scalar.reshape(1, hidden), runtime, allocations,
                ),
            }
            scratch = Qwen4ExpMoEScratch.allocate(
                rows=rows, hidden=hidden, ffn=ffn, experts=experts, top_k=top_k,
                runtime=runtime,
            )
            result = run_qwen4_exp_moe(
                d_mixed.ptr,
                weights,
                scratch=scratch,
                rows=rows,
                hidden=hidden,
                ffn=ffn,
                experts=experts,
                top_k=top_k,
                runtime=runtime,
            )
            runtime.device_synchronize()
            return _download(result.output, (rows, hidden), np.uint16, runtime)
        finally:
            if scratch is not None:
                scratch.close()
            for allocation in reversed(allocations):
                free(allocation, runtime=runtime)

    default_bits = run("1")
    strict_bits = run("0")
    np.testing.assert_array_equal(
        default_bits,
        strict_bits,
        err_msg="fork-b default MoE output bits vs FORKB_GROUPED_DOWN=0 strict",
    )
