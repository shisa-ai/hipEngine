"""PF-3 RED tests: MoE expert GEMM schedule retunes (M1 128-wide tile family).

Declaration entry:
``worklog/entries/20260903T170430.578452Z-lhl-pf-3-moe-expert-gemm-schedules-declaration-arith-195111.md``

- **Arithmetic class:** T0 for both levers, with the PF-5 T1 escalation stop
  rule. The candidates retune tile/schedule geometry only (halo-box
  ``a7ad7b7f`` ``mmq-config-rdna3-5.cuh`` M1 family: 128-thread blocks, I=64
  dst rows, Q8_1 SRAM layout, ``MMQ_ITER_K``; Q4_K config rows 116-127 and
  Q5_1 rows 66-77). The per-thread FP32 accumulation sequence and the strict
  128-thread reduction tree are preserved, so the binding contract is
  **bit-for-bit** parity against the incumbent production owners. If any
  implementation detail forces a reassociation (split-K, WMMA accumulate
  reordering, changed partial-softmax merge), the class escalates to T1 and
  the unit stops for the PF-0-based section 6.2 packet before any promotion.
- **Registered strict fallbacks (named per loop contract):** the incumbent
  owners stay registered, production, and untouched:
  - Q4_K gate/up:
    ``gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out``
    (``KernelKey("hip_gfx1100", "moe_linear", "gguf_q4_k",
    "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out")``)
    plus ``silu_mul_separate_out_bf16``.
  - Q5_1 down:
    ``qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out``
    (``KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1",
    "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out")``).

RED semantics (unmodified path): the candidate functions
``gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_m1_bf16_bf16_out``
and
``qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out``
do not exist yet, so every test below fails with ``ImportError`` before any
GPU work runs. They must pass after the M1 variants land under their NEW
four-axis variant keys with the incumbents still resolving as production.
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
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import GGMLQuantizationType, bf16_to_float32

_Q4_K_IN_FEATURES = 2560
_Q4_K_OUT_FEATURES = 640
_Q5_1_IN_FEATURES = 640
_Q5_1_OUT_FEATURES = 2560
_NUM_EXPERTS = 64

# Family tolerance for the CPU-reference sanity checks (matches the Q5_1
# selected family tests). The binding T0 contract is candidate-vs-incumbent
# bit equality; the CPU check only guards against shared fixture misencoding.
_CPU_RTOL = 2e-2
_CPU_ATOL = 2e-2


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _expert_counts(rows: int, seed: int) -> np.ndarray:
    """Deterministic uneven expert row counts (with empty experts) summing to rows."""
    rng = np.random.default_rng(seed)
    if rows >= _NUM_EXPERTS:
        raw = rng.integers(0, 24, size=_NUM_EXPERTS).astype(np.int64)
        total = int(raw.sum())
        counts = raw * rows // total
        counts[0] += rows - int(counts.sum())
    else:
        # Fewer rows than experts: one row on a deterministic expert subset.
        counts = np.zeros(_NUM_EXPERTS, dtype=np.int64)
        counts[rng.permutation(_NUM_EXPERTS)[:rows]] = 1
    assert int(counts.sum()) == rows
    return counts


def _expert_start(counts: np.ndarray) -> np.ndarray:
    start = np.zeros(len(counts) + 1, dtype=np.int64)
    start[1:] = np.cumsum(counts)
    return start


_Q4_K_BASE_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _base_q4_k_weight(out_features: int, in_features: int, seed: int) -> np.ndarray:
    key = (out_features, in_features, seed)
    if key not in _Q4_K_BASE_CACHE:
        # The int64-indexed helper is the documented large-shape equivalent of
        # the legacy uint8-indexed generator, which overflows for
        # out_features > 127 (production Q4_K gate/up rows = 640).
        from tests._gguf_synthetic_weights import make_q4_k_weight

        _Q4_K_BASE_CACHE[key] = make_q4_k_weight(out_features, in_features)
    return _Q4_K_BASE_CACHE[key]


def _make_expert_q4_k_weights(
    *, num_experts: int, out_features: int, in_features: int, seed: int
) -> np.ndarray:
    base = _base_q4_k_weight(out_features, in_features, seed)
    return np.ascontiguousarray(
        np.stack(
            [
                np.roll(base, shift=seed + expert, axis=0)
                for expert in range(num_experts)
            ],
            axis=0,
        )
    )


def _make_expert_q5_1_weights(
    *, num_experts: int, out_features: int, in_features: int, seed: int
) -> np.ndarray:
    """Q5_1 blocks of 24 bytes: d(f16), dmin(f16), qh(4), qs(16) per 32 elements."""
    rng = np.random.default_rng(seed)
    blocks_per_row = in_features // 32
    d = rng.uniform(0.01, 0.06, size=(num_experts, out_features, blocks_per_row)).astype(
        np.float16
    )
    dmin = rng.uniform(
        -0.06, 0.01, size=(num_experts, out_features, blocks_per_row)
    ).astype(np.float16)
    qh = rng.integers(0, 256, size=(num_experts, out_features, blocks_per_row, 4), dtype=np.uint8)
    qs = rng.integers(0, 256, size=(num_experts, out_features, blocks_per_row, 16), dtype=np.uint8)
    raw = np.concatenate(
        [d.view(np.uint8).reshape(*d.shape, 2),
         dmin.view(np.uint8).reshape(*dmin.shape, 2), qh, qs], axis=-1
    )
    assert raw.shape == (num_experts, out_features, blocks_per_row, 24)
    return np.ascontiguousarray(raw.reshape(num_experts, out_features, blocks_per_row * 24))


def _make_activation(rows: int, in_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """BF16 activations; returns (bits uint16, decoded f32)."""
    rng = np.random.default_rng(seed + 1)
    values = rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    bits = float_array_to_bf16_bits(values)
    return bits, bf16_to_float32(bits)


def test_pf3_m1_variant_keys_registry_contract() -> None:
    """Both M1 candidate variant keys must resolve to the candidate functions.

    RED on the unmodified path: neither candidate function exists yet, so the
    imports fail before any GPU work runs. The resolver must map each NEW
    variant key to its own candidate function (never to the incumbent owner
    or any broadened fallback), while the incumbent keys stay bound to the
    production owners.
    """
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
        gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out,
        gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_m1_bf16_bf16_out,
        register_gguf_q4_k_selected_prefill_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out,
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out,
        register_qwen4_exp_q5_1_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_gguf_q4_k_selected_prefill_kernels()
    register_qwen4_exp_q5_1_kernels()

    # New candidate keys resolve to the candidate functions only.
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_grouped_rowbatch8_out4_expertgrid64_m1_bf16_bf16_out",
            missing="none",
        )
        is gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_m1_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q5_1",
            variant="selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out",
            missing="none",
        )
        is qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out
    )

    # The incumbent production keys stay bound to the untouched owners.
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            missing="none",
        )
        is gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q5_1",
            variant="selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            missing="none",
        )
        is qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out
    )


def _q4_k_reference_per_expert(
    x_ref: np.ndarray, qweights: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    from hipengine.kernels.cpu_reference import gguf_quant_gemv

    rows = x_ref.shape[0]
    out_features = qweights.shape[1]
    reference = np.zeros((rows, out_features), dtype=np.float32)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        stop = start + int(count)
        reference[start:stop] = gguf_quant_gemv(
            x_ref[start:stop], qweights[expert], GGMLQuantizationType.Q4_K
        )
        start = stop
    return reference


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [16, 64, 512])
def test_pf3_q4_k_gate_up_m1_bit_exact_vs_incumbent(rows: int) -> None:
    """Q4_K gate/up M1 candidate must be bit-for-bit vs the incumbent owner (T0).

    Binding shape: rows=512 chunk (compact MoE rows, uneven expert counts with
    empty experts); anchor shapes: rows=16 and rows=64. Production widths:
    in_features=2560 (hidden), gate/up out_features=640 each (ffn). Verified
    from GGUF metadata + tensor shapes of the local model files.

    RED on the unmodified path: the candidate function does not exist, so the
    import fails before any GPU work runs.
    """
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
        build_gguf_q4_k_selected_prefill,
        gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out,
        gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_m1_bf16_bf16_out,
    )

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_selected_prefill(load=True)
    counts = _expert_counts(rows, seed=7000 + rows)
    num_experts = _NUM_EXPERTS
    in_features = _Q4_K_IN_FEATURES
    out_features = _Q4_K_OUT_FEATURES
    x_bits, x_ref = _make_activation(rows, in_features, seed=7100 + rows)
    qweight_a = _make_expert_q4_k_weights(
        num_experts=num_experts, out_features=out_features, in_features=in_features, seed=7201
    )
    qweight_b = _make_expert_q4_k_weights(
        num_experts=num_experts, out_features=out_features, in_features=in_features, seed=7202
    )
    expert_start = _expert_start(counts)

    out_shape = (rows, out_features)
    allocations = []
    try:
        d_x = _upload(x_bits, runtime, allocations)
        d_expert_start = _upload(expert_start, runtime, allocations)
        d_qweight_a = _upload(qweight_a, runtime, allocations)
        d_qweight_b = _upload(qweight_b, runtime, allocations)
        d_incumbent_a = _alloc(out_shape, np.uint16, runtime, allocations)
        d_incumbent_b = _alloc(out_shape, np.uint16, runtime, allocations)
        d_m1_a = _alloc(out_shape, np.uint16, runtime, allocations)
        d_m1_b = _alloc(out_shape, np.uint16, runtime, allocations)

        gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out(
            d_x.ptr,
            d_expert_start.ptr,
            d_qweight_a.ptr,
            d_qweight_b.ptr,
            d_incumbent_a.ptr,
            d_incumbent_b.ptr,
            rows,
            num_experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q4_k_selected_dual_grouped_rowbatch8_out4_expertgrid64_m1_bf16_bf16_out(
            d_x.ptr,
            d_expert_start.ptr,
            d_qweight_a.ptr,
            d_qweight_b.ptr,
            d_m1_a.ptr,
            d_m1_b.ptr,
            rows,
            num_experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        incumbent_a = _download(d_incumbent_a, out_shape, np.uint16, runtime)
        incumbent_b = _download(d_incumbent_b, out_shape, np.uint16, runtime)
        m1_a = _download(d_m1_a, out_shape, np.uint16, runtime)
        m1_b = _download(d_m1_b, out_shape, np.uint16, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    # T0 tile-retune contract: identical arithmetic, only schedule changes.
    np.testing.assert_array_equal(m1_a, incumbent_a)
    np.testing.assert_array_equal(m1_b, incumbent_b)

    if rows == 16:
        # CPU-reference sanity at the smallest anchor shape only: guards
        # against a shared fixture misencoding (both kernels equally wrong).
        reference = _q4_k_reference_per_expert(x_ref, qweight_a, counts)
        np.testing.assert_allclose(
            bf16_to_float32(incumbent_a), reference, rtol=_CPU_RTOL, atol=_CPU_ATOL
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [16, 64, 512])
def test_pf3_q5_1_down_m1_bit_exact_vs_incumbent(rows: int) -> None:
    """Q5_1 down M1 candidate must be bit-for-bit vs the incumbent owner (T0).

    Binding shape: rows=512 chunk; anchor shapes: rows=16 and rows=64.
    Production widths: in_features=640 (ffn), down out_features=2560 (hidden).
    Verified from GGUF metadata + tensor shapes of the local model files.

    RED on the unmodified path: the candidate function does not exist, so the
    import fails before any GPU work runs.
    """
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        build_qwen4_exp_q5_1,
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out,
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_q5_1(load=True)
    counts = _expert_counts(rows, seed=8000 + rows)
    num_experts = _NUM_EXPERTS
    in_features = _Q5_1_IN_FEATURES
    out_features = _Q5_1_OUT_FEATURES
    x_bits, x_ref = _make_activation(rows, in_features, seed=8100 + rows)
    qweights = _make_expert_q5_1_weights(
        num_experts=num_experts, out_features=out_features, in_features=in_features, seed=8201
    )
    expert_start = _expert_start(counts)

    out_shape = (rows, out_features)
    allocations = []
    try:
        d_x = _upload(x_bits, runtime, allocations)
        d_expert_start = _upload(expert_start, runtime, allocations)
        d_qweights = _upload(qweights, runtime, allocations)
        d_incumbent = _alloc(out_shape, np.uint16, runtime, allocations)
        d_m1 = _alloc(out_shape, np.uint16, runtime, allocations)

        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out(
            d_x.ptr,
            d_expert_start.ptr,
            d_qweights.ptr,
            d_incumbent.ptr,
            rows,
            num_experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out(
            d_x.ptr,
            d_expert_start.ptr,
            d_qweights.ptr,
            d_m1.ptr,
            rows,
            num_experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        incumbent = _download(d_incumbent, out_shape, np.uint16, runtime)
        m1 = _download(d_m1, out_shape, np.uint16, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    # T0 tile-retune contract: identical arithmetic, only schedule changes.
    np.testing.assert_array_equal(m1, incumbent)

    if rows == 16:
        # CPU-reference sanity at the smallest anchor shape only.
        from hipengine.quant.gguf import dequantize_gguf_data

        expected = np.zeros(out_shape, dtype=np.float32)
        start = 0
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            stop = start + int(count)
            weight = dequantize_gguf_data(
                qweights[expert], GGMLQuantizationType.Q5_1
            )
            expected[start:stop] = x_ref[start:stop] @ weight.T
            start = stop
        np.testing.assert_allclose(
            bf16_to_float32(incumbent), expected, rtol=_CPU_RTOL, atol=_CPU_ATOL
        )


def test_q5_1_m2_variant_key_registry_contract() -> None:
    """The M2 key must resolve exactly, with M1 retained as fallback.

    RED on the unmodified path: the M2 function does not exist, so the import
    fails before any GPU work runs.
    """
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out,
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m2_bf16_bf16_out,
        register_qwen4_exp_q5_1_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_qwen4_exp_q5_1_kernels()
    resolved = resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_q5_1",
        variant=(
            "selected_grouped_prefill_compact_rowbatch8_out8_"
            "expertgrid64_m2_bf16_bf16_out"
        ),
        missing="none",
    )
    fallback = resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_q5_1",
        variant=(
            "selected_grouped_prefill_compact_rowbatch8_out8_"
            "expertgrid64_m1_bf16_bf16_out"
        ),
        missing="none",
    )
    assert (
        resolved
        is qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m2_bf16_bf16_out
    )
    assert (
        fallback
        is qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [16, 512])
def test_q5_1_down_m2_hierarchical_reduction_bit_exact_vs_m1(rows: int) -> None:
    """M2 must preserve every M1 logical partial and strict tree addition.

    M2 may fold only the strict stride-128 add into registers, use shared
    memory for the stride-64/32 cross-wave steps, and use wave shuffles for
    the unchanged stride-16..1 additions. The binding production shape is a
    512-row chunk; 16 rows covers sparse experts and empty-expert handling.

    RED on the unmodified path: the M2 function does not exist, so the import
    fails before any GPU work runs.
    """
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        build_qwen4_exp_q5_1,
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out,
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m2_bf16_bf16_out,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_q5_1(load=True)
    counts = _expert_counts(rows, seed=8300 + rows)
    x_bits, _ = _make_activation(rows, _Q5_1_IN_FEATURES, seed=8400 + rows)
    qweights = _make_expert_q5_1_weights(
        num_experts=_NUM_EXPERTS,
        out_features=_Q5_1_OUT_FEATURES,
        in_features=_Q5_1_IN_FEATURES,
        seed=8501,
    )
    expert_start = _expert_start(counts)
    out_shape = (rows, _Q5_1_OUT_FEATURES)

    allocations = []
    try:
        d_x = _upload(x_bits, runtime, allocations)
        d_expert_start = _upload(expert_start, runtime, allocations)
        d_qweights = _upload(qweights, runtime, allocations)
        d_m1 = _alloc(out_shape, np.uint16, runtime, allocations)
        d_m2 = _alloc(out_shape, np.uint16, runtime, allocations)
        for function, output in (
            (
                qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out,
                d_m1,
            ),
            (
                qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m2_bf16_bf16_out,
                d_m2,
            ),
        ):
            function(
                d_x.ptr,
                d_expert_start.ptr,
                d_qweights.ptr,
                output.ptr,
                rows,
                _NUM_EXPERTS,
                _Q5_1_IN_FEATURES,
                _Q5_1_OUT_FEATURES,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        m1 = _download(d_m1, out_shape, np.uint16, runtime)
        m2 = _download(d_m2, out_shape, np.uint16, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(m2, m1)


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape, dtype, runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
