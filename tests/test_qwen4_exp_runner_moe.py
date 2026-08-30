from __future__ import annotations

import ctypes
from math import prod
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.qwen4_exp import (
    Qwen4ExpMoEWeights,
    qwen4_exp_moe,
)
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    float_array_to_bf16_bits,
)
from hipengine.loading.qwen4_exp_gguf import Qwen4ExpGGUFTensorRef
from hipengine.loading.qwen4_exp_materialize import (
    LAYOUT_RAW_GGUF,
    Qwen4ExpDeviceWeight,
    Qwen4ExpGGUFWeightSpec,
)
from hipengine.quant.gguf import (
    GGMLQuantizationType,
    bf16_to_float32,
    dequantize_gguf_data,
)
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpMoEScratch,
    run_qwen4_exp_moe,
)
from tests._gguf_synthetic_weights import make_q4_k_weight
from tests.test_qwen4_exp_runner_gr import _dense_f32_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.parametrize(
    ("rows", "grouped_prefill"), [(1, False), (3, False), (16, True)]
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_runner_moe_matches_reduced_topk_shared_cpu_oracle(
    rows: int, grouped_prefill: bool, monkeypatch
) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    if grouped_prefill:
        monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL", "1")
    else:
        monkeypatch.delenv("HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL", raising=False)
    rng = np.random.default_rng(510)
    hidden, ffn, experts, top_k = 256, 256, 4, 2
    mixed = rng.normal(0.0, 0.1, size=(rows, hidden)).astype(np.float32)
    router = rng.normal(0.0, 0.1, size=(experts, hidden)).astype(np.float32)
    gate_raw = np.stack([make_q4_k_weight(ffn, hidden) for _ in range(experts)])
    up_raw = np.stack([make_q4_k_weight(ffn, hidden) for _ in range(experts)])
    down_raw = np.stack([make_q4_k_weight(hidden, ffn) for _ in range(experts)])
    expert_gate = np.stack(
        [dequantize_gguf_data(value, GGMLQuantizationType.Q4_K) for value in gate_raw]
    )
    expert_up = np.stack(
        [dequantize_gguf_data(value, GGMLQuantizationType.Q4_K) for value in up_raw]
    )
    expert_down = np.stack(
        [dequantize_gguf_data(value, GGMLQuantizationType.Q4_K) for value in down_raw]
    )
    shared_gate = rng.normal(0.0, 0.1, size=(ffn, hidden)).astype(np.float32)
    shared_up = rng.normal(0.0, 0.1, size=(ffn, hidden)).astype(np.float32)
    shared_down = rng.normal(0.0, 0.1, size=(hidden, ffn)).astype(np.float32)
    shared_scalar = rng.normal(0.0, 0.1, size=(hidden,)).astype(np.float32)
    expected = qwen4_exp_moe(
        mixed,
        Qwen4ExpMoEWeights(
            router=router,
            expert_gate=expert_gate,
            expert_up=expert_up,
            expert_down=expert_down,
            shared_gate=shared_gate,
            shared_up=shared_up,
            shared_down=shared_down,
            shared_gate_weight=shared_scalar,
            experts_used=top_k,
        ),
    )

    allocations = []
    scratch = serial_scratch = None
    try:
        d_mixed = _upload(mixed, runtime, allocations)
        weights = {
            "router": _dense_f32_weight("router", router, runtime, allocations),
            "expert_gate": _q4_weight("expert_gate", gate_raw, runtime, allocations),
            "expert_up": _q4_weight("expert_up", up_raw, runtime, allocations),
            "expert_down": _q4_weight("expert_down", down_raw, runtime, allocations),
            "shared_gate": _dense_f32_weight("shared_gate", shared_gate, runtime, allocations),
            "shared_up": _dense_f32_weight("shared_up", shared_up, runtime, allocations),
            "shared_down": _dense_f32_weight("shared_down", shared_down, runtime, allocations),
            "shared_gate_weight": _dense_f32_weight(
                "shared_gate_weight",
                shared_scalar.reshape(1, hidden),
                runtime,
                allocations,
            ),
        }
        scratch = Qwen4ExpMoEScratch.allocate(
            rows=rows,
            hidden=hidden,
            ffn=ffn,
            experts=experts,
            top_k=top_k,
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
        actual_bits = _download(result.output, (rows, hidden), np.uint16, runtime)
        assert result.selected.nbytes == rows * top_k * np.dtype(np.int64).itemsize
        selected = _download(result.selected, (rows, top_k), np.int64, runtime)
        routing = _download(result.routing, (rows, top_k), np.float32, runtime)
        serial_bits = None
        serial_selected = None
        serial_routing = None
        if rows > 1:
            serial_scratch = Qwen4ExpMoEScratch.allocate(
                rows=1,
                hidden=hidden,
                ffn=ffn,
                experts=experts,
                top_k=top_k,
                runtime=runtime,
            )
            bits_rows = []
            selected_rows = []
            routing_rows = []
            for row in range(rows):
                serial = run_qwen4_exp_moe(
                    d_mixed.ptr + row * hidden * np.dtype(np.float32).itemsize,
                    weights,
                    scratch=serial_scratch,
                    rows=1,
                    hidden=hidden,
                    ffn=ffn,
                    experts=experts,
                    top_k=top_k,
                    runtime=runtime,
                )
                runtime.device_synchronize()
                bits_rows.append(
                    _download(serial.output, (hidden,), np.uint16, runtime)
                )
                selected_rows.append(
                    _download(serial.selected, (top_k,), np.int64, runtime)
                )
                routing_rows.append(
                    _download(serial.routing, (top_k,), np.float32, runtime)
                )
            serial_bits = np.asarray(bits_rows)
            serial_selected = np.asarray(selected_rows)
            serial_routing = np.asarray(routing_rows)
        finite_boundaries = {
            "expert_intermediate": bf16_to_float32(
                _download(scratch.expert_intermediate, (rows * top_k, ffn), np.uint16, runtime)
            ),
            "expert_down": bf16_to_float32(
                _download(scratch.expert_down, (rows * top_k, hidden), np.uint16, runtime)
            ),
            "routed": bf16_to_float32(
                _download(scratch.routed, (rows, hidden), np.uint16, runtime)
            ),
            "shared_down": _download(
                scratch.shared_down, (rows, hidden), np.float32, runtime
            ),
        }
        # The c1 operation-complete Q4 owner writes only the authoritative
        # BF16 post-SiLU intermediate. Multirow strict/grouped fallbacks retain
        # separate gate/up planes and keep those surfaces testable.
        if rows > 1:
            finite_boundaries["expert_gate"] = bf16_to_float32(
                _download(
                    scratch.expert_gate,
                    (rows * top_k, ffn),
                    np.uint16,
                    runtime,
                )
            )
            finite_boundaries["expert_up"] = bf16_to_float32(
                _download(
                    scratch.expert_up,
                    (rows * top_k, ffn),
                    np.uint16,
                    runtime,
                )
            )
    finally:
        if serial_scratch is not None:
            serial_scratch.close()
        if scratch is not None:
            scratch.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    for name, boundary in finite_boundaries.items():
        assert np.isfinite(boundary).all(), name
    np.testing.assert_array_equal(selected, expected.selected_experts.astype(np.int64))
    if serial_bits is not None:
        if grouped_prefill:
            np.testing.assert_allclose(
                bf16_to_float32(actual_bits),
                bf16_to_float32(serial_bits),
                rtol=2e-2,
                atol=2e-2,
            )
        else:
            np.testing.assert_array_equal(actual_bits, serial_bits)
        np.testing.assert_array_equal(selected, serial_selected)
        np.testing.assert_array_equal(routing, serial_routing)
    np.testing.assert_allclose(routing, expected.routing_weights, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        bf16_to_float32(actual_bits),
        expected.output,
        rtol=3e-2 if grouped_prefill else 2e-2,
        atol=2e-2,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_runner_moe_q8_0_down_grouped_strict_fallback(
    monkeypatch,
) -> None:
    """Regression: Q8_0 expert-down through the grouped prefill path with no
    opt-in Q8_0 grouped owner flag must fall back to the strict per-expert
    selected gemv and match the CPU oracle.

    The P1 device-driven grouped Q8_0 down owner restructured the Q8_0 down
    dispatch in the grouped branch. A regression dropped the strict fallback
    when neither ``HIPENGINE_QWEN4_EXP_Q8_0_GROUPED`` nor
    ``HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA`` is set, leaving ``expert_down``
    unwritten for Q8_0 layers (layer 2/4/30/46/47) and corrupting whole-model
    prefill output. This test pins the default (no-flag) path to the CPU
    oracle.
    """
    from hipengine.core.hip import get_hip_runtime
    from tests._gguf_synthetic_weights import make_q8_0_weight

    runtime = get_hip_runtime()
    # Grouped prefill for rows>=16 with a Q4_K/Q4_K gate/up pair; Q8_0 down.
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL", "1")
    # Ensure neither opt-in Q8_0 grouped owner is active (the buggy case).
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_0_GROUPED", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA", raising=False)
    rng = np.random.default_rng(2026)
    hidden, ffn, experts, top_k = 256, 256, 4, 2
    rows = 16
    mixed = rng.normal(0.0, 0.1, size=(rows, hidden)).astype(np.float32)
    router = rng.normal(0.0, 0.1, size=(experts, hidden)).astype(np.float32)
    gate_raw = np.stack([make_q4_k_weight(ffn, hidden) for _ in range(experts)])
    up_raw = np.stack([make_q4_k_weight(ffn, hidden) for _ in range(experts)])
    down_raw = np.stack([make_q8_0_weight(hidden, ffn) for _ in range(experts)])
    expert_gate = np.stack(
        [dequantize_gguf_data(value, GGMLQuantizationType.Q4_K) for value in gate_raw]
    )
    expert_up = np.stack(
        [dequantize_gguf_data(value, GGMLQuantizationType.Q4_K) for value in up_raw]
    )
    expert_down = np.stack(
        [dequantize_gguf_data(value, GGMLQuantizationType.Q8_0) for value in down_raw]
    )
    shared_gate = rng.normal(0.0, 0.1, size=(ffn, hidden)).astype(np.float32)
    shared_up = rng.normal(0.0, 0.1, size=(ffn, hidden)).astype(np.float32)
    shared_down = rng.normal(0.0, 0.1, size=(hidden, ffn)).astype(np.float32)
    shared_scalar = rng.normal(0.0, 0.1, size=(hidden,)).astype(np.float32)
    expected = qwen4_exp_moe(
        mixed,
        Qwen4ExpMoEWeights(
            router=router,
            expert_gate=expert_gate,
            expert_up=expert_up,
            expert_down=expert_down,
            shared_gate=shared_gate,
            shared_up=shared_up,
            shared_down=shared_down,
            shared_gate_weight=shared_scalar,
            experts_used=top_k,
        ),
    )

    allocations = []
    scratch = serial_scratch = None
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
                "shared_gate_weight",
                shared_scalar.reshape(1, hidden),
                runtime,
                allocations,
            ),
        }
        scratch = Qwen4ExpMoEScratch.allocate(
            rows=rows,
            hidden=hidden,
            ffn=ffn,
            experts=experts,
            top_k=top_k,
            runtime=runtime,
        )
        serial_scratch = Qwen4ExpMoEScratch.allocate(
            rows=1, hidden=hidden, ffn=ffn, experts=experts, top_k=top_k,
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
        actual_bits = _download(result.output, (rows, hidden), np.uint16, runtime)
        selected = _download(result.selected, (rows, top_k), np.int64, runtime)
        routing = _download(result.routing, (rows, top_k), np.float32, runtime)
        # Strict rows=1 baseline: the Q8_0 down always uses the strict
        # per-expert selected gemv. The grouped (rows=16) path with no opt-in
        # flag must reproduce it within the grouped tolerance.
        serial_bits = []
        for row in range(rows):
            serial = run_qwen4_exp_moe(
                d_mixed.ptr + row * hidden * np.dtype(np.float32).itemsize,
                weights,
                scratch=serial_scratch,
                rows=1, hidden=hidden, ffn=ffn, experts=experts, top_k=top_k,
                runtime=runtime,
            )
            runtime.device_synchronize()
            serial_bits.append(
                _download(serial.output, (hidden,), np.uint16, runtime)
            )
        serial_bits = np.asarray(serial_bits)
        np.testing.assert_array_equal(selected, expected.selected_experts.astype(np.int64))
        np.testing.assert_allclose(routing, expected.routing_weights, rtol=2e-6, atol=2e-6)
        # The grouped path must agree with the strict serial baseline. If the
        # Q8_0 down fallback were dropped (the regression), expert_down would be
        # unwritten and this would diverge far beyond tolerance.
        np.testing.assert_allclose(
            bf16_to_float32(actual_bits),
            bf16_to_float32(serial_bits),
            rtol=2e-2,
            atol=2e-2,
        )
        down = bf16_to_float32(
            _download(scratch.expert_down, (rows * top_k, hidden), np.uint16, runtime)
        )
        assert np.all(np.isfinite(down))
    finally:
        if serial_scratch is not None:
            serial_scratch.close()
        if scratch is not None:
            scratch.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def _q4_weight(name, raw, runtime, allocations):
    return _quant_weight(
        name, raw, "gguf_q4_k", GGMLQuantizationType.Q4_K, 144, runtime, allocations
    )


def _q8_0_weight(name, raw, runtime, allocations):
    return _quant_weight(
        name, raw, "gguf_q8_0", GGMLQuantizationType.Q8_0, 34, runtime, allocations
    )


def _quant_weight(name, raw, quant_key, ggml_type, block_bytes, runtime, allocations):
    host = np.ascontiguousarray(raw, dtype=np.uint8)
    buffer = malloc(host.nbytes, runtime=runtime)
    allocations.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(host), runtime=runtime)
    shape = (raw.shape[0], raw.shape[1], 256 * (raw.shape[2] // block_bytes))
    tensor = GGUFTensorInfo(
        name=f"{name}.weight",
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(ggml_type),
        ggml_type_name=ggml_type.name,
        n_elements=prod(shape),
        nbytes=int(host.nbytes),
        offset=0,
        data_offset=0,
        byte_shape=tuple(int(value) for value in host.shape),
    )
    spec = Qwen4ExpGGUFWeightSpec(
        slot_path=name,
        source_ref=Qwen4ExpGGUFTensorRef(0, Path("synthetic.gguf"), tensor),
        quant_key=quant_key,
        layout=LAYOUT_RAW_GGUF,
        allocation_names=("raw",),
        device_resident=True,
        device_nbytes=int(host.nbytes),
    )
    allocation = DeviceTensorAllocation(
        name=name,
        source=SimpleSource(name),
        buffer=buffer,
        tensor=SimpleTensor(buffer.ptr),
        owns_buffer=False,
    )
    return Qwen4ExpDeviceWeight(spec, "hip_gfx1151", {"raw": allocation})


class SimpleSource:
    def __init__(self, name):
        self.name = name


class SimpleTensor:
    def __init__(self, ptr):
        self.ptr = ptr


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
