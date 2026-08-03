"""Exact small-row reuse for resident-pack8 GGUF Q4_K projections."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_q4_k_pack8_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_pack8_gemv_bf16_bf16_out,
    gguf_q4_k_pack8_rowtile_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8
import hipengine.runtime.gguf_linear as gguf_linear_module
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    _pack8_rowtile_dispatch,
    launch_gguf_linear_pair,
    native_batch_decode_session,
)
from tests.test_gguf_q4_k_gemv import make_q4_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).copy()
    words += 0x7FFF + ((words >> 16) & 1)
    return (words >> 16).astype(np.uint16)


def _bf16_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    probs = np.exp(shifted, dtype=np.float64)
    return probs / np.sum(probs, axis=-1, keepdims=True)


def test_pack8_rowtile_registry_and_dispatch_contract() -> None:
    key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_rowtile_bf16_bf16_out",
    )
    assert resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    ) is gguf_q4_k_pack8_rowtile_bf16_bf16_out

    base = GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k",
            "pack8_prefill_bf16_bf16_out",
        ),
        "pack8",
    )
    for rows in (2, 3, 4):
        selected = _pack8_rowtile_dispatch(
            base,
            rows=rows,
            use_rowtile=True,
            native_batch=True,
        )
        assert selected.key == key
        assert selected.abi == "pack8"
    for rows in (1, 5, 8):
        assert (
            _pack8_rowtile_dispatch(
                base,
                rows=rows,
                use_rowtile=True,
                native_batch=True,
            )
            is base
        )
    assert (
        _pack8_rowtile_dispatch(
            base,
            rows=4,
            use_rowtile=False,
            native_batch=True,
        )
        is base
    )
    assert (
        _pack8_rowtile_dispatch(
            base,
            rows=4,
            use_rowtile=True,
            native_batch=False,
        )
        is base
    )


def test_native_pack8_pair_uses_two_bounded_rowtiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations = {
        "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
        "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=12)),
        "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=13)),
    }

    class Weight:
        spec = SimpleNamespace(layout="q4_k_pack8", quant_key="gguf_q4_k")
        backend = "hip_gfx1100"

        def allocation(self, name: str):
            return allocations[name]

    rowtile_calls: list[tuple[tuple, dict]] = []
    dual_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_pack8_rowtile_bf16_bf16_out",
        lambda *args, **kwargs: rowtile_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_pack8_dual_prefill_bf16_bf16_out",
        lambda *args, **kwargs: dual_calls.append((args, kwargs)),
    )
    with native_batch_decode_session(True):
        assert launch_gguf_linear_pair(
            Weight(),
            Weight(),
            100,
            200,
            300,
            4,
            5120,
            17408,
            use_wmma_prefill=False,
            use_gemv_decode=True,
        )
    assert len(rowtile_calls) == 2
    assert dual_calls == []

    assert launch_gguf_linear_pair(
        Weight(),
        Weight(),
        100,
        200,
        300,
        4,
        5120,
        17408,
        use_wmma_prefill=False,
        use_gemv_decode=True,
    )
    assert len(dual_calls) == 1


def test_pack8_rowtile_wrapper_rejects_non_verifier_shapes() -> None:
    for rows in (1, 5):
        with pytest.raises(ValueError, match="rows must be 2, 3, or 4"):
            gguf_q4_k_pack8_rowtile_bf16_bf16_out(
                1,
                2,
                3,
                4,
                5,
                rows,
                256,
                16,
            )
    with pytest.raises(ValueError, match="threads must be 0 or 32"):
        gguf_q4_k_pack8_rowtile_bf16_bf16_out(
            1,
            2,
            3,
            4,
            5,
            4,
            256,
            16,
            threads=64,
        )


def _run_projection(
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = make_q4_k_weight(out_features, in_features)
    packed = repack_gguf_q4_k_pack8(raw)
    rng = np.random.default_rng(0xD27 + rows * 17 + in_features + out_features)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.15, size=(rows, in_features)).astype(np.float32)
    )
    control = np.empty((rows, out_features), dtype=np.uint16)
    candidate = np.empty_like(control)
    arrays = (x_bits, packed.qweight, packed.scales, packed.mins)
    inputs = [malloc(array.nbytes) for array in arrays]
    outputs = [malloc(control.nbytes), malloc(candidate.nbytes)]
    library = build_gguf_q4_k_gemv(load=True)
    try:
        for array, allocation in zip(arrays, inputs, strict=True):
            copy_host_to_device(allocation, host_array_ptr(array), array.nbytes)
        x_d, q_d, s_d, m_d = inputs
        control_d, candidate_d = outputs
        common = (
            x_d.ptr,
            q_d.ptr,
            s_d.ptr,
            m_d.ptr,
        )
        gguf_q4_k_pack8_gemv_bf16_bf16_out(
            *common,
            control_d.ptr,
            rows,
            in_features,
            out_features,
            threads=32,
            library=library,
        )
        gguf_q4_k_pack8_rowtile_bf16_bf16_out(
            *common,
            candidate_d.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        for host, allocation in zip((control, candidate), outputs, strict=True):
            copy_device_to_host(host_array_ptr(host), allocation, host.nbytes)
    finally:
        for allocation in (*outputs, *inputs):
            free(allocation)
    cpu = gguf_q4_k_pack8_gemv(
        _bf16_f32(x_bits),
        packed.qweight,
        packed.scales,
        packed.mins,
    )
    return control, candidate, cpu


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (2, 3, 4))
@pytest.mark.parametrize(
    "in_features,out_features",
    ((256, 16), (512, 64), (1024, 128)),
)
def test_pack8_rowtile_is_bit_exact_and_passes_cpu_gate(
    rows: int,
    in_features: int,
    out_features: int,
) -> None:
    control, candidate, cpu = _run_projection(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    )
    np.testing.assert_array_equal(candidate, control)

    gpu = _bf16_f32(candidate)
    p = _softmax(cpu)
    q = _softmax(gpu)
    kl = np.sum(p * (np.log(p + 1.0e-30) - np.log(q + 1.0e-30)), axis=-1)
    top1 = np.mean(np.argmax(cpu, axis=-1) == np.argmax(gpu, axis=-1))
    assert float(np.max(kl)) <= 0.05
    assert float(top1) >= 0.90
