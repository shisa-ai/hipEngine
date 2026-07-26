"""Exact RED/GREEN gate for Laguna paired-output SWAR Q5 reconstruction."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100 import quant
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as kernel_mod
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import (
    Q5_K_BLOCK_BYTES,
    make_q5_k_weight,
    make_q6_k_weight,
)

_DIRECT_VARIANT = "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out"
_PAIR_VARIANT = _DIRECT_VARIANT
_MIXED_VARIANT = (
    "mixed_local32_q5_swar_pair_fixed_meta_gemv_decode_bf16_f32_out"
)
_MIXED_QUANT = "gguf_q5_k+gguf_q6_k+gguf_q6_k+gguf_q5_k"
_DIRECT_BF16 = (
    "gguf_q5_k_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out"
)
_DIRECT_F32 = "gguf_q5_k_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_f32_out"
_PAIR_BF16 = (
    "gguf_q5_k_pair_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out"
)
_PAIR_F32 = (
    "gguf_q5_k_pair_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_f32_out"
)
_MIXED = (
    "gguf_q5_q6_attention_q5_qg_mixed_local32_q5_swar_pair_fixed_meta_"
    "gemv_decode_bf16_f32_out"
)
_KEYS = (
    KernelKey("hip_gfx1100", "linear", "gguf_q5_k", _DIRECT_VARIANT),
    KernelKey("hip_gfx1100", "linear_pair", "gguf_q5_k", _PAIR_VARIANT),
    KernelKey(
        "hip_gfx1100",
        "attention_projection_quad",
        _MIXED_QUANT,
        _MIXED_VARIANT,
    ),
)
_SOURCE = Path(kernel_mod.__file__).with_suffix(".hip")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def hip_context():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return kernel_mod.build_gguf_k_gemv(load=True)


def _wrapper(name: str):
    candidate = getattr(kernel_mod, name, None)
    assert callable(candidate), f"missing selected wrapper {name}"
    return candidate


def _f32_to_bf16(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32).copy()
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    rounded[np.isnan(f32)] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(bits.shape).copy()


def _edge_bf16(in_features: int) -> np.ndarray:
    edges = np.asarray(
        (
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            0x007F,
            0x807F,
            0x0080,
            0x8080,
            0x3F7F,
            0x3F80,
            0x3F81,
            0xBF7F,
            0xBF80,
            0xBF81,
            0x4700,
            0xC700,
            0x7F00,
            0xFF00,
        ),
        dtype=np.uint16,
    )
    return np.resize(edges, (1, in_features)).copy()


def _edge_q5_weight(out_features: int, in_features: int, *, seed: int) -> np.ndarray:
    weight = make_q5_k_weight(out_features, in_features)
    blocks = weight.reshape(out_features, in_features // 256, Q5_K_BLOCK_BYTES)
    signs = np.indices(blocks.shape[:2]).sum(axis=0) & 1
    d = blocks[..., 0:2].copy().view(np.uint16).reshape(blocks.shape[:2])
    dmin = blocks[..., 2:4].copy().view(np.uint16).reshape(blocks.shape[:2])
    d |= signs.astype(np.uint16) << 15
    dmin |= (1 - signs).astype(np.uint16) << 15
    blocks[..., 0:2] = d[..., None].view(np.uint8).reshape(*d.shape, 2)
    blocks[..., 2:4] = dmin[..., None].view(np.uint8).reshape(*dmin.shape, 2)
    rng = np.random.default_rng(seed)
    blocks[..., 4:16] = rng.integers(
        0, 256, size=blocks[..., 4:16].shape, dtype=np.uint8
    )
    blocks[..., 16:48] = np.resize(
        np.asarray((0x00, 0xFF, 0x55, 0xAA), dtype=np.uint8),
        blocks[..., 16:48].shape,
    )
    blocks[..., 48:176] = rng.integers(
        0, 256, size=blocks[..., 48:176].shape, dtype=np.uint8
    )
    return weight


def _run_direct(fn, x: np.ndarray, weight: np.ndarray, *, dtype, library):
    out = np.empty((1, weight.shape[0]), dtype=dtype)
    buffers = []
    try:
        for array in (x, weight, out):
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
        copy_host_to_device(buffers[0], host_array_ptr(x), x.nbytes)
        copy_host_to_device(buffers[1], host_array_ptr(weight), weight.nbytes)
        fn(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            1,
            x.shape[1],
            weight.shape[0],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), buffers[2], out.nbytes)
        return out
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def _run_pair(fn, x: np.ndarray, weights, *, dtype, library):
    outputs = tuple(np.empty((1, weight.shape[0]), dtype=dtype) for weight in weights)
    buffers = []
    try:
        x_buf = malloc(x.nbytes)
        buffers.append(x_buf)
        weight_bufs = [malloc(weight.nbytes) for weight in weights]
        output_bufs = [malloc(output.nbytes) for output in outputs]
        buffers.extend(weight_bufs)
        buffers.extend(output_bufs)
        copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
        for weight, buffer in zip(weights, weight_bufs, strict=True):
            copy_host_to_device(buffer, host_array_ptr(weight), weight.nbytes)
        fn(
            x_buf.ptr,
            *(buffer.ptr for buffer in weight_bufs),
            *(buffer.ptr for buffer in output_bufs),
            1,
            x.shape[1],
            *(weight.shape[0] for weight in weights),
            library=library,
        )
        for output, buffer in zip(outputs, output_bufs, strict=True):
            copy_device_to_host(host_array_ptr(output), buffer, output.nbytes)
        return outputs
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def _run_mixed(fn, x: np.ndarray, weights, *, library):
    outputs = tuple(
        np.empty((1, weight.shape[0]), dtype=np.float32) for weight in weights
    )
    buffers = []
    try:
        x_buf = malloc(x.nbytes)
        buffers.append(x_buf)
        weight_bufs = [malloc(weight.nbytes) for weight in weights]
        output_bufs = [malloc(output.nbytes) for output in outputs]
        buffers.extend(weight_bufs)
        buffers.extend(output_bufs)
        copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
        for weight, buffer in zip(weights, weight_bufs, strict=True):
            copy_host_to_device(buffer, host_array_ptr(weight), weight.nbytes)
        fn(
            x_buf.ptr,
            *(buffer.ptr for buffer in weight_bufs),
            *(buffer.ptr for buffer in output_bufs),
            1,
            x.shape[1],
            *(weight.shape[0] for weight in weights),
            library=library,
        )
        for output, buffer in zip(outputs, output_bufs, strict=True):
            copy_device_to_host(host_array_ptr(output), buffer, output.nbytes)
        return outputs
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def test_q5_swar_pair_registry_package_exports_and_controls_are_distinct() -> None:
    wrappers = tuple(
        _wrapper(name)
        for name in (_DIRECT_BF16, _DIRECT_F32, _PAIR_BF16, _PAIR_F32, _MIXED)
    )
    for name, wrapper in zip(
        (_DIRECT_BF16, _DIRECT_F32, _PAIR_BF16, _PAIR_F32, _MIXED),
        wrappers,
        strict=True,
    ):
        assert getattr(quant, name, None) is wrapper

    kernel_mod.register_gguf_k_gemv_kernels(replace=True)
    assert resolve(backend=_KEYS[0].backend, layer=_KEYS[0].layer, quant=_KEYS[0].quant, variant=_KEYS[0].variant) is wrappers[0]
    assert resolve(backend=_KEYS[1].backend, layer=_KEYS[1].layer, quant=_KEYS[1].quant, variant=_KEYS[1].variant) is wrappers[2]
    assert resolve(backend=_KEYS[2].backend, layer=_KEYS[2].layer, quant=_KEYS[2].quant, variant=_KEYS[2].variant) is wrappers[4]
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k",
        variant="wave32x2_fixed_meta_gemv_decode_bf16_bf16_out",
    ) is kernel_mod.gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear_pair",
        quant="gguf_q5_k",
        variant="wave32x2_fixed_meta_gemv_decode_bf16_bf16_out",
    ) is kernel_mod.gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out


def test_q5_swar_pair_wrappers_reject_before_library_load(monkeypatch) -> None:
    direct = _wrapper(_DIRECT_BF16)
    pair = _wrapper(_PAIR_BF16)
    mixed = _wrapper(_MIXED)

    def fail_build(*args, **kwargs):
        raise AssertionError("invalid input reached library load")

    monkeypatch.setattr(kernel_mod, "build_gguf_k_gemv", fail_build)
    with pytest.raises(ValueError, match="pointers must be non-zero"):
        direct(0, 2, 3, rows=1, in_features=256, out_features=8)
    with pytest.raises(ValueError, match="rows must be exactly 1"):
        direct(1, 2, 3, rows=2, in_features=256, out_features=8)
    with pytest.raises(ValueError, match="divisible by GGUF Q5_K block size 256"):
        direct(1, 2, 3, rows=1, in_features=255, out_features=8)
    with pytest.raises(ValueError, match="divisible by 2"):
        direct(1, 2, 3, rows=1, in_features=256, out_features=7)
    with pytest.raises(ValueError, match="threads must be 32"):
        direct(1, 2, 3, rows=1, in_features=256, out_features=8, threads=64)
    with pytest.raises(ValueError, match="pointers must be non-zero"):
        pair(1, 2, 0, 4, 5, rows=1, in_features=256, out_features=8, out_features_b=8)
    with pytest.raises(ValueError, match="divisible by 2"):
        pair(1, 2, 3, 4, 5, rows=1, in_features=256, out_features=8, out_features_b=7)
    valid_mixed = dict(
        rows=1,
        in_features=256,
        q_features=8,
        k_features=8,
        v_features=8,
        gate_features=8,
    )
    with pytest.raises(ValueError, match="pointers must be non-zero"):
        mixed(0, *range(2, 10), **valid_mixed)
    with pytest.raises(ValueError, match="rows must be exactly 1"):
        mixed(*range(1, 10), **(valid_mixed | {"rows": 2}))
    with pytest.raises(ValueError, match="divisible by 8"):
        mixed(*range(1, 10), **(valid_mixed | {"gate_features": 6}))


def test_q5_swar_pair_keys_are_excluded_from_unvalidated_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.kernels.hip_gfx1151 as backend

    for key in _KEYS:
        for unvalidated_backend in ("hip_gfx1151", "cuda_sm86", "cpu_reference"):
            assert not is_registered(
                KernelKey(unvalidated_backend, key.layer, key.quant, key.variant)
            )

    registered: list[KernelKey] = []
    monkeypatch.setattr(backend, "import_module", lambda _name: None)
    monkeypatch.setattr(backend, "registered_keys", lambda: _KEYS)
    monkeypatch.setattr(backend, "is_registered", lambda _key: False)
    monkeypatch.setattr(backend, "resolve", lambda **_kwargs: object())
    monkeypatch.setattr(
        backend,
        "register",
        lambda key, _kernel, *, replace=False: registered.append(key),
    )
    backend.register_gfx1151_kernels()
    assert registered == []


def test_q5_swar_pair_source_preserves_retained_helper_and_integer_contract() -> None:
    source = _SOURCE.read_text()
    retained = source.split(
        "void gguf_q5_k_wave32x2_fixed_meta_decode_out_block(", 1
    )[1].split("uint32_t gguf_q5_k_swar_pair_u8(", 1)[0]
    candidate = source.split("uint32_t gguf_q5_k_swar_pair_u8(", 1)[1].split(
        "void gguf_q6_k_wave32x2_fixed_meta_decode_out_block(", 1
    )[0]
    assert "q5_k_quant(block0" in retained
    assert "gguf_q5_k_swar_quant_pair" not in retained
    assert "0x0F0Fu" in candidate
    assert "0x0101u" in candidate
    for offset in (16, 48, 80, 112, 144):
        assert f", {offset} + lane)" in candidate
    assert candidate.count("gguf_q5_k_swar_quant_pair<") == 2
    assert source.count(
        "gguf_q5_k_wave32x2_swar_pair_fixed_meta_decode_out_block("
    ) == 4


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("in_features", (256, 512, 1024, 3072, 6144, 9216))
def test_q5_swar_pair_direct_is_exact_for_all_frozen_k_boundaries(
    hip_context,
    in_features: int,
) -> None:
    x = _edge_bf16(in_features)
    weight = _edge_q5_weight(10, in_features, seed=0x5100 + in_features)
    for candidate_name, control, dtype, view_dtype in (
        (
            _DIRECT_BF16,
            kernel_mod.gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
            np.uint16,
            np.uint16,
        ),
        (
            _DIRECT_F32,
            kernel_mod.gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_f32_out,
            np.float32,
            np.uint32,
        ),
    ):
        expected = _run_direct(control, x, weight, dtype=dtype, library=hip_context)
        actual = _run_direct(
            _wrapper(candidate_name), x, weight, dtype=dtype, library=hip_context
        )
        np.testing.assert_array_equal(actual.view(view_dtype), expected.view(view_dtype))


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("in_features", (256, 1024, 3072, 9216))
def test_q5_swar_pair_unequal_pair_is_exact_for_bf16_and_f32(
    hip_context,
    in_features: int,
) -> None:
    x = _edge_bf16(in_features)
    weights = (
        _edge_q5_weight(16, in_features, seed=0x5200 + in_features),
        _edge_q5_weight(8, in_features, seed=0x5300 + in_features),
    )
    for candidate_name, control, dtype, view_dtype in (
        (
            _PAIR_BF16,
            kernel_mod.gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
            np.uint16,
            np.uint16,
        ),
        (
            _PAIR_F32,
            kernel_mod.gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_f32_out,
            np.float32,
            np.uint32,
        ),
    ):
        expected = _run_pair(control, x, weights, dtype=dtype, library=hip_context)
        actual = _run_pair(
            _wrapper(candidate_name), x, weights, dtype=dtype, library=hip_context
        )
        for observed, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(
                observed.view(view_dtype), reference.view(view_dtype)
            )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "in_features,dimensions",
    (
        (256, (8, 8, 8, 8)),
        (1024, (16, 8, 8, 8)),
        (3072, (16, 8, 8, 8)),
        (9216, (16, 8, 8, 8)),
    ),
)
def test_q5_swar_pair_mixed_is_exact_and_leaves_q6_outputs_unchanged(
    hip_context,
    in_features: int,
    dimensions: tuple[int, int, int, int],
) -> None:
    x = _edge_bf16(in_features)
    weights = (
        _edge_q5_weight(dimensions[0], in_features, seed=0x5400 + in_features),
        make_q6_k_weight(dimensions[1], in_features),
        make_q6_k_weight(dimensions[2], in_features),
        _edge_q5_weight(dimensions[3], in_features, seed=0x5500 + in_features),
    )
    expected = _run_mixed(
        kernel_mod.gguf_q5_q6_attention_q5_qg_mixed_local32_fixed_meta_gemv_decode_bf16_f32_out,
        x,
        weights,
        library=hip_context,
    )
    actual = _run_mixed(_wrapper(_MIXED), x, weights, library=hip_context)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            observed.view(np.uint32), reference.view(np.uint32)
        )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q5_swar_pair_passes_independent_cpu_kl_top1_gate(hip_context) -> None:
    rng = np.random.default_rng(2026072601)
    rows, in_features, out_features = 10, 1024, 20
    x_bits = _f32_to_bf16(rng.normal(0.0, 0.2, size=(rows, in_features)))
    weight = _edge_q5_weight(out_features, in_features, seed=0x5600)
    candidate = _wrapper(_DIRECT_BF16)
    control = kernel_mod.gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out
    actual = np.concatenate(
        [
            _run_direct(
                candidate,
                x_bits[row : row + 1],
                weight,
                dtype=np.uint16,
                library=hip_context,
            )
            for row in range(rows)
        ],
        axis=0,
    )
    expected = np.concatenate(
        [
            _run_direct(
                control,
                x_bits[row : row + 1],
                weight,
                dtype=np.uint16,
                library=hip_context,
            )
            for row in range(rows)
        ],
        axis=0,
    )
    np.testing.assert_array_equal(actual, expected)
    reference = gguf_quant_gemv(
        _bf16_to_f32(x_bits), weight, GGMLQuantizationType.Q5_K
    )
    quality = evaluate_logits(reference, _bf16_to_f32(actual))
    assert quality.kl_max <= 0.05
    assert quality.top1_agreement >= 0.90
