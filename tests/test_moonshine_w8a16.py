from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.registry import resolve
from hipengine.quant import (
    W8A16,
    dequantize_w8a16_per_output,
    quantize_w8a16_per_output,
    resolve_quant,
    w8a16_linear_fp16,
)


def test_w8a16_plugin_and_host_per_output_contract() -> None:
    assert resolve_quant("w8a16") is W8A16
    assert W8A16.weight_storage == "int8_row_major"
    assert W8A16.scale_granularity == "per_output_channel_symmetric_fp32"
    assert W8A16.compute_dtype == "fp32_accum_fp16_output"

    weight = np.asarray(
        [[-2.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    packed = quantize_w8a16_per_output(weight)
    assert packed.layout == "row_major_int8_per_output_channel_symmetric_f32_scale"
    assert packed.qweight.dtype == np.int8
    assert packed.scales.dtype == np.float32
    assert packed.qweight.flags.c_contiguous
    assert packed.scales.flags.c_contiguous
    assert packed.qweight.shape == weight.shape
    assert packed.scales.shape == (2,)
    assert packed.qweight[0].tolist() == [-127, -64, 0, 64]
    assert packed.scales[0] == np.float32(2.0 / 127.0)
    assert packed.scales[1] > 0
    np.testing.assert_array_equal(packed.qweight[1], 0)
    restored = dequantize_w8a16_per_output(packed.qweight, packed.scales)
    np.testing.assert_allclose(restored[0], weight[0], atol=float(packed.scales[0]) / 2)
    np.testing.assert_array_equal(restored[1], weight[1])
    assert packed.source_fp16_nbytes == 16
    assert packed.qweight_nbytes == 8
    assert packed.scale_nbytes == 8
    assert packed.packed_nbytes == 16


def test_w8a16_host_oracle_uses_fp16_boundaries_and_rejects_bad_input() -> None:
    x = np.asarray([[0.25, -0.5, 1.0, 2.0]], dtype=np.float16)
    weight = np.asarray(
        [[0.5, -0.25, 1.0, 0.125], [-1.0, 0.5, 0.25, -0.5]],
        dtype=np.float32,
    )
    packed = quantize_w8a16_per_output(weight)
    actual = w8a16_linear_fp16(x, packed.qweight, packed.scales)
    expected = (x.astype(np.float32) @ dequantize_w8a16_per_output(
        packed.qweight, packed.scales
    ).T).astype(np.float16)
    assert actual.dtype == np.float16
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="rank 2"):
        quantize_w8a16_per_output(np.zeros((2,), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        quantize_w8a16_per_output(np.asarray([[np.inf]], dtype=np.float32))
    with pytest.raises(ValueError, match="input width"):
        w8a16_linear_fp16(
            np.zeros((1, 3), dtype=np.float16), packed.qweight, packed.scales
        )


def test_moonshine_w8a16_family_selection_is_ordered_and_exact() -> None:
    from hipengine.loading.moonshine import (
        MOONSHINE_W8A16_COMPONENT_ORDER,
        MOONSHINE_W8A16_FAMILY_ORDER,
        moonshine_w8a16_source_names,
        normalize_moonshine_w8a16_families,
    )

    spec = SimpleNamespace(
        decoder_layers=8,
        embedding_weight_name="model.decoder.embed_tokens.weight",
    )
    assert MOONSHINE_W8A16_FAMILY_ORDER == ("lm_head", "mlp", "attention")
    assert MOONSHINE_W8A16_COMPONENT_ORDER == (
        "lm_head",
        "mlp_fc1",
        "mlp_fc2",
        "self_attention",
        "cross_attention",
    )
    assert normalize_moonshine_w8a16_families("lm_head,mlp") == (
        "lm_head",
        "mlp_fc1",
        "mlp_fc2",
    )
    assert normalize_moonshine_w8a16_families(("attention", "lm_head")) == (
        "lm_head",
        "self_attention",
        "cross_attention",
    )
    assert normalize_moonshine_w8a16_families(("mlp_fc2", "self_attention")) == (
        "mlp_fc2",
        "self_attention",
    )
    with pytest.raises(ValueError, match="unknown"):
        normalize_moonshine_w8a16_families(("encoder",))

    assert len(moonshine_w8a16_source_names(spec, ("lm_head",))) == 1
    assert len(moonshine_w8a16_source_names(spec, ("mlp",))) == 16
    attention = moonshine_w8a16_source_names(spec, ("attention",))
    assert len(attention) == 64
    assert "model.decoder.layers.0.self_attn.q_proj.weight" in attention
    assert "model.decoder.layers.7.encoder_attn.o_proj.weight" in attention


def test_moonshine_w8a16_kernel_registry_and_raw_pointer_abis(tmp_path) -> None:
    from hipengine.kernels.hip_gfx1100.linear.moonshine_w8a16 import (
        moonshine_w8a16_cross_kv_pair_head_major,
        moonshine_w8a16_lm_head_wave8,
        moonshine_w8a16_mlp_fc1_gated_silu,
        moonshine_w8a16_mlp_fc2_residual,
        moonshine_w8a16_projection,
        moonshine_w8a16_qkv_triple,
        plan_moonshine_w8a16_build,
        register_moonshine_w8a16_kernels,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_moonshine_w8a16_lm_head_wave8 = FakeKernel()
        hipengine_moonshine_w8a16_projection = FakeKernel()
        hipengine_moonshine_w8a16_mlp_fc1_gated_silu = FakeKernel()
        hipengine_moonshine_w8a16_mlp_fc2_residual = FakeKernel()
        hipengine_moonshine_w8a16_qkv_triple = FakeKernel()
        hipengine_moonshine_w8a16_cross_kv_pair_head_major = FakeKernel()

    register_moonshine_w8a16_kernels()
    expected = {
        ("moonshine_lm_head", "tied_wave8_per_output_f32_scale"): moonshine_w8a16_lm_head_wave8,
        ("moonshine_projection", "single_per_output_f32_scale"): moonshine_w8a16_projection,
        ("moonshine_mlp_fc1", "bias_gated_silu_per_output_f32_scale"): moonshine_w8a16_mlp_fc1_gated_silu,
        ("moonshine_mlp_fc2_residual", "bias_rounded_residual_per_output_f32_scale"): moonshine_w8a16_mlp_fc2_residual,
        ("moonshine_qkv_proj", "triple_per_output_f32_scale"): moonshine_w8a16_qkv_triple,
        ("moonshine_cross_kv_precompute", "pair_head_major_per_output_f32_scale"): moonshine_w8a16_cross_kv_pair_head_major,
    }
    for (layer, variant), function in expected.items():
        assert resolve(
            backend="hip_gfx1100", layer=layer, quant="w8a16", variant=variant
        ) is function

    library = FakeLibrary()
    common = {"stream": 9, "library": library, "runtime": object()}
    moonshine_w8a16_lm_head_wave8(1, 2, 3, 4, 1, 416, 36_864, **common)
    moonshine_w8a16_projection(1, 2, 3, 4, 1, 416, 416, threads=64, **common)
    moonshine_w8a16_mlp_fc1_gated_silu(
        1, 2, 3, 4, 5, 1, 416, 1664, **common
    )
    moonshine_w8a16_mlp_fc2_residual(
        1, 2, 3, 4, 5, 6, 1, 1664, 416, threads=64, **common
    )
    moonshine_w8a16_qkv_triple(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 416, 416, 416, 416,
        threads=32, **common
    )
    moonshine_w8a16_cross_kv_pair_head_major(
        1, 2, 3, 4, 5, 6, 7, 40, 416, 416, 416, 52,
        threads=32, **common
    )
    assert library.hipengine_moonshine_w8a16_lm_head_wave8.calls == [
        (1, 2, 3, 4, 1, 416, 36_864, 9)
    ]
    assert library.hipengine_moonshine_w8a16_projection.calls == [
        (1, 2, 3, 4, 1, 416, 416, 64, 9)
    ]
    assert library.hipengine_moonshine_w8a16_mlp_fc1_gated_silu.calls == [
        (1, 2, 3, 4, 5, 1, 416, 1664, 9)
    ]
    assert library.hipengine_moonshine_w8a16_mlp_fc2_residual.calls == [
        (1, 2, 3, 4, 5, 6, 1, 1664, 416, 64, 9)
    ]
    assert library.hipengine_moonshine_w8a16_qkv_triple.calls == [
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 416, 416, 416, 416, 32, 9)
    ]
    assert library.hipengine_moonshine_w8a16_cross_kv_pair_head_major.calls == [
        (1, 2, 3, 4, 5, 6, 7, 40, 416, 416, 416, 52, 32, 9)
    ]

    artifact = plan_moonshine_w8a16_build(
        cache_root=tmp_path / "cache", compiler_version="hipcc test"
    )
    assert artifact.family == "moonshine_w8a16"
    assert artifact.output_path.name == "moonshine_w8a16.so"
    assert not artifact.cache_dir.exists()


def _hip_available() -> bool:
    if os.environ.get("HIPENGINE_HIP_ARCH") != "gfx1151":
        return False
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_moonshine_w8a16_gpu_primitives_match_quantized_cpu_oracles() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.cpu_reference.moonshine import (
        moonshine_gated_silu,
        moonshine_residual,
    )
    from hipengine.kernels.hip_gfx1100.linear.moonshine_w8a16 import (
        build_moonshine_w8a16,
        moonshine_w8a16_cross_kv_pair_head_major,
        moonshine_w8a16_lm_head_wave8,
        moonshine_w8a16_mlp_fc1_gated_silu,
        moonshine_w8a16_mlp_fc2_residual,
        moonshine_w8a16_projection,
        moonshine_w8a16_qkv_triple,
    )

    rng = np.random.default_rng(0x8A16)
    hidden = 416
    intermediate = 128
    x = rng.normal(0.0, 0.05, size=(1, hidden)).astype(np.float16)
    rows = rng.normal(0.0, 0.05, size=(40, hidden)).astype(np.float16)
    weights = tuple(
        quantize_w8a16_per_output(
            rng.normal(0.0, 0.04, size=(hidden, hidden)).astype(np.float32)
        )
        for _ in range(3)
    )
    fc1 = quantize_w8a16_per_output(
        rng.normal(0.0, 0.04, size=(2 * intermediate, hidden)).astype(np.float32)
    )
    fc2 = quantize_w8a16_per_output(
        rng.normal(0.0, 0.04, size=(hidden, intermediate)).astype(np.float32)
    )
    fc1_bias = rng.normal(0.0, 0.02, size=(2 * intermediate,)).astype(np.float16)
    fc2_bias = rng.normal(0.0, 0.02, size=(hidden,)).astype(np.float16)
    residual = rng.normal(0.0, 0.05, size=(1, hidden)).astype(np.float16)
    expected = tuple(w8a16_linear_fp16(x, item.qweight, item.scales) for item in weights)
    expected_rows = tuple(
        w8a16_linear_fp16(rows, item.qweight, item.scales) for item in weights[:2]
    )
    expected_fc1 = moonshine_gated_silu(
        w8a16_linear_fp16(x, fc1.qweight, fc1.scales, fc1_bias)
    )
    expected_fc2 = moonshine_residual(
        residual,
        w8a16_linear_fp16(expected_fc1, fc2.qweight, fc2.scales, fc2_bias),
    )

    runtime = get_hip_runtime()
    library = build_moonshine_w8a16(load=True)
    allocations = []

    def upload(array: np.ndarray):
        host = np.ascontiguousarray(array)
        device = malloc(host.nbytes, runtime=runtime)
        allocations.append(device)
        copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
        return device

    def allocate(shape: tuple[int, ...]):
        device = malloc(
            int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float16).itemsize,
            runtime=runtime,
        )
        allocations.append(device)
        return device

    def download(device, shape: tuple[int, ...]) -> np.ndarray:
        host = np.empty(shape, dtype=np.float16)
        copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
        return host

    try:
        dx = upload(x)
        drows = upload(rows)
        dweights = tuple((upload(item.qweight), upload(item.scales)) for item in weights)
        dfc1 = (upload(fc1.qweight), upload(fc1.scales))
        dfc2 = (upload(fc2.qweight), upload(fc2.scales))
        dfc1_bias = upload(fc1_bias)
        dfc2_bias = upload(fc2_bias)
        dresidual = upload(residual)
        projection = allocate((1, hidden))
        lm_head = allocate((1, hidden))
        triple = tuple(allocate((1, hidden)) for _ in range(3))
        pair = tuple(allocate((8, 40, 52)) for _ in range(2))
        mlp_intermediate = allocate((1, intermediate))
        mlp_output = allocate((1, hidden))

        moonshine_w8a16_projection(
            dx.ptr, dweights[0][0].ptr, dweights[0][1].ptr, projection.ptr,
            1, hidden, hidden, library=library, runtime=runtime,
        )
        moonshine_w8a16_lm_head_wave8(
            dx.ptr, dweights[0][0].ptr, dweights[0][1].ptr, lm_head.ptr,
            1, hidden, hidden, library=library, runtime=runtime,
        )
        moonshine_w8a16_qkv_triple(
            dx.ptr,
            dweights[0][0].ptr, dweights[0][1].ptr,
            dweights[1][0].ptr, dweights[1][1].ptr,
            dweights[2][0].ptr, dweights[2][1].ptr,
            *(output.ptr for output in triple),
            1, hidden, hidden, hidden, hidden,
            library=library, runtime=runtime,
        )
        moonshine_w8a16_cross_kv_pair_head_major(
            drows.ptr,
            dweights[0][0].ptr, dweights[0][1].ptr,
            dweights[1][0].ptr, dweights[1][1].ptr,
            pair[0].ptr, pair[1].ptr,
            40, hidden, hidden, hidden, 52,
            library=library, runtime=runtime,
        )
        moonshine_w8a16_mlp_fc1_gated_silu(
            dx.ptr, dfc1[0].ptr, dfc1[1].ptr, dfc1_bias.ptr,
            mlp_intermediate.ptr, 1, hidden, intermediate,
            library=library, runtime=runtime,
        )
        moonshine_w8a16_mlp_fc2_residual(
            mlp_intermediate.ptr, dfc2[0].ptr, dfc2[1].ptr, dfc2_bias.ptr,
            dresidual.ptr, mlp_output.ptr, 1, intermediate, hidden,
            library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        actual_projection = download(projection, (1, hidden))
        actual_lm_head = download(lm_head, (1, hidden))
        actual_triple = tuple(download(output, (1, hidden)) for output in triple)
        actual_pair = tuple(download(output, (8, 40, 52)) for output in pair)
        actual_fc1 = download(mlp_intermediate, (1, intermediate))
        actual_fc2 = download(mlp_output, (1, hidden))
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual_projection, expected[0], rtol=3e-3, atol=3e-3)
    np.testing.assert_array_equal(actual_lm_head, actual_projection)
    for actual, reference in zip(actual_triple, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=3e-3, atol=3e-3)
    for actual, reference in zip(actual_pair, expected_rows, strict=True):
        np.testing.assert_allclose(
            actual, reference.reshape(40, 8, 52).transpose(1, 0, 2),
            rtol=3e-3, atol=3e-3,
        )
    np.testing.assert_allclose(actual_fc1, expected_fc1, rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(actual_fc2, expected_fc2, rtol=3e-3, atol=3e-3)
    assert all(
        np.isfinite(value).all()
        for value in (
            actual_projection, actual_lm_head, *actual_triple, *actual_pair,
            actual_fc1, actual_fc2,
        )
    )
