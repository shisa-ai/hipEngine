from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.moonshine import (
    moonshine_projection,
    moonshine_tied_lm_logits,
    moonshine_triple_projection,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def _cuda_sm120a_enabled() -> bool:
    if os.environ.get("HIPENGINE_RUN_CUDA_SM120A") != "1":
        return False
    if os.environ.get("HIPENGINE_CUDA_ARCH") != "sm_120a":
        return False
    try:
        ctypes.CDLL("libcudart.so.13")
    except OSError:
        return False
    return True


def setup_function() -> None:
    clear_registry_for_tests()


def test_moonshine_cuda_projection_registry_resolves_explicit_keys() -> None:
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        moonshine_f16_lm_head_projection,
        moonshine_f16_lm_head_projection_wave8,
        moonshine_f16_projection,
        moonshine_f16_projection_bias,
        moonshine_f16_projection_pair,
        moonshine_f16_projection_pair_head_major,
        moonshine_f16_projection_triple,
        register_moonshine_projection_kernels,
    )

    register_moonshine_projection_kernels()
    expected = {
        ("moonshine_projection", "single_fp32_accum"): moonshine_f16_projection,
        ("moonshine_lm_head", "tied_fp32_accum"): moonshine_f16_lm_head_projection,
        ("moonshine_lm_head", "tied_wave8_fp32_accum"): moonshine_f16_lm_head_projection_wave8,
        ("moonshine_projection_rows", "single_fp32_accum"): moonshine_f16_projection,
        ("moonshine_projection_bias", "single_fp32_accum"): moonshine_f16_projection_bias,
        ("moonshine_projection_pair", "pair_fp32_accum"): moonshine_f16_projection_pair,
        (
            "moonshine_cross_kv_precompute",
            "pair_head_major_fp32_accum",
        ): moonshine_f16_projection_pair_head_major,
        ("moonshine_qkv_proj", "triple_fp32_accum"): moonshine_f16_projection_triple,
    }
    for (layer, variant), function in expected.items():
        assert resolve(
            backend="cuda_sm120a",
            layer=layer,
            quant="fp16",
            variant=variant,
        ) is function


def test_moonshine_cuda_projection_build_plan_targets_sm120a(tmp_path) -> None:
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        plan_moonshine_projection_build,
    )

    artifact = plan_moonshine_projection_build(
        cache_root=tmp_path / "cache",
        compiler_version="nvcc Moonshine test version",
    )
    assert artifact.family == "cuda_sm120a_moonshine_projection"
    assert artifact.target_arch == "sm_120a"
    assert artifact.flags == ("-arch=sm_120a",)
    assert artifact.output_path.name == "moonshine_projection.so"
    assert not artifact.cache_dir.exists()


def test_moonshine_cuda_projection_wrappers_keep_raw_pointer_abi() -> None:
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        moonshine_f16_lm_head_projection,
        moonshine_f16_lm_head_projection_wave8,
        moonshine_f16_projection,
        moonshine_f16_projection_bias,
        moonshine_f16_projection_pair,
        moonshine_f16_projection_pair_head_major,
        moonshine_f16_projection_triple,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_cuda_sm120a_moonshine_f16_lm_head_projection = FakeKernel()
        hipengine_cuda_sm120a_moonshine_f16_lm_head_projection_wave8 = FakeKernel()
        hipengine_cuda_sm120a_moonshine_f16_projection = FakeKernel()
        hipengine_cuda_sm120a_moonshine_f16_projection_bias = FakeKernel()
        hipengine_cuda_sm120a_moonshine_f16_projection_pair = FakeKernel()
        hipengine_cuda_sm120a_moonshine_f16_projection_pair_head_major = FakeKernel()
        hipengine_cuda_sm120a_moonshine_f16_projection_triple = FakeKernel()

    library = FakeLibrary()
    common = {"threads": 256, "stream": 7, "library": library, "runtime": object()}
    moonshine_f16_projection(1, 2, 3, 1, 416, 416, **common)
    moonshine_f16_lm_head_projection(1, 2, 3, 1, 416, 36_864, **common)
    moonshine_f16_lm_head_projection_wave8(
        1, 2, 3, 1, 416, 36_864, stream=7, library=library, runtime=object()
    )
    moonshine_f16_projection_bias(1, 2, 3, 4, 1, 416, 416, **common)
    moonshine_f16_projection_pair(1, 2, 3, 4, 5, 1, 416, 416, 416, **common)
    moonshine_f16_projection_pair_head_major(
        1, 2, 3, 4, 5, 40, 416, 416, 416, 52, **common
    )
    moonshine_f16_projection_triple(
        1, 2, 3, 4, 5, 6, 7, 1, 416, 416, 416, 416, **common
    )
    assert library.hipengine_cuda_sm120a_moonshine_f16_projection.calls == [
        (1, 2, 3, 1, 416, 416, 256, 7)
    ]
    assert library.hipengine_cuda_sm120a_moonshine_f16_lm_head_projection.calls == [
        (1, 2, 3, 1, 416, 36_864, 256, 7)
    ]
    assert library.hipengine_cuda_sm120a_moonshine_f16_lm_head_projection_wave8.calls == [
        (1, 2, 3, 1, 416, 36_864, 7)
    ]
    assert library.hipengine_cuda_sm120a_moonshine_f16_projection_bias.calls == [
        (1, 2, 3, 4, 1, 416, 416, 256, 7)
    ]
    assert library.hipengine_cuda_sm120a_moonshine_f16_projection_pair.calls == [
        (1, 2, 3, 4, 5, 1, 416, 416, 416, 256, 7)
    ]
    assert (
        library.hipengine_cuda_sm120a_moonshine_f16_projection_pair_head_major.calls
        == [(1, 2, 3, 4, 5, 40, 416, 416, 416, 52, 256, 7)]
    )
    assert library.hipengine_cuda_sm120a_moonshine_f16_projection_triple.calls == [
        (1, 2, 3, 4, 5, 6, 7, 1, 416, 416, 416, 416, 256, 7)
    ]


def test_moonshine_cuda_projection_schedule_auto_selects_measured_threads() -> None:
    """Auto-select reflects the measured schedule; explicit ``threads=`` overrides.

    Batch-timed screen on exclusive GPU0: 64 threads is best for pair/head-major
    row projections across 40/207/1,248 rows (C1C-R1), and the fused fc2
    bias_residual is best at 256 threads for decode M=1 and 64 at M=40 (C1D-R2).
    """
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        moonshine_f16_projection_bias_residual,
        moonshine_f16_projection_pair,
        moonshine_f16_projection_pair_head_major,
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    class FakeLibrary:
        hipengine_cuda_sm120a_moonshine_f16_projection_pair = FakeKernel()
        hipengine_cuda_sm120a_moonshine_f16_projection_pair_head_major = FakeKernel()
        hipengine_cuda_sm120a_moonshine_f16_projection_bias_residual = FakeKernel()

    library = FakeLibrary()
    common = {"stream": 7, "library": library, "runtime": object()}

    # Pair/head-major auto-select 64 threads at every production row bucket.
    for rows in (40, 207, 1_248):
        moonshine_f16_projection_pair(
            1, 2, 3, 4, 5, rows, 416, 416, 416, **common
        )
        moonshine_f16_projection_pair_head_major(
            1, 2, 3, 4, 5, rows, 416, 416, 416, 52, **common
        )
    assert library.hipengine_cuda_sm120a_moonshine_f16_projection_pair.calls == [
        (1, 2, 3, 4, 5, rows, 416, 416, 416, 64, 7) for rows in (40, 207, 1_248)
    ]
    assert library.hipengine_cuda_sm120a_moonshine_f16_projection_pair_head_major.calls == [
        (1, 2, 3, 4, 5, rows, 416, 416, 416, 52, 64, 7)
        for rows in (40, 207, 1_248)
    ]

    # Fused fc2 auto-select: 256 at decode M=1, 64 at M=40; override wins.
    moonshine_f16_projection_bias_residual(1, 2, 3, 4, 5, 1, 1664, 416, **common)
    moonshine_f16_projection_bias_residual(1, 2, 3, 4, 5, 40, 1664, 416, **common)
    moonshine_f16_projection_bias_residual(
        1, 2, 3, 4, 5, 1, 1664, 416, threads=32, **common
    )
    assert (
        library.hipengine_cuda_sm120a_moonshine_f16_projection_bias_residual.calls
        == [(1, 2, 3, 4, 5, 1, 1664, 416, 256, 7), (1, 2, 3, 4, 5, 40, 1664, 416, 64, 7), (1, 2, 3, 4, 5, 1, 1664, 416, 32, 7)]
    )


def test_moonshine_cuda_projection_rejects_invalid_shapes_before_build() -> None:
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        moonshine_f16_projection,
        moonshine_f16_projection_pair_head_major,
    )

    with pytest.raises(ValueError, match="rows"):
        moonshine_f16_projection(1, 2, 3, 0, 416, 416)
    with pytest.raises(ValueError, match="threads"):
        moonshine_f16_projection(1, 2, 3, 1, 416, 416, threads=48)
    with pytest.raises(ValueError, match="head_dim"):
        moonshine_f16_projection_pair_head_major(1, 2, 3, 4, 5, 1, 416, 416, 416, 14)


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_projection_single_pair_triple_match_cpu_oracle() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        build_moonshine_projection,
        moonshine_f16_lm_head_projection,
        moonshine_f16_projection,
        moonshine_f16_projection_bias,
        moonshine_f16_projection_pair,
        moonshine_f16_projection_pair_head_major,
        moonshine_f16_projection_triple,
    )

    rng = np.random.default_rng(0x92B)
    hidden = 416
    x_one = rng.normal(0.0, 0.05, size=(1, hidden)).astype(np.float16)
    x_rows = rng.normal(0.0, 0.05, size=(40, hidden)).astype(np.float16)
    weights = tuple(
        rng.normal(0.0, 0.04, size=(hidden, hidden)).astype(np.float16)
        for _ in range(3)
    )
    bias = rng.normal(0.0, 0.03, size=(hidden,)).astype(np.float16)
    expected_one = moonshine_triple_projection(x_one, *weights)
    expected_bias = moonshine_projection(x_one, weights[0], bias)
    expected_rows = tuple(moonshine_projection(x_rows, weight) for weight in weights[:2])

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_projection(load=True)
    allocations = []
    try:
        dx_one = _upload(x_one, runtime, allocations)
        dx_rows = _upload(x_rows, runtime, allocations)
        device_weights = tuple(_upload(weight, runtime, allocations) for weight in weights)
        device_bias = _upload(bias, runtime, allocations)
        single = _alloc((1, hidden), runtime, allocations)
        lm_head = _alloc((1, hidden), runtime, allocations)
        biased = _alloc((1, hidden), runtime, allocations)
        triple = tuple(_alloc((1, hidden), runtime, allocations) for _ in range(3))
        pair = tuple(_alloc((40, hidden), runtime, allocations) for _ in range(2))
        head_major = tuple(_alloc((8, 40, 52), runtime, allocations) for _ in range(2))

        moonshine_f16_projection(
            dx_one.ptr, device_weights[0].ptr, single.ptr, 1, hidden, hidden,
            library=library, runtime=runtime,
        )
        moonshine_f16_lm_head_projection(
            dx_one.ptr, device_weights[0].ptr, lm_head.ptr, 1, hidden, hidden,
            library=library, runtime=runtime,
        )
        moonshine_f16_projection_bias(
            dx_one.ptr, device_weights[0].ptr, device_bias.ptr, biased.ptr,
            1, hidden, hidden, library=library, runtime=runtime,
        )
        moonshine_f16_projection_pair_head_major(
            dx_rows.ptr,
            device_weights[0].ptr,
            device_weights[1].ptr,
            head_major[0].ptr,
            head_major[1].ptr,
            40,
            hidden,
            hidden,
            hidden,
            52,
            library=library,
            runtime=runtime,
        )
        moonshine_f16_projection_triple(
            dx_one.ptr,
            *(weight.ptr for weight in device_weights),
            *(output.ptr for output in triple),
            1,
            hidden,
            hidden,
            hidden,
            hidden,
            library=library,
            runtime=runtime,
        )
        moonshine_f16_projection_pair(
            dx_rows.ptr,
            device_weights[0].ptr,
            device_weights[1].ptr,
            pair[0].ptr,
            pair[1].ptr,
            40,
            hidden,
            hidden,
            hidden,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_single = _download(single, (1, hidden), runtime)
        actual_lm_head = _download(lm_head, (1, hidden), runtime)
        actual_bias = _download(biased, (1, hidden), runtime)
        actual_triple = tuple(_download(output, (1, hidden), runtime) for output in triple)
        actual_pair = tuple(_download(output, (40, hidden), runtime) for output in pair)
        actual_head_major = tuple(
            _download(output, (8, 40, 52), runtime) for output in head_major
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual_single, expected_one[0], rtol=2e-3, atol=2e-3)
    np.testing.assert_array_equal(actual_lm_head, actual_single)
    np.testing.assert_allclose(actual_bias, expected_bias, rtol=2e-3, atol=2e-3)
    for actual, expected in zip(actual_triple, expected_one, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    for actual, expected in zip(actual_pair, expected_rows, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    for actual, expected in zip(actual_head_major, expected_rows, strict=True):
        expected_layout = expected.reshape(40, 8, 52).transpose(1, 0, 2)
        np.testing.assert_allclose(actual, expected_layout, rtol=2e-3, atol=2e-3)
    assert all(
        np.isfinite(value).all()
        for value in (*actual_triple, *actual_pair, *actual_head_major)
    )


@pytest.mark.skipif(not _cuda_sm120a_enabled(), reason="CUDA sm_120a gate is not enabled")
def test_moonshine_cuda_projection_bias_boundaries_and_lm_head_36k() -> None:
    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        build_moonshine_projection,
        moonshine_f16_lm_head_projection,
        moonshine_f16_lm_head_projection_wave8,
        moonshine_f16_projection_bias,
    )

    rng = np.random.default_rng(0xC1C1)
    hidden, vocab = 416, 36_864
    x = rng.normal(0.0, 0.05, size=(1, hidden)).astype(np.float16)
    fc1_weight = rng.normal(0.0, 0.04, size=(3328, hidden)).astype(np.float16)
    fc1_bias = rng.normal(0.0, 0.03, size=(3328,)).astype(np.float16)
    intermediate = rng.normal(0.0, 0.05, size=(1, 1664)).astype(np.float16)
    fc2_weight = rng.normal(0.0, 0.04, size=(hidden, 1664)).astype(np.float16)
    fc2_bias = rng.normal(0.0, 0.03, size=(hidden,)).astype(np.float16)
    expected_fc1 = moonshine_projection(x, fc1_weight, fc1_bias)
    expected_fc2 = moonshine_projection(intermediate, fc2_weight, fc2_bias)
    lm_weight = rng.normal(0.0, 0.03, size=(vocab, hidden)).astype(np.float16)
    expected_lm = moonshine_tied_lm_logits(x, lm_weight)

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_projection(load=True)
    allocations = []
    try:
        dx = _upload(x, runtime, allocations)
        dfc1_weight = _upload(fc1_weight, runtime, allocations)
        dfc1_bias = _upload(fc1_bias, runtime, allocations)
        dintermediate = _upload(intermediate, runtime, allocations)
        dfc2_weight = _upload(fc2_weight, runtime, allocations)
        dfc2_bias = _upload(fc2_bias, runtime, allocations)
        dlm_weight = _upload(lm_weight, runtime, allocations)
        fc1_out = _alloc((1, 3328), runtime, allocations)
        fc2_out = _alloc((1, hidden), runtime, allocations)
        lm_out = _alloc((1, vocab), runtime, allocations)
        lm_out_wave8 = _alloc((1, vocab), runtime, allocations)

        moonshine_f16_projection_bias(
            dx.ptr, dfc1_weight.ptr, dfc1_bias.ptr, fc1_out.ptr,
            1, hidden, 3328, library=library, runtime=runtime,
        )
        moonshine_f16_projection_bias(
            dintermediate.ptr, dfc2_weight.ptr, dfc2_bias.ptr, fc2_out.ptr,
            1, 1664, hidden, library=library, runtime=runtime,
        )
        moonshine_f16_lm_head_projection(
            dx.ptr, dlm_weight.ptr, lm_out.ptr,
            1, hidden, vocab, library=library, runtime=runtime,
        )
        moonshine_f16_lm_head_projection_wave8(
            dx.ptr, dlm_weight.ptr, lm_out_wave8.ptr,
            1, hidden, vocab, library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        actual_fc1 = _download(fc1_out, (1, 3328), runtime)
        actual_fc2 = _download(fc2_out, (1, hidden), runtime)
        actual_lm = _download(lm_out, (1, vocab), runtime)
        actual_lm_wave8 = _download(lm_out_wave8, (1, vocab), runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual_fc1, expected_fc1, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(actual_fc2, expected_fc2, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(actual_lm, expected_lm, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(actual_lm_wave8, actual_lm, rtol=2e-3, atol=2e-3)
    assert np.isfinite(actual_lm).all()
    assert np.isfinite(actual_lm_wave8).all()


_FIXTURE_DIR = os.environ.get(
    "HIPENGINE_MOONSHINE_FIXTURE_DIR",
    "/home/lhl/moonshine-prod-inference/results/raw/moonshine-fixtures",
)
_CHECKPOINT = os.environ.get(
    "HIPENGINE_MOONSHINE_CHECKPOINT",
    "/home/lhl/.cache/huggingface/hub/models--shisa-ai--shisa-realtime-asr-0.92b/snapshots/cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d/model.safetensors",
)
_FIXTURE_NAMES = ("audio-konichiwa-fp16", "synthetic-1s-seed1234-fp16")
# Observed worst max-absolute error on GPU0 (RTX PRO 6000 Blackwell) for the
# real-weight head-major cross K/V and tied LM logits against the pinned
# fixtures is exactly 2^-8 = 0.00390625 (matching the independent review
# diagnostic). The CUDA kernels are deterministic, so the gate asserts this
# Tier-B-style boundary directly.
_FIXTURE_ATOL = 0.00390625
_POSITIONS = (0, 1, 8, 32, 64, 128, 193)


def _projection_fixture_inputs_available() -> bool:
    return all(
        os.path.isfile(os.path.join(_FIXTURE_DIR, f"{name}.npz"))
        for name in _FIXTURE_NAMES
    ) and os.path.isfile(_CHECKPOINT)


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _projection_fixture_inputs_available(),
    reason="CUDA sm_120a gate or model-derived fixtures are not available",
)
def test_moonshine_cuda_projection_head_major_cross_kv_on_model_derived_fixtures() -> None:
    """All-layer head-major cross K/V matches the pinned model fixtures (C1C-R2)."""
    from safetensors import safe_open

    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        build_moonshine_projection,
        moonshine_f16_projection_pair_head_major,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_projection(load=True)
    allocations = []
    try:
        with safe_open(_CHECKPOINT, framework="np") as store:
            k_weights = [
                store.get_tensor(
                    f"model.decoder.layers.{n}.encoder_attn.k_proj.weight"
                ).astype(np.float16)
                for n in range(8)
            ]
            v_weights = [
                store.get_tensor(
                    f"model.decoder.layers.{n}.encoder_attn.v_proj.weight"
                ).astype(np.float16)
                for n in range(8)
            ]

        for name in _FIXTURE_NAMES:
            with np.load(os.path.join(_FIXTURE_DIR, f"{name}.npz")) as fx:
                encoder = fx["encoder.output"]  # (1, frames, 416)
                frames = int(encoder.shape[1])
                device_enc = _upload(encoder[0], runtime, allocations)
                device_k = _alloc((8, frames, 52), runtime, allocations)
                device_v = _alloc((8, frames, 52), runtime, allocations)
                for n in range(8):
                    device_kw = _upload(k_weights[n], runtime, allocations)
                    device_vw = _upload(v_weights[n], runtime, allocations)
                    moonshine_f16_projection_pair_head_major(
                        device_enc.ptr,
                        device_kw.ptr,
                        device_vw.ptr,
                        device_k.ptr,
                        device_v.ptr,
                        frames,
                        416,
                        416,
                        416,
                        52,
                        library=library,
                        runtime=runtime,
                    )
                    runtime.device_synchronize()
                    actual_k = _download(device_k, (8, frames, 52), runtime)
                    actual_v = _download(device_v, (8, frames, 52), runtime)
                    reference_k = fx[f"cross.layer_{n}.key"][0]
                    reference_v = fx[f"cross.layer_{n}.value"][0]
                    err_k = np.max(
                        np.abs(actual_k.astype(np.float32) - reference_k.astype(np.float32))
                    )
                    err_v = np.max(
                        np.abs(actual_v.astype(np.float32) - reference_v.astype(np.float32))
                    )
                    assert err_k <= _FIXTURE_ATOL, (name, n, "K", err_k)
                    assert err_v <= _FIXTURE_ATOL, (name, n, "V", err_v)
                    assert np.isfinite(actual_k).all()
                    assert np.isfinite(actual_v).all()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _projection_fixture_inputs_available(),
    reason="CUDA sm_120a gate or model-derived fixtures are not available",
)
def test_moonshine_cuda_projection_tied_lm_selected_tokens_on_model_derived_fixtures() -> None:
    """Tied LM head argmax reproduces every pinned selected token (C1C-R2)."""
    from safetensors import safe_open

    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
        build_moonshine_projection,
        moonshine_f16_lm_head_projection,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    library = build_moonshine_projection(load=True)
    allocations = []
    try:
        with safe_open(_CHECKPOINT, framework="np") as store:
            lm_weight = store.get_tensor(
                "model.decoder.embed_tokens.weight"
            ).astype(np.float16)
        device_lm = _upload(lm_weight, runtime, allocations)
        device_logits = _alloc((1, 36_864), runtime, allocations)

        for name in _FIXTURE_NAMES:
            with np.load(os.path.join(_FIXTURE_DIR, f"{name}.npz")) as fx:
                for pos in _POSITIONS:
                    final_hidden = fx[f"decoder.position_{pos}.final_hidden"]
                    reference_logits = fx[f"decoder.position_{pos}.logits"][0]
                    reference_token = int(
                        fx[f"decoder.position_{pos}.selected_token"][0, 0]
                    )
                    device_input = _upload(final_hidden[0, 0], runtime, allocations)
                    moonshine_f16_lm_head_projection(
                        device_input.ptr,
                        device_lm.ptr,
                        device_logits.ptr,
                        1,
                        416,
                        36_864,
                        library=library,
                        runtime=runtime,
                    )
                    runtime.device_synchronize()
                    actual_logits = _download(device_logits, (1, 36_864), runtime)
                    actual_token = int(np.argmax(actual_logits[0]))
                    err = np.max(
                        np.abs(
                            actual_logits[0].astype(np.float32)
                            - reference_logits.astype(np.float32)
                        )
                    )
                    assert err <= _FIXTURE_ATOL, (name, pos, err)
                    assert actual_token == reference_token, (name, pos, actual_token, reference_token)
                    assert np.isfinite(actual_logits).all()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape: tuple[int, ...], runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(np.float16).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape: tuple[int, ...], runtime) -> np.ndarray:
    host = np.empty(shape, dtype=np.float16)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
