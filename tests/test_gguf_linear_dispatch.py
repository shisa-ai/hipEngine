from __future__ import annotations

from types import SimpleNamespace

import pytest

# Import built-ins so the registry has real kernels to restore after overrides.
import hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv  # noqa: F401
import hipengine.runtime.gguf_linear as gguf_linear_module
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.registry import KernelKey, register, resolve, unregister
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q4_K_QMICRO_T16,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q5_K_T16,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
)
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_BF16,
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_BF16,
    GGUF_OUTPUT_F32,
    GGUF_OUTPUT_FP16,
    Q6T16F16RocblasPrefillSession,
    launch_gguf_q4_t16_sidecar_decode,
    launch_gguf_linear,
    launch_gguf_linear_moe_tail_host_batch,
    launch_gguf_linear_pair,
    launch_gguf_linear_pair_concat,
    launch_gguf_linear_pair_silu,
    launch_gguf_linear_triple,
    native_batch_decode_session,
    q4_pack8_dual_wmma_silu_prefill_session,
    q4_t16_unequal_pair_prefill_session,
    q6_t16_f16_rocblas_prefill_session,
    q8_mmq_prefill_session,
    resolve_gguf_linear_dispatch,
    resolve_q8_mmq_prefill_policy,
    set_wmma_prefill_enabled,
    target_verifier_production_q4_rowtile_session,
    wmma_prefill_session,
)
from hipengine.runtime.prefill import PrefillConfig


@pytest.fixture(autouse=True)
def _isolate_wmma_axis_from_rowtile():
    """These tests exercise the WMMA-prefill axis. The default-on raw row-tile
    rewrite is a separate small-B path (covered in test_gguf_{q4_k,k}_rowtile_gemv);
    disable it here so the WMMA on/off assertions see the per-row baseline."""

    from hipengine.runtime.gguf_linear import set_q4k_rowtile_enabled

    set_q4k_rowtile_enabled(False)
    try:
        yield
    finally:
        set_q4k_rowtile_enabled(None)


def _fake_weight(
    *,
    layout: str,
    quant_key: str,
    decode_tiles: bool = False,
    decode_tiles_dual: bool = False,
):
    allocations = {
        "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=10)),
        "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
        "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=12)),
        "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=13)),
        "tiles": SimpleNamespace(tensor=SimpleNamespace(ptr=14)),
    }
    if decode_tiles:
        allocations["decode_tiles"] = SimpleNamespace(
            tensor=SimpleNamespace(ptr=15)
        )
    if decode_tiles_dual:
        allocations["decode_tiles_dual"] = SimpleNamespace(
            tensor=SimpleNamespace(ptr=16)
        )

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(layout=layout, quant_key=quant_key)
            self.allocations = allocations

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


def test_resolve_gguf_linear_dispatch_uses_weight_quant_for_raw_layouts() -> None:
    q4 = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    q5 = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k")
    q6 = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k")
    q41 = _fake_weight(layout=LAYOUT_DENSE_BF16, quant_key="gguf_q4_1")
    f32 = _fake_weight(layout=LAYOUT_DENSE_F32, quant_key="f32")

    assert resolve_gguf_linear_dispatch(q4).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q4_k", "pack8_bf16_bf16_out"
    )
    assert resolve_gguf_linear_dispatch(q5).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q5_k", "gemv_bf16_bf16_out"
    )
    assert resolve_gguf_linear_dispatch(q6, output_dtype=GGUF_OUTPUT_F32).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q6_k", "gemv_bf16_f32_out"
    )
    assert resolve_gguf_linear_dispatch(q4, rows=4).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q4_k", "pack8_prefill_bf16_bf16_out"
    )
    assert resolve_gguf_linear_dispatch(q5, rows=4, output_dtype=GGUF_OUTPUT_FP16).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q5_k", "prefill_bf16_fp16_out"
    )
    assert resolve_gguf_linear_dispatch(q41, rows=4).key == KernelKey(
        "hip_gfx1100", "dense_gemv", "bf16", "prefill_out"
    )
    assert resolve_gguf_linear_dispatch(q41, output_dtype=GGUF_OUTPUT_F32).key == KernelKey(
        "hip_gfx1100", "dense_gemv", "bf16", "f32_out"
    )
    assert resolve_gguf_linear_dispatch(f32).key == KernelKey(
        "hip_gfx1100", "dense_gemv", "f32", "bf16_hidden_bf16_out"
    )
    assert resolve_gguf_linear_dispatch(
        f32,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
    ).key == KernelKey("hip_gfx1100", "dense_gemv", "f32", "f32_hidden_f32_out")
    q5_t16 = _fake_weight(layout=LAYOUT_GGUF_Q5_K_T16, quant_key="gguf_q5_k_t16_v1")
    assert resolve_gguf_linear_dispatch(q5_t16).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q5_k_t16_v1", "t16_gemv_decode_bf16_bf16_out"
    )
    q6_t16 = _fake_weight(layout=LAYOUT_GGUF_Q6_K_T16, quant_key="gguf_q6_k_t16_v1")
    assert resolve_gguf_linear_dispatch(q6_t16).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_gemv_decode_bf16_bf16_out"
    )
    assert resolve_gguf_linear_dispatch(q6_t16, output_dtype=GGUF_OUTPUT_F32).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_gemv_decode_bf16_f32_out"
    )
    q6_qmicro = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    assert resolve_gguf_linear_dispatch(
        q6_qmicro,
        output_dtype=GGUF_OUTPUT_F32,
    ).key == KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_gemv_decode_bf16_f32_out",
    )
    q8_t16 = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    assert resolve_gguf_linear_dispatch(q8_t16).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_bf16_bf16_out"
    )
    assert resolve_gguf_linear_dispatch(q8_t16, output_dtype=GGUF_OUTPUT_FP16).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_fp16_fp16_out"
    )
    assert resolve_gguf_linear_dispatch(q8_t16, activation_dtype=GGUF_ACTIVATION_F32).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_f32_bf16_out"
    )


def test_gfx1100_t16_f16_rocblas_solution_policy_is_version_and_shape_scoped() -> None:
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_T16_F16_ROCBLAS_SOLUTION_VERSION_PREFIX",
        None,
    ) == "5.2.0.dabb6df2b98"
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_T16_F16_ROCBLAS_SOLUTION_INDICES",
        None,
    ) == {
        (512, 5_120, 1_024): -1_140_856_081,
        (512, 5_120, 2_048): -1_140_856_092,
        (4_096, 5_120, 512): -1_140_855_996,
        (4_096, 17_408, 512): -1_140_855_997,
        (4_096, 6_144, 512): -1_140_855_996,
    }
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_Q5_T16_F16_ROCBLAS_PREFILL_POLICIES",
        None,
    ) == {
        (6_144, 5_120): {512: 1_280, 1_024: 1_280, 4_096: 1_024},
    }
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_T16_F16_ROCBLAS_MAX_ROWS_BY_QUANT_SHAPE",
        None,
    ) == {
        "gguf_q4_k_t16_v1": {(5_120, 12_288): 2_047},
    }
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_T16_F16_ROCBLAS_SOLUTION_INDICES",
        None,
    ) is None
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_Q5_T16_F16_ROCBLAS_PREFILL_POLICIES",
        None,
    ) == {
        (6_144, 5_120): {512: 1_280, 1_024: 1_280, 4_096: 1_024},
    }
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_T16_F16_ROCBLAS_VARIANT_POLICIES",
        None,
    ) == {
        "gguf_q4_k_t16_v1": {
            (17_408, 5_120): {
                (512, 4_096): "f16_rocblas_t16_pair_bf16_bf16_out",
            },
            (5_120, 1_024): {
                (512, 1_024): "f16_rocblas_t16_pair_bf16_bf16_out",
                (4_096, 4_096): "f16_rocblas_t16_pair_bf16_bf16_out",
            },
            (6_144, 5_120): {
                (512, 768): "f16_rocblas_t16_pair_bf16_bf16_out",
            },
            (5_120, 12_288): {
                (512, 2_047): "f16_rocblas_t16_pair_bf16_bf16_out",
            },
        },
        "gguf_q5_k_t16_v1": {
            (6_144, 5_120): {
                (512, 4_096): "f16_rocblas_t16_octet_bf16_bf16_out",
            },
        },
    }
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_T16_F16_ROCBLAS_VARIANT_POLICIES",
        None,
    ) == {
        "gguf_q5_k_t16_v1": {
            (6_144, 5_120): {
                (512, 4_096): "f16_rocblas_t16_octet_bf16_bf16_out",
            },
        },
    }
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_T16_F16_ROCBLAS_PAIR_ONLY_POLICIES",
        None,
    ) == {
        (
            "gguf_q6_k_t16_qmicro_planar_v1",
            5_120,
            10_240,
            "gguf_q4_k_t16_v1",
            6_144,
        ): {
            (512, 1_023): (
                2_048,
                "f16_rocblas_t16_pair_bf16_bf16_out",
                False,
            ),
            (1_024, 2_047): (
                512,
                "f16_rocblas_t16_pair_bf16_bf16_out",
                False,
            ),
        },
    }
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_T16_F16_ROCBLAS_PAIR_ONLY_POLICIES",
        None,
    ) is None


def test_q6_t16_f16_rocblas_context_routes_only_bounded_planar_prefill() -> None:
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    exact_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    )
    candidate_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "f16_rocblas_t16_qmicro_planar_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (exact_key, candidate_key)
    }
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def capture(label: str):
        def fake(*args, **kwargs):
            calls.append((label, args, kwargs))

        return fake

    session = Q6T16F16RocblasPrefillSession(
        min_rows=512,
        max_rows=4096,
        x_f16_ptr=0x30000000,
        x_f16_nbytes=4096 * 17_408 * 2,
        weight_f16_ptr=0x40000000,
        weight_f16_nbytes=2048 * 17_408 * 2,
        out_f16_ptr=0x50000000,
        out_f16_nbytes=4096 * 2048 * 2,
        tile_out_features_by_shape={(512, 5120, 10240): 2048},
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
    )
    register(exact_key, capture("exact"), replace=True)
    register(candidate_key, capture("candidate"), replace=True)
    try:
        # Outside the owner context, the production exact WMMA route remains.
        launch_gguf_linear(
            weight,
            x_ptr=0x10000000,
            out_ptr=0x20000000,
            rows=512,
            in_features=5120,
            out_features=10240,
            use_wmma_prefill=True,
            stream=7,
            runtime="runtime-sentinel",
        )
        with q6_t16_f16_rocblas_prefill_session(session):
            launch_gguf_linear(
                weight,
                x_ptr=0x10000000,
                out_ptr=0x20000000,
                rows=512,
                in_features=5120,
                out_features=10240,
                use_wmma_prefill=True,
                stream=7,
                runtime="runtime-sentinel",
            )
            # Ordinary row counts inherit the nearest measured policy anchor.
            launch_gguf_linear(
                weight,
                x_ptr=0x10000000,
                out_ptr=0x20000000,
                rows=513,
                in_features=5120,
                out_features=10240,
                use_wmma_prefill=True,
                stream=7,
                runtime="runtime-sentinel",
            )
            # Decode/verifier-sized rows remain exact even inside the context.
            launch_gguf_linear(
                weight,
                x_ptr=0x10000000,
                out_ptr=0x20000000,
                rows=4,
                in_features=5120,
                out_features=10240,
                use_wmma_prefill=True,
                stream=7,
                runtime="runtime-sentinel",
            )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert [label for label, _args, _kwargs in calls] == [
        "exact",
        "candidate",
        "candidate",
        "exact",
    ]
    _, args, kwargs = calls[1]
    assert args == (
        0x10000000,
        14,
        0x20000000,
        0x30000000,
        0x40000000,
        0x50000000,
        512,
        5120,
        10240,
    )
    assert kwargs == {
        "tile_out_features": 2048,
        "stream": 7,
        "dequant_library": "dequant-library",
        "cast_library": "cast-library",
        "rocblas": "rocblas-handle",
        "solution_index": None,
        "cast_activation": True,
        "runtime": "runtime-sentinel",
    }


def test_q5_t16_f16_rocblas_context_routes_only_admitted_shapes() -> None:
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q5_K_T16,
        quant_key="gguf_q5_k_t16_v1",
    )
    exact_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q5_k_t16_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    )
    candidate_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q5_k_t16_v1",
        "f16_rocblas_t16_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (exact_key, candidate_key)
    }
    calls = []

    def capture(label):
        def fake(*args, **kwargs):
            calls.append((label, args, kwargs))

        return fake

    session = Q6T16F16RocblasPrefillSession(
        min_rows=512,
        max_rows=4096,
        x_f16_ptr=0x30000000,
        x_f16_nbytes=4096 * 6144 * 2,
        weight_f16_ptr=0x40000000,
        weight_f16_nbytes=1280 * 6144 * 2,
        out_f16_ptr=0x50000000,
        out_f16_nbytes=4096 * 1280 * 2,
        tile_out_features_by_shape={(512, 17_408, 5120): 512},
        q5_tile_out_features_by_shape={
            (512, 6144, 5120): 1280,
            (4096, 6144, 5120): 1024,
        },
        q5_x_inplace_shapes=frozenset(
            {(512, 6144, 5120), (4096, 6144, 5120)}
        ),
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
    )
    register(exact_key, capture("exact"), replace=True)
    register(candidate_key, capture("candidate"), replace=True)
    try:
        with q6_t16_f16_rocblas_prefill_session(session):
            launch_gguf_linear(
                weight,
                x_ptr=0x10000000,
                out_ptr=0x20000000,
                rows=512,
                in_features=6144,
                out_features=5120,
                backend="hip_gfx1100",
                activation_dtype=GGUF_ACTIVATION_BF16,
                output_dtype=GGUF_OUTPUT_BF16,
                use_wmma_prefill=True,
            )
            launch_gguf_linear(
                weight,
                x_ptr=0x10000000,
                out_ptr=0x20000000,
                rows=256,
                in_features=6144,
                out_features=5120,
                backend="hip_gfx1100",
                activation_dtype=GGUF_ACTIVATION_BF16,
                output_dtype=GGUF_OUTPUT_BF16,
                use_wmma_prefill=True,
            )
    finally:
        for key, fn in originals.items():
            register(key, fn, replace=True)

    assert [label for label, _args, _kwargs in calls] == ["candidate", "exact"]
    _, args, kwargs = calls[0]
    assert args[:4] == (
        0x10000000,
        weight.allocation("tiles").tensor.ptr,
        0x20000000,
        0x10000000,
    )
    assert kwargs["tile_out_features"] == 1280


def test_q4_t16_f16_rocblas_context_routes_only_admitted_shapes() -> None:
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    exact_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    )
    candidate_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "f16_rocblas_t16_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (exact_key, candidate_key)
    }
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def capture(label: str):
        def fake(*args, **kwargs):
            calls.append((label, args, kwargs))

        return fake

    session = Q6T16F16RocblasPrefillSession(
        min_rows=512,
        max_rows=4096,
        x_f16_ptr=0x30000000,
        x_f16_nbytes=4096 * 17_408 * 2,
        weight_f16_ptr=0x40000000,
        weight_f16_nbytes=2048 * 17_408 * 2,
        out_f16_ptr=0x50000000,
        out_f16_nbytes=4096 * 2048 * 2,
        tile_out_features_by_shape={(512, 5120, 10240): 2048},
        q4_tile_out_features_by_shape={(512, 17_408, 5120): 1024},
        q4_x_inplace_shapes={(512, 17_408, 5120)},
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
    )
    register(exact_key, capture("exact"), replace=True)
    register(candidate_key, capture("candidate"), replace=True)
    try:
        with q6_t16_f16_rocblas_prefill_session(session):
            launch_gguf_linear(
                weight,
                x_ptr=0x10000000,
                out_ptr=0x20000000,
                rows=512,
                in_features=17_408,
                out_features=5_120,
                use_wmma_prefill=True,
                stream=7,
                runtime="runtime-sentinel",
            )
            launch_gguf_linear(
                weight,
                x_ptr=0x10000000,
                out_ptr=0x20000000,
                rows=512,
                in_features=5_120,
                out_features=17_408,
                use_wmma_prefill=True,
                stream=7,
                runtime="runtime-sentinel",
            )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert [label for label, _args, _kwargs in calls] == ["candidate", "exact"]
    _, args, kwargs = calls[0]
    assert args == (
        0x10000000,
        14,
        0x20000000,
        0x10000000,
        0x40000000,
        0x50000000,
        512,
        17_408,
        5_120,
    )
    assert kwargs["tile_out_features"] == 1024


def test_q4_t16_f16_rocblas_variant_policy_is_fail_closed_by_row_interval() -> None:
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    exact_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    )
    scalar_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "f16_rocblas_t16_bf16_bf16_out",
    )
    pair_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "f16_rocblas_t16_pair_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (exact_key, scalar_key, pair_key)
    }
    calls: list[str] = []
    session = Q6T16F16RocblasPrefillSession(
        min_rows=512,
        max_rows=4096,
        x_f16_ptr=0x30000000,
        x_f16_nbytes=4096 * 17_408 * 2,
        weight_f16_ptr=0x40000000,
        weight_f16_nbytes=1024 * 17_408 * 2,
        out_f16_ptr=0x50000000,
        out_f16_nbytes=4096 * 1024 * 2,
        tile_out_features_by_shape={(512, 5120, 10240): 2048},
        q4_tile_out_features_by_shape={
            (512, 17_408, 5_120): 1_024,
            (512, 6_144, 5_120): 1_024,
            (512, 5_120, 12_288): 2_048,
            (1_024, 5_120, 12_288): 512,
        },
        max_rows_by_quant_shape={
            "gguf_q4_k_t16_v1": {(5_120, 12_288): 2_047},
        },
        linear_variant_intervals_by_quant={
            "gguf_q4_k_t16_v1": {
                (17_408, 5_120): {(512, 4_096): pair_key.variant},
                (6_144, 5_120): {(512, 768): pair_key.variant},
                (5_120, 12_288): {(512, 2_047): pair_key.variant},
            },
        },
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
    )
    register(exact_key, lambda *args, **kwargs: calls.append("exact"), replace=True)
    register(scalar_key, lambda *args, **kwargs: calls.append("scalar"), replace=True)
    register(pair_key, lambda *args, **kwargs: calls.append("pair"), replace=True)
    try:
        with q6_t16_f16_rocblas_prefill_session(session):
            for rows, hidden, outputs in (
                (512, 17_408, 5_120),
                (4_096, 17_408, 5_120),
                (768, 6_144, 5_120),
                (769, 6_144, 5_120),
                (512, 5_120, 12_288),
                (1_024, 5_120, 12_288),
                (2_047, 5_120, 12_288),
                (2_048, 5_120, 12_288),
                (4_096, 5_120, 12_288),
            ):
                launch_gguf_linear(
                    weight,
                    x_ptr=0x10000000,
                    out_ptr=0x20000000,
                    rows=rows,
                    in_features=hidden,
                    out_features=outputs,
                    use_wmma_prefill=True,
                    runtime="runtime-sentinel",
                )
            unregister(pair_key)
            gguf_linear_module.clear_gguf_linear_dispatch_cache()
            launch_gguf_linear(
                weight,
                x_ptr=0x10000000,
                out_ptr=0x20000000,
                rows=512,
                in_features=17_408,
                out_features=5_120,
                use_wmma_prefill=True,
                runtime="runtime-sentinel",
            )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        "pair",
        "pair",
        "pair",
        "scalar",
        "pair",
        "pair",
        "pair",
        "exact",
        "exact",
        "scalar",
    ]


def test_q5_t16_f16_rocblas_variant_policy_routes_octet_owner() -> None:
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q5_K_T16,
        quant_key="gguf_q5_k_t16_v1",
    )
    scalar_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q5_k_t16_v1",
        "f16_rocblas_t16_bf16_bf16_out",
    )
    octet_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q5_k_t16_v1",
        "f16_rocblas_t16_octet_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (scalar_key, octet_key)
    }
    calls: list[str] = []
    session = Q6T16F16RocblasPrefillSession(
        min_rows=512,
        max_rows=4096,
        x_f16_ptr=0x30000000,
        x_f16_nbytes=4096 * 6_144 * 2,
        weight_f16_ptr=0x40000000,
        weight_f16_nbytes=1280 * 6_144 * 2,
        out_f16_ptr=0x50000000,
        out_f16_nbytes=4096 * 1280 * 2,
        tile_out_features_by_shape={},
        q5_tile_out_features_by_shape={
            (512, 6_144, 5_120): 1_280,
            (1_024, 6_144, 5_120): 1_280,
            (4_096, 6_144, 5_120): 1_024,
        },
        q5_x_inplace_shapes=frozenset(
            {
                (512, 6_144, 5_120),
                (1_024, 6_144, 5_120),
                (4_096, 6_144, 5_120),
            }
        ),
        linear_variant_intervals_by_quant={
            "gguf_q5_k_t16_v1": {
                (6_144, 5_120): {(512, 4_096): octet_key.variant},
            },
        },
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
    )
    register(scalar_key, lambda *args, **kwargs: calls.append("scalar"), replace=True)
    register(octet_key, lambda *args, **kwargs: calls.append("octet"), replace=True)
    try:
        with q6_t16_f16_rocblas_prefill_session(session):
            for rows in (512, 1_024, 4_096):
                launch_gguf_linear(
                    weight,
                    x_ptr=0x10000000,
                    out_ptr=0x20000000,
                    rows=rows,
                    in_features=6_144,
                    out_features=5_120,
                    use_wmma_prefill=True,
                    runtime="runtime-sentinel",
                )
            unregister(octet_key)
            gguf_linear_module.clear_gguf_linear_dispatch_cache()
            launch_gguf_linear(
                weight,
                x_ptr=0x10000000,
                out_ptr=0x20000000,
                rows=512,
                in_features=6_144,
                out_features=5_120,
                use_wmma_prefill=True,
                runtime="runtime-sentinel",
            )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == ["octet", "octet", "octet", "scalar"]


def test_t16_f16_rocblas_variant_policy_rejects_overlapping_intervals() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        Q6T16F16RocblasPrefillSession(
            min_rows=512,
            max_rows=4096,
            x_f16_ptr=1,
            x_f16_nbytes=1,
            weight_f16_ptr=2,
            weight_f16_nbytes=1,
            out_f16_ptr=3,
            out_f16_nbytes=1,
            tile_out_features_by_shape={(512, 5_120, 10_240): 2_048},
            q4_tile_out_features_by_shape={(512, 17_408, 5_120): 1_024},
            linear_variant_intervals_by_quant={
                "gguf_q4_k_t16_v1": {
                    (17_408, 5_120): {
                        (512, 1_024): "candidate_a",
                        (768, 4_096): "candidate_b",
                    },
                },
            },
            dequant_library="dequant-library",
            cast_library="cast-library",
            rocblas="rocblas-handle",
        )


def test_q6_qkv_q4_gate_pair_reuses_pair_producer_and_one_activation_cast() -> None:
    q6_weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    q4_weight = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    q6_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "f16_rocblas_t16_qmicro_planar_bf16_bf16_out",
    )
    q4_pair_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "f16_rocblas_t16_pair_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (q6_key, q4_pair_key)
    }
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def capture(label: str):
        def fake(*args, **kwargs):
            calls.append((label, args, kwargs))

        return fake

    session = Q6T16F16RocblasPrefillSession(
        min_rows=512,
        max_rows=4096,
        x_f16_ptr=0x30000000,
        x_f16_nbytes=4096 * 5120 * 2,
        weight_f16_ptr=0x40000000,
        weight_f16_nbytes=2048 * 5120 * 2,
        out_f16_ptr=0x50000000,
        out_f16_nbytes=4096 * 2048 * 2,
        tile_out_features_by_shape={
            (512, 5120, 10240): 2048,
            (1024, 5120, 10240): 512,
        },
        pair_only_second_operand_policies={
            (
                "gguf_q6_k_t16_qmicro_planar_v1",
                5120,
                10240,
                "gguf_q4_k_t16_v1",
                6144,
            ): {
                (512, 1023): (2048, q4_pair_key.variant, False),
                (1024, 2047): (512, q4_pair_key.variant, False),
            },
        },
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
        solution_indices_by_gemm_shape={
            (512, 5120, 2048): -1_140_856_092,
        },
    )
    register(q6_key, capture("q6"), replace=True)
    register(q4_pair_key, capture("q4_pair"), replace=True)
    try:
        with q6_t16_f16_rocblas_prefill_session(session):
            assert launch_gguf_linear_pair(
                q6_weight,
                q4_weight,
                x_ptr=0x10000000,
                out_a_ptr=0x20000000,
                out_b_ptr=0x22000000,
                rows=512,
                in_features=5120,
                out_features=10240,
                out_features_b=6144,
                use_wmma_prefill=True,
                stream=7,
                runtime="runtime-sentinel",
            )
            assert not launch_gguf_linear_pair(
                q6_weight,
                q4_weight,
                x_ptr=0x10000000,
                out_a_ptr=0x20000000,
                out_b_ptr=0x22000000,
                rows=2048,
                in_features=5120,
                out_features=10240,
                out_features_b=6144,
                use_wmma_prefill=True,
                stream=7,
                runtime="runtime-sentinel",
            )
            unregister(q4_pair_key)
            gguf_linear_module.clear_gguf_linear_dispatch_cache()
            assert not launch_gguf_linear_pair(
                q6_weight,
                q4_weight,
                x_ptr=0x10000000,
                out_a_ptr=0x20000000,
                out_b_ptr=0x22000000,
                rows=512,
                in_features=5120,
                out_features=10240,
                out_features_b=6144,
                use_wmma_prefill=True,
                stream=7,
                runtime="runtime-sentinel",
            )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert [label for label, _args, _kwargs in calls] == ["q6", "q4_pair"]
    assert calls[0][2]["cast_activation"] is True
    assert calls[1][2]["cast_activation"] is False
    assert all(
        kwargs["solution_index"] == -1_140_856_092
        for _label, _args, kwargs in calls
    )


def test_q4_t16_f16_rocblas_inplace_policy_is_shape_scoped() -> None:
    session = Q6T16F16RocblasPrefillSession(
        min_rows=512,
        max_rows=4096,
        x_f16_ptr=0x10000000,
        x_f16_nbytes=4096 * 17_408 * 2,
        weight_f16_ptr=0x20000000,
        weight_f16_nbytes=1024 * 17_408 * 2,
        out_f16_ptr=0x30000000,
        out_f16_nbytes=4096 * 1024 * 2,
        tile_out_features_by_shape={(512, 5120, 10240): 1024},
        q4_tile_out_features_by_shape={
            (512, 17_408, 5120): 1024,
            (512, 5120, 1024): 1024,
        },
        q4_x_inplace_shapes={(512, 17_408, 5120)},
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
    )

    assert session.activation_is_inplace(
        512, 17_408, 5120, quant="gguf_q4_k_t16_v1"
    )
    assert not session.activation_is_inplace(
        512, 5120, 1024, quant="gguf_q4_k_t16_v1"
    )


def test_q6_t16_f16_rocblas_context_supports_bounded_inplace_down_activation() -> None:
    session = Q6T16F16RocblasPrefillSession(
        min_rows=512,
        max_rows=4096,
        x_f16_ptr=0x30000000,
        x_f16_nbytes=512 * 17_408 * 2,
        weight_f16_ptr=0x40000000,
        weight_f16_nbytes=1024 * 17_408 * 2,
        out_f16_ptr=0x50000000,
        out_f16_nbytes=512 * 2048 * 2,
        tile_out_features_by_shape={(512, 17_408, 5_120): 1024},
        x_inplace_shapes=frozenset({(512, 17_408, 5_120)}),
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
    )
    assert session.activation_is_inplace(512, 17_408, 5_120)
    assert session.activation_is_inplace(513, 17_408, 5_120)
    assert not session.activation_is_inplace(512, 5_120, 10_240)


def test_q6_t16_f16_rocblas_policy_uses_nearest_measured_row_anchor() -> None:
    session = Q6T16F16RocblasPrefillSession(
        min_rows=512,
        max_rows=4096,
        x_f16_ptr=0x30000000,
        x_f16_nbytes=4096 * 5120 * 2,
        weight_f16_ptr=0x40000000,
        weight_f16_nbytes=2048 * 5120 * 2,
        out_f16_ptr=0x50000000,
        out_f16_nbytes=4096 * 2048 * 2,
        tile_out_features_by_shape={
            (512, 5120, 10240): 2048,
            (768, 5120, 10240): 1024,
            (1024, 5120, 10240): 512,
        },
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
        solution_indices_by_gemm_shape={
            (512, 5120, 2048): -1_140_856_092,
            (4096, 5120, 512): -1_140_855_996,
        },
    )
    assert session.tile_out_features(511, 5120, 10240) is None
    assert session.tile_out_features(513, 5120, 10240) == 2048
    assert session.tile_out_features(767, 5120, 10240) == 2048
    assert session.tile_out_features(768, 5120, 10240) == 1024
    assert session.tile_out_features(1000, 5120, 10240) == 1024
    assert session.tile_out_features(1024, 5120, 10240) == 512
    assert session.tile_out_features(1024, 5120, 1024) is None
    assert session.solution_index(512, 5120, 2048) == -1_140_856_092
    assert session.solution_index(4096, 5120, 512) == -1_140_855_996
    assert session.solution_index(1024, 5120, 512) is None


def test_q4_t16_bulk_unequal_pair_routes_only_admitted_shape() -> None:
    q4_a = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    q4_b = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair",
        "gguf_q4_k_t16_v1",
        "dense_unequal_dual_wmma_prefill_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls = []
    try:
        register(key, lambda *args, **kwargs: calls.append((args, kwargs)), replace=True)
        assert not launch_gguf_linear_pair(
            q4_a,
            q4_b,
            x_ptr=0x1000,
            out_a_ptr=0x2000,
            out_b_ptr=0x3000,
            rows=512,
            in_features=5_120,
            out_features=10_240,
            out_features_b=6_144,
            use_wmma_prefill=True,
        )
        with q4_t16_unequal_pair_prefill_session(True):
            assert launch_gguf_linear_pair(
                q4_a,
                q4_b,
                x_ptr=0x1000,
                out_a_ptr=0x2000,
                out_b_ptr=0x3000,
                rows=512,
                in_features=5_120,
                out_features=10_240,
                out_features_b=6_144,
                use_wmma_prefill=True,
                runtime="runtime-sentinel",
            )
        with q4_t16_unequal_pair_prefill_session(True):
            # rows<=8 keep the GEMV/rowtile decode owners and 15 is the last row
            # count under the qualified rows16 floor (2026-08-30 W7900 re-
            # qualification; 511 routed here before that change).
            for rows in (1, 4, 15):
                assert not launch_gguf_linear_pair(
                    q4_a,
                    q4_b,
                    x_ptr=0x1000,
                    out_a_ptr=0x2000,
                    out_b_ptr=0x3000,
                    rows=rows,
                    in_features=5_120,
                    out_features=10_240,
                    out_features_b=6_144,
                    use_wmma_prefill=True,
                )
            assert not launch_gguf_linear_pair(
                q4_a,
                q4_b,
                x_ptr=0x1000,
                out_a_ptr=0x2000,
                out_b_ptr=0x3000,
                rows=512,
                in_features=5_120,
                out_features=10_240,
                out_features_b=6_112,
                use_wmma_prefill=True,
            )
    finally:
        register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (0x1000, 14, 14, 0x2000, 0x3000, 512, 5_120, 10_240, 6_144),
            {"stream": 0, "runtime": "runtime-sentinel"},
        )
    ]


def test_q4_t16_unequal_pair_context_is_nested_and_cache_safe() -> None:
    context = gguf_linear_module._q4_t16_unequal_pair_prefill_enabled
    assert context.get() is False
    with q4_t16_unequal_pair_prefill_session(True):
        assert context.get() is True
        with q4_t16_unequal_pair_prefill_session(False):
            assert context.get() is False
        assert context.get() is True
    assert context.get() is False


def test_q6_t16_f16_rocblas_context_declines_mixed_pair_for_singleton_owner() -> None:
    q6_weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    q8_weight = _fake_weight(
        layout=LAYOUT_RAW_GGUF,
        quant_key="gguf_q8_0",
    )
    session = Q6T16F16RocblasPrefillSession(
        min_rows=512,
        max_rows=4096,
        x_f16_ptr=0x30000000,
        x_f16_nbytes=512 * 5120 * 2,
        weight_f16_ptr=0x40000000,
        weight_f16_nbytes=2048 * 5120 * 2,
        out_f16_ptr=0x50000000,
        out_f16_nbytes=512 * 2048 * 2,
        tile_out_features_by_shape={(512, 5120, 10240): 2048},
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
    )

    with q6_t16_f16_rocblas_prefill_session(session):
        assert not launch_gguf_linear_pair(
            q6_weight,
            q8_weight,
            x_ptr=0x10000000,
            out_a_ptr=0x20000000,
            out_b_ptr=0x21000000,
            rows=512,
            in_features=5120,
            out_features=10240,
            out_features_b=6144,
            use_wmma_prefill=True,
        )


def test_launch_gguf_linear_residual_routes_registered_q6_and_q4_owners() -> None:
    launch = getattr(gguf_linear_module, "launch_gguf_linear_residual", None)
    assert callable(launch)
    q6_key = KernelKey(
        "hip_gfx1100",
        "linear+residual",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_gemv_rowtile_bf16_residual_bf16_out",
    )
    q4_key = KernelKey(
        "hip_gfx1100",
        "linear+residual",
        "gguf_q4_k_t16_v1",
        "dense_rowtile_bf16_residual_bf16_out",
    )
    # Earlier registry-plan tests deliberately clear global registrations. Make
    # both primitive owners present before installing fake-pointer captures;
    # otherwise the launcher's lazy bootstrap can restore the real composite
    # over a capture and execute it with these sentinel addresses.
    for primitive_key in (
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_rowtile_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_rowtile_bf16_bf16_out",
        ),
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k_t16_v1",
            "t16_gemv_rowtile_bf16_bf16_out",
        ),
    ):
        gguf_linear_module._ensure_linear_kernel_registered(primitive_key)
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (q6_key, q4_key)
    }
    calls = []

    def capture(label):
        def fake(*args, **kwargs):
            calls.append((label, args, kwargs))

        return fake

    register(q6_key, capture("q6"), replace=True)
    register(q4_key, capture("q4"), replace=True)
    q6 = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    q4 = _fake_weight(
        layout=LAYOUT_Q4_K_PACK8,
        quant_key="gguf_q4_k",
        decode_tiles=True,
    )
    q5 = _fake_weight(
        layout=LAYOUT_GGUF_Q5_K_T16,
        quant_key="gguf_q5_k_t16_v1",
    )
    try:
        assert not launch(q6, 100, 300, 400, 3, 17_408, 5_120)
        with native_batch_decode_session(True):
            assert not launch(q5, 100, 300, 400, 3, 6_144, 5_120)
            assert not launch(q6, 100, 300, 400, 4, 17_408, 5_120)
            assert not launch(q6, 100, 300, 400, 5, 17_408, 5_120)
            with wmma_prefill_session(True):
                assert not launch(q6, 100, 300, 400, 3, 17_408, 5_120)
            assert launch(
                q6,
                100,
                300,
                400,
                3,
                17_408,
                5_120,
                stream=7,
                runtime="runtime-sentinel",
            )
            assert launch(
                q4,
                101,
                301,
                401,
                4,
                17_408,
                5_120,
                stream=8,
                runtime="runtime-sentinel",
            )
    finally:
        for key, fn in originals.items():
            register(key, fn, replace=True)

    assert calls == [
        (
            "q6",
            (100, 14, 300, 400, 3, 17_408, 5_120),
            {"stream": 7, "runtime": "runtime-sentinel"},
        ),
        (
            "q4",
            (101, 15, 301, 401, 4, 17_408, 5_120),
            {"stream": 8, "runtime": "runtime-sentinel"},
        ),
    ]


def test_launch_gguf_linear_residual_routes_registered_c1_pack8_and_dense_abis() -> None:
    from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
        register_dense_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        register_gguf_q4_k_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    launch = gguf_linear_module.launch_gguf_linear_residual
    register_dense_gemv_kernels()
    register_gguf_q4_k_gemv_kernels()
    register_gfx1151_kernels(replace=True)
    keys = (
        KernelKey(
            "hip_gfx1151",
            "linear+residual",
            "gguf_q4_k",
            "pack8_bf16_residual_bf16_out",
        ),
        KernelKey(
            "hip_gfx1151",
            "linear+residual",
            "bf16",
            "out_bf16_residual_bf16_out",
        ),
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in keys
    }
    calls: list[tuple[str, tuple, dict]] = []

    def capture(label):
        def fake(*args, **kwargs):
            calls.append((label, args, kwargs))

        return fake

    for key, label in zip(keys, ("q4", "dense"), strict=True):
        register(key, capture(label), replace=True)
    q4 = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    dense = _fake_weight(layout=LAYOUT_DENSE_BF16, quant_key="bf16")
    try:
        assert not launch(
            q4, 100, 300, 400, 1, 3_584, 1_024, backend="hip_gfx1151"
        )
        with wmma_prefill_session(True):
            assert launch(
                q4,
                100,
                300,
                400,
                1,
                3_584,
                1_024,
                backend="hip_gfx1151",
                registered_decode=True,
                stream=7,
                runtime="runtime-sentinel",
            )
            assert launch(
                dense,
                101,
                301,
                401,
                1,
                3_584,
                1_024,
                backend="hip_gfx1151",
                registered_decode=True,
                stream=8,
                runtime="runtime-sentinel",
            )
    finally:
        for key, fn in originals.items():
            register(key, fn, replace=True)

    assert calls == [
        (
            "q4",
            (100, 11, 12, 13, 300, 400, 1, 3_584, 1_024),
            {"stream": 7, "runtime": "runtime-sentinel"},
        ),
        (
            "dense",
            (101, 10, 301, 401, 1, 3_584, 1_024),
            {"stream": 8, "runtime": "runtime-sentinel"},
        ),
    ]


def test_launch_gguf_linear_residual_routes_registered_c1_t16_abi() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    keys = (
        KernelKey(
            "hip_gfx1151",
            "linear+residual",
            "gguf_q4_k_t16_v1",
            "dense_single_local32_bf16_residual_bf16_out",
        ),
        KernelKey(
            "hip_gfx1151",
            "linear+residual",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_decode_bf16_residual_bf16_out",
        ),
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in keys
    }
    calls: list[tuple[str, tuple, dict]] = []

    def capture(label):
        def fake(*args, **kwargs):
            calls.append((label, args, kwargs))

        return fake

    for key, label in zip(keys, ("q4", "q6"), strict=True):
        register(key, capture(label), replace=True)
    q4 = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    q6 = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    try:
        with wmma_prefill_session(True):
            for weight, stream in ((q4, 7), (q6, 8)):
                assert gguf_linear_module.launch_gguf_linear_residual(
                    weight,
                    100,
                    300,
                    400,
                    1,
                    17_408,
                    5_120,
                    backend="hip_gfx1151",
                    registered_decode=True,
                    stream=stream,
                    runtime="runtime-sentinel",
                )
    finally:
        for key, fn in originals.items():
            register(key, fn, replace=True)

    assert calls == [
        (
            "q4",
            (100, 14, 300, 400, 1, 17_408, 5_120),
            {"stream": 7, "runtime": "runtime-sentinel"},
        ),
        (
            "q6",
            (100, 14, 300, 400, 1, 17_408, 5_120),
            {"stream": 8, "runtime": "runtime-sentinel"},
        ),
    ]


def test_launch_gguf_linear_residual_routes_dense_wmma_bulk_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
        register_dense_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_dense_gemv_kernels()
    register_gfx1151_kernels(replace=True)
    key = KernelKey(
        "hip_gfx1151",
        "linear+residual",
        "bf16",
        "prefill_wmma_out_bf16_residual_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, capture, replace=True)
    dense = _fake_weight(layout=LAYOUT_DENSE_BF16, quant_key="bf16")
    pack8 = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    resolve_calls = []
    real_resolve = gguf_linear_module.resolve_gguf_linear_dispatch

    def traced_resolve(*args, **kwargs):
        resolve_calls.append((args, kwargs))
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(
        gguf_linear_module,
        "resolve_gguf_linear_dispatch",
        traced_resolve,
    )
    try:
        assert not gguf_linear_module.launch_gguf_linear_residual(
            dense,
            101,
            301,
            401,
            512,
            3_584,
            1_024,
            backend="hip_gfx1151",
        )
        with wmma_prefill_session(True):
            assert not gguf_linear_module.launch_gguf_linear_residual(
                pack8,
                100,
                300,
                400,
                512,
                3_584,
                1_024,
                backend="hip_gfx1151",
            )
            assert resolve_calls == []
            monkeypatch.setenv("HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL", "0")
            assert not gguf_linear_module.launch_gguf_linear_residual(
                dense, 101, 301, 401, 512, 3_584, 1_024, backend="hip_gfx1151"
            )
            monkeypatch.delenv("HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL")
            assert gguf_linear_module.launch_gguf_linear_residual(
                dense,
                101,
                301,
                401,
                512,
                3_584,
                1_024,
                backend="hip_gfx1151",
                stream=9,
                runtime="runtime-sentinel",
            )
    finally:
        register(key, original, replace=True)

    assert calls == [
        (
            (101, 10, 301, 401, 512, 3_584, 1_024),
            {"stream": 9, "runtime": "runtime-sentinel"},
        )
    ]
    assert len(resolve_calls) == 1


def test_launch_gguf_linear_q8_1_routes_only_registered_planar_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = getattr(gguf_linear_module, "launch_gguf_linear_q8_1", None)
    assert callable(launch)
    projection_key = KernelKey(
        "hip_gfx1100",
        "linear_q8_1",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_q8_1_dp4a_gemv_bf16_bf16_out",
    )
    residual_key = KernelKey(
        "hip_gfx1100",
        "linear_q8_1+residual",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_q8_1_dp4a_gemv_bf16_residual_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (projection_key, residual_key)
    }
    quantize_calls = []
    kernel_calls = []
    resolve_dispatch_calls = []

    def quantize(*args, **kwargs):
        quantize_calls.append((args, kwargs))

    def capture(label):
        def fake(*args, **kwargs):
            kernel_calls.append((label, args, kwargs))

        return fake

    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_quantize_bf16_q8_1",
        quantize,
        raising=False,
    )
    original_resolve_dispatch = gguf_linear_module.resolve_gguf_linear_dispatch

    def counted_resolve_dispatch(*args, **kwargs):
        resolve_dispatch_calls.append((args, kwargs))
        return original_resolve_dispatch(*args, **kwargs)

    monkeypatch.setattr(
        gguf_linear_module,
        "resolve_gguf_linear_dispatch",
        counted_resolve_dispatch,
    )
    gguf_linear_module.clear_gguf_linear_dispatch_cache()

    def reject_global_registry_restore(_key):
        raise AssertionError("optional q8-input misses must not rewrite the registry")

    monkeypatch.setattr(
        gguf_linear_module,
        "_ensure_linear_kernel_registered",
        reject_global_registry_restore,
    )
    register(projection_key, capture("projection"), replace=True)
    register(residual_key, capture("residual"), replace=True)
    q6 = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    q5 = _fake_weight(
        layout=LAYOUT_GGUF_Q5_K_T16,
        quant_key="gguf_q5_k_t16_v1",
    )
    try:
        assert launch(
            q6,
            100,
            200,
            300,
            1,
            5_120,
            10_240,
            stream=7,
            runtime="runtime-sentinel",
            enabled=True,
        )
        assert launch(
            q6,
            100,
            200,
            300,
            1,
            5_120,
            10_240,
            stream=7,
            runtime="runtime-sentinel",
            enabled=True,
        )
        assert launch(
            q6,
            101,
            201,
            301,
            4,
            17_408,
            5_120,
            residual_ptr=401,
            stream=8,
            runtime="runtime-sentinel",
            enabled=True,
        )
        assert not launch(q5, 100, 200, 300, 4, 17_408, 5_120, enabled=True)
        assert not launch(q6, 100, 200, 300, 5, 17_408, 5_120, enabled=True)
        assert not launch(q6, 100, 0, 300, 4, 17_408, 5_120, enabled=True)
        assert not launch(q6, 100, 200, 300, 4, 17_408, 5_120, enabled=False)
    finally:
        for key, fn in originals.items():
            register(key, fn, replace=True)

    assert len(resolve_dispatch_calls) == 2
    assert quantize_calls == [
        (
            (100, 200, 1, 5_120),
            {"stream": 7, "runtime": "runtime-sentinel"},
        ),
        (
            (100, 200, 1, 5_120),
            {"stream": 7, "runtime": "runtime-sentinel"},
        ),
        (
            (101, 201, 4, 17_408),
            {"stream": 8, "runtime": "runtime-sentinel"},
        ),
    ]
    assert kernel_calls == [
        (
            "projection",
            (200, 14, 300, 1, 5_120, 10_240),
            {"stream": 7, "runtime": "runtime-sentinel"},
        ),
        (
            "projection",
            (200, 14, 300, 1, 5_120, 10_240),
            {"stream": 7, "runtime": "runtime-sentinel"},
        ),
        (
            "residual",
            (201, 14, 401, 301, 4, 17_408, 5_120),
            {"stream": 8, "runtime": "runtime-sentinel"},
        ),
    ]

    # The restored production leaf must fail closed before quantization when a
    # registered planar weight has no measured shape policy. Misses are cached
    # too, but remain invalidated by registry-generation changes.
    assert not launch(q6, 102, 202, 302, 4, 512, 256, enabled=True)
    assert len(resolve_dispatch_calls) == 3
    assert len(quantize_calls) == 3
    gguf_linear_module.clear_gguf_linear_dispatch_cache()


def test_launch_q4_pack8_wmma_prefill_uses_resident_pack8_abi() -> None:
    weight = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_wmma_prefill_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, fake_kernel, replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=512,
            in_features=3072,
            out_features=1024,
            stream=7,
            libraries={
                "gguf_q4_k:pack8_wmma_prefill_bf16_bf16_out": "wmma-library"
            },
            runtime="runtime-sentinel",
            use_wmma_prefill=False,
            use_gemv_decode=False,
            use_q4_pack8_wmma=True,
        )
    finally:
        register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (100, 11, 12, 13, 200, 512, 3072, 1024),
            {
                "stream": 7,
                "runtime": "runtime-sentinel",
                "library": "wmma-library",
            },
        )
    ]


def test_gfx1151_q4_pack8_bulk_prefill_prefers_wmma_over_exact_tile8x8() -> None:
    import hipengine.kernels.hip_gfx1151  # noqa: F401  (registers aliases + capability)

    weight = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k",
        "pack8_wmma_prefill_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, fake_kernel, replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=512,
            in_features=1024,
            out_features=3584,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
            use_gemv_decode=False,
        )
    finally:
        register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (100, 11, 12, 13, 200, 512, 1024, 3584)
    assert kwargs["runtime"] == "runtime-sentinel"


def test_gfx1151_q4_pack8_bulk_prefill_keeps_exact_outside_qualified_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The campaign route must fail closed beyond its p512 shape matrix."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    exact_key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k",
        "pack8_prefill_bf16_bf16_out",
    )
    candidate_key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k",
        "pack8_wmma_prefill_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (exact_key, candidate_key)
    }

    def exact(*args, **kwargs):
        return None

    def wmma(*args, **kwargs):
        return None

    calls: list[object] = []
    register(exact_key, exact, replace=True)
    register(candidate_key, wmma, replace=True)
    monkeypatch.setitem(
        gguf_linear_module._LAUNCH_ABI,
        "pack8",
        lambda fn, *args, **kwargs: calls.append(fn),
    )
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=256,
            in_features=1024,
            out_features=3584,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
            use_gemv_decode=False,
        )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [exact]


def test_gfx1100_q4_pack8_bulk_prefill_keeps_exact_tile8x8() -> None:
    weight = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_exact_prefill_tile8x8_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, fake_kernel, replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=512,
            in_features=1024,
            out_features=3584,
            backend="hip_gfx1100",
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
            use_gemv_decode=False,
        )
    finally:
        register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert len(calls) == 1
    assert calls[0][0] == (100, 11, 12, 13, 200, 512, 1024, 3584)


@pytest.mark.parametrize(
    ("rows", "in_features", "out_features", "expected"),
    [
        (16, 1_000, 128, "fallback"),
        (256, 3_584, 1_024, "fallback"),
        (512, 2_048, 1_024, "fallback"),
        (512, 1_024, 512, "wmma"),
        (512, 3_584, 1_024, "wmma"),
    ],
)
def test_gfx1151_dense_bf16_bulk_prefill_honors_qualified_shapes(
    rows: int,
    in_features: int,
    out_features: int,
    expected: str,
) -> None:
    """The WMMA leaf owns only safe campaign-qualified p512 shapes."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight = _fake_weight(layout=LAYOUT_DENSE_BF16, quant_key="gguf_q4_1")
    fallback_key = KernelKey("hip_gfx1151", "dense_gemv", "bf16", "prefill_out")
    candidate_key = KernelKey(
        "hip_gfx1151", "dense_gemv", "bf16", "prefill_wmma_out"
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (fallback_key, candidate_key)
    }
    calls: list[str] = []
    register(
        fallback_key,
        lambda *args, **kwargs: calls.append("fallback"),
        replace=True,
    )
    register(
        candidate_key,
        lambda *args, **kwargs: calls.append("wmma"),
        replace=True,
    )
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
        )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [expected]


def test_gfx1151_bulk_wmma_policy_matches_qwen35_08b_campaign_shapes() -> None:
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_Q4_PACK8_WMMA_BULK_PREFILL_SHAPES",
        frozenset(),
    ) == frozenset(
        {
            (512, 1_024, 512),
            (512, 1_024, 2_048),
            (512, 1_024, 3_584),
            (512, 2_048, 1_024),
            (512, 3_584, 1_024),
        }
    )
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL_SHAPES",
        frozenset(),
    ) == frozenset({(512, 1_024, 3_584)})
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_DENSE_BF16_WMMA_BULK_PREFILL_SHAPES",
        frozenset(),
    ) == frozenset(
        {
            (512, 1_024, 512),
            (512, 3_584, 1_024),
        }
    )
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_LINEAR_RESIDUAL_MAX_ROWS_BY_QUANT",
        {},
    )["bf16"] == 512


def test_qwen35_dense_pack8_wmma_tile_policy_covers_ffn_shapes() -> None:
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
        _default_q4_pack8_tiles,
    )

    assert _default_q4_pack8_tiles(512, 1024, 3584) == (16, 32)
    assert _default_q4_pack8_tiles(512, 3584, 1024) == (64, 16)


@pytest.mark.parametrize(
    ("layout", "quant_key"),
    [
        (LAYOUT_GGUF_Q6_K_T16, "gguf_q6_k_t16_v1"),
        (
            LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
            "gguf_q6_k_t16_qmicro_planar_v1",
        ),
    ],
)
def test_q6_t16_routes_decode_native_rowtile_and_dense_wmma(
    layout: str,
    quant_key: str,
) -> None:
    weight = _fake_weight(
        layout=layout,
        quant_key=quant_key,
    )
    keys = {
        "decode": KernelKey(
            "hip_gfx1100",
            "linear",
            quant_key,
            "t16_gemv_decode_bf16_bf16_out",
        ),
        "rowtile": KernelKey(
            "hip_gfx1100",
            "linear",
            quant_key,
            "t16_gemv_rowtile_bf16_bf16_out",
        ),
        "wmma": KernelKey(
            "hip_gfx1100",
            "linear",
            quant_key,
            "t16_wmma_prefill_bf16_bf16_out",
        ),
    }
    originals = {
        label: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
            missing="none",
        )
        for label, key in keys.items()
    }
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def capture(label: str):
        def fake_kernel(*args, **kwargs):
            calls.append((label, args, kwargs))

        return fake_kernel

    for label, key in keys.items():
        register(key, capture(label), replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=1,
            in_features=5120,
            out_features=10240,
            stream=7,
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
        )
        with native_batch_decode_session(True):
            launch_gguf_linear(
                weight,
                x_ptr=101,
                out_ptr=201,
                rows=4,
                in_features=5120,
                out_features=10240,
                stream=8,
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
        launch_gguf_linear(
            weight,
            x_ptr=102,
            out_ptr=202,
            rows=64,
            in_features=5120,
            out_features=10240,
            stream=9,
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
        )
    finally:
        for label, key in keys.items():
            original = originals[label]
            if original is None:
                unregister(key)
            else:
                register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            "decode",
            (100, 14, 200, 1, 5120, 10240),
            {"stream": 7, "runtime": "runtime-sentinel"},
        ),
        (
            "rowtile",
            (101, 14, 201, 4, 5120, 10240),
            {"stream": 8, "runtime": "runtime-sentinel"},
        ),
        (
            "wmma",
            (102, 14, 202, 64, 5120, 10240),
            {"stream": 9, "runtime": "runtime-sentinel"},
        ),
    ]


def test_q5_t16_routes_decode_bounded_native_rowtile_and_dense_wmma() -> None:
    quant_key = "gguf_q5_k_t16_v1"
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q5_K_T16,
        quant_key=quant_key,
    )
    keys = {
        "decode": KernelKey(
            "hip_gfx1100",
            "linear",
            quant_key,
            "t16_gemv_decode_bf16_bf16_out",
        ),
        "rowtile": KernelKey(
            "hip_gfx1100",
            "linear",
            quant_key,
            "t16_gemv_rowtile_bf16_bf16_out",
        ),
        "wmma": KernelKey(
            "hip_gfx1100",
            "linear",
            quant_key,
            "t16_wmma_prefill_bf16_bf16_out",
        ),
    }
    originals = {
        label: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
            missing="none",
        )
        for label, key in keys.items()
    }
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def capture(label: str):
        def fake_kernel(*args, **kwargs):
            calls.append((label, args, kwargs))

        return fake_kernel

    for label, key in keys.items():
        register(key, capture(label), replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=1,
            in_features=6144,
            out_features=5120,
            stream=7,
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
        )
        with native_batch_decode_session(True):
            launch_gguf_linear(
                weight,
                x_ptr=101,
                out_ptr=201,
                rows=4,
                in_features=6144,
                out_features=5120,
                stream=8,
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
            launch_gguf_linear(
                weight,
                x_ptr=102,
                out_ptr=202,
                rows=5,
                in_features=6144,
                out_features=5120,
                stream=9,
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
        launch_gguf_linear(
            weight,
            x_ptr=103,
            out_ptr=203,
            rows=64,
            in_features=6144,
            out_features=5120,
            stream=10,
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
        )
    finally:
        for label, key in keys.items():
            original = originals[label]
            if original is None:
                unregister(key)
            else:
                register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            "decode",
            (100, 14, 200, 1, 6144, 5120),
            {"stream": 7, "runtime": "runtime-sentinel"},
        ),
        (
            "rowtile",
            (101, 14, 201, 4, 6144, 5120),
            {"stream": 8, "runtime": "runtime-sentinel"},
        ),
        (
            "rowtile",
            (102, 14, 202, 5, 6144, 5120),
            {"stream": 9, "runtime": "runtime-sentinel"},
        ),
        (
            "wmma",
            (103, 14, 203, 64, 6144, 5120),
            {"stream": 10, "runtime": "runtime-sentinel"},
        ),
    ]


def test_gfx1151_q5_t16_ssm_out_c1_uses_exact_tile8_shape_owner() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    quant_key = "gguf_q5_k_t16_v1"
    weight = _fake_weight(layout=LAYOUT_GGUF_Q5_K_T16, quant_key=quant_key)
    keys = {
        "direct": KernelKey(
            "hip_gfx1151",
            "linear",
            quant_key,
            "t16_gemv_decode_bf16_bf16_out",
        ),
        "tile8": KernelKey(
            "hip_gfx1151",
            "linear",
            quant_key,
            "t16_gemv_decode_tile8_bf16_bf16_out",
        ),
    }
    originals = {
        label: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for label, key in keys.items()
    }
    calls: list[tuple[str, tuple[object, ...]]] = []

    def capture(label: str):
        def fake_kernel(*args, **_kwargs):
            calls.append((label, args))

        return fake_kernel

    for label, key in keys.items():
        register(key, capture(label), replace=True)
    try:
        launch_gguf_linear(
            weight,
            100,
            200,
            rows=1,
            in_features=6_144,
            out_features=5_120,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
        )
        with native_batch_decode_session(True):
            launch_gguf_linear(
                weight,
                101,
                201,
                rows=1,
                in_features=6_144,
                out_features=5_120,
                backend="hip_gfx1151",
                runtime="runtime-sentinel",
            )
        launch_gguf_linear(
            weight,
            102,
            202,
            rows=1,
            in_features=2_048,
            out_features=1_024,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
        )
    finally:
        for label, key in keys.items():
            register(key, originals[label], replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        ("tile8", (100, 14, 200, 1, 6_144, 5_120)),
        ("direct", (101, 14, 201, 1, 6_144, 5_120)),
        ("direct", (102, 14, 202, 1, 2_048, 1_024)),
    ]


def test_gfx1151_q5_t16_ssm_out_uses_direct_through_c8_and_wmma_for_bulk() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    quant_key = "gguf_q5_k_t16_v1"
    weight = _fake_weight(layout=LAYOUT_GGUF_Q5_K_T16, quant_key=quant_key)
    keys = {
        "direct": KernelKey(
            "hip_gfx1151",
            "linear",
            quant_key,
            "t16_gemv_decode_bf16_bf16_out",
        ),
        "rowtile": KernelKey(
            "hip_gfx1151",
            "linear",
            quant_key,
            "t16_gemv_rowtile_bf16_bf16_out",
        ),
        "wmma": KernelKey(
            "hip_gfx1151",
            "linear",
            quant_key,
            "t16_wmma_prefill_bf16_bf16_out",
        ),
    }
    originals = {
        label: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for label, key in keys.items()
    }
    calls: list[tuple[str, tuple[object, ...]]] = []

    def capture(label: str):
        def fake_kernel(*args, **_kwargs):
            calls.append((label, args))

        return fake_kernel

    for label, key in keys.items():
        register(key, capture(label), replace=True)
    try:
        launch_gguf_linear(
            weight,
            100,
            200,
            rows=1,
            in_features=2_048,
            out_features=1_024,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
        )
        with native_batch_decode_session(True):
            launch_gguf_linear(
                weight,
                101,
                201,
                rows=4,
                in_features=2_048,
                out_features=1_024,
                backend="hip_gfx1151",
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
            launch_gguf_linear(
                weight,
                102,
                202,
                rows=8,
                in_features=2_048,
                out_features=1_024,
                backend="hip_gfx1151",
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
            launch_gguf_linear(
                weight,
                103,
                203,
                rows=8,
                in_features=1_024,
                out_features=6_144,
                backend="hip_gfx1151",
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
        launch_gguf_linear(
            weight,
            104,
            204,
            rows=512,
            in_features=2_048,
            out_features=1_024,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
        )
    finally:
        for label, key in keys.items():
            register(key, originals[label], replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        ("direct", (100, 14, 200, 1, 2_048, 1_024)),
        ("rowtile", (101, 14, 201, 4, 2_048, 1_024)),
        ("direct", (102, 14, 202, 8, 2_048, 1_024)),
        ("wmma", (103, 14, 203, 8, 1_024, 6_144)),
        ("wmma", (104, 14, 204, 512, 2_048, 1_024)),
    ]


def test_gfx1151_q5_t16_27b_shapes_use_rowtile_through_c8() -> None:
    """27B Q5 ssm_out/ffn_down/qkv/v shapes rowtile to c8 (per-shape cap)."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    quant_key = "gguf_q5_k_t16_v1"
    weight = _fake_weight(layout=LAYOUT_GGUF_Q5_K_T16, quant_key=quant_key)
    keys = {
        "rowtile": KernelKey(
            "hip_gfx1151",
            "linear",
            quant_key,
            "t16_gemv_rowtile_bf16_bf16_out",
        ),
        "wmma": KernelKey(
            "hip_gfx1151",
            "linear",
            quant_key,
            "t16_wmma_prefill_bf16_bf16_out",
        ),
    }
    originals = {
        label: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for label, key in keys.items()
    }
    calls: list[tuple[str, tuple[object, ...]]] = []

    def capture(label: str):
        def fake_kernel(*args, **_kwargs):
            calls.append((label, args))

        return fake_kernel

    for label, key in keys.items():
        register(key, capture(label), replace=True)
    try:
        with native_batch_decode_session(True):
            # 27B ssm_out (6144, 5120) rowtiles to c8 (cap 8 via per-shape cap).
            for rows in (2, 4, 8):
                launch_gguf_linear(
                    weight,
                    100,
                    200,
                    rows=rows,
                    in_features=6_144,
                    out_features=5_120,
                    backend="hip_gfx1151",
                    runtime="runtime-sentinel",
                    use_wmma_prefill=False,
                )
            # 27B ffn_down (17408, 5120) rowtiles to c8.
            launch_gguf_linear(
                weight,
                101,
                201,
                rows=8,
                in_features=17_408,
                out_features=5_120,
                backend="hip_gfx1151",
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
    finally:
        for label, key in keys.items():
            register(key, originals[label], replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        ("rowtile", (100, 14, 200, 2, 6_144, 5_120)),
        ("rowtile", (100, 14, 200, 4, 6_144, 5_120)),
        ("rowtile", (100, 14, 200, 8, 6_144, 5_120)),
        ("rowtile", (101, 14, 201, 8, 17_408, 5_120)),
    ]


def test_gfx1151_q4_t16_single_c_n_chunks_to_rowtile8_in_native_session() -> None:
    """Native c=N: Q4 single projections chunk rows 9..511 into rowtile8 groups.

    Q5 keeps its native direct grid.y=rows leaf, and rows >= 512 stay on WMMA
    prefill (bulk regime), so no decode concurrency silently falls to WMMA.
    """

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    q4 = _fake_weight(layout=LAYOUT_GGUF_Q4_K_T16, quant_key="gguf_q4_k_t16_v1")
    q5 = _fake_weight(layout=LAYOUT_GGUF_Q5_K_T16, quant_key="gguf_q5_k_t16_v1")
    keys = {
        "q4_rowtile": KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_rowtile_bf16_bf16_out",
        ),
        "q4_rowtile_w2": KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_rowtile16_w2_bf16_bf16_out",
        ),
        "q4_wmma": KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q4_k_t16_v1",
            "t16_wmma_prefill_bf16_bf16_out",
        ),
        "q5_direct": KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q5_k_t16_v1",
            "t16_gemv_decode_bf16_bf16_out",
        ),
    }
    originals = {
        label: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for label, key in keys.items()
    }
    calls: list[tuple[str, int, int, int]] = []

    def capture(label: str):
        def fake_kernel(*args, **_kwargs):
            # (x_ptr, tiles, out_ptr, rows, in_features, out_features)
            calls.append((label, args[0], args[3], args[5]))

        return fake_kernel

    for label, key in keys.items():
        register(key, capture(label), replace=True)
    try:
        with native_batch_decode_session(True):
            # c=9 -> (7,0),(2,7); c=16 -> (8,0),(8,8); c=33 -> 8,8,8,7,2.
            for rows, inf, outf in ((9, 5_120, 10_240), (16, 17_408, 5_120)):
                launch_gguf_linear(
                    q4,
                    0,
                    0,
                    rows=rows,
                    in_features=inf,
                    out_features=outf,
                    backend="hip_gfx1151",
                    runtime="runtime-sentinel",
                    use_wmma_prefill=False,
                )
            # c=512 stays on WMMA prefill (bulk regime), not chunked.
            launch_gguf_linear(
                q4,
                0,
                0,
                rows=512,
                in_features=5_120,
                out_features=10_240,
                backend="hip_gfx1151",
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
            # Q5 keeps its native direct leaf (single launch, no chunking).
            launch_gguf_linear(
                q5,
                0,
                0,
                rows=16,
                in_features=6_144,
                out_features=5_120,
                backend="hip_gfx1151",
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
    finally:
        for label, key in keys.items():
            register(key, originals[label], replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        # c=9 Q4: (7 rows @ row 0), (2 rows @ row 7).
        ("q4_rowtile", 0, 7, 10_240),
        ("q4_rowtile", 7 * 5_120 * 2, 2, 10_240),
        # c=16 Q4 ffn_down: the gfx1151-qualified row8 two-wave owner.
        ("q4_rowtile_w2", 0, 8, 5_120),
        ("q4_rowtile_w2", 8 * 17_408 * 2, 8, 5_120),
        # c=512 stays WMMA prefill.
        ("q4_wmma", 0, 512, 10_240),
        # Q5 direct leaf: one launch, rows=16, out 5120.
        ("q5_direct", 0, 16, 5_120),
    ]


def test_gfx1151_q4_t16_full_kv_c1_uses_exact_col4_shape_owner() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    quant_key = "gguf_q4_k_t16_v1"
    weight = _fake_weight(layout=LAYOUT_GGUF_Q4_K_T16, quant_key=quant_key)
    keys = {
        "direct": KernelKey(
            "hip_gfx1151",
            "linear",
            quant_key,
            "dense_single_local32_bf16_bf16_out",
        ),
        "col4": KernelKey(
            "hip_gfx1151",
            "linear",
            quant_key,
            "dense_single_col4_bf16_bf16_out",
        ),
    }
    originals = {
        label: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for label, key in keys.items()
    }
    calls: list[tuple[str, tuple[object, ...]]] = []

    def capture(label: str):
        def fake_kernel(*args, **_kwargs):
            calls.append((label, args))

        return fake_kernel

    for label, key in keys.items():
        register(key, capture(label), replace=True)
    try:
        launch_gguf_linear(
            weight,
            100,
            200,
            rows=1,
            in_features=5_120,
            out_features=1_024,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
        )
        with native_batch_decode_session(True):
            launch_gguf_linear(
                weight,
                101,
                201,
                rows=1,
                in_features=5_120,
                out_features=1_024,
                backend="hip_gfx1151",
                runtime="runtime-sentinel",
            )
        launch_gguf_linear(
            weight,
            102,
            202,
            rows=1,
            in_features=5_120,
            out_features=6_144,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
        )
    finally:
        for label, key in keys.items():
            register(key, originals[label], replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        ("col4", (100, 14, 200, 1, 5_120, 1_024)),
        ("direct", (101, 14, 201, 1, 5_120, 1_024)),
        ("direct", (102, 14, 202, 1, 5_120, 6_144)),
    ]


def test_gfx1151_q4_t16_attn_q_splits_c8_into_exact_c4_rowtiles() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    quant_key = "gguf_q4_k_t16_v1"
    weight = _fake_weight(layout=LAYOUT_GGUF_Q4_K_T16, quant_key=quant_key)
    keys = {
        "direct": KernelKey(
            "hip_gfx1151",
            "linear",
            quant_key,
            "dense_single_local32_bf16_bf16_out",
        ),
        "rowtile": KernelKey(
            "hip_gfx1151",
            "linear",
            quant_key,
            "dense_rowtile_bf16_bf16_out",
        ),
        "wmma": KernelKey(
            "hip_gfx1151",
            "linear",
            quant_key,
            "t16_wmma_prefill_bf16_bf16_out",
        ),
    }
    originals = {
        label: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for label, key in keys.items()
    }
    calls: list[tuple[str, tuple[object, ...]]] = []

    def capture(label: str):
        def fake_kernel(*args, **_kwargs):
            calls.append((label, args))

        return fake_kernel

    for label, key in keys.items():
        register(key, capture(label), replace=True)
    try:
        launch_gguf_linear(
            weight,
            100,
            200,
            rows=1,
            in_features=1_024,
            out_features=4_096,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
        )
        with native_batch_decode_session(True):
            launch_gguf_linear(
                weight,
                101,
                201,
                rows=4,
                in_features=1_024,
                out_features=4_096,
                backend="hip_gfx1151",
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
            launch_gguf_linear(
                weight,
                102,
                202,
                rows=8,
                in_features=1_024,
                out_features=4_096,
                backend="hip_gfx1151",
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
        launch_gguf_linear(
            weight,
            103,
            203,
            rows=512,
            in_features=1_024,
            out_features=4_096,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
        )
    finally:
        for label, key in keys.items():
            register(key, originals[label], replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        ("direct", (100, 14, 200, 1, 1_024, 4_096)),
        ("rowtile", (101, 14, 201, 4, 1_024, 4_096)),
        ("rowtile", (102, 14, 202, 4, 1_024, 4_096)),
        ("rowtile", (8_294, 14, 32_970, 4, 1_024, 4_096)),
        ("wmma", (103, 14, 203, 512, 1_024, 4_096)),
    ]


def test_launch_q6_raw_wmma_prefill_uses_resident_raw_abi() -> None:
    weight = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k")
    key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k",
        "wmma_prefill_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, fake_kernel, replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=512,
            in_features=3072,
            out_features=1024,
            stream=7,
            libraries={
                "gguf_q6_k:wmma_prefill_bf16_bf16_out": "wmma-library"
            },
            runtime="runtime-sentinel",
            use_wmma_prefill=False,
            use_gemv_decode=False,
            use_q4_pack8_wmma=True,
        )
    finally:
        register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (100, 10, 200, 512, 3072, 1024),
            {
                "stream": 7,
                "runtime": "runtime-sentinel",
                "library": "wmma-library",
            },
        )
    ]


@pytest.mark.parametrize(
    ("weight", "output_dtype", "key", "expected_args"),
    [
        (
            _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k"),
            GGUF_OUTPUT_BF16,
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_prefill_bf16_bf16_out"),
            (100, 11, 12, 13, 200, 2, 1024, 2048),
        ),
        (
            _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k"),
            GGUF_OUTPUT_BF16,
            KernelKey("hip_gfx1100", "linear", "gguf_q5_k", "prefill_bf16_bf16_out"),
            (100, 10, 200, 2, 1024, 2048),
        ),
        (
            _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k"),
            GGUF_OUTPUT_F32,
            KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "prefill_bf16_f32_out"),
            (100, 10, 200, 2, 1024, 2048),
        ),
        (
            _fake_weight(layout=LAYOUT_DENSE_BF16, quant_key="gguf_q4_1"),
            GGUF_OUTPUT_BF16,
            KernelKey("hip_gfx1100", "dense_gemv", "bf16", "rowtile_out"),
            (100, 10, 200, 2, 1024, 2048),
        ),
        (
            _fake_weight(layout=LAYOUT_DENSE_F32, quant_key="f32"),
            GGUF_OUTPUT_BF16,
            KernelKey("hip_gfx1100", "dense_gemv", "f32", "bf16_hidden_bf16_out"),
            (100, 10, 200, 2, 1024, 2048),
        ),
        (
            _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1"),
            GGUF_OUTPUT_BF16,
            KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_bf16_bf16_out"),
            (100, 14, 200, 2, 1024, 2048),
        ),
    ],
)
def test_launch_gguf_linear_calls_registry_kernel_with_expected_abi(
    weight, output_dtype: str, key: KernelKey, expected_args: tuple[int, ...]
) -> None:
    original = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, fake_kernel, replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=2,
            in_features=1024,
            out_features=2048,
            output_dtype=output_dtype,
            threads=128,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        register(key, original, replace=True)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == expected_args
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel", "threads": 128}


def test_native_batch_decode_routes_dense_bf16_c1_to_exact_virtual256() -> None:
    weight = _fake_weight(layout=LAYOUT_DENSE_BF16, quant_key="gguf_q4_1")
    baseline_key = KernelKey("hip_gfx1100", "dense_gemv", "bf16", "out")
    candidate_key = KernelKey(
        "hip_gfx1100", "dense_gemv", "bf16", "virtual256_out"
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (baseline_key, candidate_key)
    }
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def capture(label: str):
        def fake_kernel(*args, **kwargs):
            calls.append((label, args, kwargs))

        return fake_kernel

    register(baseline_key, capture("local256"), replace=True)
    register(candidate_key, capture("virtual256_local128"), replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=1,
            in_features=5120,
            out_features=10240,
            stream=7,
            runtime="runtime-sentinel",
        )
        with native_batch_decode_session(True):
            launch_gguf_linear(
                weight,
                x_ptr=100,
                out_ptr=200,
                rows=1,
                in_features=5120,
                out_features=10240,
                stream=7,
                runtime="runtime-sentinel",
            )
            launch_gguf_linear(
                weight,
                x_ptr=100,
                out_ptr=200,
                rows=1,
                in_features=12288,
                out_features=5120,
                stream=7,
                runtime="runtime-sentinel",
            )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    expected_args = (100, 10, 200, 1, 5120, 10240)
    expected_kwargs = {"stream": 7, "runtime": "runtime-sentinel"}
    assert calls == [
        ("local256", expected_args, expected_kwargs),
        ("virtual256_local128", expected_args, expected_kwargs),
        (
            "local256",
            (100, 10, 200, 1, 12288, 5120),
            expected_kwargs,
        ),
    ]


def test_native_batch_decode_routes_only_dense_bf16_ssm_out_rowtile_to_virtual256() -> None:
    weight = _fake_weight(layout=LAYOUT_DENSE_BF16, quant_key="gguf_q4_1")
    baseline_key = KernelKey("hip_gfx1100", "dense_gemv", "bf16", "rowtile_out")
    candidate_key = KernelKey(
        "hip_gfx1100", "dense_gemv", "bf16", "virtual256_rowtile_out"
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (baseline_key, candidate_key)
    }
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def capture(label: str):
        def fake_kernel(*args, **kwargs):
            calls.append((label, args, kwargs))

        return fake_kernel

    register(baseline_key, capture("local256_rowtile"), replace=True)
    register(candidate_key, capture("virtual256_local128_rowtile"), replace=True)
    try:
        # Ordinary prefill retains the established local256 rowtile.
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=4,
            in_features=6144,
            out_features=5120,
            stream=7,
            runtime="runtime-sentinel",
        )
        with native_batch_decode_session(True):
            for rows in (2, 3, 4):
                launch_gguf_linear(
                    weight,
                    x_ptr=100,
                    out_ptr=200,
                    rows=rows,
                    in_features=6144,
                    out_features=5120,
                    stream=7,
                    runtime="runtime-sentinel",
                )
            for in_features, out_features in (
                (5120, 10240),
                (17408, 5120),
                (5120, 1024),
            ):
                launch_gguf_linear(
                    weight,
                    x_ptr=100,
                    out_ptr=200,
                    rows=4,
                    in_features=in_features,
                    out_features=out_features,
                    stream=7,
                    runtime="runtime-sentinel",
                )

        # A missing exact candidate key must fail closed to the local256 owner.
        unregister(candidate_key)
        with native_batch_decode_session(True):
            launch_gguf_linear(
                weight,
                x_ptr=100,
                out_ptr=200,
                rows=4,
                in_features=6144,
                out_features=5120,
                stream=7,
                runtime="runtime-sentinel",
            )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert [label for label, _, _ in calls] == [
        "local256_rowtile",
        "virtual256_local128_rowtile",
        "virtual256_local128_rowtile",
        "virtual256_local128_rowtile",
        "local256_rowtile",
        "local256_rowtile",
        "local256_rowtile",
        "local256_rowtile",
    ]
    assert [args[3:] for _, args, _ in calls] == [
        (4, 6144, 5120),
        (2, 6144, 5120),
        (3, 6144, 5120),
        (4, 6144, 5120),
        (4, 5120, 10240),
        (4, 17408, 5120),
        (4, 5120, 1024),
        (4, 6144, 5120),
    ]
    assert all(
        kwargs == {"stream": 7, "runtime": "runtime-sentinel"}
        for _, _, kwargs in calls
    )


def test_launch_gguf_linear_dense_f32_activation_f32_output_calls_registry_kernel() -> None:
    weight = _fake_weight(layout=LAYOUT_DENSE_F32, quant_key="f32")
    key = KernelKey("hip_gfx1100", "dense_gemv", "f32", "f32_hidden_f32_out")
    original = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, fake_kernel, replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=2,
            in_features=1024,
            out_features=4,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            threads=128,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        register(key, original, replace=True)

    assert calls == [((100, 10, 200, 2, 1024, 4), {"stream": 7, "runtime": "runtime-sentinel", "threads": 128})]


def test_gguf_linear_dispatch_rejects_unsupported_dtype() -> None:
    weight = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    with pytest.raises(ValueError, match="unsupported GGUF linear dispatch"):
        resolve_gguf_linear_dispatch(weight, output_dtype="int8")


# ---------------------------------------------------------------------------
# P8: WMMA batched prefill opt-in dispatch (docs/GGUF.md "P8: real batched
# prefill GEMM").
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_wmma_prefill_state(monkeypatch):
    """Clear the env var + session toggle before/after every test in this file.

    Without this, a test that flips ``HIPENGINE_GGUF_WMMA_PREFILL`` or calls
    ``set_wmma_prefill_enabled`` would silently leak into the next test
    case (and the next test module, since pytest runs the file in-process).
    """

    monkeypatch.delenv("HIPENGINE_GGUF_WMMA_PREFILL", raising=False)
    set_wmma_prefill_enabled(None)
    yield
    set_wmma_prefill_enabled(None)


_WMMA_BF16 = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "wmma_prefill_bf16_bf16_out")
_PREFILL_BF16 = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "prefill_bf16_bf16_out")
_DECODE_PACK8_BF16 = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out")
_PREFILL_PACK8_BF16 = _DECODE_PACK8_BF16
_PREFILL_TILE8X2_BF16 = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0", "exact_prefill_tile8x2_bf16_bf16_out"
)
_PREFILL_TILE8X4_BF16 = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0", "exact_prefill_tile8x4_bf16_bf16_out"
)
_PREFILL_TILE16X4_BF16 = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0", "exact_prefill_tile16x4_bf16_bf16_out"
)
_PREFILL_MMQ128_X3_GUARDED_BF16 = KernelKey(
    "hip_gfx1100",
    "linear",
    "gguf_q8_0",
    "mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out",
)
_Q4_WMMA_BF16 = KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "wmma_prefill_bf16_bf16_out")
_Q4_PREFILL_BF16 = KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "prefill_bf16_bf16_out")
_Q4_GEMV_BF16 = KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "gemv_bf16_bf16_out")
_Q4_PACK8_PREFILL_BF16 = KernelKey(
    "hip_gfx1100", "linear", "gguf_q4_k", "pack8_prefill_bf16_bf16_out"
)


def _capture_launch(
    *,
    rows: int,
    in_features: int = 1024,
    out_features: int = 2048,
    use_wmma_prefill: bool | None = None,
    quant_key: str = "gguf_q8_0",
    layout: str = LAYOUT_RAW_GGUF,
    output_dtype: str = GGUF_OUTPUT_BF16,
    threads: int = 0,
    runtime: object = "runtime-sentinel",
    extra_keys: tuple[KernelKey, ...] = (),
) -> tuple[KernelKey, tuple, dict]:
    """Drive ``launch_gguf_linear`` against a fake kernel + capture the call.

    Returns ``(key, args, kwargs)`` for the kernel that fired.
    """

    weight = _fake_weight(layout=layout, quant_key=quant_key)
    captured: dict[str, object] = {"key": None, "args": None, "kwargs": None}
    # Pre-resolve which key the dispatch should pick so we can register a
    # fake kernel under exactly that key (and the alternates we care about,
    # so the dispatch doesn't fall through to the real .so kernel).
    keys = (
        _WMMA_BF16,
        _PREFILL_BF16,
        _DECODE_PACK8_BF16,
        _PREFILL_TILE8X2_BF16,
        _PREFILL_TILE8X4_BF16,
        _PREFILL_TILE16X4_BF16,
        _Q4_WMMA_BF16,
        _Q4_PREFILL_BF16,
        _Q4_GEMV_BF16,
        _Q4_PACK8_PREFILL_BF16,
    ) + extra_keys
    originals = {k: resolve(backend=k.backend, layer=k.layer, quant=k.quant, variant=k.variant) for k in keys}

    def make_fake(key: KernelKey):
        def fake(*args, **kwargs):
            captured["key"] = key
            captured["args"] = args
            captured["kwargs"] = kwargs

        return fake

    try:
        for k in keys:
            register(k, make_fake(k), replace=True)
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            output_dtype=output_dtype,
            threads=threads,
            stream=7,
            runtime=runtime,
            use_wmma_prefill=use_wmma_prefill,
        )
    finally:
        for k, fn in originals.items():
            register(k, fn, replace=True)

    return captured["key"], captured["args"], captured["kwargs"]  # type: ignore[return-value]


def test_prefill_config_exposes_wmma_prefill_field() -> None:
    """PrefillConfig.use_wmma_prefill is a real field with safe default."""

    cfg = PrefillConfig()
    assert cfg.use_wmma_prefill is False  # safe default: opt-in only
    on = PrefillConfig(use_wmma_prefill=True)
    assert on.use_wmma_prefill is True
    # Coercion: non-bool truthy values become True
    coerced = PrefillConfig(use_wmma_prefill=1)  # type: ignore[arg-type]
    assert coerced.use_wmma_prefill is True


def test_exact_pack8_prefill_is_default_for_q8_0_rows_gt_1() -> None:
    """Without WMMA opt-in, raw Q8_0 prefill uses the exact pack8 kernel."""

    key, args, kwargs = _capture_launch(rows=4)
    assert key == _PREFILL_PACK8_BF16
    assert args == (100, 10, 200, 4, 1024, 2048)
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel"}


def test_exact_pack8_prefill_requires_eight_output_columns() -> None:
    key, _, _ = _capture_launch(rows=4, out_features=2049)
    assert key == _PREFILL_BF16


def test_exact_q8_prefill_reuses_weights_across_rows() -> None:
    narrow, _, _ = _capture_launch(rows=8, out_features=512)
    wide, _, _ = _capture_launch(rows=8, out_features=2048)
    large_narrow, _, _ = _capture_launch(rows=32, out_features=512)
    small, _, _ = _capture_launch(rows=7, out_features=2048)

    assert narrow == _PREFILL_TILE8X2_BF16
    assert wide == _PREFILL_TILE8X4_BF16
    assert large_narrow == _PREFILL_TILE8X4_BF16
    assert small == _PREFILL_PACK8_BF16


def test_exact_q8_prefill_widens_columns_at_measured_row_thresholds() -> None:
    narrow, _, _ = _capture_launch(rows=512, out_features=512)
    medium, _, _ = _capture_launch(rows=64, out_features=2048)
    wide, _, _ = _capture_launch(rows=32, out_features=8192)
    narrow_control, _, _ = _capture_launch(rows=256, out_features=512)
    medium_control, _, _ = _capture_launch(rows=32, out_features=2048)
    wide_control, _, _ = _capture_launch(rows=16, out_features=8192)

    assert narrow == _PREFILL_TILE16X4_BF16
    assert medium == _PREFILL_TILE16X4_BF16
    assert wide == _PREFILL_TILE16X4_BF16
    assert narrow_control == _PREFILL_TILE8X4_BF16
    assert medium_control == _PREFILL_TILE8X4_BF16
    assert wide_control == _PREFILL_TILE8X4_BF16


def test_q3_mmq_prefill_policy_is_registry_selected() -> None:
    q3_policy = resolve_q8_mmq_prefill_policy("gguf_ud_q3_k_m")
    assert q3_policy is not None
    assert q3_policy(512, 2048, 8192)
    assert not q3_policy(512, 512, 2048)
    assert not q3_policy(4097, 2048, 8192)
    assert q3_policy.risk_threshold == 1.0e-5
    assert q3_policy.risk_indices_nbytes(512) == 16_777_216
    assert resolve_q8_mmq_prefill_policy("gguf_qwen35") is None


def test_q3_mmq_prefill_session_uses_bounded_workspace(monkeypatch) -> None:
    quantize_calls = []
    correction_calls = []

    class FakeRuntime:
        def __init__(self) -> None:
            self.memset_calls = []

        def memset_async(self, *args) -> None:
            self.memset_calls.append(args)

    def fake_quantize(*args, **kwargs):
        quantize_calls.append((args, kwargs))

    def fake_correct(*args, **kwargs):
        correction_calls.append((args, kwargs))

    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q8_0_mmq128_quantize_bf16_d4x3",
        fake_quantize,
    )
    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q8_0_mmq128_sparse_exact_correct_bf16",
        fake_correct,
    )
    policy = resolve_q8_mmq_prefill_policy("gguf_ud_q3_k_m")
    assert policy is not None
    library = object()
    runtime = FakeRuntime()
    with q8_mmq_prefill_session(
        workspace_ptr=10_000_000,
        workspace_nbytes=3_538_944,
        risk_count_ptr=14_000_000,
        risk_count_nbytes=4,
        risk_indices_ptr=15_000_000,
        risk_indices_nbytes=16_777_216,
        policy=policy,
        library=library,  # type: ignore[arg-type]
    ):
        key, args, kwargs = _capture_launch(
            rows=512,
            in_features=2048,
            out_features=8192,
            runtime=runtime,
            extra_keys=(_PREFILL_MMQ128_X3_GUARDED_BF16,),
        )
    assert key == _PREFILL_MMQ128_X3_GUARDED_BF16
    assert runtime.memset_calls == [(14_000_000, 0, 4, 7)]
    common_kwargs = {"stream": 7, "runtime": runtime, "library": library}
    assert quantize_calls == [((100, 10_000_000, 512, 2048), common_kwargs)]
    assert args == (
        10_000_000,
        10,
        200,
        14_000_000,
        15_000_000,
        4_194_304,
        1.0e-5,
        512,
        2048,
        8192,
    )
    assert kwargs == common_kwargs
    assert correction_calls == [
        (
            (100, 10, 200, 14_000_000, 15_000_000, 4_194_304, 512, 2048, 8192),
            common_kwargs,
        )
    ]


def test_q3_mmq_prefill_session_keeps_exact_below_crossover() -> None:
    policy = resolve_q8_mmq_prefill_policy("gguf_ud_q3_k_m")
    assert policy is not None
    with q8_mmq_prefill_session(
        workspace_ptr=10_000_000,
        workspace_nbytes=3_538_944,
        risk_count_ptr=14_000_000,
        risk_count_nbytes=4,
        risk_indices_ptr=15_000_000,
        risk_indices_nbytes=16_777_216,
        policy=policy,
    ):
        key, _, _ = _capture_launch(
            rows=31,
            in_features=2048,
            out_features=8192,
            extra_keys=(_PREFILL_MMQ128_X3_GUARDED_BF16,),
        )
    assert key == _PREFILL_TILE8X4_BF16


def test_q3_mmq_prefill_session_rejects_undersized_workspace() -> None:
    policy = resolve_q8_mmq_prefill_policy("gguf_ud_q3_k_m")
    assert policy is not None
    with q8_mmq_prefill_session(
        workspace_ptr=10_000_000,
        workspace_nbytes=3_538_943,
        risk_count_ptr=14_000_000,
        risk_count_nbytes=4,
        risk_indices_ptr=15_000_000,
        risk_indices_nbytes=16_777_216,
        policy=policy,
    ):
        with pytest.raises(ValueError, match="workspace is too small"):
            _capture_launch(
                rows=512,
                in_features=2048,
                out_features=8192,
                extra_keys=(_PREFILL_MMQ128_X3_GUARDED_BF16,),
            )


def test_wmma_prefill_kwarg_opts_in_q8_0_rows_gt_1() -> None:
    """Passing ``use_wmma_prefill=True`` rewrites to the WMMA family for Q8_0 rows>1."""

    key, args, kwargs = _capture_launch(rows=4, use_wmma_prefill=True)
    assert key == _WMMA_BF16
    # WMMA ABI is the raw-pointer signature: (x, qweight, out, rows, in_f, out_f)
    assert args == (100, 10, 200, 4, 1024, 2048)
    # threads should NOT be present on the WMMA path (it takes tile_m / tile_n)
    assert "threads" not in kwargs
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel"}


def test_wmma_prefill_kwarg_can_force_off_even_with_session_on() -> None:
    """Per-call ``use_wmma_prefill=False`` wins over an enabled session."""

    set_wmma_prefill_enabled(True)
    key, _, _ = _capture_launch(rows=4, use_wmma_prefill=False)
    assert key == _PREFILL_PACK8_BF16


def test_wmma_prefill_env_var_opts_in(monkeypatch) -> None:
    """``HIPENGINE_GGUF_WMMA_PREFILL=1`` enables the rewrite without any kwarg."""

    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "1")
    key, _, _ = _capture_launch(rows=4)
    assert key == _WMMA_BF16


def test_wmma_prefill_env_var_accepts_common_truthy_values(monkeypatch) -> None:
    for value in ("1", "true", "TRUE", "yes", "On"):
        monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", value)
        key, _, _ = _capture_launch(rows=4)
        assert key == _WMMA_BF16, f"env value {value!r} should enable WMMA"


def test_wmma_prefill_env_var_falsy_values_keep_exact_pack8_path(monkeypatch) -> None:
    for value in ("", "0", "false", "no", "off"):
        monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", value)
        key, _, _ = _capture_launch(rows=4)
        assert key == _PREFILL_PACK8_BF16, f"env value {value!r} should keep exact pack8"


def test_wmma_prefill_session_toggle_persists_until_cleared() -> None:
    """``set_wmma_prefill_enabled(True)`` enables until cleared with ``None``."""

    set_wmma_prefill_enabled(True)
    key, _, _ = _capture_launch(rows=4)
    assert key == _WMMA_BF16
    set_wmma_prefill_enabled(False)
    key, _, _ = _capture_launch(rows=4)
    assert key == _PREFILL_PACK8_BF16
    set_wmma_prefill_enabled(None)
    # Back to the env default (unset in this fixture) -> exact pack8.
    key, _, _ = _capture_launch(rows=4)
    assert key == _PREFILL_PACK8_BF16


def test_wmma_prefill_session_context_manager_restores_previous_state() -> None:
    set_wmma_prefill_enabled(False)  # baseline: explicit off
    with wmma_prefill_session(True):
        key, _, _ = _capture_launch(rows=4)
        assert key == _WMMA_BF16
    # Restored to the previous explicit-off session state.
    key, _, _ = _capture_launch(rows=4)
    assert key == _PREFILL_PACK8_BF16


def test_wmma_prefill_decode_path_unaffected_by_opt_in() -> None:
    """rows==1 never gets the WMMA rewrite, regardless of opt-in state."""

    set_wmma_prefill_enabled(True)
    key, _, _ = _capture_launch(rows=1)
    # rows==1 Q8_0 raw resolves to the pack8 decode GEMV alias (out_features %
    # 8 == 0 path), never to WMMA prefill.
    assert key == _DECODE_PACK8_BF16


def test_wmma_prefill_raw_q4_k_off_by_default_rows_gt_1() -> None:
    """Raw Q4_K rows>1 keeps the decode-shaped prefill alias unless opted in."""

    key, _, _ = _capture_launch(rows=4, quant_key="gguf_q4_k", layout=LAYOUT_RAW_GGUF)
    assert key == _Q4_PREFILL_BF16


def test_wmma_prefill_kwarg_opts_in_raw_q4_k_rows_gt_1() -> None:
    """Per-call opt-in routes raw Q4_K rows>1 to the new P8.2 WMMA family."""

    key, args, kwargs = _capture_launch(
        rows=4,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        in_features=1024,
        out_features=2048,
        use_wmma_prefill=True,
        threads=128,
    )
    assert key == _Q4_WMMA_BF16
    assert args == (100, 10, 200, 4, 1024, 2048)
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel"}


@pytest.mark.parametrize(
    ("output_dtype", "prefill_key", "wmma_key"),
    [
        (
            GGUF_OUTPUT_FP16,
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "prefill_bf16_fp16_out"),
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "wmma_prefill_bf16_fp16_out"),
        ),
        (
            GGUF_OUTPUT_F32,
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "prefill_bf16_f32_out"),
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "wmma_prefill_bf16_f32_out"),
        ),
    ],
)
def test_wmma_prefill_raw_q4_k_output_dtype_variants_route(
    output_dtype: str, prefill_key: KernelKey, wmma_key: KernelKey
) -> None:
    """Raw Q4_K FP16/F32 output variants also rewrite to matching WMMA keys."""

    key, _, _ = _capture_launch(
        rows=4,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        output_dtype=output_dtype,
        use_wmma_prefill=False,
        extra_keys=(prefill_key, wmma_key),
    )
    assert key == prefill_key
    key, args, kwargs = _capture_launch(
        rows=4,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        output_dtype=output_dtype,
        use_wmma_prefill=True,
        extra_keys=(prefill_key, wmma_key),
    )
    assert key == wmma_key
    assert args == (100, 10, 200, 4, 1024, 2048)
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel"}


def test_wmma_prefill_env_var_opts_in_raw_q4_k(monkeypatch) -> None:
    """The env opt-in applies to raw Q4_K as well as Q8_0."""

    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "1")
    key, _, _ = _capture_launch(rows=4, quant_key="gguf_q4_k", layout=LAYOUT_RAW_GGUF)
    assert key == _Q4_WMMA_BF16


def test_wmma_prefill_session_opts_in_raw_q4_k() -> None:
    """The session toggle applies to raw Q4_K and can be forced off per call."""

    set_wmma_prefill_enabled(True)
    key, _, _ = _capture_launch(rows=4, quant_key="gguf_q4_k", layout=LAYOUT_RAW_GGUF)
    assert key == _Q4_WMMA_BF16
    key, _, _ = _capture_launch(
        rows=4,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        use_wmma_prefill=False,
    )
    assert key == _Q4_PREFILL_BF16


def test_wmma_prefill_raw_q4_k_decode_path_unaffected_by_opt_in() -> None:
    """rows==1 raw Q4_K stays on the scalar raw GEMV path."""

    key, _, _ = _capture_launch(
        rows=1,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        use_wmma_prefill=True,
    )
    assert key == _Q4_GEMV_BF16


def test_wmma_prefill_q4_k_pack8_layout_keeps_pack8_fallback_under_opt_in() -> None:
    """Dense 2D Q4_K pack8 materialization is not silently reinterpreted as raw."""

    key, args, kwargs = _capture_launch(
        rows=4,
        quant_key="gguf_q4_k",
        layout=LAYOUT_Q4_K_PACK8,
        use_wmma_prefill=True,
    )
    assert key == _Q4_PACK8_PREFILL_BF16
    assert args == (100, 11, 12, 13, 200, 4, 1024, 2048)
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel"}


def test_wmma_prefill_raw_q4_k_requires_256_aligned_in_features() -> None:
    """Raw Q4_K WMMA requires in_features % 256 == 0; otherwise fallback."""

    key, _, _ = _capture_launch(
        rows=4,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        in_features=1000,
        use_wmma_prefill=True,
    )
    assert key == _Q4_PREFILL_BF16


def test_wmma_prefill_q5_k_not_yet_supported_keeps_decode_path() -> None:
    """Q5_K does not yet ship a WMMA prefill family (lands in P8.5)."""

    q5_prefill = KernelKey("hip_gfx1100", "linear", "gguf_q5_k", "prefill_bf16_bf16_out")
    key, _, _ = _capture_launch(
        rows=4, quant_key="gguf_q5_k", use_wmma_prefill=True, extra_keys=(q5_prefill,)
    )
    assert key == q5_prefill


def test_wmma_prefill_unaligned_in_features_falls_back_to_exact_pack8_path() -> None:
    """Q8_0 shapes outside WMMA policy retain the exact pack8 schedule."""

    key, _, _ = _capture_launch(rows=4, in_features=1000, use_wmma_prefill=True)
    assert key == _PREFILL_PACK8_BF16


def test_wmma_prefill_threads_silently_dropped_on_wmma_path() -> None:
    """The caller's ``threads`` value applies to the decode path only."""

    key, _, kwargs = _capture_launch(rows=4, use_wmma_prefill=True, threads=128)
    assert key == _WMMA_BF16
    assert "threads" not in kwargs
    # And confirm threads still flows through on the exact pack8 path:
    key2, _, kwargs2 = _capture_launch(rows=4, threads=128)
    assert key2 == _PREFILL_PACK8_BF16
    assert kwargs2.get("threads") == 128


def test_t16_pair_concat_fuses_q8_shared_gate_up() -> None:
    """Resident Q8T16 shared gate/up can use the concatenated dual ABI."""

    weight_a = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_b = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    import hipengine.runtime.gguf_linear as gl

    pair_calls: list[tuple] = []

    def fake_pair(*args, **kwargs):
        pair_calls.append((args, kwargs))

    original = gl.gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out
    gl.gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out = fake_pair  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_pair_concat(
            weight_a,
            weight_b,
            x_ptr=100,
            out_ptr=200,
            rows=1,
            in_features=2048,
            out_features=512,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        gl.gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out = original  # type: ignore[assignment]

    assert fused is True
    assert pair_calls == [
        ((100, 14, 14, 200, 1, 2048, 512, 512), {"threads": 0, "stream": 7, "runtime": "runtime-sentinel"})
    ]


def test_t16_pair_fuses_q8_separate_outputs() -> None:
    """Resident Q8T16 same-input pairs can share one split-output launch."""

    weight_a = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_b = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    import hipengine.runtime.gguf_linear as gl

    pair_calls: list[tuple] = []

    def fake_pair(*args, **kwargs):
        pair_calls.append((args, kwargs))

    original = gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out
    gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = fake_pair  # type: ignore[assignment]
    try:
        fused_equal = launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=2048,
            out_features=512,
            stream=7,
            runtime="runtime-sentinel",
        )
        fused_unequal = launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=101,
            out_a_ptr=201,
            out_b_ptr=301,
            rows=1,
            in_features=2048,
            out_features=1536,
            out_features_b=512,
            stream=8,
            runtime="runtime-sentinel",
        )
    finally:
        gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = original  # type: ignore[assignment]

    assert fused_equal is True
    assert fused_unequal is True
    assert pair_calls == [
        ((100, 14, 14, 200, 300, 1, 2048, 512, 512), {"threads": 0, "stream": 7, "runtime": "runtime-sentinel"}),
        ((101, 14, 14, 201, 301, 1, 2048, 1536, 512), {"threads": 0, "stream": 8, "runtime": "runtime-sentinel"}),
    ]


def test_t16_pair_qwen35_rows_gt1_routes_to_rowtile4_when_opted_in(monkeypatch) -> None:
    """The large qwen35 attn_qkv+attn_gate verifier pair can opt into rowtile4."""

    weight_a = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_b = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    import hipengine.runtime.gguf_linear as gl

    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_THREADS", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE", "1")
    gl.clear_gguf_linear_dispatch_cache()
    calls: list[tuple[str, tuple, dict]] = []

    def fake_exact(*args, **kwargs):
        calls.append(("exact", args, kwargs))

    def fake_rowtile(*args, **kwargs):
        calls.append(("rowtile4", args, kwargs))

    original_exact = gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out
    original_rowtile = gl.gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out
    gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = fake_exact  # type: ignore[assignment]
    gl.gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out = fake_rowtile  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=3,
            in_features=2048,
            out_features=8192,
            out_features_b=4096,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = original_exact  # type: ignore[assignment]
        gl.gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out = original_rowtile  # type: ignore[assignment]
        gl.clear_gguf_linear_dispatch_cache()

    assert fused is True
    assert calls == [
        (
            "rowtile4",
            (100, 14, 14, 200, 300, 3, 2048, 8192, 4096),
            {"threads": 128, "stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_t16_pair_rowtile_session_scopes_exact_c8_width(monkeypatch) -> None:
    weight_a = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_b = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    import hipengine.runtime.gguf_linear as gl

    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_PAIR_COL8", raising=False)
    gl.clear_gguf_linear_dispatch_cache()
    calls: list[tuple[str, int, int]] = []

    def fake_exact(*args, **kwargs):
        calls.append(("exact", int(args[5]), int(kwargs["threads"])))

    def fake_rowtile(*args, **kwargs):
        calls.append(("rowtile4", int(args[5]), int(kwargs["threads"])))

    def fake_col8(*args, **kwargs):
        calls.append(("rowtile4_col8", int(args[5]), int(kwargs["threads"])))

    original_exact = gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out
    original_rowtile = gl.gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out
    original_col8 = gl.gguf_q8_0_t16_dual_gemv_decode_rowtile4_col8_bf16_bf16_out
    gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = fake_exact  # type: ignore[assignment]
    gl.gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out = fake_rowtile  # type: ignore[assignment]
    gl.gguf_q8_0_t16_dual_gemv_decode_rowtile4_col8_bf16_bf16_out = fake_col8  # type: ignore[assignment]
    try:
        with gl.q8_t16_pair_rowtile_min_rows_session(8):
            for rows in (4, 8):
                assert launch_gguf_linear_pair(
                    weight_a,
                    weight_b,
                    x_ptr=100,
                    out_a_ptr=200,
                    out_b_ptr=300,
                    rows=rows,
                    in_features=2048,
                    out_features=8192,
                    out_features_b=4096,
                )
        monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_PAIR_COL8", "0")
        with gl.q8_t16_pair_rowtile_min_rows_session(8):
            assert launch_gguf_linear_pair(
                weight_a,
                weight_b,
                x_ptr=100,
                out_a_ptr=200,
                out_b_ptr=300,
                rows=8,
                in_features=2048,
                out_features=8192,
                out_features_b=4096,
            )
        monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE", "0")
        with gl.q8_t16_pair_rowtile_min_rows_session(8):
            assert launch_gguf_linear_pair(
                weight_a,
                weight_b,
                x_ptr=100,
                out_a_ptr=200,
                out_b_ptr=300,
                rows=8,
                in_features=2048,
                out_features=8192,
                out_features_b=4096,
            )
    finally:
        gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = original_exact  # type: ignore[assignment]
        gl.gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out = original_rowtile  # type: ignore[assignment]
        gl.gguf_q8_0_t16_dual_gemv_decode_rowtile4_col8_bf16_bf16_out = original_col8  # type: ignore[assignment]
        gl.clear_gguf_linear_dispatch_cache()

    assert calls == [
        ("exact", 4, 0),
        ("rowtile4_col8", 8, 128),
        ("rowtile4", 8, 128),
        ("exact", 8, 0),
    ]


def test_t16_pair_rowtile_opt_out_keeps_exact_wrapper(monkeypatch) -> None:
    weight_a = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_b = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    import hipengine.runtime.gguf_linear as gl

    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE", "0")
    gl.clear_gguf_linear_dispatch_cache()
    calls: list[tuple[str, tuple, dict]] = []

    def fake_exact(*args, **kwargs):
        calls.append(("exact", args, kwargs))

    def fake_rowtile(*args, **kwargs):
        calls.append(("rowtile4", args, kwargs))

    original_exact = gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out
    original_rowtile = gl.gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out
    gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = fake_exact  # type: ignore[assignment]
    gl.gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out = fake_rowtile  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=3,
            in_features=2048,
            out_features=8192,
            out_features_b=4096,
            runtime="runtime-sentinel",
        )
    finally:
        gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = original_exact  # type: ignore[assignment]
        gl.gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out = original_rowtile  # type: ignore[assignment]
        gl.clear_gguf_linear_dispatch_cache()

    assert fused is True
    assert calls == [
        (
            "exact",
            (100, 14, 14, 200, 300, 3, 2048, 8192, 4096),
            {"threads": 0, "stream": 0, "runtime": "runtime-sentinel"},
        )
    ]


def test_t16_single_qwen35_rows_gt1_routes_to_rowtile4_when_all_opted_in(monkeypatch) -> None:
    weight = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    import hipengine.runtime.gguf_linear as gl

    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_THREADS", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL", "1")
    gl.clear_gguf_linear_dispatch_cache()
    calls: list[tuple[str, tuple, dict]] = []

    def fake_exact(*args, **kwargs):
        calls.append(("exact", args, kwargs))

    def fake_rowtile(*args, **kwargs):
        calls.append(("rowtile4", args, kwargs))

    original_exact = gl._LAUNCH_ABI["t16"]
    original_rowtile = gl.gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out
    gl._LAUNCH_ABI["t16"] = fake_exact
    gl.gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out = fake_rowtile  # type: ignore[assignment]
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=3,
            in_features=2048,
            out_features=4096,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        gl._LAUNCH_ABI["t16"] = original_exact
        gl.gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out = original_rowtile  # type: ignore[assignment]
        gl.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            "rowtile4",
            (100, 14, 200, 3, 2048, 4096),
            {"threads": 128, "stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_t16_rowtile_all_uses_packed_session_with_explicit_opt_out(monkeypatch) -> None:
    weight = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    import hipengine.runtime.gguf_linear as gl

    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_THREADS", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL", raising=False)
    gl.clear_gguf_linear_dispatch_cache()
    calls: list[str] = []

    def fake_exact(*args, **kwargs):
        calls.append("exact")

    def fake_rowtile(*args, **kwargs):
        calls.append("rowtile4")

    original_exact = gl._LAUNCH_ABI["t16"]
    original_rowtile = gl.gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out
    gl._LAUNCH_ABI["t16"] = fake_exact
    gl.gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out = fake_rowtile  # type: ignore[assignment]
    try:
        with gl.q8_t16_rowtile_all_session(True):
            launch_gguf_linear(
                weight,
                x_ptr=100,
                out_ptr=200,
                rows=4,
                in_features=2048,
                out_features=4096,
                backend="hip_gfx1151",
            )
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=4,
            in_features=2048,
            out_features=4096,
            backend="hip_gfx1151",
        )
        monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL", "0")
        gl.clear_gguf_linear_dispatch_cache()
        with gl.q8_t16_rowtile_all_session(True):
            launch_gguf_linear(
                weight,
                x_ptr=100,
                out_ptr=200,
                rows=4,
                in_features=2048,
                out_features=4096,
                backend="hip_gfx1151",
            )
    finally:
        gl._LAUNCH_ABI["t16"] = original_exact
        gl.gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out = original_rowtile  # type: ignore[assignment]
        gl.clear_gguf_linear_dispatch_cache()

    assert calls == ["rowtile4", "exact", "exact"]


def test_t16_triple_fuses_q8_separate_outputs() -> None:
    """Resident Q8T16 full-attention Q/K/V can share one split-output launch."""

    weight_a = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_b = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_c = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    import hipengine.runtime.gguf_linear as gl

    triple_calls: list[tuple] = []

    def fake_triple(*args, **kwargs):
        triple_calls.append((args, kwargs))

    original = gl.gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out
    gl.gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out = fake_triple  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_triple(
            weight_a,
            weight_b,
            weight_c,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            out_c_ptr=400,
            rows=1,
            in_features=2048,
            out_features=1024,
            out_features_b=512,
            out_features_c=512,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        gl.gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out = original  # type: ignore[assignment]

    assert fused is True
    assert triple_calls == [
        (
            (100, 14, 14, 14, 200, 300, 400, 1, 2048, 1024, 512, 512),
            {"threads": 0, "stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_t16_triple_qwen35_rows_gt1_routes_to_rowtile4_when_all_opted_in(monkeypatch) -> None:
    weight_a = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_b = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_c = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    import hipengine.runtime.gguf_linear as gl

    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_THREADS", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL", "1")
    gl.clear_gguf_linear_dispatch_cache()
    calls: list[tuple[str, tuple, dict]] = []

    def fake_exact(*args, **kwargs):
        calls.append(("exact", args, kwargs))

    def fake_rowtile(*args, **kwargs):
        calls.append(("rowtile4", args, kwargs))

    original_exact = gl.gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out
    original_rowtile = gl.gguf_q8_0_t16_triple_gemv_decode_rowtile4_bf16_bf16_out
    gl.gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out = fake_exact  # type: ignore[assignment]
    gl.gguf_q8_0_t16_triple_gemv_decode_rowtile4_bf16_bf16_out = fake_rowtile  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_triple(
            weight_a,
            weight_b,
            weight_c,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            out_c_ptr=400,
            rows=3,
            in_features=2048,
            out_features=4096,
            out_features_b=1024,
            out_features_c=1024,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        gl.gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out = original_exact  # type: ignore[assignment]
        gl.gguf_q8_0_t16_triple_gemv_decode_rowtile4_bf16_bf16_out = original_rowtile  # type: ignore[assignment]
        gl.clear_gguf_linear_dispatch_cache()

    assert fused is True
    assert calls == [
        (
            "rowtile4",
            (100, 14, 14, 14, 200, 300, 400, 3, 2048, 4096, 1024, 1024),
            {"threads": 128, "stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_wmma_prefill_pair_declines_fusion_when_q8_0_rows_gt_1() -> None:
    """Pair fast paths defer to two singletons when Q8_0 rows>1 + WMMA opt-in.

    There is no Q8_0 dual WMMA prefill yet (follow-up P8 step). The pair
    function must return ``False`` so the caller falls back to two
    ``launch_gguf_linear`` calls that each hit the WMMA family.
    """

    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    fused = launch_gguf_linear_pair(
        weight_a,
        weight_b,
        x_ptr=100,
        out_a_ptr=200,
        out_b_ptr=300,
        rows=4,
        in_features=1024,
        out_features=2048,
        use_wmma_prefill=True,
    )
    assert fused is False


def test_gfx1151_q8_t16_dual_wmma_prefill_routes_exact_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    key = KernelKey(
        "hip_gfx1151",
        "linear_pair",
        "gguf_q8_0_t16_v1",
        "t16_dual_wmma_prefill_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant
    )
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, capture, replace=True)
    weight_a = _fake_weight(
        layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1"
    )
    weight_b = _fake_weight(
        layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1"
    )
    try:
        with gguf_linear_module.q8_t16_dual_wmma_prefill_session(True):
            assert launch_gguf_linear_pair(
                weight_a, weight_b, 100, 200, 300, 512, 1_024, 16,
                backend="hip_gfx1151", use_wmma_prefill=True, stream=7,
                runtime="runtime-sentinel",
            )
            monkeypatch.setenv("HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL", "0")
            assert not launch_gguf_linear_pair(
                weight_a, weight_b, 100, 200, 300, 512, 1_024, 16,
                backend="hip_gfx1151", use_wmma_prefill=True,
            )
            monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL")
            assert not launch_gguf_linear_pair(
                weight_a, weight_b, 100, 200, 300, 511, 1_024, 16,
                backend="hip_gfx1151", use_wmma_prefill=True,
            )
    finally:
        register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (100, 14, 14, 200, 300, 512, 1_024, 16),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_wmma_prefill_pair_fuses_raw_q4_k_dual_prefill_when_opted_in() -> None:
    """Raw Q4_K gate+up pair routes to the P8.2 dual WMMA path."""

    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    import hipengine.runtime.gguf_linear as gl

    pair_calls: list[tuple] = []

    def fake_pair(*args, **kwargs):
        pair_calls.append((args, kwargs))

    original = gl.gguf_q4_k_wmma_prefill_dual_bf16_bf16_out
    gl.gguf_q4_k_wmma_prefill_dual_bf16_bf16_out = fake_pair  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=4,
            in_features=1024,
            out_features=2048,
            stream=7,
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
        )
    finally:
        gl.gguf_q4_k_wmma_prefill_dual_bf16_bf16_out = original  # type: ignore[assignment]
    assert fused is True
    assert pair_calls == [
        ((100, 10, 10, 200, 300, 4, 1024, 2048), {"stream": 7, "runtime": "runtime-sentinel"})
    ]


def test_wmma_prefill_pair_raw_q4_k_requires_opt_in() -> None:
    """Raw Q4_K pair has no default-off pair fast path; callers fall back."""

    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    fused = launch_gguf_linear_pair(
        weight_a,
        weight_b,
        x_ptr=100,
        out_a_ptr=200,
        out_b_ptr=300,
        rows=4,
        in_features=1024,
        out_features=2048,
        use_wmma_prefill=False,
    )
    assert fused is False


def test_wmma_prefill_pair_raw_q4_k_unaligned_falls_back() -> None:
    """Raw Q4_K dual WMMA pair requires the 256-wide Q4_K block alignment."""

    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    import hipengine.runtime.gguf_linear as gl

    pair_calls: list[tuple] = []

    def fake_pair(*args, **kwargs):
        pair_calls.append((args, kwargs))

    original = gl.gguf_q4_k_wmma_prefill_dual_bf16_bf16_out
    gl.gguf_q4_k_wmma_prefill_dual_bf16_bf16_out = fake_pair  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=4,
            in_features=1000,
            out_features=2048,
            use_wmma_prefill=True,
        )
    finally:
        gl.gguf_q4_k_wmma_prefill_dual_bf16_bf16_out = original  # type: ignore[assignment]
    assert fused is False
    assert pair_calls == []


def test_wmma_prefill_pair_still_fuses_q4_k_pack8_dual_prefill() -> None:
    """WMMA opt-in does NOT poison the Q4_K pack8 dual prefill fast path."""

    weight_a = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    # Stub out the actual pair kernel so we don't touch the GPU.
    import hipengine.runtime.gguf_linear as gl

    pair_calls: list[tuple] = []

    def fake_pair(*args, **kwargs):
        pair_calls.append((args, kwargs))

    original = gl.gguf_q4_k_pack8_dual_prefill_bf16_bf16_out
    gl.gguf_q4_k_pack8_dual_prefill_bf16_bf16_out = fake_pair  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=4,
            in_features=1024,
            out_features=2048,
            use_wmma_prefill=True,
        )
    finally:
        gl.gguf_q4_k_pack8_dual_prefill_bf16_bf16_out = original  # type: ignore[assignment]
    assert fused is True
    assert len(pair_calls) == 1


def test_gfx1151_pack8_bulk_pair_invokes_registered_wmma_owner(monkeypatch) -> None:
    """Bulk pair routing must execute the function behind its four-axis key."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    import hipengine.runtime.gguf_linear as gl

    register_gfx1151_kernels(replace=True)
    weight_a = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k",
        "pack8_wmma_prefill_bf16_bf16_out",
    )
    exact_key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k",
        "pack8_exact_prefill_tile8x8_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    exact_original = resolve(
        backend=exact_key.backend,
        layer=exact_key.layer,
        quant=exact_key.quant,
        variant=exact_key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def registered(*args, **kwargs):
        calls.append((args, kwargs))

    def unexpected_direct(*args, **kwargs):
        raise AssertionError("pair route bypassed the registered WMMA owner")

    register(key, registered, replace=True)
    # A populated exact singleton normally makes the pair owner decline so the
    # caller launches two registered singletons. Remove it to exercise the
    # pair function's capability fallback, which must still honor the registry.
    unregister(exact_key)
    monkeypatch.setattr(
        gl,
        "gguf_q4_k_pack8_wmma_prefill_gfx1151_bf16_bf16_out",
        unexpected_direct,
        raising=False,
    )
    try:
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=512,
            in_features=1024,
            out_features=3584,
            backend="hip_gfx1151",
            stream=7,
            libraries={
                "gguf_q4_k:pack8_wmma_prefill_bf16_bf16_out": "wmma-library"
            },
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
        )
    finally:
        register(key, original, replace=True)
        register(exact_key, exact_original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (100, 11, 12, 13, 200, 512, 1024, 3584),
            {
                "stream": 7,
                "runtime": "runtime-sentinel",
                "library": "wmma-library",
            },
        ),
        (
            (100, 11, 12, 13, 300, 512, 1024, 3584),
            {
                "stream": 7,
                "runtime": "runtime-sentinel",
                "library": "wmma-library",
            },
        ),
    ]


def test_gfx1151_pack8_dual_wmma_silu_is_shape_scoped_and_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operation-complete bulk owner must fail closed around p512 H1024."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight_a = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    key = KernelKey(
        "hip_gfx1151",
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_wmma_prefill_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple[tuple, dict]] = []
    register(key, lambda *args, **kwargs: calls.append((args, kwargs)), replace=True)
    try:
        with (
            wmma_prefill_session(True),
            q4_pack8_dual_wmma_silu_prefill_session(True),
        ):
            assert launch_gguf_linear_pair_silu(
                weight_a,
                weight_b,
                x_ptr=100,
                out_ptr=200,
                rows=512,
                in_features=1024,
                out_features=3584,
                backend="hip_gfx1151",
                stream=7,
                libraries={"gguf_q4_k": "wmma-library"},
                runtime="runtime-sentinel",
            )
            assert not launch_gguf_linear_pair_silu(
                weight_a,
                weight_b,
                x_ptr=100,
                out_ptr=200,
                rows=511,
                in_features=1024,
                out_features=3584,
                backend="hip_gfx1151",
            )
            assert not launch_gguf_linear_pair_silu(
                weight_a,
                weight_b,
                x_ptr=100,
                out_ptr=200,
                rows=512,
                in_features=1024,
                out_features=3584,
                backend="hip_gfx1100",
            )
            monkeypatch.setenv(
                "HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL",
                "0",
            )
            assert not launch_gguf_linear_pair_silu(
                weight_a,
                weight_b,
                x_ptr=100,
                out_ptr=200,
                rows=512,
                in_features=1024,
                out_features=3584,
                backend="hip_gfx1151",
            )
    finally:
        register(key, original, replace=True)

    assert calls == [
        (
            (100, 11, 12, 13, 11, 12, 13, 200, 512, 1024, 3584),
            {
                "stream": 7,
                "runtime": "runtime-sentinel",
                "library": "wmma-library",
            },
        )
    ]


def test_gfx1151_q4_k_pack8_decode_pair_uses_registered_dual_owner() -> None:
    """Laguna c=1 gate/up may reuse the exact local32 dual pack8 body."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight_a = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    pair_key = KernelKey(
        "hip_gfx1151",
        "linear_pair",
        "gguf_q4_k",
        "pack8_dual_decode_bf16_bf16_out",
    )
    original = resolve(
        backend=pair_key.backend,
        layer=pair_key.layer,
        quant=pair_key.quant,
        variant=pair_key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def fake_pair(*args, **kwargs):
        calls.append((args, kwargs))

    register(pair_key, fake_pair, replace=True)
    try:
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=3072,
            out_features=1024,
            backend="hip_gfx1151",
            stream=7,
            runtime="runtime-sentinel",
            registered_decode_only=True,
        )
        assert not launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=3072,
            out_features=1024,
            backend="hip_gfx1100",
        )
    finally:
        register(pair_key, original, replace=True)

    assert calls == [
        (
            (100, 11, 12, 13, 11, 12, 13, 200, 300, 1, 3072, 1024),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_gfx1151_q4_k_pack8_decode_pair_silu_uses_registered_owner() -> None:
    """The exact fused consumer remains gfx1151/Q4/rows-one scoped."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight_a = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    fused_key = KernelKey(
        "hip_gfx1151",
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_decode_bf16_bf16_out",
    )
    original = resolve(
        backend=fused_key.backend,
        layer=fused_key.layer,
        quant=fused_key.quant,
        variant=fused_key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def fake_fused(*args, **kwargs):
        calls.append((args, kwargs))

    register(fused_key, fake_fused, replace=True)
    try:
        assert launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            x_ptr=100,
            out_ptr=400,
            rows=1,
            in_features=3072,
            out_features=1024,
            backend="hip_gfx1151",
            stream=7,
            runtime="runtime-sentinel",
            use_gemv_decode=True,
        )
        assert not launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            x_ptr=100,
            out_ptr=400,
            rows=2,
            in_features=3072,
            out_features=1024,
            backend="hip_gfx1151",
            use_gemv_decode=True,
        )
        assert not launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            x_ptr=100,
            out_ptr=400,
            rows=1,
            in_features=3072,
            out_features=1024,
            backend="hip_gfx1100",
            use_gemv_decode=True,
        )
    finally:
        register(fused_key, original, replace=True)

    assert calls == [
        (
            (100, 11, 12, 13, 11, 12, 13, 400, 1, 3072, 1024),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_gfx1151_q4_k_pack8_decode_pair_silu_t128_uses_registered_owner() -> None:
    """The qualified 0.8B schedule resolves a distinct four-axis variant."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight_a = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    fused_key = KernelKey(
        "hip_gfx1151",
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_decode_t128_bf16_bf16_out",
    )
    original = resolve(
        backend=fused_key.backend,
        layer=fused_key.layer,
        quant=fused_key.quant,
        variant=fused_key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def fake_fused(*args, **kwargs):
        calls.append((args, kwargs))

    register(fused_key, fake_fused, replace=True)
    try:
        assert launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            x_ptr=100,
            out_ptr=400,
            rows=1,
            in_features=1024,
            out_features=3584,
            backend="hip_gfx1151",
            stream=7,
            runtime="runtime-sentinel",
            use_gemv_decode=True,
            registered_decode_variant="pack8_dual_decode_t128_bf16_bf16_out",
        )
    finally:
        register(fused_key, original, replace=True)

    assert calls == [
        (
            (100, 11, 12, 13, 11, 12, 13, 400, 1, 1024, 3584),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


@pytest.mark.parametrize(
    "variant",
    [
        "dense_dual_q8_1x2_dp4a_bf16_bf16_out",
        "dense_dual_q8_1x2_split_weight_dp4a_bf16_bf16_out",
    ],
)
def test_gfx1151_q4_t16_dense_pair_q8x2_quantizes_workspace(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    """The qualified dense route packs two Q8_1 planes before its T16 owner."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight_a = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    weight_b = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    fused_key = KernelKey(
        "hip_gfx1151",
        "linear_pair_silu",
        "gguf_q4_k_t16_v1",
        variant,
    )
    original = resolve(
        backend=fused_key.backend,
        layer=fused_key.layer,
        quant=fused_key.quant,
        variant=fused_key.variant,
    )
    quantize_calls: list[tuple[tuple, dict]] = []
    fused_calls: list[tuple[tuple, dict]] = []

    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_quantize_bf16_q8_1x2",
        lambda *args, **kwargs: quantize_calls.append((args, kwargs)),
    )
    register(
        fused_key,
        lambda *args, **kwargs: fused_calls.append((args, kwargs)),
        replace=True,
    )
    try:
        assert not launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            x_ptr=100,
            out_ptr=400,
            rows=1,
            in_features=5_120,
            out_features=17_408,
            backend="hip_gfx1151",
            use_gemv_decode=True,
            registered_decode_variant=variant,
        )
        assert launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            x_ptr=100,
            out_ptr=400,
            rows=1,
            in_features=5_120,
            out_features=17_408,
            backend="hip_gfx1151",
            stream=7,
            runtime="runtime-sentinel",
            use_gemv_decode=True,
            registered_decode_variant=variant,
            q8_1_workspace_ptr=900,
        )
    finally:
        register(fused_key, original, replace=True)

    assert quantize_calls == [
        (
            (100, 900, 1, 5_120),
            {
                "stream": 7,
                "library": None,
                "runtime": "runtime-sentinel",
            },
        )
    ]
    assert fused_calls == [
        (
            (900, 14, 14, 400, 1, 5_120, 17_408),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


@pytest.mark.parametrize(
    ("rows", "chunks"),
    (
        (6, ((6, 0),)),
        (9, ((7, 0), (2, 7))),
        (12, ((8, 0), (4, 8))),
    ),
)
def test_gfx1151_production_verifier_q4_scope_chunks_single_rowtiles(
    rows: int,
    chunks: tuple[tuple[int, int], ...],
) -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    candidate_key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k_t16_v1",
        "dense_rowtile_bf16_bf16_out",
    )
    original = resolve(
        backend=candidate_key.backend,
        layer=candidate_key.layer,
        quant=candidate_key.quant,
        variant=candidate_key.variant,
    )
    calls: list[tuple[tuple, dict]] = []
    register(
        candidate_key,
        lambda *args, **kwargs: calls.append((args, kwargs)),
        replace=True,
    )
    try:
        with target_verifier_production_q4_rowtile_session(True):
            launch_gguf_linear(
                weight,
                x_ptr=100,
                out_ptr=400,
                rows=rows,
                in_features=5_120,
                out_features=12_288,
                backend="hip_gfx1151",
                stream=7,
                runtime="runtime-sentinel",
                use_wmma_prefill=False,
            )
    finally:
        register(candidate_key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (
                100 + row_base * 5_120 * 2,
                14,
                400 + row_base * 12_288 * 2,
                chunk_rows,
                5_120,
                12_288,
            ),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
        for chunk_rows, row_base in chunks
    ]


@pytest.mark.parametrize(
    ("rows", "chunks"),
    (
        (6, ((6, 0),)),
        (9, ((7, 0), (2, 7))),
        (12, ((8, 0), (4, 8))),
    ),
)
def test_gfx1151_production_verifier_q4_scope_chunks_gate_up_rowtiles(
    rows: int,
    chunks: tuple[tuple[int, int], ...],
) -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight_a = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    weight_b = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    candidate_key = KernelKey(
        "hip_gfx1151",
        "linear_pair_silu",
        "gguf_q4_k_t16_v1",
        "dense_dual_rowtile_bf16_bf16_out",
    )
    original = resolve(
        backend=candidate_key.backend,
        layer=candidate_key.layer,
        quant=candidate_key.quant,
        variant=candidate_key.variant,
    )
    calls: list[tuple[tuple, dict]] = []
    register(
        candidate_key,
        lambda *args, **kwargs: calls.append((args, kwargs)),
        replace=True,
    )
    try:
        with target_verifier_production_q4_rowtile_session(True):
            assert launch_gguf_linear_pair_silu(
                weight_a,
                weight_b,
                x_ptr=100,
                out_ptr=400,
                rows=rows,
                in_features=5_120,
                out_features=17_408,
                backend="hip_gfx1151",
                stream=7,
                runtime="runtime-sentinel",
            )
    finally:
        register(candidate_key, original, replace=True)

    assert calls == [
        (
            (
                100 + row_base * 5_120 * 2,
                14,
                14,
                400 + row_base * 17_408 * 2,
                chunk_rows,
                5_120,
                17_408,
            ),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
        for chunk_rows, row_base in chunks
    ]


def test_gfx1151_qmicro_q8x2_rowbatch_quantizes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native rows2-4 policy shares weights without changing c1 math."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight_a = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_QMICRO_T16,
        quant_key="gguf_q4_k_qmicro_t16_v1",
    )
    weight_b = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_QMICRO_T16,
        quant_key="gguf_q4_k_qmicro_t16_v1",
    )
    candidate_key = KernelKey(
        "hip_gfx1151",
        "linear_pair_silu",
        "gguf_q4_k_qmicro_t16_v1",
        "dense_dual_q8_1x2_rowtile8_dp4a_bf16_bf16_out",
    )
    rowtile_key = KernelKey(
        "hip_gfx1151",
        "linear_pair_silu",
        "gguf_q4_k_qmicro_t16_v1",
        "dense_dual_rowtile_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (candidate_key, rowtile_key)
    }
    quantize_calls: list[tuple[tuple, dict]] = []
    candidate_calls: list[tuple[tuple, dict]] = []
    rowtile_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_quantize_bf16_q8_1x2",
        lambda *args, **kwargs: quantize_calls.append((args, kwargs)),
    )
    register(
        candidate_key,
        lambda *args, **kwargs: candidate_calls.append((args, kwargs)),
        replace=True,
    )
    register(
        rowtile_key,
        lambda *args, **kwargs: rowtile_calls.append((args, kwargs)),
        replace=True,
    )
    try:
        with native_batch_decode_session(True):
            assert launch_gguf_linear_pair_silu(
                weight_a,
                weight_b,
                x_ptr=100,
                out_ptr=400,
                rows=4,
                in_features=5_120,
                out_features=17_408,
                backend="hip_gfx1151",
                stream=7,
                runtime="runtime-sentinel",
                use_gemv_decode=True,
                registered_decode_variant=candidate_key.variant,
                q8_1_workspace_ptr=900,
            )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)

    assert quantize_calls == [
        (
            (100, 900, 4, 5_120),
            {
                "stream": 7,
                "library": None,
                "runtime": "runtime-sentinel",
            },
        )
    ]
    assert candidate_calls == [
        (
            (900, 14, 14, 400, 4, 5_120, 17_408),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]
    assert rowtile_calls == []


def test_gfx1151_q4_t16_gate_up_rowtile8_chunks_rows_65_plus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate/up dual rowtile8 covers the full decode regime (rows 65..511).

    The policy admits rows up to 511, so c=70 decomposes into eight 8-row
    groups plus a 6-row tail via ``_rowtile8_row_chunks``; each group shares
    one q8_1 quantization and one fused gate/up+SiLU launch. No gate/up
    concurrency below 512 silently falls to WMMA.
    """

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight_a = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_QMICRO_T16,
        quant_key="gguf_q4_k_qmicro_t16_v1",
    )
    weight_b = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_QMICRO_T16,
        quant_key="gguf_q4_k_qmicro_t16_v1",
    )
    fused_key = KernelKey(
        "hip_gfx1151",
        "linear_pair_silu",
        "gguf_q4_k_qmicro_t16_v1",
        "dense_dual_q8_1x2_rowtile8_dp4a_bf16_bf16_out",
    )
    original = resolve(
        backend=fused_key.backend,
        layer=fused_key.layer,
        quant=fused_key.quant,
        variant=fused_key.variant,
    )
    quantize_calls: list[tuple[tuple, dict]] = []
    fused_calls: list[tuple[tuple, dict]] = []

    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_quantize_bf16_q8_1x2",
        lambda *args, **kwargs: quantize_calls.append((args, kwargs)),
    )
    register(
        fused_key,
        lambda *args, **kwargs: fused_calls.append((args, kwargs)),
        replace=True,
    )
    try:
        with native_batch_decode_session(True):
            assert launch_gguf_linear_pair_silu(
                weight_a,
                weight_b,
                x_ptr=100,
                out_ptr=400,
                rows=70,
                in_features=5_120,
                out_features=17_408,
                backend="hip_gfx1151",
                stream=7,
                runtime="runtime-sentinel",
                use_gemv_decode=True,
                registered_decode_variant=fused_key.variant,
                q8_1_workspace_ptr=900,
            )
    finally:
        register(fused_key, original, replace=True)

    groups = [(8, 0), (8, 8), (8, 16), (8, 24), (8, 32), (8, 40), (8, 48), (8, 56), (6, 64)]
    assert quantize_calls == [
        (
            (100 + row_base * 5_120 * 2, 900, chunk_rows, 5_120),
            {"stream": 7, "library": None, "runtime": "runtime-sentinel"},
        )
        for chunk_rows, row_base in groups
    ]
    assert fused_calls == [
        (
            (900, 14, 14, 400 + row_base * 17_408 * 2, chunk_rows, 5_120, 17_408),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
        for chunk_rows, row_base in groups
    ]


def test_gfx1151_q4_t16_split_weight_keeps_native_b1_on_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native B1-B3 keeps the exact non-regressive Q8_1x2 control owner."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight_a = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    weight_b = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    control_key = KernelKey(
        "hip_gfx1151",
        "linear_pair_silu",
        "gguf_q4_k_t16_v1",
        "dense_dual_q8_1x2_dp4a_bf16_bf16_out",
    )
    candidate_key = KernelKey(
        "hip_gfx1151",
        "linear_pair_silu",
        "gguf_q4_k_t16_v1",
        "dense_dual_q8_1x2_split_weight_dp4a_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (control_key, candidate_key)
    }
    quantize_calls: list[tuple[tuple, dict]] = []
    control_calls: list[tuple[tuple, dict]] = []
    candidate_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_quantize_bf16_q8_1x2",
        lambda *args, **kwargs: quantize_calls.append((args, kwargs)),
    )
    register(
        control_key,
        lambda *args, **kwargs: control_calls.append((args, kwargs)),
        replace=True,
    )
    register(
        candidate_key,
        lambda *args, **kwargs: candidate_calls.append((args, kwargs)),
        replace=True,
    )
    try:
        with native_batch_decode_session(True):
            assert launch_gguf_linear_pair_silu(
                weight_a,
                weight_b,
                x_ptr=100,
                out_ptr=400,
                rows=1,
                in_features=5_120,
                out_features=17_408,
                backend="hip_gfx1151",
                stream=7,
                runtime="runtime-sentinel",
                use_gemv_decode=True,
                registered_decode_variant=candidate_key.variant,
                q8_1_workspace_ptr=900,
            )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)

    assert quantize_calls == [
        (
            (100, 900, 1, 5_120),
            {
                "stream": 7,
                "library": None,
                "runtime": "runtime-sentinel",
            },
        )
    ]
    assert control_calls == [
        (
            (900, 14, 14, 400, 1, 5_120, 17_408),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]
    assert candidate_calls == []


def test_gfx1151_q4_k_decode_sidecar_pair_silu_uses_t16_owner() -> None:
    """A decode-only T16 sidecar overrides pack8 without changing its layout."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight_a = _fake_weight(
        layout=LAYOUT_Q4_K_PACK8,
        quant_key="gguf_q4_k",
        decode_tiles=True,
    )
    weight_b = _fake_weight(
        layout=LAYOUT_Q4_K_PACK8,
        quant_key="gguf_q4_k",
        decode_tiles=True,
    )
    fused_key = KernelKey(
        "hip_gfx1151",
        "linear_pair_silu",
        "gguf_q4_k",
        "t16_sidecar_dual_decode_bf16_bf16_out",
    )
    original = resolve(
        backend=fused_key.backend,
        layer=fused_key.layer,
        quant=fused_key.quant,
        variant=fused_key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def fake_fused(*args, **kwargs):
        calls.append((args, kwargs))

    register(fused_key, fake_fused, replace=True)
    try:
        assert launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            x_ptr=100,
            out_ptr=400,
            rows=1,
            in_features=3072,
            out_features=1024,
            backend="hip_gfx1151",
            stream=7,
            runtime="runtime-sentinel",
            use_gemv_decode=True,
        )
    finally:
        register(fused_key, original, replace=True)

    assert calls == [
        (
            (100, 15, 15, 400, 1, 3072, 1024),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_gfx1151_q4_k_decode_sidecar_single_uses_t16_owner() -> None:
    """The single-output helper selects exact T16 or declines cleanly."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight = _fake_weight(
        layout=LAYOUT_Q4_K_PACK8,
        quant_key="gguf_q4_k",
        decode_tiles=True,
    )
    key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k_t16_v1",
        "dense_single_local32_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def fake_single(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, fake_single, replace=True)
    try:
        assert launch_gguf_q4_t16_sidecar_decode(
            weight,
            x_ptr=100,
            out_ptr=400,
            rows=1,
            in_features=1024,
            out_features=3072,
            backend="hip_gfx1151",
            stream=7,
            runtime="runtime-sentinel",
            enabled=True,
        )
        assert not launch_gguf_q4_t16_sidecar_decode(
            weight,
            x_ptr=100,
            out_ptr=400,
            rows=1,
            in_features=1024,
            out_features=3072,
            backend="hip_gfx1151",
            enabled=False,
        )
    finally:
        register(key, original, replace=True)

    assert calls == [
        (
            (100, 15, 400, 1, 1024, 3072),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_gfx1151_q4_k_decode_sidecar_row8_uses_two_wave_owner() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight = _fake_weight(
        layout=LAYOUT_Q4_K_PACK8,
        quant_key="gguf_q4_k",
        decode_tiles=True,
    )
    key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k_t16_v1",
        "dense_rowtile16_w2_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls = []

    def fake_rowtile(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, fake_rowtile, replace=True)
    try:
        assert launch_gguf_q4_t16_sidecar_decode(
            weight,
            x_ptr=100,
            out_ptr=400,
            rows=8,
            in_features=17_408,
            out_features=5_120,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
        )
    finally:
        register(key, original, replace=True)

    assert calls == [
        (
            (100, 15, 400, 8, 17_408, 5_120),
            {"stream": 0, "runtime": "runtime-sentinel"},
        )
    ]


@pytest.mark.parametrize(
    "rows,in_features,out_features,expect_two_wave",
    [
        (2, 5_120, 12_288, True),
        (3, 5_120, 12_288, True),
        (4, 5_120, 12_288, True),
        (2, 5_120, 10_240, False),
        (3, 5_120, 10_240, True),
        (4, 5_120, 10_240, True),
        (3, 17_408, 5_120, False),
    ],
)
def test_gfx1151_q4_k_decode_smallm_two_wave_shape_policy(
    rows: int,
    in_features: int,
    out_features: int,
    expect_two_wave: bool,
) -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight = _fake_weight(
        layout=LAYOUT_Q4_K_PACK8,
        quant_key="gguf_q4_k",
        decode_tiles=True,
    )
    two_wave_key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k_t16_v1",
        "dense_rowtile16_w2_bf16_bf16_out",
    )
    parent_key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q4_k_t16_v1",
        "dense_rowtile_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (two_wave_key, parent_key)
    }
    calls: list[str] = []

    register(two_wave_key, lambda *args, **kwargs: calls.append("two_wave"), replace=True)
    register(parent_key, lambda *args, **kwargs: calls.append("parent"), replace=True)
    try:
        assert launch_gguf_q4_t16_sidecar_decode(
            weight,
            x_ptr=100,
            out_ptr=400,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            backend="hip_gfx1151",
            runtime="runtime-sentinel",
        )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)

    assert calls == ["two_wave" if expect_two_wave else "parent"]


@pytest.mark.parametrize(
    "rows,in_features,out_features,expect_two_wave",
    [
        (6, 5_120, 1_024, True),
        (6, 5_120, 6_144, True),
        (6, 5_120, 12_288, True),
        (6, 5_120, 17_408, True),
        (6, 6_144, 5_120, True),
        (4, 5_120, 12_288, False),
        (6, 17_408, 5_120, False),
    ],
)
def test_gfx1100_q4_k_decode_row6_two_wave_shape_policy(
    rows: int,
    in_features: int,
    out_features: int,
    expect_two_wave: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2", "1")
    gguf_linear_module._rowtile_variant_policy_env_cache.clear()
    weight = _fake_weight(
        layout=LAYOUT_Q4_K_PACK8,
        quant_key="gguf_q4_k",
        decode_tiles=True,
    )
    two_wave_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "dense_rowtile16_w2_bf16_bf16_out",
    )
    parent_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "dense_rowtile_bf16_bf16_out",
    )
    for key in (two_wave_key, parent_key):
        gguf_linear_module._ensure_linear_kernel_registered(key)
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (two_wave_key, parent_key)
    }
    calls: list[str] = []

    register(two_wave_key, lambda *args, **kwargs: calls.append("two_wave"), replace=True)
    register(parent_key, lambda *args, **kwargs: calls.append("parent"), replace=True)
    try:
        assert launch_gguf_q4_t16_sidecar_decode(
            weight,
            x_ptr=100,
            out_ptr=400,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            backend="hip_gfx1100",
            runtime="runtime-sentinel",
        )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)
        gguf_linear_module._rowtile_variant_policy_env_cache.clear()

    assert calls == ["two_wave" if expect_two_wave else "parent"]


def test_gfx1100_q4_k_decode_row6_two_wave_defaults_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2", raising=False)
    gguf_linear_module._rowtile_variant_policy_env_cache.clear()
    try:
        variants = gguf_linear_module._q4_t16_sidecar_decode_variants(
            rows=6,
            in_features=5_120,
            out_features=12_288,
            backend="hip_gfx1100",
        )
    finally:
        gguf_linear_module._rowtile_variant_policy_env_cache.clear()
    assert variants[0] == "dense_rowtile16_w2_bf16_bf16_out"


@pytest.mark.parametrize(
    "backend,enabled,expected_variant",
    [
        ("hip_gfx1100", True, "dense_rowtile16_w2_bf16_bf16_out"),
        ("hip_gfx1100", False, "dense_rowtile_bf16_bf16_out"),
        ("hip_gfx1151", True, "dense_rowtile_bf16_bf16_out"),
    ],
)
def test_q4_t16_row6_two_wave_canonical_scope(
    backend: str,
    enabled: bool,
    expected_variant: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2",
        "1" if enabled else "0",
    )
    monkeypatch.setattr(
        gguf_linear_module,
        "_native_batch_decode_session_enabled",
        True,
    )
    gguf_linear_module._rowtile_variant_policy_env_cache.clear()
    dispatch = gguf_linear_module.GGUFLinearDispatch(
        KernelKey(
            backend,
            "linear",
            "gguf_q4_k_t16_v1",
            "t16_gemv_decode_bf16_bf16_out",
        ),
        "t16",
    )
    for variant in (
        "dense_rowtile16_w2_bf16_bf16_out",
        "dense_rowtile_bf16_bf16_out",
    ):
        gguf_linear_module._ensure_linear_kernel_registered(
            KernelKey(backend, "linear", "gguf_q4_k_t16_v1", variant)
        )
    try:
        resolved = gguf_linear_module._q4_t16_dense_native_dispatch(
            dispatch,
            rows=6,
            in_features=5_120,
            out_features=12_288,
        )
    finally:
        gguf_linear_module._rowtile_variant_policy_env_cache.clear()
    assert resolved.key.variant == expected_variant


def test_gfx1151_q4_k_t16_shared_down_selects_native_tail_batch() -> None:
    """The production tail boundary consumes the resident T16 sidecar."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight = _fake_weight(
        layout=LAYOUT_Q4_K_PACK8,
        quant_key="gguf_q4_k",
        decode_tiles=True,
    )
    key = KernelKey(
        "hip_gfx1151",
        "linear+moe_tail+next_rmsnorm_host_batch",
        "gguf_q4_k_t16_v1",
        "dense_single_local32_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def fake_batch(*args, **kwargs):
        calls.append((args, kwargs))

    fake_batch.projection_symbol = "projection"
    fake_batch.tail_symbol = "tail"
    register(key, fake_batch, replace=True)
    libraries = {
        "gguf_q4_k_t16_v1": SimpleNamespace(projection="projection-fn"),
        "launch_batch": "batch-library",
        "moe_tail": SimpleNamespace(tail="tail-fn"),
    }
    try:
        assert launch_gguf_linear_moe_tail_host_batch(
            weight,
            x_ptr=100,
            shared_out_ptr=200,
            routed_ptr=300,
            post_attention_ptr=400,
            norm_weight_ptr=500,
            norm_out_ptr=600,
            hidden_out_ptr=700,
            rows=1,
            in_features=1024,
            out_features=3072,
            eps=1e-6,
            backend="hip_gfx1151",
            stream=7,
            libraries=libraries,
            runtime="runtime-sentinel",
            use_q4_t16_sidecar=True,
        )
    finally:
        register(key, original, replace=True)

    assert calls == [
        (
            (
                "projection-fn",
                "tail-fn",
                100,
                15,
                200,
                300,
                400,
                500,
                600,
                700,
                1,
                1024,
                3072,
            ),
            {
                "eps": 1e-6,
                "stream": 7,
                "library": "batch-library",
                "runtime": "runtime-sentinel",
            },
        )
    ]


def test_gfx1151_q4_k_dual_interleaved_sidecar_uses_paired_owner() -> None:
    """The paired sidecar is selectable without losing separate rollback."""

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    weight_a = _fake_weight(
        layout=LAYOUT_Q4_K_PACK8,
        quant_key="gguf_q4_k",
        decode_tiles=True,
        decode_tiles_dual=True,
    )
    weight_b = _fake_weight(
        layout=LAYOUT_Q4_K_PACK8,
        quant_key="gguf_q4_k",
        decode_tiles=True,
    )
    fused_key = KernelKey(
        "hip_gfx1151",
        "linear_pair_silu",
        "gguf_q4_k",
        "t16_dual_interleaved_sidecar_decode_bf16_bf16_out",
    )
    original = resolve(
        backend=fused_key.backend,
        layer=fused_key.layer,
        quant=fused_key.quant,
        variant=fused_key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def fake_fused(*args, **kwargs):
        calls.append((args, kwargs))

    register(fused_key, fake_fused, replace=True)
    try:
        assert launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            x_ptr=100,
            out_ptr=400,
            rows=1,
            in_features=3072,
            out_features=1024,
            backend="hip_gfx1151",
            stream=7,
            runtime="runtime-sentinel",
            use_gemv_decode=True,
        )
    finally:
        register(fused_key, original, replace=True)

    assert calls == [
        (
            (100, 16, 400, 1, 3072, 1024),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_registered_q5_decode_pair_is_exact_scope_and_falls_back() -> None:
    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k")
    pair_key = KernelKey(
        "hip_gfx1100",
        "linear_pair",
        "gguf_q5_k",
        "pack8_gemv_decode_bf16_bf16_out",
    )
    original = resolve(
        backend=pair_key.backend,
        layer=pair_key.layer,
        quant=pair_key.quant,
        variant=pair_key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def fake_pair(*args, **kwargs):
        calls.append((args, kwargs))

    register(pair_key, fake_pair, replace=True)
    try:
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=3072,
            out_features=1024,
            use_wmma_prefill=False,
            use_gemv_decode=True,
            registered_decode_only=True,
        )
        assert not launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=2,
            in_features=3072,
            out_features=1024,
            use_wmma_prefill=False,
            use_gemv_decode=True,
            registered_decode_only=True,
        )
        assert not launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=3072,
            out_features=1024,
            out_features_b=2048,
            use_wmma_prefill=False,
            use_gemv_decode=True,
            registered_decode_only=True,
        )
        assert not launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=3072,
            out_features=1024,
            use_wmma_prefill=False,
            use_gemv_decode=False,
            registered_decode_only=True,
        )
    finally:
        register(pair_key, original, replace=True)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (100, 10, 10, 200, 300, 1, 3072, 1024)
    assert kwargs["stream"] == 0


def test_registered_q5_f32_decode_pair_accepts_unequal_widths_only_at_c1() -> None:
    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k")
    pair_key = KernelKey(
        "hip_gfx1100",
        "linear_pair",
        "gguf_q5_k",
        "pack8_gemv_decode_bf16_f32_out",
    )
    original = resolve(
        backend=pair_key.backend,
        layer=pair_key.layer,
        quant=pair_key.quant,
        variant=pair_key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def fake_pair(*args, **kwargs):
        calls.append((args, kwargs))

    register(pair_key, fake_pair, replace=True)
    try:
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=3072,
            out_features=9216,
            out_features_b=72,
            output_dtype=GGUF_OUTPUT_F32,
            use_wmma_prefill=False,
            use_gemv_decode=True,
            registered_decode_only=True,
        )
        assert not launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=2,
            in_features=3072,
            out_features=9216,
            out_features_b=72,
            output_dtype=GGUF_OUTPUT_F32,
            use_wmma_prefill=False,
            use_gemv_decode=True,
            registered_decode_only=True,
        )
    finally:
        register(pair_key, original, replace=True)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (100, 10, 10, 200, 300, 1, 3072, 9216, 72)
    assert kwargs["stream"] == 0


def test_registered_q5_wave32x2_singleton_is_explicit_and_fails_closed() -> None:
    weight = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k")
    candidate_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q5_k",
        "wave32x2_gemv_decode_bf16_bf16_out",
    )
    fallback_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q5_k",
        "pack8_gemv_decode_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (candidate_key, fallback_key)
    }
    calls: list[str] = []

    register(candidate_key, lambda *args, **kwargs: calls.append("candidate"), replace=True)
    register(fallback_key, lambda *args, **kwargs: calls.append("fallback"), replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=1,
            in_features=6144,
            out_features=3072,
            use_gemv_decode=True,
            registered_variant=candidate_key.variant,
        )
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=1,
            in_features=6144,
            out_features=3072,
            use_gemv_decode=True,
            registered_variant="missing_wave32x2_variant",
        )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)

    assert calls == ["candidate", "fallback"]


def test_registered_q4_lm_head_local32_is_explicit_and_fails_closed() -> None:
    weight = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    candidate_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "local32_fixed_meta_gemv_decode_bf16_f32_out",
    )
    fallback_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k",
        "pack8_gemv_decode_bf16_f32_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (candidate_key, fallback_key)
    }
    calls: list[str] = []

    register(candidate_key, lambda *args, **kwargs: calls.append("candidate"), replace=True)
    register(fallback_key, lambda *args, **kwargs: calls.append("fallback"), replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=1,
            in_features=3072,
            out_features=100_352,
            output_dtype=GGUF_OUTPUT_F32,
            use_gemv_decode=True,
            registered_variant=candidate_key.variant,
        )
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=1,
            in_features=3072,
            out_features=100_352,
            output_dtype=GGUF_OUTPUT_F32,
            use_gemv_decode=True,
            registered_variant="missing_local32_fixed_meta_variant",
        )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)

    assert calls == ["candidate", "fallback"]


def test_registered_q5_wave32x2_pair_is_c1_only_with_pack8_fallback() -> None:
    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k")
    candidate_key = KernelKey(
        "hip_gfx1100",
        "linear_pair",
        "gguf_q5_k",
        "wave32x2_gemv_decode_bf16_f32_out",
    )
    fallback_key = KernelKey(
        "hip_gfx1100",
        "linear_pair",
        "gguf_q5_k",
        "pack8_gemv_decode_bf16_f32_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (candidate_key, fallback_key)
    }
    calls: list[tuple[str, tuple]] = []

    def candidate(*args, **kwargs):
        calls.append(("candidate", args))

    def fallback(*args, **kwargs):
        calls.append(("fallback", args))

    register(candidate_key, candidate, replace=True)
    register(fallback_key, fallback, replace=True)
    try:
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=3072,
            out_features=9216,
            out_features_b=72,
            output_dtype=GGUF_OUTPUT_F32,
            use_gemv_decode=True,
            registered_decode_only=True,
            registered_decode_variant=candidate_key.variant,
        )
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=3072,
            out_features=9216,
            out_features_b=72,
            output_dtype=GGUF_OUTPUT_F32,
            use_gemv_decode=True,
            registered_decode_only=True,
            registered_decode_variant="missing_wave32x2_variant",
        )
        assert not launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=2,
            in_features=3072,
            out_features=9216,
            out_features_b=72,
            output_dtype=GGUF_OUTPUT_F32,
            use_gemv_decode=True,
            registered_decode_only=True,
            registered_decode_variant=candidate_key.variant,
        )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)

    assert [name for name, _ in calls] == ["candidate", "fallback"]
    assert calls[0][1] == (100, 10, 10, 200, 300, 1, 3072, 9216, 72)


def test_registered_q5_fixed_meta_bf16_pair_is_c1_only_with_pack8_fallback() -> None:
    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k")
    candidate_key = KernelKey(
        "hip_gfx1100",
        "linear_pair",
        "gguf_q5_k",
        "wave32x2_fixed_meta_gemv_decode_bf16_bf16_out",
    )
    fallback_key = KernelKey(
        "hip_gfx1100",
        "linear_pair",
        "gguf_q5_k",
        "pack8_gemv_decode_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (candidate_key, fallback_key)
    }
    calls: list[tuple[str, tuple]] = []

    def candidate(*args, **kwargs):
        calls.append(("candidate", args))

    def fallback(*args, **kwargs):
        calls.append(("fallback", args))

    register(candidate_key, candidate, replace=True)
    register(fallback_key, fallback, replace=True)
    try:
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=3072,
            out_features=1024,
            use_gemv_decode=True,
            registered_decode_only=True,
            registered_decode_variant=candidate_key.variant,
        )
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=3072,
            out_features=1024,
            use_gemv_decode=True,
            registered_decode_only=True,
            registered_decode_variant="missing_wave32x2_variant",
        )
        assert not launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=2,
            in_features=3072,
            out_features=1024,
            use_gemv_decode=True,
            registered_decode_only=True,
            registered_decode_variant=candidate_key.variant,
        )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)

    assert [name for name, _ in calls] == ["candidate", "fallback"]
    assert calls[0][1] == (100, 10, 10, 200, 300, 1, 3072, 1024, 1024)


def test_registered_q6_f32_decode_pair_is_c1_only_and_fail_closed() -> None:
    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k")
    pair_key = KernelKey(
        "hip_gfx1100",
        "linear_pair",
        "gguf_q6_k",
        "pack8_gemv_decode_bf16_f32_out",
    )
    original = resolve(
        backend=pair_key.backend,
        layer=pair_key.layer,
        quant=pair_key.quant,
        variant=pair_key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def fake_pair(*args, **kwargs):
        calls.append((args, kwargs))

    register(pair_key, fake_pair, replace=True)
    try:
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=3072,
            out_features=1024,
            output_dtype=GGUF_OUTPUT_F32,
            use_wmma_prefill=False,
            use_gemv_decode=True,
            registered_decode_only=True,
        )
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=101,
            out_a_ptr=201,
            out_b_ptr=301,
            rows=1,
            in_features=3072,
            out_features=9216,
            out_features_b=72,
            output_dtype=GGUF_OUTPUT_F32,
            use_wmma_prefill=False,
            use_gemv_decode=True,
            registered_decode_only=True,
        )
        assert not launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=2,
            in_features=3072,
            out_features=1024,
            output_dtype=GGUF_OUTPUT_F32,
            use_wmma_prefill=False,
            use_gemv_decode=True,
            registered_decode_only=True,
        )
        assert not launch_gguf_linear_pair(
            weight_a,
            _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k"),
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=3072,
            out_features=1024,
            output_dtype=GGUF_OUTPUT_F32,
            use_wmma_prefill=False,
            use_gemv_decode=True,
            registered_decode_only=True,
        )
    finally:
        register(pair_key, original, replace=True)

    assert len(calls) == 2
    args, kwargs = calls[0]
    assert args == (100, 10, 10, 200, 300, 1, 3072, 1024, 1024)
    assert kwargs["stream"] == 0
    args, kwargs = calls[1]
    assert args == (101, 10, 10, 201, 301, 1, 3072, 9216, 72)
    assert kwargs["stream"] == 0


def test_physical_q4_pair_chunks_rows6_and_preserves_unfused_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.core.specdec2_scope import (
        q4_t16_physical_extra_rowtiles_session,
    )

    weight_a = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    weight_b = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    calls: list[tuple[object, int, int, int, int, int]] = []

    monkeypatch.setattr(
        gguf_linear_module,
        "backend_package_capability",
        lambda backend, name, default: (
            (6,) if name == "GGUF_SPECDEC2_TARGET_VERIFY_PAD_ROW_COUNTS" else default
        ),
    )
    monkeypatch.setattr(
        gguf_linear_module,
        "_resolve_gguf_linear_pair_kind",
        lambda *args, **kwargs: None,
    )

    def fake_single(weight, x_ptr, out_ptr, rows, in_features, out_features, **kwargs):
        calls.append((weight, x_ptr, out_ptr, rows, in_features, out_features))

    monkeypatch.setattr(gguf_linear_module, "launch_gguf_linear", fake_single)

    with q4_t16_physical_extra_rowtiles_session(True):
        assert launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=12,
            in_features=10,
            out_features=20,
            backend="hip_gfx1100",
        )

    assert calls == [
        (weight_a, 100, 200, 6, 10, 20),
        (weight_b, 100, 300, 6, 10, 20),
        (weight_a, 220, 440, 6, 10, 20),
        (weight_b, 220, 540, 6, 10, 20),
    ]


def test_w7900_q4_k_t16_ffn_pair_silu_scope_pins_the_rows33_floor() -> None:
    """The dense Q4T16 bulk FFN fused SiLU owner stays admitted from 33 rows.

    The floor moved 512 -> 33 on 2026-08-30: a same-process gate ladder measured
    +4.21%/+4.21%/+4.89% prefill at 45/96/192 rows with bit-identical fused/unfused
    outputs, and rows<33 keeps the strict unfused pair + SiLU chain. Nothing
    asserted this scope before, so a dispatch change could send 45-row prefill back
    to the unfused chain and only show up as wall time -- the same failure shape as
    the packed-wiring regression that surfaced as a 3.5x slowdown with green tests.
    """

    weight_a = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    weight_b = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k_t16_v1",
    )
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k_t16_v1",
        "dense_dual_wmma_prefill_bf16_bf16_out",
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple] = []
    register(
        key,
        lambda *args, **kwargs: calls.append(args),
        replace=True,
    )

    def launch(
        weight_a,
        weight_b,
        *,
        rows: int,
        out_features: int = 17_408,
        backend: str = "hip_gfx1100",
    ):
        return launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            x_ptr=0x1000,
            out_ptr=0x2000,
            rows=rows,
            in_features=5_120,
            out_features=out_features,
            backend=backend,
            runtime="runtime-sentinel",
        )

    try:
        with wmma_prefill_session(True):
            for rows in (33, 45, 192, 512):
                before = len(calls)
                assert launch(weight_a, weight_b, rows=rows), rows
                assert len(calls) == before + 1, rows
            # 32 is the last row count that keeps the strict unfused pair + SiLU chain
            assert not launch(weight_a, weight_b, rows=32)
            # the owner is shape-qualified: a near-miss FFN shape stays unfused
            assert not launch(weight_a, weight_b, rows=512, out_features=17_344)
            # Measured 2026-08-30: with wmma_prefill_session(False) the owner is still
            # selected at rows 512, so this t16 owner has no per-request opt-out (the
            # q4_pack8_dual_wmma_silu_prefill_session and the
            # HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL kill-switch gate the pack8
            # owner). Rows and shape are therefore the only scope controls, which is
            # exactly why this test pins both.
    finally:
        register(key, original, replace=True)

    assert len(calls) == 4
