"""WPF-H7S exact raw-Q6 packed-activation cross-row-reuse RED."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.quant import (
    gguf_q5_k_f32_rocblas_prefill as q5_f32,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch
from tests.test_gguf_q5_k_f32_rocblas_prefill import (
    _bf16_bits,
    _bf16_to_f32,
    _device,
    _edge_q6_weight,
)
from tests.test_laguna_h6e_q6_activation_tile_k_row import (
    _Q5_PRODUCTION_POLICY,
    _Q6_PRODUCTION_POLICY,
    _activation_plane_shape,
    _expected_activation_plane,
    _hip_available,
    _sampled_cpu_gate,
)
from tests.test_laguna_h6u_q6_dpp_wave_reduction import (
    _H6U_KERNEL,
    _H6U_REDUCE_HELPER,
    _h6u_keys,
    _h6u_names,
)

_ROOT = Path(__file__).resolve().parents[1]
_TARGET_ARTIFACT = (
    _ROOT
    / "benchmarks/results/"
    "2026-08-02-gfx1100-laguna-q2-xl-post-h7r-matched-raw-q6-"
    "cross-row-reuse-target.json"
)
_TARGET_ARTIFACT_SHA256 = (
    "eee146c11ffe80ab93eefbce0042b53d2875c34cc249fdbc3dc4741b27c15f62"
)
# role, output dtype, exact K, exact N, production call weight, H6U row batch
# The three roles and the c2r32 source form are inseparable.
_ROLES = (
    ("bf16-k3072-n1024", "bf16", 3_072, 1_024, 2, 5),
    ("bf16-k1024-n3072", "bf16", 1_024, 3_072, 46, 4),
    ("f32-k3072-n1024", "f32", 3_072, 1_024, 94, 5),
)
_ROWS = 512
_H7S_COL_TILE = 2
_H7S_ROW_BATCH = 32
_H7S_KERNEL = "gguf_q6_k_raw_q6_c2r32_activation_tile_kernel"
_H7S_LAUNCHER = "launch_q6_k_raw_q6_c2r32_activation_tile"
_H7S_PRIMITIVE_VARIANT = "raw_q6_c2r32_activation_tile_bf16_{output_dtype}_out"
_H7S_COMPOSITE_VARIANT = "c2r32_packed_activation_direct_bf16_{output_dtype}_out"
_H7S_PRIMITIVE_NAME = "gguf_q6_k_" + _H7S_PRIMITIVE_VARIANT
_H7S_COMPOSITE_NAME = "gguf_q6_k_" + _H7S_COMPOSITE_VARIANT
_H7S_PRIMITIVE_SYMBOL = "hipengine_gguf_q6_k_" + _H7S_PRIMITIVE_VARIANT
_H7S_PACK_NAME = (
    "gguf_bf16_activation_pack_tile_k_row_coltile2_rowbatch32_"
    "bf16_{output_dtype}_out"
)
_H7S_PACK_SYMBOL = "hipengine_" + _H7S_PACK_NAME
_H7S_GFX1151_EXCLUSION = "c2r32_packed_activation_direct"
_H6U_PRIMITIVE_WRAPPER_SHA256 = (
    "ac756f99aff7492b52ca5eb386c1e5eb9e1d30829b4ada71f307ba065d79e364"
)
_H6U_COMPOSITE_WRAPPER_SHA256 = (
    "d799437ebc6f0fc43c21c251d0e342440fce492b1722ced9965ec38701ae8196"
)
_FROZEN_UNCHANGED_FILES = {
    "hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip": (
        "377f3f966968350f63956516faa539e616ac031d362b6e6312d8b35572362336"
    ),
    "hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.py": (
        "9943de24fb179770480921811ebc83456d4ac42a3e584e1e280bfc72bb6a808d"
    ),
    "hipengine/kernels/hip_gfx1100/__init__.py": (
        "9cf784e41d9f77983373f60c543342cfb7659a736ce84cdc762bb2bc93ab6abf"
    ),
    "hipengine/runtime/gguf_linear.py": (
        "85e7faa7d5129a40cf5afcedfa538e8733b263d3c058f676a9aae9f9736fe8d1"
    ),
}
_H6U_DECLARATION_SHA256 = {
    "pack": "ec2806349b837140ab3b2b90435975741b55d86faf323e11cc55319c6ae6a714",
    "consumer": "f4c70eb4b9b78b785abfa69a897cb7949124ba799d4e1846c3fbba9de06b1df4",
    "reduce": "b71f4a713ffed5f1702b04fc383587892baee35fac324577b2654f26469d4555",
    "q6_exact_value": (
        "7465a1f8252ebd34483f8148f40797b89f8900209425d4cf8892beec9e9ed332"
    ),
}
_PHYSICAL_CONTRACT = {
    "consumer_activation_global_load_b128_sites": 4,
    "consumer_barriers": 1,
    "consumer_code_bytes_max": 14_000,
    "consumer_dpp_add_operations": 256,
    "consumer_instruction_slots_max": 2_400,
    "consumer_lds_bytes": 1_024,
    "consumer_local_size": 128,
    "consumer_metadata_sgpr_max": 96,
    "consumer_metadata_vgpr_max": 136,
    "consumer_ordered_fma_operations": 64,
    "consumer_permlanex16_operations": 64,
    "consumer_private_spill_scratch_bytes": 0,
    "consumer_raw_q6_field_load_sites": 8,
    "consumer_total_global_load_sites": 12,
    "consumer_wavefront_size": 32,
    "pack_local_size": 256,
    "pack_private_spill_scratch_bytes": 0,
}
_TRACE_CONTRACT = {
    "ordered_kernels": (
        "gguf_bf16_activation_pack_tile_k_row_kernel<32>",
        _H7S_KERNEL,
    ),
    "forbidden_kernel": "gguf_q6_k_dequantize_f32_exact_kernel",
    "require_cached_build": True,
    "new_compiler_processes": 0,
    "positive_duration": True,
    "consumer_local_size": 128,
    "pack_local_size": 256,
    "consumer_runtime_scratch_bytes": 0,
}
_TIMING_CONTRACT = {
    "rows": 512,
    "warmups": 5,
    "counter_rotated_repetitions": 15,
    "launches_per_sample": 5,
    "production_invocations": 142,
    "call_weights": (2, 46, 94),
    "clocks": ("hip_event", "synchronized_wall"),
    "require_every_role_both_clocks": True,
    "require_weighted_aggregate_both_clocks": True,
    "one_shot_only": True,
}
_REJECT_RULE = (
    "Any correctness, physical, cached-trace, compiler, lifecycle, per-role "
    "both-clock, or aggregate both-clock miss removes every H7S surface "
    "without role/c4r16/c8r8/prompt subset, tuning, recompile, or favorable "
    "rerun."
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode())


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _declaration(source: str, anchor: str) -> str:
    start = source.index(anchor)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated declaration: {anchor}")


def _candidate_names(output_dtype: str) -> tuple[str, str, str]:
    return (
        _H7S_PACK_NAME.format(output_dtype=output_dtype),
        _H7S_PRIMITIVE_NAME.format(output_dtype=output_dtype),
        _H7S_COMPOSITE_NAME.format(output_dtype=output_dtype),
    )


def _candidate_keys(
    output_dtype: str,
    *,
    backend: str = "hip_gfx1100",
) -> tuple[KernelKey, KernelKey]:
    return (
        KernelKey(
            backend,
            "linear",
            "gguf_q6_k",
            _H7S_PRIMITIVE_VARIANT.format(output_dtype=output_dtype),
        ),
        KernelKey(
            backend,
            "linear",
            "gguf_q6_k",
            _H7S_COMPOSITE_VARIANT.format(output_dtype=output_dtype),
        ),
    )


def _candidate(output_dtype: str):
    _, primitive_name, composite_name = _candidate_names(output_dtype)
    return getattr(q5_f32, primitive_name), getattr(q5_f32, composite_name)


def _allocation_lifecycle() -> tuple[int, int]:
    stats = memory_stats()
    return stats["current_allocated_bytes"], stats["active_allocations"]


@pytest.fixture(scope="module")
def library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    return q5_f32.build_gguf_q5_k_f32_rocblas_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )


@pytest.fixture(scope="module")
def qweights() -> dict[tuple[int, int], np.ndarray]:
    return {
        (in_features, out_features): _edge_q6_weight(
            out_features,
            in_features,
        )
        for _, _, in_features, out_features, _, _ in _ROLES
    }


def test_h7s_frozen_target_physical_trace_timing_and_rejection_contract() -> None:
    artifact_bytes = _TARGET_ARTIFACT.read_bytes()
    assert _sha256_bytes(artifact_bytes) == _TARGET_ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["status"] == (
        "accepted_matched_production_rerank_and_exact_h7s_target"
    )
    assert artifact["target"]["id"] == "WPF-H7S"
    assert artifact["target"]["implementation_absent"] is True
    assert artifact["decision"]["candidate_implemented"] is False
    assert artifact["decision"]["production_changed"] is False
    assert artifact["decision"]["performance_claim"] == "target_selection_only"
    assert artifact["production"]["wall_tok_s"] == pytest.approx(
        431.31016450993457
    )
    assert artifact["production"]["matched_llamacpp_hip_tok_s"] == 690.791
    assert artifact["production"]["kernel_sum_ms"] == pytest.approx(
        1_172.2412389999988
    )

    roles = artifact["h6u_control"]["roles"]
    assert tuple(role["production_invocations"] for role in roles) == (2, 46, 94)
    assert sum(role["production_invocations"] for role in roles) == 142
    assert artifact["h6u_control"]["weighted_event_ms"] == pytest.approx(
        48.26689233779907
    )
    assert artifact["h6u_control"]["weighted_wall_ms"] == pytest.approx(
        48.519960790872574
    )
    assert artifact["h7n_control"]["q6_column_decodes_per_k_iteration"] == 16
    assert artifact["h7n_control"]["static_global_load_sites"] == 68
    source_model = artifact["target"]["source_operation_model"]
    assert source_model["q6_column_decodes_per_k_iteration"] == 2
    assert source_model["q6_field_load_sites_per_k_iteration"] == 8
    assert source_model["aligned_activation_b128_load_sites_per_k_iteration"] == 4
    assert source_model["total_global_load_sites_per_k_iteration"] == 12
    assert source_model["removed_producer_launches_per_request"] == 142
    assert source_model["request_dispatches_if_selected_model"] == 2_050
    assert source_model["weighted_input_bytes_ratio_vs_current"] == pytest.approx(
        0.9371989347476998
    )
    assert source_model["weighted_input_bytes_ratio_vs_h7n"] == pytest.approx(
        3.1072485207100593
    )

    assert artifact["target"]["physical_gate"] == _PHYSICAL_CONTRACT
    assert _TRACE_CONTRACT["ordered_kernels"][1] == _H7S_KERNEL
    assert _TRACE_CONTRACT["new_compiler_processes"] == 0
    assert _TRACE_CONTRACT["forbidden_kernel"] not in _TRACE_CONTRACT[
        "ordered_kernels"
    ]
    assert _TIMING_CONTRACT["call_weights"] == tuple(
        role[4] for role in _ROLES
    )
    assert sum(_TIMING_CONTRACT["call_weights"]) == 142
    assert _TIMING_CONTRACT["require_every_role_both_clocks"] is True
    assert _TIMING_CONTRACT["require_weighted_aggregate_both_clocks"] is True
    assert _TIMING_CONTRACT["one_shot_only"] is True
    assert "every H7S surface" in _REJECT_RULE
    assert "role/c4r16/c8r8/prompt subset" in _REJECT_RULE
    assert all(
        term not in _H7S_COMPOSITE_VARIANT
        for term in ("prompt", "token", "layer")
    )


def test_h7s_source_registry_workspace_and_backend_isolation() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    artifact = json.loads(_TARGET_ARTIFACT.read_text())
    for relative, digest in _FROZEN_UNCHANGED_FILES.items():
        assert _sha256_file(_ROOT / relative) == digest
        assert artifact["repo"]["implementation_sha256"][relative] == digest

    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert q5_f32._Q6_DPP_WAVE_REDUCTION_ROLES == tuple(
        (16, row_batch, output_dtype, in_features, out_features)
        for _, output_dtype, in_features, out_features, _, row_batch in _ROLES
    )
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_H5Y_POLICY == (
        _Q5_PRODUCTION_POLICY
    )
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == (
        _Q6_PRODUCTION_POLICY
    )
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_H6U_POLICY == (
        _Q6_PRODUCTION_POLICY
    )
    assert hip_gfx1151.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == {}
    assert LagunaQ5F32OrderedScratch.weight_f32_planned_nbytes() == 150_994_944
    assert LagunaQ5F32OrderedScratch.activation_bf16_planned_nbytes(
        max_rows=512
    ) == 10_125_312
    assert q5_f32.q5_k_f32_activation_tile_k_row_nbytes(
        512, 3_072, 32
    ) == 3_145_728
    assert q5_f32.q5_k_f32_activation_tile_k_row_nbytes(
        512, 1_024, 32
    ) == 1_048_576
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256

    source = Path(q5_f32.__file__).with_suffix(".hip").read_text()
    declarations = {
        "pack": _declaration(
            source,
            "gguf_bf16_activation_pack_tile_k_row_kernel",
        ),
        "consumer": _declaration(source, _H6U_KERNEL),
        "reduce": _declaration(source, _H6U_REDUCE_HELPER),
        "q6_exact_value": _declaration(source, "q6_k_exact_value"),
    }
    assert {
        name: _sha256_text(body) for name, body in declarations.items()
    } == _H6U_DECLARATION_SHA256
    for row_batch, output_dtype in ((4, "bf16"), (5, "bf16"), (5, "f32")):
        h6u_primitive_name, h6u_composite_name = _h6u_names(
            16,
            row_batch,
            output_dtype,
        )
        assert _sha256_text(
            inspect.getsource(getattr(q5_f32, h6u_primitive_name))
        ) == _H6U_PRIMITIVE_WRAPPER_SHA256
        assert _sha256_text(
            inspect.getsource(getattr(q5_f32, h6u_composite_name))
        ) == _H6U_COMPOSITE_WRAPPER_SHA256

    for row_batch, output_dtype in ((4, "bf16"), (5, "bf16"), (5, "f32")):
        for key in _h6u_keys(16, row_batch, output_dtype):
            assert is_registered(key)
            assert not is_registered(
                KernelKey(
                    "hip_gfx1151",
                    key.layer,
                    key.quant,
                    key.variant,
                )
            )

    # Intentional RED only after H6U bytes, policies, allocation, registry,
    # runtime/source isolation, and current gfx1151 controls are proven.
    for output_dtype in ("bf16", "f32"):
        pack_name, primitive_name, composite_name = _candidate_names(output_dtype)
        primitive, composite = _candidate(output_dtype)
        pack = getattr(q5_f32, pack_name)
        assert primitive.__name__ == primitive_name
        assert composite.__name__ == composite_name
        assert primitive is not composite
        for key, function in zip(
            _candidate_keys(output_dtype),
            (primitive, composite),
            strict=True,
        ):
            assert resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            ) is function
            assert not is_registered(
                KernelKey(
                    "hip_gfx1151",
                    key.layer,
                    key.quant,
                    key.variant,
                )
            )
        assert pack.__name__ == pack_name
        assert source.count(
            _H7S_PRIMITIVE_SYMBOL.format(output_dtype=output_dtype)
        ) == 1
        assert source.count(
            _H7S_PACK_SYMBOL.format(output_dtype=output_dtype)
        ) == 1

    assert source.count(f"__global__ void {_H7S_KERNEL}(") == 1
    assert source.count(_H7S_LAUNCHER) >= 3
    candidate_body = _declaration(source, _H7S_KERNEL)
    assert "constexpr int ROW_BATCH = 32;" in candidate_body
    assert "constexpr int COL_TILE = 2;" in candidate_body
    assert "float acc[ROW_BATCH][COL_TILE] = {};" in candidate_body
    assert "q6_k_exact_value" in candidate_body
    assert "h6u_reduce_wave_accumulators_dpp" in candidate_body
    assert candidate_body.count("__syncthreads();") == 1
    gfx1151_source = inspect.getsource(hip_gfx1151)
    assert gfx1151_source.count(_H7S_GFX1151_EXCLUSION) == 1


@pytest.mark.parametrize(
    ("role", "output_dtype", "in_features", "out_features", "calls", "h6u_row_batch"),
    _ROLES,
    ids=tuple(role[0] for role in _ROLES),
)
def test_h7s_strict_m512_role_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    output_dtype: str,
    in_features: int,
    out_features: int,
    calls: int,
    h6u_row_batch: int,
) -> None:
    del role, calls
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("valid H7S role reached HIP loader")

    monkeypatch.setattr(
        q5_f32,
        "build_gguf_q5_k_f32_rocblas_prefill",
        fail_if_loaded,
    )
    _, h6u_composite_name = _h6u_names(
        16,
        h6u_row_batch,
        output_dtype,
    )
    control = getattr(q5_f32, h6u_composite_name)
    invalid_control = (
        (0, in_features, out_features, "rows must be positive"),
        (17, in_features - 256, out_features, f"exactly {in_features}"),
        (17, in_features, out_features - 16, f"exactly {out_features}"),
    )
    for rows, hidden, outputs, message in invalid_control:
        with pytest.raises(ValueError, match=message):
            control(1, 2, 3, 4, 5, rows, hidden, outputs)
    assert load_attempts == 0

    # Intentional RED only after retained H6U preflight rejects before loading.
    _, candidate = _candidate(output_dtype)
    for rows in (511, 513):
        with pytest.raises(ValueError, match="rows must be exactly 512"):
            candidate(1, 2, 3, 4, 5, rows, in_features, out_features)
    if output_dtype == "bf16":
        invalid_candidate = (
            (512, 768, out_features, "exactly 1024 or 3072"),
            (512, in_features, out_features - 16, "unsupported exact H7S role"),
        )
    else:
        invalid_candidate = (
            (512, in_features - 256, out_features, "exactly 3072"),
            (512, in_features, out_features - 16, "exactly 1024"),
        )
    for rows, hidden, outputs, message in invalid_candidate:
        with pytest.raises(ValueError, match=message):
            candidate(1, 2, 3, 4, 5, rows, hidden, outputs)
    with pytest.raises(ValueError, match="threads must be exactly 128"):
        candidate(
            1,
            2,
            3,
            4,
            5,
            512,
            in_features,
            out_features,
            threads=64,
        )
    assert load_attempts == 0
    with pytest.raises(AssertionError, match="valid H7S role reached HIP loader"):
        candidate(1, 2, 3, 4, 5, 512, in_features, out_features)
    assert load_attempts == 1


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("role", "output_dtype", "in_features", "out_features", "calls", "h6u_row_batch"),
    _ROLES,
    ids=tuple(role[0] for role in _ROLES),
)
def test_h7s_m512_composite_primitive_pack_cpu_poison_and_lifecycle(
    role: str,
    output_dtype: str,
    in_features: int,
    out_features: int,
    calls: int,
    h6u_row_batch: int,
    library: Any,
    qweights: dict[tuple[int, int], np.ndarray],
) -> None:
    from hipengine.core.hip import get_hip_runtime

    assert calls in (2, 46, 94)
    rng = np.random.default_rng(20260802 + 11 * in_features + out_features)
    x_bf16 = _bf16_bits(
        rng.normal(0.0, 0.2, size=(_ROWS, in_features)).astype(np.float32)
    )
    qweight = qweights[(in_features, out_features)]
    host_dtype = np.uint16 if output_dtype == "bf16" else np.float32
    expected = np.empty((_ROWS, out_features), dtype=host_dtype)
    h6u_primitive_actual = np.empty_like(expected)
    candidate_composite_actual = np.empty_like(expected)
    candidate_primitive_actual = np.empty_like(expected)
    h6u_plane_shape = _activation_plane_shape(
        _ROWS,
        in_features,
        h6u_row_batch,
    )
    h7s_plane_shape = _activation_plane_shape(
        _ROWS,
        in_features,
        _H7S_ROW_BATCH,
    )
    h6u_plane = np.empty(h6u_plane_shape, dtype=np.uint16)
    h7s_plane = np.empty(h7s_plane_shape, dtype=np.uint16)
    expected_h6u_plane = _expected_activation_plane(x_bf16, h6u_row_batch)
    expected_h7s_plane = _expected_activation_plane(x_bf16, _H7S_ROW_BATCH)

    runtime = get_hip_runtime()
    baseline = _allocation_lifecycle()
    buffers: list[Any] = []
    try:
        x_dev = _device(x_bf16, runtime)
        weight_dev = _device(qweight, runtime)
        weight_f32_dev = malloc(
            q5_f32.q6_k_f32_ordered_workspace_nbytes(
                in_features,
                out_features,
            ),
            runtime=runtime,
        )
        expected_dev = malloc(expected.nbytes, runtime=runtime)
        h6u_primitive_dev = malloc(expected.nbytes, runtime=runtime)
        candidate_composite_dev = malloc(expected.nbytes, runtime=runtime)
        candidate_primitive_dev = malloc(expected.nbytes, runtime=runtime)
        activation_dev = malloc(
            max(h6u_plane.nbytes, h7s_plane.nbytes),
            runtime=runtime,
        )
        buffers.extend(
            (
                x_dev,
                weight_dev,
                weight_f32_dev,
                expected_dev,
                h6u_primitive_dev,
                candidate_composite_dev,
                candidate_primitive_dev,
                activation_dev,
            )
        )

        h6u_primitive_name, h6u_composite_name = _h6u_names(
            16,
            h6u_row_batch,
            output_dtype,
        )
        h6u_primitive = getattr(q5_f32, h6u_primitive_name)
        h6u_composite = getattr(q5_f32, h6u_composite_name)
        runtime.memset(expected_dev.ptr, 0x5A, expected.nbytes)
        runtime.memset(activation_dev.ptr, 0xA5, h6u_plane.nbytes)
        h6u_composite(
            x_dev.ptr,
            weight_dev.ptr,
            expected_dev.ptr,
            weight_f32_dev.ptr,
            activation_dev.ptr,
            _ROWS,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(expected),
            expected_dev,
            expected.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(h6u_plane),
            activation_dev,
            h6u_plane.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(h6u_plane, expected_h6u_plane)
        _sampled_cpu_gate(
            expected,
            x_bf16,
            qweight,
            row_batch=h6u_row_batch,
            output_dtype=output_dtype,
            in_features=in_features,
            out_features=out_features,
        )
        runtime.memset(h6u_primitive_dev.ptr, 0x5A, expected.nbytes)
        h6u_primitive(
            activation_dev.ptr,
            weight_f32_dev.ptr,
            h6u_primitive_dev.ptr,
            _ROWS,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(h6u_primitive_actual),
            h6u_primitive_dev,
            h6u_primitive_actual.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(h6u_primitive_actual, expected)
        assert np.isfinite(
            _bf16_to_f32(expected)
            if output_dtype == "bf16"
            else np.asarray(expected, dtype=np.float32)
        ).all()

        # Intentional RED only after complete H6U composite/primitive bytes,
        # exact H6U pack, sampled CPU values, poison overwrite, and finiteness.
        candidate_primitive, candidate_composite = _candidate(output_dtype)
        runtime.memset(activation_dev.ptr, 0xA5, h7s_plane.nbytes)
        runtime.memset(candidate_composite_dev.ptr, 0x5A, expected.nbytes)
        candidate_composite(
            x_dev.ptr,
            weight_dev.ptr,
            candidate_composite_dev.ptr,
            weight_f32_dev.ptr,
            activation_dev.ptr,
            _ROWS,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(candidate_composite_actual),
            candidate_composite_dev,
            candidate_composite_actual.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(h7s_plane),
            activation_dev,
            h7s_plane.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(h7s_plane, expected_h7s_plane)
        np.testing.assert_array_equal(candidate_composite_actual, expected)
        assert np.isfinite(
            _bf16_to_f32(candidate_composite_actual)
            if output_dtype == "bf16"
            else np.asarray(candidate_composite_actual, dtype=np.float32)
        ).all()

        runtime.memset(candidate_primitive_dev.ptr, 0x5A, expected.nbytes)
        candidate_primitive(
            activation_dev.ptr,
            weight_dev.ptr,
            candidate_primitive_dev.ptr,
            _ROWS,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(candidate_primitive_actual),
            candidate_primitive_dev,
            candidate_primitive_actual.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(candidate_primitive_actual, expected)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    assert _allocation_lifecycle() == baseline, role
