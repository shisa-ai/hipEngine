"""WPF-H7V dequantized-Q6 full-batch/live-tail RED contract."""

from __future__ import annotations

import hashlib
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
from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.kernels.hip_gfx1100.quant import (
    gguf_q5_k_f32_rocblas_prefill as q5_f32,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch
from tests.test_gguf_q5_k_f32_rocblas_prefill import (
    _bf16_bits,
    _device,
    _edge_q6_weight,
)
from tests.test_laguna_h6e_q6_activation_tile_k_row import (
    _activation_plane_shape,
    _expected_activation_plane,
    _hip_available,
    _sampled_cpu_gate,
    _suffix,
)
from tests.test_laguna_h6u_q6_dpp_wave_reduction import (
    _H6U_ADD_HELPER,
    _H6U_KERNEL,
    _H6U_MOVE_HELPER,
    _H6U_PERMLANE_HELPER,
    _H6U_REDUCE_HELPER,
    _ROLES,
    _declaration,
    _h6u_keys,
    _h6u_names,
)

_ROOT = Path(__file__).parents[1]
_ARTIFACT = (
    _ROOT
    / "benchmarks/results/"
    "2026-08-02-gfx1100-laguna-q2-xl-post-h7u-"
    "q6-full-batch-live-tail-target.json"
)
_ARTIFACT_SHA256 = (
    "e8ca5e66ab37c27c37e56d24492dda8e0997ec4dbcf90f83d42d6aaec07548c8"
)
_H7V_KERNEL = (
    "gguf_q6_k_f32_weight_ordered_weight_major_row_major_"
    "activation_tile_k_row_dpp_wave_reduction_full_batch_kernel"
)
_H7V_LAUNCH = (
    "launch_q6_k_f32_weight_ordered_weight_major_row_major_"
    "activation_tile_k_row_dpp_wave_reduction_full_batch_live_tail"
)
_H7V_STEM = (
    "weight_major_row_major_activation_tile_k_row_"
    "dpp_wave_reduction_full_batch_live_tail"
)
_H6U_POLICY = {
    ("bf16", 3_072, 1_024): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch5"
    ),
    ("bf16", 1_024, 3_072): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch4"
    ),
    ("f32", 3_072, 72): "coltile8_rowbatch4",
    ("f32", 3_072, 1_024): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch5"
    ),
}
_CALL_WEIGHTS = {
    (16, 5, "bf16", 3_072, 1_024): 2,
    (16, 4, "bf16", 1_024, 3_072): 46,
    (16, 5, "f32", 3_072, 1_024): 94,
}
_PHYSICAL_CONTRACT = {
    (16, 4, "bf16"): {
        "fmas": 64,
        "permlanex16": 64,
        "dpp_adds": 256,
        "lds_metadata_max": 1_024,
        "code_bytes_max": 5_992,
        "slots_max": 923,
        "metadata_vgpr_max": 111,
        "runtime_vgpr_max": 112,
    },
    (16, 5, "bf16"): {
        "fmas": 80,
        "permlanex16": 80,
        "dpp_adds": 320,
        "lds_metadata_max": 1_280,
        "code_bytes_max": 7_128,
        "slots_max": 1_082,
        "metadata_vgpr_max": 139,
        "runtime_vgpr_max": 144,
    },
    (16, 5, "f32"): {
        "fmas": 80,
        "permlanex16": 80,
        "dpp_adds": 320,
        "lds_metadata_max": 1_280,
        "code_bytes_max": 7_100,
        "slots_max": 1_079,
        "metadata_vgpr_max": 139,
        "runtime_vgpr_max": 144,
    },
}
_TRACE_CONTRACT = {
    "activation_packs": 142,
    "q6_f32_producers": 143,
    "h7v_full_consumers": 142,
    "h6u_tail_consumers": 96,
    "consumer_launches": 238,
    "request_dispatches": 2_382,
    "queues": 1,
    "compiler_processes": 0,
}
_TIMING_CONTRACT = {
    "warmups": 5,
    "samples": 15,
    "launch_repeats": 5,
    "role_weights": (2, 46, 94),
    "required_clocks": ("hip_event", "synchronized_wall"),
    "require_every_role_positive": True,
    "require_weighted_aggregate_positive": True,
    "allow_subset_salvage": False,
    "allow_retune": False,
    "allow_recompile": False,
    "allow_favorable_rerun": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_names(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
) -> tuple[str, str]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = f"{_H7V_STEM}_{suffix}"
    return (
        f"gguf_q5_k_f32_weight_ordered_{stem}",
        f"gguf_q6_k_f32_ordered_{stem}",
    )


def _candidate_keys(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    *,
    backend: str = "hip_gfx1100",
) -> tuple[KernelKey, KernelKey]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = f"{_H7V_STEM}_{suffix}"
    return (
        KernelKey(backend, "linear", "f32_weight", f"ordered_{stem}"),
        KernelKey(backend, "linear", "gguf_q6_k", f"f32_ordered_{stem}"),
    )


def _candidate(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
):
    _, composite_name = _candidate_names(col_tile, row_batch, output_dtype)
    return getattr(q5_f32, composite_name)


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
def production_qweights() -> dict[tuple[int, int], np.ndarray]:
    return {
        (in_features, out_features): _edge_q6_weight(
            out_features,
            in_features,
        )
        for _, _, _, in_features, out_features, _ in _ROLES
    }


def test_h7v_target_artifact_source_arithmetic_and_admission_are_frozen() -> None:
    artifact_bytes = _ARTIFACT.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == _ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["status"] == "accepted_target_only_no_candidate_run"
    assert artifact["target"]["id"] == "WPF-H7V"
    assert artifact["production"]["wall_tok_s"] == 437.1892736544479
    assert artifact["production"]["kernel_sum_ms"] == 1_160.833477
    assert artifact["production"]["dispatches"] == 2_286
    assert artifact["target"]["current_consumer_calls"] == 142
    assert artifact["target"]["current_consumer_ms"] == 49.191039999999994
    assert artifact["target"]["full_workgroups"] == 1_757_184
    assert artifact["target"]["tail_workgroups"] == 6_144
    assert artifact["target"]["candidate_consumer_launches"] == 238
    assert artifact["target"]["expected_candidate_request_dispatches"] == 2_382
    assert artifact["target"]["new_device_allocation_bytes"] == 0
    assert artifact["target"]["new_workspace_bytes"] == 0
    assert artifact["target"]["weight_producer_unchanged"]
    assert artifact["target"]["activation_pack_unchanged"]
    assert artifact["target"]["fma_reduction_store_order_unchanged"]
    assert artifact["target"]["tail_uses_exact_h6u_fallback"]
    assert not artifact["decision"]["candidate_implemented"]
    assert not artifact["decision"]["candidate_executed"]
    assert not artifact["decision"]["speed_result_exists"]
    assert artifact["decision"]["performance_claim"] == "target_selection_only"
    for term in (
        "all three 2/46/94-call roles",
        "full-prefix plus exact H6U tail",
        "rows17/33/M512",
    ):
        assert term in artifact["admission"]["correctness"]
    assert "142 H7V full launches" in artifact["admission"]["trace"]
    assert "2,382 request dispatches" in artifact["admission"]["trace"]
    assert "each of BF16 K3072/N1024 r5" in artifact["admission"]["timing"]
    assert "no r4-only/r5-only" in artifact["admission"]["no_salvage"]

    for relative, expected in artifact["source_sha256"].items():
        assert _sha256(_ROOT / relative) == expected
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == _H6U_POLICY
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_H6U_POLICY == _H6U_POLICY
    assert hip_gfx1151.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == {}
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256
    assert sum(_CALL_WEIGHTS.values()) == 142
    assert sum(weight for role, weight in _CALL_WEIGHTS.items() if role[1] == 5) == 96
    assert _TRACE_CONTRACT["consumer_launches"] == 142 + 96
    assert _TRACE_CONTRACT["request_dispatches"] == 2_286 + 96
    assert _TIMING_CONTRACT["role_weights"] == (2, 46, 94)
    assert _TIMING_CONTRACT["required_clocks"] == (
        "hip_event",
        "synchronized_wall",
    )
    assert not any(
        _TIMING_CONTRACT[name]
        for name in (
            "allow_subset_salvage",
            "allow_retune",
            "allow_recompile",
            "allow_favorable_rerun",
        )
    )


def test_h7v_candidate_source_structure_and_physical_contract() -> None:
    source = Path(q5_f32.__file__).with_suffix(".hip").read_text()
    h6u = _declaration(source, f"__global__ void {_H6U_KERNEL}(")
    assert h6u.count("if (row < rows)") == 2
    assert _H6U_REDUCE_HELPER in h6u
    assert h6u.count("__syncthreads();") == 1
    assert f"{_H6U_PERMLANE_HELPER}(" in source
    assert f"{_H6U_MOVE_HELPER}<0x108>" in source
    assert f"{_H6U_ADD_HELPER}(" in source

    # Intentional RED only after the current H6U source is frozen.
    primitives = []
    composites = []
    for col_tile, row_batch, output_dtype, *_ in _ROLES:
        primitive_name, composite_name = _candidate_names(
            col_tile,
            row_batch,
            output_dtype,
        )
        primitives.append(getattr(q5_f32, primitive_name))
        composites.append(getattr(q5_f32, composite_name))
    assert len(set(primitives)) == len(set(composites)) == 3
    assert source.count(f"__global__ void {_H7V_KERNEL}(") == 1
    full = _declaration(source, f"__global__ void {_H7V_KERNEL}(")
    assert "if (row < rows)" not in full
    assert "row_base >= rows" not in full
    assert "out_col_base >= out_features" not in full
    assert full.count(_H6U_REDUCE_HELPER) == 1
    assert full.count("__syncthreads();") == 1
    assert "for (int wave_index = 0; wave_index < 4; ++wave_index)" in full
    assert source.count(f"int {_H7V_LAUNCH}(") == 1
    launch = _declaration(source, f"int {_H7V_LAUNCH}(")
    assert "rows != 512" in launch
    assert _H7V_KERNEL in launch
    assert _H6U_KERNEL in launch
    assert "rows % ROW_BATCH" in launch
    assert "ROW_BATCH == 5" in launch
    assert set(_PHYSICAL_CONTRACT) == {
        (col_tile, row_batch, output_dtype)
        for col_tile, row_batch, output_dtype, *_ in _ROLES
    }


def test_h7v_registry_preserves_h6u_source_workspace_and_gfx1151() -> None:
    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    for col_tile, row_batch, output_dtype, *_ in _ROLES:
        for key, function in zip(
            _h6u_keys(col_tile, row_batch, output_dtype),
            (
                getattr(q5_f32, _h6u_names(col_tile, row_batch, output_dtype)[0]),
                getattr(q5_f32, _h6u_names(col_tile, row_batch, output_dtype)[1]),
            ),
            strict=True,
        ):
            assert resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            ) is function

    # Intentional RED after all retained H6U/package/workspace controls pass.
    for col_tile, row_batch, output_dtype, *_ in _ROLES:
        primitive_name, composite_name = _candidate_names(
            col_tile,
            row_batch,
            output_dtype,
        )
        primitive = getattr(q5_f32, primitive_name)
        composite = getattr(q5_f32, composite_name)
        for key, function in zip(
            _candidate_keys(col_tile, row_batch, output_dtype),
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
                KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
            )
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == _H6U_POLICY
    assert not hasattr(hip_gfx1100, "GGUF_Q6_F32_ORDERED_PREFILL_H7V_POLICY")
    assert not hasattr(hip_gfx1151, "GGUF_Q6_F32_ORDERED_PREFILL_H7V_POLICY")


@pytest.mark.parametrize(
    (
        "col_tile",
        "row_batch",
        "output_dtype",
        "in_features",
        "out_features",
        "pack_col_tile",
    ),
    _ROLES,
    ids=("bf16-k3072-n1024-r5", "bf16-k1024-n3072-r4", "f32-k3072-n1024-r5"),
)
def test_h7v_strict_m512_preflight_rejects_before_hip_loading(
    monkeypatch: pytest.MonkeyPatch,
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    in_features: int,
    out_features: int,
    pack_col_tile: int,
) -> None:
    del pack_col_tile
    load_attempts = 0

    def _unexpected_build(**_kwargs: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H7V role reached HIP loading")

    monkeypatch.setattr(
        q5_f32,
        "build_gguf_q5_k_f32_rocblas_prefill",
        _unexpected_build,
    )
    h6u = getattr(q5_f32, _h6u_names(col_tile, row_batch, output_dtype)[1])
    with pytest.raises(ValueError, match="rows must be positive"):
        h6u(1, 2, 3, 4, 5, 0, in_features, out_features)
    assert load_attempts == 0

    # Intentional RED only after retained H6U rejects before loading.
    candidate = _candidate(col_tile, row_batch, output_dtype)
    for rows in (0, 17, 33, 511, 513):
        with pytest.raises(ValueError, match="rows must be exactly 512"):
            candidate(1, 2, 3, 4, 5, rows, in_features, out_features)
    for hidden, outputs, message in (
        (in_features - 256, out_features, f"exactly {in_features}"),
        (in_features, out_features - 16, f"exactly {out_features}"),
    ):
        with pytest.raises(ValueError, match=message):
            candidate(1, 2, 3, 4, 5, 512, hidden, outputs)
    assert load_attempts == 0


@pytest.mark.parametrize(
    (
        "col_tile",
        "row_batch",
        "output_dtype",
        "in_features",
        "out_features",
        "pack_col_tile",
    ),
    _ROLES,
    ids=("bf16-k3072-n1024-r5", "bf16-k1024-n3072-r4", "f32-k3072-n1024-r5"),
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h7v_m512_full_tail_output_pack_cpu_repeat_and_lifecycle(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    in_features: int,
    out_features: int,
    pack_col_tile: int,
    library: Any,
    production_qweights: dict[tuple[int, int], np.ndarray],
) -> None:
    del pack_col_tile
    from hipengine.core.hip import get_hip_runtime

    rows = 512
    rng = np.random.default_rng(20260802 + 17 * in_features + out_features)
    x_bf16 = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    qweight = production_qweights[(in_features, out_features)]
    host_dtype = np.uint16 if output_dtype == "bf16" else np.float32
    expected = np.empty((rows, out_features), dtype=host_dtype)
    actual = np.empty_like(expected)
    repeat = np.empty_like(expected)
    plane_shape = _activation_plane_shape(rows, in_features, row_batch)
    plane = np.empty(plane_shape, dtype=np.uint16)
    expected_plane = _expected_activation_plane(x_bf16, row_batch)

    runtime = get_hip_runtime()
    before = memory_stats()
    buffers: list[Any] = []
    try:
        x_dev = _device(x_bf16, runtime)
        qweight_dev = _device(qweight, runtime)
        weight_f32_dev = malloc(
            q5_f32.q6_k_f32_ordered_workspace_nbytes(
                in_features,
                out_features,
            ),
            runtime=runtime,
        )
        expected_dev = malloc(expected.nbytes, runtime=runtime)
        actual_dev = malloc(actual.nbytes, runtime=runtime)
        repeat_dev = malloc(repeat.nbytes, runtime=runtime)
        activation_dev = malloc(plane.nbytes, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                qweight_dev,
                weight_f32_dev,
                expected_dev,
                actual_dev,
                repeat_dev,
                activation_dev,
            )
        )

        h6u = getattr(q5_f32, _h6u_names(col_tile, row_batch, output_dtype)[1])
        h6u(
            x_dev.ptr,
            qweight_dev.ptr,
            expected_dev.ptr,
            weight_f32_dev.ptr,
            activation_dev.ptr,
            rows,
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
            host_array_ptr(plane),
            activation_dev,
            plane.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(plane, expected_plane)
        _sampled_cpu_gate(
            expected,
            x_bf16,
            qweight,
            row_batch=row_batch,
            output_dtype=output_dtype,
            in_features=in_features,
            out_features=out_features,
        )

        # Intentional RED only after complete H6U, activation, and CPU bytes.
        candidate = _candidate(col_tile, row_batch, output_dtype)
        for out_dev, host in ((actual_dev, actual), (repeat_dev, repeat)):
            runtime.memset(activation_dev.ptr, 0xA5, plane.nbytes)
            runtime.memset(out_dev.ptr, 0x5A, host.nbytes)
            candidate(
                x_dev.ptr,
                qweight_dev.ptr,
                out_dev.ptr,
                weight_f32_dev.ptr,
                activation_dev.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(
                host_array_ptr(host),
                out_dev,
                host.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(plane),
                activation_dev,
                plane.nbytes,
                runtime=runtime,
            )
            np.testing.assert_array_equal(host, expected)
            np.testing.assert_array_equal(plane, expected_plane)
            assert np.isfinite(
                host.view(np.uint16 if output_dtype == "bf16" else np.float32)
            ).all()
        np.testing.assert_array_equal(repeat, actual)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
