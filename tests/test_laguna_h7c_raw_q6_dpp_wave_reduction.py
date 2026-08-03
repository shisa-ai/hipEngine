"""WPF-H7C exact raw-Q6 DPP-add wave-reduction RED contract."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    free,
    host_array_ptr,
    memory_stats,
)
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from tests.test_gguf_k_rowtile_gemv import _bf16_bits, _bf16_to_f32
from tests.test_gguf_q5_k_f32_rocblas_prefill import (
    _device,
    _exact_q6_f32_cpu,
)

# role, col tile, row batch, output dtype, exact K, exact N
_ROLES = (
    ("layer-0-dense-ffn-down", 4, 8, "bf16", 12_288, 3_072),
    ("layer-47-attention-q", 2, 16, "f32", 3_072, 9_216),
    ("layer-47-attention-output", 4, 8, "bf16", 9_216, 3_072),
)
_ROWS = (1, 7, 8, 9, 512)
_H7C_KERNEL = "gguf_q6_k_prefill_out_coltile_rowbatch_dpp_wave_reduction_kernel"
_H7C_REDUCE_HELPER = "h7c_reduce_wave_accumulators_dpp"
_H7C_PERMLANE_HELPER = "h7c_permlanex16_f32"
_H7C_MOVE_HELPER = "h7c_dpp_move_f32"
_H7C_ADD_HELPER = "h7c_dpp_add_row_shl1_f32"
_GENERIC_KERNEL = "gguf_k_prefill_out_coltile_rowbatch_kernel"
_GENERIC_KERNEL_SHA256 = (
    "087a8fa301661f159dbd7625aac7226b0772a4a75a149c069988cb2ea6832e51"
)
_GENERIC_LAUNCH_SHA256 = (
    "ff7a1bd655828d7880c62d57550704793aea339369af4c459f9d4d34d070dd31"
)
_GENERIC_WRAPPER_FACTORY_SHA256 = (
    "86e9b569b72c127e8b5be9633bf01da01b01940a83aef187fb6c126b26564cea"
)
_GENERIC_LAUNCH_WRAPPER_SHA256 = (
    "3e67bcf8b32884ed203530024dbe4e77112c40aa84712667de258b02d8734208"
)
_GENERIC_VALIDATE_SHA256 = (
    "bf1cd90230a61312a75acd401fe3a8c67eac00b59476d655c0d67db354a63e11"
)
_RAW_DISPATCH_SHA256 = (
    "c12ea5317a53973204166ffe5589d44674e4f9d8df3b9f9964fb5656bce4df92"
)
_SELECTION_ARITHMETIC = {
    "bf16_calls": 2,
    "bf16_workgroups_per_call": 49_152,
    "f32_workgroups": 147_456,
    "total_workgroups": 245_760,
    "waves_per_workgroup": 4,
    "wave_instances": 983_040,
    "source_static_ds_bpermute_b32": 160,
    "candidate_static_permlanex16": 32,
    "candidate_static_dpp_add": 128,
    "source_dynamic_bpermute_wave_instructions": 157_286_400,
    "candidate_dynamic_permlanex_wave_instructions": 31_457_280,
    "candidate_dynamic_dpp_wave_instructions": 125_829_120,
    "reduction_steps_changed": 0,
    "logical_weight_activation_output_bytes_changed": 0,
}
_COMMON_PHYSICAL = {
    "source_ds_bpermute_b32": 160,
    "candidate_ds_bpermute_b32": 0,
    "source_permlanex16": 0,
    "candidate_permlanex16": 32,
    "source_dpp_add": 0,
    "candidate_dpp_add": 128,
    "source_global_loads": 24,
    "candidate_global_loads": 24,
    "source_global_stores": 1,
    "candidate_global_stores": 1,
    "source_ds_store_b128": 8,
    "candidate_ds_store_b128": 8,
    "source_ds_load_2addr_b32": 2,
    "candidate_ds_load_2addr_b32": 2,
    "source_barriers": 1,
    "candidate_barriers": 1,
    "source_ordered_fma_operations": 32,
    "candidate_ordered_fma_operations": 32,
    "metadata_vgpr_max": 72,
    "runtime_vgpr_max": 72,
    "metadata_lds_bytes": 512,
    "runtime_lds_bytes": 512,
    "private_bytes": 0,
    "vgpr_spills": 0,
    "sgpr_spills": 0,
    "runtime_scratch_bytes": 0,
    "local_size": 128,
}
_PHYSICAL_BY_OUTPUT = {
    "bf16": {
        "source_code_bytes": 4_840,
        "candidate_code_bytes_max": 4_840,
        "source_instruction_slots": 843,
        "candidate_instruction_slots_max": 843,
        "source_metadata_sgpr": 50,
        "candidate_metadata_sgpr_max": 50,
        "grid_x": 98_304,
        "grid_y": 64,
    },
    "f32": {
        "source_code_bytes": 5_040,
        "candidate_code_bytes_max": 5_040,
        "source_instruction_slots": 909,
        "candidate_instruction_slots_max": 909,
        "source_metadata_sgpr": 69,
        "candidate_metadata_sgpr_max": 69,
        "grid_x": 589_824,
        "grid_y": 32,
    },
}
_TRACE_CONTRACT = {
    "kernel_name": _H7C_KERNEL,
    "require_cached_build": True,
    "new_compiler_processes": 0,
    "local_size": 128,
    "positive_duration": True,
    "required_grids": ((98_304, 64), (589_824, 32)),
}
_TIMING_CONTRACT = {
    "warmups": 5,
    "repetitions": 15,
    "launches_per_sample": 5,
    "order": "counter_rotated",
    "clocks": ("hip_event", "synchronized_wall"),
    "roles": tuple(role[0] for role in _ROLES),
    "role_weights": (1, 1, 1),
    "require_each_role_both_clocks": True,
    "require_weighted_aggregate_both_clocks": True,
    "one_shot_only": True,
}
_REJECTION_SURFACES = (
    "gfx1100 HIP kernel/exports",
    "gfx1100 Python wrappers/registry keys",
    "gfx1151 explicit exclusions",
    "H7C tests",
)
_REJECT_RULE = (
    "Any correctness, physical, resource, cached-trace, compiler, lifecycle, "
    "per-role both-clock, or aggregate both-clock miss removes every H7C "
    "surface without tuning or rerun."
)


def _module():
    return importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv"
    )


def _suffix(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    *,
    candidate: bool,
) -> str:
    prefix = "dpp_wave_reduction_" if candidate else ""
    return (
        f"{prefix}coltile{col_tile}_rowbatch{row_batch}_"
        f"bf16_{output_dtype}_out"
    )


def _wrapper_name(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    *,
    candidate: bool,
) -> str:
    return "gguf_q6_k_gemv_" + _suffix(
        col_tile,
        row_batch,
        output_dtype,
        candidate=candidate,
    )


def _control(col_tile: int, row_batch: int, output_dtype: str):
    return getattr(
        _module(),
        _wrapper_name(
            col_tile,
            row_batch,
            output_dtype,
            candidate=False,
        ),
    )


def _candidate(col_tile: int, row_batch: int, output_dtype: str):
    return getattr(
        _module(),
        _wrapper_name(
            col_tile,
            row_batch,
            output_dtype,
            candidate=True,
        ),
    )


def _key(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    *,
    backend: str = "hip_gfx1100",
    candidate: bool,
) -> KernelKey:
    return KernelKey(
        backend,
        "linear",
        "gguf_q6_k",
        _suffix(
            col_tile,
            row_batch,
            output_dtype,
            candidate=candidate,
        ),
    )


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


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _allocation_lifecycle() -> tuple[int, int]:
    stats = memory_stats()
    return stats["current_allocated_bytes"], stats["active_allocations"]


def _sample_columns(out_features: int) -> np.ndarray:
    return np.unique(
        np.asarray(
            (
                0,
                1,
                2,
                3,
                4,
                7,
                8,
                15,
                16,
                31,
                32,
                out_features // 2 - 1,
                out_features // 2,
                out_features // 2 + 1,
                out_features - 33,
                out_features - 32,
                out_features - 17,
                out_features - 16,
                out_features - 9,
                out_features - 8,
                out_features - 5,
                out_features - 4,
                out_features - 2,
                out_features - 1,
            ),
            dtype=np.int64,
        )
    )


def _q6_weight(out_features: int, in_features: int) -> np.ndarray:
    """Build finite deterministic raw-Q6 bytes without a giant Python loop."""

    blocks_per_row = in_features // 256
    rng = np.random.default_rng(20260801 + in_features + 7 * out_features)
    blocks = rng.integers(
        0,
        256,
        size=(out_features, blocks_per_row, 210),
        dtype=np.uint8,
    )
    scales = rng.integers(
        -128,
        128,
        size=(out_features, blocks_per_row, 16),
        dtype=np.int16,
    ).astype(np.int8)
    blocks[:, :, 192:208] = scales.view(np.uint8)
    d_values = np.asarray(
        (0.0, 0.001953125, -0.00390625, 0.0078125),
        dtype=np.float16,
    )
    d = d_values[
        (
            np.arange(out_features)[:, None]
            + np.arange(blocks_per_row)[None, :]
        )
        % len(d_values)
    ]
    blocks[:, :, 208:210] = d.view(np.uint8).reshape(
        out_features,
        blocks_per_row,
        2,
    )
    return np.ascontiguousarray(blocks.reshape(out_features, -1))


@pytest.fixture(scope="module")
def library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    module = _module()
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    return module.build_gguf_k_gemv(
        load=True,
        compiler_version=compiler_version,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )


@pytest.fixture(scope="module")
def qweights() -> dict[tuple[int, int], np.ndarray]:
    return {
        (in_features, out_features): _q6_weight(out_features, in_features)
        for _, _, _, _, in_features, out_features in _ROLES
    }


@pytest.fixture(scope="module")
def sampled_cpu_weights(
    qweights: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    result = {}
    for _, _, _, _, in_features, out_features in _ROLES:
        columns = _sample_columns(out_features)
        result[(in_features, out_features)] = (
            columns,
            _exact_q6_f32_cpu(
                qweights[(in_features, out_features)][columns],
                in_features,
            ),
        )
    return result


def _run(
    wrapper,
    *,
    library: Any,
    x_bf16: np.ndarray,
    qweight: np.ndarray,
    output_dtype: str,
) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rows, in_features = x_bf16.shape
    out_features = qweight.shape[0]
    dtype = np.uint16 if output_dtype == "bf16" else np.float32
    poison = (
        np.full((rows, out_features), 0x7FC0, dtype=np.uint16)
        if output_dtype == "bf16"
        else np.full((rows, out_features), np.nan, dtype=np.float32)
    )
    actual = np.empty((rows, out_features), dtype=dtype)
    before = _allocation_lifecycle()
    buffers = []
    try:
        x_dev = _device(x_bf16, runtime)
        weight_dev = _device(qweight, runtime)
        output_dev = _device(poison, runtime)
        buffers.extend((x_dev, weight_dev, output_dev))
        wrapper(
            x_dev.ptr,
            weight_dev.ptr,
            output_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(actual),
            output_dev,
            actual.nbytes,
            runtime=runtime,
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    assert _allocation_lifecycle() == before
    if output_dtype == "bf16":
        assert not np.any(actual == np.uint16(0x7FC0))
        assert np.isfinite(_bf16_to_f32(actual)).all()
    else:
        assert np.isfinite(actual).all()
    return actual


def _sampled_cpu_gate(
    actual: np.ndarray,
    x_bf16: np.ndarray,
    *,
    row_batch: int,
    output_dtype: str,
    columns: np.ndarray,
    cpu_weight: np.ndarray,
) -> None:
    sample_rows = np.unique(
        np.asarray(
            (
                0,
                min(row_batch - 1, actual.shape[0] - 1),
                min(row_batch, actual.shape[0] - 1),
                actual.shape[0] - 1,
            ),
            dtype=np.int64,
        )
    )
    cpu = np.asarray(
        _bf16_to_f32(x_bf16[sample_rows]) @ cpu_weight.T,
        dtype=np.float32,
    )
    sampled = actual[np.ix_(sample_rows, columns)]
    sampled_f32 = (
        _bf16_to_f32(sampled)
        if output_dtype == "bf16"
        else np.asarray(sampled, dtype=np.float32)
    )
    relative = np.abs(sampled_f32 - cpu) / np.maximum(np.abs(cpu), 1.0)
    assert float(np.max(relative)) <= 0.05
    assert evaluate_logits(cpu, sampled_f32).passed


def test_h7c_frozen_selection_physical_trace_timing_and_rejection_contract() -> None:
    total_workgroups = 49_152 * 2 + 147_456
    wave_instances = total_workgroups * 4
    assert _SELECTION_ARITHMETIC == {
        "bf16_calls": 2,
        "bf16_workgroups_per_call": 49_152,
        "f32_workgroups": 147_456,
        "total_workgroups": total_workgroups,
        "waves_per_workgroup": 4,
        "wave_instances": wave_instances,
        "source_static_ds_bpermute_b32": 160,
        "candidate_static_permlanex16": 32,
        "candidate_static_dpp_add": 128,
        "source_dynamic_bpermute_wave_instructions": wave_instances * 160,
        "candidate_dynamic_permlanex_wave_instructions": wave_instances * 32,
        "candidate_dynamic_dpp_wave_instructions": wave_instances * 128,
        "reduction_steps_changed": 0,
        "logical_weight_activation_output_bytes_changed": 0,
    }
    assert _COMMON_PHYSICAL == {
        "source_ds_bpermute_b32": 160,
        "candidate_ds_bpermute_b32": 0,
        "source_permlanex16": 0,
        "candidate_permlanex16": 32,
        "source_dpp_add": 0,
        "candidate_dpp_add": 128,
        "source_global_loads": 24,
        "candidate_global_loads": 24,
        "source_global_stores": 1,
        "candidate_global_stores": 1,
        "source_ds_store_b128": 8,
        "candidate_ds_store_b128": 8,
        "source_ds_load_2addr_b32": 2,
        "candidate_ds_load_2addr_b32": 2,
        "source_barriers": 1,
        "candidate_barriers": 1,
        "source_ordered_fma_operations": 32,
        "candidate_ordered_fma_operations": 32,
        "metadata_vgpr_max": 72,
        "runtime_vgpr_max": 72,
        "metadata_lds_bytes": 512,
        "runtime_lds_bytes": 512,
        "private_bytes": 0,
        "vgpr_spills": 0,
        "sgpr_spills": 0,
        "runtime_scratch_bytes": 0,
        "local_size": 128,
    }
    assert _PHYSICAL_BY_OUTPUT["bf16"] == {
        "source_code_bytes": 4_840,
        "candidate_code_bytes_max": 4_840,
        "source_instruction_slots": 843,
        "candidate_instruction_slots_max": 843,
        "source_metadata_sgpr": 50,
        "candidate_metadata_sgpr_max": 50,
        "grid_x": 98_304,
        "grid_y": 64,
    }
    assert _PHYSICAL_BY_OUTPUT["f32"] == {
        "source_code_bytes": 5_040,
        "candidate_code_bytes_max": 5_040,
        "source_instruction_slots": 909,
        "candidate_instruction_slots_max": 909,
        "source_metadata_sgpr": 69,
        "candidate_metadata_sgpr_max": 69,
        "grid_x": 589_824,
        "grid_y": 32,
    }
    assert _TRACE_CONTRACT == {
        "kernel_name": _H7C_KERNEL,
        "require_cached_build": True,
        "new_compiler_processes": 0,
        "local_size": 128,
        "positive_duration": True,
        "required_grids": ((98_304, 64), (589_824, 32)),
    }
    assert _TIMING_CONTRACT == {
        "warmups": 5,
        "repetitions": 15,
        "launches_per_sample": 5,
        "order": "counter_rotated",
        "clocks": ("hip_event", "synchronized_wall"),
        "roles": tuple(role[0] for role in _ROLES),
        "role_weights": (1, 1, 1),
        "require_each_role_both_clocks": True,
        "require_weighted_aggregate_both_clocks": True,
        "one_shot_only": True,
    }
    assert len(_REJECTION_SURFACES) == 4
    assert "H7C tests" in _REJECTION_SURFACES
    assert _REJECT_RULE.endswith("without tuning or rerun.")
    assert "every H7C surface" in _REJECT_RULE


@pytest.mark.parametrize(
    ("role", "col_tile", "row_batch", "output_dtype", "in_features", "out_features"),
    _ROLES,
    ids=tuple(role[0] for role in _ROLES),
)
def test_h7c_registry_source_policy_and_generic_immutability(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    in_features: int,
    out_features: int,
) -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.runtime.gguf_linear import (
        GGUFLinearDispatch,
        _raw_k_prefill_rowbatch_dispatch,
    )

    del role
    module = _module()
    source = Path(module.__file__).with_suffix(".hip").read_text()
    generic_kernel = _declaration(
        source,
        f"__global__ void {_GENERIC_KERNEL}(",
    )
    generic_launch = _declaration(
        source,
        "int launch_gguf_k_gemv_coltile_rowbatch_out(",
    )
    assert _sha256(generic_kernel) == _GENERIC_KERNEL_SHA256
    assert _sha256(generic_launch) == _GENERIC_LAUNCH_SHA256
    assert _sha256(inspect.getsource(module._make_wrapper)) == (
        _GENERIC_WRAPPER_FACTORY_SHA256
    )
    assert _sha256(inspect.getsource(module._launch)) == (
        _GENERIC_LAUNCH_WRAPPER_SHA256
    )
    assert _sha256(inspect.getsource(module._validate)) == (
        _GENERIC_VALIDATE_SHA256
    )
    assert _sha256(inspect.getsource(_raw_k_prefill_rowbatch_dispatch)) == (
        _RAW_DISPATCH_SHA256
    )
    assert generic_kernel.count("__shfl_down") == 1
    assert generic_kernel.count("__syncthreads();") == 1
    assert "float acc[ROW_BATCH][COL_TILE] = {};" in generic_kernel

    expected_coltile2 = frozenset(
        {
            ("gguf_q5_k", "bf16_bf16_out", 3_072, 12_288),
            ("gguf_q5_k", "bf16_f32_out", 3_072, 6_144),
            ("gguf_q5_k", "bf16_f32_out", 3_072, 9_216),
            ("gguf_q6_k", "bf16_f32_out", 3_072, 9_216),
        }
    )
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED is True
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_ROWBATCH == 32
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED is True
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_COLTILE2_SHAPES == expected_coltile2
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_VARIANT == "coltile"
    expected_h7c_policy = {
        (
            "gguf_q6_k",
            f"bf16_{role_output_dtype}_out",
            512,
            role_in_features,
            role_out_features,
        ): _suffix(
            role_col_tile,
            role_row_batch,
            role_output_dtype,
            candidate=True,
        )
        for (
            _,
            role_col_tile,
            role_row_batch,
            role_output_dtype,
            role_in_features,
            role_out_features,
        ) in _ROLES
    }
    expected_h7i_policy = {
        role: variant.replace(
            "dpp_wave_reduction_",
            "dpp_wave_reduction_full_group_compute_",
            1,
        )
        for role, variant in expected_h7c_policy.items()
    }
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_GENERIC_ROLE_VARIANTS == {}
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_H7C_ROLE_VARIANTS == expected_h7c_policy
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_ROLE_VARIANTS == expected_h7i_policy
    monkeypatch.setattr(
        hip_gfx1100,
        "GGUF_RAW_K_PREFILL_ROLE_VARIANTS",
        {},
    )

    control = _control(col_tile, row_batch, output_dtype)
    control_key = _key(
        col_tile,
        row_batch,
        output_dtype,
        candidate=False,
    )
    assert resolve(
        backend=control_key.backend,
        layer=control_key.layer,
        quant=control_key.quant,
        variant=control_key.variant,
    ) is control
    output_variant = f"bf16_{output_dtype}_out"
    base = GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k",
            f"prefill_{output_variant}",
        ),
        "raw",
    )
    selected = _raw_k_prefill_rowbatch_dispatch(
        base,
        rows=512,
        in_features=in_features,
        out_features=out_features,
        row_batch=32,
        variant="coltile",
    )
    assert selected.key == control_key
    assert selected.abi == "raw"

    load_backend_kernel_package("hip_gfx1151")
    assert hip_gfx1151.GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED is False
    assert hip_gfx1151.GGUF_RAW_K_PREFILL_ROWBATCH == 0
    assert hip_gfx1151.GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED is False
    assert hip_gfx1151.GGUF_RAW_K_PREFILL_COLTILE2_SHAPES == frozenset()
    assert hip_gfx1151.GGUF_RAW_K_PREFILL_VARIANT == "rowbatch"

    # Intentional RED: generic source bytes, package/runtime selection, and
    # backend isolation pass before the separately named H7C wrapper lookup.
    candidate = _candidate(col_tile, row_batch, output_dtype)
    candidate_key = _key(
        col_tile,
        row_batch,
        output_dtype,
        candidate=True,
    )
    assert candidate.__name__ == _wrapper_name(
        col_tile,
        row_batch,
        output_dtype,
        candidate=True,
    )
    assert resolve(
        backend=candidate_key.backend,
        layer=candidate_key.layer,
        quant=candidate_key.quant,
        variant=candidate_key.variant,
    ) is candidate
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            candidate_key.layer,
            candidate_key.quant,
            candidate_key.variant,
        )
    )
    assert selected.key == control_key
    assert source.count("hipengine_" + candidate.__name__) == 1
    assert source.count(_H7C_KERNEL) == 2

    helper = _declaration(
        source,
        f"__device__ inline void {_H7C_REDUCE_HELPER}(",
    )
    steps = (
        f"acc[row][col] += {_H7C_PERMLANE_HELPER}(acc[row][col]);",
        f"acc[row][col] += {_H7C_MOVE_HELPER}<0x108>(acc[row][col]);",
        f"acc[row][col] += {_H7C_MOVE_HELPER}<0x104>(acc[row][col]);",
        f"acc[row][col] += {_H7C_MOVE_HELPER}<0x102>(acc[row][col]);",
        f"acc[row][col] = {_H7C_ADD_HELPER}(acc[row][col]);",
    )
    offsets = [helper.index(step) for step in steps]
    assert offsets == sorted(offsets)
    assert "__shfl_down" not in helper
    assert f"{_H7C_MOVE_HELPER}<0x101>" not in helper
    assert "__syncthreads" not in helper

    permlane = _declaration(
        source,
        f"__device__ inline float {_H7C_PERMLANE_HELPER}(",
    )
    assert "__builtin_amdgcn_permlanex16" in permlane
    assert "0x76543210U" in permlane
    assert "0xFEDCBA98U" in permlane
    direct_add = _declaration(
        source,
        f"__device__ inline float {_H7C_ADD_HELPER}(",
    )
    assert "v_add_f32_dpp %0, %1, %1 row_shl:1" in direct_add
    assert "row_mask:0xf bank_mask:0xf bound_ctrl:1" in direct_add

    candidate_kernel = _declaration(
        source,
        f"__global__ void {_H7C_KERNEL}(",
    )
    assert candidate_kernel.count(_H7C_REDUCE_HELPER) == 1
    assert "__shfl_down" not in candidate_kernel
    assert candidate_kernel.count("__syncthreads();") == 1
    assert "float acc[ROW_BATCH][COL_TILE] = {};" in candidate_kernel
    assert "COL_TILE * ROW_BATCH == 32" in candidate_kernel
    assert "q6_k_weight(weight_rows[col], k)" in candidate_kernel
    assert "q5_k_weight" not in candidate_kernel
    assert "qtype" not in candidate_kernel
    assert "__launch_bounds__(128, 2)" in source[
        source.rfind("__launch_bounds__", 0, source.index(candidate_kernel)) :
        source.index(candidate_kernel)
    ]


@pytest.mark.parametrize(
    ("role", "col_tile", "row_batch", "output_dtype", "in_features", "out_features"),
    _ROLES,
    ids=tuple(role[0] for role in _ROLES),
)
def test_h7c_strict_role_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    in_features: int,
    out_features: int,
) -> None:
    del role
    module = _module()
    control = _control(col_tile, row_batch, output_dtype)
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H7C raw-Q6 role reached HIP loader")

    monkeypatch.setattr(module, "build_gguf_k_gemv", fail_if_loaded)
    generic_invalid = (
        ((1, 2, 3, 0, in_features, out_features), {}, "rows must be positive"),
        (
            (1, 2, 3, 7, in_features - 1, out_features),
            {},
            "divisible by GGUF",
        ),
        ((1, 2, 3, 7, in_features, 0), {}, "out_features must be positive"),
        (
            (1, 2, 3, 7, in_features, out_features),
            {"threads": 96},
            "threads must be one of",
        ),
    )
    for args, kwargs, message in generic_invalid:
        with pytest.raises(ValueError, match=message):
            control(*args, **kwargs)
    assert load_attempts == 0

    # Intentional RED only after all retained generic bounds are proven.
    candidate = _candidate(col_tile, row_batch, output_dtype)
    if output_dtype == "bf16":
        candidate_invalid = (
            ((1, 2, 3, 0, in_features, out_features), {}, "rows must be positive"),
            (
                (1, 2, 3, 7, 3_072, out_features),
                {},
                "in_features must be exactly 9216 or 12288",
            ),
            (
                (1, 2, 3, 7, in_features, out_features - 4),
                {},
                "out_features must be exactly 3072",
            ),
            (
                (1, 2, 3, 7, in_features, out_features),
                {"threads": 64},
                "threads must be exactly 128",
            ),
        )
    else:
        candidate_invalid = (
            ((1, 2, 3, 0, in_features, out_features), {}, "rows must be positive"),
            (
                (1, 2, 3, 7, in_features + 256, out_features),
                {},
                "in_features must be exactly 3072",
            ),
            (
                (1, 2, 3, 7, in_features, out_features - 2),
                {},
                "out_features must be exactly 9216",
            ),
            (
                (1, 2, 3, 7, in_features, out_features),
                {"threads": 64},
                "threads must be exactly 128",
            ),
        )
    for args, kwargs, message in candidate_invalid:
        with pytest.raises(ValueError, match=message):
            candidate(*args, **kwargs)
    assert load_attempts == 0


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", _ROWS, ids=tuple(f"rows{rows}" for rows in _ROWS))
@pytest.mark.parametrize(
    ("role", "col_tile", "row_batch", "output_dtype", "in_features", "out_features"),
    _ROLES,
    ids=tuple(role[0] for role in _ROLES),
)
def test_h7c_complete_outputs_match_generic_and_sampled_cpu(
    rows: int,
    role: str,
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    in_features: int,
    out_features: int,
    library: Any,
    qweights: dict[tuple[int, int], np.ndarray],
    sampled_cpu_weights: dict[
        tuple[int, int], tuple[np.ndarray, np.ndarray]
    ],
) -> None:
    rng = np.random.default_rng(
        20260801 + 17 * rows + 3 * in_features + out_features
    )
    x_bf16 = _bf16_bits(
        rng.uniform(
            0.0078125,
            0.03125,
            size=(rows, in_features),
        ).astype(np.float32)
    )
    qweight = qweights[(in_features, out_features)]
    control = _control(col_tile, row_batch, output_dtype)
    expected = _run(
        control,
        library=library,
        x_bf16=x_bf16,
        qweight=qweight,
        output_dtype=output_dtype,
    )
    columns, cpu_weight = sampled_cpu_weights[(in_features, out_features)]
    _sampled_cpu_gate(
        expected,
        x_bf16,
        row_batch=row_batch,
        output_dtype=output_dtype,
        columns=columns,
        cpu_weight=cpu_weight,
    )

    # Intentional RED only after complete poisoned generic bytes, independent
    # sampled CPU values, finiteness, and allocation lifecycle pass.
    candidate = _candidate(col_tile, row_batch, output_dtype)
    actual = _run(
        candidate,
        library=library,
        x_bf16=x_bf16,
        qweight=qweight,
        output_dtype=output_dtype,
    )
    np.testing.assert_array_equal(actual, expected, err_msg=f"{role}: rows={rows}")
