"""WPF-H8C exact shared-Q5 dual-weight consumer RED contract."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

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
from tests.test_laguna_h5y_q5_activation_tile_k_row import (
    _activation_plane_shape,
    _candidate_names as _h5y_names,
    _expected_activation_plane,
    _hip_available,
)
from tests.test_laguna_h7g_q5_padded_compute import (
    _H7G_KERNEL,
    _bf16_bits,
    _bf16_to_f32,
    _declaration,
    _device,
    _production_q5_weight,
    _sampled_cpu_gate,
    _sha256,
)
from tests.test_laguna_h7h_q5_full_group_compute import (
    _H7G_KERNEL_SHA256,
    _h7h_names,
)

_ROOT = Path(__file__).resolve().parents[1]
_TARGET_ARTIFACT = _ROOT / (
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8b-"
    "shared-q5-dual-consumer-target.json"
)
_TARGET_ARTIFACT_SHA256 = (
    "6188a5de4d971180c764cd755d30b026ce902bfa45d63e725c0cf1170cd4bd45"
)
_PRODUCTION_ARTIFACT_SHA256 = (
    "d7f62709a66d255caf8e7a8a4ec2eaf9ce713916c7b78ffaecd7e68d2ff69d91"
)
_TARGET_SOURCE_SHA256 = {
    "hipengine/kernels/activation_pack.py": (
        "2b10234b49ee19417e439fa598b0b069ae4b832eebb3751357c7819891072f67"
    ),
    "hipengine/kernels/hip_gfx1100/__init__.py": (
        "3638a8fb56d7f87b928bd4f9c8f533f3923381db4d7d7a6e1929ae283b37968d"
    ),
    "hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.hip": (
        "1a06011ea6e7bda8e0b48fd357cbcbadaff76793a1b5c49bd217cc83d32b7110"
    ),
    "hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.py": (
        "fb9b2ae1a88300ac1e754b8c3214310db65d3e2343598b7631ac185ec141f33e"
    ),
    "hipengine/kernels/hip_gfx1151/__init__.py": (
        "a5838ffc8fd8df367cd828f397e701f94f2268c7992d0a5e143c8d7e2b8ba3b3"
    ),
    "hipengine/runtime/gguf_linear.py": (
        "f9ebb089b31937dcaea27f8bb43bfc2936b294d541c2841465c498d6f6dbd363"
    ),
    "hipengine/runtime/laguna_gguf_runner.py": (
        "edea1fc2df3c8ca46fe3396663ac14f9000b4ee0cc967ebafb55208afad50654"
    ),
    "hipengine/runtime/laguna_moe.py": (
        "0507c0ab9bcabddfda9d0390c66d46f80aaaf7c42357a58dfa24c692d43414fd"
    ),
}
_ROWS = (17, 33, 512)
_COL_TILE = 8
_ROW_BATCH = 4
_OUTPUT_DTYPE = "bf16"
_WEIGHT_LAYOUT = "tile_k_col"
_IN_FEATURES = 3_072
_OUT_FEATURES = 1_024
_GROUPS = 46
_PLANE_NBYTES = _IN_FEATURES * _OUT_FEATURES * 4
_PRIMITIVE_VARIANT = (
    "ordered_weight_major_tile_k_col_activation_tile_k_row_"
    "dual_weight_full_group_compute_coltile8_rowbatch4_bf16_bf16_out"
)
_PRIMITIVE_NAME = f"gguf_q5_k_f32_weight_{_PRIMITIVE_VARIANT}"
_KERNEL = (
    "gguf_q5_k_f32_weight_ordered_weight_major_tile_k_col_activation_"
    "tile_k_row_dual_weight_full_group_compute_kernel"
)
_SYMBOL = f"hipengine_{_PRIMITIVE_NAME}"
_KEY = KernelKey("hip_gfx1100", "linear_pair", "f32_weight", _PRIMITIVE_VARIANT)
_SUPPORTED_CAPABILITY = "LAGUNA_SHARED_Q5_DUAL_WEIGHT_SUPPORTED"
_SOURCE_CAPABILITY = "LAGUNA_SHARED_Q5_DUAL_WEIGHT"
_SESSION_PARAMETER = "use_shared_q5_dual_weight"


def _candidate():
    return getattr(q5_f32, _PRIMITIVE_NAME)


def test_h8c_frozen_target_sources_topology_and_admission_contract() -> None:
    artifact_bytes = _TARGET_ARTIFACT.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == _TARGET_ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)

    assert artifact["status"] == "accepted_target_only_no_candidate_implementation"
    assert artifact["head"] == "f0069b89a9a7b11d74fdbd40b888e0e1bc8794a3"
    assert artifact["source_sha256"] == _TARGET_SOURCE_SHA256
    assert (
        artifact["execution_unchanged_audit"]["production_artifact_sha256"]
        == _PRODUCTION_ARTIFACT_SHA256
    )

    target = artifact["target"]
    assert target["id"] == "WPF-H8C"
    assert target["implemented"] is False
    assert target["candidate_executed"] is False
    assert target["new_device_body_exists"] is False
    assert target["new_jit_object_exists"] is False
    assert target["new_allocation_bytes"] == 0
    assert target["new_workspace_bytes"] == 0
    assert target["source_default_changed"] is False
    assert target["raw_pointer_abi"] == [
        "activation_tile_k_row_ptr",
        "weight_gate_f32_tile_k_col_ptr",
        "weight_up_f32_tile_k_col_ptr",
        "gate_out_ptr",
        "up_out_ptr",
        "rows",
        "in_features",
        "out_features_each",
        "stream",
    ]
    assert "all 46 architecture-defined" in target["scope"]

    selected = artifact["execution_unchanged_audit"]["selected_class"]
    assert selected["name"] == "complete_shared_q5_gate_up"
    assert selected["groups"] == _GROUPS
    assert selected["current_consumer_calls"] == 92
    assert selected["target_consumer_calls"] == _GROUPS
    assert selected["current_pack_calls"] == selected["target_pack_calls"] == 46
    assert selected["current_producer_calls"] == selected["target_producer_calls"] == 92
    assert selected["in_features"] == _IN_FEATURES
    assert selected["out_features_each"] == _OUT_FEATURES
    assert selected["rows"] == 512
    assert selected["current_local_size"] == 128
    assert selected["current_vgpr"] == 72
    assert selected["current_lds_bytes"] == 512
    assert selected["current_scratch_bytes"] == 0

    operation = artifact["operation_model"]
    assert operation["dispatches_before"] == 2_155
    assert operation["dispatch_delta_model"] == -_GROUPS
    assert operation["dispatches_after_model"] == 2_109
    assert operation["current_logical_activation_bytes"] == 37_044_092_928
    assert operation["target_logical_activation_bytes"] == 18_522_046_464
    assert operation["logical_activation_load_reduction_percent"] == 50.0
    assert operation["fmas_unchanged"] == 148_176_371_712
    assert operation["packs_unchanged"] is True
    assert operation["weight_producers_unchanged"] is True
    assert operation["output_association_unchanged"] is True
    assert operation["not_a_speed_claim"] is True

    physical = artifact["physical_gate"]
    assert physical == {
        "local_size": 128,
        "max_lds_bytes": 1_024,
        "max_runtime_vgpr": 136,
        "private_bytes": 0,
        "requires_each_output_to_preserve_current_k_fma_reduction_store_sequence": True,
        "requires_one_activation_load_shared_across_both_weight_accumulator_sets": True,
        "scratch_bytes": 0,
        "sgpr_spills": 0,
        "vgpr_spills": 0,
    }
    admission = artifact["admission"]
    assert admission["red_first"] is True
    assert admission["target_commit_before_red"] is True
    assert "all 46 actual pairs are inseparable" in admission["all_class_screen"]
    assert "rows17/33/M512" in admission["leaf_correctness"]
    assert "2,155→2,109" in admission["trace"]
    assert "no layer/role/prompt/token/length" in admission["no_salvage"]


def test_h8c_primitive_registry_source_and_backend_exclusion() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    primitive = _candidate()  # Intentional RED: no H8C function exists yet.
    assert getattr(hip_gfx1100, _SUPPORTED_CAPABILITY) is True
    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) is False
    assert _SUPPORTED_CAPABILITY in hip_gfx1100.__all__
    assert _SOURCE_CAPABILITY in hip_gfx1100.__all__
    assert not hasattr(hip_gfx1151, _SUPPORTED_CAPABILITY)
    assert not hasattr(hip_gfx1151, _SOURCE_CAPABILITY)

    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert resolve(
        backend=_KEY.backend,
        layer=_KEY.layer,
        quant=_KEY.quant,
        variant=_KEY.variant,
    ) is primitive
    for backend in ("hip_gfx1151", "cuda_sm86", "cpu_reference"):
        assert not is_registered(
            KernelKey(backend, _KEY.layer, _KEY.quant, _KEY.variant)
        )
    assert q5_f32._Q5_DUAL_WEIGHT_FULL_GROUP_COMPUTE_ROLES == (
        (
            _COL_TILE,
            _ROW_BATCH,
            _OUTPUT_DTYPE,
            _WEIGHT_LAYOUT,
            _IN_FEATURES,
            _OUT_FEATURES,
        ),
    )

    source = Path(q5_f32.__file__).with_suffix(".hip").read_text()
    retained = _declaration(source, f"__global__ void {_H7G_KERNEL}(")
    assert _sha256(retained) == _H7G_KERNEL_SHA256
    assert source.count(f"__global__ void {_KERNEL}(") == 1
    assert source.count(f'extern "C" int {_SYMBOL}(') == 1
    start = source.index(f"__global__ void {_KERNEL}(")
    assert "__launch_bounds__(128, 1)" in source[max(0, start - 160) : start]
    candidate = _declaration(source, f"__global__ void {_KERNEL}(")
    assert "const float* __restrict__ weight_gate" in candidate
    assert "const float* __restrict__ weight_up" in candidate
    assert candidate.count("*reinterpret_cast<const uint2*>(activation_record)") == 1
    assert candidate.count("bf16_bits_to_float(input_bits)") == 1
    assert candidate.count("fmaf(input_value,") == 2
    assert "__shared__ float wave_sums[2][4][4][8];" in candidate
    assert candidate.count("store_output<out_t>(") == 2
    assert candidate.count("__syncthreads();") == 1


def test_h8c_strict_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primitive = _candidate()  # Intentional RED: no H8C function exists yet.
    load_attempts = 0

    def _unexpected_build(**_kwargs: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H8C input reached HIP loading")

    monkeypatch.setattr(
        q5_f32,
        "build_gguf_q5_k_f32_rocblas_prefill",
        _unexpected_build,
    )
    valid_ptrs = (0x1000, 0x2000, 0x3000, 0x4000, 0x5000)
    with pytest.raises(ValueError, match="pointers must be non-zero"):
        primitive(0, *valid_ptrs[1:], 17, _IN_FEATURES, _OUT_FEATURES)
    with pytest.raises(ValueError, match="weight planes must be distinct"):
        primitive(
            valid_ptrs[0],
            valid_ptrs[1],
            valid_ptrs[1],
            valid_ptrs[3],
            valid_ptrs[4],
            17,
            _IN_FEATURES,
            _OUT_FEATURES,
        )
    with pytest.raises(ValueError, match="output planes must be distinct"):
        primitive(
            valid_ptrs[0],
            valid_ptrs[1],
            valid_ptrs[2],
            valid_ptrs[3],
            valid_ptrs[3],
            17,
            _IN_FEATURES,
            _OUT_FEATURES,
        )
    for rows, hidden, outputs, message in (
        (0, _IN_FEATURES, _OUT_FEATURES, "rows must be positive"),
        (17, _IN_FEATURES - 256, _OUT_FEATURES, "exactly 3072"),
        (17, _IN_FEATURES, _OUT_FEATURES - 8, "exactly 1024"),
    ):
        with pytest.raises(ValueError, match=message):
            primitive(*valid_ptrs, rows, hidden, outputs)
    assert load_attempts == 0


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h8c_rows17_33_m512_gate_up_bytes_and_cpu_values_match_h7h() -> None:
    primitive = _candidate()  # Intentional RED before any build or allocation.
    from hipengine.core.hip import get_hip_runtime

    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = (
        Path(version_file).read_text(encoding="utf-8").strip()
        if version_file
        else None
    )
    library = q5_f32.build_gguf_q5_k_f32_rocblas_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )
    pack_name, _consumer_name, _composite_name = _h5y_names(
        _COL_TILE,
        _ROW_BATCH,
        _OUTPUT_DTYPE,
        _WEIGHT_LAYOUT,
    )
    pack = getattr(q5_f32, pack_name)
    producer = getattr(
        q5_f32,
        "gguf_q5_k_dequantize_f32_exact_tile_k_col_"
        "coltile8_rowbatch4_bf16_bf16_out",
    )
    control_name, _ = _h7h_names(
        _COL_TILE,
        _ROW_BATCH,
        _OUTPUT_DTYPE,
        _WEIGHT_LAYOUT,
    )
    control = getattr(q5_f32, control_name)

    gate_qweight = _production_q5_weight(_OUT_FEATURES, _IN_FEATURES)
    up_qweight = np.ascontiguousarray(np.roll(gate_qweight, 1, axis=0))
    runtime = get_hip_runtime()
    before = memory_stats()
    for rows in _ROWS:
        rng = np.random.default_rng(20260803 + 97 * rows)
        x_bf16 = _bf16_bits(
            rng.normal(0.0, 0.2, size=(rows, _IN_FEATURES)).astype(np.float32)
        )
        plane_shape = _activation_plane_shape(rows, _IN_FEATURES, _ROW_BATCH)
        activation = np.empty(plane_shape, dtype=np.uint16)
        expected_activation = _expected_activation_plane(x_bf16, _ROW_BATCH)
        expected_gate = np.empty((rows, _OUT_FEATURES), dtype=np.uint16)
        expected_up = np.empty_like(expected_gate)
        actual_gate = np.empty_like(expected_gate)
        actual_up = np.empty_like(expected_gate)
        buffers = []
        try:
            x_dev = _device(x_bf16, runtime)
            gate_qweight_dev = _device(gate_qweight, runtime)
            up_qweight_dev = _device(up_qweight, runtime)
            activation_dev = malloc(activation.nbytes, runtime=runtime)
            gate_plane_dev = malloc(_PLANE_NBYTES, runtime=runtime)
            up_plane_dev = malloc(_PLANE_NBYTES, runtime=runtime)
            expected_gate_dev = malloc(expected_gate.nbytes, runtime=runtime)
            expected_up_dev = malloc(expected_up.nbytes, runtime=runtime)
            actual_gate_dev = malloc(actual_gate.nbytes, runtime=runtime)
            actual_up_dev = malloc(actual_up.nbytes, runtime=runtime)
            buffers.extend(
                (
                    x_dev,
                    gate_qweight_dev,
                    up_qweight_dev,
                    activation_dev,
                    gate_plane_dev,
                    up_plane_dev,
                    expected_gate_dev,
                    expected_up_dev,
                    actual_gate_dev,
                    actual_up_dev,
                )
            )

            pack(
                x_dev.ptr,
                activation_dev.ptr,
                rows,
                _IN_FEATURES,
                library=library,
                runtime=runtime,
            )
            producer(
                gate_qweight_dev.ptr,
                gate_plane_dev.ptr,
                _IN_FEATURES,
                _OUT_FEATURES,
                library=library,
                runtime=runtime,
            )
            producer(
                up_qweight_dev.ptr,
                up_plane_dev.ptr,
                _IN_FEATURES,
                _OUT_FEATURES,
                library=library,
                runtime=runtime,
            )
            control(
                activation_dev.ptr,
                gate_plane_dev.ptr,
                expected_gate_dev.ptr,
                rows,
                _IN_FEATURES,
                _OUT_FEATURES,
                library=library,
                runtime=runtime,
            )
            control(
                activation_dev.ptr,
                up_plane_dev.ptr,
                expected_up_dev.ptr,
                rows,
                _IN_FEATURES,
                _OUT_FEATURES,
                library=library,
                runtime=runtime,
            )
            primitive(
                activation_dev.ptr,
                gate_plane_dev.ptr,
                up_plane_dev.ptr,
                actual_gate_dev.ptr,
                actual_up_dev.ptr,
                rows,
                _IN_FEATURES,
                _OUT_FEATURES,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (activation, activation_dev),
                (expected_gate, expected_gate_dev),
                (expected_up, expected_up_dev),
                (actual_gate, actual_gate_dev),
                (actual_up, actual_up_dev),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_array_equal(activation, expected_activation)
            np.testing.assert_array_equal(actual_gate, expected_gate)
            np.testing.assert_array_equal(actual_up, expected_up)
            assert not np.array_equal(expected_gate, expected_up)
            assert np.isfinite(_bf16_to_f32(actual_gate)).all()
            assert np.isfinite(_bf16_to_f32(actual_up)).all()
            _sampled_cpu_gate(
                actual_gate,
                x_bf16,
                gate_qweight,
                row_batch=_ROW_BATCH,
                output_dtype=_OUTPUT_DTYPE,
                out_features=_OUT_FEATURES,
            )
            _sampled_cpu_gate(
                actual_up,
                x_bf16,
                up_qweight,
                row_batch=_ROW_BATCH,
                output_dtype=_OUTPUT_DTYPE,
                out_features=_OUT_FEATURES,
            )
        finally:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


def test_h8c_default_off_complete_runtime_owner_and_no_growth() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.runtime import gguf_linear
    from hipengine.runtime import laguna_gguf_runner as runner
    from hipengine.runtime import laguna_moe

    assert getattr(hip_gfx1100, _SUPPORTED_CAPABILITY) is True
    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) is False
    assert not hasattr(hip_gfx1151, _SUPPORTED_CAPABILITY)
    assert not hasattr(hip_gfx1151, _SOURCE_CAPABILITY)

    resolver = getattr(runner, "resolve_laguna_shared_q5_dual_weight")
    assert resolver("hip_gfx1100", None) is False
    assert resolver("hip_gfx1100", False) is False
    assert resolver("hip_gfx1100", True) is True
    assert resolver("hip_gfx1151", None) is False
    assert resolver("hip_gfx1151", False) is False
    with pytest.raises(ValueError, match="not supported"):
        resolver("hip_gfx1151", True)

    parameters = inspect.signature(runner.LagunaGGUFResidentSession.__init__).parameters
    assert _SESSION_PARAMETER in parameters
    init_source = inspect.getsource(runner.LagunaGGUFResidentSession.__init__)
    assert "resolve_laguna_shared_q5_dual_weight(" in init_source
    assert "self.use_shared_q5_dual_weight" in init_source

    pair_source = inspect.getsource(gguf_linear.launch_gguf_linear_pair)
    shared_source = inspect.getsource(laguna_moe._launch_laguna_shared_rows)
    sparse_source = inspect.getsource(
        runner.LagunaGGUFResidentSession._run_sparse_ffn_rows
    )
    run_source = inspect.getsource(laguna_moe.run_laguna_moe_rows)
    assert "_q5_f32_ordered_prefill_session" in pair_source
    assert "activation_bf16_ptr" in pair_source
    assert "linear_pair" in pair_source
    assert "use_shared_q5_dual_weight" in shared_source
    assert "launch_gguf_linear_pair(" in shared_source
    assert shared_source.count("launch_gguf_linear(") == 3
    assert "use_shared_q5_dual_weight: bool = False" in run_source
    assert "use_shared_q5_dual_weight=self.use_shared_q5_dual_weight" in sparse_source
    combined = "\n".join((pair_source, shared_source, sparse_source, run_source))
    for forbidden in (
        "layer_id in",
        "layer_id ==",
        "next_token_id",
        "position == 511",
        "rows == 512",
        "tokens == 512",
    ):
        assert forbidden not in combined

    assert _GROUPS == 46
    assert _GROUPS * 2 == 92
    assert 2_155 - _GROUPS == 2_109
    assert 2 * _PLANE_NBYTES == 25_165_824
    assert (
        2 * _PLANE_NBYTES
        <= LagunaQ5F32OrderedScratch.weight_f32_planned_nbytes()
        == 150_994_944
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256
