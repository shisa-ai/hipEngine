"""RED contracts for WPF-H7I exact raw-Q6 full-group compute."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch
from tests.test_laguna_h7c_raw_q6_dpp_wave_reduction import (
    _bf16_bits,
    _declaration,
    _hip_available,
    _q6_weight,
    _run,
    _sample_columns,
    _sampled_cpu_gate,
)
from tests.test_gguf_q5_k_f32_rocblas_prefill import _exact_q6_f32_cpu

_ROOT = Path(__file__).parents[1]
_ARTIFACT = (
    _ROOT
    / "benchmarks/results/"
    "2026-08-02-gfx1100-laguna-q2-xl-post-h7h-matched-raw-q6-full-group-target.json"
)
_ARTIFACT_SHA256 = (
    "c3b4d6b08c11dd8dde8f1178552ea450a1b54cef8015220e6a2dee9bc2424856"
)
# role, col tile, row batch, output dtype, exact K, exact N
# All three roles are inseparable. No post-timing role subset is admissible.
_ROLES = (
    ("layer-0-dense-ffn-down", 4, 8, "bf16", 12_288, 3_072),
    ("layer-47-attention-q", 2, 16, "f32", 3_072, 9_216),
    ("layer-47-attention-output", 4, 8, "bf16", 9_216, 3_072),
)
_FALLBACK_ROWS = (1, 7, 8, 9)
_REJECTED_ROWS = (*_FALLBACK_ROWS, 511, 513)
_H7C_KERNEL = "gguf_q6_k_prefill_out_coltile_rowbatch_dpp_wave_reduction_kernel"
_H7I_KERNEL = (
    "gguf_q6_k_prefill_out_coltile_rowbatch_dpp_wave_reduction_"
    "full_group_compute_kernel"
)
_H7C_KERNEL_SHA256 = (
    "646770cb382d39b9078ce3a6073c1c722aec74fc2b551ee893dc562ee8f095b9"
)
_H7C_VALIDATE_SHA256 = (
    "909c047979c39e0f6b2cb048c198229d496bfc45289d7f08cf1012f65eeef363"
)
_H7C_LAUNCH_SHA256 = (
    "b57710003350204130409ce003f7204f061e699b284d5fab4bc6b373e19e7e04"
)
_H7C_WRAPPER_SHA256 = {
    "bf16": "f9b928774c2033f100e3e9c1c516f5cabc3193f204e0b21a8998464e5e70e69c",
    "f32": "a9b0ae6e9f228a3568298611840b21d219ac18c68ecd0d63d1f6c5fb5f857fcd",
}
_H7C_POLICY = {
    ("gguf_q6_k", "bf16_bf16_out", 512, 12_288, 3_072): (
        "dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out"
    ),
    ("gguf_q6_k", "bf16_f32_out", 512, 3_072, 9_216): (
        "dpp_wave_reduction_coltile2_rowbatch16_bf16_f32_out"
    ),
    ("gguf_q6_k", "bf16_bf16_out", 512, 9_216, 3_072): (
        "dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out"
    ),
}
_H7I_POLICY = {
    role: variant.replace(
        "dpp_wave_reduction_",
        "dpp_wave_reduction_full_group_compute_",
        1,
    )
    for role, variant in _H7C_POLICY.items()
}
_EXPECTED_SELECTION = {
    "revision": "4fd68fd6f+out-of-tree-h7i-selection-probe",
    "rows": 512,
    "warmups": 5,
    "counter_rotated_repetitions": 15,
    "launches_per_sample": 5,
    "production_invocations": 3,
    "harness_sha256": (
        "17f36f061432f8617cf1fa3556fa5f6cc82cd6c394208ef47e1938301ca4ebd6"
    ),
    "candidate_source_sha256": (
        "fcce137bd3daa2aa16c962c7d59223a72316f47c803cea71adb83dd057a57f7d"
    ),
    "raw_json_sha256": (
        "99c0f9385a7d35cc4cfc95128516247d03caaf8f29d778a35a1ddf831c9cee9e"
    ),
    "h7c_event_weighted_ms": 35.83960113525391,
    "h7i_event_weighted_ms": 20.322689819335938,
    "event_speedup": 1.7635264551031267,
    "h7c_wall_weighted_ms": 34.85432397574186,
    "h7i_wall_weighted_ms": 21.974391071125865,
    "wall_speedup": 1.5861337801319146,
    "all_roles_both_clock_positive": True,
    "all_candidate_bytes_exact": True,
    "first_and_only_timing_run": True,
    "subset_salvage_allowed": False,
}
_EXPECTED_PHYSICAL = {
    "bf16": {
        "control_code_bytes": 4_228,
        "candidate_code_bytes_max": 4_060,
        "control_instruction_slots": 681,
        "candidate_instruction_slots_max": 623,
        "control_row_comparisons": 9,
        "candidate_row_comparisons_max": 2,
        "control_dual_fmac": 1,
        "candidate_dual_fmac_min": 10,
        "control_scalar_fmac": 31,
        "candidate_scalar_fmac_max": 14,
        "control_vgpr": 60,
        "candidate_vgpr_max": 69,
        "control_sgpr": 50,
        "candidate_sgpr_max": 44,
        "grid_x": 98_304,
        "grid_y": 64,
    },
    "f32": {
        "control_code_bytes": 4_452,
        "candidate_code_bytes_max": 4_032,
        "control_instruction_slots": 749,
        "candidate_instruction_slots_max": 631,
        "control_row_comparisons": 17,
        "candidate_row_comparisons_max": 2,
        "control_dual_fmac": 1,
        "candidate_dual_fmac_min": 11,
        "control_scalar_fmac": 31,
        "candidate_scalar_fmac_max": 16,
        "control_vgpr": 55,
        "candidate_vgpr_max": 64,
        "control_sgpr": 69,
        "candidate_sgpr_max": 54,
        "grid_x": 589_824,
        "grid_y": 32,
    },
}
_COMMON_PHYSICAL = {
    "global_loads": 24,
    "global_stores": 1,
    "permlanex16": 32,
    "dpp_add": 128,
    "ordered_fma_operations": 32,
    "barriers": 1,
    "lds_bytes": 512,
    "private_bytes": 0,
    "vgpr_spills": 0,
    "sgpr_spills": 0,
    "runtime_scratch_bytes": 0,
    "local_size": 128,
    "metadata_vgpr_ceiling": 72,
}
_TRACE_CONTRACT = {
    "kernel_name": _H7I_KERNEL,
    "require_cached_build": True,
    "new_compiler_processes": 0,
    "local_size": 128,
    "positive_duration": True,
    "required_grids": ((98_304, 64), (589_824, 32)),
    "runtime_scratch_bytes": 0,
    "runtime_vgpr_ceiling": 72,
}


def _module():
    return importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode())


def _suffix(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    *,
    candidate: bool,
) -> str:
    prefix = (
        "dpp_wave_reduction_full_group_compute_"
        if candidate
        else "dpp_wave_reduction_"
    )
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


@pytest.fixture(scope="module")
def library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    return _module().build_gguf_k_gemv(load=True)


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


def test_h7i_frozen_target_artifact_selection_physical_and_trace_contract() -> None:
    artifact_bytes = _ARTIFACT.read_bytes()
    assert _sha256_bytes(artifact_bytes) == _ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["repo"]["measurement_revision"].startswith("4fd68fd6f")
    assert artifact["decision"]["h7i_target_selected"]
    assert not artifact["decision"]["candidate_implemented"]
    assert not artifact["decision"]["production_changed"]
    assert artifact["decision"]["performance_claim"] == "target_selection_only"
    assert artifact["target"]["all_three_roles_required"]
    assert not artifact["target"]["subset_salvage_allowed"]
    assert artifact["target"]["first_and_only_timing_run"]
    assert len(artifact["target"]["selection_rows"]) == len(_ROLES) == 3

    selection = _EXPECTED_SELECTION
    assert math.isclose(
        selection["h7c_event_weighted_ms"]
        / selection["h7i_event_weighted_ms"],
        selection["event_speedup"],
    )
    assert math.isclose(
        selection["h7c_wall_weighted_ms"]
        / selection["h7i_wall_weighted_ms"],
        selection["wall_speedup"],
    )
    assert selection["event_speedup"] > 1.0
    assert selection["wall_speedup"] > 1.0
    assert selection["all_roles_both_clock_positive"]
    assert selection["all_candidate_bytes_exact"]
    assert selection["first_and_only_timing_run"]
    assert not selection["subset_salvage_allowed"]
    assert artifact["target"]["candidate_source_sha256"] == (
        selection["candidate_source_sha256"]
    )
    assert artifact["raw_evidence_sha256"][
        "/tmp/laguna_wpfh7i_raw_q6_full_group_actual_role_screen.py"
    ] == selection["harness_sha256"]
    assert artifact["raw_evidence_sha256"][
        "/tmp/laguna-wpfh7i-raw-q6-full-group-actual-role-screen.json"
    ] == selection["raw_json_sha256"]
    assert artifact["target"]["aggregate"]["event_speedup"] == (
        selection["event_speedup"]
    )
    assert artifact["target"]["aggregate"]["wall_speedup"] == (
        selection["wall_speedup"]
    )

    assert set(_EXPECTED_PHYSICAL) == {"bf16", "f32"}
    for output_dtype, expected in _EXPECTED_PHYSICAL.items():
        control = artifact["physical"]["control"][output_dtype]
        candidate = artifact["physical"]["candidate"][output_dtype]
        assert control["code_bytes"] == expected["control_code_bytes"]
        assert candidate["code_bytes"] <= expected["candidate_code_bytes_max"]
        assert control["instruction_slots"] == (
            expected["control_instruction_slots"]
        )
        assert candidate["instruction_slots"] <= (
            expected["candidate_instruction_slots_max"]
        )
        assert control["scalar_row_comparisons"] == (
            expected["control_row_comparisons"]
        )
        assert candidate["scalar_row_comparisons"] <= (
            expected["candidate_row_comparisons_max"]
        )
        assert control["dual_fmac_sites"] == expected["control_dual_fmac"]
        assert candidate["dual_fmac_sites"] >= expected["candidate_dual_fmac_min"]
        assert control["scalar_fmac_sites"] == expected["control_scalar_fmac"]
        assert candidate["scalar_fmac_sites"] <= (
            expected["candidate_scalar_fmac_max"]
        )
        assert control["vgpr"] == expected["control_vgpr"]
        assert candidate["vgpr"] <= expected["candidate_vgpr_max"]
        assert control["sgpr"] == expected["control_sgpr"]
        assert candidate["sgpr"] <= expected["candidate_sgpr_max"]
        for field, value in _COMMON_PHYSICAL.items():
            if field in {"runtime_scratch_bytes", "local_size", "metadata_vgpr_ceiling"}:
                continue
            assert control[field] == candidate[field] == value
    assert _TRACE_CONTRACT["kernel_name"] == _H7I_KERNEL
    assert _TRACE_CONTRACT["required_grids"] == tuple(
        (expected["grid_x"], expected["grid_y"])
        for expected in _EXPECTED_PHYSICAL.values()
    )
    assert _TRACE_CONTRACT["runtime_vgpr_ceiling"] == 72
    assert _TRACE_CONTRACT["runtime_scratch_bytes"] == 0


@pytest.mark.parametrize(
    ("role", "col_tile", "row_batch", "output_dtype", "in_features", "out_features"),
    _ROLES,
    ids=tuple(role[0] for role in _ROLES),
)
def test_h7i_source_registry_package_workspace_and_backend_isolation(
    role: str,
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    in_features: int,
    out_features: int,
) -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.runtime.gguf_linear import (
        GGUFLinearDispatch,
        _raw_k_prefill_rowbatch_dispatch,
    )

    del role
    module = _module()
    module.register_gguf_k_gemv_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_GENERIC_ROLE_VARIANTS == {}
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_H7C_ROLE_VARIANTS == _H7C_POLICY
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_ROLE_VARIANTS == _H7I_POLICY
    assert hip_gfx1100.GGUF_RAW_K_PREFILL_H7I_ROLE_VARIANTS == _H7I_POLICY
    assert not hasattr(hip_gfx1151, "GGUF_RAW_K_PREFILL_H7I_ROLE_VARIANTS")
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256

    source = Path(module.__file__).with_suffix(".hip").read_text()
    h7c_kernel = _declaration(source, f"__global__ void {_H7C_KERNEL}(")
    assert _sha256_text(h7c_kernel) == _H7C_KERNEL_SHA256
    assert _sha256_text(inspect.getsource(module._validate_h7c_raw_q6_role)) == (
        _H7C_VALIDATE_SHA256
    )
    assert _sha256_text(inspect.getsource(module._launch_h7c_raw_q6)) == (
        _H7C_LAUNCH_SHA256
    )
    assert _sha256_text(inspect.getsource(_control(col_tile, row_batch, output_dtype))) == (
        _H7C_WRAPPER_SHA256[output_dtype]
    )

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
    assert selected == GGUFLinearDispatch(
        _key(
            col_tile,
            row_batch,
            output_dtype,
            candidate=True,
        ),
        "raw",
    )
    for fallback_rows in _REJECTED_ROWS:
        fallback = _raw_k_prefill_rowbatch_dispatch(
            base,
            rows=fallback_rows,
            in_features=in_features,
            out_features=out_features,
            row_batch=32,
            variant="coltile",
        )
        assert "full_group_compute" not in fallback.key.variant
        assert fallback.abi == "raw"

    # H7I source selection preserves H7C bytes/wrappers, strict fallback,
    # workspace, and backend controls.
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
    assert source.count("hipengine_" + candidate.__name__) == 1
    assert source.count(_H7I_KERNEL) == 2
    assert selected.key.variant == _H7I_POLICY[
        ("gguf_q6_k", output_variant, 512, in_features, out_features)
    ]


@pytest.mark.parametrize(
    ("role", "col_tile", "row_batch", "output_dtype", "in_features", "out_features"),
    _ROLES,
    ids=tuple(role[0] for role in _ROLES),
)
def test_h7i_strict_m512_full_group_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    in_features: int,
    out_features: int,
) -> None:
    del role, row_batch
    module = _module()
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("valid H7I role reached HIP loader")

    monkeypatch.setattr(module, "build_gguf_k_gemv", fail_if_loaded)
    control = _control(col_tile, 32 // col_tile, output_dtype)
    with pytest.raises(ValueError, match="rows must be positive"):
        control(1, 2, 3, 0, in_features, out_features)
    assert load_attempts == 0

    # Intentional RED only after the retained control rejects before loading.
    candidate = _candidate(col_tile, 32 // col_tile, output_dtype)
    for rows in _REJECTED_ROWS:
        with pytest.raises(ValueError, match="rows must be exactly 512"):
            candidate(1, 2, 3, rows, in_features, out_features)
    if output_dtype == "bf16":
        invalid = (
            (512, 3_072, out_features, "exactly 9216 or 12288"),
            (512, in_features, out_features - 4, "exactly 3072"),
        )
    else:
        invalid = (
            (512, in_features + 256, out_features, "exactly 3072"),
            (512, in_features, out_features - 2, "exactly 9216"),
        )
    for rows, hidden, outputs, message in invalid:
        with pytest.raises(ValueError, match=message):
            candidate(1, 2, 3, rows, hidden, outputs)
    with pytest.raises(ValueError, match="threads must be exactly 128"):
        candidate(
            1,
            2,
            3,
            512,
            in_features,
            out_features,
            threads=64,
        )
    assert load_attempts == 0
    with pytest.raises(AssertionError, match="valid H7I role reached HIP loader"):
        candidate(1, 2, 3, 512, in_features, out_features)
    assert load_attempts == 1


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows",
    _FALLBACK_ROWS,
    ids=tuple(f"rows{rows}" for rows in _FALLBACK_ROWS),
)
@pytest.mark.parametrize(
    ("role", "col_tile", "row_batch", "output_dtype", "in_features", "out_features"),
    _ROLES,
    ids=tuple(role[0] for role in _ROLES),
)
def test_h7i_short_rows_retain_complete_h7c_and_sampled_cpu_fallback(
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
        20260804 + 17 * rows + 3 * in_features + out_features
    )
    x_bf16 = _bf16_bits(
        rng.uniform(
            0.0078125,
            0.03125,
            size=(rows, in_features),
        ).astype(np.float32)
    )
    qweight = qweights[(in_features, out_features)]
    expected = _run(
        _control(col_tile, row_batch, output_dtype),
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
    assert expected.shape == (rows, out_features), role


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("role", "col_tile", "row_batch", "output_dtype", "in_features", "out_features"),
    _ROLES,
    ids=tuple(role[0] for role in _ROLES),
)
def test_h7i_m512_complete_outputs_match_h7c_and_sampled_cpu(
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
    rows = 512
    rng = np.random.default_rng(20260804 + 3 * in_features + out_features)
    x_bf16 = _bf16_bits(
        rng.uniform(
            0.0078125,
            0.03125,
            size=(rows, in_features),
        ).astype(np.float32)
    )
    qweight = qweights[(in_features, out_features)]
    expected = _run(
        _control(col_tile, row_batch, output_dtype),
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

    # Intentional RED only after complete poisoned H7C bytes, independent CPU
    # values, finiteness, and tracked lifecycle pass for natural M512.
    actual = _run(
        _candidate(col_tile, row_batch, output_dtype),
        library=library,
        x_bf16=x_bf16,
        qweight=qweight,
        output_dtype=output_dtype,
    )
    np.testing.assert_array_equal(actual, expected, err_msg=role)
