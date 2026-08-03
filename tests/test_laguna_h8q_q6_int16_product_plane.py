"""WPF-H8Q exact Q6 int16-product plus tiled-F32-scale RED contract."""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import math
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
    _exact_q6_f32_cpu,
)
from tests.test_laguna_h6e_q6_activation_tile_k_row import (
    _Q6_PRODUCTION_POLICY,
    _activation_plane_shape,
    _expected_activation_plane,
    _sampled_cpu_gate,
    _suffix,
)

_ROOT = Path(__file__).resolve().parents[1]
_TARGET_ARTIFACT = _ROOT / (
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8p-q6-int16-product-plane-target.json"
)
_TARGET_ARTIFACT_SHA256 = "f6f616f985b08dd9f7c52fb28d8cb253ef1d3ed2da93a19dc7cfad4639387a63"
_QK_K = 256
_Q6_BLOCK_BYTES = 210
_COL_TILE = 16

# Role, output dtype, exact K, exact N, row batch, production calls,
# exact runtime LDS bytes, runtime VGPR ceiling.
_ROLES = (
    ("bf16-k3072-n1024", "bf16", 3_072, 1_024, 5, 2, 1_536, 160),
    ("bf16-k1024-n3072", "bf16", 1_024, 3_072, 4, 46, 1_024, 128),
    ("f32-k3072-n1024", "f32", 3_072, 1_024, 5, 94, 1_536, 160),
)
_ROLE_KEYS = tuple((dtype, hidden, outputs) for _, dtype, hidden, outputs, *_ in _ROLES)
_CALL_WEIGHTS = {role: calls for role, _, _, _, _, calls, _, _ in _ROLES}

_H8Q_CAPABILITY = "GGUF_Q6_INT16_PRODUCT_F32_SCALE_PREFILL_H8Q_POLICY"
_H8Q_POLICY = {
    ("bf16", 3_072, 1_024): (
        "int16_product_f32_scale_weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch5"
    ),
    ("bf16", 1_024, 3_072): (
        "int16_product_f32_scale_weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch4"
    ),
    # The one H5I N72 route is part of the complete fallback map, not H8Q.
    ("f32", 3_072, 72): "coltile8_rowbatch4",
    ("f32", 3_072, 1_024): (
        "int16_product_f32_scale_weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch5"
    ),
}

_PRODUCER_NAME = "gguf_q6_k_expand_int16_product_f32_scale_exact"
_PRODUCER_KERNEL = "gguf_q6_k_expand_int16_product_f32_scale_exact_kernel"
_CONSUMER_KERNEL = (
    "gguf_q6_k_int16_product_f32_scale_weight_ordered_weight_major_"
    "row_major_activation_tile_k_row_dpp_wave_reduction_kernel"
)
_SCALE_BROADCAST_HELPER = "h8q_readfirstlane_f32"
_PRODUCER_VARIANT = "raw_int16_product_f32_scale_exact_local64"
_PLANE_QUANT = "q6_int16_product_f32_scale"

_RETAINED_H6U_KERNEL = (
    "gguf_q6_k_f32_weight_ordered_weight_major_row_major_"
    "activation_tile_k_row_dpp_wave_reduction_kernel"
)
_RETAINED_H6U_KERNEL_SHA256 = "5bd38a734cf4169b0ab141aa3a942ed36cb654279a94517ccb18fb915ffba226"
_RETAINED_Q6_PRODUCER_KERNEL_SHA256 = (
    "acab7bab84b5933c7c021ccd707a6cd553fc3b24ef27b4c4a905e0b9c3e74d01"
)
_RETAINED_Q6_PRODUCER_WRAPPER_SHA256 = (
    "4a653d5395a8c287aa9d76d1aa75fbd2954e3f46336cd7e645035df0be84cdb1"
)
_RETAINED_H6U_PRIMITIVE_WRAPPER_SHA256 = (
    "ac756f99aff7492b52ca5eb386c1e5eb9e1d30829b4ada71f307ba065d79e364"
)
_RETAINED_H6U_COMPOSITE_WRAPPER_SHA256 = (
    "f320fe2db79f40cda0fa8f8558c441a639381fca4bc36e41568cb29916046393"
)

_PHYSICAL_CONTRACT = {
    "producer": {
        "local_size": 64,
        "metadata_vgpr_max": 24,
        "runtime_vgpr_max": 24,
        "lds_bytes": 0,
        "private_bytes": 0,
        "vgpr_spills": 0,
        "sgpr_spills": 0,
        "runtime_scratch_bytes": 0,
    },
    "consumers": {
        role: {
            "local_size": 128,
            "runtime_lds_bytes": lds_bytes,
            "runtime_vgpr_max": runtime_vgpr,
            "private_bytes": 0,
            "vgpr_spills": 0,
            "sgpr_spills": 0,
            "runtime_scratch_bytes": 0,
        }
        for role, _, _, _, _, _, lds_bytes, runtime_vgpr in _ROLES
    },
    "scale_load_path": "scalar_or_readfirstlane_uniform",
    "reconstruction": "one_int16_to_f32_plus_one_f32_multiply_before_fmaf",
}
_TRACE_CONTRACT = {
    "require_cached_build": True,
    "new_compiler_processes": 0,
    "queues": 1,
    "streams": 1,
    "positive_duration": True,
    "producer": {
        "kernel_name": _PRODUCER_KERNEL,
        "local_size": 64,
        "grid_x": 12_288,
    },
    "consumers": {
        "bf16-k3072-n1024": {"local_size": 128, "grid_x": 6_592},
        "bf16-k1024-n3072": {"local_size": 128, "grid_x": 24_576},
        "f32-k3072-n1024": {"local_size": 128, "grid_x": 6_592},
    },
}
_TIMING_CONTRACT = {
    "rows": 512,
    "warmups": 5,
    "repetitions": 15,
    "launches_per_sample": 5,
    "order": "counter_rotated",
    "clocks": ("hip_event", "synchronized_wall"),
    "producer_inclusive": True,
    "role_weights": dict(_CALL_WEIGHTS),
    "aggregate": "weighted-142-call",
    "require_each_role_both_clocks": True,
    "require_weighted_aggregate_both_clocks": True,
    "first_and_only_screen": True,
}
_NO_SALVAGE = (
    "role",
    "dtype",
    "shape",
    "layer",
    "length",
    "layout",
    "scale-dtype",
    "grouping",
    "resource",
    "recompile",
    "favorable-rerun",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode())


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


def _primitive_name(output_dtype: str, row_batch: int) -> str:
    return (
        "gguf_q6_k_int16_product_f32_scale_weight_ordered_weight_major_"
        "row_major_activation_tile_k_row_dpp_wave_reduction_"
        + _suffix(_COL_TILE, row_batch, output_dtype)
    )


def _composite_name(output_dtype: str, row_batch: int) -> str:
    return (
        "gguf_q6_k_f32_ordered_int16_product_f32_scale_weight_major_"
        "row_major_activation_tile_k_row_dpp_wave_reduction_"
        + _suffix(_COL_TILE, row_batch, output_dtype)
    )


def _primitive_key(
    output_dtype: str,
    row_batch: int,
    *,
    backend: str = "hip_gfx1100",
) -> KernelKey:
    return KernelKey(
        backend,
        "linear",
        _PLANE_QUANT,
        "ordered_weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_" + _suffix(_COL_TILE, row_batch, output_dtype),
    )


def _composite_key(
    output_dtype: str,
    row_batch: int,
    *,
    backend: str = "hip_gfx1100",
) -> KernelKey:
    return KernelKey(
        backend,
        "linear",
        "gguf_q6_k",
        "f32_ordered_int16_product_f32_scale_weight_major_row_major_"
        "activation_tile_k_row_dpp_wave_reduction_" + _suffix(_COL_TILE, row_batch, output_dtype),
    )


def _retained_h6u_names(output_dtype: str, row_batch: int) -> tuple[str, str]:
    stem = "weight_major_row_major_activation_tile_k_row_dpp_wave_reduction_" + _suffix(
        _COL_TILE, row_batch, output_dtype
    )
    return (
        "gguf_q5_k_f32_weight_ordered_" + stem,
        "gguf_q6_k_f32_ordered_" + stem,
    )


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _allocation_lifecycle() -> tuple[int, int]:
    stats = memory_stats()
    return stats["current_allocated_bytes"], stats["active_allocations"]


def _expected_product_scale_planes(
    qweight: np.ndarray,
    in_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw Q6 independently into the two proposed exact H8Q planes."""

    raw = np.ascontiguousarray(qweight, dtype=np.uint8)
    blocks_per_row = in_features // _QK_K
    if in_features <= 0 or in_features % _QK_K:
        raise ValueError("in_features must be a positive multiple of 256")
    if raw.ndim != 2 or raw.shape[1] != blocks_per_row * _Q6_BLOCK_BYTES:
        raise ValueError("invalid Q6 fixture bytes")
    out_features = raw.shape[0]
    if out_features <= 0 or out_features % _COL_TILE:
        raise ValueError("out_features must be a positive multiple of 16")

    products = np.empty((out_features, in_features), dtype=np.int16)
    tiled_scales = np.empty(
        (out_features // _COL_TILE, blocks_per_row, _COL_TILE),
        dtype=np.float32,
    )
    for out_col in range(out_features):
        for qblock in range(blocks_per_row):
            start = qblock * _Q6_BLOCK_BYTES
            block = raw[out_col, start : start + _Q6_BLOCK_BYTES]
            ql = block[:128]
            qh = block[128:192]
            scales = block[192:208].view(np.int8)
            tiled_scales[out_col // _COL_TILE, qblock, out_col % _COL_TILE] = np.float32(
                block[208:210].view(np.float16)[0]
            )
            for within in range(_QK_K):
                group32 = within >> 5
                lane = within & 31
                base64 = 64 if group32 >= 4 else 0
                ql_byte = ql[base64 + (group32 & 1) * 32 + lane]
                low = ql_byte & np.uint8(0x0F) if (group32 & 2) == 0 else ql_byte >> np.uint8(4)
                high = (
                    qh[(32 if group32 >= 4 else 0) + lane] >> np.uint8(2 * (group32 & 3))
                ) & np.uint8(3)
                quant = (int(low) | (int(high) << 4)) - 32
                products[out_col, qblock * _QK_K + within] = int(scales[within >> 4]) * quant
    return products, tiled_scales


def _reconstruct_f32(
    products: np.ndarray,
    tiled_scales: np.ndarray,
) -> np.ndarray:
    out_features, in_features = products.shape
    blocks_per_row = in_features // _QK_K
    out = np.empty_like(products, dtype=np.float32)
    for out_col in range(out_features):
        scales = tiled_scales[out_col // _COL_TILE, :, out_col % _COL_TILE]
        out[out_col] = np.asarray(products[out_col], dtype=np.float32) * np.repeat(
            scales,
            _QK_K,
        )
    assert tiled_scales.shape == (
        out_features // _COL_TILE,
        blocks_per_row,
        _COL_TILE,
    )
    return out


def _edge_product_qweight() -> np.ndarray:
    raw = _edge_q6_weight(out_features=16, in_features=512)
    # block0/within0: scale=-128, quant=-32 => +4096.
    raw[0, 0] &= np.uint8(0xF0)
    raw[0, 128] &= np.uint8(0xFC)
    raw[0, 192] = np.asarray([-128], dtype=np.int8).view(np.uint8)[0]
    # block0/within16: scale=+127, quant=-32 => -4064.
    raw[0, 16] &= np.uint8(0xF0)
    raw[0, 128 + 16] &= np.uint8(0xFC)
    raw[0, 192 + 1] = np.asarray([127], dtype=np.int8).view(np.uint8)[0]
    return raw


def _screen_admitted(measurements: dict[str, dict[str, Any]]) -> bool:
    required = (*_CALL_WEIGHTS, "weighted-142-call")
    if tuple(measurements) != required:
        return False
    for name in required:
        row = measurements[name]
        if row.get("producer_inclusive") is not True:
            return False
        for clock in _TIMING_CONTRACT["clocks"]:
            clock_row = row.get(clock)
            if not isinstance(clock_row, dict):
                return False
            control = float(clock_row.get("control_ms", math.nan))
            candidate = float(clock_row.get("candidate_ms", math.nan))
            if not (math.isfinite(control) and math.isfinite(candidate)):
                return False
            if not candidate < control:
                return False
    return True


_H8Q_PRESENT = hasattr(q5_f32, _PRODUCER_NAME)
_H8Q_REQUIRED = pytest.mark.skipif(
    not _H8Q_PRESENT,
    reason="H8Q RED: candidate producer is intentionally absent",
)


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
        for _, _, in_features, out_features, *_ in _ROLES
    }


def test_h8q_frozen_target_format_physical_trace_timing_and_rejection_contract() -> None:
    artifact_bytes = _TARGET_ARTIFACT.read_bytes()
    assert _sha256_bytes(artifact_bytes) == _TARGET_ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["target_id"] == "WPF-H8Q"
    assert artifact["status"] == "target_only_no_candidate"
    assert artifact["decision"]["production_changed"] is False
    assert artifact["lineage"]["red_test_added"] is False
    assert artifact["lineage"]["candidate_source_added"] is False
    assert artifact["lineage"]["gpu_candidate_executed"] is False
    assert artifact["lineage"]["performance_result_exists"] is False
    assert artifact["complete_q6_product_audit"]["product_min"] == -4_064
    assert artifact["complete_q6_product_audit"]["product_max"] == 4_096
    assert artifact["complete_q6_product_audit"]["all_products_fit_int16"]

    record = artifact["traffic_model"]["record"]
    assert record["candidate_int16_product_bytes_per_q6_block"] == 512
    assert record["candidate_f32_scale_bytes_per_q6_block"] == 4
    assert record["candidate_bytes_per_q6_block"] == 516
    assert record["current_bytes_per_q6_block"] == 1_024
    maximum = artifact["traffic_model"]["maximum_role_workspace_bytes"]
    assert maximum == {
        "candidate_int16_products": 6_291_456,
        "candidate_tiled_f32_scales": 49_152,
        "candidate_total": 6_340_608,
        "current_f32_plane": 12_582_912,
        "shared_owner_unchanged": 161_120_256,
    }
    assert sum(_CALL_WEIGHTS.values()) == 142
    assert artifact["traffic_model"]["totals"]["candidate_reconstructed_weights"] == (
        49_627_004_928
    )
    assert artifact["traffic_model"]["totals"]["unchanged_useful_fmas"] == (228_707_008_512)
    assert artifact["traffic_only_ceiling"]["not_a_performance_claim"] is True

    assert _PHYSICAL_CONTRACT["producer"] == {
        "local_size": 64,
        "metadata_vgpr_max": 24,
        "runtime_vgpr_max": 24,
        "lds_bytes": 0,
        "private_bytes": 0,
        "vgpr_spills": 0,
        "sgpr_spills": 0,
        "runtime_scratch_bytes": 0,
    }
    assert tuple(
        contract["runtime_lds_bytes"] for contract in _PHYSICAL_CONTRACT["consumers"].values()
    ) == (1_536, 1_024, 1_536)
    assert tuple(
        contract["runtime_vgpr_max"] for contract in _PHYSICAL_CONTRACT["consumers"].values()
    ) == (160, 128, 160)
    assert _TRACE_CONTRACT["producer"] == {
        "kernel_name": _PRODUCER_KERNEL,
        "local_size": 64,
        "grid_x": 12_288,
    }
    assert tuple(row["grid_x"] for row in _TRACE_CONTRACT["consumers"].values()) == (
        6_592,
        24_576,
        6_592,
    )
    assert _TIMING_CONTRACT == {
        "rows": 512,
        "warmups": 5,
        "repetitions": 15,
        "launches_per_sample": 5,
        "order": "counter_rotated",
        "clocks": ("hip_event", "synchronized_wall"),
        "producer_inclusive": True,
        "role_weights": {"bf16-k3072-n1024": 2, "bf16-k1024-n3072": 46, "f32-k3072-n1024": 94},
        "aggregate": "weighted-142-call",
        "require_each_role_both_clocks": True,
        "require_weighted_aggregate_both_clocks": True,
        "first_and_only_screen": True,
    }
    assert _NO_SALVAGE == (
        "role",
        "dtype",
        "shape",
        "layer",
        "length",
        "layout",
        "scale-dtype",
        "grouping",
        "resource",
        "recompile",
        "favorable-rerun",
    )


def test_h8q_independent_cpu_planes_cover_both_format_extrema_exactly() -> None:
    raw = _edge_product_qweight()
    products, tiled_scales = _expected_product_scale_planes(raw, 512)
    assert products.dtype == np.int16
    assert tiled_scales.dtype == np.float32
    assert products.shape == (16, 512)
    assert tiled_scales.shape == (1, 2, 16)
    assert int(products[0, 0]) == 4_096
    assert int(products[0, 16]) == -4_064
    assert int(products.min()) == -4_064
    assert int(products.max()) == 4_096
    reconstructed = _reconstruct_f32(products, tiled_scales)
    expected = _exact_q6_f32_cpu(raw, 512)
    np.testing.assert_array_equal(reconstructed.view(np.uint32), expected.view(np.uint32))


def test_h8q_source_registry_workspace_policy_and_fallback_contract() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == _Q6_PRODUCTION_POLICY
    assert set(_H8Q_POLICY) == set(_Q6_PRODUCTION_POLICY)
    assert {key for key in _H8Q_POLICY if _H8Q_POLICY[key] != _Q6_PRODUCTION_POLICY[key]} == set(
        _ROLE_KEYS
    )
    assert _H8Q_POLICY[("f32", 3_072, 72)] == "coltile8_rowbatch4"
    assert not hasattr(hip_gfx1151, _H8Q_CAPABILITY)
    assert (
        LagunaQ5F32OrderedScratch.planned_nbytes(
            max_rows=512,
            use_activation_tile_k_row=True,
        )
        == 161_120_256
    )

    source = Path(q5_f32.__file__).with_suffix(".hip").read_text()
    retained_h6u = _declaration(source, f"__global__ void {_RETAINED_H6U_KERNEL}(")
    retained_q6_producer = _declaration(
        source,
        "__global__ void gguf_q6_k_dequantize_f32_exact_kernel(",
    )
    assert _sha256_text(retained_h6u) == _RETAINED_H6U_KERNEL_SHA256
    assert _sha256_text(retained_q6_producer) == _RETAINED_Q6_PRODUCER_KERNEL_SHA256
    assert _sha256_text(inspect.getsource(q5_f32.gguf_q6_k_dequantize_f32_exact)) == (
        _RETAINED_Q6_PRODUCER_WRAPPER_SHA256
    )
    for _, output_dtype, _, _, row_batch, _, _, _ in _ROLES:
        primitive_name, composite_name = _retained_h6u_names(
            output_dtype,
            row_batch,
        )
        assert _sha256_text(inspect.getsource(getattr(q5_f32, primitive_name))) == (
            _RETAINED_H6U_PRIMITIVE_WRAPPER_SHA256
        )
        assert _sha256_text(inspect.getsource(getattr(q5_f32, composite_name))) == (
            _RETAINED_H6U_COMPOSITE_WRAPPER_SHA256
        )

    # Intentional RED after target/oracle/current-production controls: one new
    # producer and its complete three-role family must appear together.
    assert _H8Q_PRESENT, "H8Q producer and three-consumer family are absent"

    assert getattr(hip_gfx1100, _H8Q_CAPABILITY) == _H8Q_POLICY
    producer = getattr(q5_f32, _PRODUCER_NAME)
    producer_key = KernelKey(
        "hip_gfx1100",
        "dequant",
        "gguf_q6_k",
        _PRODUCER_VARIANT,
    )
    assert (
        resolve(
            backend=producer_key.backend,
            layer=producer_key.layer,
            quant=producer_key.quant,
            variant=producer_key.variant,
        )
        is producer
    )
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            producer_key.layer,
            producer_key.quant,
            producer_key.variant,
        )
    )

    for _, output_dtype, _, _, row_batch, _, _, _ in _ROLES:
        primitive = getattr(q5_f32, _primitive_name(output_dtype, row_batch))
        composite = getattr(q5_f32, _composite_name(output_dtype, row_batch))
        for key, function in (
            (_primitive_key(output_dtype, row_batch), primitive),
            (_composite_key(output_dtype, row_batch), composite),
        ):
            assert (
                resolve(
                    backend=key.backend,
                    layer=key.layer,
                    quant=key.quant,
                    variant=key.variant,
                )
                is function
            )
            assert not is_registered(KernelKey("hip_gfx1151", key.layer, key.quant, key.variant))

    assert q5_f32._Q6_INT16_PRODUCT_F32_SCALE_ROLES == tuple(
        (_COL_TILE, row_batch, output_dtype, in_features, out_features)
        for _, output_dtype, in_features, out_features, row_batch, *_ in _ROLES
    )
    for in_features, out_features in ((3_072, 1_024), (1_024, 3_072)):
        product_nbytes = q5_f32.q6_k_int16_product_plane_nbytes(
            in_features,
            out_features,
        )
        scale_nbytes = q5_f32.q6_k_tiled_f32_scale_plane_nbytes(
            in_features,
            out_features,
        )
        total_nbytes = q5_f32.q6_k_int16_product_f32_scale_workspace_nbytes(
            in_features,
            out_features,
        )
        assert product_nbytes == 6_291_456
        assert scale_nbytes == 49_152
        assert total_nbytes == product_nbytes + scale_nbytes == 6_340_608
        assert total_nbytes < q5_f32.q6_k_f32_ordered_workspace_nbytes(
            in_features,
            out_features,
        )

    assert source.count(f"__global__ void {_PRODUCER_KERNEL}(") == 1
    assert source.count(f"__global__ void {_CONSUMER_KERNEL}(") == 1
    assert source.count("hipengine_" + _PRODUCER_NAME) == 1
    assert source.count("hipengine_gguf_q6_k_int16_product_f32_scale_weight_ordered_") == len(
        _ROLES
    )
    producer_kernel = _declaration(source, f"__global__ void {_PRODUCER_KERNEL}(")
    assert "int16_t* __restrict__ products" in producer_kernel
    assert "float* __restrict__ tiled_scales" in producer_kernel
    assert "static_cast<int>(scale) * quant" in producer_kernel
    assert "out_tile" in producer_kernel and "qblock" in producer_kernel

    scale_helper = _declaration(
        source,
        f"__device__ __forceinline__ float {_SCALE_BROADCAST_HELPER}(",
    )
    assert "__builtin_amdgcn_readfirstlane" in scale_helper
    consumer_kernel = _declaration(source, f"__global__ void {_CONSUMER_KERNEL}(")
    assert "const int16_t* __restrict__ products" in consumer_kernel
    assert "const float* __restrict__ tiled_scales" in consumer_kernel
    assert "for (int qblock = 0; qblock < qblocks; ++qblock)" in consumer_kernel
    assert "qblock * QK_K + tid" in consumer_kernel
    assert "k + 128" in consumer_kernel
    assert _SCALE_BROADCAST_HELPER in consumer_kernel
    assert "static_cast<float>" in consumer_kernel
    assert "fmaf(input_value, weights[col], acc[row_index][col])" in consumer_kernel
    assert consumer_kernel.count("h6u_reduce_wave_accumulators_dpp") == 1
    assert "__shfl_down" not in consumer_kernel
    assert consumer_kernel.count("__syncthreads();") == 1


@_H8Q_REQUIRED
def test_h8q_strict_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_attempts = 0

    def _unexpected_build(**_kwargs: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("valid H8Q role reached HIP loading")

    monkeypatch.setattr(
        q5_f32,
        "build_gguf_q5_k_f32_rocblas_prefill",
        _unexpected_build,
    )
    producer = getattr(q5_f32, _PRODUCER_NAME)
    # The encoder is shape-generic for small independent fixtures; ownership is
    # constrained by the composite role check, package policy, and registry.
    with pytest.raises(ValueError, match="multiple of 256"):
        producer(1, 2, 3, 384, 16)
    with pytest.raises(ValueError, match="multiple of 16"):
        producer(1, 2, 3, 512, 15)

    for _, output_dtype, in_features, out_features, row_batch, *_ in _ROLES:
        primitive = getattr(q5_f32, _primitive_name(output_dtype, row_batch))
        composite = getattr(q5_f32, _composite_name(output_dtype, row_batch))
        for function, pointers in (
            (primitive, (1, 2, 3, 4)),
            (composite, (1, 2, 3, 4, 5)),
        ):
            with pytest.raises(ValueError, match="rows must be positive"):
                function(*pointers, 0, in_features, out_features)
            with pytest.raises(ValueError, match=f"exactly {in_features}"):
                function(*pointers, 17, in_features - 256, out_features)
            with pytest.raises(ValueError, match=f"exactly {out_features}"):
                function(*pointers, 17, in_features, out_features - 16)
    assert load_attempts == 0

    with pytest.raises(AssertionError, match="valid H8Q role reached HIP loading"):
        producer(1, 2, 3, 512, 16)
    assert load_attempts == 1


@_H8Q_REQUIRED
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h8q_producer_planes_match_independent_cpu_and_repeat(library: Any) -> None:
    from hipengine.core.hip import get_hip_runtime

    raw = _edge_product_qweight()
    expected_products, expected_scales = _expected_product_scale_planes(raw, 512)
    actual_products = np.empty_like(expected_products)
    actual_scales = np.empty_like(expected_scales)
    repeat_products = np.empty_like(expected_products)
    repeat_scales = np.empty_like(expected_scales)
    runtime = get_hip_runtime()
    before = _allocation_lifecycle()
    buffers: list[Any] = []
    try:
        raw_dev = _device(raw, runtime)
        product_dev = malloc(actual_products.nbytes, runtime=runtime)
        scale_dev = malloc(actual_scales.nbytes, runtime=runtime)
        buffers.extend((raw_dev, product_dev, scale_dev))
        producer = getattr(q5_f32, _PRODUCER_NAME)
        for product_out, scale_out in (
            (actual_products, actual_scales),
            (repeat_products, repeat_scales),
        ):
            runtime.memset(product_dev.ptr, 0xA5, actual_products.nbytes)
            runtime.memset(scale_dev.ptr, 0xFF, actual_scales.nbytes)
            producer(
                raw_dev.ptr,
                product_dev.ptr,
                scale_dev.ptr,
                512,
                16,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(
                host_array_ptr(product_out),
                product_dev,
                product_out.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(scale_out),
                scale_dev,
                scale_out.nbytes,
                runtime=runtime,
            )
        np.testing.assert_array_equal(actual_products, expected_products)
        np.testing.assert_array_equal(
            actual_scales.view(np.uint32),
            expected_scales.view(np.uint32),
        )
        np.testing.assert_array_equal(repeat_products, actual_products)
        np.testing.assert_array_equal(
            repeat_scales.view(np.uint32),
            actual_scales.view(np.uint32),
        )
        assert int(actual_products.min()) == -4_064
        assert int(actual_products.max()) == 4_096
        np.testing.assert_array_equal(
            _reconstruct_f32(actual_products, actual_scales).view(np.uint32),
            _exact_q6_f32_cpu(raw, 512).view(np.uint32),
        )
        assert np.isfinite(actual_scales).all()
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    assert _allocation_lifecycle() == before


@_H8Q_REQUIRED
@pytest.mark.parametrize("rows", [17, 33, 512])
@pytest.mark.parametrize(
    (
        "role",
        "output_dtype",
        "in_features",
        "out_features",
        "row_batch",
        "calls",
        "lds_bytes",
        "vgpr_max",
    ),
    _ROLES,
    ids=tuple(role[0] for role in _ROLES),
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h8q_complete_outputs_activation_plane_cpu_repeat_and_lifecycle(
    rows: int,
    role: str,
    output_dtype: str,
    in_features: int,
    out_features: int,
    row_batch: int,
    calls: int,
    lds_bytes: int,
    vgpr_max: int,
    library: Any,
    production_qweights: dict[tuple[int, int], np.ndarray],
) -> None:
    from hipengine.core.hip import get_hip_runtime

    del role, calls, lds_bytes, vgpr_max
    candidate = getattr(q5_f32, _composite_name(output_dtype, row_batch))
    _, retained_name = _retained_h6u_names(output_dtype, row_batch)
    control = getattr(q5_f32, retained_name)
    rng = np.random.default_rng(20260803 + 47 * rows + 13 * in_features + out_features)
    x_bf16 = _bf16_bits(rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32))
    qweight = production_qweights[(in_features, out_features)]
    host_dtype = np.uint16 if output_dtype == "bf16" else np.float32
    expected = np.empty((rows, out_features), dtype=host_dtype)
    actual = np.empty_like(expected)
    repeated = np.empty_like(expected)
    plane_shape = _activation_plane_shape(rows, in_features, row_batch)
    plane = np.empty(plane_shape, dtype=np.uint16)
    expected_plane = _expected_activation_plane(x_bf16, row_batch)

    runtime = get_hip_runtime()
    before = _allocation_lifecycle()
    buffers: list[Any] = []
    try:
        x_dev = _device(x_bf16, runtime)
        weight_dev = _device(qweight, runtime)
        control_workspace_dev = malloc(
            q5_f32.q6_k_f32_ordered_workspace_nbytes(
                in_features,
                out_features,
            ),
            runtime=runtime,
        )
        candidate_workspace_dev = malloc(
            q5_f32.q6_k_int16_product_f32_scale_workspace_nbytes(
                in_features,
                out_features,
            ),
            runtime=runtime,
        )
        expected_dev = malloc(expected.nbytes, runtime=runtime)
        actual_dev = malloc(actual.nbytes, runtime=runtime)
        activation_dev = malloc(plane.nbytes, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                weight_dev,
                control_workspace_dev,
                candidate_workspace_dev,
                expected_dev,
                actual_dev,
                activation_dev,
            )
        )
        control(
            x_dev.ptr,
            weight_dev.ptr,
            expected_dev.ptr,
            control_workspace_dev.ptr,
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
        _sampled_cpu_gate(
            expected,
            x_bf16,
            qweight,
            row_batch=row_batch,
            output_dtype=output_dtype,
            in_features=in_features,
            out_features=out_features,
        )

        for output in (actual, repeated):
            runtime.memset(actual_dev.ptr, 0xA5, actual.nbytes)
            runtime.memset(activation_dev.ptr, 0x5A, plane.nbytes)
            runtime.memset(
                candidate_workspace_dev.ptr,
                0xFF,
                q5_f32.q6_k_int16_product_f32_scale_workspace_nbytes(
                    in_features,
                    out_features,
                ),
            )
            candidate(
                x_dev.ptr,
                weight_dev.ptr,
                actual_dev.ptr,
                candidate_workspace_dev.ptr,
                activation_dev.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(
                host_array_ptr(output),
                actual_dev,
                output.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(plane),
                activation_dev,
                plane.nbytes,
                runtime=runtime,
            )
            np.testing.assert_array_equal(plane, expected_plane)

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(repeated, expected)
        _sampled_cpu_gate(
            actual,
            x_bf16,
            qweight,
            row_batch=row_batch,
            output_dtype=output_dtype,
            in_features=in_features,
            out_features=out_features,
        )
        finite = _bf16_to_f32(actual) if output_dtype == "bf16" else actual
        assert np.isfinite(finite).all()
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    assert _allocation_lifecycle() == before


def test_h8q_timing_admission_is_indivisible_and_producer_inclusive() -> None:
    measurements: dict[str, dict[str, Any]] = {}
    for index, role in enumerate(_CALL_WEIGHTS):
        measurements[role] = {
            "producer_inclusive": True,
            "hip_event": {"control_ms": 10.0 + index, "candidate_ms": 9.0 + index},
            "synchronized_wall": {
                "control_ms": 11.0 + index,
                "candidate_ms": 10.0 + index,
            },
        }
    measurements["weighted-142-call"] = {
        "producer_inclusive": True,
        "hip_event": {"control_ms": 12.0, "candidate_ms": 11.0},
        "synchronized_wall": {"control_ms": 13.0, "candidate_ms": 12.0},
    }
    assert _screen_admitted(measurements)

    missing = dict(measurements)
    missing.pop("bf16-k3072-n1024")
    assert not _screen_admitted(missing)
    reordered = dict(reversed(tuple(measurements.items())))
    assert not _screen_admitted(reordered)

    for name in measurements:
        for clock in _TIMING_CONTRACT["clocks"]:
            failed = json.loads(json.dumps(measurements))
            failed[name][clock]["candidate_ms"] = failed[name][clock]["control_ms"]
            assert not _screen_admitted(failed)
    downstream_only = json.loads(json.dumps(measurements))
    downstream_only["weighted-142-call"]["producer_inclusive"] = False
    assert not _screen_admitted(downstream_only)
