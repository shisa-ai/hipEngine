"""WPF-H7E IQ3 two-plane residual-D4 source-MMQ RED contract."""

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
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.quant import (
    gguf_iq_source_mmq_prefill as source_mmq,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_mmq_prefill as residual_d4
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill import (
    build_gguf_iq_selected_prefill,
    gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.quant.gguf import (
    GGMLQuantizationType,
    bf16_to_float32,
    dequantize_gguf_data,
)
from tests.test_gguf_iq_gemv import _f32_to_bf16_u16, _make_iq3_weight
from tests.test_gguf_iq_selected_prefill import (
    CompactMeta,
    _compact_meta,
    _run_single_grouped,
)

_ROWS = (1, 7, 8, 9, 512)
_TOP_K = 10
_IN_FEATURES = 1024
_OUT_FEATURES = 128
_EXPERTS = 256
_TAIL_COUNTS = (0, 1, 17, 128, 129)
_SOURCE_VARIANT = (
    "selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out"
)
_CANDIDATE_VARIANT = (
    "selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out"
)
_CANDIDATE_WRAPPER = (
    "gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_"
    "prefill_compact_bf16_bf16_out"
)
_CANDIDATE_SYMBOL = "hipengine_" + _CANDIDATE_WRAPPER
_CANDIDATE_KERNEL = "gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_kernel"
_SOURCE_HIP = Path(source_mmq.__file__).with_suffix(".hip")
_PRODUCER_HIP = Path(residual_d4.__file__).with_suffix(".hip")
_SOURCE_KERNEL_SHA256 = (
    "94002fb0d682c9373235343ff67dd6c88f9a2ea353b89eda1e46aa7a2756d8ad"
)
_SOURCE_LAUNCH_SHA256 = (
    "cb1d6c90474181d01cc89f9957d261444ce450171c8bea47de5acc06ffdfe0a7"
)
_SOURCE_IQ3_EXPORT_SHA256 = (
    "8a1fc916052de82bb20b221ae4331b5c953a73c71fc8836b756ea16cd613dbfc"
)
_SOURCE_IQ4_EXPORT_SHA256 = (
    "3dd41ffa226085cfe478286d3214dae9d80901a28cdd077beb95e83f15da1b00"
)
_SOURCE_METADATA_SHA256 = (
    "bc01f64ae5ccd0aab1723c455bf9751574fdcd530fbe01052a067bd698daf98a"
)
_SOURCE_PY_LAUNCH_SHA256 = (
    "9bee6c81f62270853f45842c13575c5642d84308b38e73552c411b248298f3dd"
)
_SOURCE_IQ3_WRAPPER_SHA256 = (
    "6031592bdf676ab7f3439b45679ba1644f496268bbd99ebf57aeaa8d63e40fce"
)
_SOURCE_IQ4_WRAPPER_SHA256 = (
    "f84b5a079e5e1d7e391e2567ad5a4941f4ffad9a3576574d201c03fef7b1d946"
)
_PRODUCER_HIP_SHA256 = (
    "e648f8f1977aa959c1ba310e6f0d13e74142a2e05cef4b9bf194eecb0d5da56e"
)
_PRODUCER_WRAPPER_SHA256 = (
    "de71344a8271ed33f148e003acb2b70ad5459b050a3319f124299d0c0cae1d0e"
)
_H6T_SOURCE_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_rowbatch8_"
    "bf16_bf16_out"
)
_SELECTION = {
    "layers": 45,
    "exact_sum_ms": 248.42099380493164,
    "candidate_sum_ms": 172.8543701171875,
    "speedup": 1.43716929827411,
    "candidate_change_percent": -30.418775213129344,
    "all_layers_faster": True,
    "bf16_mismatch_min": 0.02187004089355469,
    "bf16_mismatch_median": 0.022680918375651043,
    "bf16_mismatch_max": 0.025007883707682293,
    "max_leaf_kl": 0.0004865060822764507,
    "min_leaf_top1": 0.9994140625,
    "performance_claim": False,
}
_D4X3_CLOSURE = {
    "exact_sum_ms": 249.47145652770996,
    "candidate_sum_ms": 247.90887641906738,
    "speedup": 1.0063030421952346,
    "layers_faster": 27,
    "layers": 45,
}
_PHYSICAL = {
    "local_size": 256,
    "block": (32, 8),
    "grid_x": 24,
    "grid_y": "mmq_total_rows / 128",
    "dynamic_lds_bytes": 57_856,
    "code_bytes_max": 31_564,
    "metadata_vgpr_max": 148,
    "metadata_sgpr_max": 44,
    "private_bytes": 0,
    "vgpr_spills": 0,
    "sgpr_spills": 0,
    "dynamic_stack": False,
    "runtime_scratch_bytes": 0,
    "static_integer_wmma": 128,
    "static_barriers": 5,
    "static_bf16_stores": 64,
}
_TIMING = {
    "warmups": 5,
    "repetitions": 15,
    "launches_per_sample": 5,
    "order": "counter_rotated",
    "layers": tuple(range(1, 46)),
    "clocks": ("hip_event", "synchronized_wall"),
    "require_every_layer_both_clocks": True,
    "require_aggregate_both_clocks": True,
    "one_shot_only": True,
}
_QUALITY = {
    "prompt_suite": "benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl",
    "prompt_suite_sha256": (
        "3097ed25c6f4cf3c2986c1da90e61d1600c3b291745224313dba5100fa7a8e76"
    ),
    "prompts": 18,
    "teacher_forced_steps": 576,
    "categories": ("code", "general_en", "general_ja", "mixed_ja_en"),
    "max_kl": 0.05,
    "min_top1": 0.90,
    "poolside_required": True,
    "same_mode_determinism_required": True,
    "free_running_horizons": (16, 32),
    "lifecycle_required": True,
    "promotion_timing_only_after_pass": True,
    "prompt_or_layer_conditioning_allowed": False,
}
_WORKSPACE = {
    "d4x2_required_bytes_m512": 11_796_480,
    "reuse_buffer": "LagunaMoEScratch.expert_gate_up after gate/up and SiLU",
    "reuse_buffer_bytes_m512": 20_971_520,
    "activation_q8_bytes_m512": 5_898_240,
    "new_allocation_bytes": 0,
    "workspace_growth_bytes": 0,
}
_REJECTION_SURFACES = (
    "gfx1100 H7E HIP kernel/export",
    "gfx1100 H7E Python wrapper/registry key",
    "gfx1151 H7E exclusion",
    "H7E RED test",
)
_REJECT_RULE = (
    "Any leaf correctness, physical, resource, cached-trace, compiler, lifecycle, "
    "per-layer both-clock, or aggregate both-clock miss removes every H7E leaf "
    "surface without tuning or rerun. A complete-quality miss removes the "
    "temporary runtime owner and skips promotion timing."
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _module():
    return importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_source_mmq_prefill"
    )


def _candidate():
    # Intentional RED lookup occurs only after retained controls in each test.
    return getattr(_module(), _CANDIDATE_WRAPPER)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _declaration(source: str, anchor: str) -> str:
    start = source.index(anchor)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        depth += (source[index] == "{") - (source[index] == "}")
        if depth == 0:
            return source[start : index + 1]
    raise AssertionError(f"unterminated declaration: {anchor}")


def _device_buffer(array: np.ndarray, buffers: list[Any], runtime):
    value = np.ascontiguousarray(array)
    buffer = malloc(value.nbytes, runtime=runtime)
    copy_host_to_device(
        buffer,
        host_array_ptr(value),
        value.nbytes,
        runtime=runtime,
    )
    buffers.append(buffer)
    return buffer


def _lifecycle() -> tuple[int, int]:
    stats = memory_stats()
    return stats["current_allocated_bytes"], stats["active_allocations"]


def _counts_for_rows(rows: int) -> tuple[int, ...]:
    lanes = rows * _TOP_K
    lane = np.arange(lanes, dtype=np.int64)
    experts = (37 * lane + 11 * (lane // _TOP_K)) % _EXPERTS
    return tuple(np.bincount(experts, minlength=_EXPERTS).tolist())


def _positive_bf16(rows: int) -> np.ndarray:
    values = np.arange(rows * _IN_FEATURES, dtype=np.int32).reshape(
        rows, _IN_FEATURES
    )
    f32 = ((values % 61) + 1).astype(np.float32) / np.float32(128.0)
    return _f32_to_bf16_u16(f32)


def _sample_rows(meta: CompactMeta) -> np.ndarray:
    candidates = {0, meta.compact_rows - 1}
    for value in meta.expert_start_compact:
        index = int(value)
        if 0 <= index < meta.compact_rows:
            candidates.add(index)
        if 0 <= index - 1 < meta.compact_rows:
            candidates.add(index - 1)
    return np.asarray(sorted(candidates)[:32], dtype=np.int64)


def _sampled_cpu_gate(
    output: np.ndarray,
    x_bf16: np.ndarray,
    meta: CompactMeta,
    qweight: np.ndarray,
) -> None:
    columns = np.asarray(
        (0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 95, 96, 126, 127),
        dtype=np.int64,
    )
    rows = _sample_rows(meta)
    experts = np.repeat(
        np.arange(meta.num_experts, dtype=np.int64),
        np.asarray(meta.counts, dtype=np.int64),
    )
    weights = dequantize_gguf_data(
        np.ascontiguousarray(qweight[:, columns]),
        GGMLQuantizationType.IQ3_XXS,
    ).astype(np.float32)
    x = bf16_to_float32(x_bf16[rows]).astype(np.float32)
    reference = np.empty((len(rows), len(columns)), dtype=np.float32)
    for index, (row, expert) in enumerate(zip(rows, experts[rows], strict=True)):
        reference[index] = weights[int(expert)] @ x[index]
    sampled = bf16_to_float32(output[np.ix_(rows, columns)])
    quality = evaluate_logits(reference, sampled)
    assert quality.kl_max <= 0.05, quality
    assert quality.top1_agreement >= 0.90, quality


def _run_candidate(
    wrapper,
    *,
    libraries: dict[str, Any],
    meta: CompactMeta,
    x_bf16: np.ndarray,
    qweight: np.ndarray,
) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    metadata = source_mmq.build_iq_source_mmq128_metadata(meta.counts)
    poison = np.full(
        (meta.compact_rows, qweight.shape[1]),
        np.uint16(0x7FC0),
        dtype=np.uint16,
    )
    actual = np.empty_like(poison)
    compact_before = meta.expert_start_compact.copy()
    mmq_before = metadata.expert_start_mmq.copy()
    tiles_before = metadata.tile_expert.copy()
    before = _lifecycle()
    buffers: list[Any] = []
    try:
        x_dev = _device_buffer(x_bf16, buffers, runtime)
        packed_dev = malloc(
            residual_d4.q8_mmq_d4x2_nbytes(meta.compact_rows, _IN_FEATURES),
            runtime=runtime,
        )
        buffers.append(packed_dev)
        compact_dev = _device_buffer(meta.expert_start_compact, buffers, runtime)
        mmq_dev = _device_buffer(metadata.expert_start_mmq, buffers, runtime)
        tile_dev = _device_buffer(metadata.tile_expert, buffers, runtime)
        weight_dev = _device_buffer(qweight, buffers, runtime)
        output_dev = _device_buffer(poison, buffers, runtime)
        residual_d4.gguf_q8_0_mmq128_quantize_bf16_d4x2(
            x_dev.ptr,
            packed_dev.ptr,
            meta.compact_rows,
            _IN_FEATURES,
            library=libraries["producer"],
            runtime=runtime,
        )
        wrapper(
            packed_dev.ptr,
            compact_dev.ptr,
            mmq_dev.ptr,
            tile_dev.ptr,
            weight_dev.ptr,
            output_dev.ptr,
            compact_rows=meta.compact_rows,
            in_features=_IN_FEATURES,
            out_features=qweight.shape[1],
            num_experts=meta.num_experts,
            mmq_total_rows=metadata.mmq_total_rows,
            library=libraries["consumer"],
            runtime=runtime,
        )
        runtime.device_synchronize()
        compact_after = np.empty_like(compact_before)
        mmq_after = np.empty_like(mmq_before)
        tiles_after = np.empty_like(tiles_before)
        copy_device_to_host(
            host_array_ptr(actual), output_dev, actual.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(compact_after),
            compact_dev,
            compact_after.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(mmq_after), mmq_dev, mmq_after.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(tiles_after),
            tile_dev,
            tiles_after.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(compact_after, compact_before)
        np.testing.assert_array_equal(mmq_after, mmq_before)
        np.testing.assert_array_equal(tiles_after, tiles_before)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    assert _lifecycle() == before
    assert not np.any(actual == np.uint16(0x7FC0))
    assert np.isfinite(bf16_to_float32(actual)).all()
    return actual


@pytest.fixture(scope="module")
def libraries() -> dict[str, Any]:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    require_cached = os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1"
    return {
        "exact": build_gguf_iq_selected_prefill(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached,
        ),
        "producer": residual_d4.build_gguf_q8_0_mmq_prefill(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached,
        ),
        "consumer": source_mmq.build_gguf_iq_source_mmq_prefill(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached,
        ),
    }


@pytest.fixture(scope="module")
def qweight() -> np.ndarray:
    return _make_iq3_weight(_EXPERTS, _OUT_FEATURES, _IN_FEATURES)


def test_h7e_frozen_selection_physical_timing_quality_workspace_and_rejection() -> None:
    assert _SELECTION["candidate_sum_ms"] < _SELECTION["exact_sum_ms"]
    assert _SELECTION["all_layers_faster"] and _SELECTION["layers"] == 45
    assert _SELECTION["speedup"] == pytest.approx(
        _SELECTION["exact_sum_ms"] / _SELECTION["candidate_sum_ms"]
    )
    assert _SELECTION["performance_claim"] is False
    assert _D4X3_CLOSURE["layers_faster"] == 27
    assert _D4X3_CLOSURE["layers"] == 45
    assert _D4X3_CLOSURE["speedup"] == pytest.approx(
        _D4X3_CLOSURE["exact_sum_ms"] / _D4X3_CLOSURE["candidate_sum_ms"]
    )
    assert _PHYSICAL == {
        "local_size": 256,
        "block": (32, 8),
        "grid_x": 24,
        "grid_y": "mmq_total_rows / 128",
        "dynamic_lds_bytes": 57_856,
        "code_bytes_max": 31_564,
        "metadata_vgpr_max": 148,
        "metadata_sgpr_max": 44,
        "private_bytes": 0,
        "vgpr_spills": 0,
        "sgpr_spills": 0,
        "dynamic_stack": False,
        "runtime_scratch_bytes": 0,
        "static_integer_wmma": 128,
        "static_barriers": 5,
        "static_bf16_stores": 64,
    }
    assert _TIMING["layers"] == tuple(range(1, 46))
    assert _TIMING["warmups"] == 5
    assert _TIMING["repetitions"] == 15
    assert _TIMING["launches_per_sample"] == 5
    assert _TIMING["require_every_layer_both_clocks"]
    assert _TIMING["require_aggregate_both_clocks"]
    prompt_suite = Path(_QUALITY["prompt_suite"])
    assert hashlib.sha256(prompt_suite.read_bytes()).hexdigest() == (
        _QUALITY["prompt_suite_sha256"]
    )
    assert _QUALITY["prompts"] == 18
    assert _QUALITY["teacher_forced_steps"] == 576
    assert _QUALITY["max_kl"] == 0.05
    assert _QUALITY["min_top1"] == 0.90
    assert _QUALITY["prompt_or_layer_conditioning_allowed"] is False
    assert _WORKSPACE["d4x2_required_bytes_m512"] == (
        residual_d4.q8_mmq_d4x2_nbytes(512 * _TOP_K, _IN_FEATURES)
    )
    assert _WORKSPACE["reuse_buffer_bytes_m512"] >= (
        _WORKSPACE["d4x2_required_bytes_m512"]
    )
    assert _WORKSPACE["activation_q8_bytes_m512"] < (
        _WORKSPACE["d4x2_required_bytes_m512"]
    )
    assert _WORKSPACE["new_allocation_bytes"] == 0
    assert _WORKSPACE["workspace_growth_bytes"] == 0
    assert len(_REJECTION_SURFACES) == 4
    assert _REJECT_RULE.endswith("skips promotion timing.")
    assert "without tuning or rerun" in _REJECT_RULE


def test_h7e_registry_source_policy_fallback_and_gfx1151_isolation() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    module = _module()
    source = _SOURCE_HIP.read_text()
    source_kernel = _declaration(
        source,
        "__global__ void "
        "gguf_iq_selected_mmq_i128_j128_k256_q8_1_ds4_kernel(",
    )
    source_launch = _declaration(source, "int launch_iq_source_mmq(")
    source_iq3_export = _declaration(
        source,
        'extern "C" int\nhipengine_gguf_iq3_xxs_selected_mmq',
    )
    source_iq4_export = _declaration(
        source,
        'extern "C" int\nhipengine_gguf_iq4_xs_selected_mmq',
    )
    assert _sha256(source_kernel) == _SOURCE_KERNEL_SHA256
    assert _sha256(source_launch) == _SOURCE_LAUNCH_SHA256
    assert _sha256(source_iq3_export) == _SOURCE_IQ3_EXPORT_SHA256
    assert _sha256(source_iq4_export) == _SOURCE_IQ4_EXPORT_SHA256
    assert _sha256(inspect.getsource(module.build_iq_source_mmq128_metadata)) == (
        _SOURCE_METADATA_SHA256
    )
    assert _sha256(inspect.getsource(module._launch_iq_source_mmq)) == (
        _SOURCE_PY_LAUNCH_SHA256
    )
    assert _sha256(
        inspect.getsource(
            module.gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out
        )
    ) == _SOURCE_IQ3_WRAPPER_SHA256
    assert _sha256(
        inspect.getsource(
            module.gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out
        )
    ) == _SOURCE_IQ4_WRAPPER_SHA256
    assert hashlib.sha256(_PRODUCER_HIP.read_bytes()).hexdigest() == (
        _PRODUCER_HIP_SHA256
    )
    assert _sha256(
        inspect.getsource(residual_d4.gguf_q8_0_mmq128_quantize_bf16_d4x2)
    ) == _PRODUCER_WRAPPER_SHA256
    for quant, wrapper in (
        (
            "gguf_iq3_xxs",
            module.gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            module.gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out,
        ),
    ):
        source_key = KernelKey(
            "hip_gfx1100", "moe_linear", quant, _SOURCE_VARIANT
        )
        assert resolve(
            backend=source_key.backend,
            layer=source_key.layer,
            quant=source_key.quant,
            variant=source_key.variant,
        ) is wrapper
    assert hip_gfx1100.LAGUNA_SELECTED_DOWN_MODE == "grouped_exact"
    assert hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS["gguf_iq3_xxs"] == (
        _H6T_SOURCE_VARIANT
    )
    register_gfx1151_kernels(replace=True)
    assert not is_registered(
        KernelKey("hip_gfx1151", "moe_linear", "gguf_iq3_xxs", _SOURCE_VARIANT)
    )

    # Intentional RED only after all source/fallback controls are frozen.
    candidate = _candidate()
    candidate_key = KernelKey(
        "hip_gfx1100", "moe_linear", "gguf_iq3_xxs", _CANDIDATE_VARIANT
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
    assert source.count(_CANDIDATE_SYMBOL) == 1
    assert source.count(_CANDIDATE_KERNEL) == 1
    assert _CANDIDATE_VARIANT not in (
        hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS.values()
    )


def test_h7e_wrapper_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    source_wrapper = (
        module.gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out
    )
    kwargs = dict(
        xq_ptr=1,
        expert_start_compact_ptr=2,
        expert_start_mmq_ptr=3,
        tile_expert_ptr=4,
        qweight_ptr=5,
        out_ptr=6,
        compact_rows=17,
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        num_experts=4,
        mmq_total_rows=128,
    )
    load_attempts = 0

    def fail_if_loaded(**_: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H7E shape reached HIP loader")

    monkeypatch.setattr(module, "build_gguf_iq_source_mmq_prefill", fail_if_loaded)
    for fn in (source_wrapper,):
        with pytest.raises(ValueError, match="compact_rows must be positive"):
            fn(**{**kwargs, "compact_rows": 0})
        with pytest.raises(ValueError, match="divisible by 256"):
            fn(**{**kwargs, "in_features": 896})
        with pytest.raises(ValueError, match="multiple of 128"):
            fn(**{**kwargs, "out_features": 127})
        with pytest.raises(ValueError, match="num_experts must be positive"):
            fn(**{**kwargs, "num_experts": 0})
        with pytest.raises(ValueError, match="multiple of 128"):
            fn(**{**kwargs, "mmq_total_rows": 127})
    assert load_attempts == 0

    # Intentional RED after retained validation behavior is proven.
    candidate = _candidate()
    for changed, message in (
        ({"compact_rows": 0}, "compact_rows must be positive"),
        ({"in_features": 896}, "divisible by 256"),
        ({"out_features": 127}, "multiple of 128"),
        ({"num_experts": 0}, "num_experts must be positive"),
        ({"mmq_total_rows": 127}, "multiple of 128"),
    ):
        with pytest.raises(ValueError, match=message):
            candidate(**{**kwargs, **changed})
    assert load_attempts == 0


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("case", "counts"),
    (
        ("rows1", _counts_for_rows(1)),
        ("rows7", _counts_for_rows(7)),
        ("rows8", _counts_for_rows(8)),
        ("rows9", _counts_for_rows(9)),
        ("rows512", _counts_for_rows(512)),
        ("empty-uneven-127-128-129-tails", _TAIL_COUNTS),
    ),
    ids=("rows1", "rows7", "rows8", "rows9", "rows512", "tails"),
)
def test_h7e_complete_output_passes_exact_and_independent_cpu_quality(
    case: str,
    counts: tuple[int, ...],
    libraries: dict[str, Any],
    qweight: np.ndarray,
) -> None:
    del case
    meta = _compact_meta(counts)
    selected_weight = np.ascontiguousarray(qweight[: meta.num_experts])
    x_bf16 = _positive_bf16(meta.compact_rows)
    before = _lifecycle()
    control = _run_single_grouped(
        gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
        libraries["exact"],
        x_bf16=x_bf16,
        meta=meta,
        qweight=selected_weight,
        wmma=False,
    )
    assert _lifecycle() == before
    assert np.isfinite(bf16_to_float32(control)).all()
    _sampled_cpu_gate(control, x_bf16, meta, selected_weight)

    # Intentional RED only after the exact/CPU/finite/lifecycle controls pass.
    candidate = _candidate()
    actual = _run_candidate(
        candidate,
        libraries=libraries,
        meta=meta,
        x_bf16=x_bf16,
        qweight=selected_weight,
    )
    quality = evaluate_logits(
        bf16_to_float32(control),
        bf16_to_float32(actual),
    )
    assert quality.kl_max <= 0.05, quality
    assert quality.top1_agreement >= 0.90, quality
    assert quality.passed, quality
    _sampled_cpu_gate(actual, x_bf16, meta, selected_weight)
