"""WPF-H7T quality-gated late-start QK-only score-replay RED."""

from __future__ import annotations

import ast
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION
from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch
from tests.test_laguna_h6w_swa_global_score_replay import (
    _cpu_rows as _swa_cpu_rows,
)
from tests.test_laguna_h6w_swa_global_score_replay import (
    _swa_spans,
    test_h6w_strict_late_start_and_scratch_preflight_before_hip as _h6w_preflight_control,
)
from tests.test_laguna_h6z_global_score_weight_replay import (
    _cpu_rows as _global_cpu_rows,
)
from tests.test_laguna_h6z_global_score_weight_replay import (
    _global_spans,
    test_h6z_strict_late_start_and_caller_plane_preflight_before_hip as _h6z_preflight_control,
)

_ROOT = Path(__file__).resolve().parents[1]
_TARGET = (
    _ROOT
    / "benchmarks/results/"
    "2026-08-02-gfx1100-laguna-q2-xl-post-h7s-"
    "qk-only-score-replay-target.json"
)
_TARGET_SHA256 = "9fb06a29c0d4e78566b3d8dde981723cd6d52654e8175978455fbbd9a114aae9"
_PROMPT_SUITE = (
    _ROOT / "benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl"
)
_PROMPT_SUITE_SHA256 = (
    "3097ed25c6f4cf3c2986c1da90e61d1600c3b291745224313dba5100fa7a8e76"
)

_ROWS = 128
_STARTS = (256, 384)
_CONTEXTS = (384, 512)
_KV_HEADS = 8
_HEAD_DIM = 128
_GLOBAL_HEADS = 48
_SWA_HEADS = 72
_GLOBAL_CAPACITY = 4096
_SWA_CAPACITY = 512
_GLOBAL_BLOCK_SIZE = 256
_SCORE_CAPACITY = 512
_SCORE_F32_BYTES = _SWA_HEADS * _ROWS * _SCORE_CAPACITY * 4
_GLOBAL_SCORE_F32_BYTES = _GLOBAL_HEADS * _ROWS * _SCORE_CAPACITY * 4
_KEY_F32_BYTES = _SWA_CAPACITY * _KV_HEADS * _HEAD_DIM * 4
_QUERY_PACKED_F32_BYTES = _ROWS * _SWA_HEADS * _HEAD_DIM * 4
_KEY_F32_OFFSET = _SCORE_F32_BYTES
_QUERY_PACKED_F32_OFFSET = _KEY_F32_OFFSET + _KEY_F32_BYTES
_PLANNED_BYTES = _QUERY_PACKED_F32_OFFSET + _QUERY_PACKED_F32_BYTES
_EXISTING_WORKSPACE_BYTES = 161_120_256
_CALL_WEIGHTS = {
    ("global", 256): 12,
    ("global", 384): 12,
    ("swa", 256): 36,
    ("swa", 384): 36,
}
_CURRENT_MS = {
    "global": 12.198789000000001,
    "swa": 62.62723900000001,
}

_KEY_WIDEN_FUNCTION = "laguna_dense_initial_key_cache_bf16_to_f32_spans"
_KEY_WIDEN_SYMBOL = "hipengine_laguna_dense_initial_key_cache_bf16_to_f32_spans"
_GLOBAL_FUNCTION = (
    "laguna_global_attention_prefill_qrow4_dense_initial_"
    "qk_only_score_weight_replay_exact_bf16_spans"
)
_GLOBAL_SYMBOL = "hipengine_" + _GLOBAL_FUNCTION.removesuffix("_bf16_spans") + "_bf16_spans"
_GLOBAL_VARIANT = (
    "global_context_rows_qrow4_dense_initial_"
    "qk_only_score_weight_replay_exact_spans"
)
_GLOBAL_KERNEL = (
    "laguna_global_attention_prefill_qrow4_dense_initial_"
    "qk_only_score_weight_replay_exact_bf16_kernel"
)
_SWA_FUNCTION = (
    "laguna_swa_attention_prefill_qrow4_dense_initial_"
    "qk_only_score_replay_exact_bf16_spans"
)
_SWA_SYMBOL = "hipengine_" + _SWA_FUNCTION.removesuffix("_bf16_spans") + "_bf16_spans"
_SWA_VARIANT = (
    "swa_context_rows_qrow4_dense_initial_"
    "qk_only_score_replay_exact_spans"
)
_SWA_KERNEL = (
    "laguna_swa_attention_prefill_qrow4_dense_initial_"
    "qk_only_score_replay_exact_bf16_kernel"
)
_OWNER = "LagunaQkOnlyScoreReplay"

_H6W_FUNCTION = (
    "laguna_swa_attention_prefill_qrow4_"
    "dense_initial_global_score_replay_exact_bf16_spans"
)
_H6W_VARIANT = (
    "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
)
_H6W_KERNEL = (
    "laguna_swa_attention_prefill_qrow4_"
    "dense_initial_global_score_replay_exact_bf16_kernel"
)
_H6Z_FUNCTION = (
    "laguna_global_attention_prefill_qrow4_dense_initial_"
    "global_score_weight_replay_exact_bf16_spans"
)
_H6Z_VARIANT = (
    "global_context_rows_qrow4_dense_initial_"
    "global_score_weight_replay_exact_spans"
)
_H6Z_KERNEL = (
    "laguna_global_attention_prefill_qrow4_dense_initial_"
    "global_score_weight_replay_exact_bf16_kernel"
)
_H6W_BODY_SHA256 = (
    "d025a3c73c499c75b2dc0444ae4fdd479cb72729eaad2b1e1aa6e38491748aee"
)
_H6Z_BODY_SHA256 = (
    "3a21a68d13eefb1e865d193869e7fe3b4e86f559349620b091eec14594a5f66b"
)
_EXISTING_BLAS_LAUNCH_SHA256 = (
    "85e0f81cdd9bd0af6b76a553f21f5b2c13f5a52b64ef469ba3d975e5993662c9"
)
_UNCHANGED_FILES = {
    "hipengine/runtime/laguna_gguf_runner.py": (
        "61ff2f193a53eb0c48a55e3616e6c55b7cbc77705b58ac552559cf9b56814688"
    ),
    "hipengine/runtime/laguna_kv.py": (
        "a9d046bd793aca0a3085d4040a14992d80f2b7261ee16f2721ce3ad2ed8bc31f"
    ),
}
_QUALITY_CONTRACT = {
    "prompts": 18,
    "teacher_forced_steps": 576,
    "categories": ("code", "general_en", "general_ja", "mixed_ja_en"),
    "max_kl": 0.05,
    "min_top1_agreement": 0.90,
    "quality_before_timing": True,
    "no_timing_on_failure": True,
    "all_changed_role_classes_per_prompt": True,
}
_PHYSICAL_CONTRACT = {
    "consumer_local_size": 32,
    "consumer_wavefront_size": 32,
    "consumer_lds_bytes": 0,
    "consumer_metadata_vgpr_max": 64,
    "consumer_metadata_sgpr_max": 96,
    "consumer_code_bytes_max": 9_000,
    "consumer_instruction_slots_max": 1_600,
    "consumer_private_spill_scratch_bytes": 0,
}
_TRACE_CONTRACT = {
    "ordered_stages": ("key_only_widen", "query_pack", "packed_f32_qk", "h7t_consumer"),
    "forbidden_stages": ("hipblaslt_pv", "value_widen", "standalone_softmax"),
    "one_queue": True,
    "one_stream": True,
    "new_compiler_processes": 0,
    "workspace_growth_bytes": 0,
}
_TIMING_CONTRACT = {
    "warmups": 5,
    "counter_rotated_repetitions": 15,
    "launches_per_sample": 5,
    "role_count": 4,
    "production_calls": 96,
    "clocks": ("hip_event", "synchronized_wall"),
    "require_every_role_both_clocks": True,
    "require_weighted_aggregate_both_clocks": True,
    "one_shot_only": True,
}
_REJECT_RULE = (
    "Any correctness, quality, physical, trace, lifecycle, per-role, or aggregate "
    "miss removes every H7T surface without family/start/head/layer/prompt subset, "
    "algorithm retune, source rewrite, recompile, or favorable rerun."
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _extract_braced(source: str, marker: str) -> str:
    marker_at = source.index(marker)
    start = source.rfind("\n", 0, marker_at) + 1
    brace = source.index("{", marker_at)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated declaration: {marker}")


def _method_source(source: str, class_name: str, method_name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    assert child.end_lineno is not None
                    return "".join(lines[child.lineno - 1 : child.end_lineno])
    raise AssertionError((class_name, method_name))


def _kv_module():
    from hipengine.kernels.hip_gfx1100.attention import laguna_kv

    return laguna_kv


def _owner_module():
    from hipengine.runtime import laguna_attention_hipblaslt

    return laguna_attention_hipblaslt


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _require_cached() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1"


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _copy_metadata(tensor: Tensor, runtime: Any) -> np.ndarray:
    from hipengine.core.hip import HipMemcpyKind
    from hipengine.core.memory import host_array_ptr

    dtype = {"int32": np.int32, "int64": np.int64, "bool": np.uint8}[
        tensor.dtype.value
    ]
    host = np.empty(tensor.numel, dtype=dtype)
    runtime.memcpy(
        host_array_ptr(host), tensor.ptr, host.nbytes, HipMemcpyKind.DEVICE_TO_HOST
    )
    return host


def _span_snapshot(spans: KVLiveSpans, runtime: Any) -> tuple[np.ndarray, ...]:
    return tuple(
        _copy_metadata(tensor, runtime)
        for tensor in (
            spans.base_offsets,
            spans.live_counts,
            spans.token_positions,
            spans.evict_mask,
            spans.row_positions,
        )
    )


def test_h7t_frozen_target_quality_physical_trace_timing_and_rejection_contract() -> None:
    target_bytes = _TARGET.read_bytes()
    assert _sha256_bytes(target_bytes) == _TARGET_SHA256
    artifact = json.loads(target_bytes)
    assert artifact["status"] == (
        "accepted_matched_production_rerank_and_quality_gated_h7t_target"
    )
    assert artifact["target"]["id"] == "WPF-H7T"
    assert artifact["target"]["implementation_absent"] is True
    assert artifact["decision"]["candidate_implemented"] is False
    assert artifact["decision"]["production_changed"] is False
    assert artifact["production"]["wall_tok_s"] == pytest.approx(431.31016450993457)
    assert artifact["production"]["matched_llamacpp_hip_tok_s"] == 690.791
    assert artifact["production"]["kernel_sum_ms"] == pytest.approx(
        1_172.2412389999988
    )

    scope = artifact["current_target_scope"]
    assert scope["h6w_swa_calls"] == 72
    assert scope["h6z_global_calls"] == 24
    assert scope["calls"] == sum(_CALL_WEIGHTS.values()) == 96
    assert scope["current_ms"] == pytest.approx(sum(_CURRENT_MS.values()))
    assert scope["attention_share_percent"] == pytest.approx(64.88720543000937)
    assert scope["score_plane_bytes"] == _SCORE_F32_BYTES == 18_874_368
    assert scope["global_score_view_bytes"] == _GLOBAL_SCORE_F32_BYTES
    assert tuple(scope["starts"]) == _STARTS
    assert tuple(scope["contexts"]) == _CONTEXTS

    assert _SCORE_F32_BYTES == 72 * 128 * 512 * 4
    assert _GLOBAL_SCORE_F32_BYTES == 48 * 128 * 512 * 4
    assert _KEY_F32_BYTES == 2_097_152
    assert _QUERY_PACKED_F32_BYTES == 4_718_592
    assert _KEY_F32_OFFSET == 18_874_368
    assert _QUERY_PACKED_F32_OFFSET == 20_971_520
    assert _PLANNED_BYTES == 25_690_112
    assert _PLANNED_BYTES < _EXISTING_WORKSPACE_BYTES
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512, use_activation_tile_k_row=True
    ) == _EXISTING_WORKSPACE_BYTES

    quality = artifact["target"]["admission"]["quality"]
    assert _sha256_file(_PROMPT_SUITE) == _PROMPT_SUITE_SHA256
    assert quality["suite_sha256"] == _PROMPT_SUITE_SHA256
    assert quality["prompts"] == _QUALITY_CONTRACT["prompts"]
    assert quality["teacher_forced_steps"] == _QUALITY_CONTRACT["teacher_forced_steps"]
    assert tuple(quality["categories"]) == _QUALITY_CONTRACT["categories"]
    assert quality["max_kl"] == _QUALITY_CONTRACT["max_kl"]
    assert quality["min_top1_agreement"] == _QUALITY_CONTRACT["min_top1_agreement"]
    assert quality["no_timing_on_failure"] is True
    assert artifact["target"]["admission"]["quality_before_timing"] is True
    assert artifact["target"]["storage"]["workspace_growth_bytes"] == 0
    assert artifact["target"]["storage"]["new_persistent_sidecar_bytes"] == 0

    assert _PHYSICAL_CONTRACT["consumer_local_size"] == 32
    assert _PHYSICAL_CONTRACT["consumer_wavefront_size"] == 32
    assert _PHYSICAL_CONTRACT["consumer_lds_bytes"] == 0
    assert _PHYSICAL_CONTRACT["consumer_private_spill_scratch_bytes"] == 0
    assert _TRACE_CONTRACT["forbidden_stages"] == (
        "hipblaslt_pv",
        "value_widen",
        "standalone_softmax",
    )
    assert _TRACE_CONTRACT["new_compiler_processes"] == 0
    assert _TIMING_CONTRACT["production_calls"] == 96
    assert _TIMING_CONTRACT["require_every_role_both_clocks"] is True
    assert _TIMING_CONTRACT["require_weighted_aggregate_both_clocks"] is True
    assert _TIMING_CONTRACT["one_shot_only"] is True
    assert "family/start/head/layer/prompt subset" in _REJECT_RULE
    assert "favorable rerun" in _REJECT_RULE


def test_h7t_source_registry_workspace_and_backend_isolation() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package

    module = _kv_module()
    module.register_laguna_kv_attention_kernels(replace=True)
    for function_name, variant in (
        (_H6W_FUNCTION, _H6W_VARIANT),
        (_H6Z_FUNCTION, _H6Z_VARIANT),
    ):
        function = getattr(module, function_name)
        assert resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant=variant,
        ) is function

    for relative, digest in _UNCHANGED_FILES.items():
        assert _sha256_file(_ROOT / relative) == digest
    source = Path(module.__file__).with_name("laguna_kv_attention.hip").read_text()
    assert _sha256_bytes(_extract_braced(source, _H6W_KERNEL).encode()) == (
        _H6W_BODY_SHA256
    )
    assert _sha256_bytes(_extract_braced(source, _H6Z_KERNEL).encode()) == (
        _H6Z_BODY_SHA256
    )
    assert "output_acc[row_index][part] += weight * cached_values[part];" in (
        _extract_braced(source, _H6W_KERNEL)
    )
    assert "output_acc[row_index][0] += normalized_weight * cached_values[0];" in (
        _extract_braced(source, _H6Z_KERNEL)
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512, use_activation_tile_k_row=True
    ) == _EXISTING_WORKSPACE_BYTES

    # Intentional RED only after H6W/H6Z source, registry, and workspace controls.
    key_widen = getattr(module, _KEY_WIDEN_FUNCTION)
    global_consumer = getattr(module, _GLOBAL_FUNCTION)
    swa_consumer = getattr(module, _SWA_FUNCTION)
    assert key_widen.__name__ == _KEY_WIDEN_FUNCTION
    assert global_consumer.__name__ == _GLOBAL_FUNCTION
    assert swa_consumer.__name__ == _SWA_FUNCTION
    module.register_laguna_kv_attention_kernels(replace=True)
    load_backend_kernel_package("hip_gfx1151")
    for function, variant in (
        (global_consumer, _GLOBAL_VARIANT),
        (swa_consumer, _SWA_VARIANT),
    ):
        key = KernelKey(
            "hip_gfx1100", "laguna_attention_prefill", "bf16", variant
        )
        assert resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        ) is function
        assert not is_registered(
            KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
        )
    assert source.count(_KEY_WIDEN_SYMBOL) == 1
    assert source.count(_GLOBAL_SYMBOL) == 1
    assert source.count(_SWA_SYMBOL) == 1
    assert source.count(_GLOBAL_KERNEL) == 2
    assert source.count(_SWA_KERNEL) == 2


def test_h7t_zero_allocation_owner_and_qk_only_source_contract() -> None:
    owner_module = _owner_module()
    source = Path(owner_module.__file__).read_text()
    launch = _method_source(source, "LagunaAttentionHipblasLt", "launch")
    assert _sha256_bytes(launch.encode()) == _EXISTING_BLAS_LAUNCH_SHA256
    assert "problems.qk.launch(" in launch
    assert "problems.pv.launch(" in launch

    # Intentional RED only after the existing full QK/PV owner is frozen.
    owner_type = getattr(owner_module, _OWNER)
    assert owner_type.planned_nbytes() == _PLANNED_BYTES
    assert owner_type.score_f32_offset() == 0
    assert owner_type.key_f32_offset() == _KEY_F32_OFFSET
    assert owner_type.query_packed_f32_offset() == _QUERY_PACKED_F32_OFFSET
    assert owner_type.supports_global(
        rows=128,
        start_position=256,
        num_q_heads=48,
        num_kv_heads=8,
        head_dim=128,
    )
    assert owner_type.supports_swa(
        rows=128,
        start_position=384,
        num_q_heads=72,
        num_kv_heads=8,
        head_dim=128,
        sliding_window=512,
    )
    owner_source = inspect.getsource(owner_type)
    assert ".qk.launch(" in owner_source
    assert ".pv.launch(" not in owner_source
    assert _KEY_WIDEN_FUNCTION in owner_source
    assert _GLOBAL_FUNCTION in owner_source
    assert _SWA_FUNCTION in owner_source
    assert "laguna_dense_initial_causal_softmax" not in owner_source
    assert "value_f32" not in owner_source
    assert "malloc(" not in owner_source


def test_h7t_key_only_widen_preflight_rejects_before_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _kv_module()

    class FakeFn:
        argtypes = None
        restype = None

        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args: object) -> int:
            self.calls.append(args)
            return 0

    # Intentional RED at the separately named key-only widening surface.
    candidate = getattr(module, _KEY_WIDEN_FUNCTION)
    fake = FakeFn()
    library = SimpleNamespace(**{_KEY_WIDEN_SYMBOL: fake})
    candidate(
        0x10000,
        0x20000,
        _swa_spans(),
        384,
        _KV_HEADS,
        _HEAD_DIM,
        global_layout=False,
        library=library,
        runtime=SimpleNamespace(),
    )
    candidate(
        0x10000,
        0x20000,
        _global_spans(),
        512,
        _KV_HEADS,
        _HEAD_DIM,
        global_layout=True,
        library=library,
        runtime=SimpleNamespace(),
    )
    assert len(fake.calls) == 2

    def fail_build(*_: object, **__: object) -> object:
        raise AssertionError("invalid H7T key-widen preflight loaded HIP")

    monkeypatch.setattr(module, "build_laguna_kv_attention", fail_build)
    for spans, context, global_layout in (
        (_swa_spans(), 256, False),
        (_swa_spans(), 513, False),
        (_global_spans(), 384, False),
        (_swa_spans(), 384, True),
    ):
        with pytest.raises(ValueError, match="H7T key-only widening"):
            candidate(
                0x10000,
                0x20000,
                spans,
                context,
                _KV_HEADS,
                _HEAD_DIM,
                global_layout=global_layout,
            )
    assert len(fake.calls) == 2


@pytest.mark.parametrize("family", ("global", "swa"))
def test_h7t_consumer_strict_role_preflight_after_retained_controls(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    # Retained controls must pass before touching the absent H7T consumer.
    if family == "global":
        _h6z_preflight_control(monkeypatch)
    else:
        _h6w_preflight_control(monkeypatch)

    module = _kv_module()
    function_name = _GLOBAL_FUNCTION if family == "global" else _SWA_FUNCTION
    candidate = getattr(module, function_name)

    class FakeFn:
        argtypes = None
        restype = None

        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args: object) -> int:
            self.calls.append(args)
            return 0

    symbol = _GLOBAL_SYMBOL if family == "global" else _SWA_SYMBOL
    fake = FakeFn()
    library = SimpleNamespace(**{symbol: fake})
    for start in _STARTS:
        if family == "global":
            candidate(
                0x6000,
                0x7000,
                0x8000,
                _global_spans(),
                _ROWS,
                _GLOBAL_CAPACITY,
                _GLOBAL_HEADS,
                _KV_HEADS,
                _HEAD_DIM,
                _HEAD_DIM**-0.5,
                score_scratch_nbytes=_GLOBAL_SCORE_F32_BYTES,
                start_position=start,
                library=library,
                runtime=SimpleNamespace(),
            )
        else:
            candidate(
                0x6000,
                0x7000,
                0x8000,
                _swa_spans(),
                _ROWS,
                _SWA_HEADS,
                _KV_HEADS,
                _HEAD_DIM,
                _HEAD_DIM**-0.5,
                score_scratch_nbytes=_SCORE_F32_BYTES,
                sliding_window=_SWA_CAPACITY,
                start_position=start,
                library=library,
                runtime=SimpleNamespace(),
            )
    assert len(fake.calls) == 2


def _run_current_and_h7t(
    library: Any,
    *,
    family: str,
    start_position: int,
) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
        memory_stats,
    )
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    module = _kv_module()
    runtime = get_hip_runtime()
    is_global = family == "global"
    q_heads = _GLOBAL_HEADS if is_global else _SWA_HEADS
    capacity = _GLOBAL_CAPACITY if is_global else _SWA_CAPACITY
    layer_type = FULL_ATTENTION if is_global else SLIDING_ATTENTION
    config = SimpleNamespace(
        block_count=1,
        layer_types=(layer_type,),
        head_counts=(q_heads,),
        head_count_kv=_KV_HEADS,
        key_length=_HEAD_DIM,
        value_length=_HEAD_DIM,
        sliding_window=_SWA_CAPACITY,
    )
    baseline = memory_stats()
    cache = allocate_laguna_kv_cache(
        config,
        context_length=capacity,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    rng = np.random.default_rng(
        0x7A00 + start_position + (0 if is_global else 0x100)
    )
    total_rows = start_position + _ROWS
    keys = rng.normal(
        0.0, 0.12, size=(total_rows, _KV_HEADS, _HEAD_DIM)
    ).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=keys.shape).astype(np.float32)
    queries = rng.normal(
        0.0, 0.12, size=(_ROWS, q_heads, _HEAD_DIM)
    ).astype(np.float32)
    current_host = np.empty_like(queries)
    candidate_host = np.empty_like(queries)
    allocations: list[Any] = []
    owner = None
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        query_rows = malloc(queries.nbytes, runtime=runtime)
        current_out = malloc(current_host.nbytes, runtime=runtime)
        candidate_out = malloc(candidate_host.nbytes, runtime=runtime)
        workspace = malloc(_EXISTING_WORKSPACE_BYTES, runtime=runtime)
        allocations.extend(
            (key_rows, value_rows, query_rows, current_out, candidate_out, workspace)
        )
        for device, host in (
            (key_rows, keys),
            (value_rows, values),
            (query_rows, queries),
        ):
            copy_host_to_device(
                device, host_array_ptr(host), host.nbytes, runtime=runtime
            )
        if start_position:
            cache.prepare_rows(tuple(range(start_position)))
            cache.append_rows(
                0,
                key_rows.ptr,
                value_rows.ptr,
                start_position,
                library=library,
            )
            cache.commit_rows()
        cache.prepare_rows(tuple(range(start_position, total_rows)))
        row_nbytes = _KV_HEADS * _HEAD_DIM * np.dtype(np.float32).itemsize
        current_key_ptr = key_rows.ptr + start_position * row_nbytes
        current_value_ptr = value_rows.ptr + start_position * row_nbytes
        cache.append_rows(
            0,
            current_key_ptr,
            current_value_ptr,
            _ROWS,
            library=library,
        )
        state = cache.layer(0)
        spans_before = _span_snapshot(state.spans, runtime)
        runtime.memset(current_out.ptr, 0xA5, current_out.nbytes)
        if is_global:
            current = getattr(module, _H6Z_FUNCTION)
            current(
                query_rows.ptr,
                current_key_ptr,
                current_value_ptr,
                state.key_cache.ptr,
                state.value_cache.ptr,
                current_out.ptr,
                workspace.ptr,
                state.spans,
                _ROWS,
                _GLOBAL_CAPACITY,
                q_heads,
                _KV_HEADS,
                _HEAD_DIM,
                _HEAD_DIM**-0.5,
                score_scratch_nbytes=_GLOBAL_SCORE_F32_BYTES,
                start_position=start_position,
                library=library,
                runtime=runtime,
            )
            cpu_rows = _global_cpu_rows(
                queries, keys, values, start_position=start_position
            )
        else:
            current = getattr(module, _H6W_FUNCTION)
            current(
                query_rows.ptr,
                current_key_ptr,
                current_value_ptr,
                state.key_cache.ptr,
                state.value_cache.ptr,
                current_out.ptr,
                workspace.ptr,
                state.spans,
                _ROWS,
                q_heads,
                _KV_HEADS,
                _HEAD_DIM,
                _HEAD_DIM**-0.5,
                score_scratch_nbytes=_SCORE_F32_BYTES,
                sliding_window=_SWA_CAPACITY,
                start_position=start_position,
                library=library,
                runtime=runtime,
            )
            cpu_rows = _swa_cpu_rows(
                queries, keys, values, start_position=start_position
            )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(current_host),
            current_out,
            current_host.nbytes,
            runtime=runtime,
        )
        assert np.isfinite(current_host).all()
        for row, expected in cpu_rows.items():
            np.testing.assert_allclose(
                current_host[row], expected, rtol=3e-4, atol=3e-4
            )
        for actual, expected in zip(
            _span_snapshot(state.spans, runtime), spans_before, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)

        # Intentional RED only after complete retained output/CPU/span controls.
        owner_type = getattr(_owner_module(), _OWNER)
        owner = owner_type(runtime=runtime)
        runtime.memset(candidate_out.ptr, 0xA5, candidate_out.nbytes)
        if is_global:
            owner.launch_global(
                query_rows.ptr,
                state.key_cache.ptr,
                state.value_cache.ptr,
                candidate_out.ptr,
                workspace.ptr,
                state.spans,
                rows=_ROWS,
                max_context_len=_GLOBAL_CAPACITY,
                start_position=start_position,
                num_q_heads=q_heads,
                num_kv_heads=_KV_HEADS,
                head_dim=_HEAD_DIM,
                scale=_HEAD_DIM**-0.5,
                workspace_nbytes=workspace.nbytes,
                stream=0,
                kv_library=library,
            )
        else:
            owner.launch_swa(
                query_rows.ptr,
                state.key_cache.ptr,
                state.value_cache.ptr,
                candidate_out.ptr,
                workspace.ptr,
                state.spans,
                rows=_ROWS,
                start_position=start_position,
                num_q_heads=q_heads,
                num_kv_heads=_KV_HEADS,
                head_dim=_HEAD_DIM,
                scale=_HEAD_DIM**-0.5,
                sliding_window=_SWA_CAPACITY,
                workspace_nbytes=workspace.nbytes,
                stream=0,
                kv_library=library,
            )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(candidate_host),
            candidate_out,
            candidate_host.nbytes,
            runtime=runtime,
        )
        assert np.isfinite(candidate_host).all()
        np.testing.assert_allclose(
            candidate_host, current_host, rtol=2e-5, atol=2e-6
        )
        assert float(np.max(np.abs(candidate_host - current_host))) <= 1e-4
        for actual, expected in zip(
            _span_snapshot(state.spans, runtime), spans_before, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        cache.discard_rows()
    finally:
        if owner is not None:
            owner.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()
    after = memory_stats()
    assert after["current_allocated_bytes"] == baseline["current_allocated_bytes"]
    assert after["active_allocations"] == baseline["active_allocations"]


@pytest.fixture(scope="module")
def h7t_control_library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    return _kv_module().build_laguna_kv_attention(
        load=True,
        compiler_version=compiler_version,
        require_cached=_require_cached(),
    )


@pytest.mark.parametrize(
    ("family", "start_position"),
    tuple((family, start) for family in ("global", "swa") for start in _STARTS),
    ids=("global-start256", "global-start384", "swa-start256", "swa-start384"),
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h7t_all_roles_current_cpu_candidate_spans_and_lifecycle(
    h7t_control_library: Any,
    family: str,
    start_position: int,
) -> None:
    _run_current_and_h7t(
        h7t_control_library,
        family=family,
        start_position=start_position,
    )
