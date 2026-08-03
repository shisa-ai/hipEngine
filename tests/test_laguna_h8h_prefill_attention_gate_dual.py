"""WPF-H8H exact prefill attention+softplus dual-publication contract."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION

_TARGET = Path(
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-"
    "post-h8g-prefill-attention-softplus-dual-publication-target.json"
)
_TARGET_SHA256 = "70bba53e21249b3108c288cfd7f24d6375266d167898e15179bbc3d2b84b4efd"
_LAYER = "laguna_attention_prefill+attention_gate"
_QUANT = "f32"
_ROWS = 128
_HEAD_DIM = 128
_GLOBAL_HEADS = 48
_SWA_HEADS = 72
_KV_HEADS = 8
_GLOBAL_CAPACITY = 4096
_SWA_CAPACITY = 512
_SCORE_SCRATCH_BYTES = {
    "global_late": 12_582_912,
    "swa_late": 18_874_368,
}

_ROUTES = {
    "global_early": {
        "starts": (0, 128),
        "variant": (
            "global_context_rows_dense_initial_fixed512_cached_exact_"
            "softplus_bf16_spans"
        ),
        "function": (
            "laguna_global_attention_prefill_dense_initial_fixed512_"
            "cached_exact_softplus_gate_bf16_spans"
        ),
        "symbol": (
            "hipengine_laguna_global_attention_prefill_dense_initial_"
            "fixed512_cached_exact_softplus_gate_bf16_spans"
        ),
        "kernel": (
            "laguna_global_attention_prefill_dense_initial_fixed512_"
            "cached_exact_softplus_gate_bf16_kernel"
        ),
        "control": (
            "laguna_global_attention_prefill_dense_initial_fixed512_"
            "cached_exact_bf16_spans"
        ),
        "local_size": 256,
        "runtime_vgpr_max": 48,
    },
    "global_late": {
        "starts": (256, 384),
        "variant": (
            "global_context_rows_qrow4_dense_initial_global_score_weight_"
            "replay_exact_softplus_bf16_spans"
        ),
        "function": (
            "laguna_global_attention_prefill_qrow4_dense_initial_global_"
            "score_weight_replay_exact_softplus_gate_bf16_spans"
        ),
        "symbol": (
            "hipengine_laguna_global_attention_prefill_qrow4_dense_initial_"
            "global_score_weight_replay_exact_softplus_gate_bf16_spans"
        ),
        "kernel": (
            "laguna_global_attention_prefill_qrow4_dense_initial_global_"
            "score_weight_replay_exact_softplus_gate_bf16_kernel"
        ),
        "control": (
            "laguna_global_attention_prefill_qrow4_dense_initial_global_"
            "score_weight_replay_exact_bf16_spans"
        ),
        "local_size": 32,
        "runtime_vgpr_max": 56,
    },
    "swa_early": {
        "starts": (0, 128),
        "variant": (
            "swa_context_rows_qrow4_dense_initial_cached_exact_"
            "softplus_bf16_spans"
        ),
        "function": (
            "laguna_swa_attention_prefill_qrow4_dense_initial_cached_exact_"
            "softplus_gate_bf16_spans"
        ),
        "symbol": (
            "hipengine_laguna_swa_attention_prefill_qrow4_dense_initial_"
            "cached_exact_softplus_gate_bf16_spans"
        ),
        "kernel": (
            "laguna_swa_attention_prefill_qrow4_dense_initial_cached_exact_"
            "softplus_gate_bf16_kernel"
        ),
        "control": (
            "laguna_swa_attention_prefill_qrow4_dense_initial_cached_exact_"
            "bf16_spans"
        ),
        "local_size": 32,
        "runtime_vgpr_max": 72,
    },
    "swa_late": {
        "starts": (256, 384),
        "variant": (
            "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_"
            "softplus_bf16_spans"
        ),
        "function": (
            "laguna_swa_attention_prefill_qrow4_dense_initial_global_score_"
            "replay_exact_softplus_gate_bf16_spans"
        ),
        "symbol": (
            "hipengine_laguna_swa_attention_prefill_qrow4_dense_initial_"
            "global_score_replay_exact_softplus_gate_bf16_spans"
        ),
        "kernel": (
            "laguna_swa_attention_prefill_qrow4_dense_initial_global_score_"
            "replay_exact_softplus_gate_bf16_kernel"
        ),
        "control": (
            "laguna_swa_attention_prefill_qrow4_dense_initial_global_score_"
            "replay_exact_bf16_spans"
        ),
        "local_size": 32,
        "runtime_vgpr_max": 64,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _module():
    return importlib.import_module(
        "hipengine.kernels.hip_gfx1100.attention.laguna_kv"
    )


def _candidate_functions() -> dict[str, Callable[..., None]]:
    module = _module()
    return {
        name: getattr(module, str(contract["function"]))
        for name, contract in _ROUTES.items()
    }


def _source_path() -> Path:
    return Path(_module().__file__).with_name("laguna_kv_attention.hip")


def _extract_braced(source: str, declaration: str) -> str:
    start = source.index(declaration)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated body: {declaration}")


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _global_spans(capacity: int = _GLOBAL_CAPACITY) -> KVLiveSpans:
    block_size = 256
    return KVLiveSpans.paged_dense(
        block_table=_tensor(
            0x21000,
            ((capacity + block_size - 1) // block_size,),
            "int32",
        ),
        live_counts=_tensor(0x22000, (1,), "int64"),
        token_positions=_tensor(0x23000, (capacity,), "int64"),
        evict_mask=_tensor(0x24000, (capacity,), "bool"),
        row_positions=_tensor(0x25000, (1,), "int64"),
        capacity=capacity,
        block_size=block_size,
        storage_dtype="bf16",
        span_role="prefill",
    )


def _swa_spans(capacity: int = _SWA_CAPACITY) -> KVLiveSpans:
    return KVLiveSpans.sliding_ring(
        base_offsets=_tensor(0x11000, (capacity,), "int32"),
        live_counts=_tensor(0x12000, (1,), "int64"),
        token_positions=_tensor(0x13000, (capacity,), "int64"),
        evict_mask=_tensor(0x14000, (capacity,), "bool"),
        row_positions=_tensor(0x15000, (1,), "int64"),
        capacity=capacity,
        storage_dtype="bf16",
    )


def test_h8h_target_source_and_admission_are_frozen() -> None:
    assert _sha256(_TARGET) == _TARGET_SHA256
    artifact = json.loads(_TARGET.read_text())
    assert artifact["target_id"] == "WPF-H8H"
    assert artifact["status"] == "selected_target_only"
    assert artifact["performance_claim"] is False
    assert artifact["target"] == {
        "allocation_and_workspace_growth_bytes": 0,
        "attention_calls_retained": 192,
        "bf16_gated_output_bytes_retained": 415_236_096,
        "bf16_gated_output_elements_retained": 207_618_048,
        "context_f32_output_retained": True,
        "dispatches_after_modeled": 2107,
        "dispatches_before": 2155,
        "f32_context_bytes_reread_removed": 830_472_192,
        "f32_context_elements_reread_removed": 207_618_048,
        "gate_only_zero_increment_ceiling_percent": 0.343753597116514,
        "gate_only_zero_increment_ceiling_tok_s": 442.4082081037605,
        "id": "WPF-H8H",
        "kv_live_spans_abi_retained": True,
        "name": "exact prefill attention plus softplus dual-output publication",
        "not_a_candidate_or_speed_claim": True,
        "scope": (
            "all H6N/H6Z global and H6A/H6W SWA production M128 slices "
            "across all 48 layers"
        ),
        "separate_gate_calls_removed": 48,
        "softplus_and_multiply_per_element_retained": True,
        "unfused_fallback_retained": True,
    }
    assert artifact["admission"]["red_first"] is True
    assert "partial four-route subset" in artifact["admission"]["no_salvage"]
    assert artifact["decision"] == {
        "candidate_implemented": False,
        "next_action": (
            "commit this target-only boundary, then freeze RED before any "
            "HIP/Python/registry/package/runtime edit"
        ),
        "production_changed": False,
        "red_test_added": False,
        "runtime_owner_implemented": False,
        "source_default_changed": False,
    }
    for relative, expected in artifact["source_sha256"].items():
        assert _sha256(Path(relative)) == expected
    source = _source_path().read_text()
    wrapper = Path(_module().__file__).read_text()
    for contract in _ROUTES.values():
        assert str(contract["function"]) not in wrapper
        assert str(contract["symbol"]) not in source
        assert str(contract["kernel"]) not in source


def test_h8h_registry_static_resources_and_backend_exclusion() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    candidates = _candidate_functions()
    module = _module()
    module.register_laguna_kv_attention_kernels(replace=True)
    source = _source_path().read_text()
    helper = _extract_braced(
        source,
        "__device__ __forceinline__ void "
        "laguna_publish_prefill_context_softplus_gate_bf16(",
    )
    for statement in (
        "out[output_idx] = context_value;",
        "context_value * laguna_softplus_f32(gate[head_row])",
        "gated_out[output_idx] = laguna_float_to_bf16_bits(",
    ):
        assert statement in helper
    for name, contract in _ROUTES.items():
        candidate = candidates[name]
        assert (
            resolve(
                backend="hip_gfx1100",
                layer=_LAYER,
                quant=_QUANT,
                variant=str(contract["variant"]),
            )
            is candidate
        )
        body = _extract_braced(source, f"void {contract['kernel']}(")
        assert "const float* gate" in body
        assert "uint16_t* gated_out" in body
        assert "laguna_publish_prefill_context_softplus_gate_bf16(" in body
        assert body.count("laguna_publish_prefill_context_softplus_gate_bf16(") >= 1
        assert str(contract["symbol"]) in source
    load_backend_kernel_package("hip_gfx1151")
    for contract in _ROUTES.values():
        assert not is_registered(
            KernelKey("hip_gfx1151", _LAYER, _QUANT, str(contract["variant"]))
        )


def test_h8h_strict_preflight_and_raw_pointer_abi() -> None:
    candidates = _candidate_functions()
    calls: dict[str, list[tuple[object, ...]]] = {name: [] for name in _ROUTES}

    class FakeFn:
        argtypes = None
        restype = None

        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, *args: object) -> int:
            calls[self.name].append(args)
            return 0

    library = SimpleNamespace(
        **{
            str(contract["symbol"]): FakeFn(name)
            for name, contract in _ROUTES.items()
        }
    )

    def launch(name: str, **overrides: object) -> None:
        global_route = name.startswith("global")
        arguments: dict[str, object] = {
            "spans": _global_spans() if global_route else _swa_spans(),
            "rows": _ROWS,
            "capacity": _GLOBAL_CAPACITY if global_route else _SWA_CAPACITY,
            "num_q_heads": _GLOBAL_HEADS if global_route else _SWA_HEADS,
            "num_kv_heads": _KV_HEADS,
            "head_dim": _HEAD_DIM,
            "start_position": int(_ROUTES[name]["starts"][0]),
            "score_scratch_nbytes": _SCORE_SCRATCH_BYTES.get(name),
        }
        arguments.update(overrides)
        common = [0x6000, 0x7000, 0x8000, 0x9000, 0xA000, 0xB000, 0xC000, 0xD000]
        if name.endswith("late"):
            common.append(0xE000)
        candidates[name](
            *common,
            arguments.pop("spans"),
            int(arguments.pop("rows")),
            int(arguments.pop("capacity")),
            int(arguments.pop("num_q_heads")),
            int(arguments.pop("num_kv_heads")),
            int(arguments.pop("head_dim")),
            _HEAD_DIM**-0.5,
            **arguments,
            library=library,
            runtime=SimpleNamespace(),
        )

    for name, contract in _ROUTES.items():
        for start in contract["starts"]:
            launch(name, start_position=start)
        assert len(calls[name]) == 2

    invalid = (
        {"rows": 127},
        {"capacity": 2048},
        {"num_q_heads": 24},
        {"num_kv_heads": 4},
        {"head_dim": 64},
        {"start_position": 64},
    )
    for name in _ROUTES:
        for bad in invalid:
            with pytest.raises(ValueError):
                launch(name, **bad)
        if name.endswith("late"):
            with pytest.raises(ValueError):
                launch(name, score_scratch_nbytes=16)


def _span_snapshot(spans: KVLiveSpans, runtime: Any) -> tuple[bytes, ...]:
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    outputs = []
    for tensor in (
        spans.base_offsets,
        spans.live_counts,
        spans.token_positions,
        spans.evict_mask,
        spans.row_positions,
    ):
        host = np.empty(tensor.nbytes, dtype=np.uint8)
        copy_device_to_host(
            host_array_ptr(host), tensor, tensor.nbytes, runtime=runtime
        )
        outputs.append(host.tobytes())
    return tuple(outputs)


def _run_route(
    name: str,
    candidate: Callable[..., None],
    *,
    start_position: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
        memory_stats,
    )
    from hipengine.kernels.hip_gfx1100.fused.laguna_attention import (
        build_laguna_attention,
        laguna_softplus_head_gate_f32_bf16_out,
    )
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    module = _module()
    runtime = get_hip_runtime()
    before = memory_stats()
    library = module.build_laguna_kv_attention(
        load=True, require_cached=_require_cached_build()
    )
    gate_library = build_laguna_attention(
        load=True, require_cached=_require_cached_build()
    )
    global_route = name.startswith("global")
    heads = _GLOBAL_HEADS if global_route else _SWA_HEADS
    capacity = _GLOBAL_CAPACITY if global_route else _SWA_CAPACITY
    layer_type = FULL_ATTENTION if global_route else SLIDING_ATTENTION
    config = SimpleNamespace(
        block_count=1,
        layer_types=(layer_type,),
        head_counts=(heads,),
        head_count_kv=_KV_HEADS,
        key_length=_HEAD_DIM,
        value_length=_HEAD_DIM,
        sliding_window=_SWA_CAPACITY,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=_GLOBAL_CAPACITY,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    total_rows = start_position + _ROWS
    rng = np.random.default_rng(rng_seed)
    keys = rng.normal(0.0, 0.12, (total_rows, _KV_HEADS, _HEAD_DIM)).astype(np.float32)
    values = rng.normal(0.0, 0.12, keys.shape).astype(np.float32)
    queries = rng.normal(0.0, 0.12, (_ROWS, heads, _HEAD_DIM)).astype(np.float32)
    gates = rng.normal(0.0, 2.0, (_ROWS, heads)).astype(np.float32)
    gates.flat[:8] = np.array(
        [-30.0, -20.0, -1.0, 0.0, 1.0, 19.0, 20.0, 21.0],
        dtype=np.float32,
    )
    context_bytes = queries.nbytes
    gated_bytes = _ROWS * heads * _HEAD_DIM * 2
    allocations = []
    try:
        key_device = malloc(keys.nbytes, runtime=runtime)
        value_device = malloc(values.nbytes, runtime=runtime)
        query_device = malloc(queries.nbytes, runtime=runtime)
        gate_device = malloc(gates.nbytes, runtime=runtime)
        control_context = malloc(context_bytes, runtime=runtime)
        candidate_context = malloc(context_bytes, runtime=runtime)
        control_gated = malloc(gated_bytes, runtime=runtime)
        candidate_gated = malloc(gated_bytes, runtime=runtime)
        allocations.extend(
            (
                key_device,
                value_device,
                query_device,
                gate_device,
                control_context,
                candidate_context,
                control_gated,
                candidate_gated,
            )
        )
        scratch = None
        if name.endswith("late"):
            scratch = malloc(_SCORE_SCRATCH_BYTES[name], runtime=runtime)
            allocations.append(scratch)
        for device, host in (
            (key_device, keys),
            (value_device, values),
            (query_device, queries),
            (gate_device, gates),
        ):
            copy_host_to_device(
                device, host_array_ptr(host), host.nbytes, runtime=runtime
            )
        for output in (
            control_context,
            candidate_context,
            control_gated,
            candidate_gated,
        ):
            runtime.memset(output.ptr, 0xA5, output.nbytes)
        kv_row_nbytes = _KV_HEADS * _HEAD_DIM * 4
        if start_position:
            cache.prepare_rows(tuple(range(start_position)))
            cache.append_rows(
                0,
                key_device.ptr,
                value_device.ptr,
                start_position,
                library=library,
            )
            cache.commit_rows()
        cache.prepare_rows(tuple(range(start_position, total_rows)))
        current_key = key_device.ptr + start_position * kv_row_nbytes
        current_value = value_device.ptr + start_position * kv_row_nbytes
        cache.append_rows(
            0, current_key, current_value, _ROWS, library=library
        )
        state = cache.layer(0)
        spans_before = _span_snapshot(state.spans, runtime)
        control = getattr(module, str(_ROUTES[name]["control"]))
        base = [
            query_device.ptr,
            current_key,
            current_value,
            state.key_cache.ptr,
            state.value_cache.ptr,
        ]
        suffix = [
            state.spans,
            _ROWS,
            capacity,
            heads,
            _KV_HEADS,
            _HEAD_DIM,
            _HEAD_DIM**-0.5,
        ]
        if scratch is None:
            control(
                *base,
                control_context.ptr,
                *suffix,
                start_position=start_position,
                library=library,
                runtime=runtime,
            )
        else:
            control(
                *base,
                control_context.ptr,
                scratch.ptr,
                *suffix,
                score_scratch_nbytes=scratch.nbytes,
                start_position=start_position,
                library=library,
                runtime=runtime,
            )
        laguna_softplus_head_gate_f32_bf16_out(
            control_context.ptr,
            gate_device.ptr,
            control_gated.ptr,
            _ROWS,
            heads,
            _HEAD_DIM,
            library=gate_library,
            runtime=runtime,
        )
        fused_base = [
            *base,
            candidate_context.ptr,
            gate_device.ptr,
            candidate_gated.ptr,
        ]
        if scratch is None:
            candidate(
                *fused_base,
                *suffix,
                start_position=start_position,
                library=library,
                runtime=runtime,
            )
        else:
            candidate(
                *fused_base,
                scratch.ptr,
                *suffix,
                score_scratch_nbytes=scratch.nbytes,
                start_position=start_position,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        context_control_host = np.empty(queries.shape, dtype=np.float32)
        context_candidate_host = np.empty(queries.shape, dtype=np.float32)
        gated_shape = (_ROWS, heads, _HEAD_DIM)
        gated_control_host = np.empty(gated_shape, dtype=np.uint16)
        gated_candidate_host = np.empty(gated_shape, dtype=np.uint16)
        for host, device in (
            (context_control_host, control_context),
            (context_candidate_host, candidate_context),
            (gated_control_host, control_gated),
            (gated_candidate_host, candidate_gated),
        ):
            copy_device_to_host(
                host_array_ptr(host), device, host.nbytes, runtime=runtime
            )
        assert _span_snapshot(state.spans, runtime) == spans_before
        cache.discard_rows()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()
    after = memory_stats()
    assert after["active_allocations"] == before["active_allocations"]
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    return (
        context_control_host,
        context_candidate_host,
        gated_control_host,
        gated_candidate_host,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h8h_global_all_starts_f32_context_and_bf16_gate_exact() -> None:
    candidates = _candidate_functions()
    for name in ("global_early", "global_late"):
        for start in _ROUTES[name]["starts"]:
            control, actual, gated_control, gated_actual = _run_route(
                name,
                candidates[name],
                start_position=int(start),
                rng_seed=0x8800 + int(start),
            )
            assert np.isfinite(actual).all()
            np.testing.assert_array_equal(
                actual.view(np.uint32), control.view(np.uint32)
            )
            np.testing.assert_array_equal(gated_actual, gated_control)
            assert not np.all(actual.view(np.uint8) == 0xA5)
            assert not np.all(gated_actual.view(np.uint8) == 0xA5)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h8h_swa_all_starts_f32_context_and_bf16_gate_exact() -> None:
    candidates = _candidate_functions()
    for name in ("swa_early", "swa_late"):
        for start in _ROUTES[name]["starts"]:
            control, actual, gated_control, gated_actual = _run_route(
                name,
                candidates[name],
                start_position=int(start),
                rng_seed=0x8810 + int(start),
            )
            assert np.isfinite(actual).all()
            np.testing.assert_array_equal(
                actual.view(np.uint32), control.view(np.uint32)
            )
            np.testing.assert_array_equal(gated_actual, gated_control)
            assert not np.all(actual.view(np.uint8) == 0xA5)
            assert not np.all(gated_actual.view(np.uint8) == 0xA5)
