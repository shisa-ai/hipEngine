"""WPF-H7E bounded zero-growth runtime ownership RED contract."""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_source_mmq_prefill import (
    gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime.laguna_gguf_runner import (
    LagunaEagerLibraries,
    LagunaPrefillChunkPolicy,
    LagunaPrefillScratchPlan,
)
import hipengine.runtime.laguna_moe as laguna_moe_module
from hipengine.runtime.laguna_moe import (
    laguna_moe_scratch_nbytes,
    resolve_laguna_moe_plan,
)
from tests._laguna_synthetic import make_laguna_info

_QUALIFIED_CAPABILITY = "LAGUNA_GROUPED_IQ_DOWN_H7E_VARIANTS"
_LIVE_CAPABILITY = "LAGUNA_GROUPED_IQ_DOWN_RESIDUAL_VARIANTS"
_QUANT = "gguf_iq3_xxs"
_IQ4_QUANT = "gguf_iq4_xs"
_H7E_VARIANT = (
    "selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out"
)
_H6T_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_rowbatch8_"
    "bf16_bf16_out"
)
_H5J_IQ4_VARIANT = "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out"
_H7E_KEY = KernelKey("hip_gfx1100", "moe_linear", _QUANT, _H7E_VARIANT)
_H7E_ABI = "grouped_source_mmq_d4x2"
_SOURCE_ABI = "grouped_raw_iq_active_experts"
_SOURCE_LIBRARY = "grouped_iq_prefill"
_H7E_LIBRARY = "iq_source_mmq"
_PRODUCER_LIBRARY = "q8_mmq_prefill"
_PRODUCTION_MOE_SCRATCH_BYTES = 104_370_208
_PRODUCTION_WORKSPACE_BYTES = 161_120_256
_PRODUCTION_TOTAL_SCRATCH_BYTES = 600_141_856
_EXPERT_GATE_UP_BYTES = 20_971_520
_H7E_D4X2_BYTES = 11_796_480
_MMQ128_STATIC_ROWS = 37_632
_MMQ128_TILE_CAPACITY = 294
_ARTIFACT = Path(
    "benchmarks/results/"
    "2026-08-02-gfx1100-laguna-q2-xl-iq3-d4x2-source-mmq-candidate.json"
)
_PROMPTS = Path("benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl")
_PROMPTS_SHA256 = "3097ed25c6f4cf3c2986c1da90e61d1600c3b291745224313dba5100fa7a8e76"


def _prefill_scratch(config, plan) -> LagunaPrefillScratchPlan:
    return LagunaPrefillScratchPlan.build(
        config,
        plan,
        policy=LagunaPrefillChunkPolicy.resolve(
            context_length=4096,
            matrix_rows=512,
            attention_rows=128,
        ),
        use_q5_f32_ordered=True,
        use_q5_activation_tile_k_row=True,
    )


def _buffer(ptr: int, nbytes: int = 4096) -> SimpleNamespace:
    return SimpleNamespace(ptr=ptr, nbytes=nbytes)


def _weight(quant: str, ptr: int) -> SimpleNamespace:
    allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=ptr))
    return SimpleNamespace(
        spec=SimpleNamespace(quant_key=quant),
        allocation=lambda name: allocation if name == "raw" else None,
    )


def _layer(quant: str, ptr: int = 71) -> SimpleNamespace:
    weight = _weight(quant, ptr)
    return SimpleNamespace(weight=lambda name: weight if name == "ffn_down_exps" else None)


def _route(name: str, calls: list[tuple[str, tuple[object, ...], dict[str, object]]], *, abi: str, library_key: str):
    def function(*args: object, **kwargs: object) -> None:
        calls.append((name, args, kwargs))

    return SimpleNamespace(
        function=function,
        abi=abi,
        allocation_name="raw",
        library_key=library_key,
        key=_H7E_KEY,
    )


def _record(calls: list[tuple[str, tuple[object, ...], dict[str, object]]], name: str):
    def function(*args: object, **kwargs: object) -> None:
        calls.append((name, args, kwargs))

    return function


def test_h7e_runtime_capability_is_default_off_bounded_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    source = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert source.grouped_exact_down_keys[_QUANT].variant == _H6T_VARIANT
    assert source.grouped_exact_down_routes[_QUANT].abi == _SOURCE_ABI
    assert source.grouped_exact_down_keys[_IQ4_QUANT].variant == _H5J_IQ4_VARIANT
    assert is_registered(_H7E_KEY)
    assert (
        gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out
        is not None
    )
    assert not is_registered(
        KernelKey("hip_gfx1151", "moe_linear", _QUANT, _H7E_VARIANT)
    )
    source_moe_bytes = laguna_moe_scratch_nbytes(source, max_rows=512)
    source_scratch = _prefill_scratch(config, source)
    assert source_moe_bytes == _PRODUCTION_MOE_SCRATCH_BYTES
    assert source_scratch.q5_f32_ordered_nbytes == _PRODUCTION_WORKSPACE_BYTES
    assert source_scratch.total_nbytes == _PRODUCTION_TOTAL_SCRATCH_BYTES

    # Intentional RED after all retained source/size/registry controls pass.
    qualified = getattr(hip_gfx1100, _QUALIFIED_CAPABILITY)
    live = getattr(hip_gfx1100, _LIVE_CAPABILITY)
    assert qualified == {_QUANT: _H7E_VARIANT}
    assert live == {}
    assert getattr(hip_gfx1151, _QUALIFIED_CAPABILITY) == {}
    assert getattr(hip_gfx1151, _LIVE_CAPABILITY) == {}
    assert source.grouped_residual_down_keys == {}
    assert source.grouped_residual_down_routes == {}

    monkeypatch.setattr(hip_gfx1100, _LIVE_CAPABILITY, dict(qualified))
    candidate = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    key = candidate.grouped_residual_down_keys[_QUANT]
    route = candidate.grouped_residual_down_routes[_QUANT]
    assert key == _H7E_KEY
    assert route.key == key
    assert route.abi == _H7E_ABI
    assert route.allocation_name == "raw"
    assert route.library_key == _H7E_LIBRARY
    assert candidate.grouped_exact_down_keys[_QUANT].variant == _H6T_VARIANT
    assert candidate.grouped_exact_down_keys[_IQ4_QUANT].variant == _H5J_IQ4_VARIANT
    assert laguna_moe_scratch_nbytes(candidate, max_rows=512) == source_moe_bytes
    candidate_scratch = _prefill_scratch(config, candidate)
    assert candidate_scratch.q5_f32_ordered_nbytes == source_scratch.q5_f32_ordered_nbytes
    assert candidate_scratch.total_nbytes == source_scratch.total_nbytes

    for wrong_config in (
        replace(config, hidden_size=1024),
        replace(config, expert_feed_forward_length=2048),
        replace(config, expert_count=255),
    ):
        wrong = resolve_laguna_moe_plan(wrong_config, backend="hip_gfx1100")
        assert wrong.grouped_residual_down_keys == {}
        assert wrong.grouped_residual_down_routes == {}
        assert wrong.grouped_exact_down_keys[_QUANT].variant != _H7E_VARIANT

    original_is_registered = laguna_moe_module.is_registered
    monkeypatch.setattr(
        laguna_moe_module,
        "is_registered",
        lambda key: key != _H7E_KEY and original_is_registered(key),
    )
    missing = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert missing.grouped_residual_down_keys == {}
    assert missing.grouped_residual_down_routes == {}
    monkeypatch.setattr(laguna_moe_module, "is_registered", original_is_registered)

    gfx1151 = resolve_laguna_moe_plan(config, backend="hip_gfx1151")
    assert gfx1151.grouped_residual_down_keys == {}
    assert gfx1151.grouped_residual_down_routes == {}
    for malformed, message in (
        (17, "must be a mapping"),
        ({"unknown": _H7E_VARIANT}, "unsupported quant"),
        ({_QUANT: ""}, "non-empty variants"),
    ):
        monkeypatch.setattr(hip_gfx1100, _LIVE_CAPABILITY, malformed)
        with pytest.raises(ValueError, match=message):
            resolve_laguna_moe_plan(config, backend="hip_gfx1100")


def test_h7e_runtime_reuses_dead_gate_up_plane_and_exact_shape_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _H7E_D4X2_BYTES <= _EXPERT_GATE_UP_BYTES
    assert _H7E_D4X2_BYTES == 512 * 10 * (1024 // 128) * 144 * 2
    assert _MMQ128_STATIC_ROWS == 5120 + 127 * 256
    assert _MMQ128_STATIC_ROWS % 128 == 0
    assert _MMQ128_TILE_CAPACITY == _MMQ128_STATIC_ROWS // 128

    # Intentional RED before the frozen launch-order fixture is constructed.
    assert getattr(hip_gfx1100, _QUALIFIED_CAPABILITY) == {_QUANT: _H7E_VARIANT}
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    h7e = _route(calls=calls, name="h7e", abi=_H7E_ABI, library_key=_H7E_LIBRARY)
    h6t = _route(calls=calls, name="h6t", abi=_SOURCE_ABI, library_key=_SOURCE_LIBRARY)
    plan = SimpleNamespace(
        expert_ffn_size=1024,
        hidden_size=3072,
        expert_count=256,
        top_k=10,
        grouped_residual_down_routes={_QUANT: h7e},
        grouped_exact_down_routes={_QUANT: h6t, _IQ4_QUANT: h6t},
        mmq128_tile_map=_record(calls, "tile128"),
        grouped_weighted_sum=_record(calls, "weighted"),
    )
    scratch = SimpleNamespace(
        plan=plan,
        expert_gate=_buffer(11),
        expert_gate_up=_buffer(12, _EXPERT_GATE_UP_BYTES),
        expert_down=_buffer(13),
        grouped_expert_start=_buffer(14),
        grouped_expert_start_mmq32=_buffer(15),
        grouped_sorted_experts=_buffer(16),
        grouped_mmq_total=_buffer(17),
        grouped_sorted_weights=_buffer(18),
        grouped_sorted_lanes=_buffer(19),
        grouped_lane_to_row=_buffer(20),
        routed_output=_buffer(21),
        grouped_active_experts=_buffer(22),
        grouped_active_count=_buffer(23),
    )
    producer = _record(calls, "producer")
    monkeypatch.setattr(
        laguna_moe_module,
        "gguf_q8_0_mmq128_quantize_bf16_d4x2",
        producer,
    )
    libraries = {
        _SOURCE_LIBRARY: object(),
        _H7E_LIBRARY: object(),
        _PRODUCER_LIBRARY: object(),
        "grouped_metadata": object(),
        "grouped_weighted_sum": object(),
    }
    laguna_moe_module._launch_selected_down_grouped_exact(
        _layer(_QUANT, ptr=71),
        scratch,
        tokens=512,
        lanes=5120,
        stream=107,
        runtime=SimpleNamespace(),
        libraries=libraries,
    )
    assert [name for name, _, _ in calls] == [
        "tile128",
        "producer",
        "h7e",
        "weighted",
    ]
    tile = calls[0]
    assert tile[1][:5] == (14, 15, 16, 17, 256)
    assert tile[2]["tile_capacity"] == _MMQ128_TILE_CAPACITY
    producer_call = calls[1]
    assert producer_call[1][:4] == (11, 12, 5120, 1024)
    assert producer_call[2]["stream"] == 107
    assert producer_call[2]["library"] is libraries[_PRODUCER_LIBRARY]
    h7e_call = calls[2]
    assert h7e_call[1][:6] == (12, 14, 15, 16, 71, 13)
    assert h7e_call[2]["compact_rows"] == 5120
    assert h7e_call[2]["in_features"] == 1024
    assert h7e_call[2]["out_features"] == 3072
    assert h7e_call[2]["num_experts"] == 256
    assert h7e_call[2]["mmq_total_rows"] == _MMQ128_STATIC_ROWS
    assert h7e_call[2]["library"] is libraries[_H7E_LIBRARY]

    for tokens, quant in ((511, _QUANT), (513, _QUANT), (512, _IQ4_QUANT)):
        calls.clear()
        laguna_moe_module._launch_selected_down_grouped_exact(
            _layer(quant, ptr=72),
            scratch,
            tokens=tokens,
            lanes=tokens * 10,
            stream=109,
            runtime=SimpleNamespace(),
            libraries=libraries,
        )
        assert [name for name, _, _ in calls] == ["h6t", "weighted"]


def test_h7e_runtime_libraries_are_lazy_and_source_default_stays_exact() -> None:
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    source = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert source.grouped_exact_down_keys[_QUANT].variant == _H6T_VARIANT
    assert source.grouped_exact_down_keys[_IQ4_QUANT].variant == _H5J_IQ4_VARIANT

    # Intentional RED before inspecting the new optional library ownership.
    assert getattr(hip_gfx1100, _LIVE_CAPABILITY) == {}
    names = {field.name for field in fields(LagunaEagerLibraries)}
    assert {_H7E_LIBRARY, _PRODUCER_LIBRARY} <= names
    values = {name: object() for name in names}
    values[_H7E_LIBRARY] = None
    values[_PRODUCER_LIBRARY] = None
    control = LagunaEagerLibraries(**values)
    assert _H7E_LIBRARY not in control.moe
    assert _PRODUCER_LIBRARY not in control.moe
    values[_H7E_LIBRARY] = object()
    values[_PRODUCER_LIBRARY] = object()
    candidate = LagunaEagerLibraries(**values)
    assert candidate.moe[_H7E_LIBRARY] is values[_H7E_LIBRARY]
    assert candidate.moe[_PRODUCER_LIBRARY] is values[_PRODUCER_LIBRARY]
    loader = inspect.getsource(
        __import__(
            "hipengine.runtime.laguna_gguf_runner",
            fromlist=["load_laguna_eager_libraries"],
        ).load_laguna_eager_libraries
    )
    assert _LIVE_CAPABILITY in loader
    assert "if h7e_enabled" in loader


def test_h7e_runtime_keeps_full_quality_and_rejection_contract_binding() -> None:
    artifact = json.loads(_ARTIFACT.read_text())
    assert artifact["status"] == (
        "admitted_standalone_leaf_runtime_source_unchanged_complete_quality_pending"
    )
    assert artifact["production"]["candidate_not_selected"] is True
    quality = artifact["complete_quality_contract"]
    assert quality["prompts"] == 18
    assert quality["teacher_forced_steps"] == 576
    assert quality["max_kl"] == 0.05
    assert quality["min_top1"] == 0.90
    assert quality["poolside_required"] is True
    assert quality["determinism_required"] is True
    assert quality["free_running_horizons"] == [16, 32]
    assert quality["prompt_or_layer_conditioning_allowed"] is False
    assert hashlib.sha256(_PROMPTS.read_bytes()).hexdigest() == _PROMPTS_SHA256

    # Intentional RED after the immutable quality/source evidence passes.
    assert getattr(hip_gfx1100, _QUALIFIED_CAPABILITY) == {_QUANT: _H7E_VARIANT}
    assert getattr(hip_gfx1100, _LIVE_CAPABILITY) == {}
    assert _H7E_VARIANT not in hip_gfx1100.LAGUNA_GROUPED_IQ_DOWN_VARIANTS.values()
