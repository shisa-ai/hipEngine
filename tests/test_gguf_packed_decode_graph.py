from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.gguf_packed_decode_graph as packed_graph
from hipengine.runtime.gguf_packed_decode_graph import (
    build_qwen35_gguf_packed_decode_graph_key,
)
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


def _buffer(ptr: int):
    return SimpleNamespace(ptr=ptr)


def _owner(*, positions: tuple[int, ...] = (512, 516), packed_ptr: int = 100):
    weights = [
        SimpleNamespace(
            spec=SimpleNamespace(
                slot_path="layers.0.attn_q",
                quant_key="gguf_q8_0_t16_v1",
                source=SimpleNamespace(shape=(2048, 2048)),
            ),
            allocations={"t16": SimpleNamespace(tensor=_buffer(200))},
        )
    ]
    config = SimpleNamespace(
        layer_types=("linear_attention", "full_attention"),
        context_length=4096,
    )
    owner = SimpleNamespace(
        backend="hip_gfx1100",
        model_path="/models/example.gguf",
        runner=SimpleNamespace(
            target_arch="gfx1100",
            hidden_size=2048,
            vocab_size=248320,
            weights=SimpleNamespace(weights=weights, config=config),
        ),
        kv_storage_dtype=SimpleNamespace(value="bf16"),
        kv_storage_layout="uniform",
        kv_scale_dtype=SimpleNamespace(value="fp16"),
        kv_scale_granularity="per_token_head",
        use_wmma_prefill=True,
        use_gemv_decode=True,
        host_token_embedding_enabled=False,
        _lm_head_threads=128,
        _lm_head_stage1_blocks=970,
    )
    sessions = tuple(SimpleNamespace(position=position) for position in positions)
    return owner, sessions, (packed_ptr, packed_ptr + 1, packed_ptr + 2)


def _key(
    *,
    positions: tuple[int, ...] = (512, 516),
    packed_ptr: int = 100,
    active_mask: tuple[bool, ...] = (True, True),
):
    owner, sessions, pointers = _owner(positions=positions, packed_ptr=packed_ptr)
    return build_qwen35_gguf_packed_decode_graph_key(
        owner,
        sessions=sessions,
        active_mask=active_mask,
        block_size=256,
        max_positions=1024,
        steps_per_replay=1,
        max_replay_steps=128,
        record_steps=128,
        record_layer_ids=(0, 1),
        packed_buffer_ptrs=pointers,
    )


def test_packed_decode_graph_key_covers_width_state_mask_and_buffers() -> None:
    first = _key()
    same = _key()
    next_state = _key(positions=(513, 517))
    next_buffer = _key(packed_ptr=999)
    next_mask = _key(active_mask=(True, False))

    assert first == same
    assert first.active_rows == 2
    assert first.active_mask == (True, True)
    assert first.state_generations == (512, 516)
    assert first.replay_context_limit == 644
    assert first.context_bucket == 768
    assert first.record_steps == 128
    assert first.record_layer_ids == (0, 1)
    assert first.key_sha256 != next_state.key_sha256
    assert first.key_sha256 != next_buffer.key_sha256
    assert first.key_sha256 != next_mask.key_sha256


def test_packed_decode_graph_key_serializes_complete_route_axes() -> None:
    payload = _key().as_dict()

    assert payload["backend"] == "hip_gfx1100"
    assert payload["target_arch"] == "gfx1100"
    assert payload["kv_storage_dtype"] == "bf16"
    assert payload["layer_types"] == ["linear_attention", "full_attention"]
    assert payload["decode_repack"] is True
    assert payload["metadata_prepare_path"] == "device_positions_persistent"
    assert payload["token_feedback_path"] == "device_i32_to_i64"
    assert payload["buffer_count"] == 3
    assert len(payload["buffer_identity_sha256"]) == 64
    assert len(payload["weight_role_sha256"]) == 64
    assert len(payload["key_sha256"]) == 64


def test_resident_session_delegates_packed_graph_capture(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_capture(owner, **kwargs):
        observed["owner"] = owner
        observed.update(kwargs)
        return "graph"

    monkeypatch.setattr(
        packed_graph,
        "capture_qwen35_gguf_packed_decode_graph",
        fake_capture,
        raising=False,
    )
    owner = object.__new__(Qwen35GGUFResidentSession)
    peer = object.__new__(Qwen35GGUFResidentSession)

    result = owner.capture_packed_decode_graph(
        (11, 22),
        sessions=(owner, peer),
        steps_per_replay=2,
        max_replay_steps=8,
        record_steps=8,
        record_layer_output_hidden=(0, 3),
    )

    assert result == "graph"
    assert observed == {
        "owner": owner,
        "token_ids": (11, 22),
        "sessions": (owner, peer),
        "steps_per_replay": 2,
        "max_replay_steps": 8,
        "record_steps": 8,
        "record_layer_output_hidden": (0, 3),
    }


def test_packed_decode_graph_key_rejects_invalid_shape_contract() -> None:
    owner, sessions, pointers = _owner()
    with pytest.raises(ValueError, match="active_mask"):
        build_qwen35_gguf_packed_decode_graph_key(
            owner,
            sessions=sessions,
            active_mask=(True,),
            block_size=256,
            max_positions=1024,
            steps_per_replay=1,
            max_replay_steps=128,
            record_steps=0,
            record_layer_ids=(),
            packed_buffer_ptrs=pointers,
        )
    with pytest.raises(ValueError, match="at least one active"):
        build_qwen35_gguf_packed_decode_graph_key(
            owner,
            sessions=sessions,
            active_mask=(False, False),
            block_size=256,
            max_positions=1024,
            steps_per_replay=1,
            max_replay_steps=128,
            record_steps=0,
            record_layer_ids=(),
            packed_buffer_ptrs=pointers,
        )
