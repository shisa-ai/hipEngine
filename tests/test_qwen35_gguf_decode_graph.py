from __future__ import annotations

from types import SimpleNamespace

import hipengine.runtime.qwen35_gguf_runner as gguf_runner
from hipengine.core.dtype import DType
from hipengine.runtime.gguf_decode_graph import (
    Qwen35GGUFDecodeGraph,
    _decode_graph_kv_layout_admitted,
    build_qwen35_gguf_decode_graph_key,
)
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


def _buffer(ptr: int):
    return SimpleNamespace(ptr=ptr)


def _session(*, position: int = 512, hidden_ptr: int = 100):
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
    scratch = SimpleNamespace(
        block_size=256,
        max_positions=768,
        position_buf=_buffer(1),
        context_buf=_buffer(2),
        hidden_seed_fp32=_buffer(3),
        norm=_buffer(4),
        attn_out=_buffer(5),
        layer_conv_states=(_buffer(6), None),
        layer_recurrent_states=(_buffer(7), None),
        full_key_caches=(None, _buffer(8)),
        full_value_caches=(None, _buffer(9)),
        full_k_scale_caches=(None, _buffer(10)),
        full_v_scale_caches=(None, _buffer(11)),
    )
    return SimpleNamespace(
        backend="hip_gfx1151",
        model_path="/models/example.gguf",
        position=position,
        runner=SimpleNamespace(
            target_arch="gfx1151",
            hidden_size=2048,
            vocab_size=248320,
            weights=SimpleNamespace(weights=weights, config=config),
        ),
        scratch=scratch,
        kv_storage_dtype=SimpleNamespace(value="bf16"),
        kv_storage_layout="uniform",
        kv_scale_dtype=SimpleNamespace(value="fp16"),
        kv_scale_granularity="per_token_head",
        use_wmma_prefill=True,
        use_gemv_decode=True,
        host_token_embedding_enabled=False,
        _hidden_a=_buffer(hidden_ptr),
        _hidden_b=_buffer(101),
        _token_buf=_buffer(102),
        _logits_buf=_buffer(103),
        _lm_block_values=_buffer(104),
        _lm_block_indices=_buffer(105),
        _lm_out_index=_buffer(106),
        _lm_out_value=_buffer(107),
        _lm_head_threads=128,
        _lm_head_stage1_blocks=970,
    )


def test_decode_graph_key_covers_transition_window_state_and_buffers() -> None:
    first = build_qwen35_gguf_decode_graph_key(
        _session(),
        position=512,
        steps_per_replay=1,
        max_replay_steps=128,
    )
    same = build_qwen35_gguf_decode_graph_key(
        _session(),
        position=512,
        steps_per_replay=1,
        max_replay_steps=128,
    )
    next_state = build_qwen35_gguf_decode_graph_key(
        _session(position=513),
        position=513,
        steps_per_replay=1,
        max_replay_steps=127,
    )
    next_buffer = build_qwen35_gguf_decode_graph_key(
        _session(hidden_ptr=999),
        position=512,
        steps_per_replay=1,
        max_replay_steps=128,
    )
    recorded = build_qwen35_gguf_decode_graph_key(
        _session(),
        position=512,
        steps_per_replay=1,
        max_replay_steps=128,
        attention_max_context_len=768,
        record_steps=128,
        capture_hidden_seed_fp32=True,
        extra_buffer_ptrs=(300, 301),
    )

    assert first == same
    assert first.active_rows == 1
    assert first.state_generation == 512
    assert first.replay_context_limit == 640
    assert first.context_bucket == 768
    assert first.key_sha256 != next_state.key_sha256
    assert first.key_sha256 != next_buffer.key_sha256
    assert first.key_sha256 != recorded.key_sha256
    assert recorded.replay_context_limit == 768
    assert recorded.record_steps == 128
    assert recorded.capture_hidden_seed_fp32 is True


def test_decode_graph_key_serializes_complete_shape_state_axes() -> None:
    payload = build_qwen35_gguf_decode_graph_key(
        _session(),
        position=512,
        steps_per_replay=1,
        max_replay_steps=128,
    ).as_dict()

    assert payload["backend"] == "hip_gfx1151"
    assert payload["target_arch"] == "gfx1151"
    assert payload["kv_storage_dtype"] == "bf16"
    assert payload["kv_storage_layout"] == "uniform"
    assert payload["layer_types"] == ["linear_attention", "full_attention"]
    assert payload["decode_repack"] is True
    assert payload["buffer_count"] == 20
    assert len(payload["buffer_identity_sha256"]) == 64
    assert len(payload["weight_role_sha256"]) == 64
    assert len(payload["key_sha256"]) == 64


def test_decode_graph_admits_bf16_and_tail4_hadamard_only() -> None:
    session = _session()
    assert _decode_graph_kv_layout_admitted(session) is True

    session.kv_storage_dtype = DType.INT8_PER_TOKEN_HEAD
    session.kv_storage_layout = "tail4_hadamard_group32"
    session.kv_scale_granularity = "hadamard_group32"
    assert _decode_graph_kv_layout_admitted(session) is True

    session.kv_storage_layout = "uniform"
    assert _decode_graph_kv_layout_admitted(session) is False


def test_decode_graph_close_releases_device_kv_pin_after_destroy() -> None:
    calls: list[tuple[str, int]] = []
    unpinned: list[object] = []
    runtime = SimpleNamespace(
        graph_exec_destroy=lambda handle: calls.append(("exec", int(handle))),
        graph_destroy=lambda handle: calls.append(("graph", int(handle))),
        stream_destroy=lambda handle: calls.append(("stream", int(handle))),
    )
    session = SimpleNamespace(runtime=runtime, _decode_graphs=[])
    session._unpin_device_kv_graph = lambda graph: unpinned.append(graph)
    graph = Qwen35GGUFDecodeGraph(
        session=session,
        graph=11,
        graph_exec=12,
        stream=13,
        position=0,
        steps_per_replay=1,
        max_replay_steps=1,
        generated=None,
        generated_hidden_seeds=None,
        generated_index=None,
        record_steps=0,
        bucket_key=SimpleNamespace(),
        attention_max_context_len=1,
        capture_hidden_seed_fp32=False,
    )
    session._decode_graphs.append(graph)

    graph.close()
    graph.close()

    assert graph.closed is True
    assert session._decode_graphs == []
    assert unpinned == [graph]
    assert calls == [("exec", 12), ("graph", 11), ("stream", 13)]


def test_decode_graph_capability_uses_runner_resolved_backend(monkeypatch) -> None:
    observed: list[str] = []

    def capability(backend: str, name: str):
        observed.append(backend)
        assert name == "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS"
        return 128

    monkeypatch.setattr(gguf_runner, "backend_package_capability", capability)
    monkeypatch.delenv("HIPENGINE_GGUF_MOE_GRAPH", raising=False)
    session = object.__new__(Qwen35GGUFResidentSession)
    session.backend = "auto"
    session.runner = SimpleNamespace(
        backend="hip_gfx1151",
        weights=SimpleNamespace(
            weights=[SimpleNamespace(spec=SimpleNamespace(quant_key="gguf_q8_0_t16_v1"))]
        ),
    )
    session.scratch = SimpleNamespace()
    session.host_token_embedding_enabled = False
    session.kv_storage_dtype = DType.BF16
    session.use_gemv_decode = True

    assert session._resolve_decode_graph_min_replay_steps() == 128
    assert observed == ["hip_gfx1151"]
