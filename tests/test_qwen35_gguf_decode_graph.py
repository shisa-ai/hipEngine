from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen35_gguf_runner as gguf_runner
from hipengine.core.dtype import DType
from hipengine.kernels.policy import (
    QWEN35_DENSE_H5120_GEOMETRY,
    QWEN35_MOE_H2048_E256_GEOMETRY,
)
from hipengine.runtime.gguf_decode_graph import (
    Qwen35GGUFDecodeGraph,
    _decode_graph_kv_layout_admitted,
    build_qwen35_gguf_decode_graph_key,
)
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


def test_decode_graph_submission_transport_uses_backend_default_with_explicit_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = (QWEN35_MOE_H2048_E256_GEOMETRY, "MOSTLY_Q4_K_M")
    policies = {
        "hip_gfx1100": {
            identity: {
                "transport": "pm4",
                "min_replay_steps_by_physical_rows": {1: 160, 2: 64, 4: 96, 8: 80},
            }
        },
        "hip_gfx1151": {identity: {"transport": "hipgraph"}},
    }

    monkeypatch.setattr(
        gguf_runner,
        "backend_package_capability",
        lambda backend, name, default=None: policies.get(backend, default)
        if name == "GGUF_DECODE_GRAPH_SUBMISSION_POLICIES"
        else default,
    )

    resolve = gguf_runner._resolve_gguf_decode_graph_submission_transport
    common = {"geometry": identity[0], "file_type_name": identity[1]}
    assert resolve(
        "hip_gfx1100",
        **common,
        physical_rows=2,
        replay_steps=128,
        env={},
    ) == "pm4"
    assert resolve(
        "hip_gfx1100",
        **common,
        physical_rows=1,
        replay_steps=128,
        env={},
    ) == "hipgraph"
    assert resolve(
        "hip_gfx1100",
        **common,
        physical_rows=1,
        replay_steps=160,
        env={},
    ) == "pm4"
    assert resolve(
        "hip_gfx1100",
        **common,
        physical_rows=2,
        replay_steps=63,
        env={},
    ) == "hipgraph"
    assert resolve("hip_gfx1100", geometry=None, env={}) == "hipgraph"
    assert resolve(
        "hip_gfx1100",
        geometry=None,
        replay_steps=1,
        env={"HIPENGINE_SUBMISSION_TRANSPORT": "pm4"},
    ) == "pm4"
    assert resolve(
        "hip_gfx1151",
        **common,
        physical_rows=2,
        replay_steps=128,
        env={},
    ) == "hipgraph"
    assert resolve(
        "hip_gfx1100",
        **common,
        physical_rows=2,
        replay_steps=128,
        env={"HIPENGINE_SUBMISSION_TRANSPORT": "hipgraph"},
    ) == "hipgraph"
    assert resolve(
        "hip_gfx1100",
        **common,
        physical_rows=2,
        requested="aql",
        env={"HIPENGINE_SUBMISSION_TRANSPORT": "hipgraph"},
    ) == "aql"


def test_resident_session_reselects_shape_scoped_default_per_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.core.pm4.transport as transport_module
    import hipengine.runtime.gguf_decode_graph as decode_graph_module

    identity = (QWEN35_MOE_H2048_E256_GEOMETRY, "MOSTLY_Q4_K_M")
    policy = {
        identity: {
            "transport": "pm4",
            "min_replay_steps_by_physical_rows": {1: 160},
        }
    }
    calls: list[tuple[object, ...]] = []
    graphs = [SimpleNamespace(name="short"), SimpleNamespace(name="long")]
    monkeypatch.setattr(gguf_runner, "_gguf_moe_graph_enabled", lambda: False)
    monkeypatch.setattr(
        gguf_runner,
        "backend_package_capability",
        lambda backend, name, default=None: policy
        if name == "GGUF_DECODE_GRAPH_SUBMISSION_POLICIES"
        else default,
    )
    monkeypatch.setattr(
        transport_module,
        "create_graph_submission_context",
        lambda **kwargs: calls.append(("context", kwargs["transport"]))
        or SimpleNamespace(name=kwargs["transport"]),
    )
    monkeypatch.setattr(
        decode_graph_module,
        "capture_qwen35_gguf_decode_graph",
        lambda session, **kwargs: calls.append(
            ("capture", kwargs["submission_transport"])
        )
        or graphs.pop(0),
    )
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weights=SimpleNamespace(geometry=identity[0], file_type_name=identity[1]),
    )
    session.runtime = object()
    session._decode_graph_submission_contexts = {}
    session._pin_device_kv_graph = lambda graph: None

    short = session.capture_decode_graph(position=4, max_replay_steps=159)
    long = session.capture_decode_graph(position=4, max_replay_steps=160)

    assert short.name == "short"
    assert long.name == "long"
    assert calls == [
        ("context", "hipgraph"),
        ("capture", "hipgraph"),
        ("context", "pm4"),
        ("capture", "pm4"),
    ]


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


def test_decode_graph_delegates_replay_provenance_and_close_to_submission_owner() -> None:
    calls: list[tuple[object, ...]] = []

    class FakeSubmission:
        name = "pm4"
        graph_exec = 17

        def launch(self, stream: int) -> None:
            calls.append(("submit", stream))

        def provenance(self) -> dict[str, object]:
            return {"transport": "pm4", "launches": 1, "native_fallbacks": 0}

        def close(self) -> None:
            calls.append(("submission_close",))

    runtime = SimpleNamespace(
        stream_synchronize=lambda stream: calls.append(("stream_sync", int(stream))),
        graph_destroy=lambda handle: calls.append(("graph_destroy", int(handle))),
        stream_destroy=lambda handle: calls.append(("stream_destroy", int(handle))),
    )
    scratch = SimpleNamespace(
        position_host=np.zeros((1,), dtype=np.int64),
        context_host=np.zeros((1,), dtype=np.int64),
    )
    unpinned: list[object] = []
    session = SimpleNamespace(
        position=0,
        _position=0,
        runtime=runtime,
        scratch=scratch,
        _decode_graphs=[],
    )
    session._unpin_device_kv_graph = lambda graph: unpinned.append(graph)
    graph = Qwen35GGUFDecodeGraph(
        session=session,
        graph=11,
        graph_exec=17,
        stream=13,
        position=0,
        steps_per_replay=1,
        max_replay_steps=1,
        generated=None,
        generated_hidden_seeds=None,
        generated_index=None,
        record_steps=0,
        bucket_key=SimpleNamespace(key_sha256="decode-key"),
        attention_max_context_len=1,
        capture_hidden_seed_fp32=False,
        submission=FakeSubmission(),
    )
    session._decode_graphs.append(graph)

    graph.replay(1)
    provenance = graph.transport_provenance()
    graph.close()

    assert graph.replayed_steps == 1
    assert session._position == 1
    assert scratch.position_host.tolist() == [1]
    assert provenance == {
        "transport": "pm4",
        "launches": 1,
        "native_fallbacks": 0,
        "decode_graph_key_sha256": "decode-key",
        "replayed_steps": 1,
    }
    assert calls == [
        ("submit", 13),
        ("stream_sync", 13),
        ("submission_close",),
        ("graph_destroy", 11),
        ("stream_destroy", 13),
    ]
    assert graph.graph_exec == 0
    assert unpinned == [graph]


def test_decode_graph_rearm_replay_window_requires_restored_capture_cursor() -> None:
    session = SimpleNamespace(position=512)
    graph = Qwen35GGUFDecodeGraph(
        session=session,
        graph=11,
        graph_exec=12,
        stream=13,
        position=512,
        steps_per_replay=1,
        max_replay_steps=128,
        generated=None,
        generated_hidden_seeds=None,
        generated_index=None,
        record_steps=0,
        bucket_key=SimpleNamespace(),
        attention_max_context_len=640,
        capture_hidden_seed_fp32=False,
        replayed_steps=128,
    )

    graph.rearm_replay_window()

    assert graph.replayed_steps == 0
    graph.replayed_steps = 128
    session.position = 513
    with pytest.raises(RuntimeError, match="capture cursor"):
        graph.rearm_replay_window()
    assert graph.replayed_steps == 128

    session.position = 512
    graph.record_steps = 128
    with pytest.raises(RuntimeError, match="recording"):
        graph.rearm_replay_window()
    assert graph.replayed_steps == 128


def test_decode_graph_input_seed_updates_feedback_buffer_before_cross_stream_capture(
    monkeypatch,
) -> None:
    calls: list[tuple] = []
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(vocab_size=128)
    session._lm_out_index = _buffer(0x1234)
    session._runtime_state_library = object()
    session.runtime = SimpleNamespace(
        device_synchronize=lambda: calls.append(("synchronize",)),
    )
    monkeypatch.setattr(
        gguf_runner,
        "set_i64_scalar",
        lambda ptr, value, **kwargs: calls.append(
            ("seed", int(ptr), int(value), int(kwargs.get("stream", -1)))
        ),
    )

    session._seed_decode_graph_input_token(42)

    assert calls == [("seed", 0x1234, 42, 0), ("synchronize",)]


def test_resident_session_reuses_one_submission_context_across_graph_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.core.pm4.transport as transport_module
    import hipengine.runtime.gguf_decode_graph as decode_graph_module

    calls: list[tuple[object, ...]] = []
    context = object()
    graphs = [SimpleNamespace(name="first"), SimpleNamespace(name="second")]
    monkeypatch.setattr(gguf_runner, "_gguf_moe_graph_enabled", lambda: False)
    monkeypatch.setattr(
        transport_module,
        "select_submission_transport",
        lambda value=None, **_kwargs: "pm4",
    )
    monkeypatch.setattr(
        transport_module,
        "create_graph_submission_context",
        lambda **kwargs: calls.append(("create_context", kwargs["backend"], kwargs["gfx_arch"]))
        or context,
    )
    monkeypatch.setattr(
        decode_graph_module,
        "capture_qwen35_gguf_decode_graph",
        lambda session, **kwargs: calls.append(
            ("capture", kwargs["submission_transport"], kwargs["submission_context"])
        )
        or graphs.pop(0),
    )
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(backend="hip_gfx1100", target_arch="gfx1100")
    session.runtime = object()
    session._decode_graph_submission_contexts = {}
    session._pin_device_kv_graph = lambda graph: calls.append(("pin", graph.name))

    first = session.capture_decode_graph(position=4, submission_transport="pm4")
    second = session.capture_decode_graph(position=5, submission_transport="pm4")

    assert first.name == "first"
    assert second.name == "second"
    assert calls == [
        ("create_context", "hip_gfx1100", "gfx1100"),
        ("capture", "pm4", context),
        ("pin", "first"),
        ("capture", "pm4", context),
        ("pin", "second"),
    ]


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


def test_packed_decode_graph_minimum_uses_model_width_policy(monkeypatch) -> None:
    identity = (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M")

    def capability(backend: str, name: str, default=None):
        assert backend == "hip_gfx1151"
        if name == "GGUF_PACKED_DECODE_GRAPH_MIN_REPLAY_STEPS_BY_POLICY":
            return {identity: {2: 23}}
        return default

    monkeypatch.setattr(gguf_runner, "backend_package_capability", capability)
    session = object.__new__(Qwen35GGUFResidentSession)
    session.backend = "auto"
    session.runner = SimpleNamespace(
        backend="hip_gfx1151",
        weights=SimpleNamespace(
            geometry=identity[0],
            file_type_name=identity[1],
        ),
    )
    session._decode_graph_min_replay_steps_cache = 128

    assert session.packed_decode_graph_min_replay_steps(1) == 128
    assert session.packed_decode_graph_min_replay_steps(2) == 23
    assert session.packed_decode_graph_min_replay_steps(4) == 32
