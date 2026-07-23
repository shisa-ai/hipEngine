from __future__ import annotations

from types import SimpleNamespace

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime.laguna_decode_graph import (
    LagunaDecodeGraph,
    build_laguna_decode_graph_key,
    build_laguna_decode_replay_ticket,
    capture_laguna_decode_graph,
    laguna_decode_graph_ineligibility,
)


def _buffer(ptr: int, nbytes: int = 8) -> DeviceBuffer:
    return DeviceBuffer(int(ptr), int(nbytes))


def _weight(slot: str, ptr: int, *, quant: str = "gguf_q5_k"):
    source = SimpleNamespace(
        name=f"{slot}.weight",
        shape=(8, 8),
        ggml_type=13,
        nbytes=64,
        data_offset=ptr,
    )
    spec = SimpleNamespace(
        slot_path=slot,
        quant_key=quant,
        layout="raw_gguf",
        source=source,
    )
    allocation = SimpleNamespace(tensor=_buffer(ptr, 64))
    return SimpleNamespace(spec=spec, allocations={"raw": allocation})


def _spans(base: int):
    return SimpleNamespace(
        base_offsets=SimpleNamespace(ptr=base + 1),
        live_counts=SimpleNamespace(ptr=base + 2),
        token_positions=SimpleNamespace(ptr=base + 3),
        evict_mask=SimpleNamespace(ptr=base + 4),
        row_positions=SimpleNamespace(ptr=base + 5),
        max_live_count=512,
        spans_mode="sliding_ring",
    )


def _session(*, hidden_ptr: int = 0x2000, backend: str = "hip_gfx1100"):
    weights = (_weight("root.token_embedding", 0x1000), _weight("layers.0.q", 0x1100))
    config = SimpleNamespace(
        block_count=48,
        hidden_size=3072,
        vocab_size=100352,
        layer_types=tuple("full" if index % 4 == 0 else "swa" for index in range(48)),
        head_counts=tuple(48 if index % 4 == 0 else 72 for index in range(48)),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    resident = SimpleNamespace(config=config, weights=weights)
    scratch_buffers = tuple(_buffer(hidden_ptr + index * 0x100, 64) for index in range(8))
    scratch = SimpleNamespace(
        buffers=scratch_buffers,
        token_id=scratch_buffers[0],
        position=scratch_buffers[1],
        argmax_id=scratch_buffers[2],
        argmax_value=scratch_buffers[3],
    )
    moe = SimpleNamespace(buffers=tuple(_buffer(0x3000 + index * 0x100, 64) for index in range(4)))
    layer = SimpleNamespace(
        key_cache=_buffer(0x4000, 64),
        value_cache=_buffer(0x4100, 64),
        append_spans=_spans(0x4200),
        spans=_spans(0x4200),
        attention_type="swa",
        attention_variant="swa_context_token4_exact_spans",
        write_variant="swa_f32_spans",
    )
    kv = SimpleNamespace(
        layers=(layer,),
        buffers=(_buffer(0x4000, 64), _buffer(0x4100, 64), _buffer(0x4205, 8)),
        row_position=_buffer(0x4205, 8),
        context_length=4096,
        sliding_window=512,
        position=63,
        pending_positions=(),
        graph_compatible=True,
        graph_state_epoch=7,
    )

    def rope(ptr: int) -> SimpleNamespace:
        return SimpleNamespace(
            cos=SimpleNamespace(buffer=_buffer(ptr, 64)),
            sin=SimpleNamespace(buffer=_buffer(ptr + 0x100, 64)),
        )
    return SimpleNamespace(
        backend=backend,
        target_arch="gfx1100",
        context_length=4096,
        position=63,
        selected_down_mode="direct",
        swa_decode_variant="swa_context_token4_exact_spans",
        swa_prefill_variant="swa_context_rows_spans",
        weights=resident,
        scratch=scratch,
        moe_scratch=moe,
        kv_cache=kv,
        full_rope=rope(0x5000),
        swa_rope=rope(0x6000),
        _closed=False,
        _staged_verifier_tokens=None,
        last_result=SimpleNamespace(next_token_id=17),
    )


def test_laguna_decode_graph_key_covers_static_roles_variants_and_pointers() -> None:
    first = build_laguna_decode_graph_key(_session())
    same = build_laguna_decode_graph_key(_session())
    changed_pointer = build_laguna_decode_graph_key(_session(hidden_ptr=0x9000))
    changed_backend = build_laguna_decode_graph_key(
        _session(backend="hip_gfx1151"),
        require_supported_backend=False,
    )

    assert first == same
    assert first.schema_version == 1
    assert first.backend == "hip_gfx1100"
    assert first.target_arch == "gfx1100"
    assert first.active_rows == 1
    assert first.context_length == 4096
    assert first.kv_storage_dtype == "bf16"
    assert first.swa_decode_variant == "swa_context_token4_exact_spans"
    assert first.buffer_count > 20
    assert len(first.role_sha256) == 64
    assert len(first.buffer_identity_sha256) == 64
    assert len(first.key_sha256) == 64
    assert first.key_sha256 != changed_pointer.key_sha256
    assert first.key_sha256 != changed_backend.key_sha256


def test_laguna_decode_replay_ticket_tracks_reprime_and_continuation_epoch() -> None:
    session = _session()
    graph = SimpleNamespace(
        closed=False,
        primed_position=63,
        primed_kv_epoch=7,
    )

    ticket = build_laguna_decode_replay_ticket(session, graph)
    assert ticket.schema_version == 1
    assert ticket.lifecycle_epoch == 7
    assert ticket.session_position == ticket.kv_position == 63
    assert ticket.next_position == 64
    assert ticket.remaining_capacity == 4032
    assert ticket.pending_positions == ()
    assert ticket.dense_span_policy == "cursor_derived_no_eviction"
    assert ticket.graph_compatible is True
    assert ticket.primed is True

    session.kv_cache.graph_state_epoch += 1
    assert build_laguna_decode_replay_ticket(session, graph).primed is False
    graph.primed_kv_epoch = 8
    session.position = session.kv_cache.position = 64
    assert build_laguna_decode_replay_ticket(session, graph).primed is False
    graph.primed_position = 64
    continued = build_laguna_decode_replay_ticket(session, graph)
    assert continued.primed is True
    assert continued.next_position == 65


def test_laguna_decode_graph_eligibility_fails_closed(monkeypatch) -> None:
    import hipengine.runtime.laguna_decode_graph as module

    session = _session()
    monkeypatch.setattr(
        module,
        "backend_package_capability",
        lambda backend, name, default=False: backend == "hip_gfx1100"
        and name == "LAGUNA_DECODE_GRAPH_SUPPORTED",
    )
    assert laguna_decode_graph_ineligibility(session) is None
    assert laguna_decode_graph_ineligibility(session, input_token_id=17) is None
    assert (
        laguna_decode_graph_ineligibility(session, input_token_id=18)
        == "device_token_mismatch"
    )
    session.last_result = None
    assert (
        laguna_decode_graph_ineligibility(session, input_token_id=17)
        == "missing_device_token_owner"
    )
    session.last_result = SimpleNamespace(next_token_id=17)
    assert laguna_decode_graph_ineligibility(session, captures=object()) == "hidden_captures"
    assert laguna_decode_graph_ineligibility(session, stream=7) == "user_stream"

    session._staged_verifier_tokens = (1, 2)
    assert laguna_decode_graph_ineligibility(session) == "staged_verifier"
    session._staged_verifier_tokens = None
    session.kv_cache.pending_positions = (64,)
    assert laguna_decode_graph_ineligibility(session) == "pending_kv_rows"
    session.kv_cache.pending_positions = ()
    session.kv_cache.graph_compatible = False
    assert laguna_decode_graph_ineligibility(session) == "manual_kv_mutation"
    session.kv_cache.graph_compatible = True
    session.kv_cache.position = 62
    assert laguna_decode_graph_ineligibility(session) == "cursor_mismatch"
    session.kv_cache.position = 63
    session.position = 4095
    session.kv_cache.position = 4095
    assert laguna_decode_graph_ineligibility(session) == "context_exhausted"

    unsupported = _session(backend="hip_gfx1151")
    unsupported.target_arch = "gfx1151"
    assert laguna_decode_graph_ineligibility(unsupported) == "backend_not_certified"


def test_laguna_decode_graph_capture_failure_destroys_partial_resources(monkeypatch) -> None:
    import hipengine.runtime.laguna_decode_graph as module

    calls: list[tuple[str, int] | tuple[str]] = []

    class Runtime:
        def stream_create(self) -> int:
            calls.append(("stream_create",))
            return 31

        def stream_begin_capture(self, stream: int) -> None:
            calls.append(("capture_begin", int(stream)))

        def stream_end_capture(self, stream: int) -> int:
            calls.append(("capture_end", int(stream)))
            return 32

        def graph_destroy(self, graph: int) -> None:
            calls.append(("graph_destroy", int(graph)))
            raise RuntimeError("synthetic abandoned graph destroy failure")

        def stream_destroy(self, stream: int) -> None:
            calls.append(("stream_destroy", int(stream)))

    session = SimpleNamespace(
        runtime=Runtime(),
        _decode_graph=None,
        _enqueue_decode_graph_step=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic capture failure")
        ),
    )
    monkeypatch.setattr(module, "laguna_decode_graph_ineligibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "build_laguna_decode_graph_key", lambda *args, **kwargs: object())

    try:
        capture_laguna_decode_graph(session)
    except RuntimeError as exc:
        assert "synthetic capture failure" in str(exc)
    else:
        raise AssertionError("expected synthetic capture failure")

    assert session._decode_graph is None
    assert calls == [
        ("stream_create",),
        ("capture_begin", 31),
        ("capture_end", 31),
        ("graph_destroy", 32),
        ("stream_destroy", 31),
    ]


def test_laguna_decode_graph_instantiate_failure_destroys_graph_and_stream(monkeypatch) -> None:
    import hipengine.runtime.laguna_decode_graph as module

    calls: list[tuple[str, int] | tuple[str]] = []

    class Runtime:
        def stream_create(self) -> int:
            calls.append(("stream_create",))
            return 41

        def stream_begin_capture(self, stream: int) -> None:
            calls.append(("capture_begin", int(stream)))

        def stream_end_capture(self, stream: int) -> int:
            calls.append(("capture_end", int(stream)))
            return 42

        def graph_instantiate(self, graph: int) -> int:
            calls.append(("instantiate", int(graph)))
            raise RuntimeError("synthetic instantiate failure")

        def graph_destroy(self, graph: int) -> None:
            calls.append(("graph_destroy", int(graph)))

        def stream_destroy(self, stream: int) -> None:
            calls.append(("stream_destroy", int(stream)))

    session = SimpleNamespace(
        runtime=Runtime(),
        _decode_graph=None,
        _enqueue_decode_graph_step=lambda **kwargs: calls.append(
            ("enqueue", int(kwargs["stream"]))
        ),
    )
    monkeypatch.setattr(module, "laguna_decode_graph_ineligibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "build_laguna_decode_graph_key", lambda *args, **kwargs: object())

    try:
        capture_laguna_decode_graph(session)
    except RuntimeError as exc:
        assert "synthetic instantiate failure" in str(exc)
    else:
        raise AssertionError("expected synthetic instantiate failure")

    assert session._decode_graph is None
    assert calls == [
        ("stream_create",),
        ("capture_begin", 41),
        ("enqueue", 41),
        ("capture_end", 41),
        ("instantiate", 42),
        ("graph_destroy", 42),
        ("stream_destroy", 41),
    ]


def test_laguna_decode_graph_close_destroys_handles_before_session_buffers() -> None:
    calls: list[tuple[str, int]] = []
    runtime = SimpleNamespace(
        graph_exec_destroy=lambda handle: calls.append(("exec", int(handle))),
        graph_destroy=lambda handle: calls.append(("graph", int(handle))),
        stream_destroy=lambda handle: calls.append(("stream", int(handle))),
    )
    session = SimpleNamespace(runtime=runtime, _decode_graph=None)
    graph = LagunaDecodeGraph(
        session=session,
        graph=11,
        graph_exec=12,
        stream=13,
        key=SimpleNamespace(),
        capture_seconds=0.25,
    )
    session._decode_graph = graph

    graph.close()
    graph.close()

    assert graph.closed is True
    assert graph.graph_exec == graph.graph == graph.stream == 0
    assert session._decode_graph is None
    assert calls == [("exec", 12), ("graph", 11), ("stream", 13)]


def test_laguna_decode_graph_close_attempts_every_handle_after_destroy_failure() -> None:
    calls: list[tuple[str, int]] = []

    def fail_exec(handle: int) -> None:
        calls.append(("exec", int(handle)))
        raise RuntimeError("synthetic exec destroy failure")

    runtime = SimpleNamespace(
        graph_exec_destroy=fail_exec,
        graph_destroy=lambda handle: calls.append(("graph", int(handle))),
        stream_destroy=lambda handle: calls.append(("stream", int(handle))),
    )
    session = SimpleNamespace(runtime=runtime, _decode_graph=None)
    graph = LagunaDecodeGraph(
        session=session,
        graph=21,
        graph_exec=22,
        stream=23,
        key=SimpleNamespace(),
        capture_seconds=0.25,
    )
    session._decode_graph = graph

    try:
        graph.close()
    except RuntimeError as exc:
        assert "graph resources failed to free" in str(exc)
    else:
        raise AssertionError("expected synthetic graph teardown failure")

    assert graph.closed is True
    assert graph.graph_exec == graph.graph == graph.stream == 0
    assert session._decode_graph is None
    assert calls == [("exec", 22), ("graph", 21), ("stream", 23)]
