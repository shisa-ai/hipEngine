from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.gguf_packed_decode_graph as packed_graph
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime.gguf_packed_decode_graph import (
    Qwen35GGUFPackedDecodeGraph,
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
    physical_sessions = tuple(
        session if active else None
        for session, active in zip(sessions, active_mask, strict=True)
    )
    return build_qwen35_gguf_packed_decode_graph_key(
        owner,
        sessions=physical_sessions,
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


def test_packed_decode_graph_key_supports_native_c8_physical_width() -> None:
    positions = (512, 513, 514, 515, 516, 517, 518, 519)
    key = _key(positions=positions, active_mask=(True,) * 8)

    assert key.physical_rows == 8
    assert key.active_rows == 8
    assert key.active_mask == (True,) * 8
    assert key.state_generations == positions


def test_packed_decode_graph_key_preserves_sparse_c8_physical_lanes() -> None:
    mask = (True, False, True, False, False, True, False, True)
    key = _key(
        positions=(512, 0, 520, 0, 0, 528, 0, 536),
        active_mask=mask,
    )

    assert key.physical_rows == 8
    assert key.active_rows == 4
    assert key.active_mask == mask
    assert key.state_generations == (512, -1, 520, -1, -1, 528, -1, 536)
    assert key.replay_context_limit == 664
    assert key.context_bucket == 768


def test_packed_decode_graph_final_layout_keeps_inactive_lanes_inert() -> None:
    graph = SimpleNamespace(
        replayed_steps=2,
        slot_capacity=1024,
        bucket_key=SimpleNamespace(
            state_generations=(512, -1, 520, -1, -1, 528, -1, 536),
            active_mask=(True, False, True, False, False, True, False, True),
        ),
    )

    layout = packed_graph._final_transition_layout(graph)

    assert layout.row_positions.tolist() == [513, -1, 521, -1, -1, 529, -1, 537]
    assert layout.live_counts.tolist() == [514, 0, 522, 0, 0, 530, 0, 538]
    assert layout.active_mask.tolist() == [True, False, True, False, False, True, False, True]
    assert layout.block_table[1].tolist() == [-1, -1, -1, -1]
    assert layout.block_table[6].tolist() == [-1, -1, -1, -1]


def test_packed_graph_kernel_resolution_tracks_native_width(monkeypatch) -> None:
    resolved: list[tuple[str, str]] = []

    monkeypatch.setattr(packed_graph, "load_backend_kernel_package", lambda _backend: None)

    def fake_resolve(**kwargs):
        route = (str(kwargs["layer"]), str(kwargs["variant"]))
        resolved.append(route)
        return route

    monkeypatch.setattr(packed_graph, "resolve", fake_resolve)
    owner = SimpleNamespace(runner=SimpleNamespace(backend="hip_gfx1100"))

    c4 = packed_graph._resolve_packed_graph_kernels(owner, rows=4)
    c8 = packed_graph._resolve_packed_graph_kernels(owner, rows=8)

    assert c4[:2] == (
        ("decode_metadata", "packed_c4_device_positions_i64"),
        ("decode_graph_commit", "packed_c4_i32_i64"),
    )
    assert c8[:2] == (
        ("decode_metadata", "packed_c8_device_positions_i64"),
        ("decode_graph_commit", "packed_c8_i32_i64"),
    )
    assert resolved == [*c4, *c8]


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
    import hipengine.core.pm4.transport as transport_module

    observed: dict[str, object] = {}
    context = object()

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
    monkeypatch.setattr(transport_module, "select_submission_transport", lambda value=None: "pm4")
    monkeypatch.setattr(
        transport_module,
        "create_graph_submission_context",
        lambda **_kwargs: context,
    )
    owner = object.__new__(Qwen35GGUFResidentSession)
    peer = object.__new__(Qwen35GGUFResidentSession)
    owner.runner = SimpleNamespace(backend="hip_gfx1100", target_arch="gfx1100")
    owner.runtime = object()
    owner._decode_graph_submission_contexts = {}
    owner._decode_graph_default_submission_transport = None
    pins: list[tuple[str, object]] = []
    owner._pin_device_kv_graph = lambda graph: pins.append(("owner", graph))
    peer._pin_device_kv_graph = lambda graph: pins.append(("peer", graph))

    result = owner.capture_packed_decode_graph(
        (11, 22),
        sessions=(owner, peer),
        physical_rows=8,
        active_slot_indices=(1, 6),
        steps_per_replay=2,
        max_replay_steps=8,
        record_steps=8,
        record_layer_output_hidden=(0, 3),
        submission_transport="pm4",
        submission_timeout_seconds=7.5,
    )

    assert result == "graph"
    assert pins == [("owner", "graph"), ("peer", "graph")]
    assert observed == {
        "owner": owner,
        "token_ids": (11, 22),
        "sessions": (owner, peer),
        "physical_rows": 8,
        "active_slot_indices": (1, 6),
        "steps_per_replay": 2,
        "max_replay_steps": 8,
        "record_steps": 8,
        "record_layer_output_hidden": (0, 3),
        "submission_transport": "pm4",
        "submission_timeout_seconds": 7.5,
        "submission_context": context,
    }


def test_packed_decode_graph_replays_and_closes_registered_submission(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeSubmission:
        name = "pm4"
        graph_exec = 0

        def launch(self, stream: int) -> None:
            calls.append(("submission_launch", int(stream)))

        def provenance(self) -> dict[str, object]:
            calls.append(("submission_provenance",))
            return {"transport": self.name, "launches": 1, "native_fallbacks": 0}

        def close(self) -> None:
            calls.append(("submission_close",))

    runtime = SimpleNamespace(
        graph_launch=lambda *_args: calls.append(("unexpected_graph_launch",)),
        stream_synchronize=lambda stream: calls.append(("stream_synchronize", int(stream))),
        graph_exec_destroy=lambda *_args: calls.append(("unexpected_graph_exec_destroy",)),
        graph_destroy=lambda graph: calls.append(("graph_destroy", int(graph))),
        stream_destroy=lambda stream: calls.append(("stream_destroy", int(stream))),
    )
    sessions = tuple(
        SimpleNamespace(
            position=5,
            scratch=SimpleNamespace(
                position_host=np.asarray([5], dtype=np.int64),
                context_host=np.asarray([6], dtype=np.int64),
            ),
            _unpin_device_kv_graph=lambda graph: calls.append(("unpin", id(graph))),
        )
        for _ in range(2)
    )
    owner = SimpleNamespace(
        runtime=runtime,
        _decode_graphs=[],
        _packed_decode_session_ids=(),
        _packed_decode_positions=(),
        _packed_decode_last_layout=None,
        _packed_decode_state_dirty=False,
        last_packed_execution_manifest=None,
    )
    graph = Qwen35GGUFPackedDecodeGraph(
        owner=owner,
        sessions=sessions,
        graph=0xA0,
        graph_exec=0,
        stream=0xB0,
        position_tuple=(5, 5),
        steps_per_replay=1,
        max_replay_steps=2,
        slot_capacity=1024,
        rows=2,
        generated_tokens=None,
        generated_hidden=None,
        record_index=None,
        active_mask_device=None,
        record_steps=0,
        record_layer_ids=(),
        bucket_key=SimpleNamespace(
            active_mask=(True, True),
            state_generations=(5, 5),
            key_sha256="packed-key",
        ),
        execution_manifest={
            "graph": {
                "replay_count": 0,
                "replayed_steps": 0,
                "replay_call_synchronizations": 0,
                "transport": {
                    "transport": "pm4",
                    "launches": 0,
                    "native_fallbacks": 0,
                },
            }
        },
        submission=FakeSubmission(),
    )
    owner._decode_graphs.append(graph)
    monkeypatch.setattr(packed_graph, "_final_transition_layout", lambda _graph: "layout")

    graph.replay(1)

    assert calls[:2] == [("submission_launch", 0xB0), ("stream_synchronize", 0xB0)]
    assert ("submission_provenance",) not in calls
    assert not any(call[0].startswith("unexpected") for call in calls)
    assert graph.transport_provenance() == {
        "transport": "pm4",
        "launches": 1,
        "native_fallbacks": 0,
        "packed_decode_graph_key_sha256": "packed-key",
        "physical_rows": 2,
        "replayed_steps": 1,
    }
    assert calls.count(("submission_provenance",)) == 1
    assert graph.execution_manifest["graph"]["transport"]["transport"] == "pm4"

    graph.close()

    assert ("submission_close",) in calls
    assert ("graph_destroy", 0xA0) in calls
    assert ("stream_destroy", 0xB0) in calls
    assert graph not in owner._decode_graphs
    assert not any(call[0].startswith("unexpected") for call in calls)


def test_packed_decode_graph_reads_only_the_latest_token_row(monkeypatch) -> None:
    copied: list[tuple[int, int]] = []

    def fake_copy(host_ptr, source, *, runtime):
        del runtime
        copied.append((int(source.ptr), int(source.nbytes)))
        values = np.ascontiguousarray([31, 32], dtype=np.int32)
        ctypes.memmove(int(host_ptr), values.ctypes.data, int(source.nbytes))

    monkeypatch.setattr(packed_graph, "copy_device_to_host", fake_copy)
    graph = object.__new__(Qwen35GGUFPackedDecodeGraph)
    graph.closed = False
    graph.generated_tokens = DeviceBuffer(ptr=0x1000, nbytes=4 * 2 * 4)
    graph.replayed_steps = 3
    graph.rows = 2
    graph.record_steps = 4
    graph.owner = SimpleNamespace(runtime=object())

    assert graph.read_latest_generated_token_ids() == [31, 32]
    assert copied == [(0x1000 + 2 * 2 * 4, 2 * 4)]


def test_packed_decode_graph_key_rejects_invalid_shape_contract() -> None:
    owner, sessions, pointers = _owner()
    oversized_owner, oversized_sessions, oversized_pointers = _owner(
        positions=tuple(range(9))
    )
    with pytest.raises(ValueError, match="between one and eight"):
        build_qwen35_gguf_packed_decode_graph_key(
            oversized_owner,
            sessions=oversized_sessions,
            active_mask=(True,) * 9,
            block_size=256,
            max_positions=1024,
            steps_per_replay=1,
            max_replay_steps=1,
            record_steps=0,
            record_layer_ids=(),
            packed_buffer_ptrs=oversized_pointers,
        )
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
