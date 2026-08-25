from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
)
from hipengine.core.tensor import Tensor

from hipengine.runtime import gguf_native_spec_cycle as native_cycle_mod
from hipengine.runtime.gguf_native_spec_cycle import (
    NativeSpecTargetGraphUnsupportedError,
    Qwen35GGUFNativeAcceptCommitResult,
    build_native_b2_target_batch,
    run_qwen35_gguf_native_mtp_cycle,
    verify_qwen35_gguf_native_b2_target,
    verify_qwen35_gguf_native_target_from_device_proposal,
)
from hipengine.speculative.mtp_resident_draft import (
    NativeSpecProposalGraphUnsupportedError,
)


def test_native_target_graph_snapshots_mutable_linear_commit_tables(
    monkeypatch,
) -> None:
    device = Device("hip", 0)
    next_ptr = iter((0x1000, 0x2000, 0x3000, 0x4000))
    reservations: list[tuple[str, tuple[int, ...], DType]] = []
    staged: list[tuple[int, np.ndarray]] = []

    class Workspace:
        def reserve_tensor(self, name, shape, dtype):
            parsed = DType.parse(dtype)
            reservations.append((str(name), tuple(shape), parsed))
            return Tensor.from_handle(next(next_ptr), shape, parsed, device)

    monkeypatch.setattr(
        native_cycle_mod,
        "_copy_array_to_tensor",
        lambda tensor, values, *, runtime: staged.append(
            (int(tensor.ptr), np.asarray(values).copy())
        ),
    )
    session = SimpleNamespace(
        _verify_linear_state_src_conv_host=np.asarray([11, 12], dtype=np.uint64),
        _verify_linear_state_src_recurrent_host=np.asarray([21, 22], dtype=np.uint64),
        _verify_linear_state_dst_conv_host=np.asarray([31, 32], dtype=np.uint64),
        _verify_linear_state_dst_recurrent_host=np.asarray([41, 42], dtype=np.uint64),
    )

    tables = native_cycle_mod._allocate_native_linear_state_tables(
        Workspace(),
        session,
        runtime=object(),
    )
    session._verify_linear_state_src_conv_host[:] = 99

    assert tuple(table.ptr for table in tables) == (0x1000, 0x2000, 0x3000, 0x4000)
    assert all(shape == (2,) and dtype is DType.INT64 for _name, shape, dtype in reservations)
    np.testing.assert_array_equal(staged[0][1], np.asarray([11, 12], dtype=np.uint64))


def test_compact_n2_result_validates_commit_from_visible_tokens_without_row_ids() -> None:
    kwargs = {
        "input_token_ids": [100, 0, 0],
        "token_ids": [101, 102, 103],
        "accepted_draft_tokens": 2,
        "commit_row": 2,
        "commit_token": 102,
        "commit_position": 7,
        "next_token": 103,
        "full_accept": True,
        "start_position": 5,
        "end_position": 8,
        "hidden_seed_rows_ptr": 0x1000,
        "hidden_seed_row_count": 3,
        "hidden_size": 8,
        "proposal_device_handoff": True,
    }

    with pytest.raises(ValueError, match="commit_token"):
        Qwen35GGUFNativeAcceptCommitResult(**kwargs)

    result = Qwen35GGUFNativeAcceptCommitResult(
        **kwargs,
        compact_result=True,
    )

    assert result.target_top1 == []
    assert result.proposal_top1_values == ()
    assert result.compact_result


def test_build_native_b2_target_batch_uses_root_prefixed_chain_layout() -> None:
    batch = build_native_b2_target_batch(
        [101, 202, 303],
        start_position=17,
        request_id=9,
    )

    assert batch.request_ids == (9,)
    assert batch.tokens == (101, 202, 303)
    assert batch.positions == (17, 18, 19)
    assert batch.root_rows == (0,)
    assert batch.candidate_rows == (1, 2)
    assert batch.parent_rows == (-1, 0, 1)
    assert batch.draft_depths == (0, 1, 2)
    assert batch.row_to_request == (9, 9, 9)
    assert batch.active_mask == (True, True, True)
    assert batch.mode == "verify_chain"

    b1 = build_native_b2_target_batch([101, 202], start_position=17, request_id=9)
    assert b1.tokens == (101, 202)
    assert b1.positions == (17, 18)
    assert b1.parent_rows == (-1, 0)
    assert b1.draft_depths == (0, 1)
    assert b1.row_to_request == (9, 9)

    b3 = build_native_b2_target_batch([101, 202, 303, 404], start_position=17, request_id=9)
    assert b3.tokens == (101, 202, 303, 404)
    assert b3.positions == (17, 18, 19, 20)
    assert b3.parent_rows == (-1, 0, 1, 2)
    assert b3.draft_depths == (0, 1, 2, 3)
    assert b3.row_to_request == (9, 9, 9, 9)


class _FallbackSession:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], dict[str, object]]] = []

    def verify_target_block(self, input_token_ids, **kwargs):
        self.calls.append((tuple(int(token) for token in input_token_ids), dict(kwargs)))
        return SimpleNamespace(token_ids=[7, 8])


def test_native_b2_target_uses_exact_python_fallback_for_unsupported_shape() -> None:
    session = _FallbackSession()

    result = verify_qwen35_gguf_native_b2_target(
        session,
        [1, 2],
        fallback=True,
        bulk_attention_mode="native",
        capture_linear_state_rows=True,
        defer_linear_state_commit=True,
    )

    assert result.token_ids == [7, 8]
    assert session.calls == [
        (
            (1, 2),
            {
                "bulk_attention_mode": "native",
                "use_wmma_prefill": False,
                "capture_linear_state_rows": True,
                "defer_linear_state_commit": True,
            },
        )
    ]


def test_native_b2_target_can_make_unsupported_shape_a_hard_error() -> None:
    with pytest.raises(NativeSpecTargetGraphUnsupportedError, match="two to eight rows"):
        verify_qwen35_gguf_native_b2_target(
            _FallbackSession(),
            [1],
            fallback=False,
        )


@pytest.mark.parametrize(
    ("native_proposal_graph", "graph_unsupported", "proposal_call"),
    [
        (False, False, "propose"),
        (True, False, "propose_graph"),
        (True, True, "propose"),
    ],
)
def test_native_complete_cycle_owns_propose_accept_mtp_kv_and_reseed(
    native_proposal_graph: bool,
    graph_unsupported: bool,
    proposal_call: str,
) -> None:
    calls: list[tuple[object, ...]] = []

    class Draft:
        hidden_size = 4
        _device_chain_enabled = True

        def propose_chain_from_device_seed(self, hidden_seed_ptr, **kwargs):
            calls.append(("propose", int(hidden_seed_ptr), dict(kwargs)))
            return [201, 202], [[201], [202]], int(kwargs["dense_cache_len"]) + 2

        def propose_chain_from_device_seed_graph(self, hidden_seed_ptr, **kwargs):
            if graph_unsupported:
                raise NativeSpecProposalGraphUnsupportedError("unsupported test graph")
            calls.append(("propose_graph", int(hidden_seed_ptr), dict(kwargs)))
            return [201, 202], [[201], [202]], int(kwargs["dense_cache_len"]) + 2

        def write_kv_rows_from_device_seed_base(self, hidden_seed_ptr, token_ids, **kwargs):
            calls.append(
                (
                    "commit_mtp_kv",
                    int(hidden_seed_ptr),
                    tuple(int(token) for token in token_ids.tolist()),
                    tuple(int(position) for position in kwargs["positions"].tolist()),
                    int(kwargs["dense_cache_len"]),
                )
            )
            return int(kwargs["dense_cache_len"]) + len(token_ids)

    class Context:
        pending_seed = SimpleNamespace(hidden_ptr=0xA000)

        def record_verify_seeds(self, seeds):
            rows = tuple(seeds)
            calls.append(("record_verify_seeds", tuple(int(seed.hidden_ptr) for seed in rows)))
            self.verify_seeds = rows

        def accept(self, accepted):
            calls.append(("accept_seed", int(accepted)))
            self.pending_seed = self.verify_seeds[int(accepted)]
            return self.pending_seed

    class Session:
        position = 17
        backend = "hip_gfx1100"

        def verify_target_block_native_cycle(self, input_token_ids, **kwargs):
            calls.append(("verify", tuple(int(token) for token in input_token_ids), dict(kwargs)))
            self.position = 19
            return SimpleNamespace(
                input_token_ids=[101, 201, 202],
                token_ids=[201, 909],
                accepted_draft_tokens=1,
                commit_row=1,
                commit_token=201,
                commit_position=18,
                next_token=909,
                full_accept=False,
                start_position=17,
                end_position=19,
                hidden_seed_rows_ptr=0xB000,
                hidden_seed_row_count=3,
                hidden_size=4,
                device_accept_commit=True,
            )

        def mtp_verify_seed(self, row, *, token_id, position, **kwargs):
            return SimpleNamespace(
                token_id=int(token_id),
                position=int(position),
                hidden_ptr=int(kwargs["hidden_seed_base_ptr"]) + int(row) * 16,
                hidden_contract=SimpleNamespace(ready_for_mtp=True, rows=1, hidden_size=4),
            )

    rope_cos = np.ones((64, 4), dtype=np.float32)
    rope_sin = np.zeros((64, 4), dtype=np.float32)
    result = run_qwen35_gguf_native_mtp_cycle(
        Session(),
        Draft(),
        Context(),
        root_token=101,
        root_position=17,
        candidate_budget=2,
        remaining_decode=7,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        draft_key_cache=DeviceBuffer(0xC000, 4096),
        draft_value_cache=DeviceBuffer(0xD000, 4096),
        draft_cache_len=7,
        cycle_id=5,
        transaction_id=6,
        native_proposal_graph=native_proposal_graph,
    )

    assert result.draft_token_ids == (201, 202)
    assert result.output_token_ids == (201, 909)
    assert result.accepted_draft_tokens == 1
    assert result.start_position == 17
    assert result.end_position == 19
    assert result.draft_cache_len_before == 7
    assert result.draft_cache_len_after == 9
    assert result.target_result.device_accept_commit is True
    assert result.proposal_native_graph is (native_proposal_graph and not graph_unsupported)
    assert calls[0][0:2] == (proposal_call, 0xA000)
    assert calls[1] == (
        "verify",
        (101, 201, 202),
        {
            "cycle_id": 5,
            "transaction_id": 6,
            "request_id": 0,
            "bulk_attention_mode": "bulk",
            "use_wmma_prefill": False,
            "capture_linear_state_rows": True,
            "defer_linear_state_commit": True,
            "device_accept_commit": True,
            "remaining_decode": 7,
            "fallback": False,
        },
    )
    assert calls[2] == ("record_verify_seeds", (0xB000, 0xB010))
    assert calls[3] == ("accept_seed", 1)
    assert calls[4] == ("commit_mtp_kv", 0xB000, (201,), (18,), 8)


def test_device_proposal_handoff_stages_both_token_metadata_columns() -> None:
    calls: list[tuple[object, ...]] = []

    class Runtime:
        def stream_wait_event(self, stream, event):
            calls.append(("wait", int(stream), int(event)))

        def memcpy_async(self, dst, src, nbytes, kind, stream):
            calls.append(
                (
                    "copy",
                    int(dst),
                    int(src),
                    int(nbytes),
                    int(kind),
                    int(stream),
                )
            )

    native_cycle_mod._enqueue_device_proposal_handoff(
        runtime=Runtime(),
        stream=0x1000,
        dynamic_metadata_ptr=0x2000,
        result_payload_ptr=0x3000,
        proposal_result_ptr=0x4000,
        proposal_result_nbytes=24,
        proposal_event=0x5000,
        proposal_budget=3,
        target_rows=4,
    )

    assert calls[0] == ("wait", 0x1000, 0x5000)
    assert calls[1:7] == [
        ("copy", 0x2000 + 40, 0x4000, 4, int(HipMemcpyKind.DEVICE_TO_DEVICE), 0x1000),
        ("copy", 0x2000 + 48, 0x4000, 4, int(HipMemcpyKind.DEVICE_TO_DEVICE), 0x1000),
        ("copy", 0x2000 + 80, 0x4008, 4, int(HipMemcpyKind.DEVICE_TO_DEVICE), 0x1000),
        ("copy", 0x2000 + 88, 0x4008, 4, int(HipMemcpyKind.DEVICE_TO_DEVICE), 0x1000),
        ("copy", 0x2000 + 120, 0x4010, 4, int(HipMemcpyKind.DEVICE_TO_DEVICE), 0x1000),
        ("copy", 0x2000 + 128, 0x4010, 4, int(HipMemcpyKind.DEVICE_TO_DEVICE), 0x1000),
    ]
    payload_start = native_cycle_mod.ACCEPT_PACKED_PAYLOAD_FIELDS + 1 + 8
    assert calls[7] == (
        "copy",
        0x3000 + payload_start * 4,
        0x4000,
        24,
        int(HipMemcpyKind.DEVICE_TO_DEVICE),
        0x1000,
    )


@pytest.mark.parametrize(
    ("rows", "position", "remaining_decode", "expected_reason"),
    [
        (2, 1021, 2, None),
        (2, 1022, 2, "target_graph_context_bucket_miss"),
        (3, 1020, 3, None),
        (3, 1021, 3, "target_graph_context_bucket_miss"),
        (4, 1019, 4, None),
        (4, 1020, 4, "target_graph_context_bucket_miss"),
        (4, 1019, 3, "target_graph_output_room_miss"),
    ],
)
def test_native_target_graph_launch_eligibility_covers_live_context_and_output_room(
    monkeypatch,
    rows: int,
    position: int,
    remaining_decode: int,
    expected_reason: str | None,
) -> None:
    session = SimpleNamespace(position=position)
    monkeypatch.setattr(
        native_cycle_mod,
        "_native_target_binding_signature",
        lambda _session: (0xCAFE,),
    )
    graph = native_cycle_mod.Qwen35GGUFNativeB2TargetGraph.__new__(
        native_cycle_mod.Qwen35GGUFNativeB2TargetGraph
    )
    graph.closed = False
    graph.session = session
    graph.rows = rows
    graph.context_limit = 1023
    graph.device_accept_commit = True
    graph.configuration_key = native_cycle_mod._native_target_configuration_key(
        bulk_attention_mode="native",
        use_wmma_prefill=False,
        capture_linear_state_rows=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
        device_accept_commit=True,
    )
    graph.binding_signature = (0xCAFE,)

    reason = graph.launch_ineligibility_reason(
        session,
        position=position,
        rows=rows,
        remaining_decode=remaining_decode,
        bulk_attention_mode="native",
        use_wmma_prefill=False,
        capture_linear_state_rows=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
        device_accept_commit=True,
    )

    assert reason == expected_reason
    assert graph.can_launch(
        session,
        position=position,
        rows=rows,
        remaining_decode=remaining_decode,
        bulk_attention_mode="native",
        use_wmma_prefill=False,
        capture_linear_state_rows=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
        device_accept_commit=True,
    ) is (expected_reason is None)


def test_native_target_graph_rejects_proposal_cursor_drift_before_submit(
    monkeypatch,
) -> None:
    session = SimpleNamespace(position=18)
    monkeypatch.setattr(
        native_cycle_mod,
        "_native_target_binding_signature",
        lambda _session: (0xCAFE,),
    )

    class Launcher:
        def launch(self, *_args, **_kwargs):
            raise AssertionError("cursor drift must fail before target graph submission")

    graph = native_cycle_mod.Qwen35GGUFNativeB2TargetGraph.__new__(
        native_cycle_mod.Qwen35GGUFNativeB2TargetGraph
    )
    graph.closed = False
    graph.session = session
    graph.binding_signature = (0xCAFE,)
    graph.device_accept_commit = True
    graph.rows = 4
    graph.context_limit = 1023
    graph.launcher = Launcher()
    proposal = SimpleNamespace(
        budget=3,
        request_id=7,
        root_token=101,
        root_position=17,
        result_ptr=0x1000,
        result_nbytes=24,
        completion_event=0x2000,
    )

    with pytest.raises(ValueError, match="root position drifted"):
        graph.launch(
            cycle_id=5,
            transaction_id=6,
            request_id=7,
            remaining_decode=4,
            device_proposal=proposal,
        )


def test_native_target_context_miss_falls_back_eager_with_stable_reason() -> None:
    session = _FallbackSession()
    session.position = 1020

    class Graph:
        closed = False

        def compatible_with(self, _session, **_kwargs) -> bool:
            return True

        def launch(self, *_args, **_kwargs):
            raise NativeSpecTargetGraphUnsupportedError(
                "target_graph_context_bucket_miss"
            )

        def close(self) -> None:
            raise AssertionError("a clean context miss must keep the short graph cached")

    session._native_spec_b3_target_graph_n2 = Graph()

    result = verify_qwen35_gguf_native_b2_target(
        session,
        [1, 2, 3, 4],
        fallback=True,
        bulk_attention_mode="native",
        capture_linear_state_rows=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
        device_accept_commit=True,
        remaining_decode=4,
    )

    assert result.token_ids == [7, 8]
    assert (
        session.last_native_spec_target_fallback_reason
        == "target_graph_context_bucket_miss"
    )
    assert session._native_spec_b3_target_graph_n2.closed is False
    assert session.calls == [
        (
            (1, 2, 3, 4),
            {
                "bulk_attention_mode": "native",
                "use_wmma_prefill": False,
                "capture_linear_state_rows": True,
                "defer_linear_state_commit": True,
                "capture_pre_output_norm_hidden": True,
            },
        )
    ]


def test_device_proposal_handoff_rechecks_context_before_target_launch() -> None:
    proposal = SimpleNamespace(budget=3, request_id=7, root_position=1020)
    launches: list[object] = []

    class Graph:
        closed = False

        def compatible_with(self, _session, **_kwargs) -> bool:
            return True

        def launch_ineligibility_reason(self, _session, **_kwargs) -> str:
            return "target_graph_context_bucket_miss"

        def launch(self, **kwargs):
            launches.append(kwargs)
            raise AssertionError("context-ineligible target graph must not launch")

        def close(self) -> None:
            raise AssertionError("a clean context miss must not destroy the cached short graph")

    session = SimpleNamespace(
        position=1020,
        _native_spec_b3_target_graph_n2=Graph(),
        last_native_spec_target_submitted=True,
        last_native_spec_target_fallback_reason=None,
        last_native_spec_target_capture_ms=1.0,
        last_native_spec_target_submit_ms=2.0,
        last_native_spec_target_readback_ms=3.0,
    )

    with pytest.raises(
        NativeSpecTargetGraphUnsupportedError,
        match="target_graph_context_bucket_miss",
    ):
        verify_qwen35_gguf_native_target_from_device_proposal(
            session,
            proposal,
            cycle_id=5,
            transaction_id=6,
            request_id=7,
            remaining_decode=4,
        )

    assert launches == []
    assert session.last_native_spec_target_submitted is False
    assert (
        session.last_native_spec_target_fallback_reason
        == "target_graph_context_bucket_miss"
    )


def test_device_proposal_handoff_requires_and_launches_only_a_cached_n2_graph() -> None:
    proposal = SimpleNamespace(budget=3, request_id=7, root_position=17)
    launches: list[dict[str, object]] = []

    class Graph:
        closed = False

        def compatible_with(self, _session, **kwargs) -> bool:
            assert kwargs["device_accept_commit"] is True
            assert kwargs["bulk_attention_mode"] == "native"
            return True

        def launch_ineligibility_reason(self, _session, **_kwargs):
            return None

        def launch(self, input_token_ids=None, **kwargs):
            assert input_token_ids is None
            launches.append(dict(kwargs))
            return "retired"

    session = SimpleNamespace(
        position=17,
        _native_spec_b3_target_graph_n2=Graph(),
        last_native_spec_target_submitted=True,
        last_native_spec_target_fallback_reason="stale",
        last_native_spec_target_capture_ms=1.0,
        last_native_spec_target_submit_ms=2.0,
        last_native_spec_target_readback_ms=3.0,
    )

    result = verify_qwen35_gguf_native_target_from_device_proposal(
        session,
        proposal,
        cycle_id=5,
        transaction_id=6,
        request_id=7,
        remaining_decode=9,
    )

    assert result == "retired"
    assert launches == [
        {
            "cycle_id": 5,
            "transaction_id": 6,
            "request_id": 7,
            "remaining_decode": 9,
            "device_proposal": proposal,
        }
    ]
    assert session.last_native_spec_target_capture_ms == 0.0
    missing = SimpleNamespace(
        last_native_spec_target_submitted=False,
        last_native_spec_target_fallback_reason=None,
        last_native_spec_target_capture_ms=0.0,
        last_native_spec_target_submit_ms=0.0,
        last_native_spec_target_readback_ms=0.0,
    )
    with pytest.raises(NativeSpecTargetGraphUnsupportedError, match="cached N2"):
        verify_qwen35_gguf_native_target_from_device_proposal(
            missing,
            proposal,
            request_id=7,
            remaining_decode=9,
        )


@pytest.mark.parametrize("failure", ["graph_launch", "graph_readback"])
def test_rf3_target_graph_failure_closes_owner_without_eager_replay(failure: str) -> None:
    session = _FallbackSession()
    session.position = 17

    class Graph:
        closed = False

        def compatible_with(self, _session, **_kwargs) -> bool:
            return True

        def launch(self, *_args, **_kwargs):
            raise RuntimeError(failure)

        def close(self) -> None:
            self.closed = True

    graph = Graph()
    session._native_spec_b3_target_graph = graph

    with pytest.raises(RuntimeError, match=failure):
        verify_qwen35_gguf_native_b2_target(session, [1, 2, 3, 4])

    assert graph.closed is True
    assert session.calls == []


def test_rf3_target_graph_capture_allocation_failure_has_no_eager_replay(monkeypatch) -> None:
    session = _FallbackSession()
    session.position = 17
    monkeypatch.setattr(
        native_cycle_mod,
        "capture_qwen35_gguf_native_b2_target_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError("injected allocation")),
    )

    with pytest.raises(MemoryError, match="injected allocation"):
        verify_qwen35_gguf_native_b2_target(session, [1, 2, 3, 4])

    assert session.calls == []
    assert session._native_spec_target_graphs == {}


def test_native_b2_target_reuses_one_dynamic_graph_across_cycles(monkeypatch) -> None:
    session = _FallbackSession()
    launches: list[tuple[tuple[int, ...], int, int, int]] = []
    captures: list[tuple[int, ...]] = []

    class FakeReusableGraph:
        closed = False

        def compatible_with(self, _session, **_kwargs) -> bool:
            return True

        def launch(
            self,
            input_token_ids,
            *,
            cycle_id: int,
            transaction_id: int,
            request_id: int,
        ):
            launches.append(
                (
                    tuple(int(token) for token in input_token_ids),
                    int(cycle_id),
                    int(transaction_id),
                    int(request_id),
                )
            )
            return SimpleNamespace(token_ids=[7, 8, 9])

    graph = FakeReusableGraph()

    def fake_capture(_session, input_token_ids, **_kwargs):
        captures.append(tuple(int(token) for token in input_token_ids))
        return graph

    monkeypatch.setattr(
        "hipengine.runtime.gguf_native_spec_cycle.capture_qwen35_gguf_native_b2_target_graph",
        fake_capture,
    )

    first = verify_qwen35_gguf_native_b2_target(
        session,
        [1, 2, 3],
        cycle_id=4,
        transaction_id=5,
        request_id=6,
    )
    second = verify_qwen35_gguf_native_b2_target(
        session,
        [4, 5, 6],
        cycle_id=7,
        transaction_id=8,
        request_id=9,
    )
    short = verify_qwen35_gguf_native_b2_target(session, [7])

    assert first.token_ids == second.token_ids == [7, 8, 9]
    assert short.token_ids == [7, 8]
    assert captures == [(1, 2, 3)]
    assert launches == [
        ((1, 2, 3), 4, 5, 6),
        ((4, 5, 6), 7, 8, 9),
    ]
    assert session.calls == [
        (
            (7,),
            {
                "bulk_attention_mode": "bulk",
                "use_wmma_prefill": False,
                "capture_linear_state_rows": False,
                "defer_linear_state_commit": False,
            },
        )
    ]
    assert session._native_spec_b2_target_graph is graph
    assert graph.closed is False


def test_rf2_context_bucket_selection_is_power_of_two_and_capability_bounded(
    monkeypatch,
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "_gguf_prefill_device_metadata_enabled",
        lambda *, backend, prompt_tokens: backend == "hip_gfx1151"
        and int(prompt_tokens) <= 4096,
    )
    session = SimpleNamespace(
        position=0,
        backend="hip_gfx1151",
        scratch=SimpleNamespace(max_positions=65544),
    )

    assert native_cycle_mod._native_target_graph_context_limit(session, rows=1023) == 1023
    assert native_cycle_mod._native_target_graph_context_limit(session, rows=1024) is None
    session.position = 1020
    assert native_cycle_mod._native_target_graph_context_limit(session, rows=4) is None
    session.position = 1024
    assert native_cycle_mod._native_target_graph_context_limit(session, rows=4) == 1280
    session.position = 1278
    assert native_cycle_mod._native_target_graph_context_limit(session, rows=4) is None
    session.position = 2044
    assert native_cycle_mod._native_target_graph_context_limit(session, rows=4) == 2048
    session.position = 4092
    assert native_cycle_mod._native_target_graph_context_limit(session, rows=4) == 4096
    session.position = 4093
    assert native_cycle_mod._native_target_graph_context_limit(session, rows=4) is None


def test_rf2_context_bucket_respects_split_kernel_family_boundary(monkeypatch) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_module

    generic = object()
    grouped = object()
    monkeypatch.setattr(
        runner_module,
        "_gguf_prefill_device_metadata_enabled",
        lambda *, backend, prompt_tokens: int(prompt_tokens) <= 4096,
    )
    monkeypatch.setattr(
        runner_module,
        "_gguf_full_attention_split_gate_bf16_fn",
        lambda _config, *, active_context, **_kwargs: (
            generic if int(active_context) < 4096 else grouped
        ),
    )
    session = SimpleNamespace(
        position=4091,
        backend="hip_gfx1151",
        scratch=SimpleNamespace(max_positions=65544, block_size=256),
        runner=SimpleNamespace(weights=SimpleNamespace(config=object())),
    )

    assert native_cycle_mod._native_target_graph_context_limit(session, rows=4) == 4095
    session.position = 4092
    assert native_cycle_mod._native_target_graph_context_limit(session, rows=4) is None


def test_rf2_target_graph_cache_separates_short_and_split_k_context_buckets(
    monkeypatch,
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "_gguf_prefill_device_metadata_enabled",
        lambda *, backend, prompt_tokens: int(prompt_tokens) <= 4096,
    )
    session = _FallbackSession()
    session.position = 1019
    session.backend = "hip_gfx1151"
    session.scratch = SimpleNamespace(max_positions=4096)
    captures: list[int] = []
    launches: list[tuple[int, int]] = []

    class Graph:
        closed = False

        def __init__(self, context_limit: int) -> None:
            self.context_limit = int(context_limit)

        def compatible_with(self, _session, *, context_limit, **_kwargs) -> bool:
            return int(context_limit) == self.context_limit

        def launch(self, input_token_ids, **_kwargs):
            launches.append((self.context_limit, int(session.position)))
            session.position += len(tuple(input_token_ids))
            return SimpleNamespace(token_ids=[7, 8, 9, 10])

        def close(self) -> None:
            self.closed = True

    def capture(_session, _tokens, *, context_limit, **_kwargs):
        captures.append(int(context_limit))
        return Graph(int(context_limit))

    monkeypatch.setattr(native_cycle_mod, "capture_qwen35_gguf_native_b2_target_graph", capture)

    first = verify_qwen35_gguf_native_b2_target(
        session,
        [1, 2, 3, 4],
        bulk_attention_mode="native",
        capture_linear_state_rows=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
    )
    session.position = 1020
    second = verify_qwen35_gguf_native_b2_target(
        session,
        [5, 6, 7, 8],
        bulk_attention_mode="native",
        capture_linear_state_rows=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
    )
    session.position = 1024
    third = verify_qwen35_gguf_native_b2_target(
        session,
        [9, 10, 11, 12],
        bulk_attention_mode="native",
        capture_linear_state_rows=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
    )
    session.position = 1100
    fourth = verify_qwen35_gguf_native_b2_target(
        session,
        [13, 14, 15, 16],
        bulk_attention_mode="native",
        capture_linear_state_rows=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
    )

    assert first.token_ids == third.token_ids == fourth.token_ids == [7, 8, 9, 10]
    assert second.token_ids == [7, 8]
    assert captures == [1023, 1280]
    assert launches == [(1023, 1019), (1280, 1024), (1280, 1100)]
    assert sorted(session._native_spec_target_graphs) == [
        (3, False, 1023),
        (3, False, 1280),
    ]
    assert session.last_native_spec_target_fallback_reason is None
    assert session.calls == [
        (
            (5, 6, 7, 8),
            {
                "bulk_attention_mode": "native",
                "use_wmma_prefill": False,
                "capture_linear_state_rows": True,
                "defer_linear_state_commit": True,
                "capture_pre_output_norm_hidden": True,
            },
        )
    ]


def test_rf2_target_graph_cache_evicts_and_closes_oldest_owner() -> None:
    closed: list[int] = []

    class Graph:
        def __init__(self, index: int) -> None:
            self.index = index

        def close(self) -> None:
            closed.append(self.index)

    session = SimpleNamespace(_native_spec_target_graphs={})
    for index in range(native_cycle_mod._NATIVE_TARGET_GRAPH_CACHE_MAX_ENTRIES + 1):
        native_cycle_mod._cache_native_target_graph(
            session,
            (3, True, 1023 + index),
            Graph(index),
        )

    assert closed == [0]
    assert len(session._native_spec_target_graphs) == native_cycle_mod._NATIVE_TARGET_GRAPH_CACHE_MAX_ENTRIES
    assert (3, True, 1023) not in session._native_spec_target_graphs


def test_native_b3_target_reuses_one_dynamic_native_graph_across_positions(monkeypatch) -> None:
    session = _FallbackSession()
    session.position = 17
    launches: list[tuple[tuple[int, ...], int]] = []
    captures: list[tuple[int, ...]] = []

    class FakeReusableGraph:
        closed = False

        def compatible_with(self, _session, **kwargs) -> bool:
            return (
                kwargs["bulk_attention_mode"] == "native"
                and kwargs["capture_pre_output_norm_hidden"] is True
            )

        def launch(self, input_token_ids, **_kwargs):
            tokens = tuple(int(token) for token in input_token_ids)
            launches.append((tokens, int(session.position)))
            session.position += len(tokens)
            return SimpleNamespace(token_ids=[7, 8, 9, 10])

    graph = FakeReusableGraph()

    def fake_capture(_session, input_token_ids, **kwargs):
        assert kwargs["bulk_attention_mode"] == "native"
        assert kwargs["capture_pre_output_norm_hidden"] is True
        captures.append(tuple(int(token) for token in input_token_ids))
        return graph

    monkeypatch.setattr(
        "hipengine.runtime.gguf_native_spec_cycle.capture_qwen35_gguf_native_b2_target_graph",
        fake_capture,
    )

    first = verify_qwen35_gguf_native_b2_target(
        session,
        [1, 2, 3, 4],
        bulk_attention_mode="native",
        capture_linear_state_rows=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
    )
    second = verify_qwen35_gguf_native_b2_target(
        session,
        [5, 6, 7, 8],
        bulk_attention_mode="native",
        capture_linear_state_rows=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
    )

    assert first.token_ids == second.token_ids == [7, 8, 9, 10]
    assert captures == [(1, 2, 3, 4)]
    assert launches == [((1, 2, 3, 4), 17), ((5, 6, 7, 8), 21)]
    assert session.calls == []
    assert session._native_spec_b3_target_graph is graph


def test_native_linear_chain_scheduler_stages_all_rows_once(monkeypatch) -> None:
    from hipengine.loading.qwen35_gguf import LINEAR_ATTENTION
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner

    runner = object.__new__(Qwen35GGUFFullStackRunner)
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(hidden_size=16, is_moe=False)
    )
    runner.runtime = SimpleNamespace()
    calls: list[tuple[object, ...]] = []

    def staged(*args, **kwargs):
        calls.append(("staged", *args, kwargs))

    def scalar(*args, **kwargs):
        calls.append(("scalar", *args, kwargs))

    def ffn(*args, **kwargs):
        calls.append(("ffn", *args, kwargs))

    monkeypatch.setattr(
        runner,
        "_run_linear_attention_attn_chain_rows_exact",
        staged,
        raising=False,
    )
    monkeypatch.setattr(runner, "_run_linear_attention_attn_only", scalar)
    monkeypatch.setattr(runner, "_run_post_attention_ffn_rows", ffn)
    scratch = SimpleNamespace(attn_out=SimpleNamespace(ptr=0x5000))
    decode_scratch = SimpleNamespace()

    runner._run_native_attention_bulk_ffn_layer_rows(
        7,
        LINEAR_ATTENTION,
        0x1000,
        0x2000,
        scratch,
        rows=4,
        decode_scratch=decode_scratch,
        start_position=11,
        commit_final_linear_state=False,
        hidden_f32_ptr=0x3000,
        out_f32_ptr=0x4000,
    )

    assert [call[0] for call in calls] == ["staged", "ffn"]
    staged_call = calls[0]
    assert staged_call[1:5] == (7, 0x1000, 0x5000, scratch)
    assert staged_call[-1]["rows"] == 4
    assert staged_call[-1]["decode_scratch"] is decode_scratch
    assert staged_call[-1]["commit_final_linear_state"] is False
    assert staged_call[-1]["hidden_f32_ptr"] == 0x3000

    calls.clear()
    runner.weights.config.is_moe = True
    runner._run_native_attention_bulk_ffn_layer_rows(
        7,
        LINEAR_ATTENTION,
        0x1000,
        0x2000,
        scratch,
        rows=4,
        decode_scratch=decode_scratch,
    )
    assert [call[0] for call in calls] == ["scalar"] * 4 + ["ffn"]

    calls.clear()
    runner.weights.config.is_moe = False
    runner._run_native_attention_bulk_ffn_layer_rows(
        7,
        LINEAR_ATTENTION,
        0x1000,
        0x2000,
        scratch,
        rows=1,
        decode_scratch=decode_scratch,
    )
    assert [call[0] for call in calls] == ["scalar", "ffn"]


def test_native_long_context_serializes_dense_ffn_rows(monkeypatch) -> None:
    from hipengine.loading.qwen35_gguf import LINEAR_ATTENTION
    from hipengine.runtime import qwen35_gguf_runner as qgr
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner

    runner = object.__new__(Qwen35GGUFFullStackRunner)
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(hidden_size=16, is_moe=False)
    )
    runner.runtime = SimpleNamespace()
    calls: list[tuple[object, ...]] = []
    dispatch_states: list[bool] = []

    monkeypatch.setattr(
        runner,
        "_run_linear_attention_attn_chain_rows_exact",
        lambda *args, **kwargs: calls.append(("attention", args, kwargs)),
        raising=False,
    )

    def attention(*args, **kwargs):
        dispatch_states.append(qgr.gguf_native_batch_decode_enabled())
        calls.append(("attention", args, kwargs))

    def ffn(*args, **kwargs):
        dispatch_states.append(qgr.gguf_native_batch_decode_enabled())
        calls.append(("ffn", args, kwargs))

    monkeypatch.setattr(runner, "_run_linear_attention_attn_only", attention)
    monkeypatch.setattr(runner, "_run_post_attention_ffn_rows", ffn)
    monkeypatch.setattr(
        qgr,
        "_use_gguf_full_attention_split_decode",
        lambda context: int(context) >= 1024,
    )
    scratch = SimpleNamespace(attn_out=SimpleNamespace(ptr=0x5000))

    with qgr.native_batch_decode_session(True):
        runner._run_native_attention_bulk_ffn_layer_rows(
            46,
            LINEAR_ATTENTION,
            0x1000,
            0x2000,
            scratch,
            rows=4,
            decode_scratch=SimpleNamespace(),
            start_position=1020,
            hidden_f32_ptr=0x3000,
            out_f32_ptr=0x4000,
        )

    assert [call[0] for call in calls] == [
        "attention",
        "attention",
        "attention",
        "attention",
        "ffn",
        "ffn",
        "ffn",
        "ffn",
    ]
    ffn_calls = calls[4:]
    assert [call[1][1:4] for call in ffn_calls] == [
        (0x1000 + row * 32, 0x5000 + row * 32, 0x2000 + row * 32)
        for row in range(4)
    ]
    assert all(call[2]["rows"] == 1 for call in ffn_calls)
    assert [call[2]["hidden_f32_ptr"] for call in ffn_calls] == [
        0x3000 + row * 64 for row in range(4)
    ]
    assert [call[2]["out_f32_ptr"] for call in ffn_calls] == [
        0x4000 + row * 64 for row in range(4)
    ]
    assert dispatch_states == [False] * 8


def test_native_full_attention_chain_scheduler_stages_dynamic_dense_rows_once(
    monkeypatch,
) -> None:
    from hipengine.loading.qwen35_gguf import FULL_ATTENTION
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner

    runner = object.__new__(Qwen35GGUFFullStackRunner)
    runner.weights = SimpleNamespace(
        config=SimpleNamespace(hidden_size=16, is_moe=False)
    )
    runner.runtime = SimpleNamespace()
    calls: list[tuple[object, ...]] = []

    def staged(*args, **kwargs):
        calls.append(("staged", *args, kwargs))

    def scalar(*args, **kwargs):
        calls.append(("scalar", *args, kwargs))

    def ffn(*args, **kwargs):
        calls.append(("ffn", *args, kwargs))

    monkeypatch.setattr(
        runner,
        "_run_full_attention_attn_chain_rows_exact",
        staged,
        raising=False,
    )
    monkeypatch.setattr(runner, "_run_full_attention_attn_only", scalar)
    monkeypatch.setattr(runner, "_run_post_attention_ffn_rows", ffn)
    scratch = SimpleNamespace(attn_out=SimpleNamespace(ptr=0x5000))
    decode_scratch = SimpleNamespace(set_full_attention_position=lambda *_: None)
    row_scratches = tuple(SimpleNamespace() for _ in range(4))

    runner._run_native_attention_bulk_ffn_layer_rows(
        7,
        FULL_ATTENTION,
        0x1000,
        0x2000,
        scratch,
        rows=4,
        decode_scratch=decode_scratch,
        start_position=11,
        hidden_f32_ptr=0x3000,
        out_f32_ptr=0x4000,
        decode_row_scratches=row_scratches,
        attention_context_limit=768,
    )

    assert [call[0] for call in calls] == ["staged", "ffn"]
    staged_call = calls[0]
    assert staged_call[1:5] == (7, 0x1000, 0x5000, scratch)
    assert staged_call[-1]["rows"] == 4
    assert staged_call[-1]["decode_row_scratches"] is row_scratches
    assert staged_call[-1]["start_position"] == 11
    assert staged_call[-1]["hidden_f32_ptr"] == 0x3000
    assert staged_call[-1]["attention_context_limit"] == 768

    calls.clear()
    runner._run_native_attention_bulk_ffn_layer_rows(
        7,
        FULL_ATTENTION,
        0x1000,
        0x2000,
        scratch,
        rows=4,
        decode_scratch=decode_scratch,
        start_position=11,
    )
    assert [call[0] for call in calls] == ["scalar"] * 4 + ["ffn"]

    calls.clear()
    runner.weights.config.is_moe = True
    runner._run_native_attention_bulk_ffn_layer_rows(
        7,
        FULL_ATTENTION,
        0x1000,
        0x2000,
        scratch,
        rows=4,
        decode_scratch=decode_scratch,
        decode_row_scratches=row_scratches,
    )
    assert [call[0] for call in calls] == ["scalar"] * 4 + ["ffn"]

    calls.clear()
    runner.weights.config.is_moe = False
    runner._run_native_attention_bulk_ffn_layer_rows(
        7,
        FULL_ATTENTION,
        0x1000,
        0x2000,
        scratch,
        rows=1,
        decode_scratch=decode_scratch,
        decode_row_scratches=row_scratches[:1],
    )
    assert [call[0] for call in calls] == ["scalar", "ffn"]


def test_staged_full_attention_batches_shared_cache_only_with_exact_owner(
    monkeypatch,
) -> None:
    from hipengine.core.device import Device
    from hipengine.core.dtype import DType
    from hipengine.core.tensor import Tensor
    from hipengine.kvcache import KVLiveSpans
    from hipengine.runtime import qwen35_gguf_runner as qgr

    class Weight:
        def __init__(self, name: str, ptr: int):
            self.name = name
            self.ptr = ptr
            self.allocations = {
                "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=ptr + 0x10)),
                "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=ptr + 0x20)),
                "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=ptr + 0x30)),
            }

        def allocation(self, name: str | None = None):
            if name is None:
                return SimpleNamespace(tensor=SimpleNamespace(ptr=self.ptr))
            return self.allocations[name]

        def has_allocation(self, name: str) -> bool:
            return name in self.allocations

    weights = {
        name: Weight(name, 0xA000 + index * 0x100)
        for index, name in enumerate(
            ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm", "attn_output")
        )
    }
    layer = SimpleNamespace(weight=lambda name: weights[name])
    cfg = SimpleNamespace(
        hidden_size=16,
        head_count=2,
        head_count_kv=1,
        key_length=4,
        value_length=4,
        rope_dimension_count=4,
        rms_norm_eps=1.0e-6,
    )
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    runner.weights = SimpleNamespace(config=cfg, layer=lambda _layer_id: layer)
    runner.runtime = SimpleNamespace()
    runner.backend = "hip_gfx1100"
    runner._gguf_prefill_quant = "gguf_q4_k_m"
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(qgr, "resolve", lambda **_kwargs: None)

    monkeypatch.setattr(
        runner,
        "_run_attention_norm_rows",
        lambda **kwargs: calls.append(("norm", kwargs)),
    )
    monkeypatch.setattr(runner, "_cast_library", lambda: "cast-lib")
    monkeypatch.setattr(runner, "_paged_kv_write_library", lambda: "write-lib")
    monkeypatch.setattr(runner, "_paged_attn_decode_library", lambda: "attn-lib")

    def linear(weight, x_ptr, out_ptr, **kwargs):
        calls.append(("linear", weight.name, x_ptr, out_ptr, kwargs["rows"]))

    monkeypatch.setattr(qgr, "launch_gguf_linear", linear)
    monkeypatch.setattr(
        qgr,
        "launch_gguf_q4_t16_sidecar_decode",
        lambda *_args, **_kwargs: False,
        raising=False,
    )
    monkeypatch.setattr(
        qgr,
        "qwen35_split_qgate_bf16",
        lambda q, query, gate, rows, *args, **kwargs: calls.append(
            ("split", q, query, gate, rows)
        ),
    )
    monkeypatch.setattr(
        qgr,
        "bf16_to_f32",
        lambda src, dst, count, **kwargs: calls.append(("cast", src, dst, count)),
    )
    monkeypatch.setattr(
        qgr,
        "gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight",
        lambda *args, **kwargs: calls.append(
            ("rotary", args[0], args[1], args[6], args[10], args[15])
        ),
    )
    monkeypatch.setattr(
        qgr,
        "qwen35_write_paged_kv_mixed_value_bf16_spans",
        lambda key, value, key_cache, value_cache, spans, *args, **kwargs: calls.append(
            ("write", key, value, key_cache, value_cache, spans.span_role)
        ),
    )
    monkeypatch.setattr(
        qgr,
        "qwen35_paged_full_attn_decode_context_bf16_spans",
        lambda query, key_cache, value_cache, context, spans, limit, *args, **kwargs: calls.append(
            ("attn", query, key_cache, value_cache, context, spans.span_role, limit)
        ),
    )
    monkeypatch.setattr(
        qgr,
        "qwen35_full_attn_gate_mul_bf16",
        lambda context, gate, gated, width, **kwargs: calls.append(
            ("gate", context, gate, gated, width)
        ),
    )

    scratch = SimpleNamespace(
        norm=SimpleNamespace(ptr=0x1000),
        full_q=SimpleNamespace(ptr=0x2000),
        full_k=SimpleNamespace(ptr=0x3000),
        full_v=SimpleNamespace(ptr=0x4000),
        full_query_raw=SimpleNamespace(ptr=0x5000),
        full_gate=SimpleNamespace(ptr=0x6000),
        positions_tensor=SimpleNamespace(ptr=0x7000),
        full_key_raw=SimpleNamespace(ptr=0x7800),
        full_query=SimpleNamespace(ptr=0x8000),
        full_key=SimpleNamespace(ptr=0x9000),
        full_gated=SimpleNamespace(ptr=0xA000),
    )
    device = Device("hip", 0)
    block_table = Tensor.from_handle(0xD000, (1,), DType.INT32, device)
    row_scratches = []
    for row in range(4):
        position = Tensor.from_handle(0xD100 + row * 8, (1,), DType.INT64, device)
        context = Tensor.from_handle(0xD200 + row * 8, (1,), DType.INT64, device)
        row_scratches.append(
            SimpleNamespace(
                kv_storage_dtype=DType.BF16,
                position_host=np.asarray([11 + row], dtype=np.int64),
                max_positions=128,
                append_spans=KVLiveSpans.paged_uniform(
                    block_table=block_table,
                    live_counts=position,
                    max_live_count=127,
                    storage_dtype=DType.BF16,
                    row_positions=position,
                    span_role="verify_chain",
                ),
                decode_spans=KVLiveSpans.paged_uniform(
                    block_table=block_table,
                    live_counts=context,
                    max_live_count=128,
                    storage_dtype=DType.BF16,
                    row_positions=position,
                    span_role="verify_chain",
                ),
                block_size=256,
                cos_table=SimpleNamespace(ptr=0xB000),
                sin_table=SimpleNamespace(ptr=0xC000),
                full_cache=lambda _layer_id: (
                    SimpleNamespace(ptr=0xE000),
                    SimpleNamespace(ptr=0xF000),
                ),
            )
        )

    runner._run_full_attention_attn_chain_rows_exact(
        7,
        0xF000,
        0x11000,
        scratch,
        rows=4,
        decode_row_scratches=tuple(row_scratches),
        start_position=11,
        hidden_f32_ptr=0x12000,
        attention_context_limit=128,
    )

    linear_calls = [call for call in calls if call[0] == "linear"]
    assert linear_calls == [
        ("linear", "attn_q", 0x1000, 0x2000, 4),
        ("linear", "attn_k", 0x1000, 0x3000, 1),
        ("linear", "attn_k", 0x1020, 0x3008, 1),
        ("linear", "attn_k", 0x1040, 0x3010, 1),
        ("linear", "attn_k", 0x1060, 0x3018, 1),
        ("linear", "attn_v", 0x1000, 0x4000, 4),
        ("linear", "attn_output", 0xA000, 0x11000, 4),
    ]
    serial_calls = [call for call in calls if call[0] in {"write", "attn", "gate"}]
    assert [call[0] for call in serial_calls] == ["write", "attn", "gate"] * 4
    assert serial_calls[0] == ("write", 0x9000, 0x4000, 0xE000, 0xF000, "verify_chain")
    assert serial_calls[3] == ("write", 0x9010, 0x4008, 0xE000, 0xF000, "verify_chain")
    assert serial_calls[6] == ("write", 0x9020, 0x4010, 0xE000, 0xF000, "verify_chain")
    assert serial_calls[9] == ("write", 0x9030, 0x4018, 0xE000, 0xF000, "verify_chain")
    assert [call[-1] for call in serial_calls if call[0] == "attn"] == [128] * 4

    def batch_attn(
        query,
        key_cache,
        value_cache,
        context,
        spans,
        rows,
        limit,
        *args,
        **kwargs,
    ):
        calls.append(
            (
                "attn_batch",
                query,
                key_cache,
                value_cache,
                context,
                spans.base_offsets.ptr,
                spans.live_counts.ptr,
                rows,
                limit,
            )
        )

    monkeypatch.setattr(
        runner,
        "_full_attn_decode_batch_shared_native_fn",
        lambda: batch_attn,
        raising=False,
    )
    calls.clear()
    runner._run_full_attention_attn_chain_rows_exact(
        7,
        0xF000,
        0x11000,
        scratch,
        rows=4,
        decode_row_scratches=tuple(row_scratches),
        start_position=11,
        hidden_f32_ptr=0x12000,
        attention_context_limit=128,
    )

    cache_calls = [
        call for call in calls if call[0] in {"write", "attn", "attn_batch", "gate"}
    ]
    assert [call[0] for call in cache_calls] == ["write"] * 4 + [
        "attn_batch",
        "gate",
    ]
    assert cache_calls[0] == (
        "write",
        0x9000,
        0x4000,
        0xE000,
        0xF000,
        "verify_chain",
    )
    assert cache_calls[3] == (
        "write",
        0x9030,
        0x4018,
        0xE000,
        0xF000,
        "verify_chain",
    )
    assert cache_calls[4] == (
        "attn_batch",
        0x8000,
        0xE000,
        0xF000,
        0x5000,
        0xD000,
        0xD200,
        4,
        128,
    )
    assert cache_calls[5] == ("gate", 0x5000, 0x6000, 0xA000, 32)

    def sidecar_k(
        weight,
        x,
        out,
        rows,
        in_features,
        out_features,
        **kwargs,
    ):
        calls.append(
            (
                "k_sidecar",
                weight.name,
                x,
                out,
                rows,
                in_features,
                out_features,
                kwargs["backend"],
                kwargs["stream"],
            )
        )
        return True

    monkeypatch.setattr(
        qgr,
        "launch_gguf_q4_t16_sidecar_decode",
        sidecar_k,
        raising=False,
    )
    calls.clear()
    runner._run_full_attention_attn_chain_rows_exact(
        7,
        0xF000,
        0x11000,
        scratch,
        rows=4,
        decode_row_scratches=tuple(row_scratches),
        start_position=11,
        hidden_f32_ptr=0x12000,
        attention_context_limit=128,
    )
    assert [call for call in calls if call[0] == "linear"] == [
        ("linear", "attn_q", 0x1000, 0x2000, 4),
        ("linear", "attn_v", 0x1000, 0x4000, 4),
        ("linear", "attn_output", 0xA000, 0x11000, 4),
    ]
    assert [call for call in calls if call[0] == "k_sidecar"] == [
        (
            "k_sidecar",
            "attn_k",
            0x1000,
            0x3000,
            4,
            16,
            4,
            "hip_gfx1100",
            0,
        )
    ]

    def batch_k(
        x,
        qweight,
        scales,
        mins,
        out,
        rows,
        in_features,
        out_features,
        **kwargs,
    ):
        calls.append(
            (
                "k_batch",
                x,
                qweight,
                scales,
                mins,
                out,
                rows,
                in_features,
                out_features,
                kwargs["stream"],
            )
        )

    monkeypatch.setattr(
        qgr,
        "launch_gguf_q4_t16_sidecar_decode",
        lambda *_args, **_kwargs: False,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_full_attn_k_grid_y_batch_fn",
        lambda: batch_k,
        raising=False,
    )
    calls.clear()
    runner._run_full_attention_attn_chain_rows_exact(
        7,
        0xF000,
        0x11000,
        scratch,
        rows=4,
        decode_row_scratches=tuple(row_scratches),
        start_position=11,
        hidden_f32_ptr=0x12000,
        attention_context_limit=128,
    )
    assert [call for call in calls if call[0] == "linear"] == [
        ("linear", "attn_q", 0x1000, 0x2000, 4),
        ("linear", "attn_v", 0x1000, 0x4000, 4),
        ("linear", "attn_output", 0xA000, 0x11000, 4),
    ]
    assert [call for call in calls if call[0] == "k_batch"] == [
        (
            "k_batch",
            0x1000,
            weights["attn_k"].ptr + 0x10,
            weights["attn_k"].ptr + 0x20,
            weights["attn_k"].ptr + 0x30,
            0x3000,
            4,
            16,
            4,
            0,
        )
    ]
    assert [
        call[0]
        for call in calls
        if call[0] in {"write", "attn", "attn_batch", "gate"}
    ] == ["write"] * 4 + ["attn_batch", "gate"]

    bad_context = Tensor.from_handle(0xD280, (1,), DType.INT64, device)
    row_scratches[1].decode_spans = KVLiveSpans.paged_uniform(
        block_table=block_table,
        live_counts=bad_context,
        max_live_count=128,
        storage_dtype=DType.BF16,
        row_positions=row_scratches[1].append_spans.row_positions,
        span_role="verify_chain",
    )
    calls.clear()
    runner._run_full_attention_attn_chain_rows_exact(
        7,
        0xF000,
        0x11000,
        scratch,
        rows=4,
        decode_row_scratches=tuple(row_scratches),
        start_position=11,
        hidden_f32_ptr=0x12000,
        attention_context_limit=128,
    )
    fallback_calls = [
        call for call in calls if call[0] in {"write", "attn", "attn_batch", "gate"}
    ]
    assert [call[0] for call in fallback_calls] == ["write", "attn", "gate"] * 4


def test_native_b2_target_falls_back_before_capture_when_provider_key_is_missing(
    monkeypatch,
) -> None:
    session = _FallbackSession()
    session.backend = "hip_gfx1151"
    session.runner = object()
    session.scratch = object()
    session.host_token_embedding_enabled = False
    session.use_expert_sidecar = False
    session.kv_storage_dtype = "bf16"
    session.position = 8
    monkeypatch.setattr(
        "hipengine.runtime.gguf_native_spec_cycle.resolve",
        lambda **_kwargs: None,
    )

    result = verify_qwen35_gguf_native_b2_target(session, [1, 2, 3])

    assert result.token_ids == [7, 8]
    assert session.calls == [
        (
            (1, 2, 3),
            {
                "bulk_attention_mode": "bulk",
                "use_wmma_prefill": False,
                "capture_linear_state_rows": False,
                "defer_linear_state_commit": False,
            },
        )
    ]
    assert "not registered" in session.last_native_spec_target_fallback_reason


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _read_buffer(runtime, buffer) -> np.ndarray:
    host = np.empty(int(buffer.nbytes), dtype=np.uint8)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(int(buffer.ptr), int(buffer.nbytes)),
        host.nbytes,
        runtime=runtime,
    )
    return host


def _linear_state_row_buffers(session) -> tuple[object, ...]:
    return tuple(
        buffer
        for pair in zip(
            session._verify_linear_conv_state_rows,
            session._verify_linear_recurrent_state_rows,
            strict=True,
        )
        for buffer in pair
        if buffer is not None
    )


def _read_linear_state_rows(session) -> tuple[np.ndarray, ...]:
    return tuple(
        _read_buffer(session.runtime, buffer)
        for buffer in _linear_state_row_buffers(session)
    )


def _read_linear_state_row_prefix(session, rows: int) -> tuple[np.ndarray, ...]:
    row_buffers = _linear_state_row_buffers(session)
    resident = tuple(
        buffer
        for pair in zip(
            session.scratch.layer_conv_states,
            session.scratch.layer_recurrent_states,
            strict=True,
        )
        for buffer in pair
        if buffer is not None
    )
    assert len(row_buffers) == len(resident)
    values = []
    for row_buffer, state_buffer in zip(row_buffers, resident, strict=True):
        prefix = SimpleNamespace(ptr=row_buffer.ptr, nbytes=rows * int(state_buffer.nbytes))
        values.append(_read_buffer(session.runtime, prefix))
    return tuple(values)


def _read_resident_linear_state(session) -> tuple[np.ndarray, ...]:
    return tuple(
        _read_buffer(session.runtime, buffer)
        for pair in zip(
            session.scratch.layer_conv_states,
            session.scratch.layer_recurrent_states,
            strict=True,
        )
        for buffer in pair
        if buffer is not None
    )


def _full_kv_buffers(session) -> tuple[object, ...]:
    return tuple(
        buffer
        for pair in zip(
            session.scratch.full_key_caches,
            session.scratch.full_value_caches,
            strict=True,
        )
        for buffer in pair
        if buffer is not None
    )


def _read_full_kv(session) -> tuple[np.ndarray, ...]:
    return tuple(
        _read_buffer(session.runtime, buffer)
        for buffer in _full_kv_buffers(session)
    )


def _write_full_kv(session, values: tuple[np.ndarray, ...]) -> None:
    buffers = _full_kv_buffers(session)
    assert len(buffers) == len(values)
    for buffer, value in zip(buffers, values, strict=True):
        restored = np.ascontiguousarray(value, dtype=np.uint8)
        assert restored.nbytes == int(buffer.nbytes)
        copy_host_to_device(
            buffer,
            host_array_ptr(restored),
            restored.nbytes,
            runtime=session.runtime,
        )


_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
def test_native_b2_target_graph_matches_eager_hidden_state_and_kv(
    monkeypatch,
    hip_test_target_arch: str,
) -> None:
    if hip_test_target_arch not in {"gfx1100", "gfx1151"}:
        pytest.skip(f"native target graph is not registered for {hip_test_target_arch}")

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_PREFILL_DEVICE_METADATA", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_RESIDUAL", "1")
    prompt = [760, 4087, 369, 220, 16, 17, 18, 19]

    with Qwen35GGUFResidentSession(
        _MODEL,
        max_sequence_length=256,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        first = session.prefill(
            prompt,
            use_bulk=True,
            return_logits=False,
            capture_hidden_seed_fp32=True,
        )
        block_inputs = [int(first.token_id), 1, 2]
        next_inputs = [3, 4, 5]
        start_position = int(session.position)
        snapshot = session._linear_state_snapshot()
        kv_snapshot = _read_full_kv(session)
        try:
            eager = session.verify_target_block(
                block_inputs,
                use_wmma_prefill=False,
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
            )
            eager_rows = _read_linear_state_rows(session)
            eager_first_state = _read_resident_linear_state(session)
            eager_first_kv = _read_full_kv(session)
            session._commit_verify_linear_state_row(2, position=start_position + 3)
            eager_next = session.verify_target_block(
                next_inputs,
                use_wmma_prefill=False,
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
            )
            eager_next_rows = _read_linear_state_rows(session)
            eager_state = _read_resident_linear_state(session)
            eager_kv = _read_full_kv(session)

            session._restore_linear_state_snapshot(snapshot, position=start_position)
            _write_full_kv(session, kv_snapshot)
            assert session._verify_hidden_seed_buf is not None
            session.runtime.memset(
                session._verify_hidden_seed_buf.ptr,
                0xA5,
                session._verify_hidden_seed_buf.nbytes,
            )
            for buffer in _linear_state_row_buffers(session):
                session.runtime.memset(buffer.ptr, 0xA5, buffer.nbytes)
            session.runtime.device_synchronize()
            with session.capture_native_spec_target_graph(
                block_inputs,
                cycle_id=31,
                transaction_id=41,
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
            ) as graph:
                native = graph.launch()
                assert graph.launch_count == 1
                assert graph.native_result is not None
                assert graph.native_result.cycle_id == 31
                assert graph.native_result.transaction_id == 41
                native_rows = _read_linear_state_rows(session)
                native_first_state = _read_resident_linear_state(session)
                native_first_kv = _read_full_kv(session)

                session._commit_verify_linear_state_row(2, position=start_position + 3)
                replayed_native = graph.launch(
                    next_inputs,
                    cycle_id=32,
                    transaction_id=42,
                )
                assert graph.launch_count == 2
                assert graph.native_result is not None
                assert graph.native_result.cycle_id == 32
                assert graph.native_result.transaction_id == 42

            native_next_rows = _read_linear_state_rows(session)
            native_state = _read_resident_linear_state(session)
            native_kv = _read_full_kv(session)

            b1_inputs = block_inputs[:2]
            session._restore_linear_state_snapshot(snapshot, position=start_position)
            _write_full_kv(session, kv_snapshot)
            eager_b1 = session.verify_target_block(
                b1_inputs,
                use_wmma_prefill=False,
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
            )
            eager_b1_rows = _read_linear_state_row_prefix(session, 2)
            eager_b1_state = _read_resident_linear_state(session)
            eager_b1_kv = _read_full_kv(session)

            session._restore_linear_state_snapshot(snapshot, position=start_position)
            _write_full_kv(session, kv_snapshot)
            for buffer in _linear_state_row_buffers(session):
                session.runtime.memset(buffer.ptr, 0xA5, buffer.nbytes)
            session.runtime.device_synchronize()
            with session.capture_native_spec_target_graph(
                b1_inputs,
                cycle_id=33,
                transaction_id=43,
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
            ) as b1_graph:
                native_b1 = b1_graph.launch()
                assert b1_graph.launch_count == 1
                assert b1_graph.native_result is not None
                assert b1_graph.native_result.cycle_id == 33
                assert b1_graph.native_result.transaction_id == 43
            native_b1_rows = _read_linear_state_row_prefix(session, 2)
            native_b1_state = _read_resident_linear_state(session)
            native_b1_kv = _read_full_kv(session)

            # N2 oracle: target rows stay on device, acceptance selects one
            # captured recurrent/hidden row, and one bounded payload is returned.
            session._restore_linear_state_snapshot(snapshot, position=start_position)
            _write_full_kv(session, kv_snapshot)
            eager_n2 = session.verify_target_block(
                block_inputs,
                use_wmma_prefill=False,
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
            )
            n2_accepted = 0
            for draft_token, target_token in zip(block_inputs[1:], eager_n2.token_ids, strict=False):
                if int(draft_token) != int(target_token):
                    break
                n2_accepted += 1
            n2_visible_tokens = [int(token) for token in eager_n2.token_ids[:n2_accepted + 1]]
            session._commit_verify_linear_state_row(
                n2_accepted,
                position=start_position + n2_accepted + 1,
            )
            eager_n2_state = _read_resident_linear_state(session)
            eager_n2_kv = _read_full_kv(session)
            eager_n2_hidden = _read_buffer(session.runtime, session.scratch.hidden_seed_fp32)

            session._restore_linear_state_snapshot(snapshot, position=start_position)
            _write_full_kv(session, kv_snapshot)
            with session.capture_native_spec_target_graph(
                block_inputs,
                cycle_id=34,
                transaction_id=44,
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
                device_accept_commit=True,
            ) as n2_graph:
                native_n2 = n2_graph.launch(remaining_decode=3)
                assert n2_graph.launch_count == 1
                assert n2_graph.native_result is not None
                assert n2_graph.native_result.visible_output_count == n2_accepted + 1
            native_n2_state = _read_resident_linear_state(session)
            native_n2_kv = _read_full_kv(session)
            native_n2_hidden = _read_buffer(session.runtime, session.scratch.hidden_seed_fp32)

            # Construct a guaranteed full-accept B2 chain from target samples,
            # then require dynamic row-2 commit parity as well as reject parity.
            session._restore_linear_state_snapshot(snapshot, position=start_position)
            _write_full_kv(session, kv_snapshot)
            full_probe0 = session.verify_target_block(block_inputs, use_wmma_prefill=False)
            full_draft0 = int(full_probe0.token_ids[0])
            session._restore_linear_state_snapshot(snapshot, position=start_position)
            _write_full_kv(session, kv_snapshot)
            full_probe1 = session.verify_target_block(
                [block_inputs[0], full_draft0, block_inputs[2]],
                use_wmma_prefill=False,
            )
            full_draft1 = int(full_probe1.token_ids[1])
            full_inputs = [block_inputs[0], full_draft0, full_draft1]

            session._restore_linear_state_snapshot(snapshot, position=start_position)
            _write_full_kv(session, kv_snapshot)
            eager_n2_full = session.verify_target_block(
                full_inputs,
                use_wmma_prefill=False,
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
            )
            assert eager_n2_full.token_ids[:2] == full_inputs[1:]
            session._commit_verify_linear_state_row(2, position=start_position + 3)
            eager_n2_full_state = _read_resident_linear_state(session)
            eager_n2_full_kv = _read_full_kv(session)
            eager_n2_full_hidden = _read_buffer(session.runtime, session.scratch.hidden_seed_fp32)

            session._restore_linear_state_snapshot(snapshot, position=start_position)
            _write_full_kv(session, kv_snapshot)
            with session.capture_native_spec_target_graph(
                full_inputs,
                cycle_id=35,
                transaction_id=45,
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
                device_accept_commit=True,
            ) as n2_full_graph:
                native_n2_full = n2_full_graph.launch(remaining_decode=3)
                native_n2_full_hidden_rows = np.empty_like(eager_n2_full.hidden_seeds)
                copy_device_to_host(
                    host_array_ptr(native_n2_full_hidden_rows),
                    DeviceBuffer(
                        int(native_n2_full.hidden_seed_rows_ptr),
                        int(native_n2_full_hidden_rows.nbytes),
                    ),
                    native_n2_full_hidden_rows.nbytes,
                    runtime=session.runtime,
                )
            native_n2_full_state = _read_resident_linear_state(session)
            native_n2_full_kv = _read_full_kv(session)
            native_n2_full_hidden = _read_buffer(session.runtime, session.scratch.hidden_seed_fp32)
        finally:
            session._free_linear_state_snapshot(snapshot)

    assert eager.token_ids == native.token_ids
    assert eager_next.token_ids == replayed_native.token_ids
    assert eager_b1.token_ids == native_b1.token_ids
    assert np.all(np.isfinite(native.hidden_seeds))
    np.testing.assert_array_equal(native.hidden_seeds, eager.hidden_seeds)
    np.testing.assert_array_equal(replayed_native.hidden_seeds, eager_next.hidden_seeds)
    np.testing.assert_array_equal(native_b1.hidden_seeds, eager_b1.hidden_seeds)
    assert len(eager_rows) == len(native_rows) == 60
    assert all(np.array_equal(expected, actual) for expected, actual in zip(eager_rows, native_rows, strict=True))
    assert len(eager_next_rows) == len(native_next_rows) == 60
    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(eager_next_rows, native_next_rows, strict=True)
    )
    assert len(eager_b1_rows) == len(native_b1_rows) == 60
    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(eager_b1_rows, native_b1_rows, strict=True)
    )
    assert len(eager_first_state) == len(native_first_state) == 60
    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(eager_first_state, native_first_state, strict=True)
    )
    assert len(eager_state) == len(native_state) == 60
    assert all(np.array_equal(expected, actual) for expected, actual in zip(eager_state, native_state, strict=True))
    assert len(eager_first_kv) == len(native_first_kv) == 20
    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(eager_first_kv, native_first_kv, strict=True)
    )
    assert len(eager_kv) == len(native_kv) == 20
    assert all(np.array_equal(expected, actual) for expected, actual in zip(eager_kv, native_kv, strict=True))
    assert len(eager_b1_state) == len(native_b1_state) == 60
    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(eager_b1_state, native_b1_state, strict=True)
    )
    assert len(eager_b1_kv) == len(native_b1_kv) == 20
    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(eager_b1_kv, native_b1_kv, strict=True)
    )
    assert native.start_position == start_position
    assert replayed_native.start_position == start_position + 3
    assert native_b1.start_position == start_position
    assert native_n2.device_accept_commit is True
    assert native_n2.token_ids == n2_visible_tokens
    assert native_n2.accepted_draft_tokens == n2_accepted
    assert native_n2.commit_row == n2_accepted
    assert native_n2.end_position == start_position + n2_accepted + 1
    assert native_n2.hidden_seeds.shape == (0, 0)
    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(eager_n2_state, native_n2_state, strict=True)
    )
    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(eager_n2_kv, native_n2_kv, strict=True)
    )
    np.testing.assert_array_equal(native_n2_hidden, eager_n2_hidden)
    assert native_n2_full.accepted_draft_tokens == 2
    assert native_n2_full.token_ids == [int(token) for token in eager_n2_full.token_ids]
    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(eager_n2_full_state, native_n2_full_state, strict=True)
    )
    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(eager_n2_full_kv, native_n2_full_kv, strict=True)
    )
    np.testing.assert_array_equal(native_n2_full_hidden, eager_n2_full_hidden)
    np.testing.assert_array_equal(native_n2_full_hidden_rows, eager_n2_full.hidden_seeds)
    assert session.position == start_position + 3
