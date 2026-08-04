from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
)

from hipengine.runtime.gguf_native_spec_cycle import (
    NativeSpecTargetGraphUnsupportedError,
    build_native_b2_target_batch,
    run_qwen35_gguf_native_mtp_cycle,
    verify_qwen35_gguf_native_b2_target,
)
from hipengine.speculative.mtp_resident_draft import (
    NativeSpecProposalGraphUnsupportedError,
)


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
    with pytest.raises(NativeSpecTargetGraphUnsupportedError, match="two to four rows"):
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


def test_staged_full_attention_keeps_k_and_cache_attention_row_serial(
    monkeypatch,
) -> None:
    from hipengine.core.dtype import DType
    from hipengine.runtime import qwen35_gguf_runner as qgr

    class Weight:
        def __init__(self, name: str, ptr: int):
            self.name = name
            self.ptr = ptr

        def allocation(self):
            return SimpleNamespace(tensor=SimpleNamespace(ptr=self.ptr))

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
    calls: list[tuple[object, ...]] = []

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
            ("write", key, value, key_cache, value_cache, spans.name)
        ),
    )
    monkeypatch.setattr(
        qgr,
        "qwen35_paged_full_attn_decode_context_bf16_spans",
        lambda query, key_cache, value_cache, context, spans, limit, *args, **kwargs: calls.append(
            ("attn", query, key_cache, value_cache, context, spans.name, limit)
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
    row_scratches = []
    for row in range(3):
        row_scratches.append(
            SimpleNamespace(
                kv_storage_dtype=DType.BF16,
                position_host=np.asarray([11 + row], dtype=np.int64),
                max_positions=128,
                append_spans=SimpleNamespace(name=f"append-{row}", scale_metadata=None),
                decode_spans=SimpleNamespace(name=f"decode-{row}"),
                block_size=256,
                cos_table=SimpleNamespace(ptr=0xB000),
                sin_table=SimpleNamespace(ptr=0xC000),
                full_cache=lambda _layer_id, row=row: (
                    SimpleNamespace(ptr=0xD000 + row * 0x100),
                    SimpleNamespace(ptr=0xE000 + row * 0x100),
                ),
            )
        )

    runner._run_full_attention_attn_chain_rows_exact(
        7,
        0xF000,
        0x11000,
        scratch,
        rows=3,
        decode_row_scratches=tuple(row_scratches),
        start_position=11,
        hidden_f32_ptr=0x12000,
        attention_context_limit=128,
    )

    linear_calls = [call for call in calls if call[0] == "linear"]
    assert linear_calls == [
        ("linear", "attn_q", 0x1000, 0x2000, 3),
        ("linear", "attn_k", 0x1000, 0x3000, 1),
        ("linear", "attn_k", 0x1020, 0x3008, 1),
        ("linear", "attn_k", 0x1040, 0x3010, 1),
        ("linear", "attn_v", 0x1000, 0x4000, 3),
        ("linear", "attn_output", 0xA000, 0x11000, 3),
    ]
    serial_calls = [call for call in calls if call[0] in {"write", "attn", "gate"}]
    assert [call[0] for call in serial_calls] == ["write", "attn", "gate"] * 3
    assert serial_calls[0] == ("write", 0x9000, 0x4000, 0xD000, 0xE000, "append-0")
    assert serial_calls[3] == ("write", 0x9010, 0x4008, 0xD100, 0xE100, "append-1")
    assert serial_calls[6] == ("write", 0x9020, 0x4010, 0xD200, 0xE200, "append-2")
    assert [call[-1] for call in serial_calls if call[0] == "attn"] == [128, 128, 128]


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
