from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.generation.deadline import GenerationCancelled
from hipengine.kvcache import KVScaleMetadata
from hipengine.runtime import qwen35_gguf_mtp as mtp_module
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession


def test_mtp_int8_target_policy_registration_keeps_scale_metadata() -> None:
    device = Device("hip", 0)
    block_table = Tensor.from_handle(0x1000, (4,), DType.INT32, device)
    live_counts = Tensor.from_handle(0x2000, (1,), DType.INT64, device)
    metadata = KVScaleMetadata(
        k_scale=Tensor.from_handle(0x3000, (4, 256, 2), DType.FP32, device),
        v_scale=Tensor.from_handle(0x4000, (4, 256, 2), DType.FP32, device),
        scale_dtype=DType.FP32,
    )
    owner = SimpleNamespace(
        block_size=256,
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        block_table_tensor=block_table,
        context_tensor=live_counts,
        max_positions=1024,
        full_kv_scale_metadata=(None, metadata, None),
    )
    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = SimpleNamespace(_target_scratch_owner=owner, position=9)

    policy = decoder._register_kv_policy(7)

    reservation = policy.reservations[7]
    assert reservation.storage_dtype is DType.INT8_PER_TOKEN_HEAD
    assert reservation.scale_metadata is metadata


@pytest.mark.parametrize(
    ("budget", "position", "remaining_decode", "expected", "expected_reason"),
    [
        (1, 1020, 2, True, None),
        (1, 1021, 2, False, "target_graph_proposal_handoff_boundary_miss"),
        (1, 1022, 2, False, "target_graph_context_bucket_miss"),
        (2, 1019, 3, True, None),
        (2, 1020, 3, False, "target_graph_proposal_handoff_boundary_miss"),
        (2, 1021, 3, False, "target_graph_context_bucket_miss"),
        (3, 1018, 4, True, None),
        (3, 1019, 4, False, "target_graph_proposal_handoff_boundary_miss"),
        (3, 1020, 4, False, "target_graph_context_bucket_miss"),
        (3, 1019, 3, False, "target_graph_output_room_miss"),
    ],
)
def test_device_proposal_ready_checks_live_cycle_end_and_output_room(
    budget: int,
    position: int,
    remaining_decode: int,
    expected: bool,
    expected_reason: str | None,
) -> None:
    class Graph:
        closed = False
        context_limit = 1023

        def compatible_with(self, _target, **_kwargs) -> bool:
            return True

        def launch_ineligibility_reason(self, _target, **kwargs) -> str | None:
            rows = int(kwargs["rows"])
            if int(kwargs["position"]) + rows > 1023:
                return "target_graph_context_bucket_miss"
            if int(kwargs["remaining_decode"]) < rows:
                return "target_graph_output_room_miss"
            return None

    target = SimpleNamespace(position=position)
    setattr(target, f"_native_spec_b{budget}_target_graph_n2", Graph())
    verifier = mtp_module.Qwen35GGUFTransactionalVerifier.__new__(
        mtp_module.Qwen35GGUFTransactionalVerifier
    )
    verifier.closed = False
    verifier.target_verify_mode = "native"
    verifier.target = target
    verifier.last_device_proposal_fallback_reason = "stale"

    assert verifier.device_proposal_ready(
        budget,
        remaining_decode=remaining_decode,
    ) is expected
    assert verifier.last_device_proposal_fallback_reason == expected_reason


def test_ineligible_cached_target_graph_never_launches_device_proposal() -> None:
    calls: list[tuple[object, ...]] = []

    class Verifier:
        last_device_proposal_fallback_reason = None

        def device_proposal_ready(self, budget, *, remaining_decode):
            calls.append(("ready", int(budget), int(remaining_decode)))
            self.last_device_proposal_fallback_reason = (
                "target_graph_context_bucket_miss"
            )
            return False

    class Provider:
        def launch_device_proposal(self, *_args, **_kwargs):
            calls.append(("launch",))
            raise AssertionError("ineligible target graph must prevent proposal launch")

    proposal = mtp_module._maybe_launch_device_proposal(
        Verifier(),
        Provider(),
        SimpleNamespace(root_positions=(1020,)),
        candidate_budget=3,
        remaining_decode=4,
        return_cycle_logits=False,
    )

    assert proposal is None
    assert calls == [("ready", 3, 4)]


def test_native_target_rows_follow_implementation_ladder_and_context_limit() -> None:
    assert mtp_module._effective_target_verify_mode("native", rows=4) == "native"
    assert (
        mtp_module._effective_target_verify_mode(
            "native", rows=3, backend="hip_gfx1100", end_position=95
        )
        == "native"
    )
    assert (
        mtp_module._effective_target_verify_mode(
            "native", rows=4, backend="hip_gfx1100", end_position=96
        )
        == "serial_exact"
    )
    for rows in range(2, 9):
        assert mtp_module._effective_target_verify_mode("native", rows=rows) == "native"
        assert mtp_module._effective_target_verify_mode("serial_exact", rows=rows) == "serial_exact"
    assert mtp_module._effective_target_verify_mode("native", rows=9) == "serial_exact"
    assert mtp_module._initial_state_only_journal_applies(
        "native",
        max_candidate_budget=3,
    )
    for budget in range(4, 8):
        assert mtp_module._initial_state_only_journal_applies(
            "native", max_candidate_budget=budget,
        )
    assert not mtp_module._initial_state_only_journal_applies(
        "native", max_candidate_budget=8,
    )
    assert not mtp_module._initial_state_only_journal_applies(
        "serial_exact",
        max_candidate_budget=3,
    )


def test_serial_fallback_forces_consumer_owned_initial_state_snapshot() -> None:
    journal = mtp_module._StateJournal.__new__(mtp_module._StateJournal)
    journal.target = SimpleNamespace(last_target_hidden=SimpleNamespace(ptr=0x2000))
    journal.initial_hidden = DeviceBuffer(0x1000, 8)
    journal.producer_capture_initial_state = True
    journal.initial_state_captured = False
    copies: list[tuple[int, int, int, int]] = []
    state_copies: list[tuple[bool, int]] = []
    journal._copy_d2d = lambda dst, src, nbytes, *, stream: copies.append(
        (int(dst), int(src), int(nbytes), int(stream))
    )
    journal._copy_initial_state = lambda *, restore, stream: (
        state_copies.append((bool(restore), int(stream))) or True
    )
    journal._capture_state_index = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("pair-copy snapshot should own this fixture")
    )

    journal.capture_initial(stream=9, force_consumer_state=True)

    assert copies == [(0x1000, 0x2000, 8, 9)]
    assert state_copies == [(False, 9)]
    assert journal.initial_state_captured


def test_mtp_prompt_admission_streams_shifted_draft_without_full_hidden_slab(
    monkeypatch,
) -> None:
    pending = DeviceBuffer(0x1000, 8)
    freed: list[int] = []
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )

    target_calls: list[tuple[tuple[int, ...], dict[str, object]]] = []

    class Runtime:
        def memset(self, *_args) -> None:
            pass

        def memcpy_async(self, *_args) -> None:
            pass

    class Target:
        runner = SimpleNamespace(hidden_size=4)
        runtime = Runtime()

        def prefill(self, prompt, **kwargs):
            target_calls.append((tuple(prompt), kwargs))
            sink = kwargs["target_hidden_chunk_sink"]
            sink.consume(
                request_id=kwargs["target_hidden_request_id"],
                chunk_start=0,
                hidden_ptr=0x2000,
                rows=len(prompt),
                stream=0,
            )
            sink.finish(
                request_id=kwargs["target_hidden_request_id"],
                total_rows=len(prompt),
                stream=0,
            )
            return SimpleNamespace(token_id=91)

        def step(self, *_args, **_kwargs):
            raise AssertionError("bulk MTP admission must not serial-prefill the target")

    draft_calls: list[tuple[int, int, int, int]] = []
    finish_calls: list[tuple[int, int, bool]] = []

    class Executor:
        def enqueue_prompt_rows(self, request_id, token_ids, **kwargs):
            for index, token in enumerate(token_ids):
                draft_calls.append(
                    (
                        int(request_id),
                        int(token),
                        int(kwargs["position_start"]) + index,
                        int(kwargs["target_hidden_base_ptr"])
                        + index * int(kwargs["hidden_stride_bytes"]),
                    )
                )

        def finish_prompt_priming(self, request_id, *, stream, synchronize):
            finish_calls.append((int(request_id), int(stream), bool(synchronize)))

    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = Target()
    decoder.draft_provider = SimpleNamespace(executor=Executor())

    result = decoder._prefill_target_and_draft(
        (11, 22, 33),
        request_id=7,
        use_bulk=True,
    )

    assert result.token_id == 91
    assert len(target_calls) == 1
    prompt, kwargs = target_calls[0]
    assert prompt == (11, 22, 33)
    assert kwargs["use_bulk"] is True
    assert kwargs["return_logits"] is False
    assert kwargs["target_hidden_request_id"] == 7
    assert kwargs["target_hidden_chunk_sink"].total_rows == 3
    assert "capture_target_hidden_rows" not in kwargs
    assert draft_calls == [
        (7, 11, 0, 0x1000),
        (7, 22, 1, 0x2000),
        (7, 33, 2, 0x2008),
    ]
    assert finish_calls == [(7, 0, False)]
    assert freed == [0x1000]


def test_mtp_streaming_prompt_failure_drains_staging_and_frees_pending_row(
    monkeypatch,
) -> None:
    pending = DeviceBuffer(0x1000, 8)
    freed: list[int] = []
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )

    class Runtime:
        def memset(self, *_args) -> None:
            pass

        def memcpy_async(self, *_args) -> None:
            pass

    class Target:
        runner = SimpleNamespace(hidden_size=4)
        runtime = Runtime()

        def prefill(self, prompt, **kwargs):
            kwargs["target_hidden_chunk_sink"].consume(
                request_id=kwargs["target_hidden_request_id"],
                chunk_start=0,
                hidden_ptr=0x2000,
                rows=1,
                stream=0,
            )
            raise RuntimeError("injected target failure")

    finish_calls: list[tuple[int, int, bool]] = []

    class Executor:
        def enqueue_prompt_rows(self, *_args, **_kwargs) -> None:
            pass

        def finish_prompt_priming(self, request_id, *, stream, synchronize):
            finish_calls.append((int(request_id), int(stream), bool(synchronize)))

    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = Target()
    decoder.draft_provider = SimpleNamespace(executor=Executor())

    with pytest.raises(RuntimeError, match="injected target failure"):
        decoder._prefill_target_and_draft(
            (11, 22),
            request_id=7,
            use_bulk=True,
        )

    assert finish_calls == [(7, 0, True)]
    assert freed == [pending.ptr]


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 8, 9])
def test_streaming_prompt_sink_preserves_shifted_rows_across_chunk_partitions(
    monkeypatch,
    chunk_size: int,
) -> None:
    hidden_size = 4
    hidden_nbytes = hidden_size * DType.BF16.itemsize
    pending = DeviceBuffer(0x1000, hidden_nbytes)
    freed: list[int] = []
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )

    copies: list[tuple[int, int, int, int, int]] = []

    class Runtime:
        def memset(self, ptr, value, nbytes) -> None:
            copies.append((int(ptr), int(value), int(nbytes), -1, -1))

        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            copies.append((int(dst), int(src), int(nbytes), int(kind), int(stream)))

    rows: list[tuple[int, int, int, int]] = []

    class Executor:
        def enqueue_prompt_rows(
            self,
            request_id,
            token_ids,
            *,
            position_start,
            target_hidden_base_ptr,
            hidden_stride_bytes,
            stream,
        ) -> None:
            for index, token in enumerate(token_ids):
                rows.append(
                    (
                        int(request_id),
                        int(token),
                        int(position_start) + index,
                        int(target_hidden_base_ptr) + index * int(hidden_stride_bytes),
                    )
                )

    prompt = tuple(range(101, 124))
    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7,
        prompt_tokens=prompt,
        hidden_size=hidden_size,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=None,
    )
    source_base = 0x4000
    chunk_starts: set[int] = set()
    for start in range(0, len(prompt), chunk_size):
        count = min(chunk_size, len(prompt) - start)
        chunk_starts.add(start)
        sink.consume(
            request_id=7,
            chunk_start=start,
            hidden_ptr=source_base + start * hidden_nbytes,
            rows=count,
            stream=3,
        )
    sink.finish(request_id=7, total_rows=len(prompt), stream=3)

    assert [(token, position) for _, token, position, _ in rows] == [
        (token, position) for position, token in enumerate(prompt)
    ]
    for _, _, position, hidden_ptr in rows:
        if position in chunk_starts:
            assert hidden_ptr == pending.ptr
        else:
            assert hidden_ptr == source_base + (position - 1) * hidden_nbytes
    pending_copies = [copy for copy in copies if copy[3] == int(HipMemcpyKind.DEVICE_TO_DEVICE)]
    assert pending_copies == [
        (
            pending.ptr,
            source_base + (min(start + chunk_size, len(prompt)) - 1) * hidden_nbytes,
            hidden_nbytes,
            int(HipMemcpyKind.DEVICE_TO_DEVICE),
            3,
        )
        for start in range(0, len(prompt), chunk_size)
    ]
    assert sink.final_pending_hidden.ptr == pending.ptr
    sink.close()
    assert freed == [pending.ptr]


def test_streaming_prompt_sink_supports_warm_offset_and_fails_closed_on_owner_mismatch(
    monkeypatch,
) -> None:
    allocations = iter((DeviceBuffer(0x1000, 8),))
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: next(allocations))
    monkeypatch.setattr(mtp_module, "free", lambda _buffer, *, runtime: None)

    copies: list[tuple[int, int, int, int]] = []

    class Runtime:
        def memcpy(self, dst, src, nbytes, kind) -> None:
            copies.append((int(dst), int(src), int(nbytes), int(kind)))

        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            copies.append((int(dst), int(src), int(nbytes), int(kind)))

    rows: list[tuple[int, int, int, int]] = []

    class Executor:
        def enqueue_prompt_rows(self, request_id, token_ids, **kwargs) -> None:
            for index, token in enumerate(token_ids):
                rows.append(
                    (
                        int(request_id),
                        int(token),
                        int(kwargs["position_start"]) + index,
                        int(kwargs["target_hidden_base_ptr"])
                        + index * int(kwargs["hidden_stride_bytes"]),
                    )
                )

    initial = Tensor.from_handle(0x9000, (1, 4), DType.BF16, Device("hip", 0))
    sink = mtp_module._StreamingNextNPromptSink(
        request_id=11,
        prompt_tokens=(50, 51),
        hidden_size=4,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=None,
        start_position=100,
        initial_hidden=initial,
    )
    assert copies[0] == (0x1000, 0x9000, 8, int(HipMemcpyKind.DEVICE_TO_DEVICE))
    with pytest.raises(RuntimeError, match="owner"):
        sink.consume(
            request_id=12,
            chunk_start=0,
            hidden_ptr=0x5000,
            rows=1,
            stream=0,
        )
    sink.consume(
        request_id=11,
        chunk_start=0,
        hidden_ptr=0x5000,
        rows=1,
        stream=0,
    )
    with pytest.raises(RuntimeError, match="contiguous"):
        sink.consume(
            request_id=11,
            chunk_start=0,
            hidden_ptr=0x6000,
            rows=1,
            stream=0,
        )
    sink.consume(
        request_id=11,
        chunk_start=1,
        hidden_ptr=0x6000,
        rows=1,
        stream=0,
    )
    sink.finish(request_id=11, total_rows=2, stream=0)
    assert [(token, position) for _, token, position, _ in rows] == [(50, 100), (51, 101)]
    assert rows[0][3] == rows[1][3] == 0x1000
    sink.close()


def test_draft_hidden_policy_manifest_is_immutable_and_explicit() -> None:
    strict = mtp_module.Qwen35GGUFDraftHiddenPolicy()
    candidate = mtp_module.Qwen35GGUFDraftHiddenPolicy(
        target_hidden_variant="post_output_norm"
    )

    assert strict.target_hidden_variant == "pre_output_norm"
    assert strict.manifest()["prompt_target_hidden"] == "pre_output_norm"
    assert candidate.manifest() == {
        "target_hidden_variant": "post_output_norm",
        "prompt_target_hidden": "post_output_norm",
        "proposal_target_hidden": "post_output_norm",
        "target_commit_hidden": "pre_output_norm",
        "draft_chain_hidden": "nextn_post_output_norm",
    }
    with pytest.raises(ValueError, match="target_hidden_variant"):
        mtp_module.Qwen35GGUFDraftHiddenPolicy(target_hidden_variant="adaptive")
    with pytest.raises(Exception):
        candidate.target_hidden_variant = "pre_output_norm"


def test_post_output_norm_policy_normalizes_each_target_proposal_once(
    monkeypatch,
) -> None:
    calls: list[tuple[int, int, int, int, int]] = []
    syncs: list[str] = []
    target_hidden = Tensor.from_handle(0x1000, (1, 4), DType.BF16, Device("hip", 0))
    output_norm = SimpleNamespace(
        allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
    )
    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = SimpleNamespace(
        last_target_hidden=target_hidden,
        runner=SimpleNamespace(
            hidden_size=4,
            weights=SimpleNamespace(
                root=lambda name: output_norm,
                config=SimpleNamespace(rms_norm_eps=1.0e-6),
            ),
        ),
        runtime=SimpleNamespace(device_synchronize=lambda: syncs.append("sync")),
    )
    decoder.draft_hidden_policy = mtp_module.Qwen35GGUFDraftHiddenPolicy(
        target_hidden_variant="post_output_norm"
    )
    decoder._draft_post_norm_hidden = DeviceBuffer(0x3000, 8)

    monkeypatch.setattr(
        mtp_module,
        "gguf_rmsnorm_bf16_f32_weight",
        lambda src, weight, dst, **kwargs: calls.append(
            (
                int(src),
                int(weight),
                int(dst),
                int(kwargs["rows"]),
                int(kwargs["hidden_size"]),
            )
        ),
    )

    normalized = decoder._proposal_target_hidden()

    assert normalized.ptr == 0x3000
    assert calls == [(0x1000, 0x2000, 0x3000, 1, 4)]
    assert syncs == ["sync"]

    decoder.draft_hidden_policy = mtp_module.Qwen35GGUFDraftHiddenPolicy()
    strict = decoder._proposal_target_hidden()
    assert strict.ptr == 0x1000
    assert calls == [(0x1000, 0x2000, 0x3000, 1, 4)]


def test_post_output_norm_decoder_close_frees_owned_policy_row(
    monkeypatch,
) -> None:
    freed: list[int] = []
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )
    verifier_closes: list[str] = []
    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = SimpleNamespace(runtime=object())
    decoder._draft_post_norm_hidden = DeviceBuffer(0x3000, 8)
    decoder.owns_verifier = False
    decoder.verifier = SimpleNamespace(close=lambda: verifier_closes.append("close"))

    decoder.close()
    decoder.close()

    assert freed == [0x3000]
    assert verifier_closes == []


def test_streaming_prompt_sink_applies_one_consistent_target_hidden_transform(
    monkeypatch,
) -> None:
    pending = DeviceBuffer(0x1000, 8)
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(mtp_module, "free", lambda _buffer, *, runtime: None)
    copies: list[tuple[int, int, int]] = []

    class Runtime:
        def memset(self, *_args) -> None:
            pass

        def memcpy_async(self, dst, src, nbytes, _kind, _stream) -> None:
            copies.append((int(dst), int(src), int(nbytes)))

    rows: list[tuple[int, int, int]] = []

    class Executor:
        def enqueue_prompt_rows(self, _request_id, token_ids, **kwargs) -> None:
            for index, token in enumerate(token_ids):
                rows.append(
                    (
                        int(token),
                        int(kwargs["position_start"]) + index,
                        int(kwargs["target_hidden_base_ptr"])
                        + index * int(kwargs["hidden_stride_bytes"]),
                    )
                )

    transforms: list[tuple[int, int, int]] = []

    def transform(hidden_ptr: int, count: int, stream: int) -> int:
        transforms.append((int(hidden_ptr), int(count), int(stream)))
        return 0x7000

    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7,
        prompt_tokens=(11, 22, 33),
        hidden_size=4,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=None,
        transform_hidden_rows=transform,
    )
    sink.consume(
        request_id=7,
        chunk_start=0,
        hidden_ptr=0x5000,
        rows=3,
        stream=5,
    )
    sink.finish(request_id=7, total_rows=3, stream=5)

    assert transforms == [(0x5000, 3, 5)]
    assert rows == [(11, 0, 0x1000), (22, 1, 0x7000), (33, 2, 0x7008)]
    assert copies == [(0x1000, 0x7010, 8)]
    sink.close()


def test_streaming_prompt_sink_cancellation_between_chunks_releases_owned_row(
    monkeypatch,
) -> None:
    pending = DeviceBuffer(0x1000, 8)
    freed: list[int] = []
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )

    class Runtime:
        def memset(self, *_args) -> None:
            pass

        def memcpy_async(self, *_args) -> None:
            pass

    calls: list[tuple[int, ...]] = []

    class Executor:
        def enqueue_prompt_rows(self, _request_id, token_ids, **_kwargs) -> None:
            calls.append(tuple(int(token) for token in token_ids))

    checkpoints = 0

    def checkpoint() -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 2:
            raise GenerationCancelled()

    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7,
        prompt_tokens=(1, 2, 3, 4),
        hidden_size=4,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=checkpoint,
    )
    try:
        sink.consume(
            request_id=7,
            chunk_start=0,
            hidden_ptr=0x5000,
            rows=2,
            stream=0,
        )
        with pytest.raises(GenerationCancelled):
            sink.consume(
                request_id=7,
                chunk_start=2,
                hidden_ptr=0x6000,
                rows=2,
                stream=0,
            )
        assert sink.consumed_rows == 2
        assert calls == [(1,), (2,)]
    finally:
        sink.close()
    assert freed == [pending.ptr]


def test_streaming_prompt_sink_transfers_final_carried_row_ownership(
    monkeypatch,
) -> None:
    pending = DeviceBuffer(0x1000, 8)
    freed: list[int] = []
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: pending)
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )

    class Runtime:
        def memset(self, *_args) -> None:
            pass

        def memcpy_async(self, *_args) -> None:
            pass

    class Executor:
        def enqueue_prompt_rows(self, *_args, **_kwargs) -> None:
            pass

    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7,
        prompt_tokens=(11, 22),
        hidden_size=4,
        executor=Executor(),
        runtime=Runtime(),
        checkpoint=None,
    )
    sink.consume(
        request_id=7,
        chunk_start=0,
        hidden_ptr=0x5000,
        rows=2,
        stream=0,
    )
    sink.finish(request_id=7, total_rows=2, stream=0)

    transferred = sink.take_final_pending_buffer()
    sink.close()

    assert transferred is pending
    assert freed == []
    with pytest.raises(RuntimeError, match="already transferred"):
        sink.take_final_pending_buffer()


def test_mtp_prompt_admission_preserves_target_default_bulk_selector(monkeypatch) -> None:
    allocations = iter((DeviceBuffer(0x1000, 8), DeviceBuffer(0x2000, 8)))
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: next(allocations))
    monkeypatch.setattr(mtp_module, "free", lambda _buffer, *, runtime: None)
    target_calls: list[object] = []

    class Target:
        runner = SimpleNamespace(hidden_size=4)
        runtime = SimpleNamespace(memset=lambda *_args: None)

        def prefill(self, _prompt, **kwargs):
            target_calls.append(kwargs["use_bulk"])
            return SimpleNamespace(token_id=91)

    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = Target()
    decoder.draft_provider = SimpleNamespace(
        executor=SimpleNamespace(run_step=lambda *_args, **_kwargs: None)
    )

    decoder._prefill_target_and_draft((11,), request_id=7, use_bulk=None)

    assert target_calls == [None]


def test_mtp_deadline_checkpoint_raises_on_expiry() -> None:
    assert mtp_module._deadline_checkpoint(None) is None
    future = mtp_module._deadline_checkpoint(time.monotonic() + 60.0)
    assert future is not None
    future()
    past = mtp_module._deadline_checkpoint(time.monotonic() - 1.0)
    with pytest.raises(mtp_module.GenerationDeadlineExceeded) as excinfo:
        past()
    assert excinfo.value.deadline_at is not None


def test_mtp_custom_checkpoint_propagates_error() -> None:
    def boom() -> None:
        raise RuntimeError("injected fault")

    with pytest.raises(RuntimeError, match="injected fault"):
        mtp_module._mtp_cycle_checkpoint(boom)


def test_mtp_rollback_decision_never_reopens_committed_transaction() -> None:
    open_txn = SimpleNamespace(committed=False, rolled_back=False)
    assert mtp_module._mtp_should_rollback_transaction(
        target_committed=False,
        transaction=open_txn,
    ) is True
    assert mtp_module._mtp_should_rollback_transaction(
        target_committed=True,
        transaction=open_txn,
    ) is False
    assert mtp_module._mtp_should_rollback_transaction(
        target_committed=False,
        transaction=SimpleNamespace(committed=True, rolled_back=False),
    ) is False


def test_mtp_lifecycle_phase_is_stable_and_propagates_fault() -> None:
    seen: list[str] = []
    mtp_module._mtp_lifecycle_phase(seen.append, "after_target_prepare")
    assert seen == ["after_target_prepare"]

    def fault(phase: str) -> None:
        raise RuntimeError(f"injected:{phase}")

    with pytest.raises(RuntimeError, match="injected:after_target_commit"):
        mtp_module._mtp_lifecycle_phase(fault, "after_target_commit")


def test_mtp_generate_cancellation_precedes_proposal_mutation(monkeypatch) -> None:
    """A checkpoint that raises at the first cycle boundary must stop before
    any proposal/target work: no draft propose, no device proposal, no
    verifier prepare."""

    proposed: list[tuple[object, ...]] = []
    device_proposals: list[tuple[object, ...]] = []
    prepared: list[tuple[object, ...]] = []
    budget_events: list[tuple[object, ...]] = []

    class BudgetPolicy:
        def start_request(self, *, request_id, max_budget):
            budget_events.append(("start", int(request_id), int(max_budget)))

        def choose_budget(self, **kwargs):
            budget_events.append(("choose", kwargs))
            raise AssertionError("cancellation must prevent budget choice")

        def record_cycle(self, result):
            budget_events.append(("record", result))

        def summary(self):
            return {"kind": "test"}

    class FakeScheduler:
        def __init__(self, capacity: int = 1) -> None:
            self.completed: set[int] = set()
            self._rid = 7

        def submit(self, _prompt, *, max_new_tokens, request_id):
            self._rid = int(request_id)
            return self._rid

        def admit_pending(self) -> None:
            pass

        def next_prefill_work(self, chunk_size) -> None:
            pass

        def record_generated(self, _rows) -> None:
            pass

        def finish_request_at_stop(self, rid, *, eos_token_id, stop_token_ids) -> None:
            pass

        @property
        def active_batch(self):
            return SimpleNamespace(
                requests={self._rid: SimpleNamespace(remaining_decode=4)}
            )

    monkeypatch.setattr(mtp_module, "ResidentBatchScheduler", FakeScheduler)

    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.candidate_budget = 2
    decoder.target = SimpleNamespace(reset=lambda: None)
    decoder.draft_provider = SimpleNamespace(reset_request=lambda _rid: None)
    decoder.verifier = SimpleNamespace()

    def fake_prefill(_prompt, *, request_id, use_bulk, checkpoint=None):
        return SimpleNamespace(token_id=11)

    decoder._prefill_target_and_draft = fake_prefill

    def fake_register_policy(_rid):
        return None

    decoder._register_kv_policy = fake_register_policy

    def fake_propose(_context, *, candidate_budget, return_logits):
        proposed.append((int(candidate_budget), bool(return_logits)))
        raise AssertionError("cancellation must prevent draft proposal")

    decoder.draft_provider.propose = fake_propose

    def fake_device_proposal(*_args, **_kwargs):
        device_proposals.append((_args, _kwargs))
        raise AssertionError("cancellation must prevent device proposal launch")

    monkeypatch.setattr(mtp_module, "_maybe_launch_device_proposal", fake_device_proposal)

    def fake_prepare(*_args, **_kwargs):
        prepared.append((_args, _kwargs))
        raise AssertionError("cancellation must prevent target verify")

    decoder.verifier.prepare = fake_prepare

    def cancel_before_first_cycle() -> None:
        raise GenerationCancelled()

    with pytest.raises(GenerationCancelled):
        decoder.generate(
            (1, 2, 3),
            max_new_tokens=4,
            request_id=7,
            checkpoint=cancel_before_first_cycle,
            budget_policy=BudgetPolicy(),
        )

    assert budget_events == [("start", 7, 2)]
    assert proposed == []
    assert device_proposals == []
    assert prepared == []
