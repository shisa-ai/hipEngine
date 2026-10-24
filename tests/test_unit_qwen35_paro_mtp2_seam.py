from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.generation.qwen35_paro_mtp2 as paro_mtp2_module
from hipengine.generation.qwen35_paro import Qwen35ParoResidentModelRunner
from hipengine.generation.qwen35_paro_mtp2 import (
    Qwen35ParoMTP2Adapter,
    _ParoMTP2RequestState,
)
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kernels.backends import backend_package_capability
from hipengine.speculative import SpeculativeRequestSemantics


class _AdapterDouble:
    def __init__(self) -> None:
        self.calls = []

    def register_request(self, request_id, candidate_budget):
        self.calls.append(("register", request_id, candidate_budget))

    def capability(self, semantics):
        self.calls.append(("capability", tuple(semantics)))
        return "capability"

    def claims_fit(self, plan):
        return plan == "plan"

    def component_claims(self, plan):
        return {"plan": plan}

    def reserve_claims(self, claims):
        return ("reservation", claims)

    def release_claims(self, reservation):
        self.calls.append(("release", reservation))

    def prepare_requests(self, plan, semantics, *, stream=None):
        self.calls.append(("prepare", plan, tuple(semantics), stream))

    def propose_batch(self, plan, semantics, *, stream=None):
        return ("proposal", plan, tuple(semantics), stream)

    def execute_target_frontier(self, *args, **kwargs):
        return (args, kwargs)

    def rollback_cycle(self, *args):
        self.calls.append(("rollback", args))

    def recover_cycle_failure(self, plan, error):
        self.calls.append(("recover", plan, error))
        return True

    def kv_live_spans_owner(self, plan):
        return f"owner:{plan.operation_id}"


class _ProposerDouble:
    current = SimpleNamespace(token=77)
    cache_len = 12
    closed = False

    def __init__(self) -> None:
        self.advances = []
        self.prompt_kv = []

    def append_prompt_kv(self, **kwargs):
        self.prompt_kv.append(kwargs)

    def save_state(self, slot):
        return ("snapshot", slot)

    def restore_state(self, snapshot):
        self.restored = snapshot

    def reset(self):
        return None

    def advance_with_target_hidden(self, **kwargs):
        self.advances.append(kwargs)
        return self.current

    def device_candidate_token_ids(self):
        return Tensor.from_handle(0x7000, (1,), DType.INT32, Device("hip", 0))


def test_gfx11_packages_expose_only_paro_c1_scope() -> None:
    assert backend_package_capability(
        "hip_gfx1100", "PARO_SPECDEC2_MTP2_C1", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1100", "PARO_SPECDEC2_MTP2_C4", False
    ) is False
    assert backend_package_capability(
        "hip_gfx1151", "PARO_SPECDEC2_MTP2_C1", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1151", "PARO_SPECDEC2_MTP2_C4", False
    ) is False


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_paro_resident_runner_resolves_singleton_adapter(backend) -> None:
    runner = object.__new__(Qwen35ParoResidentModelRunner)
    runner.generator = SimpleNamespace(backend=backend)
    runner.capacity = 1
    runner._mtp2_adapter = None
    runner._mtp2_adapter_resolved = False
    assert isinstance(runner._resolved_mtp2_adapter(), Qwen35ParoMTP2Adapter)


def test_missing_adapter_refuses_before_marking_prompt_for_capture() -> None:
    runner = object.__new__(Qwen35ParoResidentModelRunner)
    runner._mtp2_adapter = None
    runner._mtp2_adapter_resolved = True
    runner._rows = {7: SimpleNamespace(mtp2_candidate_budget=0)}
    with pytest.raises(NotImplementedError, match="PARO MTP2 adapter is unavailable"):
        runner.register_speculative_request(7, 1)
    assert runner._rows[7].mtp2_candidate_budget == 0


def test_invalid_budget_does_not_mark_prompt_for_capture() -> None:
    runner = object.__new__(Qwen35ParoResidentModelRunner)
    runner.generator = SimpleNamespace()
    runner._mtp2_adapter = Qwen35ParoMTP2Adapter(runner)
    runner._rows = {7: SimpleNamespace(mtp2_candidate_budget=0)}
    with pytest.raises(ValueError, match="candidate budget must be positive"):
        runner.register_speculative_request(7, 0)
    assert runner._rows[7].mtp2_candidate_budget == 0


def test_paro_resident_runner_delegates_staged_methods() -> None:
    runner = object.__new__(Qwen35ParoResidentModelRunner)
    adapter = _AdapterDouble()
    runner._mtp2_adapter = adapter
    runner._mtp2_adapter_resolved = True
    runner._rows = {7: SimpleNamespace(mtp2_candidate_budget=0)}

    runner.register_speculative_request(7, 3)
    assert runner._rows[7].mtp2_candidate_budget == 1
    assert adapter.calls == [("register", 7, 3)]
    assert runner.speculative_capability(("semantics",)) == "capability"
    assert runner.speculative_claims_fit("plan") is True
    assert runner.speculative_component_claims("plan") == {"plan": "plan"}
    reservation = runner.reserve_speculative_claims("claims")
    assert reservation == ("reservation", "claims")
    runner.release_speculative_claims(reservation)
    runner.prepare_speculative_requests("plan", ("s",), stream=9)
    assert runner.propose_speculative_batch("plan", ("s",), stream=4) == (
        "proposal",
        "plan",
        ("s",),
        4,
    )
    assert runner.speculative_kv_live_spans_owner(
        SimpleNamespace(operation_id="op")
    ) == "owner:op"


@pytest.mark.parametrize("backend,variant", [
    ("hip_gfx1100", "fast_d64_candidate"),
    ("hip_gfx1151", "fast_d64_candidate"),
])
def test_paro_capability_is_c1_k1_and_profile_specific(backend, variant) -> None:
    row = SimpleNamespace(
        model_slot=0,
        native_greedy=True,
        first_token_emitted=True,
    )
    session = SimpleNamespace(max_sequence_length=4096)
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            backend=backend,
            execution_profile="production",
        ),
        _row=lambda request_id: row,
        _session=session,
    )
    adapter = Qwen35ParoMTP2Adapter(owner)
    adapter._intents[7] = 1
    adapter._states[7] = _ParoMTP2RequestState(7, _ProposerDouble())
    semantics = (
        SpeculativeRequestSemantics(
            request_id=7,
            sampling_mode="greedy",
            mode="verify_chain",
            context_tokens=128,
            remaining_decode=8,
        ),
    )

    capability = adapter.capability(semantics)

    assert capability is not None
    assert capability.max_requests == 1
    assert capability.max_candidates_per_request == 1
    assert capability.max_frontier_rows == 2
    assert capability.proposal_widths == (1,)
    assert capability.target_row_buckets == (2,)
    assert capability.execution_profile == "production"
    assert variant in capability.capability_key
    assert capability.strict_fallback_key == "paro_target_c1_loop_exact"

    owner.generator.execution_profile = "strict"
    strict = adapter.capability(semantics)
    assert strict is not None
    assert "strict_exact" in strict.capability_key

    row.model_slot = 1
    assert adapter.capability(semantics) is None


def test_provider_open_timing_covers_proposer_construction(monkeypatch) -> None:
    class FakeProposer:
        closed = False

        def __init__(self, _model, **kwargs) -> None:
            self.max_positions = kwargs["max_positions"]
            self.max_mtp_tokens = kwargs["max_mtp_tokens"]
            self.scoring_head = kwargs["scoring_head"]

        def reset(self) -> None:
            return None

    ticks = iter((10.0, 10.25))
    monkeypatch.setattr(paro_mtp2_module, "NativeMtpChainProposer", FakeProposer)
    monkeypatch.setattr(paro_mtp2_module.time, "perf_counter", lambda: next(ticks))
    session = SimpleNamespace(
        max_sequence_length=4096,
        lm_head_weight=SimpleNamespace(tensor=SimpleNamespace(ptr=0x1000)),
        lm_head_scale=SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000)),
        vocab_size=248320,
        lm_head_threads=256,
        runtime=object(),
        compiler_version=None,
    )
    row = SimpleNamespace(
        prompt_ids=(10, 11, 12),
        request=SimpleNamespace(max_tokens=8),
        mtp2_provider_open_ms=0.0,
    )
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1100",
            model_path="/tmp/model",
        ),
        _row=lambda request_id: row,
        _session=session,
    )
    adapter = Qwen35ParoMTP2Adapter(owner)
    adapter._intents[7] = 1

    adapter.begin_prompt(7)

    assert row.mtp2_provider_open_ms == 250.0
    assert adapter._proposer_builds == 1
    assert adapter._states[7].proposer.max_mtp_tokens == 256
    assert 7 in adapter._states


def test_provider_capacity_bucket_grows_without_prompt_content() -> None:
    assert paro_mtp2_module._proposer_capacity_bucket(1) == 256
    assert paro_mtp2_module._proposer_capacity_bucket(255) == 256
    assert paro_mtp2_module._proposer_capacity_bucket(256) == 256
    assert paro_mtp2_module._proposer_capacity_bucket(257) == 512


def test_streaming_prompt_priming_uses_shifted_tokens_and_final_root() -> None:
    proposer = _ProposerDouble()
    row = SimpleNamespace(prompt_ids=(10, 11, 12), mtp2_prompt_prime_ms=0.0)
    owner = SimpleNamespace(
        generator=SimpleNamespace(backend="hip_gfx1100"),
        _row=lambda request_id: row,
    )
    adapter = Qwen35ParoMTP2Adapter(owner)
    adapter._states[7] = _ParoMTP2RequestState(7, proposer)

    adapter.consume_prompt_row(7, prompt_index=0, target_hidden_ptr=0x1000, seed_token=None)
    adapter.consume_prompt_row(7, prompt_index=1, target_hidden_ptr=0x2000, seed_token=None)
    adapter.consume_prompt_row(7, prompt_index=2, target_hidden_ptr=0x3000, seed_token=99)

    priming = proposer.prompt_kv + proposer.advances
    assert [row["input_token"] for row in priming] == [11, 12, 99]
    assert [row["position"] for row in priming] == [1, 2, 3]
    assert [row["target_hidden_ptr"] for row in priming] == [0x1000, 0x2000, 0x3000]
    assert len(proposer.prompt_kv) == 2
    assert len(proposer.advances) == 1
    assert proposer.advances[0]["need_result"] is True
    assert proposer.advances[0]["read_token_id"] is False
    assert adapter._states[7].prompt_rows_consumed == 3
    assert adapter._states[7].prompt_prime_seconds > 0.0
    assert row.mtp2_prompt_prime_ms > 0.0


def test_cache_only_prompt_failure_does_not_publish_a_consumed_row() -> None:
    proposer = _ProposerDouble()
    row = SimpleNamespace(prompt_ids=(10, 11), mtp2_prompt_prime_ms=0.0)
    owner = SimpleNamespace(
        generator=SimpleNamespace(backend="hip_gfx1100"),
        _row=lambda request_id: row,
    )
    adapter = Qwen35ParoMTP2Adapter(owner)
    state = _ParoMTP2RequestState(7, proposer)
    adapter._states[7] = state

    def fail(**kwargs):
        raise RuntimeError("injected KV append failure")

    proposer.append_prompt_kv = fail
    with pytest.raises(RuntimeError, match="injected KV append failure"):
        adapter.consume_prompt_row(7, prompt_index=0, target_hidden_ptr=0x1000, seed_token=None)
    assert state.prompt_rows_consumed == 0
    assert state.prompt_prime_seconds == row.mtp2_prompt_prime_ms == 0.0
    assert not proposer.advances


@pytest.mark.parametrize(
    ("final_chunk", "prompt_total", "sample", "seed_token"),
    (
        (True, 2, True, 99),
        (False, 5, False, None),
    ),
)
def test_staged_prefill_uses_packed_target_and_streams_final_hidden_rows(
    final_chunk,
    prompt_total,
    sample,
    seed_token,
) -> None:
    """The packed path hands the provider one chunk, not one row at a time.

    The target owns hidden normalization and the K/V append; this seam owns only
    which rows the provider sees and whether the packed sample reaches it. A
    request's rows of one slab are one contiguous span, so they arrive as a
    single ``PromptPrimeSink.consume`` carrying the span's first prompt row and
    the slab's base pointer, and the sampled root is delivered only on the chunk
    that ends the prompt.
    """

    calls = []
    packed_result = SimpleNamespace(token_id=99)

    class FakeSink:
        request_id = 7
        hidden_size = 8
        total_rows = prompt_total

        def consume(self, *, request_id, chunk_start, hidden_ptr, rows, stream):
            calls.append(("consume", request_id, chunk_start, hidden_ptr, rows, stream))

        def finish(self, *, request_id, total_rows, stream):
            calls.append(("finish", request_id, total_rows, stream))

    sink = FakeSink()
    adapter = SimpleNamespace(
        begin_prompt=lambda request_id: calls.append(("begin", request_id)),
        prompt_capture_sink=lambda request_id: calls.append(("sink", request_id)) or sink,
        finish_prompt_capture=lambda request_id, *, seed_token: calls.append(
            ("finish_capture", request_id, seed_token)
        ),
    )

    class FakeSession:
        block_size = 256
        hidden_nbytes = 16
        config = SimpleNamespace(hidden_size=8)
        runtime = object()

        def prefill_native_packed(
            self,
            slab,
            *,
            sample=True,
            final_hidden_row_sink=None,
            final_hidden_chunk_sink=None,
        ):
            calls.append(("packed", slab.token_rows, sample))
            # The chunk seam replaces the per-row seam for this route.
            assert final_hidden_row_sink is None
            assert final_hidden_chunk_sink is sink
            # Rows 0 and 1 of the slab are one contiguous span: prompt index 0,
            # base pointer of the post-output-norm rows, two rows, one stream.
            final_hidden_chunk_sink.consume(
                request_id=7,
                chunk_start=0,
                hidden_ptr=0x9000,
                rows=2,
                stream=0,
            )
            if 0 + 2 == final_hidden_chunk_sink.total_rows:
                final_hidden_chunk_sink.finish(
                    request_id=7,
                    total_rows=final_hidden_chunk_sink.total_rows,
                    stream=0,
                )
            return (packed_result,) if sample else (None,)

    runner = object.__new__(Qwen35ParoResidentModelRunner)
    runner._session = FakeSession()
    runner._resolved_mtp2_adapter = lambda: adapter
    runner._clear_session_sampler = lambda: None
    runner._configure_sampled_row = lambda row: None
    runner._fallback_reasons = {"packed_prefill_unavailable": 0}
    runner._route_counts = {"prefill_chunks": 0}
    row = SimpleNamespace(
        request_id=7,
        model_slot=0,
        prompt_ids=(10, 11),
        native_greedy=True,
        mtp2_candidate_budget=1,
        mtp2_prompt_hidden_buffer=None,
        native_prefill=False,
    )

    result = runner._prefill_row_chunk(
        row,
        (10, 11),
        start_position=0,
        final_chunk=final_chunk,
    )

    assert row.native_prefill is True
    assert result is (packed_result if sample else None)
    assert calls == [
        ("begin", 7),
        ("sink", 7),
        ("packed", ((10, 11),), sample),
        ("consume", 7, 0, 0x9000, 2, 0),
        *([("finish", 7, 2, 0)] if final_chunk else []),
        ("finish_capture", 7, seed_token),
    ]


@pytest.mark.parametrize("backend,mode", [
    ("hip_gfx1100", "decode_batched"),
    ("hip_gfx1151", "decode_batched"),
])
def test_prefill_completion_reserves_c1_verify_scratch_before_first_cycle(backend, mode) -> None:
    proposer = _ProposerDouble()
    row = SimpleNamespace(
        prompt_ids=(10, 11),
        request=SimpleNamespace(max_tokens=8),
        mtp2_prompt_hidden_buffer=object(),
    )
    calls = []
    owner = SimpleNamespace(
        generator=SimpleNamespace(backend=backend),
        _row=lambda request_id: row,
        _session=SimpleNamespace(
            prepare_specdec2_verify_scratch=lambda *, rows, chain_attn_mode, max_context_tokens: calls.append(
                ("reserve", rows, chain_attn_mode, max_context_tokens)
            )
        ),
        _release_mtp2_prompt_capture=lambda selected: calls.append(("release", selected)),
    )
    adapter = Qwen35ParoMTP2Adapter(owner)
    adapter._states[7] = _ParoMTP2RequestState(
        7,
        proposer,
        prompt_rows_consumed=2,
    )

    adapter.observe_prefill_result(7)

    assert calls == [("reserve", 2, mode, 10), ("release", row)]

    owner.generator.execution_profile = "strict"
    adapter.observe_prefill_result(7)
    assert calls[-2:] == [("reserve", 2, "c1_loop", 10), ("release", row)]


def test_staged_target_commit_defers_host_sync_to_stream_order() -> None:
    calls = []
    session = SimpleNamespace(
        verify_chain_bulk_and_commit=lambda batch, **kwargs: calls.append(kwargs)
        or "verify"
    )

    result = Qwen35ParoMTP2Adapter._verify_target(
        session,
        "batch",
        chain_attn_mode="decode_batched",
        candidate_token_ids_i32=Tensor.from_handle(
            0x7000, (1,), DType.INT32, Device("hip", 0)
        ),
    )

    assert result == "verify"
    assert calls[0]["synchronize_after_commit"] is False
    assert calls[0]["candidate_token_ids_i32"].ptr == 0x7000


def test_initial_root_k0_keeps_primed_provider_live() -> None:
    proposer = _ProposerDouble()
    row = SimpleNamespace(first_token_emitted=False)
    owner = SimpleNamespace(
        generator=SimpleNamespace(backend="hip_gfx1100"),
        _row=lambda request_id: row,
    )
    adapter = Qwen35ParoMTP2Adapter(owner)
    adapter._states[7] = _ParoMTP2RequestState(7, proposer)
    plan = SimpleNamespace(request_ids=(7,))

    adapter.prepare_k0(plan, ())
    assert 7 in adapter._states
    assert 7 not in adapter._disabled_requests

    row.first_token_emitted = True
    proposer.close = lambda: None
    adapter.prepare_k0(plan, ())
    assert 7 not in adapter._states
    assert 7 in adapter._disabled_requests


def test_paro_proposal_emits_one_stable_device_candidate() -> None:
    proposer = _ProposerDouble()
    row = SimpleNamespace(mtp2_candidate_device_handoffs=0)
    owner = SimpleNamespace(
        generator=SimpleNamespace(backend="hip_gfx1100"),
        _row=lambda request_id: row,
    )
    adapter = Qwen35ParoMTP2Adapter(owner)
    adapter._states[7] = _ParoMTP2RequestState(7, proposer)
    plan = SimpleNamespace(
        request_ids=(7,),
        provider_key="qwen_paro_mtp_bf16",
        cycle_id=3,
        resident_slots=(0,),
    )
    semantics = (
        SpeculativeRequestSemantics(7, "greedy", "verify_chain", 128, 8),
    )

    graph = adapter.propose_batch(plan, semantics)

    assert graph.request_ids == (7,)
    assert graph.root_positions == (127,)
    assert graph.candidate_counts == (1,)
    assert graph.candidate_tokens == (0,)
    assert graph.token_ids is not None and graph.token_ids.ptr == 0x7000
    assert graph.provider_metadata == (("candidate_handoff", "device_i32"),)
    assert adapter._states[7].checkpoint == ("snapshot", 0)
    assert row.mtp2_candidate_device_handoffs == 1
