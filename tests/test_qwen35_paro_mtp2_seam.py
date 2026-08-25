from __future__ import annotations

from types import SimpleNamespace

import hipengine.generation.qwen35_paro_mtp2 as paro_mtp2_module
from hipengine.generation.qwen35_paro import Qwen35ParoResidentModelRunner
from hipengine.generation.qwen35_paro_mtp2 import (
    Qwen35ParoMTP2Adapter,
    _ParoMTP2RequestState,
)
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

    def save_state(self, slot):
        return ("snapshot", slot)

    def reset(self):
        return None

    def advance_with_target_hidden(self, **kwargs):
        self.advances.append(kwargs)
        return self.current


def test_gfx1100_package_exposes_only_paro_c1_scope() -> None:
    assert backend_package_capability(
        "hip_gfx1100", "PARO_SPECDEC2_MTP2_C1", False
    ) is True
    assert backend_package_capability(
        "hip_gfx1100", "PARO_SPECDEC2_MTP2_C4", False
    ) is False
    assert backend_package_capability(
        "hip_gfx1151", "PARO_SPECDEC2_MTP2_C1", False
    ) is False


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


def test_paro_capability_is_c1_k1_and_profile_specific() -> None:
    row = SimpleNamespace(
        model_slot=0,
        native_greedy=True,
        first_token_emitted=True,
    )
    session = SimpleNamespace(max_sequence_length=4096)
    owner = SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1100",
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
    assert "fast_d64_candidate" in capability.capability_key
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
    assert 7 in adapter._states


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

    assert [row["input_token"] for row in proposer.advances] == [11, 12, 99]
    assert [row["position"] for row in proposer.advances] == [1, 2, 3]
    assert adapter._states[7].prompt_rows_consumed == 3
    assert adapter._states[7].prompt_prime_seconds > 0.0
    assert row.mtp2_prompt_prime_ms > 0.0


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


def test_paro_proposal_emits_one_bounded_host_candidate() -> None:
    proposer = _ProposerDouble()
    row = SimpleNamespace()
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
    assert graph.candidate_tokens == (77,)
    assert graph.provider_metadata == (("candidate_handoff", "bounded_host_i32"),)
    assert adapter._states[7].checkpoint == ("snapshot", 0)
