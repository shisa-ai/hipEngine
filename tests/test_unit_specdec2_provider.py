from __future__ import annotations

from dataclasses import replace

import pytest

from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.speculative import (
    ProviderAttachment,
    ProviderCatchupMode,
    SpecTransactionMode,
    SpeculativeCapability,
    SpeculativeRequestSemantics,
    StagedSpeculativeRunner,
    StagedTargetAdapter,
    resolve_staged_speculative_runner,
    validate_staged_speculative_runner,
    validate_staged_target_adapter,
)


def _capability() -> SpeculativeCapability:
    return SpeculativeCapability(
        capability_key="test:mtp2:model:backend:quant",
        target_key="model",
        provider_key="test_mtp2",
        method_key="mtp2",
        policy_fingerprint="policy:v1",
        execution_profile="strict",
        kv_backend_key="paged_bf16",
        attachment=ProviderAttachment.TARGET_ATTACHED,
        catchup_mode=ProviderCatchupMode.TARGET_OUTPUT,
        supported_modes=("verify_chain",),
        supported_sampling_modes=("greedy",),
        max_requests=4,
        max_candidates_per_request=3,
        max_frontier_rows=16,
        proposal_widths=(1, 2, 4),
        target_row_buckets=(2, 4, 8, 16),
        target_transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
        provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
        graph_supported=True,
        eager_supported=True,
        strict_fallback_key="target_ar_strict",
    )


def test_request_semantics_are_typed_and_pre_mutation() -> None:
    semantics = SpeculativeRequestSemantics(
        request_id=7,
        sampling_mode="greedy",
        mode="verify_chain",
        context_tokens=128,
        remaining_decode=8,
        grammar_key=None,
        stop_policy_key="token_eos_length",
    )

    assert semantics.request_id == 7
    assert semantics.context_tokens == 128
    with pytest.raises(ValueError, match="remaining_decode"):
        replace(semantics, remaining_decode=-1)
    with pytest.raises(ValueError, match="mode"):
        replace(semantics, mode="decode")


class _StagedRunner:
    """A complete duck-typed scheduler-facing staged lifecycle owner."""

    def speculative_capability(self, request_semantics):
        return _capability()

    def speculative_component_claims(self, plan):
        return {"provider": ResourceClaimSet.from_mapping(
            f"{plan.operation_id}:provider", {"provider.kv": 1}
        )}

    def reserve_speculative_claims(self, claims):
        return claims.claim_id

    def release_speculative_claims(self, reservation):
        return None

    def prepare_speculative_requests(self, plan, request_semantics, *, stream=None):
        return None

    def propose_speculative_batch(self, plan, request_semantics, *, stream=None):
        return None

    def execute_target_frontier(
        self, plan, frontier, complete_claims, *, commit, cancelled_request_ids
    ):
        return None


class _IncompleteRunner:
    """Exposes only the cold-plan hook, not the staged-frontier lifecycle."""

    def speculative_capability(self, request_semantics):
        return None


class _OptInIncompleteRunner:
    def staged_speculative_owner(self):
        return _IncompleteRunner()


def test_staged_runner_protocol_rejects_incomplete_owners() -> None:
    runner = _StagedRunner()

    assert isinstance(runner, StagedSpeculativeRunner)
    assert validate_staged_speculative_runner(runner) is runner
    with pytest.raises(TypeError, match="missing staged methods"):
        validate_staged_speculative_runner(_IncompleteRunner())


def test_staged_runner_resolver_prefers_explicit_opt_in_and_fails_closed() -> None:
    # An explicit opt-in that cannot own a bounded cycle is rejected, not
    # silently demoted to autoregressive decoding.
    with pytest.raises(
        TypeError, match="missing staged methods: speculative_component_claims"
    ):
        resolve_staged_speculative_runner(_OptInIncompleteRunner())
    # A complete owner resolves by duck typing without an accessor.
    runner = _StagedRunner()
    assert resolve_staged_speculative_runner(runner) is runner
    # A target-only runner with no staged methods is not a staged owner.
    assert resolve_staged_speculative_runner(object()) is None


# ------------------------------------------------------- staged adapter face


class _AdapterDouble:
    """The complete staged adapter face; no optional hooks."""

    def capability(self, request_semantics):
        return _capability()

    def claims_fit(self, plan):
        return True

    def component_claims(self, plan):
        return {}

    def reserve_claims(self, claims):
        return object()

    def release_claims(self, reservation):
        return None

    def prepare_requests(self, plan, request_semantics, *, stream=None):
        return None

    def propose_batch(self, plan, request_semantics, *, stream=None):
        raise AssertionError("not driven in this test")

    def execute_target_frontier(
        self, plan, frontier, complete_claims, *, commit, cancelled_request_ids
    ):
        raise AssertionError("not driven in this test")

    def rollback_cycle(self, plan, *args):
        return None

    def recover_cycle_failure(self, plan, error):
        return False

    def prepare_k0(self, plan, request_semantics, *, stream=None):
        return None

    def register_request(self, request_id, candidate_budget):
        return None

    def close(self):
        return None


def test_staged_adapter_validator_accepts_the_complete_face_and_keeps_hooks_optional() -> None:
    adapter = _AdapterDouble()

    assert isinstance(adapter, StagedTargetAdapter)
    assert validate_staged_target_adapter(adapter) is adapter

    # Optional hooks are absent on the double and the adapter still validates,
    # exactly as an adapter without execute_cycle / kv_live_spans_owner /
    # staged_frontier / note_speculative_ar_commit / release_request keeps
    # working through the runner seam.
    for hook in (
        "execute_cycle",
        "kv_live_spans_owner",
        "staged_frontier",
        "note_speculative_ar_commit",
        "release_request",
        "close_requests",
    ):
        assert not callable(getattr(adapter, hook, None))


def test_staged_adapter_validator_rejects_each_missing_lifecycle_method() -> None:
    for name in (
        "capability",
        "claims_fit",
        "component_claims",
        "reserve_claims",
        "release_claims",
        "prepare_requests",
        "propose_batch",
        "execute_target_frontier",
        "rollback_cycle",
        "recover_cycle_failure",
        "prepare_k0",
        "register_request",
        "close",
    ):
        broken = _AdapterDouble()
        setattr(broken, name, None)
        with pytest.raises(TypeError, match=f"missing staged lifecycle methods: {name}"):
            validate_staged_target_adapter(broken)


def test_every_production_adapter_class_declares_the_complete_staged_face() -> None:
    """The three production adapters conform without needing a device.

    Construction-time validation (``validate_staged_target_adapter(self)`` at
    each adapter ``__init__`` tail) fails loudly on a GPU host; this pins the
    same conformance CPU-only by checking the class-level method surface.
    """

    from hipengine.generation.qwen35_gguf_moe_mtp2 import Qwen35GGUFMoEMTP2Adapter
    from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
    from hipengine.generation.qwen35_paro_mtp2 import Qwen35ParoMTP2Adapter

    for adapter_cls in (
        Qwen35GGUFMTP2Adapter,
        Qwen35GGUFMoEMTP2Adapter,
        Qwen35ParoMTP2Adapter,
    ):
        for name in (
            "capability",
            "claims_fit",
            "component_claims",
            "reserve_claims",
            "release_claims",
            "prepare_requests",
            "propose_batch",
            "execute_target_frontier",
            "rollback_cycle",
            "recover_cycle_failure",
            "prepare_k0",
            "register_request",
            "close",
        ):
            method = getattr(adapter_cls, name, None)
            assert callable(method), f"{adapter_cls.__name__}.{name}"
