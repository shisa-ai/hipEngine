from __future__ import annotations

from dataclasses import replace

import pytest

from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.speculative import (
    CandidateGraph,
    ProviderAttachment,
    ProviderCatchupMode,
    SpecCycleResult,
    SpecCycleTransaction,
    SpecRequestPlan,
    SpecTransactionMode,
    SpeculativeCapability,
    SpeculativeProviderKey,
    SpeculativeRequestSemantics,
    StagedSpeculativeProvider,
    construct_staged_speculative_provider,
    register_staged_speculative_provider,
    registered_staged_speculative_providers,
    resolve_staged_speculative_provider,
    validate_staged_speculative_provider,
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


class _StagedProvider:
    provider_name = "test_mtp2"

    def capability(self, target_key, request_semantics):
        assert target_key == "model"
        assert request_semantics
        return _capability()

    def resource_claims(self, plan):
        return {
            "provider": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:provider",
                {"provider.kv": sum(plan.candidate_counts)},
                lifetime=ClaimLifetime.TRANSACTION,
            )
        }

    def prepare_requests(self, plan, request_semantics, *, stream=None):
        return None

    def propose_batch(self, plan, request_semantics, *, stream=None):
        return CandidateGraph(
            provider_key="test_mtp2",
            method_key="mtp2",
            policy_fingerprint="policy:v1",
            cycle_id=plan.cycle_id,
            transaction_id=1,
            request_ids=plan.request_ids,
            resident_slots=plan.resident_slots,
            root_positions=tuple(row.context_tokens - 1 for row in request_semantics),
            row_offsets=(0, 1),
            row_to_request=plan.request_ids,
            parent_candidate_rows=(-1,),
            draft_depths=(1,),
            active_mask=(True,),
            candidate_tokens=(77,),
        )

    def commit_batch(self, result: SpecCycleResult, *, stream=None):
        return None

    def rollback_batch(self, transaction: SpecCycleTransaction, *, stream=None):
        return None

    def close_requests(self, request_ids):
        return None


class _LegacyOnlyProvider:
    provider_name = "legacy"

    def generate_detailed(self, request):
        return []

    def stream_detailed(self, request):
        return iter(())

    def capabilities(self):
        return {}

    def close(self):
        return None


class _HybridProvider(_StagedProvider):
    def generate_detailed(self, request):
        return []


def _factory(**kwargs):
    assert kwargs == {"marker": "ok"}
    return _StagedProvider()


def _legacy_factory(**kwargs):
    return _LegacyOnlyProvider()


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


def test_staged_provider_protocol_rejects_legacy_and_hybrid_generation_owners() -> None:
    provider = _StagedProvider()

    assert isinstance(provider, StagedSpeculativeProvider)
    assert validate_staged_speculative_provider(provider) is provider
    with pytest.raises(TypeError, match="staged methods"):
        validate_staged_speculative_provider(_LegacyOnlyProvider())
    with pytest.raises(TypeError, match="whole-request generation"):
        validate_staged_speculative_provider(_HybridProvider())


def test_staged_registry_resolves_exact_four_axis_key_without_legacy_alias() -> None:
    key = SpeculativeProviderKey(
        provider="test_mtp2",
        target_model="model",
        backend="backend",
        quant="quant",
    )
    register_staged_speculative_provider(key, _factory, replace=True)

    assert key in registered_staged_speculative_providers()
    assert (
        resolve_staged_speculative_provider(
            provider="test_mtp2",
            target_model="model",
            backend="backend",
            quant="quant",
        )
        is _factory
    )
    with pytest.raises(KeyError, match="unregistered staged speculative provider"):
        resolve_staged_speculative_provider(
            provider="test_mtp2",
            target_model="model",
            backend="other_backend",
            quant="quant",
        )


def test_staged_registry_constructs_and_validates_provider() -> None:
    key = SpeculativeProviderKey(
        provider="constructed_mtp2",
        target_model="model",
        backend="backend",
        quant="quant",
    )
    register_staged_speculative_provider(key, _factory, replace=True)

    provider = construct_staged_speculative_provider(
        provider="constructed_mtp2",
        target_model="model",
        backend="backend",
        quant="quant",
        marker="ok",
    )
    assert isinstance(provider, _StagedProvider)

    legacy_key = SpeculativeProviderKey(
        provider="legacy_only",
        target_model="model",
        backend="backend",
        quant="quant",
    )
    register_staged_speculative_provider(legacy_key, _legacy_factory, replace=True)
    with pytest.raises(TypeError, match="staged methods"):
        construct_staged_speculative_provider(
            provider="legacy_only",
            target_model="model",
            backend="backend",
            quant="quant",
        )
