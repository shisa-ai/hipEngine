from __future__ import annotations

import pytest

from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.speculative import TargetVerifyBatch, compose_speculative_claims
from hipengine.speculative.generic import TreeDraftRequest, compile_tree_draft
from hipengine.speculative.registry import (
    SpeculativeProviderCapabilities,
    SpeculativeProviderKey,
    register_speculative_provider,
    resolve_speculative_provider,
)


def test_provider_capabilities_explicitly_advertise_chain_tree_and_ownership() -> None:
    attached = SpeculativeProviderCapabilities(
        provider_name="nextn",
        artifact_fingerprint="sha256:nextn",
        attachment_mode="model_attached",
        supported_modes=("verify_chain",),
        max_verifier_rows=8,
        transaction_mode="journal",
        provider_state_key="shared_target_hidden",
        provider_kv_key="shared_target_kv",
        fixed_transaction_units=(("spec.results", 2),),
        per_candidate_units=(("spec.rows", 1),),
        strict_fallback="target_ar",
    )
    independent = SpeculativeProviderCapabilities(
        provider_name="dflash",
        artifact_fingerprint="sha256:dflash",
        attachment_mode="independent",
        supported_modes=("verify_chain", "verify_tree"),
        max_verifier_rows=16,
        transaction_mode="journal",
        provider_state_key="dflash_state_bf16",
        provider_kv_key="dflash_kv_bf16",
        fixed_transaction_units=(("provider.state", 4),),
        per_candidate_units=(("provider.kv", 2), ("spec.rows", 1)),
        strict_fallback="target_ar",
    )

    assert attached.supports("verify_chain")
    assert attached.supports("verify_tree") is False
    with pytest.raises(NotImplementedError, match="verify_tree"):
        attached.require_mode("verify_tree")
    claims = independent.resource_claims(
        request_id=7, candidate_rows=5, claim_id="provider:7"
    )
    assert claims.units_by_pool() == {
        "provider.kv": 10,
        "provider.state": 4,
        "spec.rows": 5,
    }
    assert all(claim.lifetime is ClaimLifetime.TRANSACTION for claim in claims.claims)
    combined = compose_speculative_claims(
        "cycle:7",
        {
            "provider": claims,
            "target": ResourceClaimSet.from_mapping(
                "target:7", {"target.txn": 8}, request_id=7,
                lifetime=ClaimLifetime.TRANSACTION,
            ),
        },
    )
    assert combined.request_id == 7
    assert combined.units_by_pool()["provider.kv"] == 10


def test_generic_provider_registry_resolves_attached_and_independent_plugins() -> None:
    def attached_factory(**kwargs):
        return "attached", kwargs

    def independent_factory(**kwargs):
        return "independent", kwargs

    attached_key = SpeculativeProviderKey("nextn", "qwen38", "hip_gfx1100", "q4km")
    independent_key = SpeculativeProviderKey("dflash", "laguna", "hip_gfx1100", "q4km")
    register_speculative_provider(attached_key, attached_factory, replace=True)
    register_speculative_provider(independent_key, independent_factory, replace=True)

    assert resolve_speculative_provider(
        provider="nextn", target_model="qwen38", backend="hip_gfx1100", quant="q4km"
    ) is attached_factory
    assert resolve_speculative_provider(
        provider="dflash", target_model="laguna", backend="hip_gfx1100", quant="q4km"
    ) is independent_factory


def test_nonuniform_tree_compiler_preserves_request_parents_depths_and_target_rows() -> None:
    draft = compile_tree_draft(
        (
            TreeDraftRequest(
                request_id=10,
                root_position=100,
                candidate_tokens=(101, 102, 103),
                parent_candidate_ids=(-1, 0, 0),
            ),
            TreeDraftRequest(
                request_id=20,
                root_position=200,
                candidate_tokens=(201, 202),
                parent_candidate_ids=(-1, 0),
            ),
        ),
        max_verifier_rows=8,
        resident_slots={10: 2, 20: 5},
        cycle_id=3,
    )
    assert draft.mode == "verify_tree"
    assert draft.row_to_request == (10, 10, 10, 20, 20)
    assert draft.tree_parents == (-1, 0, 0, -1, 3)
    assert draft.draft_depths == (1, 2, 2, 1, 2)
    assert draft.candidate_ids == (0, 1, 2, 0, 1)
    assert draft.resident_slots == (2, 2, 2, 5, 5)

    target = TargetVerifyBatch.from_draft(
        draft, root_tokens=(99, 199), root_positions=(100, 200)
    )
    assert target.rows == 7
    assert target.row_to_request == (10, 20, 10, 10, 10, 20, 20)
    assert target.parent_rows == (-1, -1, 0, 2, 2, 1, 5)
    assert target.tree_shape == (0, 1, 1, 0, 4)
    accepted = target.accept_from_top1(
        (101, 201, 103, 999, 777, 202, 888),
        transaction_id=9,
    )
    assert accepted.accepted_counts == (2, 2)
    assert accepted.accepted_tokens == ((101, 103), (201, 202))
    assert accepted.selected_candidate_rows == (4, 6)
    assert accepted.next_tokens == (777, 888)


def test_tree_compiler_fails_closed_on_cross_request_or_unbounded_parent() -> None:
    with pytest.raises(ValueError, match="earlier candidate"):
        TreeDraftRequest(
            request_id=1,
            root_position=0,
            candidate_tokens=(10, 11),
            parent_candidate_ids=(-1, 2),
        )
    with pytest.raises(ValueError, match="max_verifier_rows"):
        compile_tree_draft(
            (TreeDraftRequest(1, 0, (10, 11, 12), (-1, 0, 1)),),
            max_verifier_rows=2,
            resident_slots={1: 0},
            cycle_id=1,
        )
