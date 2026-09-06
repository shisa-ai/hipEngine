"""Packet 2 CPU seam REDs: explicitly qualified physical C1 on the staged cycle.

A one-request group is a valid one-row provider group plus one R2/R3/R4 target
frontier under the same staged adapter/resource transaction as C>1. The legacy
AR-row singleton route (`Qwen35GGUFTransactionalVerifier` /
`_ensure_active_singleton_target_verifier`) stays authoritative for gfx1151,
Qwen3.6, and capacity-1 engines; these tests pin the gfx1100 physical-C1 route
without touching those uses.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.generation.qwen35_gguf_mtp2 as mtp2_module
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.speculative import (
    SpeculativeMTPStaticEligibility,
    SpeculativeMTPStaticState,
    SpeculativeRequestSemantics,
)
from hipengine.kernels.backends import backend_package_capability


def _owner(*, backend: str, capacity: int) -> SimpleNamespace:
    return SimpleNamespace(
        generator=SimpleNamespace(
            backend=backend,
            execution_profile="production",
        ),
        capacity=capacity,
        _shared_runner=None,
        _row=lambda rid: SimpleNamespace(
            native_greedy=True,
            first_token_emitted=True,
            lease=SimpleNamespace(
                session=SimpleNamespace(
                    runner=SimpleNamespace(fp16_recurrent_state=False),
                    _target_scratch_owner=SimpleNamespace(slot_count=capacity),
                    target_layout=SimpleNamespace(max_sequence_length=1024),
                    kv_storage_dtype="bf16",
                )
            ),
            slot=SimpleNamespace(),
        ),
    )


def _c1_eligibility(rid: int) -> SpeculativeMTPStaticEligibility:
    return SpeculativeMTPStaticEligibility(
        state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
        reason="qualified_test_physical_c1",
        max_candidate_count=3,
        max_realized_group_rows=1,
        automatic_eligible=False,
        strict_fallback_key="gguf_target_ar",
        evidence_key=f"test-physical-c1-{rid}",
        evidence_fingerprint=f"sha256:test-physical-c1-{rid}",
    )


def _wide_eligibility(rid: int, *, rows: int) -> SpeculativeMTPStaticEligibility:
    return SpeculativeMTPStaticEligibility(
        state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
        reason="qualified_test_physical_wide",
        max_candidate_count=3,
        max_realized_group_rows=rows,
        automatic_eligible=False,
        strict_fallback_key="gguf_target_ar",
        evidence_key=f"test-physical-wide-{rid}",
        evidence_fingerprint=f"sha256:test-physical-wide-{rid}",
    )


def _adapter(*, backend: str, capacity: int, rid: int, eligibility) -> Qwen35GGUFMTP2Adapter:
    adapter = Qwen35GGUFMTP2Adapter(
        _owner(backend=backend, capacity=capacity),
        enabled=True,
        target_verify_mode="native",
        candidate_budget=3,
    )
    adapter._intents = {rid: 3}
    adapter._static_eligibility_by_request = {rid: eligibility}
    adapter._prompt_hidden_rows = {rid: object()}
    adapter._states = {}
    adapter._disabled_requests = set()
    adapter._active_claims = None
    return adapter


def test_physical_c1_route_flag_is_package_owned() -> None:
    """gfx1100 owns the physical-C1 route; gfx1151 keeps the legacy singleton."""

    assert (
        backend_package_capability("hip_gfx1100", "GGUF_SPECDEC2_MTP2_PHYSICAL_C1", False)
        is True
    )
    assert (
        backend_package_capability("hip_gfx1151", "GGUF_SPECDEC2_MTP2_PHYSICAL_C1", False)
        is False
    )


def test_physical_c1_partition_admits_qualified_single_request() -> None:
    """A C1-qualified request resolves partition bound 1; others stay closed."""

    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    assert adapter.partition_max_requests((7,)) == 1


def test_partition_still_rejects_unqualified_singletons() -> None:
    """Legacy singletons stay K0; wide evidence keeps its existing width bound."""

    wide = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_wide_eligibility(7, rows=8),
    )
    # Pre-existing qualified behavior: a one-request due group with wide
    # evidence partitions at its evidence width and rides the packed route.
    assert wide.partition_max_requests((7,)) == 8

    legacy = _adapter(
        backend="hip_gfx1151",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    assert legacy.partition_max_requests((7,)) == 0


def test_physical_c1_claims_fit_admits_listed_one_row_cell() -> None:
    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=(7,),
            speculative_request_ids=(7,),
            candidate_counts=(3,),
        )
    ) is True
    # A K1 claim at a depth the policy does not list for C1 stays closed.
    assert adapter.claims_fit(
        SimpleNamespace(
            request_ids=(7,),
            speculative_request_ids=(7,),
            candidate_counts=(1,),
        )
    ) is False


def test_physical_c1_capability_returns_packed_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The C1 capability resolves without touching legacy singleton owners."""

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("physical C1 must not construct the legacy verifier")

    monkeypatch.setattr(
        mtp2_module, "Qwen35GGUFTransactionalVerifier", _forbidden
    )
    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    semantics = (
        SpeculativeRequestSemantics(7, "greedy", "verify_chain", 32, 25),
    )
    capability = adapter.capability(semantics)
    assert capability is not None
    assert capability.max_candidates_per_request == 3


def test_physical_c1_single_survivor_keeps_legacy_route_for_gfx1151(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gfx1151 rows==1 requests keep the legacy singleton verifier decision."""

    constructed: list[object] = []

    class FakeVerifier:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            constructed.append(self)

    monkeypatch.setattr(
        mtp2_module, "Qwen35GGUFTransactionalVerifier", FakeVerifier
    )
    adapter = _adapter(
        backend="hip_gfx1151",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    assert adapter._singleton_only(7) is True
    assert adapter._physical_c1_request(7) is False
    assert constructed == []


def test_physical_c1_state_open_routes_to_batch_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing C1 state opens the physical batch group, not the legacy open."""

    calls: list[tuple[str, tuple[int, ...]]] = []
    adapter = _adapter(
        backend="hip_gfx1100",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    monkeypatch.setattr(
        adapter,
        "_open_batch_requests",
        lambda ids: calls.append(("batch", tuple(int(v) for v in ids))),
    )
    monkeypatch.setattr(
        adapter,
        "_open_request",
        lambda rid: calls.append(("legacy", (int(rid),))),
    )
    adapter._ensure_request_states((7,))
    assert calls == [("batch", (7,))]

    legacy = _adapter(
        backend="hip_gfx1151",
        capacity=8,
        rid=7,
        eligibility=_c1_eligibility(7),
    )
    calls.clear()
    monkeypatch.setattr(
        legacy,
        "_open_batch_requests",
        lambda ids: calls.append(("batch", tuple(int(v) for v in ids))),
    )
    monkeypatch.setattr(
        legacy,
        "_open_request",
        lambda rid: calls.append(("legacy", (int(rid),))),
    )
    legacy._ensure_request_states((7,))
    assert calls == [("legacy", (7,))]
