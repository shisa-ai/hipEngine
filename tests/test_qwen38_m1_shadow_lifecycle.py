"""B3 M1 RED: request-owned physical-C2 shadow lifecycle for logical C1.

The measured M1 screen duplicated one C1 input in a C2 benchmark process. That
is only a performance bound: production needs an explicit owner for the second
target session, provider checkpoint, hidden row, KV state, and recurrent state.
The shadow participates in physical computation but can never publish output or
own the public request commit. Cancellation restores both provider checkpoints;
compaction moves physical slots without changing request ownership; teardown
reclaims every resource exactly once.

These tests are intentionally RED until ``C1ShadowSessionLifecycle`` is
implemented in the GGUF MTP2 adapter. The constructor is resource-only: no
prompt, token, or candidate value may decide whether padding exists.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

import hipengine.generation.qwen35_gguf_mtp2 as mtp2


@dataclass(frozen=True)
class _Resource:
    name: str


_SURFACES = (
    "target_session",
    "provider_checkpoint",
    "hidden_row",
    "kv_owner",
    "recurrent_owner",
)


def _api():
    missing = [
        name
        for name in ("C1ShadowOwnershipError", "C1ShadowSessionLifecycle")
        if not hasattr(mtp2, name)
    ]
    assert not missing, f"B3 M1 shadow lifecycle API missing: {missing}"
    return mtp2.C1ShadowOwnershipError, mtp2.C1ShadowSessionLifecycle


def _resources() -> dict[str, _Resource]:
    return {
        f"{lane}_{surface}": _Resource(f"{lane}_{surface}")
        for lane in ("real", "shadow")
        for surface in _SURFACES
    }


def _lease(*, resources=None, events=None):
    _error, lifecycle = _api()
    owned = _resources() if resources is None else resources
    recorded = [] if events is None else events

    def restore_checkpoint(lane, checkpoint):
        recorded.append(("restore", lane, checkpoint.name, None))

    def reclaim(surface, lane, resource, reason):
        recorded.append(("reclaim", lane, resource.name, reason))

    lease = lifecycle(
        request_id=42,
        real_slot=0,
        shadow_slot=1,
        target_session=owned["real_target_session"],
        shadow_target_session=owned["shadow_target_session"],
        provider_checkpoint=owned["real_provider_checkpoint"],
        shadow_provider_checkpoint=owned["shadow_provider_checkpoint"],
        hidden_row=owned["real_hidden_row"],
        shadow_hidden_row=owned["shadow_hidden_row"],
        kv_owner=owned["real_kv_owner"],
        shadow_kv_owner=owned["shadow_kv_owner"],
        recurrent_owner=owned["real_recurrent_owner"],
        shadow_recurrent_owner=owned["shadow_recurrent_owner"],
        restore_provider_checkpoint=restore_checkpoint,
        reclaim=reclaim,
    )
    return lease, owned, recorded


def test_shadow_lifecycle_api_exists_red() -> None:
    _api()


def test_shadow_owns_distinct_physical_resources_and_real_publication() -> None:
    lease, resources, _events = _lease()

    assert lease.request_id == 42
    assert lease.shadow_request_id == -43
    assert lease.physical_slots == (0, 1)
    assert lease.target_sessions == (
        resources["real_target_session"],
        resources["shadow_target_session"],
    )
    assert lease.provider_checkpoints == (
        resources["real_provider_checkpoint"],
        resources["shadow_provider_checkpoint"],
    )
    assert lease.hidden_rows == (
        resources["real_hidden_row"],
        resources["shadow_hidden_row"],
    )
    assert lease.kv_owners == (
        resources["real_kv_owner"],
        resources["shadow_kv_owner"],
    )
    assert lease.recurrent_owners == (
        resources["real_recurrent_owner"],
        resources["shadow_recurrent_owner"],
    )
    assert lease.compute_mask == (True, True)
    assert lease.publish_mask == (True, False)
    assert lease.request_commit_mask == (True, False)
    assert lease.closed is False


@pytest.mark.parametrize("surface", _SURFACES)
def test_shadow_rejects_aliased_resource_ownership(surface: str) -> None:
    error, _lifecycle = _api()
    resources = _resources()
    resources[f"shadow_{surface}"] = resources[f"real_{surface}"]

    with pytest.raises(error, match=surface):
        _lease(resources=resources)


def test_shadow_only_allows_real_request_to_publish_or_own_public_commit() -> None:
    error, _lifecycle = _api()
    lease, _resources_by_name, _events = _lease()

    lease.assert_publish_owner(42)
    lease.assert_request_commit_owner(42)
    with pytest.raises(error, match="shadow.*publish"):
        lease.assert_publish_owner(lease.shadow_request_id)
    with pytest.raises(error, match="shadow.*commit"):
        lease.assert_request_commit_owner(lease.shadow_request_id)


def test_shadow_cancellation_restores_and_reclaims_both_lanes_once() -> None:
    lease, _resources_by_name, events = _lease()

    lease.cancel(42)
    lease.cancel(42)

    assert lease.closed is True
    restore_events = [event for event in events if event[0] == "restore"]
    reclaim_events = [event for event in events if event[0] == "reclaim"]
    assert [(event[1], event[2]) for event in restore_events] == [
        ("real", "real_provider_checkpoint"),
        ("shadow", "shadow_provider_checkpoint"),
    ]
    assert len(reclaim_events) == 10
    assert {event[1] for event in reclaim_events} == {"real", "shadow"}
    assert {event[2] for event in reclaim_events} == {
        f"{lane}_{surface}"
        for lane in ("real", "shadow")
        for surface in _SURFACES
    }
    assert {event[3] for event in reclaim_events} == {"cancelled"}


def test_shadow_compaction_changes_slots_not_request_or_resource_ownership() -> None:
    error, _lifecycle = _api()
    lease, resources, events = _lease()
    owned_before = (
        lease.target_sessions,
        lease.provider_checkpoints,
        lease.hidden_rows,
        lease.kv_owners,
        lease.recurrent_owners,
    )

    lease.compact(real_slot=5, shadow_slot=2)

    assert lease.physical_slots == (5, 2)
    assert lease.request_id == 42
    assert lease.shadow_request_id == -43
    assert owned_before == (
        lease.target_sessions,
        lease.provider_checkpoints,
        lease.hidden_rows,
        lease.kv_owners,
        lease.recurrent_owners,
    )
    assert all(resource in resources.values() for pair in owned_before for resource in pair)
    assert events == []
    with pytest.raises(error, match="distinct"):
        lease.compact(real_slot=3, shadow_slot=3)


def test_shadow_teardown_reclaims_without_restoring_committed_checkpoints() -> None:
    lease, _resources_by_name, events = _lease()

    lease.close(reason="teardown")
    lease.close(reason="teardown")

    assert lease.closed is True
    assert not [event for event in events if event[0] == "restore"]
    reclaim_events = [event for event in events if event[0] == "reclaim"]
    assert len(reclaim_events) == 10
    assert {event[3] for event in reclaim_events} == {"teardown"}


def test_shadow_reclaim_failure_retries_without_duplicate_success() -> None:
    lease, _resources_by_name, events = _lease()
    original_reclaim = lease.reclaim
    failed = False

    def flaky_reclaim(surface, lane, resource, reason):
        nonlocal failed
        if resource.name == "shadow_kv_owner" and not failed:
            failed = True
            raise RuntimeError("injected reclaim failure")
        original_reclaim(surface, lane, resource, reason)

    lease.reclaim = flaky_reclaim
    with pytest.raises(RuntimeError, match="injected"):
        lease.close(reason="teardown")
    assert lease.closed is False

    lease.close(reason="teardown")

    assert lease.closed is True
    reclaim_events = [event for event in events if event[0] == "reclaim"]
    assert len(reclaim_events) == 10
    assert len({event[2] for event in reclaim_events}) == 10


def test_shadow_policy_constructor_has_no_prompt_token_or_candidate_inputs() -> None:
    _error, lifecycle = _api()
    parameters = inspect.signature(lifecycle).parameters

    assert not any(
        name.startswith(("prompt", "token", "candidate")) for name in parameters
    )
    assert all(
        parameter.kind
        not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
        for parameter in parameters.values()
    )
