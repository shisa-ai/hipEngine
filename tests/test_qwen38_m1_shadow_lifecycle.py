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
from types import SimpleNamespace

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


def _adapter_resource_fixture(*, fail_second_checkpoint: bool = False):
    events = []
    resources = _resources()
    real_target = SimpleNamespace(
        name="real_target_session",
        scratch=resources["real_recurrent_owner"],
    )
    resources["real_target_session"] = real_target
    resources["shadow_target_session"] = SimpleNamespace(
        name="shadow_target_session"
    )
    row = type("Row", (), {})()
    row.lease = type("Lease", (), {"session": real_target})()
    row.slot = object()
    row.kv_allocation = resources["real_kv_owner"]
    shadow_bundle = {
        "target_session": resources["shadow_target_session"],
        "hidden_row": resources["shadow_hidden_row"],
        "kv_owner": resources["shadow_kv_owner"],
        "recurrent_owner": resources["shadow_recurrent_owner"],
    }

    def acquire(**kwargs):
        events.append(("acquire", kwargs))
        return shadow_bundle

    def reclaim(bundle, *, surface, lane, resource, reason):
        assert bundle is shadow_bundle
        events.append(("reclaim", surface, lane, resource.name, reason))

    def abort(bundle, *, reason):
        assert bundle is shadow_bundle
        events.append(("abort", reason))

    checkpoint_calls = 0

    def capture(request_id):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if fail_second_checkpoint and checkpoint_calls == 2:
            raise RuntimeError("injected checkpoint failure")
        checkpoint = _Resource(f"checkpoint_{request_id}")
        events.append(("capture", request_id, checkpoint.name))
        return checkpoint

    executor = type("Executor", (), {})()
    executor.capture_request_checkpoint = capture
    executor.restore_request_checkpoint = lambda checkpoint: events.append(
        ("restore", checkpoint.name)
    )
    executor.release_request_checkpoint = lambda checkpoint: events.append(
        ("release_checkpoint", checkpoint.name)
    )
    provider = type("Provider", (), {})()
    provider.executor = executor
    provider.reset_request = lambda request_id: events.append(("reset", request_id))
    provider.release_request = lambda request_id: events.append(
        ("release_request", request_id)
    )
    state = mtp2._MTP2RequestState(
        request_id=42,
        provider=provider,
        provider_pool_key=None,
        provider_group_key=(42,),
        verifier=None,
        root_hidden_buffer=resources["real_hidden_row"],
    )
    owner = type("Owner", (), {})()
    owner._row = lambda request_id: row
    owner.acquire_c1_shadow_resources = acquire
    owner.reclaim_c1_shadow_resource = reclaim
    owner.abort_c1_shadow_resources = abort
    adapter = object.__new__(mtp2.Qwen35GGUFMTP2Adapter)
    adapter.owner = owner
    adapter._states = {42: state}
    adapter._c1_shadow_states = {}
    return adapter, resources, events


def test_adapter_acquires_and_drops_concrete_shadow_resources() -> None:
    adapter, resources, events = _adapter_resource_fixture()

    lifecycle = adapter.acquire_c1_shadow_lifecycle(
        42,
        real_slot=7,
        shadow_slot=3,
    )

    assert lifecycle.physical_slots == (7, 3)
    assert lifecycle.target_sessions == (
        resources["real_target_session"],
        resources["shadow_target_session"],
    )
    assert lifecycle.hidden_rows == (
        resources["real_hidden_row"],
        resources["shadow_hidden_row"],
    )
    assert 42 in adapter._c1_shadow_states
    adapter.drop_c1_shadow_lifecycle(42, reason="request_drop")
    adapter.drop_c1_shadow_lifecycle(42, reason="request_drop")

    assert adapter._c1_shadow_states == {}
    assert [event for event in events if event[0] == "reset"] == [
        ("reset", -43)
    ]
    assert len([event for event in events if event[0] == "release_checkpoint"]) == 2
    assert len([event for event in events if event[0] == "reclaim"]) == 8
    assert [event for event in events if event[0] == "release_request"] == [
        ("release_request", -43)
    ]
    assert not [event for event in events if event[0] == "restore"]


def test_adapter_cancel_restores_both_provider_checkpoints() -> None:
    adapter, _resources_by_name, events = _adapter_resource_fixture()
    adapter.acquire_c1_shadow_lifecycle(42, real_slot=0, shadow_slot=1)

    adapter.drop_c1_shadow_lifecycle(42, cancel=True)

    assert [event for event in events if event[0] == "restore"] == [
        ("restore", "checkpoint_42"),
        ("restore", "checkpoint_-43"),
    ]
    assert len([event for event in events if event[0] == "release_checkpoint"]) == 2
    assert [event for event in events if event[0] == "release_request"] == [
        ("release_request", -43)
    ]


def test_adapter_acquire_failure_releases_provider_and_owner_resources() -> None:
    adapter, _resources_by_name, events = _adapter_resource_fixture(
        fail_second_checkpoint=True
    )

    with pytest.raises(RuntimeError, match="injected checkpoint failure"):
        adapter.acquire_c1_shadow_lifecycle(42, real_slot=0, shadow_slot=1)

    assert adapter._c1_shadow_states == {}
    assert [event for event in events if event[0] == "release_checkpoint"] == [
        ("release_checkpoint", "checkpoint_42")
    ]
    assert [event for event in events if event[0] == "release_request"] == [
        ("release_request", -43)
    ]
    assert [event for event in events if event[0] == "abort"] == [
        ("abort", "acquire_failed")
    ]
