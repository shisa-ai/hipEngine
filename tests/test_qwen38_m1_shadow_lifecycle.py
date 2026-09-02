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
    runtime = SimpleNamespace(
        memcpy_async=lambda dst, src, nbytes, kind, stream: events.append(
            ("memcpy", dst, src, nbytes, kind, stream)
        ),
        device_synchronize=lambda: events.append(("sync",)),
    )
    real_target = SimpleNamespace(
        name="real_target_session",
        scratch=resources["real_recurrent_owner"],
        runtime=runtime,
    )
    shadow_target = SimpleNamespace(
        name="shadow_target_session",
        clone_current_state_from=lambda source, stream: (
            events.append(("clone_target", source.name, stream)) or 1234
        ),
    )
    resources["real_target_session"] = real_target
    resources["shadow_target_session"] = shadow_target
    resources["real_hidden_row"] = SimpleNamespace(
        name="real_hidden_row", ptr=0x1000, nbytes=16
    )
    resources["shadow_hidden_row"] = SimpleNamespace(
        name="shadow_hidden_row", ptr=0x2000, nbytes=16
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
    executor.clone_request_state = lambda source, destination: events.append(
        ("clone_provider", source, destination)
    )
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
    assert [event for event in events if event[0] == "clone_target"] == [
        ("clone_target", "real_target_session", 0)
    ]
    assert [event for event in events if event[0] == "clone_provider"] == [
        ("clone_provider", 42, -43)
    ]
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


def test_adapter_provider_clone_miss_fails_closed_and_aborts_owner() -> None:
    adapter, _resources_by_name, events = _adapter_resource_fixture()
    del adapter._states[42].provider.executor.clone_request_state

    with pytest.raises(mtp2.C1ShadowOwnershipError, match="provider.*clone ABI"):
        adapter.acquire_c1_shadow_lifecycle(42, real_slot=0, shadow_slot=1)

    assert adapter._c1_shadow_states == {}
    assert [event for event in events if event[0] == "release_request"] == [
        ("release_request", -43)
    ]
    assert [event for event in events if event[0] == "abort"] == [
        ("abort", "acquire_failed")
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


def _resident_owner_fixture(monkeypatch):
    import hipengine.generation.qwen35_gguf as generation

    events = []
    real_allocation = SimpleNamespace(block_ids=(1, 2), request_id=42)
    real_session = SimpleNamespace(name="real", scratch=SimpleNamespace(name="real_scratch"))
    row = SimpleNamespace(
        lease=SimpleNamespace(session=real_session),
        kv_allocation=real_allocation,
    )
    shadow_session = SimpleNamespace(
        name="shadow",
        scratch=SimpleNamespace(name="shadow_scratch"),
        runtime="rt",
        device_kv_allocation=None,
    )

    def bind(pool, allocation):
        events.append(("bind", allocation.request_id))
        shadow_session.device_kv_allocation = allocation

    def unbind():
        events.append(("unbind",))
        allocation = shadow_session.device_kv_allocation
        shadow_session.device_kv_allocation = None
        return allocation

    shadow_session.bind_device_kv_allocation = bind
    shadow_session.unbind_device_kv_allocation = unbind
    shadow_session.invalidate_device_kv_graphs = lambda: events.append(
        ("invalidate",)
    )
    shadow_session.reset = lambda: events.append(("reset",))
    shadow_lease = SimpleNamespace(session=shadow_session, pool_key="pool")

    class Pool:
        def allocate(self, request_id, pages, *, now_seconds):
            del now_seconds
            events.append(("allocate", request_id, pages))
            return SimpleNamespace(
                block_ids=tuple(range(10, 10 + pages)),
                request_id=request_id,
            )

        def release(self, request_id, *, now_seconds):
            del now_seconds
            events.append(("release_kv", request_id))

    hidden_buffers = []

    def fake_malloc(nbytes, *, runtime):
        events.append(("malloc", nbytes, runtime))
        buffer = SimpleNamespace(name="shadow_hidden", ptr=0x5000, nbytes=nbytes)
        hidden_buffers.append(buffer)
        return buffer

    monkeypatch.setattr(generation, "malloc", fake_malloc)
    monkeypatch.setattr(
        generation,
        "free",
        lambda buffer, *, runtime: events.append(
            ("free", buffer.name, runtime)
        ),
    )
    owner = object.__new__(generation.Qwen35GGUFResidentModelRunner)
    owner._rows = {42: row}
    owner._available = [shadow_lease]
    owner._kv_pool = Pool()
    owner._shared_runner = SimpleNamespace(hidden_size=8)
    owner._c1_shadow_resource_bundles = {}
    return owner, row, shadow_lease, events, hidden_buffers


def test_resident_owner_reserves_and_reclaims_shadow_pool_bundle(monkeypatch) -> None:
    owner, row, shadow_lease, events, hidden_buffers = _resident_owner_fixture(
        monkeypatch
    )

    bundle = owner.acquire_c1_shadow_resources(
        request_id=42,
        shadow_request_id=-43,
        real_slot=0,
        shadow_slot=1,
    )

    assert owner._available == []
    assert bundle["target_session"] is shadow_lease.session
    assert bundle["kv_owner"].request_id == -43
    assert bundle["recurrent_owner"] is shadow_lease.session.scratch
    assert bundle["hidden_row"] is hidden_buffers[0]
    assert row.kv_allocation.request_id == 42
    for surface in ("target_session", "hidden_row", "kv_owner", "recurrent_owner"):
        for lane in ("real", "shadow"):
            owner.reclaim_c1_shadow_resource(
                bundle,
                surface=surface,
                lane=lane,
                resource=object(),
                reason="request_drop",
            )

    assert owner._available == [shadow_lease]
    assert owner._c1_shadow_resource_bundles == {}
    assert events == [
        ("allocate", -43, 2),
        ("bind", -43),
        ("malloc", 16, "rt"),
        ("free", "shadow_hidden", "rt"),
        ("invalidate",),
        ("unbind",),
        ("release_kv", -43),
        ("reset",),
    ]


def test_resident_owner_abort_returns_shadow_bundle(monkeypatch) -> None:
    owner, _row, shadow_lease, events, _hidden = _resident_owner_fixture(
        monkeypatch
    )
    bundle = owner.acquire_c1_shadow_resources(
        request_id=42,
        shadow_request_id=-43,
        real_slot=0,
        shadow_slot=1,
    )

    owner.abort_c1_shadow_resources(bundle, reason="acquire_failed")
    owner.abort_c1_shadow_resources(bundle, reason="acquire_failed")

    assert owner._available == [shadow_lease]
    assert owner._c1_shadow_resource_bundles == {}
    assert events.count(("release_kv", -43)) == 1
    assert events.count(("free", "shadow_hidden", "rt")) == 1


def test_resident_owner_shadow_capacity_failure_is_side_effect_free(monkeypatch) -> None:
    owner, row, _shadow_lease, events, _hidden = _resident_owner_fixture(
        monkeypatch
    )
    owner._available = []

    with pytest.raises(RuntimeError, match="additional resident session"):
        owner.acquire_c1_shadow_resources(
            request_id=42,
            shadow_request_id=-43,
            real_slot=0,
            shadow_slot=1,
        )

    assert events == []
    assert owner._c1_shadow_resource_bundles == {}
    assert row.kv_allocation.request_id == 42
