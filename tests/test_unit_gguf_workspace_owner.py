"""Packed workspace owner releases storage and pool charges in order."""
from types import SimpleNamespace

import pytest

from hipengine.runtime.qwen35_gguf_workspace import (
    PackedWorkspaceOwner, StandalonePackedBackingLease, allocate_private_packed_backing,
)


def test_standalone_backing_close_retries_only_unreleased_buffers():
    events = []
    lease = StandalonePackedBackingLease.acquire(
        pages=2, page_bytes=16, pool=None,
        allocate=lambda: SimpleNamespace(buffers=(1, 2)),
    )
    def release(buffer, **kwargs):
        if buffer == 1:
            raise RuntimeError('injected release failure')
        events.append(buffer)
    with pytest.raises(RuntimeError, match='injected release failure'):
        lease.close(runtime=None, release=release)
    assert lease.remaining_buffers == [1]
    lease.close(runtime=None, release=lambda b, **kw: events.append(b))
    assert events == [2, 1]


@pytest.mark.parametrize('with_pool', [False, True])
def test_standalone_backing_lease_owns_buffers_and_charge_once(with_pool):
    events = []
    buffers = tuple(SimpleNamespace(ptr=i, nbytes=16) for i in (1, 2))
    pool = SimpleNamespace(
        reserve_private_workspace=lambda size: events.append(('reserve', size)) or 'token',
        release_private_workspace=lambda token: events.append(('charge', token)),
    ) if with_pool else None
    lease = StandalonePackedBackingLease.acquire(
        pages=2, page_bytes=16, pool=pool,
        allocate=lambda: SimpleNamespace(buffers=buffers),
    )
    owner = PackedWorkspaceOwner(verify_state=SimpleNamespace(
        buffers=buffers, backing_lease=lease,
    ))
    assert owner.nbytes() == 32
    owner.close(runtime=SimpleNamespace(), release=lambda b, **kw: events.append(('free', b.ptr)))
    lease.close(runtime=SimpleNamespace(), release=lambda b, **kw: pytest.fail('double free'))
    assert events == (([('reserve', 32)] if with_pool else [])
                      + [('free', 2), ('free', 1)]
                      + ([('charge', 'token')] if with_pool else []))


@pytest.mark.parametrize('with_pool', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_private_backing_transfers_charge_only_after_allocation(with_pool, failure):
    events = []
    pool = SimpleNamespace(
        reserve_private_workspace=lambda size: events.append(('reserve', size)) or 'token',
        release_private_workspace=lambda token: events.append(('release', token)),
    ) if with_pool else None
    backing = object()
    def allocate():
        events.append(('allocate',))
        if failure:
            raise MemoryError('injected')
        return backing
    if failure:
        with pytest.raises(MemoryError, match='injected'):
            allocate_private_packed_backing(pages=3, page_bytes=16, pool=pool, allocate=allocate)
    else:
        result, token = allocate_private_packed_backing(pages=3, page_bytes=16, pool=pool, allocate=allocate)
        assert result is backing
        assert token == ('token' if with_pool else None)
    assert events == (
        ([('reserve', 48)] if with_pool else []) + [('allocate',)]
        + ([('release', 'token')] if with_pool and failure else [])
    )


def test_workspace_owner_counts_unique_buffers_and_releases_charges_after_storage():
    events = []
    def buffer(ptr, nbytes):
        return SimpleNamespace(ptr=ptr, nbytes=nbytes)
    common = buffer(1, 16)
    extra = buffer(2, 32)
    state = SimpleNamespace(
        buffers=(common,),
        private_workspace_pool=SimpleNamespace(release_private_workspace=lambda token: events.append(('private', token))),
        private_workspace_token='p',
        resident_charge_pool=SimpleNamespace(release_private_workspace=lambda token: events.append(('charge', token))),
        resident_charge_token='c',
        growth_lease_pool=SimpleNamespace(release_workspace=lambda token: events.append(('growth', token))),
        growth_lease_key='g',
    )
    owner = PackedWorkspaceOwner(
        verify_state=state,
        verify_scratch=SimpleNamespace(buffers=(), full_attn_split_growth_buffers=(extra,)),
        ar_attention_workspace=SimpleNamespace(buffers=()),
        decode_sessions=(object(),), decode_state_dirty=True,
    )
    assert owner.nbytes() == 48
    runtime = SimpleNamespace(device_synchronize=lambda: events.append(('sync',)))
    owner.close(runtime=runtime, release=lambda item, **kwargs: events.append(('free', item.ptr)))
    assert events == [('sync',), ('free', 2), ('free', 1), ('private', 'p'), ('charge', 'c'), ('growth', 'g')]
    assert owner.nbytes() == 0
    assert owner.verify_state is None and owner.verify_scratch is None
    assert owner.ar_attention_workspace is None
    assert owner.decode_sessions == () and not owner.decode_state_dirty


def test_workspace_owner_retry_does_not_repeat_completed_workspace_release():
    events = []
    owner = PackedWorkspaceOwner(
        ar_attention_workspace=SimpleNamespace(buffers=(SimpleNamespace(ptr=1),)),
        verify_scratch=SimpleNamespace(buffers=(SimpleNamespace(ptr=2),)),
    )
    failing = True

    def release(buffer, **kwargs):
        if buffer.ptr == 2 and failing:
            raise RuntimeError('release failed')
        events.append(buffer.ptr)

    runtime = SimpleNamespace()
    with pytest.raises(RuntimeError, match='release failed'):
        owner.close(runtime=runtime, release=release)
    assert owner.ar_attention_workspace is None
    assert owner.verify_scratch is not None
    failing = False
    owner.close(runtime=runtime, release=release)
    owner.close(runtime=runtime, release=release)
    assert events == [1, 2]
    assert owner.verify_scratch is None
