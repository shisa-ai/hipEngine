"""Packed workspace owner releases storage and pool charges in order."""
from types import SimpleNamespace

from hipengine.runtime.qwen35_gguf_workspace import PackedWorkspaceOwner


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
