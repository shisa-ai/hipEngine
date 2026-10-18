"""CPU ownership contracts for split-attention scratch (no HIP execution)."""
import pytest

from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention


class Runtime:
    def __init__(self):
        self.live = {}
        self.events = []
        self.next_ptr = 4096
        self.device = 0
        self.fail = False
        self.synced = set()
        self.event_stream = {}
        self.last_event = {}
        self.next_event = 100

    def current_device(self):
        return self.device

    def malloc(self, size):
        if self.fail:
            raise MemoryError("allocation failed")
        self.next_ptr += 4096
        self.live[self.next_ptr] = size
        self.events.append(("malloc", self.next_ptr))
        return self.next_ptr

    def free(self, ptr):
        self.events.append(("free", ptr))
        del self.live[ptr]

    def stream_synchronize(self, stream):
        self.events.append(("sync", stream))
        self.synced.add(stream)

    def event_create(self):
        self.next_event += 1
        return self.next_event

    def event_record(self, event, stream=0):
        self.events.append(("record", event, stream))
        self.event_stream[event] = stream
        self.last_event[stream] = event

    def event_query(self, event):
        if self.event_stream[event] in self.synced:
            return True
        # The device is modeled as at most one retirement behind the host on a
        # stream: an event completes once a later one is recorded behind it.
        return self.last_event.get(self.event_stream[event]) != event

    def event_destroy(self, event):
        self.events.append(("destroy", event))
        del self.event_stream[event]


def test_scratch_growth_is_bounded_and_frees_superseded_buffers_as_the_device_moves_on():
    runtime = Runtime()
    scratch = attention.Gemma4AttentionScratch()
    for size in range(1, 514):
        assert scratch.buffer(size, stream=7, runtime=runtime).nbytes >= size
    # The doubling chain no longer accumulates for the run's whole life:
    # superseded buffers are retired behind stream events and reaped once the
    # device passes them, so frees happen during the run and, after the stream
    # quiets, only the current buffer is left -- not the sum of every
    # superseded allocation, which is what the retain-until-close chain held.
    assert any(event[0] == "free" for event in runtime.events)
    runtime.stream_synchronize(7)
    current = scratch.buffer(513, stream=7, runtime=runtime)
    assert list(runtime.live) == [current.ptr]
    scratch.close()
    assert runtime.live == {}
    before = list(runtime.events)
    scratch.close()
    assert runtime.events == before
    with pytest.raises(RuntimeError, match="closed"):
        scratch.buffer(1, stream=7, runtime=runtime)


def test_scratch_retires_behind_an_event_and_reaps_after_the_stream_passes_it():
    runtime = Runtime()
    scratch = attention.Gemma4AttentionScratch()
    first = scratch.buffer(100, stream=7, runtime=runtime)
    assert scratch.buffer(100, stream=7, runtime=runtime) is first
    second = scratch.buffer(200, stream=7, runtime=runtime)
    assert second is not first
    # Parked, not freed: queued kernels may still hold the superseded buffer.
    assert first.ptr in runtime.live
    assert not any(event[0] == "free" for event in runtime.events)
    # A later retirement on the same stream models the device moving past the
    # first event, so the next call reaps the first retired buffer only.
    third = scratch.buffer(400, stream=7, runtime=runtime)
    assert not any(event[0] == "free" for event in runtime.events)
    assert scratch.buffer(400, stream=7, runtime=runtime) is third
    assert ("free", first.ptr) in runtime.events
    assert set(runtime.live) == {second.ptr, third.ptr}
    # Once the stream is synchronized the last retired buffer goes too.
    runtime.stream_synchronize(7)
    assert scratch.buffer(300, stream=7, runtime=runtime) is third
    assert set(runtime.live) == {third.ptr}
    scratch.close()
    assert runtime.live == {}


def test_scratch_reuses_capacity_but_isolates_owners_and_streams():
    runtime = Runtime()
    a, b = attention.Gemma4AttentionScratch(), attention.Gemma4AttentionScratch()
    first = a.buffer(100, stream=0, runtime=runtime)
    assert a.buffer(99, stream=0, runtime=runtime) is first
    assert a.buffer(100, stream=1, runtime=runtime).ptr != first.ptr
    assert b.buffer(100, stream=0, runtime=runtime).ptr != first.ptr
    a.close()
    assert len(runtime.live) == 1
    b.close()
    assert not runtime.live


def test_failed_growth_preserves_old_owner_and_close_releases_it():
    runtime = Runtime()
    scratch = attention.Gemma4AttentionScratch()
    first = scratch.buffer(100, stream=0, runtime=runtime)
    runtime.fail = True
    with pytest.raises(MemoryError):
        scratch.buffer(101, stream=0, runtime=runtime)
    assert scratch.buffer(100, stream=0, runtime=runtime) is first
    scratch.close()
    assert not runtime.live


def test_scratch_rejects_runtime_or_device_migration():
    runtime = Runtime()
    scratch = attention.Gemma4AttentionScratch()
    scratch.buffer(1, stream=0, runtime=runtime)
    with pytest.raises(ValueError, match="runtime"):
        scratch.buffer(1, stream=0, runtime=Runtime())
    runtime.device = 1
    with pytest.raises(ValueError, match="device"):
        scratch.buffer(1, stream=0, runtime=runtime)
    with pytest.raises(ValueError, match="device"):
        scratch.close()
    runtime.device = 0
    scratch.close()


def test_layer_scratch_owns_and_closes_attention_scratch():
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
        Gemma4LayerGeometry, Gemma4LayerScratch,
    )
    layer = Gemma4LayerScratch(
        tokens=1, hidden_size=16, dense_intermediate=32,
        geometry=Gemma4LayerGeometry(2, 1, 256),
        num_experts=4, top_k=2, expert_intermediate=8,
    )
    runtime = Runtime()
    layer.attention.buffer(100, stream=0, runtime=runtime)
    layer.free()
    layer.free()
    assert not runtime.live


@pytest.mark.parametrize("launch_error", [False, True])
def test_wrapper_without_owner_releases_temporary_even_on_launch_error(monkeypatch, launch_error):
    runtime = Runtime()
    monkeypatch.setattr(attention, "decode_slices", lambda keys, head_dim: 4)
    monkeypatch.setattr(attention, "split_workspace_bytes", lambda *args, **kw: 128)

    def launch(*args):
        assert runtime.live
        if launch_error:
            raise RuntimeError("partial launch failed")
        return 0

    monkeypatch.setattr(attention, "signed_kernel_fn", lambda *args: launch)
    kwargs = dict(tokens=1, num_heads=1, num_kv_heads=1, head_dim=256,
                  scale=1.0, keys=1024, stream=3, library=object(), runtime=runtime)
    if launch_error:
        with pytest.raises(RuntimeError, match="partial launch"):
            attention.gemma4_attention_prefill_f32(1, 2, 3, 4, 5, **kwargs)
    else:
        attention.gemma4_attention_prefill_f32(1, 2, 3, 4, 5, **kwargs)
    assert runtime.live == {}
    assert runtime.events[-2][0] == "sync"
    assert runtime.events[-1][0] == "free"


@pytest.mark.parametrize("tokens", [1, 3, 9])
@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("keys", [15857, 16640, 32771])
def test_large_context_uses_owned_global_logits(monkeypatch, tokens, head_dim, keys):
    runtime = Runtime()
    calls = []
    monkeypatch.setattr(attention, "split_workspace_bytes", lambda *args, **kw: 8192)

    def launch(*args):
        calls.append(args)
        assert runtime.live, "global logits require owned scratch even without split"
        assert args[-2].value != 0
        return 0

    monkeypatch.setattr(attention, "signed_kernel_fn", lambda lib, symbol, *args: (
        calls.append(symbol) or launch
    ))
    attention.gemma4_attention_prefill_f32(
        1, 2, 3, 4, 5, tokens=tokens, num_heads=2, num_kv_heads=1,
        head_dim=head_dim, scale=1.0, keys=keys, library=object(), runtime=runtime,
    )
    assert calls[0] == attention._SYMBOL_DECODE_F32
    launches = [call for call in calls if isinstance(call, tuple)]
    assert sum(call[5] for call in launches) == tokens
    assert all(call[5] <= 4 for call in launches)
    for index, call in enumerate(launches):
        assert call[0] == 1 + index * 4 * 2 * head_dim * 4
        assert call[3] == 4 + index * 4 * keys
    assert runtime.live == {}


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("keys", [15857, 16640, 32771])
def test_large_context_has_bounded_shared_memory(head_dim, keys):
    assert attention.gemma4_attention_shared_bytes(head_dim=head_dim, keys=keys) <= 65536
