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


def test_scratch_growth_is_bounded_and_freed_after_stream_completion():
    runtime = Runtime()
    scratch = attention.Gemma4AttentionScratch()
    for size in range(1, 514):
        assert scratch.buffer(size, stream=7, runtime=runtime).nbytes >= size
    assert sum(runtime.live.values()) < 4 * 513
    assert not any(event[0] in ("free", "sync") for event in runtime.events)
    scratch.close()
    assert runtime.live == {}
    first_free = next(i for i, event in enumerate(runtime.events) if event[0] == "free")
    assert ("sync", 7) in runtime.events[:first_free]
    before = list(runtime.events)
    scratch.close()
    assert runtime.events == before
    with pytest.raises(RuntimeError, match="closed"):
        scratch.buffer(1, stream=7, runtime=runtime)


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
