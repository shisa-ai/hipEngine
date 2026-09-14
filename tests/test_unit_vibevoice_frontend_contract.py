"""Validate recording shapes and scratch ownership before device work."""
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime,_ScratchPool


@pytest.mark.parametrize('pcm',[np.array([]),np.array([np.nan]),np.zeros((1,3200))])
def test_invalid_recording_rejected_without_device(pcm):
    r = VibevoiceFrontendRuntime.__new__(VibevoiceFrontendRuntime)
    with pytest.raises(ValueError,match='PCM'):
        r.encode(pcm)


@pytest.mark.parametrize('chunk',[0,3199,-3200,True,3200.5])
def test_invalid_chunk_rejected_without_device(chunk):
    r = VibevoiceFrontendRuntime.__new__(VibevoiceFrontendRuntime)
    with pytest.raises(ValueError,match='chunk_samples'):
        r.encode(np.zeros(3200),chunk_samples=chunk)


def test_arena_exhaustion_cannot_free_live_views():
    pool = _ScratchPool()
    arena = Mock()
    arena.allocate.side_effect = MemoryError('full')
    pool._arena = arena
    with pytest.raises(MemoryError):
        pool.take(1024)
    arena.close.assert_not_called()
    assert pool._arena is arena


def test_scratch_pool_rewinds_an_arena_that_is_already_large_enough():
    """A reset that fits the existing arena must rewind it, not reallocate.

    Freeing and reallocating the arena on every reset charged each encoder
    pass a hipMalloc/hipFree pair for the whole arena.
    """
    pool = _ScratchPool(capacity_bytes=1 << 20)
    arena = Mock()
    arena.closed = False
    arena.capacity_bytes = 1 << 20
    pool._arena = arena

    pool.reset(capacity_bytes=1 << 20)

    arena.rewind.assert_called_once_with()
    arena.close.assert_not_called()
    assert pool._arena is arena


def test_scratch_pool_grows_only_when_the_request_does_not_fit(monkeypatch):
    """A larger request replaces the arena; a smaller one reuses it."""
    import hipengine.runtime.vibevoice_encoder as encoder

    created = []

    class _FakeArena:
        @staticmethod
        def create(capacity_bytes, **kwargs):
            made = Mock()
            made.closed = False
            made.capacity_bytes = int(capacity_bytes)
            created.append(made)
            return made

    monkeypatch.setattr(encoder, "DeviceMemoryArena", _FakeArena)

    pool = _ScratchPool(capacity_bytes=1 << 20)
    small = Mock()
    small.closed = False
    small.capacity_bytes = 1 << 20
    pool._arena = small

    pool.reset(capacity_bytes=2 << 20)  # does not fit: grow
    small.close.assert_called_once_with()
    assert pool._arena is created[-1]

    pool.reset(capacity_bytes=1 << 20)  # fits again: rewind, never shrink
    assert len(created) == 1
    created[-1].rewind.assert_called_once_with()
    assert pool._arena is created[-1]


def test_scratch_pool_reset_restarts_the_bump_region():
    """Reusing the arena must still hand out the same starting offsets."""
    pool = _ScratchPool(capacity_bytes=1 << 20)
    arena = Mock()
    arena.closed = False
    arena.capacity_bytes = 1 << 20
    pool._arena = arena

    pool.reset(capacity_bytes=1 << 20)
    pool.reset(capacity_bytes=1 << 20)

    assert arena.rewind.call_count == 2
    arena.close.assert_not_called()


def test_noise_covers_partial_frame_before_encoding():
    r = VibevoiceFrontendRuntime.__new__(VibevoiceFrontendRuntime)
    r.specs = {'acoustic':SimpleNamespace(hidden_size=64)}
    with pytest.raises(ValueError,match='joined recording'):
        r.forward(np.zeros(3201),noise=np.zeros((1,64)),noise_scale=.25)
