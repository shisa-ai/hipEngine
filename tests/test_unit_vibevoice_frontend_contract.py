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


def test_noise_covers_partial_frame_before_encoding():
    r = VibevoiceFrontendRuntime.__new__(VibevoiceFrontendRuntime)
    r.specs = {'acoustic':SimpleNamespace(hidden_size=64)}
    with pytest.raises(ValueError,match='joined recording'):
        r.forward(np.zeros(3201),noise=np.zeros((1,64)),noise_scale=.25)
