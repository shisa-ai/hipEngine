"""Frozen request integrity and noise protocol regression tests; no torch."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts.vibevoice_asr_bench import array_hash, _read_request, recorded_noise


def test_tampered_request_is_rejected(tmp_path):
    path = tmp_path/'request.npz'
    pcm = np.array([0,1],dtype=np.float32)
    path.with_suffix('.json').write_text(json.dumps({'hashes':{'pcm':array_hash(pcm)}}))
    np.savez(path,pcm=pcm)
    assert np.array_equal(_read_request(path)[0]['pcm'],pcm)
    np.savez(path,pcm=pcm+1)
    with pytest.raises(ValueError,match='hash'):
        _read_request(path)


def test_noise_draws_are_supplied_once_and_hooks_restore():
    torch = SimpleNamespace(randn=Mock(),randn_like=Mock())
    original = torch.randn
    noise = SimpleNamespace(shape=(1,2,64),to=Mock(return_value='noise'))
    scale = SimpleNamespace(to=Mock(return_value='scale'))
    like = SimpleNamespace(shape=noise.shape,device='gpu',dtype='bf16')
    with recorded_noise(torch, noise, scale):
        assert torch.randn(1,device='gpu',dtype='bf16') == 'scale'
        assert torch.randn_like(like) == 'noise'
    assert torch.randn is original
    original.assert_not_called()


def test_unexpected_rng_protocol_fails():
    torch = SimpleNamespace(randn=Mock(),randn_like=Mock())
    with pytest.raises(RuntimeError,match='protocol'):
        with recorded_noise(torch, None, None):
            pass


def test_hash_includes_shape_and_dtype():
    x = np.zeros(4,dtype=np.float32)
    assert len({array_hash(x),array_hash(x.reshape(2,2)),array_hash(x.view(np.int32))}) == 3
