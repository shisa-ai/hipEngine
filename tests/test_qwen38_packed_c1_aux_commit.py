"""Exact cursor/provider-hidden commit checks independent of GPU metadata."""
from types import SimpleNamespace as NS
import hashlib
import numpy as np
import pytest


@pytest.mark.parametrize('accepted', [0, 1, 3])
def test_aux_sources_use_local_provider_row_and_cpu_cursor_values(monkeypatch, accepted):
    from scripts import qwen38_packed_c1_state as m
    from hipengine.core import DType
    monkeypatch.setattr(m, '_device_hash', lambda owner, b, **kw: f'{b.ptr}:{b.nbytes}')
    session = NS(runtime=NS(device_synchronize=lambda: None), runner=NS(hidden_size=2),
                 _hidden_a=NS(ptr=500, nbytes=4),
                 scratch=NS(position_buf=NS(ptr=600, nbytes=8), context_buf=NS(ptr=700, nbytes=8)))
    result = NS(start_position=17, row_start=8,
                pre_output_norm_hidden=NS(ptr=100, nbytes=16, shape=(4, 2), dtype=DType.BF16))
    expected = m.selected_aux_sources(session, result, accepted=accepted)
    assert expected['provider_hidden'] == dict(ptr=500, nbytes=4, hash=f'{100+accepted*4}:4')
    for name, value, ptr in [('position_device', 18+accepted, 600), ('context_device', 19+accepted, 700)]:
        assert expected[name] == dict(ptr=ptr, nbytes=8, hash=hashlib.blake2b(np.array([value], dtype=np.int64).tobytes(), digest_size=16).hexdigest())
    result.pre_output_norm_hidden = None
    with pytest.raises(ValueError):
        m.selected_aux_sources(session, result, accepted=accepted)


@pytest.mark.parametrize('fault', ['negative', 'overflow', 'dtype', 'shape', 'destination', 'cursor'])
def test_aux_source_metadata_fails_closed(fault):
    from scripts import qwen38_packed_c1_state as m
    from hipengine.core import DType
    session = NS(runtime=NS(device_synchronize=lambda: None), runner=NS(hidden_size=2),
                 _hidden_a=NS(ptr=500, nbytes=4),
                 scratch=NS(position_buf=NS(ptr=600, nbytes=8), context_buf=NS(ptr=700, nbytes=8)))
    result = NS(start_position=17, pre_output_norm_hidden=NS(ptr=100, shape=(4, 2), dtype=DType.BF16))
    accepted = 1
    if fault == 'negative':
        accepted = -1
    elif fault == 'overflow':
        accepted = 4
    elif fault == 'dtype':
        result.pre_output_norm_hidden.dtype = DType.FP32
    elif fault == 'shape':
        result.pre_output_norm_hidden.shape = (4, 3)
    elif fault == 'destination':
        session._hidden_a = None
    else:
        # Cursor validation follows source hashing. No HIP access in this CPU test.
        session.scratch.position_buf.nbytes = 4
    from unittest.mock import patch
    with patch.object(m, '_device_hash', return_value='source'), pytest.raises(ValueError):
        m.selected_aux_sources(session, result, accepted=accepted)


@pytest.mark.parametrize('fault', [None, 'provider_hidden', 'position_device', 'context_device', 'last_hidden_ptr'])
def test_aux_commit_fails_on_missing_copy_or_cursor(monkeypatch, fault):
    from scripts import qwen38_packed_c1_state as m
    b = lambda ptr, nbytes: NS(ptr=ptr, nbytes=nbytes)
    session = NS(runtime=NS(device_synchronize=lambda: None), _hidden_a=b(100, 4),
                 _last_target_hidden_ptr=999 if fault == 'last_hidden_ptr' else 100,
                 scratch=NS(position_buf=b(200, 8), context_buf=b(300, 8)))
    expected = {'provider_hidden': dict(ptr=100, nbytes=4, hash='ok'),
                'position_device': dict(ptr=200, nbytes=8, hash='ok'),
                'context_device': dict(ptr=300, nbytes=8, hash='ok')}
    bad_ptr = expected.get(fault, {}).get('ptr')
    monkeypatch.setattr(m, '_device_hash', lambda owner, b, **kw: 'bad' if b.ptr == bad_ptr else 'ok')
    if fault:
        with pytest.raises(ValueError):
            m.assert_aux_commit(session, expected)
    else:
        m.assert_aux_commit(session, expected)
