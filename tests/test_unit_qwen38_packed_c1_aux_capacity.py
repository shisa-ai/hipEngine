"""Provider commit owns one BF16 row inside a larger scratch allocation."""
from types import SimpleNamespace as NS
import pytest


@pytest.mark.parametrize('capacity', [2, 8])
@pytest.mark.parametrize('fault', [None, 'tail', 'resize'])
def test_aux_commit_preserves_unused_allocation(monkeypatch, capacity, fault):
    from scripts import qwen38_packed_c1_state as m
    from hipengine.core import DType
    session = NS(runtime=NS(device_synchronize=lambda: None), runner=NS(hidden_size=2),
                 _hidden_a=NS(ptr=500, nbytes=4*capacity), _last_target_hidden_ptr=500,
                 scratch=NS(position_buf=NS(ptr=600, nbytes=8), context_buf=NS(ptr=700, nbytes=8)))
    result = NS(start_position=17, pre_output_norm_hidden=NS(ptr=100, shape=(4, 2), dtype=DType.BF16))
    monkeypatch.setattr(m, '_device_hash', lambda owner, b, **kw: f'{b.ptr}:{b.nbytes}')
    expected = m.selected_aux_sources(session, result, accepted=1)
    assert expected['provider_hidden']['nbytes'] == 4
    assert expected['provider_tail'] == dict(ptr=504, nbytes=4*(capacity-1), hash=f'504:{4*(capacity-1)}')
    hashes = {row['ptr']: row['hash'] for row in expected.values()}
    if fault == 'tail':
        hashes[504] = 'corruption'
    if fault == 'resize':
        session._hidden_a.nbytes += 4
    monkeypatch.setattr(m, '_device_hash', lambda owner, b, **kw: hashes[b.ptr])
    if fault:
        with pytest.raises(ValueError):
            m.assert_aux_commit(session, expected)
    else:
        m.assert_aux_commit(session, expected)
