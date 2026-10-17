import pytest
from scripts.qwen4exp_mtp_boundary_http_gate import choose_boundary, validate_boundary


def test_boundary_selection_is_interior_and_unambiguous():
    result = dict(text='abcdefghijklmnopqrstuvwxyz0123456789', ids=list(range(48)))
    boundary = choose_boundary(result)
    assert boundary['prefix'] == result['text'][:10]
    assert len(boundary['stop']) == 20
    assert boundary['eos_index'] == 6
    with pytest.raises(AssertionError):
        choose_boundary(dict(text='short',ids=[1]))


@pytest.mark.parametrize('kind,result', [
    ('stop',dict(text='prefix',finish_reason='stop')),
    ('alternate_eos',dict(ids=[1,2,9],finish_reason='stop')),
    ('min_tokens',dict(ids=list(range(10,26)),finish_reason='length')),
    ('ignore_eos',dict(ids=[9]*48,finish_reason='length')),
])
def test_boundary_validation(kind,result):
    boundary=dict(prefix='prefix',eos_id=9,eos_index=2)
    validate_boundary(result,kind,boundary)
    bad=result|dict(text='wrong',ids=[9],finish_reason='length')
    with pytest.raises(AssertionError):
        validate_boundary(bad,kind,boundary)
