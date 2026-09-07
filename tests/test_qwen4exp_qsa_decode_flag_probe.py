import pytest
from scripts.qwen4exp_qsa_decode_flag_probe import orders


def test_balanced_order():
    assert orders(4)==[(0,1),(1,0),(0,1),(1,0)]


@pytest.mark.parametrize("pairs",[0,1,3,-2])
def test_reject_unbalanced(pairs):
    with pytest.raises(ValueError):
        orders(pairs)
