import pytest
from scripts.qwen4exp_gdn_owner_probe import pair_order,validate_geometry


def test_pair_order_balances_measured_pairs():
    for index in range(21):
        assert pair_order(index+1)==tuple(reversed(pair_order(index+2)))


def test_geometry_fails_closed():
    validate_geometry([1]*9+[1024,16,48,128,128])
    for shape in ([1,16,48,128,128],[1025,16,48,128,128],[512,8,48,128,128]):
        with pytest.raises(ValueError):
            validate_geometry([1]*9+shape)
