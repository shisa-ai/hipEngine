import pytest
from scripts.qwen4exp_qsa_decode_flag_probe import orders, phase_flags


def test_balanced_order():
    assert orders(4)==[(0,1),(1,0),(0,1),(1,0)]


@pytest.mark.parametrize("pairs",[0,1,3,-2])
def test_reject_unbalanced(pairs):
    with pytest.raises(ValueError):
        orders(pairs)


def test_prefill_isolation_keeps_decode_off():
    assert phase_flags(True,0)==("0","0")
    assert phase_flags(True,1)==("1","0")
    assert phase_flags(False,1)==("0","1")


def test_markers_distinguish_warmup_and_measured_arms():
    from scripts.qwen4exp_qsa_decode_flag_probe import step_marker
    assert len({step_marker("code-p4096",p,a,s)
                for p in (-1,0,1) for a in (0,1) for s in range(16)})==96
