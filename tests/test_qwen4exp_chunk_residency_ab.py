from types import SimpleNamespace

import pytest
from scripts import qwen4exp_halo_box_campaign_ab as ab


def test_chunk_mode_does_not_touch_kernel_flags():
    env = {"existing":"keep"}
    for mode in ("before","after"):
        ab._apply_mode(mode,environment=env,route_package="chunk1024")
        assert env == {"existing":"keep"}
    with pytest.raises(ValueError):
        ab._apply_mode("invalid",environment=env,route_package="chunk1024")


def test_chunk_switch_respects_preallocated_capacity():
    runner = SimpleNamespace(prefill_chunk_size=1024)
    ab.apply_chunk_mode(runner,"before",allocated_chunk_size=1024)
    assert runner.prefill_chunk_size == 512
    ab.apply_chunk_mode(runner,"after",allocated_chunk_size=1024)
    assert runner.prefill_chunk_size == 1024
    with pytest.raises(ValueError):
        ab.apply_chunk_mode(runner,"after",allocated_chunk_size=512)


def test_chunk_coverage_is_exact():
    ab.validate_chunk_coverage([1024,1024,1024,1024],4096,1024)
    ab.validate_chunk_coverage([512],512,1024)
    ab.validate_chunk_coverage([1024,1],1025,1024)
    with pytest.raises(ValueError):
        ab.validate_chunk_coverage([512]*8,4096,1024)


def test_state_gate_chunk_mode_preserves_environment():
    from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode
    runner = SimpleNamespace(prefill_chunk_size=1024)
    env = {"qsa":"page256","kernel":"keep"}
    for enabled,chunk in [("0",512),("1",1024),("0",512)]:
        apply_state_gate_mode(runner,"chunk1024",enabled,"qsa",environment=env)
        assert runner.prefill_chunk_size == chunk
        assert env == {"qsa":"page256","kernel":"keep"}
    apply_state_gate_mode(runner,"qsa-h256-page256","1","qsa",environment=env)
    assert env["qsa"] == "page256"
