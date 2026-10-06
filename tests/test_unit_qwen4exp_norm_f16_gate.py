"""DP-5 qualification checks fail when engagement or arm boundaries are wrong."""
import subprocess
import sys

import pytest

from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, build_parser
from scripts.qwen4exp_norm_f16_gate_support import ENV, expected_calls, heldout_tokens, validate_producer_call


@pytest.mark.parametrize("tokens,chunk,calls", [(512,4096,0),(1024,4096,0),(4095,4096,0),
    (4096,4096,70),(4097,4096,70),(8192,4096,140),(4096,1024,0),(4096,512,0)])
def test_expected_engagement(tokens,chunk,calls):
    assert expected_calls(tokens,chunk)==calls


@pytest.mark.parametrize("tokens,chunk",[(0,4096),(4096,0),(-1,4096)])
def test_invalid_accounting(tokens,chunk):
    with pytest.raises(ValueError): expected_calls(tokens,chunk)


def test_arm_switch_changes_only_publication():
    env={"other":"kept"}
    for mode,value in (("before","0"),("after","1")):
        _apply_mode(mode,environment=env,route_package="norm-f16-publish")
        assert env=={"other":"kept",ENV:value}
    with pytest.raises(ValueError): _apply_mode("bad",environment=env,route_package="norm-f16-publish")
    assert "norm-f16-publish" in build_parser()._option_string_actions["--route-package"].choices


def test_call_boundaries_and_uniform_heldout():
    validate_producer_call((1,2,3,4,4096,4,2560,1e-6))
    for a in ((1,2,3,4,1024,4,2560,1e-6),(1,2,3,4,4096,2,2560,1e-6),(1,2,3,4,4096,4,2560)):
        with pytest.raises(ValueError): validate_producer_call(a)
    assert len(heldout_tokens([1,2,3]))==4096
    assert heldout_tokens(list(range(5000)))==list(range(4096))
    with pytest.raises(ValueError): heldout_tokens([])


def test_native_smoke_help_has_no_hip_side_effect():
    result=subprocess.run([sys.executable,"scripts/qwen4exp_norm_f16_native_smoke.py","--help"],
                          text=True,capture_output=True,check=True)
    assert "--compiler-version-file" in result.stdout
    assert "--case-id" in result.stdout
    assert "--paired-repeats" in result.stdout
