import pytest

from scripts.qwen4exp_dense_wide_padded_gate_support import (
    call_in_scope, expected_calls, heldout_tokens,
)
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, build_parser
from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_tail_gate_modes_do_not_mutate_named_arithmetic(enabled):
    environment = {"unrelated": "kept"}
    from hipengine.runtime.gguf_linear import DENSE_WIDE_TAIL_VARIANT_ENV
    from scripts.qwen4exp_dense_wide_padded_gate_support import PARENT
    expected = {"unrelated": "kept", DENSE_WIDE_TAIL_VARIANT_ENV: PARENT}
    apply_state_gate_mode(None, "dense-wide-padded", enabled, "unused", environment=environment)
    assert environment == expected
    _apply_mode("after" if enabled == "1" else "before", route_package="dense-wide-padded",
                environment=environment)
    assert environment == expected
    assert build_parser().parse_args([
        "--model-root", "/unused", "--output", "/unused.json",
        "--route-package", "dense-wide-padded", "--prefill-chunk-size", "4096",
    ]).route_package == "dense-wide-padded"


@pytest.mark.parametrize("tokens,chunk,count", [
    (1,4096,0), (511,4096,0), (512,4096,70), (4096,4096,70),
    (4097,4096,70), (4608,4096,140), (4096,1024,280), (1024,512,140),
])
def test_tail_count_contract(tokens, chunk, count):
    assert expected_calls(tokens, chunk) == count


def test_tail_scope_and_heldout_padding():
    assert call_in_scope([1,2,3,512,10240,320])
    assert not call_in_scope([1,2,3,511,10240,320])
    assert not call_in_scope([1,2,3,512,320,10240])
    assert heldout_tokens([4,5,6]) == ([4,5,6] * 171)[:512]
    assert heldout_tokens(list(range(600))) == list(range(600))
    with pytest.raises(ValueError):
        heldout_tokens([])


def test_tail_modes_and_counts_fail_closed():
    with pytest.raises(ValueError):
        _apply_mode("invalid", route_package="dense-wide-padded", environment={})
    for tokens, chunk in ((0,4096), (512,511), (512,4097)):
        with pytest.raises(ValueError):
            expected_calls(tokens, chunk)
    with pytest.raises(ValueError):
        call_in_scope([1,2,3])
