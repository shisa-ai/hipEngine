import pytest

from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, build_parser
from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode
from scripts.qwen4exp_gdn_narrow_gate_support import expected_calls, validate_call


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_narrow_state_gate_keeps_parent_wave_norm_admission(enabled):
    environment = {"unrelated": "kept"}
    apply_state_gate_mode(None, "gdn-narrow", enabled,
                          "HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM", environment=environment)
    assert environment == {"unrelated": "kept", "HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM": "1"}
    other = {"unrelated": "kept"}
    _apply_mode("after" if enabled == "1" else "before", route_package="gdn-narrow",
                environment=other)
    assert other == environment
    assert build_parser().parse_args([
        "--model-root", "/unused", "--output", "/unused.json",
        "--route-package", "gdn-narrow", "--prefill-chunk-size", "4096",
    ]).route_package == "gdn-narrow"


def test_narrow_ab_mode_rejects_unknown():
    with pytest.raises(ValueError):
        _apply_mode("unknown", route_package="gdn-narrow", environment={})


@pytest.mark.parametrize("tokens,chunk,count", [
    (1, 4096, 0), (2, 4096, 36), (512, 512, 36), (1024, 512, 72),
    (4096, 1024, 144), (4096, 4096, 36), (4097, 4096, 36),
    (4098, 4096, 72), (17, 1, 0),
])
def test_current_ud_all_layer_counts(tokens, chunk, count):
    assert expected_calls(tokens, chunk) == count


@pytest.mark.parametrize("tokens,chunk", [(0, 4096), (512, 0), (512, 4097)])
def test_counts_fail_closed(tokens, chunk):
    with pytest.raises(ValueError):
        expected_calls(tokens, chunk)


def test_geometry():
    validate_call([1] * 9 + [4096, 16, 48, 128, 128], max_rows=4096)
    for shape in ([1, 16, 48, 128, 128], [4097, 16, 48, 128, 128],
                  [512, 8, 48, 128, 128]):
        with pytest.raises(ValueError):
            validate_call([1] * 9 + shape, max_rows=4096)
