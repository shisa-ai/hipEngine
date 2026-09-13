"""The journey baseline must not turn on an experimental owner."""

from scripts.qwen4exp_layer2_profile_gate import CANDIDATES, build_parser


def test_production_baseline_uses_named_profile_without_leaf_claim():
    spec = CANDIDATES["production_baseline"]
    assert spec.base_profile == "production"
    assert spec.environment == {}
    assert spec.classification == "diagnostic"
    assert spec.candidate_key is None
    assert spec.fallback_key is None
    assert not spec.compact_output


def test_parser_accepts_explicit_production_baseline():
    args = build_parser().parse_args([
        "--model-root", "/tmp/model", "--candidate", "production_baseline",
        "--decode-steps", "32", "--output", "/tmp/baseline.json",
    ])
    assert args.candidate == "production_baseline"
    assert args.decode_steps == 32


def test_repair_candidate_is_explicit_and_does_not_modify_baseline():
    spec = CANDIDATES["production_q8down_gdn_fallback"]
    assert spec.base_profile == "production"
    assert spec.environment == {
        "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN": "0",
        "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL": "0",
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "0",
    }
    assert CANDIDATES["production_baseline"].environment == {}
