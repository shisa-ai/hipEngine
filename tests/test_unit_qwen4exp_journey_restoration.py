from scripts.qwen4exp_layer2_profile_gate import CANDIDATES


def test_gdn_restoration_is_isolated_and_counted():
    candidate = CANDIDATES["production_gdn_restore"]
    assert candidate.base_profile == "production"
    assert candidate.classification == "T2"
    assert candidate.count_registered_dispatch
    assert candidate.candidate_key[-1] == "qwen4exp_gdn_tiled16_dpp_prefill"
    assert all(key.startswith("HIPENGINE_QWEN4_EXP_GDN_")
               for key in candidate.environment)
    assert candidate.environment["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] == "1"
    assert candidate.environment["HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL"] == "1"
    assert candidate.environment["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS"] == ""
    assert candidate.fallback_key[-1] == "qwen4exp_sigmoid_strict_prefill"


def test_multi_restoration_changes_only_the_gdn_variant():
    parent = CANDIDATES["production_gdn_restore"]
    candidate = CANDIDATES["production_gdn_multi_restore"]
    assert candidate.base_profile == "production"
    assert candidate.classification == "T2"
    assert candidate.count_registered_dispatch
    assert candidate.candidate_key[-1] == "qwen4exp_gdn_tiled16_multi_prefill"
    assert candidate.environment == {
        **parent.environment,
        "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT": "qwen4exp_gdn_tiled16_multi_prefill",
    }
