from scripts.qwen4exp_journey_localize import ARMS, OUTLIER, PREFIX
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites


def test_localization_disables_only_named_families():
    assert ARMS["bound"] == {}
    assert ARMS["decode_strict"] == {PREFIX + "Q4_DP4A64": "0"}
    for name in ("gdn_strict", "moe_strict", "dense_q8_gr_strict", "decode_strict"):
        assert ARMS[name]
        assert set(ARMS[name].values()) == {"0"}
        for key in ARMS[name]:
            assert ARMS["all_numerics_strict"][key] == "0"


def test_localization_outlier_is_a_real_heldout_not_a_performance_prompt():
    rows = _load_suites(DEFAULT_PROMPTS)
    matches = [r for r in rows if r["id"] == OUTLIER]
    assert len(matches) == 1
    assert matches[0]["category"] == "general_ja"
