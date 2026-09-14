from types import SimpleNamespace

from scripts.qwen4exp_journey_localize import ARMS, OUTLIER, PREFIX, clear_arm_graphs, strict_family_overrides, gdn_isolation_overrides
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites


def test_localization_disables_only_named_families():
    assert ARMS["bound"] == {}
    assert ARMS["decode_strict"] == {PREFIX + "Q4_DP4A64": "0"}
    for name in ("gdn_strict", "moe_strict", "dense_q8_gr_strict", "decode_strict"):
        assert ARMS[name]
        assert set(ARMS[name].values()) == {"0"}
        for key in ARMS[name]:
            assert ARMS["all_numerics_strict"][key] == "0"


def test_selected_down_control_is_a_single_numerical_override():
    assert ARMS["q8_selected_down_strict"] == {
        PREFIX + "Q8_0_SELECTED_WMMA_DOWN": "0",
    }
    assert ARMS["base_plus_q8_selected_down_strict"] == {
        **ARMS["all_numerics_strict"], PREFIX + "Q8_0_SELECTED_WMMA_DOWN": "0",
    }


def test_localization_outlier_is_a_real_heldout_not_a_performance_prompt():
    rows = _load_suites(DEFAULT_PROMPTS)
    matches = [r for r in rows if r["id"] == OUTLIER]
    assert len(matches) == 1
    assert matches[0]["category"] == "general_ja"


def test_each_arm_drains_and_replaces_both_graph_caches(monkeypatch):
    from hipengine.runtime import moe_graph

    calls = []
    runtime = SimpleNamespace(device_synchronize=lambda: calls.append("sync"))
    old = [SimpleNamespace(enabled=enabled, close=lambda: calls.append("close"))
           for enabled in (True, False)]
    runner = SimpleNamespace(runtime=runtime, moe_graph_cache=old[0], layer_graph_cache=old[1])
    monkeypatch.setattr(moe_graph, "MoeGraphCache",
                        lambda rt, enabled: SimpleNamespace(runtime=rt, enabled=enabled))
    clear_arm_graphs(runner)
    assert calls == ["sync", "close", "close"]
    assert runner.moe_graph_cache is not old[0]
    assert runner.layer_graph_cache is not old[1]
    assert runner.moe_graph_cache.enabled is True
    assert runner.layer_graph_cache.enabled is False


def test_remaining_family_flags_are_taken_from_strict_binding():
    flags = {PREFIX + key: "0" for key in (
        "Q4_PAIR_PREFILL", "Q8_WAVE_SCALE", "GDN_REGISTER_PREFILL",
    )}
    groups = strict_family_overrides(flags)
    for group in groups.values():
        assert group[PREFIX + "Q4_DP4A64"] == "0"
    assert groups["base_plus_moe_flags"][PREFIX + "Q4_PAIR_PREFILL"] == "0"
    assert PREFIX + "GDN_REGISTER_PREFILL" not in groups["base_plus_moe_flags"]
    assert groups["base_plus_q8_gr_flags"][PREFIX + "Q8_WAVE_SCALE"] == "0"
    assert groups["base_plus_gdn_flags"][PREFIX + "GDN_REGISTER_PREFILL"] == "0"


def test_gdn_isolation_keeps_other_families_strict():
    strict = {PREFIX + name: "0" for name in (
        "GDN_REGISTER_PREFILL", "GDN_PEER_PREFILL",
        "GDN_COLWARPS_PREFILL", "Q8_0_SELECTED_WMMA_DOWN",
    )}
    production = {key: "1" for key in strict}
    production["UNRELATED"] = "ignored"
    arms = gdn_isolation_overrides(strict, production)
    for values in arms.values():
        assert values[PREFIX + "Q8_0_SELECTED_WMMA_DOWN"] == "0"
        assert values[PREFIX + "GDN_REGISTER_PREFILL"] == "1"
        assert "UNRELATED" not in values
    assert arms["strict_plus_gdn"][PREFIX + "GDN_COLWARPS_PREFILL"] == "1"
    assert arms["strict_plus_gdn_serial"][PREFIX + "GDN_COLWARPS_PREFILL"] == "0"
    assert arms["strict_plus_gdn_serial"][PREFIX + "GDN_PEER_PREFILL"] == "0"
    assert arms["strict_plus_gdn_multi"][PREFIX + "GDN_TILE16_VARIANT"] == (
        "qwen4exp_gdn_tiled16_multi_prefill"
    )
