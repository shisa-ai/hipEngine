from types import SimpleNamespace

import pytest

from scripts.qwen4exp_journey_localize import (
    ARMS, OUTLIER, PREFIX, clear_arm_graphs, gdn_isolation_overrides,
    phase_peer_scope, serial_gdn, strict_family_overrides,
)
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites


def test_h256_arm_bindings_preserve_strict_and_production_choices():
    from scripts.qwen4exp_journey_localize import h256_arm_mode

    for arm in ("production_conservative", "all_flags_strict",
                "strict_plus_gdn", "strict_plus_matrix", "strict_plus_dense",
                "q8down_off_without_qsa"):
        assert h256_arm_mode(arm) == "parent"
    for arm in ("bound", "strict_plus_qsa", "gdn_flags_strict",
                "q8down_off_without_matrix"):
        assert h256_arm_mode(arm) is None


def test_split_h256_scope_restores_on_failure(monkeypatch):
    from scripts import qwen4exp_journey_localize as module

    def fail(ids):
        assert runner.qsa_h256_prefill_mode == "parent"
        raise RuntimeError("prefill failed")

    runner = SimpleNamespace(qsa_h256_prefill_mode="page256",
                             reset=lambda: None, prefill=fail)
    monkeypatch.setattr(module, "clear_arm_graphs", lambda runner: None)
    monkeypatch.setattr(module, "set_flags", lambda flags: None)
    with pytest.raises(RuntimeError, match="prefill failed"):
        module.split_trajectory(runner, [1], [], {}, {}, h256_mode="parent")
    assert runner.qsa_h256_prefill_mode == "page256"


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
        "GDN_REGISTER_PREFILL", "GDN_COLWARPS_PREFILL",
        "Q8_0_SELECTED_WMMA_DOWN",
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
    # The peer owner's layer scope is request-local runner state, so the serial
    # arm binds it on the runner instead of carrying a process value; the
    # columnwarp value is what tells the arm loop to do that.
    assert serial_gdn(arms["strict_plus_gdn_serial"]) is True
    assert serial_gdn(arms["strict_plus_gdn"]) is False
    assert arms["strict_plus_gdn_multi"][PREFIX + "GDN_TILE16_VARIANT"] == (
        "qwen4exp_gdn_tiled16_multi_prefill"
    )


def _split_runner(monkeypatch, module, seen, published="35-47"):
    import numpy as np

    result = SimpleNamespace(logits=np.array([1.0], dtype=np.float32))

    def prefill(ids):
        seen.append(("prefill", runner.gdn_peer_prefill_layers))
        return result

    def step(token):
        seen.append(("step", runner.gdn_peer_prefill_layers))
        return result

    runner = SimpleNamespace(
        gdn_peer_prefill_layers=published,
        reset=lambda: None,
        prefill=prefill,
        step=step,
    )
    monkeypatch.setattr(module, "clear_arm_graphs", lambda runner: None)
    monkeypatch.setattr(module, "set_flags", lambda flags: None)
    return runner


def test_split_trajectory_binds_the_peer_owner_off_for_a_strict_prefill_phase(monkeypatch):
    """A strict prefill phase must not inherit the production peer scope.

    The split arms set the prefill and decode flag sets independently, so the
    phase-local binding is the only thing that reaches the strict prefill.
    """

    from scripts import qwen4exp_journey_localize as module

    seen = []
    runner = _split_runner(monkeypatch, module, seen)
    module.split_trajectory(
        runner, [3], [5],
        {PREFIX + "GDN_COLWARPS_PREFILL": "0"},
        {PREFIX + "GDN_COLWARPS_PREFILL": "1"},
    )
    # Strict prefill runs the serial owners; the production decode phase leaves
    # the published scope alone, and decode never selects the owner anyway.
    assert seen == [("prefill", None), ("step", "35-47")]
    assert runner.gdn_peer_prefill_layers == "35-47"


def test_split_trajectory_keeps_the_published_scope_for_a_production_prefill_phase(monkeypatch):
    from scripts import qwen4exp_journey_localize as module

    seen = []
    runner = _split_runner(monkeypatch, module, seen)
    module.split_trajectory(
        runner, [3], [5],
        {PREFIX + "GDN_COLWARPS_PREFILL": "1"},
        {PREFIX + "GDN_COLWARPS_PREFILL": "0"},
    )
    assert seen == [("prefill", "35-47"), ("step", "35-47")]
    assert runner.gdn_peer_prefill_layers == "35-47"


def test_split_trajectory_restores_the_published_scope_after_a_failed_prefill(monkeypatch):
    from scripts import qwen4exp_journey_localize as module

    def boom(ids):
        raise RuntimeError("boom")

    runner = SimpleNamespace(
        gdn_peer_prefill_layers="35-47", reset=lambda: None, prefill=boom,
        step=lambda token: None,
    )
    monkeypatch.setattr(module, "clear_arm_graphs", lambda runner: None)
    monkeypatch.setattr(module, "set_flags", lambda flags: None)
    with pytest.raises(RuntimeError, match="boom"):
        module.split_trajectory(
            runner, [3], [5],
            {PREFIX + "GDN_COLWARPS_PREFILL": "0"},
            {PREFIX + "GDN_COLWARPS_PREFILL": "1"},
        )
    assert runner.gdn_peer_prefill_layers == "35-47"


def test_phase_scope_leaves_a_production_phase_unbound():
    runner = SimpleNamespace(gdn_peer_prefill_layers="35-47")
    with phase_peer_scope(runner, {PREFIX + "GDN_COLWARPS_PREFILL": "1"}):
        assert runner.gdn_peer_prefill_layers == "35-47"
    with phase_peer_scope(runner, {PREFIX + "GDN_COLWARPS_PREFILL": "0"}):
        assert runner.gdn_peer_prefill_layers is None
    assert runner.gdn_peer_prefill_layers == "35-47"


def test_split_trajectory_reconstructs_prefix_then_switches_without_reset(monkeypatch):
    import numpy as np
    from scripts import qwen4exp_journey_localize as module

    calls = []
    result = SimpleNamespace(logits=np.array([1, 2], dtype=np.float32))
    runner = SimpleNamespace(
        reset=lambda: calls.append("reset"),
        prefill=lambda ids: (calls.append(("prefill", ids)) or result),
        step=lambda token: (calls.append(("step", token)) or result),
    )
    monkeypatch.setattr(module, "clear_arm_graphs", lambda runner: calls.append("clear"))
    monkeypatch.setattr(module, "set_flags", lambda flags: calls.append(flags))
    rows = module.split_trajectory(runner, [3, 4], [5, 6], {"pre": "1"}, {"dec": "1"})
    assert calls == ["clear", {"pre": "1"}, "reset", ("prefill", [3, 4]),
                     "clear", {"dec": "1"}, ("step", 5), ("step", 6)]
    assert len(rows) == 3
    assert not np.shares_memory(rows[0], result.logits)


@pytest.mark.parametrize("arm,expected", [
    ("bound", True), ("all_flags_strict", False),
    ("base_plus_moe_flags", False), ("q8down_off_without_matrix", False),
    ("strict_plus_matrix", True), ("strict_plus_gdn", False),
    ("strict_plus_qsa", False), ("gdn_flags_strict", True),
])
def test_pair_arm_choice_survives_environment_retirement(arm, expected):
    from scripts.qwen4exp_journey_localize import pair_arm_enabled
    assert pair_arm_enabled(arm, True) is expected
    assert pair_arm_enabled(arm, False) is False


@pytest.mark.parametrize("enabled", [False, True])
def test_split_pair_scope_follows_prefill_and_restores(monkeypatch, enabled):
    import numpy as np
    from scripts import qwen4exp_journey_localize as module

    seen = []
    result = SimpleNamespace(logits=np.array([1.0], dtype=np.float32))
    owner = SimpleNamespace(q4_pair_prefill=True, reset=lambda: None)
    owner.prefill = lambda ids: (seen.append(owner.q4_pair_prefill) or result)
    owner.step = lambda token: (seen.append(owner.q4_pair_prefill) or result)
    monkeypatch.setattr(module, "clear_arm_graphs", lambda runner: None)
    monkeypatch.setattr(module, "set_flags", lambda flags: None)
    module.split_trajectory(owner, [3], [5], {}, {}, pair_prefill=enabled)
    assert seen == [enabled, True]
    assert owner.q4_pair_prefill is True
