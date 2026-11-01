import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode, route_scope

WIDE = "qwen4exp_sigmoid_wave_norm_prefill"
NARROW = "qwen4exp_sigmoid_wave_norm_narrow_prefill"


def test_parent_only(monkeypatch):
    key = KernelKey("hip_gfx1151","gdn_recurrence_norm_gate","f32_state",
                    "qwen4exp_sigmoid_register_prefill")
    select = runner._qwen4_exp_gdn_prefill_key
    monkeypatch.setattr(runner,"is_registered",lambda key:True)
    assert select(key) == key
    assert select(key, variant=None) == key
    assert select(key, variant=WIDE).variant == WIDE
    other=KernelKey(key.backend,key.layer,key.quant,"strict")
    assert select(other, variant=WIDE) == other
    monkeypatch.setattr(runner,"is_registered",lambda key:False)
    assert select(key, variant=WIDE) == key


def test_retired_switch_is_not_written(monkeypatch):
    # The wave-norm admission is request-local runner state now, so neither the
    # campaign arm nor the state gate may write a process-wide switch.
    for package in ("gdn-wave-norm", "gdn-narrow"):
        for mode in ("before", "after"):
            env={"other":"keep"}
            _apply_mode(mode,route_package=package,environment=env)
            assert env=={"other":"keep"}, (package, mode, env)


def test_route_scope_binds_the_runner_variant():
    owner = type("Owner", (), {"gdn_prefill_variant": NARROW})()
    env = {}
    with route_scope("gdn-wave-norm", "after", owner, environment=env):
        assert owner.gdn_prefill_variant == WIDE
    assert owner.gdn_prefill_variant == NARROW
    with route_scope("gdn-wave-norm", "before", owner, environment=env):
        assert owner.gdn_prefill_variant is None
    assert owner.gdn_prefill_variant == NARROW
    with route_scope("gdn-narrow", "after", owner, environment=env):
        assert owner.gdn_prefill_variant == NARROW
    with route_scope("gdn-narrow", "before", owner, environment=env):
        assert owner.gdn_prefill_variant == WIDE
    assert owner.gdn_prefill_variant == NARROW


@pytest.mark.parametrize("package", ["gdn-wave-norm", "gdn-narrow"])
@pytest.mark.parametrize("mode", ["before", "after"])
@pytest.mark.parametrize("previous", [False, True])
def test_route_scope_closes_the_tiled_gdn_branch_for_both_arms(
        package, mode, previous):
    # The counted serial leaf is the tiled branch's fallback: production
    # publishes the row-split recurrence on every GDN layer, so an arm that
    # leaves the tiled branch open counts zero serial calls and measures that
    # owner. The gate is request-local runner state, so the arm binds it on the
    # owner it measures and restores the published value.
    owner = type("Owner", (), {"gdn_prefill_variant": NARROW})()
    owner.gdn_colwarps_prefill = previous
    with route_scope(package, mode, owner):
        assert owner.gdn_colwarps_prefill is False
    assert owner.gdn_colwarps_prefill is previous


def _arm_variant(package, mode):
    if package == "gdn-narrow":
        return NARROW if mode == "after" else WIDE
    return WIDE if mode == "after" else None


def test_route_scope_leaves_the_tiled_gdn_branch_alone_for_other_packages():
    owner = type("Owner", (), {
        "qsa_prefill_head_mode": "parent",
        "gdn_colwarps_prefill": True,
    })()
    with route_scope("qsa-head-pair", "after", owner):
        assert owner.gdn_colwarps_prefill is True
    assert owner.gdn_colwarps_prefill is True


@pytest.mark.parametrize("package", ["gdn-wave-norm", "gdn-narrow"])
@pytest.mark.parametrize("mode", ["before", "after"])
def test_building_a_gdn_route_scope_binds_nothing(package, mode):
    # The campaign builds one scope per arm and enters it with ``with``, so a
    # scope that mutates on construction would leave the arm bound before the
    # measured run and never restore it if the caller never entered.
    owner = type("Owner", (), {"gdn_prefill_variant": NARROW})()
    owner.gdn_colwarps_prefill = True
    scope = route_scope(package, mode, owner)
    assert owner.gdn_colwarps_prefill is True
    assert owner.gdn_prefill_variant == NARROW
    with scope:
        assert owner.gdn_colwarps_prefill is False
        assert owner.gdn_prefill_variant == _arm_variant(package, mode)
    assert owner.gdn_colwarps_prefill is True
    assert owner.gdn_prefill_variant == NARROW


def test_nested_gdn_route_scopes_restore_in_order():
    owner = type("Owner", (), {"gdn_prefill_variant": NARROW})()
    owner.gdn_colwarps_prefill = True
    with route_scope("gdn-narrow", "after", owner):
        assert owner.gdn_colwarps_prefill is False
        with route_scope("gdn-wave-norm", "after", owner):
            assert owner.gdn_colwarps_prefill is False
            assert owner.gdn_prefill_variant == WIDE
        # The inner scope restores the outer arm, not the profile's value.
        assert owner.gdn_colwarps_prefill is False
        assert owner.gdn_prefill_variant == NARROW
    assert owner.gdn_colwarps_prefill is True
    assert owner.gdn_prefill_variant == NARROW


def test_gdn_owner_probe_scope_binds_nothing_until_entered():
    # The probe binds its arm around setup as well as the loop, so a failure in
    # resolve/malloc must still unwind both the variant and the tiled branch.
    from scripts.qwen4exp_gdn_owner_probe import serial_probe_scope

    owner = type("Owner", (), {"gdn_prefill_variant": NARROW})()
    owner.gdn_colwarps_prefill = True
    scope = serial_probe_scope(owner, narrow_index=False)
    assert owner.gdn_colwarps_prefill is True
    assert owner.gdn_prefill_variant == NARROW
    with scope:
        assert owner.gdn_colwarps_prefill is False
        assert owner.gdn_prefill_variant is None
    assert owner.gdn_colwarps_prefill is True
    assert owner.gdn_prefill_variant == NARROW


@pytest.mark.parametrize("narrow_index", [False, True])
def test_gdn_owner_probe_scope_restores_after_a_setup_failure(narrow_index):
    from scripts.qwen4exp_gdn_owner_probe import serial_probe_scope

    owner = type("Owner", (), {"gdn_prefill_variant": NARROW})()
    owner.gdn_colwarps_prefill = True
    with pytest.raises(RuntimeError, match="setup failed"):
        with serial_probe_scope(owner, narrow_index=narrow_index):
            raise RuntimeError("setup failed")
    assert owner.gdn_colwarps_prefill is True
    assert owner.gdn_prefill_variant == NARROW


@pytest.mark.parametrize("package", ["gdn-wave-norm", "gdn-narrow"])
def test_route_scope_restores_the_tiled_gdn_branch_after_an_exception(package):
    owner = type("Owner", (), {"gdn_prefill_variant": NARROW})()
    owner.gdn_colwarps_prefill = True
    with pytest.raises(RuntimeError):
        with route_scope(package, "after", owner):
            raise RuntimeError("arm failed")
    assert owner.gdn_colwarps_prefill is True
    assert owner.gdn_prefill_variant == NARROW


@pytest.mark.parametrize("package", ["gdn-wave-norm", "gdn-narrow"])
def test_route_scope_rejects_a_gdn_arm_without_its_runner(package):
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope(package, "after")


@pytest.mark.parametrize("tokens,expected",[(1,0),(512,36),(1024,36),(1025,36),(1026,72),(4096,144)])
def test_serial_arm_counts_every_gdn_layer(tokens,expected):
    # Both arms close the tiled branch, so all 36 GDN layers run the serial leaf.
    from scripts.qwen4exp_halo_box_campaign_ab import gdn_wave_norm_expected_calls
    assert gdn_wave_norm_expected_calls(tokens,1024)==expected
