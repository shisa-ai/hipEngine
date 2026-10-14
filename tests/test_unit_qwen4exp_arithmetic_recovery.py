import os
from types import SimpleNamespace

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND, QWEN4_EXP_MODEL, register_qwen4_exp_gfx1151_profiles,
    PRODUCTION_ARITHMETIC_RECOVERY_FLAGS,
    PRODUCTION_Q8_QSA_RESTORED_FLAGS,
)
from scripts.qwen4exp_layer2_profile_gate import CONSERVATIVE_ARITHMETIC_FLAGS
from hipengine.runtime import qwen4_exp_runner
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels


def resolve(quant):
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=quant, profile=ExecutionProfile.PRODUCTION)


def test_ud_profile_recovers_arithmetic_but_keeps_exact_owners(monkeypatch):
    # The conservative gate arm also closes the tiled GDN recurrence, which
    # the production plan runs.
    assert PRODUCTION_ARITHMETIC_RECOVERY_FLAGS == tuple(
        flag for flag in CONSERVATIVE_ARITHMETIC_FLAGS
        if flag != "GDN_COLWARPS_PREFILL")
    monkeypatch.setattr(os, "environ", os.environ.copy())
    profile = resolve("gguf_ud_q4_k_xl")
    profile.binder(SimpleNamespace(), profile)
    # GR_WIDE is closed by the recovery pass and reopened by the F16 route
    # composition this quant's production plan carries.
    expected = {**PRODUCTION_Q8_QSA_RESTORED_FLAGS, "GR_WIDE": "1"}
    for flag in PRODUCTION_ARITHMETIC_RECOVERY_FLAGS:
        assert os.environ["HIPENGINE_QWEN4_EXP_" + flag] == expected.get(flag, "0")
    for flag in ("Q4_IU8_EXACT", "Q51_IU8_EXACT", "GDN_REGISTER_PREFILL",
                 "GDN_WAVE_NORM", "FORKB_GROUPED_DOWN", "Q4_DP4A64"):
        assert os.environ["HIPENGINE_QWEN4_EXP_" + flag] == "1"
    variants = {row["selected_variant"] for row in profile.manifest["selections"]}
    assert "qwen4exp_sigmoid_wave_norm_prefill" in variants
    assert not any("mmq" in variant or "dp4a" in variant or
                   "gdn_tiled" in variant or "gdn_columnwarps" in variant
                   for variant in variants)


def test_other_quant_profile_binding_is_unchanged(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    profile = resolve("gguf_q4_k_m")
    profile.binder(SimpleNamespace(), profile)
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN"] == "1"


def _tile16_declaration(profile):
    """Return the production manifest's declared owner for the tile16 scope."""
    rows = [
        row["selected_variant"]
        for row in profile.manifest["selections"]
        if row["layer"] == "gdn_recurrence_norm_gate"
        and row["scope"] == "prefill_rows_ge16_hk16_hv32_48_d128_layers27_47"
    ]
    assert len(rows) == 1, rows
    return rows[0]


def test_production_selects_the_multi_column_gdn_prefill_owner(monkeypatch):
    """The multi-column GDN prefill route supersedes the DPP route, and the
    manifest declares the route the binder dispatches.

    The live selection is the ``HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT`` env var, so
    the manifest row has to name the same variant the env names. An earlier version
    made the row conditional on the quant and claimed that branch was inert; it was
    not. The recovery pass only replaces rows for the quant that passes
    ``recovery=True``, so the OTHER quant's conditional row reached its manifest
    unchanged and declared ``qwen4exp_gdn_tiled16_prefill`` while the env dispatched
    the multi-column route underneath it - a gate on that profile recorded one
    manifest hash for two different kernels (1350 dispatches each).

    The row is now unconditional, which is safe precisely because the recovery pass
    deactivates tiled variants by NAME: for the recovery quant both candidate values
    map to the same exact owner, so its manifest is unchanged byte for byte.

    Pinned because that binding is a one-line change the recovery path would
    silently undo: the recovery pass deactivates tiled variants by NAME, so the
    multi-column route has to be listed there too or the recovery UD profile
    starts selecting a tiled owner it must not use - which is what happened on the
    first attempt at this promotion.

    The binding is also pinned to the colwarps gate it feeds. The runner reads the
    variant only inside ``if colwarps_prefill:``, and ``GDN_COLWARPS_PREFILL`` is in
    ``PRODUCTION_ARITHMETIC_RECOVERY_FLAGS``, so the UD quant closes that gate. A
    binding that named the multi-column route for UD therefore selected a variant
    that could never dispatch: it measured as inert, and a probe that wraps the
    registered callable observed zero dispatches with the predicate never reached.
    """
    from hipengine.kernels.registry import resolve as resolve_kernel

    monkeypatch.setattr(os, "environ", os.environ.copy())
    profile = resolve("gguf_ud_q4_k_xl")
    profile.binder(SimpleNamespace(), profile)

    # The UD profile keeps the colwarps gate open and names the row-split
    # recurrence, the composition's replacement for the multi-column body.
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"] == (
        "qwen4exp_gdn_rowsplit_prefill")
    assert resolve_kernel(
        backend="hip_gfx1151", layer="gdn_recurrence_norm_gate",
        quant="f32_state", variant="qwen4exp_gdn_rowsplit_prefill",
    ) is not None

    # The recovery profile's manifest still keeps its exact owner.
    recovery_gdn = [
        row["selected_variant"]
        for row in profile.manifest["selections"]
        if row["layer"] == "gdn_recurrence_norm_gate"
    ]
    assert "qwen4exp_gdn_tiled16_multi_prefill" not in recovery_gdn, recovery_gdn

    # The profile that keeps the column-warp suffix enabled is the one that
    # selects the multi-column route, so the binding is reachable.
    other = resolve("gguf_q4_k_m")
    other.binder(SimpleNamespace(), other)
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"] == (
        "qwen4exp_gdn_tiled16_multi_prefill")

    # ...and the manifest must declare that same variant. A gate records the
    # manifest hash as the candidate's provenance, so a declaration that disagrees
    # with the binding gives two different kernels one hash - which is exactly what
    # happened while this row was conditional on the quant.
    assert _tile16_declaration(other) == os.environ[
        "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"]
    # UD declares only its reachable serial scopes, not the disabled tile16
    # scope. The checked owner covers 2-4096 rows; wide remains above4096.
    recovery_scopes = {row["scope"]: row["selected_variant"]
                       for row in profile.manifest["selections"]
                       if row["layer"] == "gdn_recurrence_norm_gate"}
    assert recovery_scopes == {
        "prefill_rows_2_4096_serial_gdn_hk16_hv48_d128": "qwen4exp_sigmoid_wave_norm_narrow_prefill",
        "prefill_rows_gt4096_serial_gdn_hk16_hv48_d128": "qwen4exp_sigmoid_wave_norm_prefill",
    }

    # The superseded route stays registered as the strict fallback, so rollback is
    # a one-line revert rather than a rebuild.
    assert resolve_kernel(
        backend="hip_gfx1151", layer="gdn_recurrence_norm_gate",
        quant="f32_state", variant="qwen4exp_gdn_tiled16_dpp_prefill",
    ) is not None
