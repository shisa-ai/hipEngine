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
    assert PRODUCTION_ARITHMETIC_RECOVERY_FLAGS == CONSERVATIVE_ARITHMETIC_FLAGS
    monkeypatch.setattr(os, "environ", os.environ.copy())
    profile = resolve("gguf_ud_q4_k_xl")
    profile.binder(SimpleNamespace(), profile)
    for flag in CONSERVATIVE_ARITHMETIC_FLAGS:
        assert os.environ["HIPENGINE_QWEN4_EXP_" + flag] == (
            PRODUCTION_Q8_QSA_RESTORED_FLAGS.get(flag, "0"))
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


def test_production_selects_the_multi_column_gdn_prefill_owner(monkeypatch):
    """The multi-column GDN prefill route supersedes the DPP route.

    The live selection is the ``HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT`` env var, not
    the manifest row: the production plan is built with
    ``dpp=quant == "gguf_ud_q4_k_xl"`` and ``recovery=quant == "gguf_ud_q4_k_xl"``,
    the SAME condition, so the recovery pass replaces every ``dpp=True`` tiled
    selection with its exact owner before the manifest is ever observed. The
    ``dpp`` branch is therefore inert and only the env binding is live.

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

    # The UD recovery profile closes the colwarps gate, so it must not name a
    # tile16 route at all - naming one is unreachable by construction.
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"] == ""

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

    # The superseded route stays registered as the strict fallback, so rollback is
    # a one-line revert rather than a rebuild.
    assert resolve_kernel(
        backend="hip_gfx1151", layer="gdn_recurrence_norm_gate",
        quant="f32_state", variant="qwen4exp_gdn_tiled16_dpp_prefill",
    ) is not None
