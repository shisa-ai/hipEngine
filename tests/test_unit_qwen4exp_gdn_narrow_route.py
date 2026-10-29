"""Bounded named-profile selection for the checked-index GDN leaf."""
import warnings
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile, clear_runtime_profile_registry_for_tests, resolve_runtime_profile,
)
from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions
from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner

WIDE = "qwen4exp_sigmoid_wave_norm_prefill"
NARROW = "qwen4exp_sigmoid_wave_norm_narrow_prefill"
PARENT = "qwen4exp_sigmoid_register_prefill"
SCOPE = "prefill_rows_2_4096_serial_gdn_hk16_hv48_d128"
OUTSIDE = "prefill_rows_gt4096_serial_gdn_hk16_hv48_d128"
RETIRED_ENVS = (
    "HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM",
    "HIPENGINE_QWEN4_EXP_GDN_PREFILL_VARIANT",
)


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    # Binders write many environment keys; restore the complete original map.
    import os
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", dict(os.environ))
        for name in RETIRED_ENVS:
            patch.delenv(name, raising=False)
        clear_runtime_profile_registry_for_tests()
        yield
        clear_runtime_profile_registry_for_tests()


def resolve(profile, quant, **kwargs):
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles(**kwargs)
    return resolve_runtime_profile(model="qwen4_exp_gguf", backend="hip_gfx1151",
                                   quant=quant, profile=profile)


def bind(result):
    generator = SimpleNamespace(runner=SimpleNamespace())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result.binder(generator, result)
    return generator


def key():
    return KernelKey("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state", PARENT)


def select(variant, rows=512, hk=16, hv=48, d=128):
    return runner._qwen4_exp_gdn_prefill_key(
        key(), rows=rows, num_k_heads=hk, num_v_heads=hv, head_dim=d,
        variant=variant)


def test_ud_manifest_binder_and_outside_parent():
    result = resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl")
    rows = [r for r in result.manifest["selections"] if r["layer"] == "gdn_recurrence_norm_gate"]
    assert {r["scope"] for r in rows} == {SCOPE, OUTSIDE, "decode_c1_d128"}
    bounded = next(r for r in rows if r["scope"] == SCOPE)
    outside = next(r for r in rows if r["scope"] == OUTSIDE)
    assert bounded["selected_variant"] == NARROW
    assert bounded["strict_fallback_variant"] == "qwen4exp_sigmoid_strict_prefill"
    assert bounded["evidence_artifact"].endswith("gdn-narrow-index/README.md")
    assert outside["selected_variant"] == WIDE
    generator = bind(result)
    # The plan's own selection reaches the runner; nothing is published through
    # a process-wide switch.
    assert generator.gdn_prefill_variant == NARROW
    assert generator.runner.gdn_prefill_variant == NARROW
    import os
    for name in RETIRED_ENVS:
        assert name not in os.environ
    assert os.environ["HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256"] == result.manifest_sha256


@pytest.mark.parametrize("profile,quant,expected,binder,fell_back", [
    (ExecutionProfile.STRICT, "gguf_ud_q4_k_xl", None, "_strict_binder", False),
    (ExecutionProfile.STRICT, "gguf_q4_k_m", None, "_strict_binder", False),
    # The Q4_K_M production plan fails its qualification envelope and resolves
    # through the strict binder, so the executed scope publishes the parent
    # even though the requested profile label is production.
    (ExecutionProfile.PRODUCTION, "gguf_q4_k_m", None, "_strict_binder", True),
])
def test_other_profiles_do_not_admit_narrow(profile, quant, expected, binder, fell_back, monkeypatch):
    import os
    # A caller-set retired switch is inert: neither profile reads it.
    monkeypatch.setenv(RETIRED_ENVS[0], "1")
    monkeypatch.setenv(RETIRED_ENVS[1], NARROW)
    result = resolve(profile, quant)
    assert all(r["selected_variant"] != NARROW for r in result.manifest["selections"])
    assert result.binder.__name__ == binder
    assert result.fell_back_to_strict is fell_back
    generator = bind(result)
    assert generator.gdn_prefill_variant == expected
    assert generator.runner.gdn_prefill_variant == expected


def test_bound_profile_reaches_runtime_and_does_not_leak_to_strict():
    production = resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl")
    assert select(bind(production).runner.gdn_prefill_variant).variant == NARROW
    strict = resolve(ExecutionProfile.STRICT, "gguf_ud_q4_k_xl")
    assert select(bind(strict).runner.gdn_prefill_variant) == key()
    assert all(r["selected_variant"] == (
                   "qwen4exp_sigmoid_strict" if r["scope"] == "decode_c1_d128"
                   else "qwen4exp_sigmoid_strict_prefill")
               for r in strict.manifest["selections"]
               if r["layer"] == "gdn_recurrence_norm_gate")


def test_explicit_parent_rollback_changes_manifest():
    original = resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl")
    clear_runtime_profile_registry_for_tests()
    rollback = resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl",
                       gdn_prefill_variant=WIDE)
    assert original.manifest_sha256 != rollback.manifest_sha256
    row = next(r for r in rollback.manifest["selections"] if r["scope"] == SCOPE)
    assert row["selected_variant"] == WIDE
    assert row["strict_fallback_variant"] == "qwen4exp_sigmoid_strict_prefill"
    # The rolled-back plan publishes the wide owner, so the dispatch hop does
    # not need a second, process-wide override.
    assert bind(rollback).runner.gdn_prefill_variant == WIDE


def test_unknown_named_variant_rejected():
    with pytest.raises(ValueError, match="wide or narrow wave-norm"):
        resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl",
                gdn_prefill_variant="unregistered")


def test_invalid_explicit_request_after_registration_fails():
    # The idempotent return must not swallow a value that names no leaf.
    resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl")
    with pytest.raises(ValueError, match="wide or narrow wave-norm"):
        register_qwen4_exp_gfx1151_profiles(gdn_prefill_variant="unregistered")


def test_conflicting_explicit_request_after_registration_fails():
    resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl")
    with pytest.raises(RuntimeError, match="registered Qwen4Exp UD production plan"):
        register_qwen4_exp_gfx1151_profiles(gdn_prefill_variant=WIDE)


def test_matching_and_default_registration_stay_idempotent():
    resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl")
    assert register_qwen4_exp_gfx1151_profiles() is False
    assert register_qwen4_exp_gfx1151_profiles(gdn_prefill_variant=NARROW) is False


def test_rollback_registration_is_fixed_and_matching_requests_repeat():
    clear_runtime_profile_registry_for_tests()
    resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl",
            gdn_prefill_variant=WIDE)
    assert register_qwen4_exp_gfx1151_profiles(gdn_prefill_variant=WIDE) is False
    with pytest.raises(RuntimeError, match="registered Qwen4Exp UD production plan"):
        register_qwen4_exp_gfx1151_profiles(gdn_prefill_variant=NARROW)


@pytest.mark.parametrize("rows,hk,hv,d,expected", [
    (2,16,48,128,NARROW), (17,16,48,128,NARROW),
    (512,16,48,128,NARROW), (1024,16,48,128,NARROW),
    (4096,16,48,128,NARROW), (4097,16,48,128,WIDE),
    (2**31,16,48,128,WIDE), (1,16,48,128,WIDE),
    (0,16,48,128,WIDE), (512,8,48,128,WIDE),
    (512,16,32,128,WIDE), (512,16,48,64,WIDE),
])
def test_runtime_shape_boundary(rows,hk,hv,d,expected,monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    assert select(NARROW, rows=rows, hk=hk, hv=hv, d=d).variant == expected


def test_wide_selection_is_not_row_bounded(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    for rows in (0, 1, 2, 4096, 4097, 2**31):
        assert select(WIDE, rows=rows).variant == WIDE


def test_missing_narrow_falls_back_to_wide_then_register(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: key.variant == WIDE)
    assert select(NARROW).variant == WIDE
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert select(NARROW) == key()
    assert select(WIDE) == key()
    other = KernelKey(key().backend, key().layer, key().quant, "strict")
    assert runner._qwen4_exp_gdn_prefill_key(other, variant=NARROW) == other


def test_selector_ignores_the_retired_environment(monkeypatch):
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.setenv(RETIRED_ENVS[0], "1")
    monkeypatch.setenv(RETIRED_ENVS[1], NARROW)
    # The retired switches cannot select anything: the explicit value decides.
    assert select(None) == key()
    assert select(WIDE).variant == WIDE
    monkeypatch.delenv(RETIRED_ENVS[1])
    assert select(NARROW).variant == NARROW
    assert select("unregistered") == key()


def test_unbound_runner_keeps_the_parent():
    assert runner.Qwen4ExpGGUFResidentModelRunner.gdn_prefill_variant is None
    assert select(None) == key()


def test_runner_scope_restores_the_published_variant():
    owner = SimpleNamespace(gdn_prefill_variant=NARROW)
    with runner.gdn_prefill_variant(owner, WIDE) as prior:
        assert prior == NARROW
        assert owner.gdn_prefill_variant == WIDE
    assert owner.gdn_prefill_variant == NARROW


def test_runner_scope_binds_the_parent_and_leaves_unset_alone():
    owner = SimpleNamespace(gdn_prefill_variant=NARROW)
    with runner.gdn_prefill_variant(owner, None):
        assert owner.gdn_prefill_variant is None
    assert owner.gdn_prefill_variant == NARROW
    with runner.gdn_prefill_variant(owner):
        assert owner.gdn_prefill_variant == NARROW
    assert owner.gdn_prefill_variant == NARROW


def test_runner_scope_restores_after_an_exception():
    owner = SimpleNamespace(gdn_prefill_variant=NARROW)
    with pytest.raises(RuntimeError, match="boom"):
        with runner.gdn_prefill_variant(owner, None):
            raise RuntimeError("boom")
    assert owner.gdn_prefill_variant == NARROW


def test_runner_scope_is_per_runner():
    first = SimpleNamespace(gdn_prefill_variant=NARROW)
    second = SimpleNamespace(gdn_prefill_variant=NARROW)
    with runner.gdn_prefill_variant(first, WIDE):
        assert second.gdn_prefill_variant == NARROW
        with runner.gdn_prefill_variant(second, None):
            assert first.gdn_prefill_variant == WIDE
            assert second.gdn_prefill_variant is None
        assert second.gdn_prefill_variant == NARROW
    assert first.gdn_prefill_variant == NARROW


def test_lazy_runner_inherits_the_published_variant():
    generator = SimpleNamespace(gdn_prefill_variant=NARROW)
    lazy = _inherit_lane_decisions(generator, SimpleNamespace())
    assert lazy.gdn_prefill_variant == NARROW
    unbound = _inherit_lane_decisions(SimpleNamespace(), SimpleNamespace())
    assert unbound.gdn_prefill_variant is None


def test_layer_forwards_the_variant_to_the_mixer(monkeypatch):
    """The layer's only mixer call site carries the runner's decision."""
    seen = []
    runtime = object()

    def gr_read(*args, **kwargs):
        return SimpleNamespace(normalized=None, gate=None,
                               mixed=SimpleNamespace(ptr=11),
                               inject_logits=SimpleNamespace(ptr=12))

    def mixer(*args, **kwargs):
        seen.append(kwargs.get("prefill_variant", "absent"))
        return SimpleNamespace(ptr=13)

    def moe(*args, **kwargs):
        return SimpleNamespace(output=SimpleNamespace(ptr=14))

    monkeypatch.setattr(runner, "run_qwen4_exp_gr_read", gr_read)
    monkeypatch.setattr(runner, "run_qwen4_exp_gdn_token_mixer", mixer)
    monkeypatch.setattr(runner, "run_qwen4_exp_moe", moe)
    monkeypatch.setattr(runner, "_qwen4_exp_gr_pending_write", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_qwen4_exp_gr_defer_write", lambda *a, **k: False)
    monkeypatch.setattr(runner, "qwen4_exp_gr_write_bf16_f32", lambda *a, **k: None)
    monkeypatch.setattr(runner, "qwen4_exp_gr_write_bf16_bf16", lambda *a, **k: None)
    output = SimpleNamespace(ptr=15)
    scratch = SimpleNamespace(
        closed=False, runtime=runtime, attention_gr=SimpleNamespace(),
        gdn=SimpleNamespace(), after_attention=SimpleNamespace(ptr=16),
        ffn_gr=SimpleNamespace(), moe=SimpleNamespace(), output=output)
    weights = SimpleNamespace(
        attention_gr=SimpleNamespace(norm_weight_ptr=1, down=2, up=3, inject=4),
        mixer=SimpleNamespace(projections={}, conv_weight_ptr=5, dt_bias_ptr=6,
                              a_ptr=7, norm_weight_ptr=8),
        ffn_gr=SimpleNamespace(norm_weight_ptr=9, down=10, up=11, inject=12),
        moe=SimpleNamespace())
    kwargs = dict(conv_state_ptr=0, recurrent_state_ptr=0, scratch=scratch,
                  rows=512, branches=1, hidden=2, low_rank=2, num_k_heads=16,
                  num_v_heads=48, head_dim=128, conv_kernel=4, ffn=8,
                  experts=1, top_k=1, runtime=runtime)
    assert runner.run_qwen4_exp_gdn_layer(100, weights, **kwargs,
                                          prefill_variant=NARROW) is output
    assert runner.run_qwen4_exp_gdn_layer(100, weights, **kwargs) is output
    assert seen == [NARROW, None]


def test_native_smoke_help_does_not_touch_hip():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "scripts/qwen4exp_gdn_narrow_native_smoke.py", "--help"],
                            text=True, capture_output=True, check=True)
    assert "--compiler-version-file" in result.stdout
    assert "--model-root" in result.stdout


def test_native_manifest_record_is_json_safe_and_hash_preserving():
    import json
    from hipengine.execution_profiles import manifest_sha256
    from scripts.qwen4exp_gdn_narrow_native_smoke import profile_record
    production = resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl")
    strict = resolve(ExecutionProfile.STRICT, "gguf_ud_q4_k_xl")
    record = json.loads(json.dumps(profile_record(production, strict)))
    assert manifest_sha256(record["manifest"]) == production.manifest_sha256
    assert manifest_sha256(record["strict_manifest"]) == strict.manifest_sha256
