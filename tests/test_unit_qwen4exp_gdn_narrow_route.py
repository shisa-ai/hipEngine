"""Bounded named-profile selection for the checked-index GDN leaf."""
import warnings
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile, clear_runtime_profile_registry_for_tests, resolve_runtime_profile,
)
from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner

ENV = "HIPENGINE_QWEN4_EXP_GDN_PREFILL_VARIANT"
WIDE = "qwen4exp_sigmoid_wave_norm_prefill"
NARROW = "qwen4exp_sigmoid_wave_norm_narrow_prefill"
SCOPE = "prefill_rows_2_4096_serial_gdn_hk16_hv48_d128"
OUTSIDE = "prefill_rows_gt4096_serial_gdn_hk16_hv48_d128"


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    # Binders write many environment keys; restore the complete original map.
    import os
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", dict(os.environ))
        patch.delenv(ENV, raising=False)
        clear_runtime_profile_registry_for_tests()
        yield
        clear_runtime_profile_registry_for_tests()


def resolve(profile, quant):
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    return resolve_runtime_profile(model="qwen4_exp_gguf", backend="hip_gfx1151",
                                   quant=quant, profile=profile)


def test_ud_manifest_binder_and_outside_parent():
    import os
    result = resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl")
    rows = [r for r in result.manifest["selections"] if r["layer"] == "gdn_recurrence_norm_gate"]
    assert {r["scope"] for r in rows} == {SCOPE, OUTSIDE}
    bounded = next(r for r in rows if r["scope"] == SCOPE)
    outside = next(r for r in rows if r["scope"] == OUTSIDE)
    assert bounded["selected_variant"] == NARROW
    assert bounded["strict_fallback_variant"] == "qwen4exp_sigmoid_strict_prefill"
    assert bounded["evidence_artifact"].endswith("gdn-narrow-index/README.md")
    assert outside["selected_variant"] == WIDE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result.binder(SimpleNamespace(), result)
    assert os.environ[ENV] == NARROW
    assert os.environ["HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256"] == result.manifest_sha256


@pytest.mark.parametrize("profile,quant", [
    (ExecutionProfile.STRICT, "gguf_ud_q4_k_xl"),
    (ExecutionProfile.STRICT, "gguf_q4_k_m"),
    (ExecutionProfile.PRODUCTION, "gguf_q4_k_m"),
])
def test_other_profiles_do_not_admit_narrow(profile, quant, monkeypatch):
    import os
    monkeypatch.setenv(ENV, NARROW)
    result = resolve(profile, quant)
    assert all(r["selected_variant"] != NARROW for r in result.manifest["selections"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result.binder(SimpleNamespace(), result)
    assert os.environ[ENV] == WIDE


def test_bound_profile_reaches_runtime_and_does_not_leak_to_strict(monkeypatch):
    import os
    production = resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl")
    key = KernelKey("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                    "qwen4exp_sigmoid_register_prefill")
    kwargs = dict(rows=512, num_k_heads=16, num_v_heads=48, head_dim=128)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        production.binder(SimpleNamespace(), production)
    assert runner._qwen4_exp_gdn_wave_norm_key(key, **kwargs).variant == NARROW
    monkeypatch.setenv(ENV, WIDE)  # mismatched diagnostic caller value is overwritten
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        production.binder(SimpleNamespace(), production)
    assert os.environ[ENV] == NARROW
    strict = resolve(ExecutionProfile.STRICT, "gguf_ud_q4_k_xl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        strict.binder(SimpleNamespace(), strict)
    assert runner._qwen4_exp_gdn_wave_norm_key(key, **kwargs) == key
    assert all(r["selected_variant"] == "qwen4exp_sigmoid_strict_prefill"
               for r in strict.manifest["selections"] if r["layer"] == "gdn_recurrence_norm_gate")


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


def test_explicit_parent_rollback_changes_manifest(monkeypatch):
    original = resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl")
    clear_runtime_profile_registry_for_tests()
    monkeypatch.setenv(ENV, WIDE)
    rollback = resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl")
    assert original.manifest_sha256 != rollback.manifest_sha256
    row = next(r for r in rollback.manifest["selections"] if r["scope"] == SCOPE)
    assert row["selected_variant"] == WIDE
    assert row["strict_fallback_variant"] == "qwen4exp_sigmoid_strict_prefill"


def test_unknown_named_variant_rejected(monkeypatch):
    monkeypatch.setenv(ENV, "unregistered")
    with pytest.raises(ValueError, match="GDN_PREFILL_VARIANT"):
        resolve(ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl")


@pytest.mark.parametrize("rows,hk,hv,d,expected", [
    (2,16,48,128,NARROW), (17,16,48,128,NARROW),
    (512,16,48,128,NARROW), (1024,16,48,128,NARROW),
    (4096,16,48,128,NARROW), (4097,16,48,128,WIDE),
    (2**31,16,48,128,WIDE), (1,16,48,128,WIDE),
    (0,16,48,128,WIDE), (512,8,48,128,WIDE),
    (512,16,32,128,WIDE), (512,16,48,64,WIDE),
])
def test_runtime_shape_boundary(rows,hk,hv,d,expected,monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM", "1")
    monkeypatch.setenv(ENV, NARROW)
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    key = KernelKey("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                    "qwen4exp_sigmoid_register_prefill")
    assert runner._qwen4_exp_gdn_wave_norm_key(
        key, rows=rows, num_k_heads=hk, num_v_heads=hv, head_dim=d).variant == expected


def test_missing_narrow_falls_back_to_wide_then_register(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM", "1")
    monkeypatch.setenv(ENV, NARROW)
    key = KernelKey("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                    "qwen4exp_sigmoid_register_prefill")
    kwargs = dict(rows=512, num_k_heads=16, num_v_heads=48, head_dim=128)
    monkeypatch.setattr(runner, "is_registered", lambda key: key.variant == WIDE)
    assert runner._qwen4_exp_gdn_wave_norm_key(key, **kwargs).variant == WIDE
    monkeypatch.setattr(runner, "is_registered", lambda key: False)
    assert runner._qwen4_exp_gdn_wave_norm_key(key, **kwargs) == key
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM", "0")
    assert runner._qwen4_exp_gdn_wave_norm_key(key, **kwargs) == key
    other = KernelKey(key.backend,key.layer,key.quant,"strict")
    assert runner._qwen4_exp_gdn_wave_norm_key(other, **kwargs) == other
