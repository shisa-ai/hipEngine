"""Bounded padded HC-tail admission must agree with the named manifest."""
import os
import warnings
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile, clear_runtime_profile_registry_for_tests, resolve_runtime_profile,
)
from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear

ENV = "HIPENGINE_QWEN4_EXP_DENSE_WIDE_TAIL_VARIANT"
PARENT = "dense_wide256_f16in_f32_f32_out"
PADDED = "dense_wide256_padded_f16in_f32_f32_out"
SCOPE = "prefill_rows_512_4096_k10240_n320_dense_wide_layers13_47"


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", dict(os.environ))
        patch.delenv(ENV, raising=False)
        clear_runtime_profile_registry_for_tests()
        yield
        clear_runtime_profile_registry_for_tests()


def profile(kind=ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl"):
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    return resolve_runtime_profile(model="qwen4_exp_gguf", backend="hip_gfx1151",
                                   quant=quant, profile=kind)


def bind(result):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result.binder(SimpleNamespace(), result)


def test_ud_manifest_and_binder_select_bounded_leaf():
    result = profile()
    row = next(r for r in result.manifest["selections"] if r["scope"] == SCOPE)
    assert row["selected_variant"] == PADDED
    assert row["strict_fallback_variant"] == "coltile8_rowbatch4_f32_f32_out"
    assert row["evidence_artifact"].endswith("dense-wide-padded-gate/README.md")
    bind(result)
    assert os.environ[ENV] == PADDED
    assert os.environ["HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256"] == result.manifest_sha256


@pytest.mark.parametrize("kind,quant", [
    (ExecutionProfile.STRICT, "gguf_ud_q4_k_xl"),
    (ExecutionProfile.STRICT, "gguf_q4_k_m"),
    (ExecutionProfile.PRODUCTION, "gguf_q4_k_m"),
])
def test_other_profiles_do_not_admit_padded(kind, quant, monkeypatch):
    monkeypatch.setenv(ENV, PADDED)
    result = profile(kind, quant)
    assert all(r["selected_variant"] != PADDED for r in result.manifest["selections"])
    bind(result)
    assert os.environ[ENV] == PARENT


def test_rollback_is_manifested_and_rebind_does_not_leak(monkeypatch):
    result = profile()
    bind(result)
    monkeypatch.setenv(ENV, PARENT)
    bind(result)
    assert os.environ[ENV] == PADDED
    clear_runtime_profile_registry_for_tests()
    monkeypatch.setenv(ENV, PARENT)
    rollback = profile()
    assert rollback.manifest_sha256 != result.manifest_sha256
    assert next(r for r in rollback.manifest["selections"] if r["scope"] == SCOPE)["selected_variant"] == PARENT
    bind(rollback)
    assert os.environ[ENV] == PARENT
    bind(profile(ExecutionProfile.STRICT))
    assert os.environ[ENV] == PARENT


def test_unknown_selection_fails_closed(monkeypatch):
    monkeypatch.setenv(ENV, "unknown")
    with pytest.raises(ValueError, match="DENSE_WIDE_TAIL_VARIANT"):
        profile()


@pytest.mark.parametrize("backend,quant,target,rows,k,n,expected", [
    ("hip_gfx1151", "gguf_q8_0", PARENT, 512, 10240, 320, PADDED),
    ("hip_gfx1151", "gguf_q8_0", PARENT, 1024, 10240, 320, PADDED),
    ("hip_gfx1151", "gguf_q8_0", PARENT, 4096, 10240, 320, PADDED),
    ("hip_gfx1151", "gguf_q8_0", PARENT, 511, 10240, 320, PARENT),
    ("hip_gfx1151", "gguf_q8_0", PARENT, 4097, 10240, 320, PARENT),
    ("hip_gfx1151", "gguf_q8_0", PARENT, 512, 2560, 10240, PARENT),
    ("hip_gfx1151", "gguf_q8_0", PARENT, 512, 10240, 321, PARENT),
    ("hip_gfx1100", "gguf_q8_0", PARENT, 512, 10240, 320, PARENT),
    ("hip_gfx1151", "gguf_q4_k", PARENT, 512, 10240, 320, PARENT),
    ("hip_gfx1151", "gguf_q8_0", "dense_wide128x128_f16in_f32_f32_out", 512, 10240, 320,
     "dense_wide128x128_f16in_f32_f32_out"),
])
def test_capability_shape_and_parent_boundary(backend, quant, target, rows, k, n, expected, monkeypatch):
    monkeypatch.setenv(ENV, PADDED)
    key = KernelKey(backend, "linear", quant, target)
    assert linear._dense_wide_f16_tail_key(key, rows=rows, in_features=k, out_features=n).variant == expected


def test_native_smoke_help_does_not_touch_hip():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "scripts/qwen4exp_dense_wide_tail_native_smoke.py", "--help"],
                            text=True, capture_output=True, check=True)
    assert "--compiler-version-file" in result.stdout
    assert "--model-root" in result.stdout


class Weight:
    def allocation(self, name):
        assert name == "raw"
        return SimpleNamespace(tensor=SimpleNamespace(ptr=0x9000))


@pytest.mark.parametrize("workspace_bytes,enabled,registered,expected", [
    (512 * 10240 * 2, "1", True, PADDED),
    (512 * 10240 * 2, "1", False, PARENT),
    (512 * 10240 * 2 - 1, "1", True, None),
    (512 * 10240 * 2, "0", True, None),
])
def test_launch_stage_abi_library_and_fail_closed(workspace_bytes, enabled, registered, expected, monkeypatch):
    profile()
    monkeypatch.setenv(ENV, PADDED)
    monkeypatch.setenv(linear.WIDE_F16_ACTIVATION_ENV, enabled)
    monkeypatch.setattr(linear, "is_registered", lambda key: registered or key.variant == PARENT)
    monkeypatch.setattr(linear, "_ensure_linear_kernel_registered", lambda key: None)
    seen, casts = [], []
    monkeypatch.setattr(linear, "resolve", lambda **key: lambda *a, **kw: seen.append((key, a, kw)))
    monkeypatch.setattr("hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide.f32_to_f16",
                        lambda *a, **kw: casts.append((a, kw)))
    libraries = {f"gguf_q8_0:{PARENT}": object(), f"gguf_q8_0:{PADDED}": object()}
    with linear.wide_f16_activation_session(workspace_ptr=0x5000, workspace_nbytes=workspace_bytes):
        swapped = linear._launch_dense_wide_f16_staged(
            None, Weight(), 0x1000, 0x2000, 512, 10240, 320, {"stream": 7},
            backend="hip_gfx1151", quant="gguf_q8_0", layer="linear",
            variant="dense_wide256_f32_f32_out", runtime=object(), libraries=libraries)
    assert swapped == (expected is not None)
    if expected is None:
        assert seen == casts == []
    else:
        assert len(seen) == len(casts) == 1
        key, args, kwargs = seen[0]
        assert key["variant"] == expected
        assert args == (0x5000, 0x9000, 0x2000, 512, 10240, 320)
        assert casts[0][0] == (0x1000, 0x5000, 512 * 10240)
        assert kwargs["stream"] == casts[0][1]["stream"] == 7
        assert kwargs["library"] is casts[0][1]["library"] is libraries[f"gguf_q8_0:{expected}"]
    assert linear.wide_f16_activation_workspace() is None
