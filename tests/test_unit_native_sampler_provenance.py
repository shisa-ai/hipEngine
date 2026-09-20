"""Native sampler policy provenance is separate from model arithmetic profile."""
from types import SimpleNamespace

import pytest


def test_sorted_and_strict_declare_distinct_sampler_manifests():
    from hipengine.runtime.native_sampler import NativeSamplerWorkspace
    sorted_ws = NativeSamplerWorkspace(runtime=None, vocab_size=8, sampler_library=None)
    strict_ws = NativeSamplerWorkspace(runtime=None, vocab_size=8, sampler_library=None,
                                       full_vocab_algorithm="strict")
    production = sorted_ws.provenance
    strict = strict_ws.provenance
    assert production["sampler_execution_profile"] == "production"
    assert strict["sampler_execution_profile"] == "strict"
    assert production["manifest_sha256"] != strict["manifest_sha256"]
    filtered = production["selections"]["full_vocab_filtered"]
    assert filtered["selected_variant"] == "sorted_rows_i32"
    assert filtered["strict_fallback_variant"] == "top_p_temperature_rows_i32"
    assert strict["selections"]["full_vocab_filtered"]["selected_variant"] == "top_p_temperature_rows_i32"
    assert strict["selections"]["full_vocab_unfiltered"]["selected_variant"] == "temperature_rows_i32"
    production["selections"]["full_vocab_filtered"]["selected_variant"] = "bogus"
    assert sorted_ws.provenance["selections"]["full_vocab_filtered"]["selected_variant"] == "sorted_rows_i32"
    with pytest.raises(AttributeError):
        sorted_ws.full_vocab_algorithm = "strict"


def test_gguf_explicit_sampler_policy_is_not_model_profile_inference():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
    generator = object.__new__(Qwen35GGUFBringupGenerator)
    generator.execution_profile = "strict"
    generator.native_sampler_algorithm = "sorted"
    generator.backend = "hip_gfx1151"
    generator._configure_session(SimpleNamespace())
    provenance = generator.native_sampler_provenance
    assert provenance["model_execution_profile"] == "strict"
    assert provenance["sampler_execution_profile"] == "production"
    assert provenance["full_vocab_algorithm"] == "sorted"
    generator.native_sampler_algorithm = "strict"
    session = SimpleNamespace()
    generator._configure_session(session)
    assert session.native_sampler_algorithm == "strict"
    assert generator.native_sampler_provenance["sampler_execution_profile"] == "strict"


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_manifest_variants_are_registered_implementations(backend):
    from hipengine.runtime.native_sampler import native_sampler_provenance
    from hipengine.kernels.hip_gfx1100.sampling.sampler import register_sampler_kernels, sample_sorted_f32_rows_i32
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve
    register_sampler_kernels()
    register_gfx1151_kernels()
    for algorithm in ("sorted", "strict"):
        manifest = native_sampler_provenance(algorithm)
        for selection in manifest["selections"].values():
            for field in ("selected_variant", "strict_fallback_variant"):
                impl = resolve(backend=backend, layer="sampler", quant="f32", variant=selection[field])
                assert callable(impl)
                if selection[field] == "sorted_rows_i32":
                    assert impl is sample_sorted_f32_rows_i32


def test_gguf_debug_opt_out_reaches_workspace(monkeypatch):
    from contextlib import nullcontext
    import hipengine.runtime.qwen35_gguf_runner as module
    session = object.__new__(module.Qwen35GGUFResidentSession)
    session.native_sampler_algorithm = "strict"
    session.backend = "hip_gfx1151"
    session.runtime = object()
    session.runner = SimpleNamespace(target_arch="gfx1151", vocab_size=8)
    session._lm_head_library = object()
    session._native_sampler_workspace = None
    monkeypatch.setattr(module, "build_sampler", lambda **kwargs: object())
    monkeypatch.setattr(module, "hip_target_arch_environment", lambda arch: nullcontext())
    workspace = session._native_sampler()
    assert workspace.full_vocab_algorithm == "strict"
    assert session.native_sampler_provenance["sampler_execution_profile"] == "strict"
    assert session.native_sampler_provenance["manifest_sha256"] == workspace.provenance["manifest_sha256"]


def test_live_session_cannot_silently_change_declared_sampler():
    from hipengine.runtime.native_sampler import NativeSamplerWorkspace
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    session = object.__new__(Qwen35GGUFResidentSession)
    session.native_sampler_algorithm = "strict"
    session._native_sampler_workspace = NativeSamplerWorkspace(
        runtime=None, vocab_size=8, sampler_library=None)
    with pytest.raises(RuntimeError, match="sampler.*selection"):
        session._native_sampler()
