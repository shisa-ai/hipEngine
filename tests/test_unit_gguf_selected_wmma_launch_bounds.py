"""CPU-only compiler-plan contracts for the selected-WMMA tuning control."""

from importlib import import_module

import pytest


_FAMILIES = (
    "gguf_q4_k_selected_prefill",
    "gguf_q4_k_t16_selected_prefill",
    "gguf_k_selected_prefill",
    "gguf_k_t16_selected_prefill",
    "gguf_q5_k_q8_1_selected_prefill",
)
_ENV = "HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS"
_MACRO = "-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS="


@pytest.mark.parametrize("family", _FAMILIES)
def test_launch_bounds_are_validated_and_key_the_compiler_plan(monkeypatch, family):
    module = import_module(f"hipengine.kernels.hip_gfx1100.quant.{family}")
    plan = getattr(module, f"plan_{family}_build")
    build = getattr(module, f"build_{family}")
    monkeypatch.delenv(_ENV, raising=False)
    default = plan(compiler_version="test-compiler")
    assert "-mcumode" in default.flags
    assert not any(flag.startswith(_MACRO) for flag in default.flags)
    keys = {default.cache_key}

    for blocks in (1, 2, 4, 8):
        monkeypatch.setenv(_ENV, str(blocks))
        artifact = plan(compiler_version="test-compiler")
        assert _MACRO + str(blocks) in artifact.flags
        assert artifact.sources == default.sources
        assert artifact.cache_key not in keys
        keys.add(artifact.cache_key)
        dry_run = build(dry_run=True, compiler_version="test-compiler")
        assert dry_run.cache_key == artifact.cache_key
        assert dry_run.flags == artifact.flags
        assert dry_run.output_path == artifact.output_path

    monkeypatch.delenv(_ENV)
    assert plan(compiler_version="test-compiler").cache_key == default.cache_key


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("value", ["0", "3", "16", "invalid"])
def test_launch_bounds_refuse_invalid_values_before_compilation(monkeypatch, family, value):
    module = import_module(f"hipengine.kernels.hip_gfx1100.quant.{family}")
    monkeypatch.setenv(_ENV, value)
    with pytest.raises(ValueError):
        getattr(module, f"plan_{family}_build")(compiler_version="test-compiler")
    with pytest.raises(ValueError):
        getattr(module, f"build_{family}")(dry_run=True, compiler_version="test-compiler")
