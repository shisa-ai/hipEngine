from __future__ import annotations

from hipengine.kernels.hip_gfx1100.smoke import plan_smoke_add_build, register_smoke_add_kernel
from hipengine.kernels.hip_gfx1100.smoke.smoke_add import smoke_add_f32
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_smoke_add_registers_lazy_wrapper_without_building() -> None:
    register_smoke_add_kernel()

    assert resolve(backend="hip_gfx1100", layer="smoke_add", quant="fp16") is smoke_add_f32


def test_smoke_add_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_smoke_add_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc smoke test version",
    )

    assert artifact.family == "smoke"
    assert artifact.profile.name == "baseline"
    assert artifact.output_path.name == "smoke_add.so"
    assert artifact.compiler_version == "hipcc smoke test version"
    assert artifact.flags == ()
    assert any(str(path).endswith("smoke_add.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()
