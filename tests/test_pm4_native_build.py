from __future__ import annotations

from hipengine.core.pm4.native_build import plan_pm4_native_build


def test_pm4_native_build_plan_is_lazy_targeted_and_hsa_linked(tmp_path) -> None:
    artifact = plan_pm4_native_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc pm4 unit-test version",
        target_arch="gfx1100",
    )

    assert artifact.family == "pm4-native"
    assert artifact.profile.name == "baseline"
    assert artifact.output_path.name == "pm4_native.so"
    assert artifact.target_arch == "gfx1100"
    assert "-std=c++17" in artifact.flags
    assert "-lhsa-runtime64" in artifact.flags
    assert any(str(path).endswith("native.cpp") for path in artifact.sources)
    assert not artifact.cache_dir.exists()
