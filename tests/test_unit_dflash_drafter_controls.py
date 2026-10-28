"""DFlash drafter choices and dense fallback dispatch need no HIP runtime."""
from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.speculative import dflash_drafter as drafter


@pytest.mark.parametrize("value,expected", [
    (None, True), ("", True), ("wmma", True), (" TRUE ", True),
    ("naive", False), ("0", False), ("off", False),
])
def test_dense_default_and_reference_choice(monkeypatch, value, expected):
    name = "HIPENGINE_DFLASH_DRAFTER_DENSE"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert drafter._drafter_dense_use_wmma() is expected


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("off", False), ("naive", False),
    ("on", True), ("fused", True), (" TRUE ", True),
])
def test_benchmark_add_norm_diagnostic_choice(monkeypatch, value, expected):
    name = "HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert drafter._drafter_dense_use_add_rmsnorm() is expected


@pytest.mark.parametrize("name,resolver", [
    ("HIPENGINE_DFLASH_DRAFTER_DENSE", drafter._drafter_dense_use_wmma),
    ("HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM", drafter._drafter_dense_use_add_rmsnorm),
])
def test_invalid_choice_fails_by_name_without_loading_runtime(monkeypatch, name, resolver):
    monkeypatch.setenv(name, "typo")
    with pytest.raises(ValueError, match=name):
        resolver()


@pytest.mark.parametrize("output", ["bf16", "f32"])
@pytest.mark.parametrize("in_features", [15, 16, 17, 32])
@pytest.mark.parametrize("choice", [None, "naive"])
def test_dense_geometry_fallback_preserves_raw_launch_arguments(monkeypatch, output, in_features, choice):
    name = "HIPENGINE_DFLASH_DRAFTER_DENSE"
    if choice is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, choice)
    launches = []
    monkeypatch.setattr(drafter, "_launch_dense", lambda *args: launches.append(("naive", args)))
    monkeypatch.setattr(drafter, f"dflash_dense_bf16_to_{output}_wmma",
                        lambda *args, **kwargs: launches.append(("wmma", args, kwargs)))
    library, runtime = object(), object()
    getattr(drafter, f"dflash_dense_bf16_to_{output}")(
        0x100, 0x200, 0x300, 3, in_features, 19,
        threads=64, stream=7, library=library, runtime=runtime,
    )
    use_wmma = choice is None and in_features % 16 == 0
    assert len(launches) == 1
    raw_args = (0x100, 0x200, 0x300, 3, in_features, 19)
    if use_wmma:
        assert launches == [("wmma", raw_args, {"stream": 7, "library": library, "runtime": runtime})]
    else:
        symbol = getattr(drafter, f"_SYMBOL_DENSE_BF16_TO_{output.upper()}")
        assert launches == [("naive", (symbol, *raw_args, 64, 7, library, runtime))]
