"""Runtime-rejection contract for mapped-host Laguna argmax output."""

from __future__ import annotations

import inspect

import pytest

from hipengine.core.hip import HipRuntime
from hipengine.runtime import laguna_gguf_runner as runner
from hipengine.runtime.laguna_gguf_runner import (
    LagunaEagerScratch,
    LagunaGGUFResidentSession,
)


def test_mapped_argmax_runtime_owner_is_removed_but_host_mapping_abi_remains() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert not hasattr(gfx1100, "LAGUNA_MAPPED_ARGMAX_OUTPUT")
    assert not hasattr(gfx1151, "LAGUNA_MAPPED_ARGMAX_OUTPUT")
    assert not hasattr(runner, "resolve_laguna_mapped_argmax_output")
    assert not hasattr(runner, "LagunaMappedArgmaxOutput")
    assert "mapped_argmax_output" not in inspect.signature(
        LagunaEagerScratch
    ).parameters
    assert "mapped_argmax_output" not in inspect.signature(
        LagunaEagerScratch.allocate
    ).parameters
    assert "use_mapped_argmax_output" not in inspect.signature(
        LagunaGGUFResidentSession
    ).parameters
    assert hasattr(HipRuntime, "host_register")
    assert hasattr(HipRuntime, "host_get_device_pointer")
    assert hasattr(HipRuntime, "host_unregister")


def test_mapped_argmax_benchmark_opt_in_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-mapped-argmax-output"],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()


def test_scalar_sampling_sites_keep_separate_device_fallback() -> None:
    for method in (
        LagunaGGUFResidentSession._project_rows_last,
        LagunaGGUFResidentSession._project_and_sample,
    ):
        source = inspect.getsource(method)
        assert source.count("_read_laguna_argmax_result") == 1

    helper_source = inspect.getsource(runner._read_laguna_argmax_result)
    assert "mapped_argmax_output" not in helper_source
    assert "_read_i64" in helper_source
    assert "_read_f32" in helper_source
