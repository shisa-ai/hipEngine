"""Runtime-rejection contract for Laguna argmax pair readback."""

from __future__ import annotations

import inspect

import pytest

from hipengine.runtime import laguna_gguf_runner as runner
from hipengine.runtime.laguna_gguf_runner import (
    LagunaEagerScratch,
    LagunaGGUFResidentSession,
)


def test_argmax_pair_runtime_owner_is_removed() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert not hasattr(gfx1100, "LAGUNA_ARGMAX_PAIR_READBACK")
    assert not hasattr(gfx1151, "LAGUNA_ARGMAX_PAIR_READBACK")
    assert not hasattr(runner, "resolve_laguna_argmax_pair_readback")
    assert not hasattr(runner, "_read_argmax_pair")
    assert "argmax_result" not in inspect.signature(LagunaEagerScratch).parameters
    assert "argmax_pair_readback" not in inspect.signature(
        LagunaEagerScratch.allocate
    ).parameters
    assert "use_argmax_pair_readback" not in inspect.signature(
        LagunaGGUFResidentSession
    ).parameters


def test_argmax_pair_benchmark_opt_in_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-argmax-pair-readback"],
    )
    with pytest.raises(SystemExit):
        benchmark._parse_args()


def test_scalar_sampling_sites_keep_separate_read_fallback() -> None:
    for method in (
        LagunaGGUFResidentSession._project_rows_last,
        LagunaGGUFResidentSession._project_and_sample,
    ):
        source = inspect.getsource(method)
        assert "_read_argmax_pair" not in source
        assert source.count("_read_laguna_argmax_result") == 1

    helper_source = inspect.getsource(runner._read_laguna_argmax_result)
    assert "_read_argmax_pair" not in helper_source
    assert "_read_i64" in helper_source
    assert "_read_f32" in helper_source
