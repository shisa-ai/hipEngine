"""MoE decode-graph admission is capability-controlled or explicitly set.

The process env override is retired. Harnesses and fixtures now pass the explicit
runner/generator control, so their eager isolation stays real instead of turning
inert. The cache's own fail-closed eager fallback is unchanged.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.generation import qwen4_exp_gguf
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.runtime import qwen4_exp_runner
from hipengine.runtime.moe_graph import MoeGraphCache
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpGGUFResidentModelRunner,
    qwen4_exp_moe_graph_enabled,
)

OVERRIDE = "HIPENGINE_QWEN4_EXP_MOE_GRAPH"


def test_capability_decides_admission_when_no_explicit_control_is_given(monkeypatch):
    monkeypatch.delenv(OVERRIDE, raising=False)
    assert qwen4_exp_moe_graph_enabled("hip_gfx1151") is True
    assert qwen4_exp_moe_graph_enabled("hip_gfx1100") is False


def test_retired_override_no_longer_changes_admission(monkeypatch):
    """The env value is ignored in both directions, on both backends."""

    for value in ("0", "1", "", "false"):
        monkeypatch.setenv(OVERRIDE, value)
        assert qwen4_exp_moe_graph_enabled("hip_gfx1151") is True
        assert qwen4_exp_moe_graph_enabled("hip_gfx1100") is False


@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_control_outranks_the_backend_capability(explicit):
    for backend in ("hip_gfx1151", "hip_gfx1100"):
        assert qwen4_exp_moe_graph_enabled(backend, explicit) is explicit


def test_runner_module_no_longer_reads_the_retired_override():
    """No read of the retired switch remains; a prose mention is not a read."""

    for line in inspect.getsource(qwen4_exp_runner).splitlines():
        if OVERRIDE in line:
            assert "environ" not in line and "getenv" not in line, line


class _StubRunner:
    max_sequence_length = 8
    prefill_chunk_size = 4

    def close(self) -> None:
        pass


def test_runner_and_generator_expose_the_explicit_control():
    runner = inspect.signature(Qwen4ExpGGUFResidentModelRunner.__init__).parameters
    generator = inspect.signature(Qwen4ExpGGUFTextGenerator.__init__).parameters
    assert runner["moe_graph_enabled"].default is None
    assert generator["moe_graph_enabled"].default is None


def test_generator_forwards_the_explicit_control_to_resident_runners(monkeypatch):
    """Both construction sites (own runner and c2 pool) take the same control."""

    calls: list[dict] = []

    def fake_runner(resident, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        qwen4_exp_gguf, "Qwen4ExpGGUFResidentModelRunner", fake_runner
    )
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        runner=_StubRunner(),
        moe_graph_enabled=False,
    )
    try:
        generator._new_resident_runner(
            SimpleNamespace(), max_sequence_length=8, prefill_chunk_size=4
        )
    finally:
        generator.close()
    assert calls and calls[0]["moe_graph_enabled"] is False


def test_disabled_cache_runs_the_eager_fallback():
    calls: list[int] = []
    cache = MoeGraphCache(SimpleNamespace(), enabled=False)
    outcome = cache.run(
        "layer-0",
        eager=lambda stream: calls.append(stream),
        out_ptr=4096,
        out_nbytes=64,
        stream=0,
    )
    assert outcome == "eager"
    assert calls == [0]
    assert cache.stats["eager"] == 1
    assert cache.stats["capture"] == 0


def test_set_enabled_switches_arms_and_recreates_a_missing_capture_stream():
    created: list[bool] = []

    def stream_create(*, nonblocking):
        created.append(nonblocking)
        return 7

    runtime = SimpleNamespace(stream_create=stream_create)
    cache = MoeGraphCache(runtime, enabled=False)
    assert cache.enabled is False
    cache.set_enabled(True)
    assert cache.enabled is True
    assert created == [True]
    cache.set_enabled(False)
    assert cache.enabled is False
    # Disabling keeps the capture stream; only a missing one is recreated.
    assert created == [True]
