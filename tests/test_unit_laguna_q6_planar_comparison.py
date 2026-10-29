"""Host contracts for Q6 interleaved-permute versus planar comparison."""

import argparse
import sys
from types import SimpleNamespace

import pytest

from hipengine.runtime import laguna_gguf_runner as runner
from scripts import laguna_q6_planar_ab as comparison
from scripts import laguna_long_context_profile as profiler


@pytest.mark.parametrize("mode", comparison._MODES)
def test_layout_comparison_constructs_each_real_session_without_later_prefetch(monkeypatch, mode):
    sessions = []
    config = SimpleNamespace(context_length=4096)
    monkeypatch.setattr(comparison, "get_hip_runtime", lambda: object())
    monkeypatch.setattr(comparison, "memory_stats", lambda: {})
    monkeypatch.setattr(comparison, "_compiler_version", lambda path: "synthetic")
    monkeypatch.setattr(runner, "GGUFReader", lambda path: SimpleNamespace(info=object()))
    monkeypatch.setattr(runner, "laguna_gguf_config_from_metadata", lambda info: config)

    def stop(*args, **kwargs):
        raise RuntimeError("layout comparison reached kernel planning")

    monkeypatch.setattr(runner, "resolve_laguna_eager_kernel_plan", stop)

    def construct(*args, **kwargs):
        session = object.__new__(runner.LagunaGGUFResidentSession)
        sessions.append(session)
        return session.__init__(*args, **kwargs)

    monkeypatch.setattr(comparison, "LagunaGGUFResidentSession", construct)
    with pytest.raises(RuntimeError, match="layout comparison reached kernel planning"):
        comparison._run_mode(mode, args=SimpleNamespace(compiler_version_file=None, require_cached_build=True), tokens=[1] * 512)
    session, = sessions
    assert session.q6_qmicro_planar is (mode == "planar_candidate")
    assert session.q6_qmicro_permute is (mode == "current_permute")
    assert not session.q6_wmma_prefetch_weight
    assert not session.q6_wmma_prefetch_activation
    assert not session.q6_precomputed_activation_sums
    assert session._closed


@pytest.mark.parametrize("mode", comparison._MODES)
@pytest.mark.parametrize("raises", [False, True])
def test_layout_comparison_records_fixed_policy_and_closes_owner(monkeypatch, mode, raises):
    constructors = []
    events = []
    runtime = SimpleNamespace(device_synchronize=lambda: events.append("sync"))
    owner = SimpleNamespace(prefill=lambda tokens: events.append("warm"), close=lambda: events.append("close"))
    monkeypatch.setattr(comparison, "get_hip_runtime", lambda: runtime)
    monkeypatch.setattr(comparison, "memory_stats", lambda: {"tracked": 0})
    monkeypatch.setattr(comparison, "_compiler_version", lambda path: "synthetic")
    ticks = iter((10.0, 11.0))
    monkeypatch.setattr(comparison.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(comparison, "LagunaGGUFResidentSession", lambda *a, **kw: constructors.append(kw) or owner)

    def run_once(selected, tokens):
        assert selected is owner and len(tokens) == 512
        if raises:
            raise RuntimeError("synthetic prefill failure")
        return {"tok_s": 1.0, "next_token": 2}

    monkeypatch.setattr(comparison, "_run_once", run_once)
    args = SimpleNamespace(compiler_version_file=None, require_cached_build=True, repetitions=1)
    if raises:
        with pytest.raises(RuntimeError, match="synthetic prefill failure"):
            comparison._run_mode(mode, args=args, tokens=[1] * 512)
    else:
        record = comparison._run_mode(mode, args=args, tokens=[1] * 512)
        expected = {
            "q6_wmma_prefetch_weight": False,
            "q6_wmma_prefetch_activation": False,
            "q6_precomputed_activation_sums": False,
        }
        assert record["comparison_policy"] == expected
        assert {key: constructors[0][key] for key in expected} == expected
        assert record["q6_qmicro_planar"] is (mode == "planar_candidate")
        assert record["q6_qmicro_permute"] is (mode == "current_permute")
        assert record["memory"] == {"tracked_before": {"tracked": 0}, "tracked_after": {"tracked": 0}}
    assert events == ["warm", "sync", "close"]


@pytest.mark.parametrize("order", [comparison._MODES, tuple(reversed(comparison._MODES))])
def test_layout_order_preserves_both_distinct_arms(order):
    assert comparison._parse_mode_order(",".join(order)) == order


@pytest.mark.parametrize("order", ["", "current_permute", "current_permute,current_permute", "unknown,planar_candidate"])
def test_invalid_layout_order_fails_loudly(order):
    with pytest.raises(argparse.ArgumentTypeError, match="mode order"):
        comparison._parse_mode_order(order)


@pytest.mark.parametrize("planar,permute,expected_planar,expected_prefetch", [
    (None, None, None, None), (True, None, True, None),
    (False, None, False, False), (None, True, False, False),
    (False, True, False, False), (None, False, None, None),
])
def test_profile_layout_override_preserves_defaults_or_satisfies_interleaved_prerequisites(planar, permute, expected_planar, expected_prefetch):
    overrides = profiler._q6_layout_overrides(SimpleNamespace(q6_qmicro_planar=planar, q6_qmicro_permute=permute))
    assert overrides["q6_qmicro_planar"] is expected_planar
    assert overrides["q6_qmicro_permute"] is permute
    for name in ("q6_wmma_prefetch_weight", "q6_wmma_prefetch_activation", "q6_precomputed_activation_sums"):
        assert overrides.get(name) is expected_prefetch


@pytest.mark.parametrize("flags,planar,permute", [
    ([], None, None), (["--q6-qmicro-permute"], False, True),
    (["--no-q6-qmicro-planar"], False, None),
])
def test_profile_cli_run_forwards_layout_prerequisites_to_real_constructor(monkeypatch, tmp_path, flags, planar, permute):
    model = tmp_path / "synthetic.gguf"
    model.touch()
    monkeypatch.setattr(sys, "argv", [
        "laguna_long_context_profile.py", str(model), "--lengths", "512",
        "--context-length", "512", "--allow-dirty", *flags,
    ])
    args = profiler._parse_args()
    monkeypatch.setattr(profiler, "_repo_state", lambda: {"tracked_clean": True})
    monkeypatch.setattr(profiler, "get_hip_runtime", lambda: SimpleNamespace(mem_get_info=lambda: (10, 20)))
    monkeypatch.setattr(profiler, "collect_artifact_provenance", lambda **kw: {})
    monkeypatch.setattr(profiler, "GGUFReader", lambda path: SimpleNamespace(info=object()))
    monkeypatch.setattr(profiler.LagunaGGUFTokenizer, "from_gguf_info", lambda info: object())
    monkeypatch.setattr(profiler, "_load_prompts", lambda *a: [])
    monkeypatch.setattr(profiler, "_profile_token_stream", lambda *a: ([1] * 512, "synthetic"))
    monkeypatch.setattr(profiler, "memory_stats", lambda: {})
    monkeypatch.setattr(profiler, "_compiler_version", lambda path: "synthetic")
    captured = []

    def stop(*a, **kw):
        captured.append(kw)
        raise RuntimeError("profile reached session constructor")

    monkeypatch.setattr(profiler, "LagunaGGUFResidentSession", stop)
    with pytest.raises(RuntimeError, match="profile reached session constructor"):
        profiler.run(args)
    kw, = captured
    assert kw["q6_qmicro_planar"] is planar and kw["q6_qmicro_permute"] is permute
    for key in ("q6_wmma_prefetch_weight", "q6_wmma_prefetch_activation", "q6_precomputed_activation_sums"):
        assert kw.get(key) is (False if planar is False else None)
