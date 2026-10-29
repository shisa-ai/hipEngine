"""Keep the natural tile8 profiler comparison distinct from later decode owners."""

import sys
from types import SimpleNamespace

import pytest

from hipengine.runtime import laguna_moe as moe
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from scripts import laguna_long_context_profile as profiler


@pytest.mark.parametrize("compare", [False, True])
@pytest.mark.parametrize("row_count", [1, 2, 17])
def test_profile_tile8_comparison_owns_residency_and_holds_later_overlays_off(monkeypatch, tmp_path, compare, row_count):
    model = tmp_path / "synthetic.gguf"
    model.touch()
    flags = ["--compare-selected-natural-tile8-decode"] if compare else []
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
    if compare:
        assert kw["use_q4_expert_t16_dual_interleaved"] is False
        assert kw["use_selected_natural_decode"] is True
        assert kw["use_selected_natural_tile8_parallel_decode"] is False
        assert kw["use_selected_natural_tile8_parallel_silu_decode"] is False
        assert kw["use_selected_natural_tile8_decode"] is False
    else:
        for name in (
            "use_q4_expert_t16_dual_interleaved", "use_selected_natural_decode",
            "use_selected_natural_tile8_parallel_decode", "use_selected_natural_tile8_parallel_silu_decode",
            "use_selected_natural_tile8_decode",
        ):
            assert kw.get(name) is None

    selected = []
    routes = {label: SimpleNamespace(abi="test", allocation_name="tiles", label=label)
              for label in ("retained", "natural16", "tile8", "parallel", "silu", "interleaved")}
    plan = SimpleNamespace(**{
        name: {"q4": routes[label]}
        for name, label in (
            ("selected_gate_up_routes", "retained"), ("c1_selected_gate_up_routes", "retained"),
            ("natural_selected_gate_up_routes", "natural16"),
            ("natural_tile8_selected_gate_up_routes", "tile8"),
            ("natural_tile8_parallel_selected_gate_up_routes", "parallel"),
            ("natural_tile8_parallel_silu_selected_gate_up_routes", "silu"),
        )
    })
    plan.interleaved_natural_selected_gate_up_route = routes["interleaved"]
    weight = SimpleNamespace(
        spec=SimpleNamespace(quant_key="q4"),
        allocations={} if kw["use_q4_expert_t16_dual_interleaved"] is False else {"tiles_dual": object()},
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
    )
    monkeypatch.setattr(moe, "_SELECTED_GATE_UP_ABIS", {"test": lambda route, *a, **kw: selected.append(route.label)})
    session = object.__new__(LagunaGGUFResidentSession)
    for enabled in (False, True):
        session.set_selected_natural_tile8_decode(enabled)
        moe._launch_selected_gate_up(
            10, SimpleNamespace(weight=lambda name: weight),
            SimpleNamespace(plan=plan, selected_experts=SimpleNamespace(ptr=12)),
            x_rows=row_count, lanes=row_count * 3, stream=7, runtime=object(), libraries=None,
            use_natural=True, use_natural_tile8=session.use_selected_natural_tile8_decode,
            use_natural_tile8_parallel=kw.get("use_selected_natural_tile8_parallel_decode") is not False,
            use_natural_tile8_parallel_silu=kw.get("use_selected_natural_tile8_parallel_silu_decode") is not False,
        )
    expected = ["natural16", "tile8"] if compare else ["interleaved", "interleaved"]
    assert selected == (expected if row_count == 1 or not compare else ["retained", "retained"])
