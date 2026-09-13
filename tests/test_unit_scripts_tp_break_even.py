"""CPU-only tests for scripts/tp_break_even.py arithmetic and verdicts.

No HIP/ROCm is touched: the projection is pure arithmetic over measured inputs.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tp_break_even.py"


def _load():
    spec = importlib.util.spec_from_file_location("tp_break_even_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_parse_tp1_accepts_named_triples(mod) -> None:
    device = mod.parse_tp1("W7900=27.9:15.652:8.646")
    assert device == {"name": "W7900", "tok_s": 27.9, "tp1_gib": 15.652, "rank_gib": 8.646}


def test_parse_tp1_rejects_malformed_specs(mod) -> None:
    with pytest.raises(ValueError):
        mod.parse_tp1("27.9:15.652:8.646")
    with pytest.raises(ValueError):
        mod.parse_tp1("W7900=27.9:15.652")
    with pytest.raises(ValueError):
        mod.parse_tp1("=27.9:15.652:8.646")
    with pytest.raises(ValueError):
        mod.parse_tp1("W7900=0:15.652:8.646")
    with pytest.raises(ValueError):
        mod.parse_tp1("W7900=27.9:15.652:20.0")


def test_project_reproduces_the_hand_computed_row(mod) -> None:
    device = mod.parse_tp1("W7900=25.0:16.0:8.0")
    row = mod.project(device, collective_ms=1.0, fixed_share=0.0)
    assert row["tp1_ms_per_token"] == pytest.approx(40.0)
    assert row["rank_weight_fraction"] == pytest.approx(0.5)
    assert row["rank_weight_ms_per_token"] == pytest.approx(20.0)
    assert row["tp2_ms_per_token"] == pytest.approx(21.0)
    assert row["projected_speedup"] == pytest.approx(40.0 / 21.0)
    assert row["break_even_collective_ms"] == pytest.approx(20.0)


def test_project_hits_break_even_exactly_at_the_budget(mod) -> None:
    device = mod.parse_tp1("W7900=25.0:16.0:8.0")
    budget = mod.project(device, collective_ms=1.0, fixed_share=0.25)["break_even_collective_ms"]
    row = mod.project(device, collective_ms=budget, fixed_share=0.25)
    assert row["projected_speedup"] == pytest.approx(1.0, abs=1e-9)


def test_project_moves_the_right_way(mod) -> None:
    device = mod.parse_tp1("W7900=25.0:16.0:8.0")
    cheap = mod.project(device, collective_ms=1.0, fixed_share=0.0)
    expensive = mod.project(device, collective_ms=8.0, fixed_share=0.0)
    assert cheap["projected_speedup"] > expensive["projected_speedup"]
    more_fixed = mod.project(device, collective_ms=1.0, fixed_share=0.4)
    assert more_fixed["projected_speedup"] < cheap["projected_speedup"]
    assert more_fixed["break_even_collective_ms"] < cheap["break_even_collective_ms"]


def test_implied_bandwidth_matches_the_tp1_row(mod) -> None:
    device = mod.parse_tp1("W7900=27.9:15.652:8.646")
    row = mod.project(device, collective_ms=1.3, fixed_share=0.0)
    # 15.652 GiB in the TP1 token time is the bandwidth the model implies.
    assert row["implied_bandwidth_gbs"] == pytest.approx(15.652 / (row["tp1_ms_per_token"] / 1000.0))


def test_build_report_verdict_and_serializability(mod) -> None:
    devices = [mod.parse_tp1("W7900=27.9:15.652:8.646"), mod.parse_tp1("XTX=29.82:15.652:7.009")]
    report = mod.build_report(devices, collective_ms=(1.3, 1.5), fixed_shares=(0.0, 0.3))
    assert report["verdict"]["passes_target_in_every_row"] is True
    assert report["verdict"]["collective_would_have_to_be_worse_by"] > 1.0
    assert len(report["devices"]) == 2
    for entry in report["devices"]:
        assert len(entry["rows"]) == 4
        assert entry["worst_case_speedup"] <= entry["best_case_speedup"]
    assert json.loads(json.dumps(report))["kind"] == "tp2_break_even"


def test_build_report_flags_a_losing_projection(mod) -> None:
    # A 1 tok/s baseline with a huge collective budget cannot reach 1.3x.
    devices = [mod.parse_tp1("slow=1.0:16.0:8.0")]
    report = mod.build_report(devices, collective_ms=(800.0,), fixed_shares=(0.0,))
    assert report["verdict"]["passes_target_in_every_row"] is False
    assert report["verdict"]["collective_would_have_to_be_worse_by"] < 1.0
    assert report["devices"][0]["rows"][0]["projected_speedup"] < 1.0


def test_main_writes_the_artifact(tmp_path: pathlib.Path, mod, capsys) -> None:
    output = tmp_path / "break_even.json"
    exit_code = mod.main(
        [
            "--tp1",
            "W7900=27.9:15.652:8.646",
            "--collective-ms",
            "1.3",
            "--fixed-share",
            "0.2",
            "--json",
            str(output),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["devices"][0]["rows"][0]["collective_ms_per_token"] == pytest.approx(1.3)
    assert "verdict" in capsys.readouterr().out
