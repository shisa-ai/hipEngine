"""Guards around the wave-admission prefill row rollup.

These exist because a refactor of the rollup silently changed the metric (completion tokens
instead of prompt-admission tokens), dropped ``--prior-config-changed`` that a published
artifact's own ``source_command`` invokes, and replaced the per-width max spread with an
aggregate delta that hides a single moving width. Each of those is a test here.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gguf_prefill_row_rollup.py"

spec = importlib.util.spec_from_file_location("gguf_prefill_row_rollup", SCRIPT)
assert spec and spec.loader
rollup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rollup)

PROTOCOL_SHA = "a" * 64
MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"


def packet(*, widths_to_wall: dict[int, float], prompt_tokens: int = 40, status: str = "ok",
           protocol_sha: str = PROTOCOL_SHA, model: str = MODEL, observed=None) -> dict:
    """Build a one-token-protocol packet: cell rate is lanes/second."""
    cells = []
    for width, wall in widths_to_wall.items():
        actual = width if observed is None else observed.get(width, width)
        cells.append({
            "width": width,
            "prompt_id": f"p{width}",
            "ar": {
                # Rate measured over `actual` lanes that actually ran.
                "tok_s": actual / wall,
                "observed_width": actual,
                "rows": [{"wall_seconds": wall, "usage": {"prompt_tokens": prompt_tokens,
                                                          "completion_tokens": actual}}],
            },
        })
    return {"protocol": {"protocol_sha256": protocol_sha, "model": model},
            "status": status, "cells": cells}


def test_row_is_prompt_admission_throughput_per_lane_not_completion_tokens():
    # One lane admits 40 prompt tokens in 0.25 s -> 160 tok/s. Counting the single completion
    # token instead would report 4 tok/s, which is the mistake this pins.
    agg = rollup.aggregate(packet(widths_to_wall={1: 0.25}))
    assert agg["per_width"]["C1"]["tok_per_s"] == pytest.approx(160.0)
    assert agg["requests"] == 1


def test_a_wave_that_admitted_fewer_lanes_than_asked_is_refused_not_relabeled():
    # A C4 cell that ran 2 lanes measures a wave the protocol did not ask for. Using the
    # observed lane count would mix wave sizes into the C4 bucket; excluding and naming it
    # keeps the width label honest.
    agg = rollup.aggregate(packet(widths_to_wall={1: 0.25, 4: 0.5}, observed={4: 2}))
    assert "C4" not in agg["per_width"]
    refusals = [name for name in agg["excluded_cells"] if name.startswith("C4:")]
    assert refusals and refusals[0].endswith("observed_width")


def test_missing_rate_or_prompt_tokens_is_refused_and_named():
    pkt = packet(widths_to_wall={1: 0.25, 2: 0.3})
    pkt["cells"][1]["ar"]["tok_s"] = 0.0
    agg = rollup.aggregate(pkt)
    assert "C2" not in agg["per_width"]
    assert any(name.startswith("C2:") for name in agg["excluded_cells"])


def test_repeatability_is_max_per_width_spread_not_the_aggregate():
    # Aggregate can look calm while one width moves: the band must catch the width.
    base = packet(widths_to_wall={1: 0.25, 2: 0.30})
    moved = packet(widths_to_wall={1: 0.25, 2: 0.3001})
    agg_a, agg_b = rollup.aggregate(base), rollup.aggregate(moved)
    frac, _, _ = rollup.drift(agg_a, agg_b)
    max_spread, deltas = rollup.spread(agg_a, agg_b)
    assert frac is not None and frac * 100 < 0.5          # aggregate looks fine
    assert "C2" in deltas and max_spread >= 0.03          # the width moved


def test_same_status_ok_prior_gives_two_sided_drift(tmp_path, capsys):
    here = tmp_path / "here.json"
    prior = tmp_path / "prior.json"
    here.write_text(__import__("json").dumps(packet(widths_to_wall={1: 0.25, 2: 0.3})))
    prior.write_text(__import__("json").dumps(packet(widths_to_wall={1: 0.252, 2: 0.30})))
    assert rollup.main([str(here), "--prior", str(prior)]) == 0
    out = capsys.readouterr().out
    assert "cross-session drift vs prior:" in out and "max per-width" in out
    assert "NOT two-sided" not in out


def test_non_ok_prior_warns_by_default_and_refuses_under_strict(tmp_path, capsys):
    here = tmp_path / "here.json"
    prior = tmp_path / "prior.json"
    here.write_text(__import__("json").dumps(packet(widths_to_wall={1: 0.25})))
    prior.write_text(__import__("json").dumps(packet(widths_to_wall={1: 0.25}, status="failed")))
    assert rollup.main([str(here), "--prior", str(prior)]) == 0
    assert "NOT two-sided" in capsys.readouterr().out
    capsys.readouterr()
    assert rollup.main([str(here), "--prior", str(prior), "--strict-prior"]) == 3
    assert "not comparable" in capsys.readouterr().err


def test_config_changed_prior_reports_effect_size_and_skips_the_band(tmp_path, capsys):
    # An A/B pair: the prior differs by configuration, so the delta is an effect size and the
    # drift band must not be applied. Published artifacts call this flag directly.
    here = tmp_path / "on.json"
    prior = tmp_path / "off.json"
    here.write_text(__import__("json").dumps(
        packet(widths_to_wall={1: 0.25, 2: 0.1}, protocol_sha="b" * 64)))
    prior.write_text(__import__("json").dumps(packet(widths_to_wall={1: 0.25, 2: 0.3})))
    assert rollup.main([str(here), "--prior", str(prior), "--prior-config-changed"]) == 0
    out = capsys.readouterr().out
    assert "effect size, not drift" in out
    assert "C2:+200.0%" in out
    assert "drift band" not in out and "NOT two-sided" not in out


def test_strict_prior_refuses_a_different_model_even_when_status_ok(tmp_path, capsys):
    here = tmp_path / "here.json"
    prior = tmp_path / "prior.json"
    here.write_text(__import__("json").dumps(packet(widths_to_wall={1: 0.25})))
    prior.write_text(__import__("json").dumps(
        packet(widths_to_wall={1: 0.25}, model="/models/gguf/Other.gguf")))
    assert rollup.main([str(here), "--prior", str(prior), "--strict-prior"]) == 3
    assert "protocol or model differs" in capsys.readouterr().err


def test_empty_packet_is_an_error_not_a_zero_rate(tmp_path, capsys):
    empty = tmp_path / "empty.json"
    empty.write_text(__import__("json").dumps({"protocol": {"protocol_sha256": PROTOCOL_SHA,
                                                           "model": MODEL}, "status": "ok",
                                               "cells": []}))
    assert rollup.main([str(empty)]) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
