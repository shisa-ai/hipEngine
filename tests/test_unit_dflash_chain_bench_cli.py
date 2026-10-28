"""Default DFlash benchmark commands are distinct from serving and experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts import dflash_chain_e2e_bench as bench


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_documented_command_keeps_default_comparisons_explicit(monkeypatch, tmp_path, backend):
    class ParsedCommand(Exception):
        pass

    parse_args = argparse.ArgumentParser.parse_args
    captured = []

    def capture(parser, args=None, namespace=None):
        captured.append(parse_args(parser, args, namespace))
        raise ParsedCommand

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(bench, "validate_dflash_artifact_pair",
                        lambda **kwargs: pytest.fail("parser check opened model artifacts"))
    output = tmp_path / "result.json"
    with pytest.raises(ParsedCommand):
        bench.main([
            "--target-model", str(tmp_path / "target"),
            "--drafter-model", str(tmp_path / "drafter"),
            "--backend", backend, "--max-prompts", "0", "--decode-tokens", "128",
            "--json", str(output),
        ])
    args, = captured
    assert args.backend == backend
    assert args.decode_tokens == 128
    assert args.max_prompts == 0
    assert args.prompt_groups == "code_promotion,robustness"
    assert args.draft_budgets == "4"
    assert args.tree_mode == "chain"
    assert args.verifier_mode == "native_bulk_bplus1"
    assert args.canonical_commit_mode == "replay"
    assert args.verifier_graph == args.drafter_graph == "off"
    assert args.drafter_fusion == "off"
    assert args.drafter_query_mode == "block"
    assert args.profile_route_manifest is None
    assert args.whole_cycle_gate == args.draft_p_min == 0
    assert args.adaptive_budget == "off"
    assert args.terminal_ar_tokens == 0
    assert args.json == output
    assert not output.exists()


def test_zero_prompt_limit_keeps_all_selected_groups(monkeypatch):
    rows = [
        {"id": str(i), "benchmark_group": group, "prompt_ids": [i + 1]}
        for i, group in enumerate(["code_promotion", "robustness", "synthetic_stress", "code_promotion"])
    ]
    monkeypatch.setattr(bench, "load_prompt_records", lambda path: rows)
    assert bench._select_prompts(
        Path("unused"), groups={"code_promotion", "robustness"}, limit=0,
    ) == [rows[0], rows[1], rows[3]]
