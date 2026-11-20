"""CPU-only regression tests for speculative baseline accounting."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def metrics():
    path = Path(__file__).resolve().parents[1] / "scripts/speculative_baseline_metrics.py"
    spec = importlib.util.spec_from_file_location("speculative_baseline_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(metrics, ids=(1, 2), ar_ids=(1, 2), seconds=1.0, ar_seconds=2.0):
    return metrics.measurement(
        method="eagle4", prompt={"id": "example", "category": "code"},
        prompt_len=4, output_ids=ids, ar_ids=ar_ids, expected_tokens=2,
        seconds=seconds, ar_seconds=ar_seconds, mean_acceptance=0.5,
        accepted_per_draft=1.0,
    )


def test_exactness_rejects_equal_prefix_with_different_lengths(metrics):
    assert not sample(metrics, ids=(1,))["exact"]
    assert not sample(metrics, ar_ids=(1,))["exact"]
    assert not sample(metrics, ids=(1,), ar_ids=(1,))["exact"]
    assert sample(metrics)["exact"]


def test_weighted_summary_pools_tokens_and_times_not_rates(metrics):
    per_prompt = {
        "fast": [sample(metrics, seconds=1, ar_seconds=2), sample(metrics, seconds=2, ar_seconds=3)],
        "slow": [sample(metrics, seconds=4, ar_seconds=5), sample(metrics, seconds=8, ar_seconds=6)],
    }
    rows, summary = metrics.aggregate(per_prompt, method="eagle4")
    assert summary["eagle4_tok_s_weighted"] == pytest.approx(8 / 15)
    assert summary["ar_tok_s_weighted"] == pytest.approx(8 / 16)
    assert summary["ratio_vs_ar"] == pytest.approx(16 / 15)
    assert summary["measured_generated_tokens"] == 8
    assert summary["prompt_count"] == 2
    assert rows[0]["ar_tok_s_reps"] == pytest.approx([1, 2 / 3])
    assert rows[0]["decode_seconds_reps"] == [1, 2]
    assert rows[0]["ar_seconds_reps"] == [2, 3]
    assert rows[0]["generated_tokens_reps"] == [2, 2]
    assert summary["exact_all"]


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_timings_are_refused(metrics, seconds):
    with pytest.raises(ValueError, match="seconds"):
        sample(metrics, seconds=seconds)


def test_empty_suite_is_refused(metrics):
    with pytest.raises(ValueError, match="empty"):
        metrics.aggregate({}, method="eagle4")


def test_harness_default_matches_shipped_depth():
    import ast
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "scripts/qwen38_eagle4_baseline.py").read_text())
    defaults = [next(k.value.value for k in node.keywords if k.arg == "default")
                for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--n-draft"]
    assert defaults == [3]


@pytest.mark.parametrize("method", ["eagle4", "dspine", "dflash2"])
@pytest.mark.parametrize("short_output", [False, True])
def test_harness_records_paired_timings_and_fails_short_output(
    monkeypatch, tmp_path, method, short_output,
):
    import json
    import sys
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        f"test_baseline_{method}", root / "scripts" / f"qwen38_{method}_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"block_size": 16}
    nd = SimpleNamespace(config=SimpleNamespace(block_size=16))
    native = SimpleNamespace(close=lambda: None)
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(module, "Qwen35GGUFResidentSession", Session)
    monkeypatch.setattr(module, "get_hip_runtime", lambda: None)
    monkeypatch.setattr(module, "_load_target_arrays", lambda path: (
        SimpleNamespace(encode=lambda text: [1, 2, 3]), None, None))
    monkeypatch.setattr(module, "_run_ar", lambda *a, **kw: ([7, 8], 2.0))
    result = {"output_ids": [7] if short_output else [7, 8], "elapsed_s": 1.0,
              "decode_s": 1.0, "mean_acceptance": 0.5, "accepted_per_draft": 1.0}
    if method == "dflash2":
        monkeypatch.setattr(module, "load_and_build_drafter", lambda path: (nd, {}))
        monkeypatch.setattr(module, "DFlash2NativeDrafter", lambda *a, **kw: native)
        monkeypatch.setattr(module, "_run_dflash2_cycle_batch", lambda *a, **kw: result)
    else:
        prefix = "EAGLE4" if method == "eagle4" else "DSpine"
        monkeypatch.setattr(module, f"load_{method}_weights", lambda path: (config, {}))
        monkeypatch.setattr(module, f"{prefix}NumpyDrafter", lambda *a: nd)
        monkeypatch.setattr(module, f"{prefix}NativeDrafter", lambda *a, **kw: native)
        monkeypatch.setattr(module, f"run_{method}_cycle", lambda *a, **kw: result)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps({"id": pid, "category": "code",
        "messages": [{"content": "test"}]}) for pid in module.FULL))
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["baseline", "--prompts-file", str(prompts),
        "--output", str(output), "--max-new-tokens", "2", "--reps", "2"])
    assert module.main() == int(short_output)
    artifact = json.loads(output.read_text())
    assert artifact["schema"] == 2
    assert artifact["summary"]["exact_all"] is not short_output
    assert artifact["summary"]["ar_tok_s_weighted"] == 1.0
    assert artifact["rows"][0]["ar_seconds_reps"] == [2.0, 2.0]
    assert artifact["rows"][0]["generated_tokens_reps"] == ([1, 1] if short_output else [2, 2])


def test_archived_reaggregation_is_reproducible(tmp_path):
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    committed = root / "benchmarks/results/2026-11-12-qwen38-specdec-reaggregation.json"
    regenerated = tmp_path / "reaggregation.json"
    subprocess.run([sys.executable, "scripts/qwen38_specdec_reaggregate.py",
                    "--output", str(regenerated)], cwd=root, check=True, capture_output=True)
    import json
    actual, expected = json.loads(regenerated.read_text()), json.loads(committed.read_text())
    actual.pop("command")
    expected.pop("command")
    assert actual == expected
