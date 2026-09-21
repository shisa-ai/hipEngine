"""Self-test of the AR replay gate: a gate that cannot fail proves nothing.

``scripts/yue2_ar_replay.py`` turns replay rows into the production numerical
envelope (KL, top-1, top-8 recall). These tests feed it synthetic rows with known
answers - identical rows must score perfectly, perturbed rows must not - so a
reporting bug cannot masquerade as a passing gate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
VOCAB = 184704


def _load_gate():
    spec = importlib.util.spec_from_file_location("yue2_ar_replay", REPO / "scripts/yue2_ar_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load_gate()


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float32)
    wide = array.view(np.uint32).astype(np.uint64)
    return (
        (wide + np.uint64(0x7FFF) + ((wide >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)
    ).astype(np.uint16)


LADDER = np.array([30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0], dtype=np.float32)


def _logits(rng, rows: int, scale: float = 2.0) -> np.ndarray:
    """Row-wise logits with a clear top-8 and a realistic noise tail.

    The top-8 is a widely separated ladder so bf16 rounding cannot create a
    rank tie at the eighth place; ties would make the recall metric ambiguous.
    """
    values = rng.standard_normal((rows, VOCAB)).astype(np.float32) * scale
    order = np.argsort(-values, axis=-1)
    for row in range(rows):
        values[row, order[row, :8]] = LADDER
    return values


def _rank(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top_ids = np.argsort(-rows, axis=-1)[..., :8]
    top_vals = np.take_along_axis(rows, top_ids, axis=-1)
    return top_ids.astype(np.int64), top_vals.astype(np.float32)


def _fixture(gate, rng, *, steps: int = 4, branches: int = 1, prefix: int = 8, cfg: bool = False):
    reference = [_logits(rng, branches) for _ in range(steps)]
    # Rank the bf16-rounded rows: that is what the runtime can reproduce, so an
    # identical replay must score perfectly.
    rounded = [gate.bf16_bits_to_f32(f32_to_bf16_bits(row)) for row in reference]
    top_ids = np.stack([_rank(row)[0] for row in rounded])
    top_vals = np.stack([_rank(row)[1] for row in rounded])
    argmax = top_ids[..., 0].astype(np.int64)
    fixture = {
        "prefix_positive": np.arange(prefix, dtype=np.int32),
        "tokens": np.arange(steps, dtype=np.int32),
        "prefill_logits": f32_to_bf16_bits(_logits(rng, 1))[None, :],
        "step_argmax": argmax,
        "step_top_ids": top_ids,
        "step_top_vals": top_vals,
        "full_logits_steps": np.asarray([[index, branch] for index in range(steps) for branch in range(branches)], dtype=np.int32).reshape(-1),
    }
    for index in range(steps):
        for branch in range(branches):
            fixture[f"full_logits_{index}_{branch}"] = f32_to_bf16_bits(reference[index][branch])[None, :]
    if cfg:
        fixture["prefix_negative"] = np.arange(prefix, dtype=np.int32)
        fixture["prefill_logits_negative"] = f32_to_bf16_bits(_logits(rng, 1))[None, :]
    manifest = {
        "phase": "semantic",
        "cfg": cfg,
        "prefix_lengths": [prefix, prefix] if cfg else [prefix],
    }
    # The candidate rows mirror what the runtime returns: the fixture's own
    # bf16 logits, upcast exactly. Using the unrounded reference here would make
    # the test disagree with the gate on argmax ties.
    rows = {
        "prefill": (
            gate.bf16_bits_to_f32(fixture["prefill_logits"][0]),
            gate.bf16_bits_to_f32(fixture["prefill_logits"][0]),
        ),
        "steps": [
            [
                gate.bf16_bits_to_f32(fixture[f"full_logits_{index}_{branch}"][0])
                for branch in range(branches)
            ]
            for index in range(steps)
        ],
    }
    if cfg:
        rows["prefill_negative"] = (
            gate.bf16_bits_to_f32(fixture["prefill_logits_negative"][0]),
            gate.bf16_bits_to_f32(fixture["prefill_logits_negative"][0]),
        )
    return fixture, manifest, rows, reference


def test_identical_rows_score_perfectly(gate):
    rng = np.random.default_rng(3)
    fixture, manifest, rows, _ = _fixture(gate, rng, steps=3)
    result = gate.evaluate("synthetic-nocfg", manifest, fixture, rows)
    assert result["kl"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert result["top1_agreement"] == 1.0
    assert result["top8_recall"] == 1.0
    assert result["argmax_flips"] == []
    assert result["full_vocab_rows"] == 4  # prefill + 3 steps


def test_cfg_case_scores_both_branches(gate):
    rng = np.random.default_rng(5)
    fixture, manifest, rows, _ = _fixture(gate, rng, steps=2, branches=2, cfg=True)
    result = gate.evaluate("synthetic-cfg", manifest, fixture, rows)
    assert result["full_vocab_rows"] == 2 + 2 * 2  # prefill x2 + steps x branches
    assert result["top1_agreement"] == 1.0
    assert set(result["top1_by_scope"]) == {"semantic:cfg:L8"}


def test_perturbed_rows_fail_the_gate(gate):
    rng = np.random.default_rng(7)
    fixture, manifest, rows, reference = _fixture(gate, rng, steps=3)
    # Sharpen the candidate rows: the argmax survives while KL grows, which is
    # exactly the case a top-1-only gate would miss. A uniform offset would not
    # change the distribution at all.
    perturbed = {
        "prefill": (rows["prefill"][0], rows["prefill"][1] * 1.5),
        "steps": [[row * 1.5 for row in step] for step in rows["steps"]],
    }
    result = gate.evaluate("synthetic-perturbed", manifest, fixture, perturbed)
    assert result["kl"]["mean"] > 0.0
    assert result["top1_agreement"] is not None
    assert result["top8_recall"] is not None

    flipped = np.array(rows["steps"][0][0], copy=True)
    flipped[int(reference[0][0].argmax())] = -50.0
    flipped[int(reference[0][0].argmin())] = 50.0
    rows_flipped = {
        "prefill": (rows["prefill"][0], rows["prefill"][1]),
        "steps": [[flipped]] + [[row for row in step] for step in rows["steps"][1:]],
    }
    result = gate.evaluate("synthetic-flipped", manifest, fixture, rows_flipped)
    assert result["top1_agreement"] < 1.0
    assert result["argmax_flips"], "a deliberate argmax flip must be reported"


def test_missing_full_logits_reduces_row_count_without_faking_kl(gate):
    rng = np.random.default_rng(11)
    fixture, manifest, rows, _ = _fixture(gate, rng, steps=3)
    fixture["full_logits_steps"] = np.asarray([0, 0], dtype=np.int32)
    result = gate.evaluate("synthetic-partial", manifest, fixture, rows)
    assert result["full_vocab_rows"] == 2  # prefill + step 0 only
    assert result["top1_agreement"] == 1.0


def test_kl_divergence_is_zero_for_identical_and_positive_otherwise(gate):
    rng = np.random.default_rng(13)
    values = _logits(rng, 1)[0]
    assert gate.kl_divergence(values, values) == pytest.approx(0.0, abs=1e-12)
    # A uniform shift leaves the distribution unchanged; a sharpened row must not.
    assert gate.kl_divergence(values, values + 1.0) == pytest.approx(0.0, abs=1e-9)
    sharpened = values * 1.05
    assert gate.kl_divergence(values, sharpened) > 0.0
    # Asymmetry: KL is not symmetric, and the gate must use reference || candidate.
    forward = gate.kl_divergence(values, sharpened)
    backward = gate.kl_divergence(sharpened, values)
    assert forward != pytest.approx(backward, rel=1e-6)


def test_softmax_is_normalized(gate):
    rng = np.random.default_rng(17)
    values = _logits(rng, 4)
    probabilities = gate.softmax_f32(values[0])
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(probabilities >= 0)


def test_top8_recall_counts_each_missing_reference_id(gate):
    """One dropped reference id must cost exactly one eighth of the recall."""
    rng = np.random.default_rng(23)
    fixture, manifest, rows, _ = _fixture(gate, rng, steps=2)
    reference_ids = fixture["step_top_ids"][0, 0]
    dropped = int(reference_ids[-1])
    candidate = np.array(rows["steps"][0][0], copy=True)
    candidate[dropped] = -1e9
    rows_dropped = {
        "prefill": rows["prefill"],
        "steps": [[candidate]] + [[row for row in step] for step in rows["steps"][1:]],
    }
    result = gate.evaluate("synthetic-recall", manifest, fixture, rows_dropped)
    assert result["top8_recall"] == pytest.approx(15 / 16)
    assert result["top8_kl"]["max"] > 0.0
    assert result["top1_agreement"] == 1.0  # the dropped id was not the argmax


def test_rows_equal_is_bit_exact(gate):
    rng = np.random.default_rng(29)
    _, _, rows, _ = _fixture(gate, rng, steps=2)
    same = {"prefill": rows["prefill"], "steps": [[row.copy() for row in step] for step in rows["steps"]]}
    assert gate.rows_equal(rows, same)
    same["steps"][1][0][0] += np.float32(1e-6)
    assert not gate.rows_equal(rows, same)
    shorter = {"prefill": rows["prefill"], "steps": rows["steps"][:1]}
    assert not gate.rows_equal(rows, shorter)
    missing_negative = {"prefill": rows["prefill"], "steps": rows["steps"], "prefill_negative": rows["prefill"]}
    assert not gate.rows_equal(rows, missing_negative)


def test_summarize_reports_tails(gate):
    summary = gate.summarize([0.0, 1e-4, 2e-3, 0.05])
    assert summary["rows"] == 4
    assert summary["max"] == pytest.approx(0.05)
    assert summary["mean"] == pytest.approx((0.0 + 1e-4 + 2e-3 + 0.05) / 4)
    assert summary["p95"] >= summary["mean"]


def test_gate_requires_the_broad_floor(gate):
    """The retained status is the AGENTS.md floor, not "the run finished"."""

    assert gate._gate_passes({"pooled_mean_kl": 1e-3, "top1_agreement": 0.95})
    assert not gate._gate_passes({"pooled_mean_kl": 0.2, "top1_agreement": 0.95})
    assert not gate._gate_passes({"pooled_mean_kl": 1e-3, "top1_agreement": 0.5})
    # A missing or empty matrix must not read as a pass.
    assert not gate._gate_passes({})
    assert not gate._gate_passes({"pooled_mean_kl": None, "top1_agreement": None})


def test_artifact_provenance_names_the_machine_and_the_command(gate):
    """The evidence-policy fields are collected, never typed into the artifact."""

    provenance = gate._provenance(
        ["--prefill-variant", "strict", "--json", "out.json"],
        REPO / "tests/fixtures/yue2",
        "bit-identical",
    )
    assert provenance["model"] == "m-a-p/YuE2-3B"
    assert provenance["quant"] == "bf16"
    assert provenance["command"] == (
        "python3 scripts/yue2_ar_replay.py --prefill-variant strict --json out.json"
    )
    assert provenance["host"]["name"]
    assert provenance["host"]["gpu"]
    assert provenance["date"].endswith("+00:00")
    assert len(provenance["fixtures_integrity_sha256"]) == 64
    assert provenance["torch_imported"] is False
    assert provenance["correctness_gate"]["repeat"] == "bit-identical"
    assert provenance["correctness_gate"]["broad_floor"] == {"mean_kl_max": 0.05, "top1_min": 0.9}
