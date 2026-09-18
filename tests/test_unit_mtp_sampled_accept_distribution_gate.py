"""Oracle for the sampled-accept distribution gate's own integrator.

The gate decides whether a speculative cycle is exactly ``p``-distributed by
comparing the implemented decision rule's *induced* law against the autoregressive
law. That comparison is only meaningful if the induced law is integrated
correctly, so these tests pin the closed form against a brute-force enumeration
of the same rule on small supports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from hipengine.speculative.sampling import (
    SparseDistribution,
    acceptance_probability,
    residual_distribution,
)

_GATE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mtp_sampled_accept_distribution_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_mtp_sampled_accept_gate", _GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _brute_force(target: SparseDistribution, draft: SparseDistribution) -> dict[int, float]:
    """Enumerate the rule: draft ``t`` with ``q(t)``, accept with ``min(1, p/q)``."""

    residual = residual_distribution(target, draft)
    support = sorted(set(target.token_ids) | set(draft.token_ids) | set(residual.token_ids))
    induced = {token_id: 0.0 for token_id in support}
    for token_id in draft.token_ids:
        weight = draft.probability(token_id)
        accepted = acceptance_probability(target, draft, token_id)
        induced[token_id] += weight * accepted
        for emitted in residual.token_ids:
            induced[emitted] += weight * (1.0 - accepted) * residual.probability(emitted)
    return induced


def _random_pair(rng: np.random.Generator, *, vocab: int = 200):
    size = int(rng.integers(3, 9))
    ids = tuple(sorted(rng.choice(vocab, size=size, replace=False).tolist()))
    weights = rng.random(size)
    target = SparseDistribution(ids, weights / weights.sum())
    draft_size = int(rng.integers(1, 7))
    draft_ids = tuple(sorted(rng.choice(vocab, size=draft_size, replace=False).tolist()))
    if not set(draft_ids) & set(ids):
        draft_ids = tuple(sorted(set(draft_ids) | {ids[0]}))
    draft_weights = rng.random(len(draft_ids))
    draft = SparseDistribution(draft_ids, draft_weights / draft_weights.sum())
    return target, draft


@pytest.mark.parametrize("seed", range(4))
def test_closed_form_matches_brute_force_enumeration(seed: int) -> None:
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(120):
        target, draft = _random_pair(rng)
        try:
            reference = _brute_force(target, draft)
        except ValueError:
            # p == q exactly; covered by its own case below.
            continue
        induced_ids, induced_weights = gate._induced_first_token(target, draft)
        assert abs(float(np.sum(induced_weights)) - 1.0) < 1e-12
        observed = dict(zip(induced_ids.tolist(), induced_weights.tolist(), strict=True))
        for token_id in set(reference) | set(observed):
            assert reference.get(token_id, 0.0) == pytest.approx(
                observed.get(token_id, 0.0), abs=1e-12
            )
        checked += 1
    assert checked > 100


def test_identical_draft_law_is_accepted_whole() -> None:
    rng = np.random.default_rng(11)
    for _ in range(50):
        size = int(rng.integers(2, 12))
        ids = tuple(sorted(rng.choice(400, size=size, replace=False).tolist()))
        weights = rng.random(size)
        target = SparseDistribution(ids, weights / weights.sum())
        induced_ids, induced_weights = gate._induced_first_token(target, target)
        metrics = gate._distribution_metrics(induced_ids, induced_weights, target)
        assert metrics["total_variation"] == pytest.approx(0.0, abs=1e-15)
        assert metrics["max_abs_deviation"] == pytest.approx(0.0, abs=1e-15)
        assert metrics["top1_agreement"] == 1.0


def test_induced_law_equals_the_autoregressive_law_for_any_draft() -> None:
    """The whole point of the rule: the emitted token is still ``p``-distributed."""

    rng = np.random.default_rng(7)
    for _ in range(150):
        target, draft = _random_pair(rng)
        try:
            induced_ids, induced_weights = gate._induced_first_token(target, draft)
        except ValueError:
            continue
        metrics = gate._distribution_metrics(induced_ids, induced_weights, target)
        assert metrics["total_variation"] < 1e-12
        assert metrics["kl_p_to_induced"] < 1e-12
        assert metrics["kl_induced_to_p"] < 1e-12
        assert metrics["top1_agreement"] == 1.0


def test_full_vocabulary_support_is_linear_not_quadratic() -> None:
    """A real 151k-token row must be integrable; the quadratic form never was."""

    rng = np.random.default_rng(3)
    size = 200_000
    ids = np.arange(size, dtype=np.int64)
    weights = rng.random(size)
    target = SparseDistribution(tuple(ids.tolist()), weights / weights.sum())
    draft_weights = rng.random(size)
    draft = SparseDistribution(tuple(ids.tolist()), draft_weights / draft_weights.sum())
    induced_ids, induced_weights = gate._induced_first_token(target, draft)
    assert induced_ids.size == size
    metrics = gate._distribution_metrics(induced_ids, induced_weights, target)
    assert metrics["total_variation"] < 1e-12
    point_mass = SparseDistribution.point_mass(int(ids[0]))
    induced_ids, induced_weights = gate._induced_first_token(target, point_mass)
    metrics = gate._distribution_metrics(induced_ids, induced_weights, target)
    assert metrics["total_variation"] < 1e-12
    assert metrics["support"] == size


def test_monte_carlo_arm_drives_the_walk_on_a_bounded_support() -> None:
    """The black-box arm must agree with p inside its own sampling error."""

    rng = np.random.default_rng(5)
    size = 4096
    ids = tuple(sorted(rng.choice(250_000, size=size, replace=False).tolist()))
    # Peaked like a real next-token row (the regime the gate runs in), so the
    # histogram's top token is a meaningful discriminator.
    weights = np.exp(-np.arange(size, dtype=np.float64) / 5.0)
    target = SparseDistribution(ids, weights / weights.sum())
    report = gate._monte_carlo_check(target, draws=4000, seed=17)
    assert report["checked_support"] == gate.MONTE_CARLO_SUPPORT
    assert report["distinct_emitted_tokens"] > 1
    assert report["max_cell_excess"] <= 0.0
    assert report["top1_agreement"] == 1.0


def test_monte_carlo_arm_accepts_a_small_support_unchanged() -> None:
    rng = np.random.default_rng(9)
    size = 64
    ids = tuple(sorted(rng.choice(1_000, size=size, replace=False).tolist()))
    weights = rng.random(size)
    target = SparseDistribution(ids, weights / weights.sum())
    report = gate._monte_carlo_check(target, draws=2000, seed=3)
    assert report["checked_support"] == size
    assert report["max_cell_excess"] <= 0.0


def test_monte_carlo_arm_rejects_a_biased_walk(monkeypatch) -> None:
    """The discriminator must fire when the walk emits the wrong law."""

    rng = np.random.default_rng(13)
    size = 64
    ids = tuple(sorted(rng.choice(50_000, size=size, replace=False).tolist()))
    weights = np.exp(-np.arange(size, dtype=np.float64) / 5.0)
    target = SparseDistribution(ids, weights / weights.sum())
    healthy = gate._monte_carlo_check(target, draws=4000, seed=2)
    assert healthy["max_cell_excess"] <= 0.0

    import dataclasses

    real_walk = gate.sampled_accept_from_distributions

    def always_accept(batch, targets, drafts, *, draws, **kwargs):
        result = real_walk(batch, targets, drafts, draws=draws, **kwargs)
        # The drafted token wins every draw: the accept test is inverted.
        accepted = tuple(
            (int(batch.tokens[1]),) for _ in result.accepted_tokens
        )
        return dataclasses.replace(
            result,
            accepted_tokens=accepted,
            accepted_counts=tuple(len(tokens) for tokens in accepted),
        )

    import dataclasses

    monkeypatch.setattr(gate, "sampled_accept_from_distributions", always_accept)
    biased = gate._monte_carlo_check(target, draws=4000, seed=2)
    assert biased["max_cell_excess"] > 0.0
    assert biased["top1_agreement"] == 1.0  # the top token is unchanged
    monkeypatch.setattr(gate, "sampled_accept_from_distributions", real_walk)
