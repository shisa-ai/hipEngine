"""Exactness gate for sampled speculative acceptance (min(1, p/q) + residual).

The binding contract of ``hipengine.speculative.sampling`` is distributional: a
request that samples must receive tokens distributed exactly as the
autoregressive sampler would emit them, not merely tokens that look plausible.
These tests prove that contract directly instead of sampling from it:

* the induced first-token distribution is computed in closed form by
  integrating the algorithm's own decision rule over the acceptance draw, and
  compared against ``p`` elementwise;
* the same code path is driven end-to-end over a stratified draw grid, which
  catches wiring mistakes (wrong branch, wrong draw order) that a closed form
  written next to the implementation could share.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hipengine.speculative.interfaces import TargetVerifyBatch
from hipengine.speculative.sampling import (
    SparseDistribution,
    acceptance_probability,
    residual_distribution,
    sampled_accept_from_distributions,
)


def _chain_batch(tokens: tuple[int, ...]) -> TargetVerifyBatch:
    """Build the linear chain layout the MTP route verifies: root + K drafts."""

    rows = len(tokens) + 1
    return TargetVerifyBatch(
        request_ids=(7,),
        tokens=(100, *tokens),
        positions=tuple(range(rows)),
        row_to_request=(7,) * rows,
        parent_rows=(-1, *range(rows - 1)),
        root_rows=(0,),
        candidate_rows=tuple(range(1, rows)),
        draft_depths=(0, *range(1, rows)),
        active_mask=(True,) * rows,
    )


def _distribution(pairs: dict[int, float]) -> SparseDistribution:
    ids = tuple(sorted(pairs))
    return SparseDistribution(ids, np.asarray([pairs[i] for i in ids]))


def _induced_first_token(
    target: SparseDistribution,
    draft: SparseDistribution,
) -> dict[int, float]:
    """Integrate the decision rule: accept with min(1, p/q), else residual."""

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


def _assert_matches_target(induced: dict[int, float], target: SparseDistribution) -> None:
    total = sum(induced.values())
    assert total == pytest.approx(1.0, abs=1e-12)
    for token_id in sorted(set(induced) | set(target.token_ids)):
        assert induced.get(token_id, 0.0) == pytest.approx(
            target.probability(token_id), abs=1e-12
        ), f"token {token_id} is not target-distributed"


# --------------------------------------------------------------------------- #
# Math gate: the coupling reproduces the target distribution.
# --------------------------------------------------------------------------- #

_TARGET_CASES = {
    "peaked": {0: 0.7, 1: 0.2, 2: 0.07, 3: 0.03},
    "flat": {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
    "long_tail": {0: 0.5, **{index: 0.5 / 31 for index in range(1, 32)}},
    "two_mass": {0: 0.5, 1: 0.5},
}

_DRAFT_CASES = {
    "identical": None,  # filled per target
    "shifted": {1: 0.6, 2: 0.3, 3: 0.1},
    "tail_heavy": {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.85},
    "disjoint_tail": {4: 0.5, 5: 0.5},
    "point_mass_argmax": None,  # filled per target
}


@pytest.mark.parametrize("target_name", sorted(_TARGET_CASES))
@pytest.mark.parametrize("draft_name", sorted(_DRAFT_CASES))
def test_min_ratio_coupling_reproduces_target_exactly(
    target_name: str, draft_name: str
) -> None:
    target = _distribution(_TARGET_CASES[target_name])
    if draft_name == "identical":
        draft = target
    elif draft_name == "point_mass_argmax":
        draft = SparseDistribution.point_mass(target.token_ids[0])
    else:
        draft = _distribution(_DRAFT_CASES[draft_name])
    identical = draft.token_ids == target.token_ids and np.allclose(
        draft.probabilities, target.probabilities
    )
    if identical:
        pytest.skip("identical distributions have no residual mass by construction")
    induced = _induced_first_token(target, draft)
    _assert_matches_target(induced, target)


@pytest.mark.parametrize("token_id", [0, 1, 2, 3])
def test_point_mass_coupling_is_exact_for_every_drafted_token(token_id: int) -> None:
    """A greedy draft chain supplies q = delta(x); the output is still exactly p."""

    target = _distribution({0: 0.55, 1: 0.25, 2: 0.15, 3: 0.05})
    draft = SparseDistribution.point_mass(token_id)
    induced = _induced_first_token(target, draft)
    _assert_matches_target(induced, target)


def test_point_mass_coupling_acceptance_is_the_target_probability() -> None:
    target = _distribution({0: 0.55, 1: 0.25, 2: 0.15, 3: 0.05})
    for token_id in target.token_ids:
        draft = SparseDistribution.point_mass(token_id)
        assert acceptance_probability(target, draft, token_id) == pytest.approx(
            target.probability(token_id)
        )


def test_residual_is_normalized_positive_mass_over_union_support() -> None:
    target = _distribution({0: 0.5, 1: 0.3, 2: 0.2})
    draft = _distribution({1: 0.9, 2: 0.1})
    residual = residual_distribution(target, draft)
    # max(0, p - q) leaves 0.5 at token 0 and 0.1 at token 2, renormalized by 0.6.
    assert residual.token_ids == (0, 2)
    assert float(residual.probabilities.sum()) == pytest.approx(1.0)
    assert residual.probability(0) == pytest.approx(0.5 / 0.6)
    assert residual.probability(1) == pytest.approx(0.0)
    assert residual.probability(2) == pytest.approx(0.1 / 0.6)


def test_residual_rejects_identical_distributions() -> None:
    target = _distribution({0: 0.5, 1: 0.5})
    with pytest.raises(ValueError, match="no residual mass"):
        residual_distribution(target, target)


def test_zero_draft_probability_is_rejected() -> None:
    target = _distribution({0: 0.5, 1: 0.5})
    draft = _distribution({0: 1.0})
    with pytest.raises(ValueError, match="zero draft probability"):
        acceptance_probability(target, draft, 1)


# --------------------------------------------------------------------------- #
# Wiring gate: the implemented walk matches the closed form under real draws.
# --------------------------------------------------------------------------- #

def _stratified_grid(size: int) -> tuple[float, ...]:
    return tuple((index + 0.5) / size for index in range(size))


def _draw_sequence(values: tuple[float, ...]):
    remaining = list(values)

    def draw() -> float:
        if not remaining:
            raise AssertionError("acceptance walk consumed more draws than expected")
        return remaining.pop(0)

    return draw


@pytest.mark.parametrize("draft_name", ["point_mass_argmax", "shifted"])
def test_walk_emits_the_closed_form_distribution(draft_name: str) -> None:
    """Drive the real walk over a stratified (acceptance, residual) draw grid."""

    target = _distribution({0: 0.7, 1: 0.2, 2: 0.07, 3: 0.03})
    if draft_name == "point_mass_argmax":
        draft = SparseDistribution.point_mass(0)
        draft_support = ((0, 1.0),)
    else:
        draft = _distribution(_DRAFT_CASES[draft_name])
        draft_support = tuple(
            (token_id, draft.probability(token_id)) for token_id in draft.token_ids
        )
    induced = _induced_first_token(target, draft)
    grid = _stratified_grid(128)
    observed: dict[int, float] = {}
    for drafted, drafted_weight in draft_support:
        batch = _chain_batch((drafted,))
        for acceptance_draw in grid:
            for residual_draw in grid:
                result = sampled_accept_from_distributions(
                    batch,
                    (target, target),
                    (draft, draft),
                    draws=_draw_sequence((acceptance_draw, residual_draw, residual_draw)),
                )
                emitted = (
                    result.accepted_tokens[0][0]
                    if result.accepted_tokens[0]
                    else result.next_tokens[0]
                )
                observed[emitted] = observed.get(emitted, 0.0) + drafted_weight / (
                    len(grid) * len(grid)
                )
    assert sum(observed.values()) == pytest.approx(1.0, abs=1e-12)
    for token_id in sorted(set(observed) | set(induced)):
        # The grid resolves each branch to 1/128, so the wiring check is a
        # distribution comparison at grid resolution, not an exact identity.
        assert observed.get(token_id, 0.0) == pytest.approx(
            induced.get(token_id, 0.0), abs=2.0 / len(grid)
        ), f"token {token_id} diverged from the closed form"


def test_first_rejection_resamples_and_discards_the_rest_of_the_chain() -> None:
    """A rejected depth 0 must emit the residual sample and drop later drafts."""

    target = _distribution({0: 0.5, 1: 0.5})
    draft = _distribution({0: 0.9, 1: 0.1})
    batch = _chain_batch((0, 1, 0))
    result = sampled_accept_from_distributions(
        batch,
        (target, target, target, target),
        (draft, draft, draft, draft),
        # accept draw 0.9 -> reject (acceptance 0.5/0.9), residual draw 0.99 -> token 1
        draws=_draw_sequence((0.99, 0.99)),
    )
    assert result.accepted_counts == (0,)
    assert result.accepted_tokens == ((),)
    assert result.next_tokens == (1,)
    assert result.selected_candidate_rows == (0,)


def test_full_accept_samples_bonus_from_the_last_verified_prefix() -> None:
    target = _distribution({0: 1.0})
    draft = SparseDistribution.point_mass(0)
    batch = _chain_batch((0, 0))
    bonus = _distribution({5: 1.0})
    result = sampled_accept_from_distributions(
        batch,
        (target, target, bonus),
        (draft, draft, draft),
        draws=_draw_sequence((0.0, 0.0, 0.0)),
    )
    assert result.accepted_counts == (2,)
    assert result.accepted_tokens == ((0, 0),)
    assert result.next_tokens == (5,)
    assert result.selected_candidate_rows == (2,)


def test_partial_accept_then_rejection_emits_residual_at_that_depth() -> None:
    target = _distribution({0: 1.0})
    accept_all = SparseDistribution.point_mass(0)
    rejecting_target = _distribution({9: 1.0})
    batch = _chain_batch((0, 0))
    result = sampled_accept_from_distributions(
        batch,
        (target, rejecting_target, target),
        (accept_all, accept_all, accept_all),
        draws=_draw_sequence((0.0, 0.5, 0.5)),
    )
    assert result.accepted_counts == (1,)
    assert result.accepted_tokens == ((0,),)
    assert result.next_tokens == (9,)
    assert result.selected_candidate_rows == (1,)


def test_remaining_decode_caps_accepted_drafts_and_keeps_bonus() -> None:
    target = _distribution({0: 1.0})
    draft = SparseDistribution.point_mass(0)
    batch = _chain_batch((0, 0, 0))
    result = sampled_accept_from_distributions(
        batch,
        (target, target, target, target),
        (draft, draft, draft, draft),
        draws=_draw_sequence((0.0, 0.0, 0.0)),
        remaining_decode=(3,),
    )
    # budget 3 means two accepted drafts plus one visible bonus token.
    assert result.accepted_counts == (2,)
    assert result.next_tokens == (0,)


def test_zero_remaining_decode_emits_no_next_token() -> None:
    target = _distribution({0: 1.0})
    draft = SparseDistribution.point_mass(0)
    batch = _chain_batch((0,))
    result = sampled_accept_from_distributions(
        batch,
        (target, target),
        (draft, draft),
        draws=_draw_sequence(()),
        remaining_decode=(0,),
    )
    assert result.accepted_counts == (0,)
    assert result.next_tokens == (None,)


def test_inactive_candidate_rows_are_not_walked() -> None:
    target = _distribution({0: 1.0})
    draft = SparseDistribution.point_mass(0)
    batch = _chain_batch((0, 1))
    batch = TargetVerifyBatch(
        request_ids=batch.request_ids,
        tokens=batch.tokens,
        positions=batch.positions,
        row_to_request=batch.row_to_request,
        parent_rows=batch.parent_rows,
        root_rows=batch.root_rows,
        candidate_rows=batch.candidate_rows,
        draft_depths=batch.draft_depths,
        active_mask=(True, True, False),
    )
    result = sampled_accept_from_distributions(
        batch,
        (target, target, target),
        (draft, draft, draft),
        draws=_draw_sequence((0.0, 0.0)),
    )
    assert result.accepted_counts == (1,)
    assert result.accepted_tokens == ((0,),)
    assert result.next_tokens == (0,)


def test_multiple_requests_walk_independently() -> None:
    target = _distribution({0: 1.0})
    draft = SparseDistribution.point_mass(0)
    batch = TargetVerifyBatch(
        request_ids=(7, 8),
        tokens=(100, 200, 0, 0),
        positions=(0, 0, 1, 1),
        row_to_request=(7, 8, 7, 8),
        parent_rows=(-1, -1, 0, 1),
        root_rows=(0, 1),
        candidate_rows=(2, 3),
        draft_depths=(0, 0, 1, 1),
        active_mask=(True, True, True, True),
    )
    result = sampled_accept_from_distributions(
        batch,
        (target, target, target, target),
        (draft, draft, draft, draft),
        draws=_draw_sequence((0.0, 0.0, 0.0, 0.0)),
    )
    assert result.accepted_counts == (1, 1)
    assert result.accepted_tokens == ((0,), (0,))
    assert result.next_tokens == (0, 0)


def test_tree_shaped_children_are_rejected_as_unsupported() -> None:
    target = _distribution({0: 1.0})
    draft = SparseDistribution.point_mass(0)
    batch = TargetVerifyBatch(
        request_ids=(7,),
        tokens=(100, 0, 0),
        positions=(0, 1, 1),
        row_to_request=(7, 7, 7),
        parent_rows=(-1, 0, 0),
        root_rows=(0,),
        candidate_rows=(1, 2),
        draft_depths=(0, 1, 1),
        active_mask=(True, True, True),
    )
    with pytest.raises(ValueError, match="single drafted chain"):
        sampled_accept_from_distributions(
            batch,
            (target, target, target),
            (draft, draft, draft),
            draws=_draw_sequence((0.0,)),
        )


def test_distribution_alignment_is_validated() -> None:
    target = _distribution({0: 1.0})
    draft = SparseDistribution.point_mass(0)
    batch = _chain_batch((0,))
    with pytest.raises(ValueError, match="target_distributions"):
        sampled_accept_from_distributions(
            batch, (target,), (draft, draft), draws=_draw_sequence(())
        )
    with pytest.raises(ValueError, match="draft_distributions"):
        sampled_accept_from_distributions(
            batch, (target, target), (draft,), draws=_draw_sequence(())
        )
    with pytest.raises(ValueError, match="remaining_decode"):
        sampled_accept_from_distributions(
            batch,
            (target, target),
            (draft, draft),
            draws=_draw_sequence(()),
            remaining_decode=(1, 1),
        )
    with pytest.raises(ValueError, match="remaining_decode"):
        sampled_accept_from_distributions(
            batch,
            (target, target),
            (draft, draft),
            draws=_draw_sequence(()),
            remaining_decode=(-1,),
        )


# --------------------------------------------------------------------------- #
# SparseDistribution contract.
# --------------------------------------------------------------------------- #

def test_sparse_distribution_normalizes_and_looks_up() -> None:
    distribution = SparseDistribution((1, 2, 3), np.asarray([0.25, 0.25, 0.5]))
    assert distribution.token_ids == (1, 2, 3)
    assert float(distribution.probabilities.sum()) == pytest.approx(1.0)
    assert distribution.probability(2) == pytest.approx(0.25)
    assert distribution.probability(9) == 0.0
    assert distribution.as_dict() == {1: 0.25, 2: 0.25, 3: 0.5}


def test_sparse_distribution_from_pairs_merges_duplicates() -> None:
    distribution = SparseDistribution.from_pairs((2, 1, 2), (0.25, 0.5, 0.25))
    assert distribution.token_ids == (1, 2)
    assert distribution.probability(2) == pytest.approx(0.5)


def test_sparse_distribution_rejects_invalid_rows() -> None:
    with pytest.raises(ValueError, match="at least one retained id"):
        SparseDistribution((), np.asarray([], dtype=np.float64))
    with pytest.raises(ValueError, match="unique"):
        SparseDistribution((1, 1), np.asarray([0.5, 0.5]))
    with pytest.raises(ValueError, match="sorted"):
        SparseDistribution((2, 1), np.asarray([0.5, 0.5]))
    with pytest.raises(ValueError, match="align"):
        SparseDistribution((1, 2), np.asarray([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        SparseDistribution((1, 2), np.asarray([1.0, -1.0]))
    with pytest.raises(ValueError, match="positive value"):
        SparseDistribution((1, 2), np.asarray([0.0, 0.0]))


def test_sampling_draw_must_be_a_unit_interval_value() -> None:
    target = _distribution({0: 1.0})
    draft = SparseDistribution.point_mass(0)
    batch = _chain_batch(())
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        sampled_accept_from_distributions(
            batch,
            (target,),
            (draft,),
            draws=_draw_sequence((1.0,)),
            remaining_decode=(1,),
        )


@dataclass(frozen=True)
class _RecordedStream:
    """Replay helper: the walk must consume draws in a documented order."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("a recorded stream needs at least one draw")


def test_walk_consumes_one_draw_per_acceptance_test_then_one_per_sample() -> None:
    target = _distribution({0: 0.5, 1: 0.5})
    draft = SparseDistribution.point_mass(0)
    batch = _chain_batch((0, 0))
    consumed: list[float] = []

    def draw() -> float:
        value = 0.0 if not consumed else 0.0
        consumed.append(value)
        return value

    result = sampled_accept_from_distributions(
        batch,
        (target, target, target),
        (draft, draft, draft),
        draws=draw,
    )
    # two acceptance tests (both accepted) plus the closing bonus sample.
    assert len(consumed) == 3
    assert result.accepted_counts == (2,)
    assert result.next_tokens == (0,)


def test_draw_order_is_deterministic_for_a_recorded_stream() -> None:
    target = _distribution({0: 0.6, 1: 0.4})
    draft = SparseDistribution.point_mass(1)
    batch = _chain_batch((1,))
    first = sampled_accept_from_distributions(
        batch,
        (target, target),
        (draft, draft),
        draws=_draw_sequence((0.5, 0.1, 0.2)),
    )
    second = sampled_accept_from_distributions(
        batch,
        (target, target),
        (draft, draft),
        draws=_draw_sequence((0.5, 0.1, 0.2)),
    )
    assert first == second
    # acceptance probability is p(1) = 0.4, so draw 0.5 rejects and the residual
    # sample at 0.1 picks token 0 (mass 0.6 / 0.6).
    assert first.accepted_counts == (0,)
    assert first.next_tokens == (0,)


# --------------------------------------------------------------------------- #
# Gate self-check: the distribution gate must pass exact couplings and reject a
# wrong residual, so a green gate result means something.
# --------------------------------------------------------------------------- #

def test_distribution_gate_helpers_accept_the_exact_coupling() -> None:
    from scripts.mtp_sampled_accept_distribution_gate import (
        _distribution_metrics,
        _induced_first_token,
        _monte_carlo_check,
    )

    target = _distribution({0: 0.55, 1: 0.25, 2: 0.15, 3: 0.05})
    for drafted in target.token_ids:
        draft = SparseDistribution.point_mass(drafted)
        induced_ids, induced_weights = _induced_first_token(target, draft)
        metrics = _distribution_metrics(induced_ids, induced_weights, target)
        assert metrics["max_abs_deviation"] < 1e-12
        assert metrics["total_variation"] < 1e-12
        assert metrics["top1_agreement"] == 1.0
    monte_carlo = _monte_carlo_check(target, draws=20000, seed=11)
    assert monte_carlo["max_cell_excess"] <= 0.0
    assert monte_carlo["top1_agreement"] == 1.0
    assert monte_carlo["accepted_rate"] == pytest.approx(
        monte_carlo["target_drafted_probability"], abs=0.03
    )


def test_distribution_gate_metrics_reject_a_wrong_residual() -> None:
    """Resampling from p instead of (p - q)+ must be visible to the gate."""

    from scripts.mtp_sampled_accept_distribution_gate import _distribution_metrics

    target = _distribution({0: 0.55, 1: 0.25, 2: 0.15, 3: 0.05})
    draft = SparseDistribution.point_mass(0)
    accepted = target.probability(0)
    wrong_ids = np.asarray(target.token_ids, dtype=np.int64)
    wrong_weights = np.asarray(
        [
            (accepted if token_id == 0 else 0.0)
            + (1.0 - accepted) * target.probability(token_id)
            for token_id in target.token_ids
        ],
        dtype=np.float64,
    )
    metrics = _distribution_metrics(wrong_ids, wrong_weights, target)
    assert metrics["total_variation"] > 0.1
    assert metrics["max_abs_deviation"] > 0.1


def test_distribution_gate_general_coupling_helper_accepts_perturbed_draft() -> None:
    from scripts.mtp_sampled_accept_distribution_gate import (
        _distribution_metrics,
        _induced_first_token,
    )

    target = _distribution({0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1})
    draft = _distribution({1: 0.5, 2: 0.3, 3: 0.2})
    induced_ids, induced_weights = _induced_first_token(target, draft)
    metrics = _distribution_metrics(induced_ids, induced_weights, target)
    assert metrics["max_abs_deviation"] < 1e-12
    assert metrics["total_variation"] < 1e-12
    assert metrics["top1_agreement"] == 1.0
