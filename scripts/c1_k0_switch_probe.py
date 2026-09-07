"""Pure planning/validation logic for the K0<->MTP lifecycle switch probe.

The probe drives sequential single requests through the actual product server
on one resident owner, alternating explicit-MTP and automatic-K0 legs, and
proves from outside the API that every switch direction keeps outputs exact,
engages only the MTP legs, and drains cleanly. This module owns the leg
planning and the per-prompt verdict so the GPU runner stays thin and the
rules are unit-testable.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

LEGS_PER_PROMPT = 4


def plan_legs(prompt_index: int, legs_per_prompt: int = LEGS_PER_PROMPT) -> tuple[str, ...]:
    """Alternate explicit-MTP and automatic-K0 legs, counterbalanced start.

    Even prompt indices start with MTP (switch chain M->K->M->K); odd indices
    start with K0 (K->M->K->M), so both switch directions are exercised first
    across the suite and every prompt contains two of each direction.
    """

    if legs_per_prompt < 2 or legs_per_prompt % 2 != 0:
        raise ValueError("legs_per_prompt must be a positive even number")
    start_mtp = prompt_index % 2 == 0
    return tuple(
        ("mtp" if (step % 2 == 0) == start_mtp else "k0")
        for step in range(legs_per_prompt)
    )


def _leg_verdict(
    leg: str,
    row: Mapping[str, Any],
    *,
    budget: int,
) -> list[str]:
    """Return the failure reasons for one leg, or an empty list."""

    reasons: list[str] = []
    summary = row.get("mtp")
    summary = summary if isinstance(summary, Mapping) else {}
    used = bool(summary.get("used"))
    cycles = int(summary.get("draft_cycles", 0) or 0)
    generated = int(summary.get("draft_tokens", 0) or 0)
    if leg == "mtp":
        if not used or cycles <= 0 or generated <= 0:
            reasons.append("mtp_leg_not_engaged")
        elif generated > budget * cycles:
            reasons.append("mtp_leg_budget_exceeded")
    else:
        if used or cycles > 0 or generated > 0:
            reasons.append("k0_leg_engaged")
    ids = row.get("generated_ids")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or not ids:
        reasons.append("missing_generated_ids")
    return reasons


def validate_sequence(
    legs: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    budget: int,
) -> list[str]:
    """Validate one prompt's switch sequence; return all failure reasons.

    Rules: the rows follow the planned alternating legs; every MTP leg engages
    inside its budget; every K0 leg stays on plain AR; and all legs of the
    prompt emit identical token IDs (greedy determinism, so an MTP leg that
    diverges from the AR legs proves broken provider catch-up across a K0
    switch in either direction).
    """

    if len(legs) != len(rows):
        return [f"row_count_{len(rows)}_planned_{len(legs)}"]
    reasons: list[str] = []
    for index, (leg, row) in enumerate(zip(legs, rows)):
        for reason in _leg_verdict(leg, row, budget=budget):
            reasons.append(f"leg{index}_{leg}_{reason}" if leg == "mtp" else f"leg{index}_{reason}")
    sequences = [tuple(row.get("generated_ids") or ()) for row in rows]
    if all(sequences) and any(seq != sequences[0] for seq in sequences):
        diverging = [
            index for index, seq in enumerate(sequences) if seq != sequences[0]
        ]
        reasons.append(f"divergent_legs_{','.join(map(str, diverging))}")
    return reasons


def summarize_switches(legs: Sequence[str]) -> dict[str, int]:
    """Count the switch directions exercised by one planned sequence."""

    transitions = {"mtp_to_k0": 0, "k0_to_mtp": 0}
    for before, after in zip(legs, legs[1:]):
        if before == "mtp" and after == "k0":
            transitions["mtp_to_k0"] += 1
        elif before == "k0" and after == "mtp":
            transitions["k0_to_mtp"] += 1
    return transitions
