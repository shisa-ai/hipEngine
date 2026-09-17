"""Per-request MTP-versus-AR output attribution.

Admission, route selection, and execution are three separate questions:

* admission asks whether a request may enter the speculative path at all,
* route selection records which route the frontend asked for,
* execution records what the backend actually did.

A request can be admitted and routed to MTP and still emit every token through
plain autoregressive decoding (a refused prompt activation, a K0 planner
decision, a recoverable cycle failure, or a mid-generation context miss).  A
boolean ``used`` flag cannot describe that; only an output count per execution
mode can.  This module owns the row-level counters that answer it, so every
adapter records them the same way:

* ``mtp2_mtp_output_tokens`` - visible tokens emitted by committed speculative
  cycles (accepted drafts plus the target-verified bonus token),
* ``mtp2_ar_output_tokens`` - visible tokens emitted by a non-speculative
  cycle that still ran through the speculative owner (a K0 row inside a mixed
  plan),
* ``mtp2_ar_step_output_tokens`` - visible tokens emitted by a K0 step of an
  AR-only cycle of the speculative owner,
* ``mtp2_first_fallback_position`` - output index of the first token produced
  after the request left the speculative path,
* ``mtp2_ar_step_reasons`` - non-speculative steps counted under the planner
  reason that selected them.

Autoregressive tokens produced outside the speculative owner (the prefill root
token, or decode steps after the request was dropped from speculation) are
counted by the response layer as ``completion_tokens - mtp2_mtp_output_tokens``
rather than tracked here; that subtraction is what makes the accounting
reconcile against the emitted token list by construction.

Rows are duck-typed: the dense and MoE GGUF adapters use the resident loop row
dataclass, while seam tests drive them with attribute-only doubles.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

REQUESTED_BUDGET_ATTRIBUTE = "mtp2_requested_budget"
MTP_OUTPUT_ATTRIBUTE = "mtp2_mtp_output_tokens"
AR_OUTPUT_ATTRIBUTE = "mtp2_ar_output_tokens"
AR_STEP_OUTPUT_ATTRIBUTE = "mtp2_ar_step_output_tokens"
FIRST_FALLBACK_ATTRIBUTE = "mtp2_first_fallback_position"
AR_STEP_REASON_ATTRIBUTE = "mtp2_ar_step_reasons"


def _row_int(row: Any, name: str) -> int:
    try:
        return max(0, int(getattr(row, name, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _row_mapping(row: Any, name: str) -> Mapping[str, Any]:
    value = getattr(row, name, None)
    return value if isinstance(value, Mapping) else {}


def record_speculative_outputs(
    row: Any,
    *,
    candidate_count: int,
    accepted_count: int,
    visible_count: int,
    output_position: int,
    plan_reason: Any | None = None,
) -> None:
    """Record one committed cycle and attribute its visible tokens.

    One call owns the whole cycle record: the selected depth, the accepted draft
    count, and the visible-token attribution are written together so a reported
    depth histogram can never disagree with the reported output split.

    ``output_position`` is the row's generated-token index before this cycle's
    tokens were appended, so a non-speculative cycle that follows speculative
    output identifies the exact fallback position.
    """

    depth = max(0, int(candidate_count))
    accepted = max(0, int(accepted_count))
    visible = max(0, int(visible_count))
    row.mtp2_cycles = _row_int(row, "mtp2_cycles") + 1
    candidate_counts = getattr(row, "mtp2_candidate_counts", None)
    if not isinstance(candidate_counts, list):
        candidate_counts = []
        setattr(row, "mtp2_candidate_counts", candidate_counts)
    candidate_counts.append(depth)
    accepted_counts = getattr(row, "mtp2_accepted_counts", None)
    if not isinstance(accepted_counts, list):
        accepted_counts = []
        setattr(row, "mtp2_accepted_counts", accepted_counts)
    accepted_counts.append(accepted)
    if visible == 0:
        return
    if depth > 0:
        setattr(
            row,
            MTP_OUTPUT_ATTRIBUTE,
            _row_int(row, MTP_OUTPUT_ATTRIBUTE) + visible,
        )
        return
    setattr(
        row,
        AR_OUTPUT_ATTRIBUTE,
        _row_int(row, AR_OUTPUT_ATTRIBUTE) + visible,
    )
    if (
        _row_int(row, MTP_OUTPUT_ATTRIBUTE) > 0
        and getattr(row, FIRST_FALLBACK_ATTRIBUTE, None) is None
    ):
        setattr(row, FIRST_FALLBACK_ATTRIBUTE, max(0, int(output_position)))
    # The visible tokens of this cycle are attributed above, so the step record
    # must not attribute them a second time.
    record_autoregressive_step(row, plan_reason=plan_reason, emitted_tokens=0)


def record_autoregressive_step(
    row: Any,
    *,
    plan_reason: Any | None,
    emitted_tokens: int = 1,
    output_position: int | None = None,
) -> None:
    """Record one non-speculative decode step, its output, and why it ran.

    Called for the K0 rows of an AR-only plan before the resident owner emits
    their token, and for the K0 rows of a mixed plan from the cycle commit that
    emitted them (which passes ``emitted_tokens=0`` because it already
    attributed the cycle's visible output).

    ``output_position`` is the generated-token index this step's token will
    occupy, so the first step that follows speculative output reports where the
    request left the speculative path. A step that runs before any speculative
    output is not a fallback and leaves the position unset.
    """

    tokens = max(0, int(emitted_tokens))
    if tokens:
        setattr(
            row,
            AR_STEP_OUTPUT_ATTRIBUTE,
            _row_int(row, AR_STEP_OUTPUT_ATTRIBUTE) + tokens,
        )
        if (
            output_position is not None
            and _row_int(row, MTP_OUTPUT_ATTRIBUTE) > 0
            and getattr(row, FIRST_FALLBACK_ATTRIBUTE, None) is None
        ):
            setattr(
                row,
                FIRST_FALLBACK_ATTRIBUTE,
                max(0, int(output_position)),
            )
    reason = None if plan_reason is None else str(plan_reason).strip()
    if not reason:
        return
    counts = getattr(row, AR_STEP_REASON_ATTRIBUTE, None)
    if not isinstance(counts, dict):
        counts = {}
        setattr(row, AR_STEP_REASON_ATTRIBUTE, counts)
    counts[reason] = int(counts.get(reason, 0)) + 1


def _counts(values: Sequence[Any] | None) -> list[int]:
    if not values:
        return []
    return [max(0, int(value)) for value in values]


def speculative_output_accounting(row: Any) -> dict[str, Any] | None:
    """Return the per-request execution accounting block.

    ``None`` means the request never carried speculative intent, so callers must
    not report an MTP-versus-AR split for it at all.
    """

    requested_budget = _row_int(row, REQUESTED_BUDGET_ATTRIBUTE)
    cycles = _row_int(row, "mtp2_cycles")
    prompt_reason = getattr(row, "mtp2_prompt_fallback_reason", None)
    ar_step_reasons = dict(_row_mapping(row, AR_STEP_REASON_ATTRIBUTE))
    recoverable_failures = _row_int(row, "mtp2_recoverable_failures")
    # A request with no speculative intent has no MTP-versus-AR question to
    # answer; refusing to report one is what keeps plain AR responses unchanged.
    if not requested_budget and not cycles:
        return None
    candidate_counts = _counts(getattr(row, "mtp2_candidate_counts", None))
    accepted_counts = _counts(getattr(row, "mtp2_accepted_counts", None))
    depth_histogram = Counter(candidate_counts)
    # Failure reasons are recorded as (category, detail) pairs.
    failure_reasons = [
        str(value) for value in (getattr(row, "mtp2_failure_reasons", None) or ())
    ]
    failure_counts = Counter(failure_reasons[0::2])
    first_fallback = getattr(row, FIRST_FALLBACK_ATTRIBUTE, None)
    return {
        "requested_budget": requested_budget,
        "candidate_budget": _row_int(row, "mtp2_candidate_budget"),
        "prompt_streaming": bool(getattr(row, "mtp2_prompt_streaming", False)),
        "prompt_fallback_reason": (
            None if prompt_reason is None else str(prompt_reason)
        ),
        "cycles": cycles,
        "generated_draft_tokens": sum(candidate_counts),
        "accepted_draft_tokens": sum(accepted_counts),
        "selected_depth_histogram": {
            str(depth): int(count)
            for depth, count in sorted(depth_histogram.items())
        },
        "mtp_output_tokens": _row_int(row, MTP_OUTPUT_ATTRIBUTE),
        # Every autoregressive token this request emitted while it was still
        # owned by the speculative plan: committed by a depth-zero cycle or by
        # a K0 step of an AR-only cycle. The remainder of the response's
        # autoregressive output was emitted after it left the plan.
        "ar_output_tokens_in_cycles": (
            _row_int(row, AR_OUTPUT_ATTRIBUTE)
            + _row_int(row, AR_STEP_OUTPUT_ATTRIBUTE)
        ),
        "first_fallback_position": (
            None if first_fallback is None else max(0, int(first_fallback))
        ),
        "ar_step_reason_counts": {
            str(name): int(value)
            for name, value in sorted(ar_step_reasons.items())
        },
        "recoverable_failures": recoverable_failures,
        "failure_reason_counts": {
            str(name): int(value) for name, value in sorted(failure_counts.items())
        },
        "k0_catchups": _row_int(row, "mtp2_k0_catchups"),
    }


def accounting_timing_fields(accounting: Mapping[str, Any] | None) -> dict[str, float]:
    """Scalar timing mirrors for the response ``timing`` map.

    Only committed cycles publish these fields: a request that carried
    speculative intent but ran no speculative cycle must stay indistinguishable
    from plain AR in ``timing``, or a zero-cycle refusal would be reported as
    realized MTP.
    """

    if not accounting:
        return {}
    cycles = max(0, int(accounting.get("cycles") or 0))
    if cycles <= 0:
        return {}
    generated = max(0, int(accounting.get("generated_draft_tokens") or 0))
    accepted = max(0, int(accounting.get("accepted_draft_tokens") or 0))
    return {
        "mtp_cycles_count": float(cycles),
        "mtp_generated_draft_tokens": float(generated),
        "mtp_accepted_draft_tokens": float(accepted),
        "mtp_visible_output_tokens": float(
            max(0, int(accounting.get("mtp_output_tokens") or 0))
        ),
        "mtp_ar_output_tokens": float(
            max(0, int(accounting.get("ar_output_tokens_in_cycles") or 0))
        ),
        "mtp_accept_per_draft": (float(accepted) / float(generated) if generated else 0.0),
    }
