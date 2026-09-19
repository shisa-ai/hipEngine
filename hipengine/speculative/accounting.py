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
import os
from typing import Any, Mapping, Sequence

REQUESTED_BUDGET_ATTRIBUTE = "mtp2_requested_budget"
MTP_OUTPUT_ATTRIBUTE = "mtp2_mtp_output_tokens"
AR_OUTPUT_ATTRIBUTE = "mtp2_ar_output_tokens"
AR_STEP_OUTPUT_ATTRIBUTE = "mtp2_ar_step_output_tokens"
FIRST_FALLBACK_ATTRIBUTE = "mtp2_first_fallback_position"
AR_STEP_REASON_ATTRIBUTE = "mtp2_ar_step_reasons"
SPAN_ATTRIBUTE = "mtp2_output_spans"
PROMPT_FALLBACK_ATTRIBUTE = "mtp2_prompt_fallback_reason"
PROVIDER_READINESS_ATTRIBUTE = "mtp2_provider_readiness"
PROVIDER_DECLINE_ATTRIBUTE = "mtp2_provider_decline_reason"
PLAN_GROUP_ROWS_ATTRIBUTE = "mtp2_plan_group_rows"
PLAN_AR_ONLY_ATTRIBUTE = "mtp2_plan_ar_only"
PLAN_REASON_ATTRIBUTE = "mtp2_plan_reason"

# Readiness states for ``PROVIDER_READINESS_ATTRIBUTE``. They are deliberately
# a closed set: a reader must be able to tell a row whose prompt activation was
# refused from one whose activation never ran, because both used to report the
# planner's ``no_provider`` while meaning opposite things.
PROVIDER_READY = "ready"
PROVIDER_PROMPT_BUFFERED = "prompt_buffered"
PROVIDER_ABSENT = "absent"
PROVIDER_DECLINED = "declined"
PROVIDER_READINESS_STATES = (
    PROVIDER_READY,
    PROVIDER_PROMPT_BUFFERED,
    PROVIDER_ABSENT,
    PROVIDER_DECLINED,
)

# Where a row's draft-provider state comes from. This is the routing rule the
# refusal paths answer to: an unsupported priming source is a property of the
# row on this route and keeps it autoregressive for its whole life, while
# temporary contention is a property of the moment and must not, because whether
# the row can speculate is decided by its source rather than by the refusal.
PRIMING_SOURCE_LIVE = "live_provider_state"
PRIMING_SOURCE_BUFFERED = "prompt_buffer"
PRIMING_SOURCE_ABSENT = "none"
PRIMING_SOURCES = (
    PRIMING_SOURCE_LIVE,
    PRIMING_SOURCE_BUFFERED,
    PRIMING_SOURCE_ABSENT,
)

# Diagnostic switch for committed output spans. Off by default: the span list is
# attribution evidence for an audit, not production telemetry, and it grows with
# the cycle count.
OUTPUT_SPANS_ENV = "HIPENGINE_MTP2_OUTPUT_SPANS"
_ENV_TRUE = frozenset({"1", "true", "yes", "on"})


def output_span_recording_enabled() -> bool:
    """Return whether committed output spans are recorded for this process."""

    value = os.environ.get(OUTPUT_SPANS_ENV)
    if value is None:
        return False
    return str(value).strip().lower() in _ENV_TRUE


def _record_span(
    row: Any,
    *,
    mode: str,
    reason: Any | None,
    position: int,
    tokens: int,
) -> None:
    """Append one committed output span when diagnostic recording is on.

    A span is the unit of attribution: which execution mode emitted the tokens,
    why that mode ran, and where in the emitted output they sit. The counters
    above answer the same question in aggregate; the spans are what make the
    aggregate checkable against the emitted token list instead of against a
    subtraction.
    """

    if tokens <= 0 or not output_span_recording_enabled():
        return
    spans = getattr(row, SPAN_ATTRIBUTE, None)
    if not isinstance(spans, list):
        spans = []
        try:
            setattr(row, SPAN_ATTRIBUTE, spans)
        except AttributeError:
            # A slots row that does not declare the span field. Diagnostic
            # recording must never break generation, so the span is dropped and
            # the response reports that no proof was recorded.
            return
    spans.append(
        {
            "mode": str(mode),
            "reason": None if reason is None else str(reason),
            "position": max(0, int(position)),
            "tokens": int(tokens),
        }
    )


def _row_int(row: Any, name: str) -> int:
    try:
        return max(0, int(getattr(row, name, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _row_mapping(row: Any, name: str) -> Mapping[str, Any]:
    value = getattr(row, name, None)
    return value if isinstance(value, Mapping) else {}


def _row_float(row: Any, name: str) -> float:
    try:
        return max(0.0, float(getattr(row, name, 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


# Per-row cycle phase windows, all measured around host wall on the serving
# thread. They are the phase split of a committed cycle, and they are host wall
# rather than device-busy: a window that ends by reading a result includes the
# wait for that result. Read them beside ``cycles``; a sum without the cycle
# count cannot be turned into a per-cycle cost.
CYCLE_PHASE_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("proposal", "mtp2_proposal_ms"),
    ("target", "mtp2_target_ms"),
    ("provider_update", "mtp2_provider_update_ms"),
    ("accept", "mtp2_accept_ms"),
    ("candidate_readback", "mtp2_candidate_readback_ms"),
    ("target_readback", "mtp2_target_readback_ms"),
    ("accept_upload", "mtp2_accept_upload_ms"),
    ("accept_tail", "mtp2_accept_tail_ms"),
    ("accept_enqueue", "mtp2_accept_enqueue_ms"),
    ("selected_commit", "mtp2_selected_commit_ms"),
)


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
        _record_span(
            row,
            mode="mtp",
            reason=plan_reason,
            position=output_position,
            tokens=visible,
        )
        return
    setattr(
        row,
        AR_OUTPUT_ATTRIBUTE,
        _row_int(row, AR_OUTPUT_ATTRIBUTE) + visible,
    )
    _record_span(
        row,
        mode="ar",
        reason=plan_reason,
        position=output_position,
        tokens=visible,
    )
    if (
        _row_int(row, MTP_OUTPUT_ATTRIBUTE) > 0
        and getattr(row, FIRST_FALLBACK_ATTRIBUTE, None) is None
    ):
        setattr(row, FIRST_FALLBACK_ATTRIBUTE, max(0, int(output_position)))
    # The visible tokens of this cycle are attributed above, so the step record
    # must not attribute them a second time.
    record_autoregressive_step(row, plan_reason=plan_reason, emitted_tokens=0)


def record_speculative_plan(
    row: Any,
    *,
    group_rows: int,
    ar_only: bool,
    plan_reason: Any | None,
) -> None:
    """Record the group-level plan decision this row ran under.

    The plan is the only place that knows the row's realized group width and
    whether the group as a whole was planned autoregressively. Both used to be
    unobservable per request: the served response reported a route-level
    ``k0_class`` (which says what the route intended, not what the plan chose)
    and a serving-key width, so a row that ran 127 autoregressive steps beside
    a ``not_k0`` route claim looked self-contradictory rather than refused.
    """

    setattr(row, PLAN_GROUP_ROWS_ATTRIBUTE, max(0, int(group_rows)))
    setattr(row, PLAN_AR_ONLY_ATTRIBUTE, bool(ar_only))
    setattr(
        row,
        PLAN_REASON_ATTRIBUTE,
        None if plan_reason is None else str(plan_reason),
    )


def record_provider_readiness(
    row: Any,
    *,
    readiness: str,
    decline_reason: Any | None = None,
) -> None:
    """Record whether the row's draft provider was ready, and why not.

    ``readiness`` is one of ``PROVIDER_READINESS_STATES``. ``decline_reason``
    is retained only for a non-ready state so a stale reason cannot outlive a
    row that has since acquired a provider.
    """

    state = str(readiness)
    if state not in PROVIDER_READINESS_STATES:
        raise ValueError(
            f"provider readiness must be one of {PROVIDER_READINESS_STATES}, got {state!r}"
        )
    setattr(row, PROVIDER_READINESS_ATTRIBUTE, state)
    setattr(
        row,
        PROVIDER_DECLINE_ATTRIBUTE,
        None
        if state == PROVIDER_READY or decline_reason is None
        else str(decline_reason),
    )


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
        if output_position is not None:
            _record_span(
                row,
                mode="ar",
                reason=plan_reason,
                position=output_position,
                tokens=tokens,
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


def _row_spans(row: Any) -> list[dict[str, Any]]:
    """Return the row's committed output spans as normalized dictionaries."""

    spans = getattr(row, SPAN_ATTRIBUTE, None)
    if not isinstance(spans, list):
        return []
    normalized: list[dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, Mapping):
            continue
        mode = str(span.get("mode") or "")
        if mode not in {"mtp", "ar"}:
            continue
        reason = span.get("reason")
        normalized.append(
            {
                "mode": mode,
                "reason": None if reason is None else str(reason),
                "position": max(0, int(span.get("position") or 0)),
                "tokens": max(0, int(span.get("tokens") or 0)),
            }
        )
    return normalized


def span_accounting(
    spans: Sequence[Mapping[str, Any]],
    *,
    completion_tokens: int,
    mtp_output_tokens: int,
    ar_output_tokens: int,
    ar_output_tokens_in_cycles: int,
) -> dict[str, Any]:
    """Check committed output spans against the reported MTP-versus-AR split.

    The aggregate counters reconcile arithmetically by construction, because
    autoregressive output is defined as whatever speculation did not cover. The
    spans are independent evidence: each committed span records the execution
    mode, the reason it ran, the emitted-token position it starts at, and how
    many tokens it emitted, so together they must tile the committed output with
    no gap and no overlap, every speculative token must be inside a span, and
    the spanned autoregressive tokens must be exactly the in-plan ones.

    Spans need not reach the completion count. A request can emit autoregressive
    tokens outside the speculative plan — the prompt-prefill root token before
    it, and the final token when the decode that produced it retires the row —
    and those are deliberately not recorded as spans. The exact requirement is
    the identity ``unspanned == ar_output_tokens - ar_output_tokens_in_cycles``:
    every emitted token is either spanned or an autoregressive token produced
    outside the plan, so a speculative token can never hide in the remainder.
    """

    normalized = [
        {
            "mode": str(span.get("mode") or ""),
            "reason": (
                None
                if span.get("reason") is None
                else str(span.get("reason"))
            ),
            "position": max(0, int(span.get("position") or 0)),
            "tokens": max(0, int(span.get("tokens") or 0)),
        }
        for span in spans
        if isinstance(span, Mapping)
    ]
    tokens_total = sum(span["tokens"] for span in normalized)
    mtp_tokens = sum(
        span["tokens"] for span in normalized if span["mode"] == "mtp"
    )
    ar_tokens = sum(span["tokens"] for span in normalized if span["mode"] == "ar")
    ar_tokens_by_reason: Counter[str] = Counter()
    for span in normalized:
        if span["mode"] == "ar":
            ar_tokens_by_reason[span["reason"] or ""] += span["tokens"]
    first_position = normalized[0]["position"] if normalized else None
    last_end = (
        max(span["position"] + span["tokens"] for span in normalized)
        if normalized
        else None
    )
    contiguous = True
    cursor: int | None = None
    for span in normalized:
        if cursor is not None and span["position"] != cursor:
            contiguous = False
            break
        cursor = span["position"] + span["tokens"]
    unspanned = int(completion_tokens) - tokens_total
    expected_unspanned = max(0, int(ar_output_tokens)) - int(
        ar_output_tokens_in_cycles
    )
    reasons: list[str] = []
    if not normalized:
        reasons.append("no_committed_spans")
    else:
        if not contiguous:
            reasons.append("spans_not_contiguous")
        if unspanned < 0:
            reasons.append("spans_exceed_the_completion_count")
    if mtp_tokens != int(mtp_output_tokens):
        reasons.append("span_mtp_tokens_do_not_match_mtp_output")
    if ar_tokens != int(ar_output_tokens_in_cycles):
        reasons.append("span_ar_tokens_do_not_match_in_cycle_ar_output")
    if unspanned != expected_unspanned:
        # Some emitted token is neither spanned nor an autoregressive token from
        # outside the plan, so the split cannot be traced to committed work.
        reasons.append("unspanned_tokens_are_not_autoregressive")
    return {
        "spans": len(normalized),
        "tokens": tokens_total,
        "mtp_tokens": mtp_tokens,
        "ar_tokens": ar_tokens,
        "ar_tokens_by_reason": {
            reason: int(count) for reason, count in sorted(ar_tokens_by_reason.items())
        },
        "unspanned_tokens": unspanned,
        "expected_unspanned_ar_tokens": expected_unspanned,
        "first_position": first_position,
        "last_end": last_end,
        "contiguous": contiguous,
        "mtp_tokens_match": mtp_tokens == int(mtp_output_tokens),
        "ar_in_cycle_tokens_match": ar_tokens
        == int(ar_output_tokens_in_cycles),
        "unspanned_tokens_match": unspanned == expected_unspanned,
        "reconciled": not reasons,
        "reconciled_reasons": reasons,
    }


def speculative_output_accounting(row: Any) -> dict[str, Any] | None:
    """Return the per-request execution accounting block.

    ``None`` means the request never carried speculative intent, so callers must
    not report an MTP-versus-AR split for it at all.
    """

    requested_budget = _row_int(row, REQUESTED_BUDGET_ATTRIBUTE)
    cycles = _row_int(row, "mtp2_cycles")
    prompt_reason = getattr(row, PROMPT_FALLBACK_ATTRIBUTE, None)
    provider_readiness = getattr(row, PROVIDER_READINESS_ATTRIBUTE, None)
    provider_decline = getattr(row, PROVIDER_DECLINE_ATTRIBUTE, None)
    plan_group_rows = getattr(row, PLAN_GROUP_ROWS_ATTRIBUTE, None)
    plan_ar_only = getattr(row, PLAN_AR_ONLY_ATTRIBUTE, None)
    plan_reason = getattr(row, PLAN_REASON_ATTRIBUTE, None)
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
    block = {
        "requested_budget": requested_budget,
        "candidate_budget": _row_int(row, "mtp2_candidate_budget"),
        "prompt_streaming": bool(getattr(row, "mtp2_prompt_streaming", False)),
        "prompt_fallback_reason": (
            None if prompt_reason is None else str(prompt_reason)
        ),
        # The four facts a refusal diagnosis needs, each published on its own so
        # no one has to reconstruct them from the folded ``fallback_reason``:
        # why this row's prompt activation was refused, whether its provider was
        # ready, how wide the group it was planned in actually was, and whether
        # that group was planned autoregressively.
        "activation_reason": (
            None if prompt_reason is None else str(prompt_reason)
        ),
        "provider_readiness": (
            None if provider_readiness is None else str(provider_readiness)
        ),
        "provider_decline_reason": (
            None if provider_decline is None else str(provider_decline)
        ),
        "plan_group_rows": (
            None if plan_group_rows is None else max(0, int(plan_group_rows))
        ),
        "plan_ar_only": None if plan_ar_only is None else bool(plan_ar_only),
        "plan_reason": None if plan_reason is None else str(plan_reason),
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
        # Host-wall phase sums for the committed cycles. Published here because
        # the row already carries them and the response did not: the phase split
        # was measured but unreachable, so every attribution question needed a
        # new profiler run against a route that might not even be the one under
        # test. Divide by ``cycles`` for a per-cycle cost.
        "cycle_timing_ms": {
            name: _row_float(row, attribute)
            for name, attribute in CYCLE_PHASE_ATTRIBUTES
        },
    }
    if output_span_recording_enabled():
        # Diagnostic only: the span list is added, never substituted for the
        # counters above, so a response read without it is unchanged.
        block["output_spans"] = _row_spans(row)
    return block


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
