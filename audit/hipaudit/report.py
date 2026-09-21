"""Computed tables and the dated run report.

Every number here is derived from the inventory and triage stores when the
command runs, so a count in a table cannot drift from the rows behind it.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import pathlib
from typing import Any

from .core import AUDIT_ROOT, REPO_ROOT, Row, Triage, load_inventory, load_triage, reconcile, orphaned

RUNS_DIR = AUDIT_ROOT / "runs"
BUDGET = AUDIT_ROOT / "budget.json"


def table(header: list[str], body: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in body:
        lines.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(lines)


def state_table(rows: list[Row], decisions: dict[str, Triage]) -> str:
    by_kind: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        decision = decisions.get(row.id)
        if decision is None:
            by_kind[row.kind]["open"] += 1
        elif decision.resolved:
            by_kind[row.kind]["resolved"] += 1
        elif decision.evidence_hash and decision.evidence_hash != row.evidence_hash:
            by_kind[row.kind]["stale"] += 1
        else:
            by_kind[row.kind]["triaged"] += 1
    body = []
    total = collections.Counter()
    for kind in sorted(by_kind):
        counts = by_kind[kind]
        total.update(counts)
        body.append([kind, sum(counts.values()), counts["open"], counts["triaged"],
                     counts["stale"], counts["resolved"]])
    body.append(["**total**", sum(total.values()), total["open"], total["triaged"],
                 total["stale"], total["resolved"]])
    return table(["Kind", "Rows", "Open", "Triaged", "Stale", "Resolved"], body)


def tag_table(decisions: dict[str, Triage]) -> str:
    counts = collections.Counter(d.tag for d in decisions.values())
    if not counts:
        return "_No rows triaged yet._"
    return table(["Tag", "Rows"], [[tag, n] for tag, n in counts.most_common()])


def disposition_table(decisions: dict[str, Triage]) -> str:
    counts = collections.Counter(d.disposition for d in decisions.values())
    if not counts:
        return "_No rows triaged yet._"
    return table(["Disposition", "Rows"], [[d, n] for d, n in counts.most_common()])


def act_first(rows: list[Row], decisions: dict[str, Triage], limit: int = 20) -> str:
    """Triaged, unresolved, highest severity first."""
    order = {"high": 0, "medium": 1, "low": 2}
    live = {r.id: r for r in rows}
    items = [d for d in decisions.values() if not d.resolved and d.id in live]
    items.sort(key=lambda d: (order.get(d.severity, 3), d.disposition, d.id))
    if not items:
        return "_No triaged, unresolved rows._"
    body = [[d.severity, d.tag, d.disposition, f"`{live[d.id].location}`", live[d.id].title[:70]]
            for d in items[:limit]]
    return table(["Severity", "Tag", "Do", "Location", "Row"], body)


def signal_table(rows: list[Row], decisions: dict[str, Triage], limit: int = 15) -> str:
    """The most common observations across rows nobody has triaged yet."""
    counts = collections.Counter()
    for row in rows:
        if row.id in decisions:
            continue
        for signal in row.signals:
            counts[signal.split(" — ")[0].split(":")[0][:80]] += 1
    if not counts:
        return "_No untriaged rows._"
    return table(["Observation on untriaged rows", "Rows"],
                 [[s, n] for s, n in counts.most_common(limit)])


def load_budget() -> dict[str, Any]:
    if BUDGET.exists():
        return json.loads(BUDGET.read_text(encoding="utf-8"))
    return {}


def gate_spec(budget: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per-kind gate selectors, as recorded in the budget's `select` block.

    A spec is `{field: value-or-list}` with AND semantics; a list of specs is
    ORed. `evidence.<name>` reads the row's evidence, anything else a row field.
    """
    payload = load_budget() if budget is None else budget
    select = payload.get("select", {})
    return select if isinstance(select, dict) else {}


def in_gate(row: Row, spec: Any) -> bool:
    """Whether the row counts toward its kind's ceiling. No spec = everything counts."""
    if spec is None:
        return True
    if isinstance(spec, list):
        return any(in_gate(row, item) for item in spec)
    for key, wanted in spec.items():
        if key.startswith("evidence."):
            value = row.evidence.get(key.split(".", 1)[1])
        else:
            value = getattr(row, key, None)
        if isinstance(wanted, list):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


def open_counts(rows: list[Row], decisions: dict[str, Triage],
                budget: dict[str, Any] | None = None) -> tuple[collections.Counter, collections.Counter]:
    """`(gated, ungated)` untriaged rows per kind.

    A kind whose population grows with normal work selects the rows that need
    action; the rest stay untriaged and visible without setting the ceiling.
    """
    select = gate_spec(budget)
    gated: collections.Counter = collections.Counter()
    ungated: collections.Counter = collections.Counter()
    for row in rows:
        if row.id in decisions:
            continue
        (gated if in_gate(row, select.get(row.kind)) else ungated)[row.kind] += 1
    return gated, ungated


def save_budget(rows: list[Row], decisions: dict[str, Triage],
                lower_only: bool = False) -> dict[str, Any]:
    """Record the untriaged ceiling.

    `lower_only` is what an automatic refresh uses: cleanup ratchets the budget
    down, but only a person may raise it, and only with a recorded reason.
    Otherwise a refresh would quietly absorb every new piece of debt.

    Only rows inside their kind's gate set the ceiling; a row outside it is
    still untriaged, and still shows up in the state table.
    """
    counts, _ = open_counts(rows, decisions)
    payload = load_budget()          #  keep `select`: a policy is not a count
    if lower_only:
        previous = payload.get("open", {})
        for kind, was in previous.items():
            #  Keep every previously recorded kind, including at zero: dropping a
            #  fully triaged kind from the payload would leave its next untriaged
            #  row ungated. `min` is the ratchet - a refresh may lower a ceiling,
            #  never raise one.
            counts[kind] = min(counts.get(kind, 0), was)
    payload.update({
        "recorded": dt.date.today().isoformat(),
        "note": "Untriaged rows per kind. `audit.py check` fails when a count rises. "
                "Lower it by triaging rows; raise it only with a recorded reason. "
                "`select` narrows a kind's gate to matching rows; rows outside the gate "
                "stay visible in the state table.",
        "open": dict(sorted(counts.items())),
    })
    BUDGET.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def write_run(rows: list[Row], decisions: dict[str, Triage], meta: dict[str, Any]) -> pathlib.Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = RUNS_DIR / stamp
    run.mkdir(parents=True, exist_ok=True)
    state = reconcile(rows, decisions)
    lost = orphaned(rows, decisions)

    body = f"""# hipEngine cleanup audit — {stamp}

Generated by `python3 audit/audit.py report`. Do not edit; re-run to refresh.

Inventory rows are mechanically extracted candidates carrying evidence, never
verdicts. A row becomes a finding when someone triages it. See
[`../../README.md`](../../README.md).

## State

{state_table(rows, decisions)}

## Act on first

{act_first(rows, decisions)}

## Tags

{tag_table(decisions)}

## Dispositions

{disposition_table(decisions)}

## What the extractors observed, on rows nobody has triaged

{signal_table(rows, decisions)}

## Extractor metadata

{table(["Extractor", "Detail"], [[k, json.dumps(v)] for k, v in sorted(meta.items())])}

## Decisions whose row disappeared

{len(lost)} triaged row(s) are no longer produced by any extractor. Either the debt
was removed, or an extractor key changed. Confirm before marking them resolved.

{table(["Row", "Tag", "Do"], [[d.id, d.tag, d.disposition] for d in lost[:20]]) if lost else "_None._"}

## What this does not establish

The extractors see what greps see. They do not resolve dispatch, run kernels, or
measure anything. A row with no signals is not thereby healthy, and a row with
several is not thereby debt — {len(state['open'])} rows have never been read by
anyone. This report states what is inventoried, not what is true.
"""
    (run / "REPORT.md").write_text(body, encoding="utf-8")
    snapshot = {
        "stamp": stamp,
        "rows": len(rows),
        "open": len(state["open"]),
        "triaged": len(state["triaged"]),
        "stale": len(state["stale"]),
        "by_kind": dict(collections.Counter(r.kind for r in rows)),
        "tags": dict(collections.Counter(d.tag for d in decisions.values())),
        "dispositions": dict(collections.Counter(d.disposition for d in decisions.values())),
    }
    (run / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return run
