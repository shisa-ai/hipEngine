"""Rows, triage, and the durable store.

An *inventory row* is a mechanically extracted candidate: a flag, a kernel, a
ledger heading, a campaign candidate. A row carries **evidence and never a
verdict**, because an extractor can only see what greps can see and is often
wrong about what it means.

A *triage* decision is a human or agent judgement about a row. It lives in its
own store, keyed by the row's stable id, so re-running an extractor never
destroys it. Each decision records a hash of the evidence it was made against;
when that evidence changes the decision is reported as **stale** and comes back
for review instead of quietly standing on facts that no longer hold.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AUDIT_ROOT = REPO_ROOT / "audit"
INVENTORY_DIR = AUDIT_ROOT / "inventory"      # the standing catalogue of recorded debt
FINDINGS_DIR = AUDIT_ROOT / "findings"        # what code scanning queued up to fix
TRIAGE_DIR = AUDIT_ROOT / "triage"            # decisions, shared by both

#  What a row turns out to be, once someone has looked.
TAGS = (
    "DEAD-FLAG",       # a flag nothing reads, or whose behaviour is now unconditional
    "LOST-OPT",        # a measured win that was never promoted to the default path
    "EXACTNESS-REJECT",  # a candidate rejected for not being bit-exact, which
                         # docs/OPTIMIZATION.md 4.1 says cannot stand alone as a
                         # reason; it must be re-reviewed under the production gate
    "ORPHAN-KERNEL",   # a kernel with no reachable registry key
    "STALE-LEDGER",    # a ledger entry whose referent is gone or whose condition fired
    "UNREACHABLE",     # a path production never selects
    "SKELETON",        # declared but unimplemented
    "GATE-CATCH22",    # a restriction with no command that could lift it
    "DUP-DISPATCH",    # two routes to the same work
    "BENCH-INVALID",   # a retained row that fails the current evidence policy
    "DOC-DRIFT",       # documentation disagrees with the tree
    "DEAD-CODE",
    "TEST-GAP",
    "NOT-DEBT",        # the extractor was wrong; this row is fine as it stands
)

#  hipEngine cleanup verbs. Most debt here resolves by turning something on or
#  deleting it, not by fixing a bug, so the vocabulary differs from a bug tracker.
DISPOSITIONS = (
    "promote",    # make it the default path
    "remove",     # delete the dead flag / path / entry
    "qualify",    # run the gate that is missing, then decide
    "keep",       # justified as-is; the note says why
    "document",   # the code is right, the docs are not
    "defer",      # real, accepted, not scheduled; the note says what unblocks it
    "wontfix",    # a real problem we are deliberately not fixing; the note says why
)

SEVERITIES = ("high", "medium", "low")


def slug(text: str, limit: int = 60) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return out[:limit].rstrip("-") or "unnamed"


def digest(*parts: Any) -> str:
    payload = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class Row:
    """One mechanically extracted candidate."""

    kind: str                                   # flag | kernel | ledger | candidate
    key: str                                    # stable within kind
    title: str
    location: str = ""                          # path or path:line
    evidence: dict[str, Any] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)   # observations, not verdicts
    hints: dict[str, Any] = field(default_factory=dict)  # for re-matching across rescans

    @property
    def id(self) -> str:
        return f"{self.kind}/{self.key}"

    @property
    def evidence_hash(self) -> str:
        """Hash of what a triage decision would have been made against."""
        return digest(self.location, json.dumps(self.evidence, sort_keys=True), sorted(self.signals))

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["id"] = self.id
        out["evidence_hash"] = self.evidence_hash
        return out


@dataclass
class Triage:
    """A durable decision about a row."""

    id: str
    tag: str
    disposition: str
    severity: str = "low"
    note: str = ""
    decided: str = ""
    by: str = ""
    evidence_hash: str = ""
    resolved: bool = False        # the work is done; keep the record
    expires: str = ""             # ISO date after which this decision must be re-confirmed
    last_reviewed: str = ""       # when a human or agent last re-affirmed it
    hints: dict[str, Any] = field(default_factory=dict)   # the row's identity at decision time
    rebound_from: str = ""        # the id this decision carried before a rescan re-matched it

    def is_expired(self, today: str) -> bool:
        return bool(self.expires) and self.expires < today

    def problems(self) -> list[str]:
        out = []
        if self.tag not in TAGS:
            out.append(f"tag {self.tag!r} is not one of {'/'.join(TAGS)}")
        if self.disposition not in DISPOSITIONS:
            out.append(f"disposition {self.disposition!r} is not one of {'/'.join(DISPOSITIONS)}")
        if self.severity not in SEVERITIES:
            out.append(f"severity {self.severity!r} is not one of {'/'.join(SEVERITIES)}")
        if not self.note.strip():
            out.append("note is empty — say why this disposition is right")
        return out


def load_triage() -> dict[str, Triage]:
    """Every recorded decision, keyed by row id."""
    out: dict[str, Triage] = {}
    if not TRIAGE_DIR.exists():
        return out
    for path in sorted(TRIAGE_DIR.glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: invalid JSON — {exc}") from exc
            known = {f.name for f in dataclasses.fields(Triage)}
            out[payload["id"]] = Triage(**{k: v for k, v in payload.items() if k in known})
    return out


def save_triage(decisions: dict[str, Triage]) -> None:
    """Rewrite the store, one file per kind, sorted so diffs stay readable."""
    TRIAGE_DIR.mkdir(parents=True, exist_ok=True)
    by_kind: dict[str, list[Triage]] = {}
    for decision in decisions.values():
        by_kind.setdefault(decision.id.split("/", 1)[0], []).append(decision)
    for kind, items in by_kind.items():
        lines = [json.dumps(dataclasses.asdict(d), sort_keys=True) for d in sorted(items, key=lambda d: d.id)]
        (TRIAGE_DIR / f"{kind}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_rows(directory: pathlib.Path, kind: str | None = None) -> list[Row]:
    rows: list[Row] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.json")):
        if path.name == "meta.json":          # run metadata, not rows
            continue
        if kind and path.stem != kind:
            continue
        for payload in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            rows.append(Row(
                kind=payload["kind"], key=payload["key"], title=payload["title"],
                location=payload.get("location", ""), evidence=payload.get("evidence", {}),
                signals=payload.get("signals", []), hints=payload.get("hints", {}),
            ))
    return rows


def load_inventory(kind: str | None = None) -> list[Row]:
    return load_rows(INVENTORY_DIR, kind)


def load_findings(kind: str | None = None) -> list[Row]:
    return load_rows(FINDINGS_DIR, kind)


def load_all() -> list[Row]:
    """Inventory plus findings. Both share one triage store, so a `wontfix`
    recorded against either is honoured everywhere."""
    return load_inventory() + load_findings()


def save_rows(directory: pathlib.Path, kind: str, rows: list[Row], meta: dict[str, Any]) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}.json"
    payload = {"kind": kind, "meta": meta, "rows": [r.to_dict() for r in sorted(rows, key=lambda r: r.id)]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def save_inventory(kind: str, rows: list[Row], meta: dict[str, Any]) -> pathlib.Path:
    return save_rows(INVENTORY_DIR, kind, rows, meta)


def save_findings(kind: str, rows: list[Row], meta: dict[str, Any]) -> pathlib.Path:
    return save_rows(FINDINGS_DIR, kind, rows, meta)


def similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """How likely two rows are the same audit item, from their match hints.

    Identity must survive ordinary editing. A REFACTOR heading gets reworded, a
    campaign row gains a column, a path is corrected — none of that makes it a
    different item, so matching leans on a stable anchor plus overlap of what
    the row names, not on the text being byte-identical.
    """
    if not a or not b:
        return 0.0
    if a.get("anchor") != b.get("anchor"):
        return 0.0                              # different file: never the same item
    refs_a, refs_b = set(a.get("refs") or ()), set(b.get("refs") or ())
    tokens_a, tokens_b = set(a.get("tokens") or ()), set(b.get("tokens") or ())
    def jaccard(x: set, y: set) -> float | None:
        if not x and not y:
            return None
        return len(x & y) / len(x | y)
    parts = [p for p in (jaccard(refs_a, refs_b), jaccard(tokens_a, tokens_b)) if p is not None]
    return sum(parts) / len(parts) if parts else 0.0


REBIND_THRESHOLD = 0.62


def rebind(rows: list[Row], decisions: dict[str, Triage],
           threshold: float = REBIND_THRESHOLD) -> list[tuple[str, str, float]]:
    """Re-attach decisions whose row id changed because its text was edited.

    Returns `(old id, new id, score)` for each rebind and mutates `decisions`
    in place. Only rows that nothing has decided on are eligible, so a rebind can
    never steal a decision from another item.
    """
    live = {row.id for row in rows}
    homeless = [d for d in decisions.values() if d.id not in live and not d.resolved and d.hints]
    free = [r for r in rows if r.id not in decisions]
    moved: list[tuple[str, str, float]] = []
    for decision in sorted(homeless, key=lambda d: d.id):
        best, best_score = None, threshold
        for row in free:
            if row.kind != decision.id.split("/", 1)[0]:
                continue
            score = similarity(decision.hints, row.hints)
            if score > best_score:
                best, best_score = row, score
        if best is None:
            continue
        old = decision.id
        decisions.pop(old, None)
        decision.rebound_from = old
        decision.id = best.id
        decision.hints = best.hints
        #  The text moved, so the decision must be re-confirmed against it.
        decision.evidence_hash = decision.evidence_hash or ""
        decisions[best.id] = decision
        free.remove(best)
        moved.append((old, best.id, round(best_score, 3)))
    return moved


def expired(decisions: dict[str, Triage], today: str | None = None) -> list[Triage]:
    """Decisions past their `expires` date, which must be re-confirmed."""
    import datetime as _dt
    stamp = today or _dt.date.today().isoformat()
    return sorted((d for d in decisions.values() if not d.resolved and d.is_expired(stamp)),
                  key=lambda d: d.expires)


def reconcile(rows: list[Row], decisions: dict[str, Triage]) -> dict[str, list[Row]]:
    """Split rows by triage state.

    `stale` is the important one: the row was triaged, but the evidence behind
    that decision has since changed, so the conclusion may no longer hold.
    """
    import datetime as _dt
    today = _dt.date.today().isoformat()
    state: dict[str, list[Row]] = {"open": [], "triaged": [], "stale": [], "expired": [], "resolved": []}
    for row in rows:
        decision = decisions.get(row.id)
        if decision is None:
            state["open"].append(row)
        elif decision.resolved:
            state["resolved"].append(row)
        elif decision.is_expired(today):
            state["expired"].append(row)
        elif decision.evidence_hash and decision.evidence_hash != row.evidence_hash:
            state["stale"].append(row)
        else:
            state["triaged"].append(row)
    return state


def orphaned(rows: list[Row], decisions: dict[str, Triage]) -> list[Triage]:
    """Decisions whose row no longer exists — the debt went away, or the key moved."""
    live = {row.id for row in rows}
    return [d for d in decisions.values() if d.id not in live and not d.resolved]
