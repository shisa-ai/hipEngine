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
ARCHIVE_DIR = TRIAGE_DIR / "archive"          # decisions whose row is gone; never matched again

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


# The hash covers what a decision rests on, not how big things are. Positions,
# sizes and age move without anything changing. Counts and reader lists move with
# every ordinary commit (a new test that mentions a kernel, a tenth script that
# reads a flag), and the signals already state the part a decision depends on:
# whether a runtime caller exists, whether any test names it, whether branching
# is spread out. Hashing the counts too made most "stale" decisions noise.
VOLATILE_EVIDENCE = frozenset({
    "line", "lines", "age_days", "body_chars",            # position, size, age
    "referrers", "runtime_referrers", "test_referrers",   # kernels: signals carry presence
    "sites_by_root", "runtime_sites",                      # flags: signals carry roots and spread
    "continuation_entries",                                # worklog: signal carries presence
})
VOLATILE_SIGNAL = re.compile(r"\b\d+ days? old\b|\bwritten \d+ days ago\b")
# Counts inside the audit's own signal templates. Numbers anywhere else in a
# signal (a quoted dependency, a phase) are content and still age a decision.
COUNTED_SIGNAL = re.compile(
    r"\bread at \d+ runtime sites\b|^\d+ later entr(?:y|ies)\b|^\d+ __global__ entry point"
    r"|\bnames \d+ path\(s\)|\b\d+ no longer exist\b")


def _uncount(match: re.Match) -> str:
    text = re.sub(r"\d+", "#", match.group(0))
    return re.sub(r"later entr(?:y|ies)", "later entries", text)


def basis_signals(signals: list[str]) -> list[str]:
    """Ignore elapsed age and template counts, not other numbers in a signal."""
    return sorted({COUNTED_SIGNAL.sub(_uncount, s) for s in signals if not VOLATILE_SIGNAL.search(s)})


def basis_value(value: Any) -> Any:
    """Drop line positions in structured evidence, retaining referenced paths."""
    if isinstance(value, dict):
        return {k: basis_value(v) for k, v in value.items() if k != "line"}
    if isinstance(value, list):
        return [basis_value(v) for v in value]
    return value


def compact_basis(basis: dict[str, Any]) -> dict[str, Any]:
    """A basis small enough to keep on a decision: long text becomes its digest."""
    def shrink(value: Any) -> Any:
        if isinstance(value, str) and len(value) > 160:
            return f"sha:{digest(value)}"
        if isinstance(value, dict):
            return {k: shrink(v) for k, v in value.items()}
        if isinstance(value, list):
            return [shrink(v) for v in value]
        return value
    return shrink(basis)


def basis_changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """What moved between the basis a decision recorded and the row now."""
    out: list[str] = []
    if before.get("title") != after.get("title"):
        out.append(f"title: {before.get('title')!r} -> {after.get('title')!r}")
    old, new = before.get("evidence", {}), after.get("evidence", {})
    for name in sorted(old.keys() | new.keys()):
        if old.get(name) != new.get(name):
            out.append(f"{name}: {json.dumps(old.get(name))[:80]} -> {json.dumps(new.get(name))[:80]}")
    was, now = set(before.get("signals", [])), set(after.get("signals", []))
    out += [f"- {s}" for s in sorted(was - now)] + [f"+ {s}" for s in sorted(now - was)]
    return out


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
        """Hash of the facts a triage decision rests on.

        Positions, sizes, elapsed age and counts do not make a decision stale.
        A changed title, default, ledger or docs entry, missing path, or signal
        (a runtime caller appearing or going away, say) does.
        """
        return digest(json.dumps(self.basis(), sort_keys=True))

    def basis(self) -> dict[str, Any]:
        """What the row says, minus what moves when nothing about it changed."""
        evidence: dict[str, Any] = {}
        for name, value in self.evidence.items():
            if name in VOLATILE_EVIDENCE:
                continue
            if name == "flag_signals":
                value = basis_signals(value)
            evidence[name] = basis_value(value)
        return {"title": self.title, "evidence": evidence, "signals": basis_signals(self.signals)}

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
    archived: str = ""            # when its row vanished and the decision left the live store
    basis: dict[str, Any] = field(default_factory=dict)   # compact_basis() it was decided against

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
        if self.expires:
            import datetime as dt
            try:
                parsed = dt.date.fromisoformat(self.expires)
                if parsed.isoformat() != self.expires:
                    raise ValueError(self.expires)
            except ValueError:
                out.append("expires must be a valid YYYY-MM-DD date")
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
    for kind in by_kind.keys() | {p.stem for p in TRIAGE_DIR.glob("*.jsonl")}:
        items = by_kind.get(kind, [])
        lines = [json.dumps(dataclasses.asdict(d), sort_keys=True) for d in sorted(items, key=lambda d: d.id)]
        (TRIAGE_DIR / f"{kind}.jsonl").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
    if len({row.id for row in rows}) != len(rows):
        raise ValueError(f"{kind}: duplicate row IDs; refusing to overwrite decision identities")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}.json"
    payload = {"kind": kind, "meta": meta, "rows": [r.to_dict() for r in sorted(rows, key=lambda r: r.id)]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def save_inventory(kind: str, rows: list[Row], meta: dict[str, Any]) -> pathlib.Path:
    return save_rows(INVENTORY_DIR, kind, rows, meta)


def save_findings(kind: str, rows: list[Row], meta: dict[str, Any]) -> pathlib.Path:
    return save_rows(FINDINGS_DIR, kind, rows, meta)


def _ref_identity(refs: set[str]) -> set[str]:
    """Refs with their line number dropped.

    A finding's ref is ``path:line`` or ``path:line:detail``. Inserting a
    line above moves either form without changing its referent. Strip only the
    first line-number field so numeric fields in the detail remain distinct.
    """

    return {_LINE_REF.sub("", ref, count=1) for ref in refs}


_LINE_REF = re.compile(r":\d+(?=:|$)")


def similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """How likely two rows are the same audit item, from their match hints.

    Identity must survive ordinary editing. A REFACTOR heading gets reworded, a
    campaign row gains a column, a path is corrected, code moves down a file —
    none of that makes it a different item, so matching leans on a stable anchor
    plus overlap of what the row names, not on the text being byte-identical.

    Sharing only a file is weak evidence: two findings in one file differ exactly
    by the code each points at, which is what their tokens carry. So a ref match
    that needs the line number dropped counts half, and the token overlap has to
    carry the rest. Ref matches that keep their line numbers are unchanged.
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
    parts: list[float] = []
    exact_refs = jaccard(refs_a, refs_b)
    if exact_refs is not None:
        if exact_refs > 0.0:
            parts.append(exact_refs)
        else:
            moved = jaccard(_ref_identity(refs_a), _ref_identity(refs_b)) or 0.0
            parts.append(0.5 * moved)
    tokens = jaccard(tokens_a, tokens_b)
    if tokens is not None:
        parts.append(tokens)
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
    # Each decision must have one best site. Several old decisions may agree
    # about that site, but conflicting judgments must never depend on scan order.
    candidates = list(free)
    scores = {
        (decision.id, row.id): similarity(decision.hints, row.hints)
        for decision in homeless
        for row in free
        if row.kind == decision.id.split("/", 1)[0]
    }
    for decision in sorted(homeless, key=lambda d: d.id):
        eligible = [(row, scores[(decision.id, row.id)]) for row in candidates
                    if scores.get((decision.id, row.id), 0.0) > threshold]
        if not eligible:
            continue
        best_score = max(score for _, score in eligible)
        best_rows = [row for row, score in eligible if score == best_score]
        if len(best_rows) != 1:
            continue
        best = best_rows[0]
        if best not in free:
            continue
        competing = [(other.id, scores[(other.id, best.id)]) for other in homeless
                     if scores.get((other.id, best.id), 0.0) > threshold]
        row_best = max(score for _, score in competing)
        winners = [other for other in homeless
                   if scores.get((other.id, best.id), 0.0) == row_best]
        # Agreement is not just a tag: do not lose severity or a review deadline.
        verdicts = {(d.tag, d.disposition, d.severity, d.expires) for d in winners}
        if len(verdicts) != 1:
            continue
        winner = max(winners, key=lambda d: (d.last_reviewed or d.decided, d.decided, d.id))
        if winner is not decision:
            continue
        old = decision.id
        decisions.pop(old, None)
        decision.rebound_from = old
        decision.id = best.id
        decision.hints = best.hints
        # Preserve the original evidence and review date. Matching is not review.
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
        elif not decision.resolved and decision.is_expired(today):
            state["expired"].append(row)
        elif decision.evidence_hash and decision.evidence_hash != row.evidence_hash:
            state["stale"].append(row)
        elif decision.resolved:
            state["resolved"].append(row)
        else:
            state["triaged"].append(row)
    return state


def orphaned(rows: list[Row], decisions: dict[str, Triage]) -> list[Triage]:
    """Decisions whose row no longer exists — the debt went away, or the key moved."""
    live = {row.id for row in rows}
    return [d for d in decisions.values() if d.id not in live and not d.resolved]


def load_archive() -> list[Triage]:
    out: list[Triage] = []
    known = {f.name for f in dataclasses.fields(Triage)}
    for path in sorted(ARCHIVE_DIR.glob("*.jsonl")) if ARCHIVE_DIR.exists() else []:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                out.append(Triage(**{k: v for k, v in payload.items() if k in known}))
    return out


def archive_orphans(rows: list[Row], decisions: dict[str, Triage],
                    today: str | None = None) -> list[Triage]:
    """Move decisions whose row no longer exists out of the live store.

    Run only after a complete inventory and findings scan and `rebind`.
    Decisions that could match an open row stay live for manual review; this
    preserves conflicting or ambiguous candidates. Other absent decisions move
    to the archive without claiming the underlying work was resolved. The
    archive is history, never an automatic source of decisions for new rows.
    """
    import datetime as _dt
    stamp = today or _dt.date.today().isoformat()
    live = {row.id for row in rows}
    free = [r for r in rows if r.id not in decisions]
    gone = [d for d in decisions.values() if d.id not in live and not any(
        r.kind == d.id.split("/", 1)[0]
        and similarity(d.hints, r.hints) > REBIND_THRESHOLD for r in free)]
    if not gone:
        return []
    by_kind: dict[str, list[Triage]] = {}
    def identity(decision: Triage) -> str:
        return json.dumps(dataclasses.asdict(dataclasses.replace(decision, archived="")), sort_keys=True)

    known = set()
    for decision in load_archive():
        by_kind.setdefault(decision.id.split("/", 1)[0], []).append(decision)
        known.add(identity(decision))
    for decision in gone:
        decisions.pop(decision.id, None)
        decision.archived = stamp
        if identity(decision) not in known:
            by_kind.setdefault(decision.id.split("/", 1)[0], []).append(decision)
            known.add(identity(decision))
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for kind, items in by_kind.items():
        lines = [json.dumps(dataclasses.asdict(d), sort_keys=True)
                 for d in sorted(items, key=lambda d: (d.id, d.archived, d.decided))]
        (ARCHIVE_DIR / f"{kind}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sorted(gone, key=lambda d: d.id)
