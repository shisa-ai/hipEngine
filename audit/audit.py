#!/usr/bin/env python3
"""hipEngine cleanup audit.

Inventory the debt mechanically, triage it durably, and gate against it growing.

    python3 audit/audit.py inventory          # re-extract every inventory
    python3 audit/audit.py status             # where the cleanup stands
    python3 audit/audit.py open kernels -n 20 # untriaged rows, most signals first
    python3 audit/audit.py show flag/HIPENGINE_X
    python3 audit/audit.py triage flag/HIPENGINE_X --tag LOST-OPT \
        --do promote --severity high --note "why"
    python3 audit/audit.py check              # the gate
    python3 audit/audit.py report             # write a dated run

Stdlib only. See audit/README.md for the model.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hipaudit import core, report                     # noqa: E402
from hipaudit.core import Row, Triage                 # noqa: E402
from hipaudit.inventory import EXTRACTORS             # noqa: E402


def run_inventory(kinds: list[str] | None) -> dict:
    meta: dict = {}
    for name, extract in sorted(EXTRACTORS.items()):
        if kinds and name not in kinds:
            continue
        rows, info = extract()
        path = core.save_inventory(name, rows, info)
        meta[name] = info
        print(f"{name:12} {len(rows):5} rows -> {path.relative_to(core.REPO_ROOT)}")
    meta_path = core.INVENTORY_DIR / "meta.json"
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    existing.update(meta)
    existing["generated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    existing["commit"] = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=core.REPO_ROOT,
        capture_output=True, text=True).stdout.strip()
    meta_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return existing


def load_meta() -> dict:
    path = core.INVENTORY_DIR / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def cmd_inventory(args) -> int:
    run_inventory(args.kind or None)
    return 0


def cmd_status(args) -> int:
    rows, decisions = core.load_inventory(), core.load_triage()
    if not rows:
        print("no inventory yet — run `python3 audit/audit.py inventory`", file=sys.stderr)
        return 1
    meta = load_meta()
    print(f"inventory generated {meta.get('generated', '?')} at {meta.get('commit', '?')}\n")
    print(report.state_table(rows, decisions))
    budget = report.load_budget().get("open", {})
    if budget:
        state = core.reconcile(rows, decisions)
        current: dict[str, int] = {}
        for row in state["open"]:
            current[row.kind] = current.get(row.kind, 0) + 1
        over = {k: (current.get(k, 0), v) for k, v in budget.items() if current.get(k, 0) > v}
        print("\nbudget: " + ("over in " + ", ".join(f"{k} {a}>{b}" for k, (a, b) in over.items())
                              if over else "within budget"))
    stale = core.reconcile(rows, decisions)["stale"]
    if stale:
        print(f"\n{len(stale)} triaged row(s) have changed since the decision was made — "
              f"`audit.py open --stale` to review")
    return 0


def _rank(row: Row) -> tuple:
    return (-len(row.signals), row.id)


def cmd_open(args) -> int:
    rows, decisions = core.load_inventory(), core.load_triage()
    if args.kind:                     # accept either the row kind or the extractor name
        want = args.kind.rstrip("s")
        rows = [r for r in rows if r.kind == want]
    state = core.reconcile(rows, decisions)
    picked = state["stale"] if args.stale else state["open"]
    if args.signal:
        picked = [r for r in picked if any(args.signal.lower() in s.lower() for s in r.signals)]
    picked.sort(key=_rank)
    label = "stale" if args.stale else "untriaged"
    print(f"{len(picked)} {label} row(s)" + (f" matching {args.signal!r}" if args.signal else ""))
    for row in picked[:args.number]:
        print(f"\n{row.id}\n  {row.title[:100]}\n  {row.location}")
        for signal in row.signals:
            print(f"    - {signal}")
    if len(picked) > args.number:
        print(f"\n... {len(picked) - args.number} more; raise -n or narrow with --signal")
    return 0


def cmd_show(args) -> int:
    rows = {r.id: r for r in core.load_inventory()}
    row = rows.get(args.id)
    if row is None:
        print(f"no such row: {args.id}", file=sys.stderr)
        return 1
    print(json.dumps(row.to_dict(), indent=2))
    decision = core.load_triage().get(args.id)
    if decision:
        print("\ntriage:")
        print(json.dumps(dataclasses.asdict(decision), indent=2))
    return 0


def cmd_triage(args) -> int:
    rows = {r.id: r for r in core.load_inventory()}
    row = rows.get(args.id)
    if row is None:
        print(f"no such row: {args.id} — run `inventory` first, or check `open`", file=sys.stderr)
        return 1
    decisions = core.load_triage()
    decision = Triage(
        id=args.id, tag=args.tag, disposition=args.do, severity=args.severity,
        note=args.note, decided=dt.date.today().isoformat(),
        by=args.by or subprocess.run(["git", "config", "user.name"], capture_output=True,
                                     text=True).stdout.strip() or "unknown",
        evidence_hash=row.evidence_hash, resolved=args.resolved,
    )
    problems = decision.problems()
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    decisions[args.id] = decision
    core.save_triage(decisions)
    print(f"recorded {args.tag}/{args.do} for {args.id}")
    return 0


def cmd_check(args) -> int:
    errors: list[str] = []
    rows, decisions = core.load_inventory(), core.load_triage()
    if not rows:
        print("error: no inventory — run `python3 audit/audit.py inventory`", file=sys.stderr)
        return 1

    for decision in decisions.values():
        for problem in decision.problems():
            errors.append(f"{decision.id}: {problem}")

    #  The inventory must reflect the tree, or every count below is fiction.
    before = {p.name: p.read_bytes() for p in core.INVENTORY_DIR.glob("*.json") if p.name != "meta.json"}
    run_inventory(None)
    after = {p.name: p.read_bytes() for p in core.INVENTORY_DIR.glob("*.json") if p.name != "meta.json"}
    drifted = sorted(k for k in after if before.get(k) != after[k])
    if drifted and not args.refresh:
        errors.append(
            "inventory was stale and has been regenerated: " + ", ".join(drifted)
            + " — review the new rows and commit them")

    rows, decisions = core.load_inventory(), core.load_triage()
    state = core.reconcile(rows, decisions)
    budget = report.load_budget().get("open", {})
    current: dict[str, int] = {}
    for row in state["open"]:
        current[row.kind] = current.get(row.kind, 0) + 1
    for kind, allowed in budget.items():
        if current.get(kind, 0) > allowed:
            errors.append(
                f"{kind}: {current[kind]} untriaged rows exceeds the budget of {allowed}. "
                f"Triage the new rows, or raise the budget with a recorded reason.")

    print(report.state_table(rows, decisions))
    if state["stale"]:
        print(f"\nwarning: {len(state['stale'])} triaged row(s) changed since the decision — "
              f"`audit.py open --stale`")
    for message in errors:
        print(f"error: {message}", file=sys.stderr)
    if errors:
        return 1
    print("\naudit: ok")
    return 0


def cmd_report(args) -> int:
    rows, decisions = core.load_inventory(), core.load_triage()
    if not rows:
        print("no inventory — run `inventory` first", file=sys.stderr)
        return 1
    run = report.write_run(rows, decisions, load_meta())
    print(f"wrote {run.relative_to(core.REPO_ROOT)}/REPORT.md")
    return 0


def cmd_budget(args) -> int:
    rows, decisions = core.load_inventory(), core.load_triage()
    payload = report.save_budget(rows, decisions)
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inventory", help="re-extract the inventories")
    p.add_argument("kind", nargs="*", choices=sorted(EXTRACTORS) or None)
    p.set_defaults(fn=cmd_inventory)

    p = sub.add_parser("status", help="where the cleanup stands")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("open", help="rows nobody has triaged, most signals first")
    p.add_argument("kind", nargs="?", default=None)
    p.add_argument("-n", "--number", type=int, default=15)
    p.add_argument("--signal", help="only rows whose observations mention this")
    p.add_argument("--stale", action="store_true", help="rows whose evidence changed after triage")
    p.set_defaults(fn=cmd_open)

    p = sub.add_parser("show", help="one row, with its triage")
    p.add_argument("id")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("triage", help="record a decision about a row")
    p.add_argument("id")
    p.add_argument("--tag", required=True, choices=core.TAGS)
    p.add_argument("--do", required=True, choices=core.DISPOSITIONS, help="disposition")
    p.add_argument("--severity", default="low", choices=core.SEVERITIES)
    p.add_argument("--note", required=True, help="why this disposition is right")
    p.add_argument("--by", default="")
    p.add_argument("--resolved", action="store_true", help="the work is already done")
    p.set_defaults(fn=cmd_triage)

    p = sub.add_parser("check", help="the gate")
    p.add_argument("--refresh", action="store_true",
                   help="regenerate the inventory without failing on drift")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("report", help="write a dated run under audit/runs/")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("budget", help="record the current untriaged counts as the ceiling")
    p.set_defaults(fn=cmd_budget)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
