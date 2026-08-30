#!/usr/bin/env python3
"""Compare MTP draft acceptance across C1-C8 packets, per prompt and per width.

The published matrices report throughput only, so a change in *acceptance* - how many
drafted tokens the verifier keeps - is invisible in them. Acceptance is the lever that
decides whether speculative decoding beats AR at all, and it can regress while every
cell stays content-exact: a rejected draft just falls back to the verified token, so
output equality cannot see it. Only acceptance telemetry can.

Reads two or more ``gguf_mtp_c1c8_server_bench.py`` packets (read-only) and prints, per
prompt, the accepted/drafted counts at each width, plus the per-width aggregate. Use it
to A/B a verify or draft change: ``--baseline`` is the reference packet and every other
label is diffed against it.

Usage:
    python3 scripts/gguf_mtp_acceptance_compare.py \
        --baseline native=/tmp/he-c1c8-k3/c1c8-k3-packet.json \
        serial_exact=/tmp/he-k3-serialexact/k3-serialexact.json \
        [--widths 1,2,4,8] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def _acceptance(payload: Mapping[str, Any], arm: str) -> dict[tuple[str, int], dict[str, int]]:
    """Return {(prompt_id, width): accepted/drafted/cycles/tokens} for one arm."""
    out: dict[tuple[str, int], dict[str, int]] = {}
    for cell in payload.get("cells", ()):
        arm_block = cell.get(arm)
        if not isinstance(arm_block, dict):
            continue
        rows = arm_block.get("rows") or ()
        stats = {
            "accepted": 0,
            "drafted": 0,
            "cycles": 0,
            "lanes": 0,
            "tokens": 0,
        }
        for row in rows:
            mtp = row.get("mtp")
            if not isinstance(mtp, dict) or not mtp.get("used"):
                continue
            stats["accepted"] += int(mtp.get("accepted_draft_tokens") or 0)
            stats["drafted"] += int(mtp.get("draft_tokens") or 0)
            stats["cycles"] += int(mtp.get("draft_cycles") or 0)
            stats["tokens"] += int(mtp.get("accepted_draft_tokens") or 0)
            stats["lanes"] += 1
        if stats["lanes"]:
            out[(str(cell.get("prompt_id")), int(cell["width"]))] = stats
        arm_block_tokens = int(arm_block.get("generated_tokens") or 0)
        if arm_block_tokens:
            stats["tokens"] = arm_block_tokens
    return out


def _route_deltas(payload: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, int]]:
    """Per-arm per-width route counters, reconstructed from cumulative snapshots.

    ``resident_observability.routes.counts`` is a process-global cumulative counter set
    and AR/K3 share a process, so a cell's snapshot includes the other arm's work. It is
    still attributable: the packet's cell list order is execution order, and inside a
    cell ``order`` names which arm's snapshot was taken first, so consecutive differences
    give each arm's work exactly. Validated on four retained packets - every delta comes
    out non-negative (a wrong ordering produces negatives immediately), and two
    independent native-mode K3 packets attribute identically (AR 1831 / MTP 298 packed
    decode steps). The durable fix is bench-side per-cell deltas; see docs/REFACTOR.md.
    """
    keys = (
        "native_packed_decode_steps",
        "native_c1_decode_steps",
        "native_packed_graph_replays",
        "native_full_prefill_rows",
        "native_full_prefill_groups",
    )
    out: dict[tuple[str, int], dict[str, int]] = {}
    prev: Mapping[str, int] = {}
    for cell in payload.get("cells", ()):
        snaps: dict[str, Mapping[str, int]] = {}
        for arm in ("ar", "mtp"):
            block = cell.get(arm)
            if not isinstance(block, Mapping):
                continue
            counts = ((block.get("resident_observability") or {}).get("routes") or {}).get("counts") or {}
            if counts:
                snaps[arm] = counts
        seq = [arm for arm in (cell.get("order") or ()) if arm in snaps] or list(snaps)
        last = prev
        for arm in seq:
            cur = snaps[arm]
            bucket = out.setdefault((arm, int(cell["width"])), {})
            for key in keys:
                bucket[key] = bucket.get(key, 0) + max(0, int(cur.get(key, 0)) - int(last.get(key, 0)))
            last = cur
        prev = last
    return out


def _invariance(payload: Mapping[str, Any], arm_order: tuple[str, ...] = ("ar", "mtp")):
    """Cross-width generated-id gate: same prompt alone vs the same prompt in a lane.

    ``correctness.ar_mtp_equal`` in a cell proves the arms agree at *that* width, so if
    batching moved the target model's argmax, both arms would move together and the cell
    would still pass. Batch-composition invariance needs a cross-width comparison, which
    retained packets never asserted. Lanes are joined to prompts by ``prompt_tokens``
    (per-cell rows carry `usage`); where two suite prompts share a length the join is
    ambiguous, so those are reported as excluded rather than guessed.
    """
    by_len: dict[int, list[str]] = {}
    refs: dict[tuple[str, int], list[int]] = {}
    for cell in payload.get("cells", ()):
        if int(cell.get("width") or 0) != 1:
            continue
        for arm in arm_order:
            for row in ((cell.get(arm) or {}).get("rows") or ()):
                ids = row.get("generated_ids")
                pt = int((row.get("usage") or {}).get("prompt_tokens") or 0)
                if not ids or not pt:
                    continue
                pid = str(cell.get("prompt_id"))
                lengths = by_len.setdefault(pt, [])
                if pid not in lengths:
                    lengths.append(pid)
                refs.setdefault((arm, pt), list(ids))
    ambiguous = sorted(pt for pt, pids in by_len.items() if len(pids) > 1)
    usable = {pt: pids[0] for pt, pids in by_len.items() if len(pids) == 1}
    per_width: dict[int, dict[str, object]] = {}
    for cell in payload.get("cells", ()):
        w = int(cell.get("width") or 0)
        if w <= 1:
            continue
        for arm in arm_order:
            for row in ((cell.get(arm) or {}).get("rows") or ()):
                ids = row.get("generated_ids")
                pt = int((row.get("usage") or {}).get("prompt_tokens") or 0)
                if not ids or pt not in usable:
                    continue
                base = refs.get((arm, pt))
                if not base:
                    continue
                n = min(len(ids), len(base))
                first = next((i for i in range(n) if ids[i] != base[i]), None)
                bucket = per_width.setdefault(w, {"matches": 0, "divergent": 0, "examples": []})
                if first is None:
                    bucket["matches"] = int(bucket["matches"]) + 1
                else:
                    bucket["divergent"] = int(bucket["divergent"]) + 1
                    if len(bucket["examples"]) < 4:
                        bucket["examples"].append(f"{arm}/{usable[pt]}@C{w}:token {first}/{n}")
    return {"per_width": per_width, "prompts_joined": usable, "ambiguous_prompt_lengths": ambiguous}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", required=True, metavar="LABEL=PACKET.json")
    ap.add_argument("comparisons", nargs="*", metavar="LABEL=PACKET.json")
    ap.add_argument("--arm", default="mtp")
    ap.add_argument("--widths", default="", help="comma list, default: all present")
    ap.add_argument("--invariance", action="store_true",
                    help="also gate cross-width generated-id equality")
    ap.add_argument("--routes", action="store_true", help="also print per-arm route deltas")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    def parse(spec: str) -> tuple[str, dict[str, Any]]:
        label, sep, path = spec.partition("=")
        if not sep:
            ap.error(f"--{label}: expected LABEL=PACKET.json")
        return label, json.loads(Path(path).read_text(encoding="utf-8"))

    base_label, base_payload = parse(args.baseline)
    others = [parse(spec) for spec in args.comparisons]
    base = _acceptance(base_payload, args.arm)
    if not base:
        print(f"baseline {base_label}: no {args.arm} acceptance rows", file=sys.stderr)
        return 2
    widths = (
        sorted({w for _, w in base})
        if not args.widths
        else sorted(int(w) for w in args.widths.split(",") if w.strip())
    )
    prompts = sorted({p for p, _ in base})

    for label, payload in [(base_label, base_payload)] + others:
        stats = _acceptance(payload, args.arm)
        acc = sum(s["accepted"] for (p, w), s in stats.items() if w in widths)
        dft = sum(s["drafted"] for (p, w), s in stats.items() if w in widths)
        rate = acc / dft if dft else 0.0
        per_lane = acc / max(sum(s["cycles"] for (p, w), s in stats.items() if w in widths), 1)
        print(f"{label}: accepted {acc}/{dft} = {rate:.4f}  (accepted per cycle {per_lane:.3f})")
    print()

    header = " ".join(f"C{w:<12}" for w in widths)
    print(f"{'prompt':26} {header}")
    for prompt in prompts:
        cells = []
        for w in widths:
            s = base.get((prompt, w))
            if not s or not s["drafted"]:
                cells.append("     -/----    ")
                continue
            rate = s["accepted"] / s["drafted"]
            cells.append(f"{s['accepted']:>4}/{s['drafted']:<4}={rate:.2f} ")
        print(f"{prompt[:26]:26} {' '.join(cells)}")

    if others:
        print("\nper-width aggregate rate vs baseline (positive = comparison accepts more):")
        for label, payload in others:
            stats = _acceptance(payload, args.arm)
            parts = []
            for w in widths:
                b = [s for (p, ww), s in base.items() if ww == w]
                c = [s for (p, ww), s in stats.items() if ww == w]
                if not b or not c or not sum(s["drafted"] for s in b) or not sum(s["drafted"] for s in c):
                    parts.append(f"C{w}=   n/a ")
                    continue
                rb = sum(s["accepted"] for s in b) / sum(s["drafted"] for s in b)
                rc = sum(s["accepted"] for s in c) / sum(s["drafted"] for s in c)
                parts.append(f"C{w}={rc - rb:+.3f} ")
            print(f"  {label:16} {' '.join(parts)}")

    if args.invariance:
        inv = _invariance(base_payload)
        print("\ncross-width generated-id gate (lane output vs the same prompt run alone):")
        print(f"  joined by prompt_tokens: {len(inv['prompts_joined'])} prompts; "
              f"ambiguous lengths excluded: {inv['ambiguous_prompt_lengths'] or 'none'}")
        tot_ok = tot_bad = 0
        for w in sorted(int(k) for k in inv["per_width"]):
            b = inv["per_width"][str(w)] if str(w) in inv["per_width"] else inv["per_width"][w]
            tot_ok += int(b["matches"]); tot_bad += int(b["divergent"])
            flag = f"  {b['examples']}" if b["divergent"] else ""
            print(f"  C{w}: matches={b['matches']} divergent={b['divergent']}{flag}")
        verdict = "PASS" if tot_bad == 0 else "FAIL"
        print(f"  {verdict}: {tot_ok + tot_bad} comparisons, {tot_bad} divergent")

    if args.routes:
        print("\nroute counters attributable per arm (cumulative snapshots differenced "
              "in cell list order):")
        print(f"{'C':>2} | {'AR packed':>9} {'AR c1':>6} {'AR grp':>6} | {'MTP packed':>10} {'MTP c1':>7}")
        routes = _route_deltas(base_payload)
        for w in widths:
            def g(arm: str, key: str, _r: Mapping[str, Any] = routes) -> int:
                return _r.get((arm, w), {}).get(key, 0)
            print(f"{w:>2} | {g('ar', 'native_packed_decode_steps'):>9} "
                  f"{g('ar', 'native_c1_decode_steps'):>6} "
                  f"{g('ar', 'native_full_prefill_groups'):>6} | "
                  f"{g('mtp', 'native_packed_decode_steps'):>10} "
                  f"{g('mtp', 'native_c1_decode_steps'):>7}")

    if args.json_out:
        out = {
            "schema": "hipengine.mtp_acceptance_compare.v1",
            "arm": args.arm,
            "baseline": base_label,
            "widths": widths,
            "per_prompt_per_width": {
                f"{p}|{w}": base.get((p, w)) for p in prompts for w in widths if (p, w) in base
            },
            "packets": {label: str(Path(spec.partition('=')[2])) for label, spec in
                        [args.baseline, *args.comparisons]},
        }
        Path(args.json_out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
