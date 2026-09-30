#!/usr/bin/env python3
"""Re-derive every verdict in this artifact from its raw numbers.

Run next to `artifact.json`. Nothing here trusts a recorded verdict, including
`hard_gates_passed`: the envelope is recomputed from the per-arm KL and top-1
values against the recorded thresholds, so a gate-revision change that silently
relaxed a bar would show up as a disagreement rather than pass unnoticed.

A role is only read as "passing alone" when both hold:

1. The arm claimed something (``candidate_dispatch_calls > 0``). An arm that
   claimed nothing is the route-off control wearing a role name; its green
   envelope is not evidence about that role.
2. Its envelope clears every binding threshold.

Three roles in this sweep cannot be claimed by the route at all, so they are
reported inert rather than as passes.
"""

from __future__ import annotations

import json
import pathlib
import sys

ARTIFACT = pathlib.Path(__file__).resolve().parent / "artifact.json"

ENVELOPE = (
    ("kl_mean", "mean_kl_max", "le"),
    ("kl_p95", "p95_kl_max", "le"),
    ("kl_p99", "p99_kl_max", "le"),
    ("kl_max", "max_kl_max", "le"),
    ("top1_agreement", "top1_min", "ge"),
)


def rederive(env: dict, thresholds: dict) -> tuple[bool, list[str]]:
    fails: list[str] = []
    for key, limit_key, cmp_ in ENVELOPE:
        v = env.get(key)
        limit = thresholds.get(limit_key)
        if v is None or limit is None:
            fails.append(f"{key}: missing")
            continue
        if cmp_ == "le" and v > limit:
            fails.append(f"{key}={v:.6g} > {limit}")
        if cmp_ == "ge" and v < limit:
            fails.append(f"{key}={v:.6g} < {limit}")
    return (not fails), fails


def main() -> int:
    if not ARTIFACT.exists():
        print(f"missing {ARTIFACT}", file=sys.stderr)
        return 2
    doc = json.loads(ARTIFACT.read_text())
    arms = doc["arms"]
    thresholds = doc["thresholds"]

    print("=" * 104)
    print("CONTROLS")
    print("=" * 104)
    header = (f"{'arm':14s} {'claims':>7s} {'kl_mean':>10s} {'kl_p95':>10s} "
              f"{'kl_p99':>10s} {'kl_max':>10s} {'top1':>8s}  verdict")
    print(header)

    disagreements: list[str] = []
    for name in ("baseline", "full_route"):
        a = arms[name]
        env = a["envelope"]
        ok, fails = rederive(env, thresholds)
        if a["hard_gates_passed"] is not None and a["hard_gates_passed"] != ok:
            disagreements.append(
                f"{name}: recorded hard_gates_passed={a['hard_gates_passed']} "
                f"but re-derived {ok} ({'; '.join(fails) or 'clean'})"
            )
        print(f"{name:14s} {a['candidate_dispatch_calls'] or 0:7d} "
              f"{env['kl_mean']:10.6f} {env['kl_p95']:10.6f} {env['kl_p99']:10.6f} "
              f"{env['kl_max']:10.6f} {env['top1_agreement']:8.5f}  "
              f"{'PASS' if ok else 'FAIL ' + '; '.join(fails)}")

    t = thresholds
    print(f"\nbinding envelope: mean<={t['mean_kl_max']} p95<={t['p95_kl_max']} "
          f"p99<={t['p99_kl_max']} max<={t['max_kl_max']} top1>={t['top1_min']}")

    b_ok, _ = rederive(arms["baseline"]["envelope"], thresholds)
    f_ok, _ = rederive(arms["full_route"]["envelope"], thresholds)
    if not b_ok:
        print("!! baseline does not pass - the sweep's reference is invalid")
    if f_ok:
        print("!! full route PASSES - it does not reproduce the recorded "
              "rejection; the sweep is invalid")

    print()
    print("=" * 104)
    print("ROLE ARMS")
    print("=" * 104)
    print(header)
    rows = []
    for name in sorted(n for n in arms if n.startswith("role_")):
        a = arms[name]
        env = a["envelope"]
        ok, fails = rederive(env, thresholds)
        n = a["candidate_dispatch_calls"] or 0
        if a["hard_gates_passed"] is not None and a["hard_gates_passed"] != ok:
            disagreements.append(
                f"{a['roles']}: recorded hard_gates_passed="
                f"{a['hard_gates_passed']} but re-derived {ok}"
            )
        if n == 0:
            verdict = "INERT (claimed nothing - not evidence about this role)"
        elif ok:
            verdict = "PASS (claims and clears every binding threshold)"
        else:
            verdict = "FAIL " + "; ".join(fails)
        rows.append((a["roles"], n, env, ok, verdict))
        print(f"{a['roles']:14s} {n:7d} {env['kl_mean']:10.6f} "
              f"{env['kl_p95']:10.6f} {env['kl_p99']:10.6f} {env['kl_max']:10.6f} "
              f"{env['top1_agreement']:8.5f}  {verdict}")

    inert = [e for r, n, e, ok, v in rows if n == 0]
    if inert:
        first = inert[0]
        agree = all(
            abs(e["kl_mean"] - first["kl_mean"]) < 1e-12
            and abs(e["kl_max"] - first["kl_max"]) < 1e-12
            and abs(e["top1_agreement"] - first["top1_agreement"]) < 1e-12
            for e in inert
        )
        i_ok, i_fails = rederive(first, thresholds)
        print()
        print("-" * 104)
        print(f"inert-arm control ({len(inert)} arms, mutually identical: {agree})")
        print(f"  kl_mean={first['kl_mean']:.6f} kl_p95={first['kl_p95']:.6f} "
              f"kl_p99={first['kl_p99']:.6f} kl_max={first['kl_max']:.6f} "
              f"top1={first['top1_agreement']:.5f}  -> "
              f"{'PASS' if i_ok else 'FAIL ' + '; '.join(i_fails)}")
        if not agree:
            print("  !! inert arms disagree - they are not one configuration")
        if not i_ok:
            print("  !! the route-off control fails - the route is not the sole cause")
        print("  (this is the correct route-off control: the candidate inherits from")
        print("   production_gr_up_restore and so runs production defaults for every")
        print("   flag it does not name, unlike production_conservative which zeroes")
        print("   all conservative flags)")

    print()
    print("=" * 104)
    print("SUMMARY")
    print("=" * 104)
    print(f"verdict            : {doc['verdict']}")
    print(f"claiming + passing : {[r for r, n, e, ok, v in rows if n > 0 and ok] or 'none'}")
    print(f"claiming + failing : {[r for r, n, e, ok, v in rows if n > 0 and not ok]}")
    print(f"inert (no claims)  : {[r for r, n, e, ok, v in rows if n == 0]}")
    shapes = arms["full_route"].get("candidate_dispatch_shapes") or []
    print(f"\nfull route claims {len(shapes)} shapes, "
          f"{sum(s['calls'] for s in shapes)} calls: "
          f"{sorted({(s['arguments'][1], s['arguments'][2]) for s in shapes})}")

    print()
    if disagreements:
        print("!! RE-DERIVATION DISAGREEMENTS:")
        for d in disagreements:
            print(f"   {d}")
        return 1
    print("re-derivation agrees with every recorded hard_gates_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
