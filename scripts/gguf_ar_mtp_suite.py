#!/usr/bin/env python3
"""Reliable apple-to-apple AR vs MTP end-to-end suite for the GGUF 35B-A3B path.

ONE entry point that measures the true no-MTP autoregressive baseline and the
MTP speculative path under a single, enforced, recorded decode configuration,
then emits ONE consolidated artifact with both rows, the MTP/AR ratio per draft
budget, per-category detail, and full provenance.

Why this exists
---------------
Historically AR and MTP were measured by two separate scripts and compared
post-hoc via ``--true-ar-baseline-json``. The apple-to-apple invariants (same
decode kernels, same prompt construction, matched warmup, matched token regime)
were *advisory*. This wrapper makes them *enforced*: it pins one canonical decode
config, drives both validated benches with it, asserts the protocol + prompt
hashes match, and fails loudly otherwise. It reuses the proven measurement code
in ``gguf_true_ar_category_bench.py`` and ``gguf_mtp_category_bench.py`` (which
already validate the timing protocol) rather than reimplementing timing.

Canonical decode config (production path, recorded in every artifact)
---------------------------------------------------------------------
- ``HIPENGINE_GGUF_DECODE_REPACK=1`` (T16 decode-repack; required path)
- ``--decode-repack --use-gemv-decode --use-wmma-prefill`` (all on)
- eager decode (the HIP decode graph was retired; see WORKLOG #8)
- greedy / temp 0
- ``--prompt-reasoning off`` forced on BOTH sides (the one real apple-to-apple
  prompt-construction knob the underlying benches did not force symmetrically)

Scope presets
-------------
- ``smoke``   : 1 prompt,  3 cycles, budget [3]            (~2 model loads; quick sanity)
- ``partial`` : 4 prompts, 5 cycles, budgets [1,3,5]       (category-representative)
- ``full``    : all prompts (10), 10 cycles, budgets [1..5] (retainable claim; SLOW,
                ~prompts*budgets child model loads -- use --reuse-existing to resume)

Each MTP child reloads the model in its own process (faithful + isolated); the
full suite is therefore long. Use ``smoke``/``partial`` for iteration.

Example
-------
    PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/gguf_ar_mtp_suite.py \
        --scope partial \
        --output benchmarks/results/2026-06-29-ar-mtp-suite-partial.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl"

# Canonical production decode config -- enforced identically on AR and MTP.
CANONICAL_DECODE = {
    "decode_repack": True,
    "use_gemv_decode": True,
    "use_wmma_prefill": True,
    "eager_decode": True,
    "greedy": True,
    "prompt_reasoning": "off",
    "warmup_decode_tokens": 1,
}

# Named MTP routes (mirrors gguf_mtp_parity_workbench candidates). The default is
# the production selector. Every route's exact extra-args are recorded.
MTP_ROUTES: dict[str, list[str]] = {
    "resident-production": [
        "--resident-mtp-draft",
        "--adaptive-block-after-full-accept",
        "--adaptive-probe-draft-n-max", "3",
        "--adaptive-ar-fallback",
    ],
    "resident-serial-fallback": [
        "--resident-mtp-draft",
        "--adaptive-ar-fallback",
        "--no-target-block-verify",
    ],
    "resident-cap32k-recover": [
        "--resident-mtp-draft",
        "--adaptive-ar-fallback",
        "--no-target-block-verify",
        "--mtp-draft-vocab-cap", "32768",
        "--adaptive-full-vocab-after-cap-miss",
    ],
    "resident-strict-context": [
        "--resident-mtp-draft",
        "--root-topk-accept", "1",
        "--sibling-topk-accept", "1",
        "--mtp-context-replay",
        "--mtp-device-kv-cache",
        "--no-target-block-verify",
    ],
    "resident-draft": ["--resident-mtp-draft"],
    "resident-block": ["--resident-mtp-draft", "--target-block-verify"],
}

SCOPES = {
    "smoke": {"limit": 1, "cycles": 3, "budgets": [3]},
    "partial": {"limit": 4, "cycles": 5, "budgets": [1, 3, 5]},
    "full": {"limit": None, "cycles": 10, "budgets": [1, 2, 3, 4, 5]},
}


class SuiteError(RuntimeError):
    pass


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _hardware() -> str:
    return os.environ.get("HIPENGINE_HW_LABEL") or (
        "AMD Radeon 8060S / Ryzen AI Max+ 395 (gfx1151)"
    )


def _run(cmd: list[str], env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "\n".join(log.read_text().splitlines()[-25:])
        raise SuiteError(f"child failed ({proc.returncode}): {' '.join(cmd)}\n--- tail ---\n{tail}")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _enforce_apple_to_apple(ar: dict[str, Any], mtp: dict[str, Any]) -> list[str]:
    """Assert the AR and MTP runs share the decode protocol + prompt set."""
    problems: list[str] = []
    tp = ar.get("timing_protocol", {})
    for key in ("decode_repack", "use_gemv_decode", "use_wmma_prefill"):
        if not tp.get(key, False):
            problems.append(f"AR timing_protocol.{key} is not True ({tp.get(key)!r})")
    if tp.get("decode_path") not in (None, "eager_step"):
        problems.append(f"AR decode_path is {tp.get('decode_path')!r}, expected eager_step")
    # Prompt-set hash parity: the MTP summary records prompt sha256 per prompt;
    # the AR baseline records prompt_metrics[].prompt_sha256. Compare the sets.
    ar_hashes = {r.get("prompt_sha256") for r in ar.get("prompt_metrics", []) if r.get("prompt_sha256")}
    mtp_hashes = {p.get("prompt_sha256") for p in mtp.get("prompts", []) if p.get("prompt_sha256")}
    if mtp_hashes and ar_hashes and ar_hashes != mtp_hashes:
        problems.append(
            f"prompt-set hash mismatch: AR has {len(ar_hashes)} prompts, MTP has {len(mtp_hashes)}, "
            f"intersection {len(ar_hashes & mtp_hashes)}"
        )
    return problems


def _ar_row(ar: dict[str, Any]) -> dict[str, Any]:
    totals = ar.get("totals", {})
    return {
        "decode_tok_s_weighted": totals.get("decode_tok_s_weighted"),
        "total_output_tokens": totals.get("total_output_tokens"),
        "decode_ms": totals.get("decode_ms"),
        "prompts": totals.get("prompts"),
        "per_category": {
            c: v.get("decode_tok_s_weighted")
            for c, v in (ar.get("categories", {}) or {}).items()
        },
    }


def _mtp_rows(mtp: dict[str, Any]) -> dict[str, Any]:
    """Extract the per-budget MTP rows + AR ratio from the category summary.

    The category summary stores ``totals`` as a budget-label-keyed dict
    (``totals["b3"] -> row``); each row carries the metrics + the attached
    ``mtp_vs_true_ar_decode_ratio``.
    """
    rows: dict[str, Any] = {}
    totals = mtp.get("totals", {})
    if not isinstance(totals, dict):
        return rows
    for label, row in totals.items():
        if not isinstance(row, dict):
            continue
        # The category bench also emits an "off"/"b0" verifier-derived AR proxy
        # row. Per the anti-gaming rules those are diagnostic only and must NOT
        # count as an MTP result; keep only real positive draft budgets (b1..bN).
        digits = "".join(ch for ch in str(label) if ch.isdigit())
        is_real_budget = str(label).lower().startswith("b") and digits and int(digits) >= 1
        rows[str(label)] = {
            "diagnostic_only": not is_real_budget,
            "decode_tok_s_weighted": row.get("decode_tok_s_weighted"),
            "accepted_per_output": row.get("accepted_per_output"),
            "draft_acceptance": row.get("draft_acceptance"),
            "mtp_vs_true_ar_decode_ratio": row.get("mtp_vs_true_ar_decode_ratio"),
            "true_ar_decode_tok_s_weighted": row.get("true_ar_decode_tok_s_weighted"),
            "total_accepted": row.get("total_accepted"),
            "total_output_tokens": row.get("total_output_tokens"),
        }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scope", choices=tuple(SCOPES), default="smoke")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--mtp-route", choices=tuple(MTP_ROUTES), default="resident-serial-fallback")
    ap.add_argument("--budgets", default=None, help="override scope budgets, e.g. 1,3,5")
    ap.add_argument("--cycles", type=int, default=None, help="override scope cycles")
    ap.add_argument("--limit", type=int, default=None, help="override scope prompt limit")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--raw-root", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--timestamp", default=None, help="ISO stamp for the artifact (default: now)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scope = SCOPES[args.scope]
    cycles = args.cycles if args.cycles is not None else scope["cycles"]
    limit = args.limit if args.limit is not None else scope["limit"]
    budgets = (
        [int(x) for x in args.budgets.split(",")] if args.budgets else list(scope["budgets"])
    )
    route_args = MTP_ROUTES[args.mtp_route]
    # Some routes carry a fixed --adaptive-probe-draft-n-max N that gguf_mtp_bench
    # requires N <= draft budget; drop incompatible low budgets so a multi-budget
    # sweep doesn't error (e.g. resident-production probes at 3, so B1/B2 are out).
    _probe = None
    for i, a in enumerate(route_args):
        if a == "--adaptive-probe-draft-n-max" and i + 1 < len(route_args):
            _probe = int(route_args[i + 1])
    dropped_budgets: list[int] = []
    if _probe is not None:
        dropped_budgets = [b for b in budgets if b < _probe]
        if dropped_budgets:
            print(f"[ar-mtp-suite] route '{args.mtp_route}' needs draft budget >= probe "
                  f"{_probe}; dropping budgets {dropped_budgets}", flush=True)
            budgets = [b for b in budgets if b >= _probe]
    if not budgets:
        raise SuiteError(f"no budgets left after route '{args.mtp_route}' probe filter")
    max_budget = max(budgets)
    stamp = args.timestamp or time.strftime("%Y-%m-%dT%H:%M:%S")
    raw_root = args.raw_root or Path(f"/tmp/hipengine-ar-mtp-suite-{args.scope}-{int(time.time())}")
    raw_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    env.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")
    # Load-once: the MTP category bench loops prompts in-process and reuses one
    # resident session (gguf_mtp_bench's opt-in cache) instead of reloading the
    # ~20GB model per (prompt, budget). The AR baseline already loads once.
    mtp_env = dict(env)
    mtp_env["HIPENGINE_MTP_BENCH_CACHE_SESSION"] = "1"

    # AR decode-tokens: a representative steady-state count (tok/s is a rate, so a
    # single AR baseline serves every budget's ratio). Match the largest budget's
    # cycle output so the regime is comparable.
    ar_decode_tokens = cycles * (max_budget + 1)
    ar_json = raw_root / "true-ar-baseline.json"
    mtp_json = raw_root / "mtp-category.json"

    ar_cmd = [
        args.python, str(REPO_ROOT / "scripts" / "gguf_true_ar_category_bench.py"),
        "--model", str(args.model),
        "--prompts", str(args.prompts),
        "--decode-tokens", str(ar_decode_tokens),
        "--warmup-decode-tokens", str(CANONICAL_DECODE["warmup_decode_tokens"]),
        "--decode-repack", "--use-gemv-decode", "--use-wmma-prefill",
        "--output", str(ar_json),
        "--raw-root", str(raw_root / "true-ar"),
    ]
    if limit is not None:
        ar_cmd += ["--limit", str(limit)]
    if args.require_cached_build:
        ar_cmd += ["--require-cached-build"]

    mtp_cmd = [
        args.python, str(REPO_ROOT / "scripts" / "gguf_mtp_category_bench.py"),
        "--model", str(args.model),
        "--prompts", str(args.prompts),
        "--cycles", str(cycles),
        "--budgets", ",".join(str(b) for b in budgets),
        "--raw-root", str(raw_root / "mtp"),
        "--output", str(mtp_json),
        "--python", args.python,
        # enforce the one prompt-construction knob symmetrically + the route flags
        "--extra-arg=--prompt-reasoning", "--extra-arg=off",
    ]
    # NOTE: we deliberately do NOT pass --true-ar-baseline-json. That attach path
    # in gguf_mtp_category_bench still validates against the RETIRED graph_replay
    # AR contract (TRUE_AR_PRODUCTION_TIMING_REQUIRED, stale since #8) and rejects
    # the current eager AR baseline. This orchestrator owns the apple-to-apple
    # comparison instead: it runs both benches under one enforced config and
    # computes the ratio itself (see _enforce_apple_to_apple + the verdict below).
    for a in route_args:
        mtp_cmd.append(f"--extra-arg={a}")
    if limit is not None:
        mtp_cmd += ["--limit", str(limit)]
    if args.reuse_existing:
        mtp_cmd += ["--reuse-existing"]

    shared_config = {
        **CANONICAL_DECODE,
        "model": str(args.model),
        "prompts": str(args.prompts),
        "cycles": cycles,
        "budgets": budgets,
        "prompt_limit": limit,
        "ar_decode_tokens": ar_decode_tokens,
        "mtp_route": args.mtp_route,
        "mtp_route_extra_args": route_args,
    }

    if args.dry_run:
        print("AR :", " ".join(ar_cmd))
        print("MTP:", " ".join(mtp_cmd))
        print("shared_config:", json.dumps(shared_config, indent=2))
        return 0

    print(f"[ar-mtp-suite] scope={args.scope} route={args.mtp_route} "
          f"cycles={cycles} budgets={budgets} limit={limit}", flush=True)
    print("[ar-mtp-suite] running true-AR baseline...", flush=True)
    _run(ar_cmd, env, raw_root / "true-ar.log")
    print("[ar-mtp-suite] running MTP category suite (in-process load-once)...", flush=True)
    _run(mtp_cmd, mtp_env, raw_root / "mtp.log")

    ar = _load(ar_json)
    mtp = _load(mtp_json)
    problems = _enforce_apple_to_apple(ar, mtp)

    ar_row = _ar_row(ar)
    mtp_rows = _mtp_rows(mtp)
    ar_tok_s = ar_row.get("decode_tok_s_weighted") or 0.0
    best_budget, best_ratio = None, 0.0
    for label, row in mtp_rows.items():
        mtp_tok_s = row.get("decode_tok_s_weighted") or 0.0
        ratio = (mtp_tok_s / ar_tok_s) if ar_tok_s else 0.0
        row["vs_ar_ratio"] = round(ratio, 4) if ratio else None
        if row.get("diagnostic_only"):
            continue  # never let the verifier-derived off/b0 proxy win the verdict
        if ratio and ratio > best_ratio:
            best_budget, best_ratio = label, ratio

    artifact = {
        "schema": "hipengine.gguf_ar_mtp_suite.v1",
        "timestamp": stamp,
        "scope": args.scope,
        "status": "complete" if not problems else "apple_to_apple_violation",
        "apple_to_apple_ok": not problems,
        "apple_to_apple_problems": problems,
        "hardware": _hardware(),
        "git_commit": _git_commit(),
        "host": platform.node(),
        "shared_config": shared_config,
        "ar": ar_row,
        "mtp_by_budget": mtp_rows,
        "verdict": {
            "ar_tok_s": ar_tok_s,
            "best_mtp_budget": best_budget,
            "best_mtp_vs_ar_ratio": round(best_ratio, 4) if best_ratio else None,
            "mtp_beats_ar": bool(best_ratio and best_ratio > 1.0),
        },
        "child_artifacts": {"true_ar": str(ar_json), "mtp_category": str(mtp_json)},
        "command": " ".join([Path(sys.executable).name] + sys.argv),
    }

    out = args.output or (raw_root / f"ar-mtp-suite-{args.scope}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")

    # Clean summary table.
    print("\n==================== AR vs MTP ====================")
    print(f"scope={args.scope}  route={args.mtp_route}  cycles={cycles}  prompts={ar_row.get('prompts')}")
    print(f"apple_to_apple_ok = {not problems}" + (f"  PROBLEMS: {problems}" if problems else ""))
    print(f"AR (no-MTP)      : {ar_tok_s:.2f} tok/s")
    print(f"{'budget':>6} {'tok/s':>8} {'vs_AR':>7} {'acc/out':>8} {'draft_acc':>10}")
    for label in sorted(mtp_rows, key=lambda x: int(''.join(ch for ch in x if ch.isdigit()) or 0)):
        r = mtp_rows[label]
        tag = "  (diag)" if r.get("diagnostic_only") else ""
        print(f"{label:>6} {(r.get('decode_tok_s_weighted') or 0):8.2f} "
              f"{(r.get('vs_ar_ratio') or 0):7.3f} {(r.get('accepted_per_output') or 0):8.3f} "
              f"{(r.get('draft_acceptance') or 0):10.3f}{tag}")
    v = artifact["verdict"]
    print(f"verdict: best={v['best_mtp_budget']} @ {v['best_mtp_vs_ar_ratio']}x AR  "
          f"-> MTP {'WINS' if v['mtp_beats_ar'] else 'does NOT beat'} AR")
    print(f"[ar-mtp-suite] wrote {out}")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
