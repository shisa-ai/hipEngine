#!/usr/bin/env python3
"""V5 boundary comparison: chunked prefill vs tokenwise at matched positions.

Punchlist V5 asks for prompt lengths 511/512/513 (the sliding window is
``sliding_window: 512``) and 1023/1024/1025 (the decode split's 1024-key entry
threshold), comparing chunked prefill against tokenwise execution and recording
block sizes and selected attention routes.

The campaign ``gate`` subcommand cannot do this: it validates that both arms
share one chain geometry and refuses to compare arms whose ``--prefill``
differs, because their row counts differ. That refusal is correct for its own
job and is exactly the gap this script fills.

Alignment rule, from ``capture_chain``'s contract: row ``t`` of an arm taken at
prefill ``P`` is the distribution after consuming ``prompt_ids[P + t]``. Two
arms therefore match at absolute position ``s`` as ``A[s - P_a]`` against
``B[s - P_b]``, over the positions both arms scored, i.e.
``s in [max(P_a, P_b), len(ids) - 2]``.

Both arms are judged with the gate's own ``evaluate``, so the KL arithmetic,
nearest-rank percentiles, thresholds and top-1 accounting are the same code
that gates promotion -- this script contributes alignment only.

Limit: two hipEngine arms agree-or-not; they cannot catch a wrong mask both
arms share. That is V2's job (an independent oracle), and V5's own Why column
says so. A passing run here establishes cross-path consistency, not correctness.

Usage::

    # capture every arm in one process (the model loads once), then compare
    gemma4_boundary_compare.py run --lengths 511 512 513 --prefills 0 256

    # compare arms already captured (no GPU)
    gemma4_boundary_compare.py compare --dir ~/.cache/hipengine/tmp/v5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gemma4_teacher_forced_gate import (  # noqa: E402
    THRESHOLDS,
    _provenance,
    capture_chain,
    evaluate,
    observe_decode_routes,
    route_summary,
    save_capture,
    load_capture,
    sha256_file,
)

DEFAULT_DIR = Path.home() / ".cache" / "hipengine" / "tmp" / "v5"


def arm_path(out_dir: Path, length: int, prefill: int) -> Path:
    return out_dir / f"v5_prompt{length}_prefill{prefill}.npz"


def align(a: np.ndarray, b: np.ndarray, pa: int, pb: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the rows both arms scored, aligned on absolute position."""
    start = max(pa, pb)
    left = a[start - pa:]
    right = b[start - pb:]
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"alignment mismatch: arm@{pa} gives {left.shape[0]} rows, "
            f"arm@{pb} gives {right.shape[0]} after offset {start}"
        )
    return left, right


def compare_dir(out_dir: Path, lengths: Sequence[int], prefills: Sequence[int]) -> dict[str, Any]:
    """Judge every non-zero-prefill arm against the tokenwise arm."""
    report: dict[str, Any] = {"thresholds": dict(THRESHOLDS), "lengths": []}
    for length in lengths:
        base_path = arm_path(out_dir, length, 0)
        if not base_path.exists():
            report["lengths"].append(
                {"length": length, "skipped": f"missing tokenwise capture {base_path.name}"}
            )
            continue
        base, base_ids, base_prov = load_capture(base_path)
        entry: dict[str, Any] = {"length": length, "arms": []}
        for prefill in prefills:
            if prefill == 0:
                continue
            arm_path_ = arm_path(out_dir, length, prefill)
            if not arm_path_.exists():
                entry["arms"].append({"prefill": prefill, "skipped": "not captured"})
                continue
            cand, cand_ids, cand_prov = load_capture(arm_path_)
            if base_ids.shape != cand_ids.shape:
                entry["arms"].append(
                    {"prefill": prefill, "skipped": "chain differs -- lengths must match"}
                )
                continue
            if not np.array_equal(base_ids, cand_ids):
                entry["arms"].append(
                    {"prefill": prefill, "skipped": "prompt_ids differ -- not a paired chain"}
                )
                continue
            b_slice, c_slice = align(base, cand, 0, prefill)
            verdict = evaluate(b_slice, c_slice)
            verdict["matched_position_range"] = [
                max(0, prefill), int(base_ids.shape[0]) - 2
            ]
            verdict["tokenwise_route"] = base_prov.get("observed_routes")
            verdict["chunked_route"] = cand_prov.get("observed_routes")
            verdict["max_block"] = cand_prov.get("max_block")
            entry["arms"].append({"prefill": prefill, "verdict": verdict})
        report["lengths"].append(entry)
    return report


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.gemma4_campaign_bench import (
        DEFAULT_ARTIFACT,
        _resolve_generator,
        exact_prompt_ids,
    )

    artifact = Path(args.artifact) if args.artifact else DEFAULT_ARTIFACT
    timings: list[dict[str, Any]] = []

    # Load the 17 GB weights once and reuse the runner for every arm;
    # capture_chain resets the cache itself before each chain.
    print(f"loading {artifact.name} once for {len(args.lengths)} lengths...", flush=True)
    load_t0 = time.monotonic()
    _llm, runner, loading = _resolve_generator(artifact, args.context)
    generator = _llm._get_text_generator()
    print(f"loaded in {time.monotonic() - load_t0:.1f}s", flush=True)
    base_prov = _provenance(artifact, loading)

    for length in args.lengths:
        prompt_ids = exact_prompt_ids(generator.tokenize, length)
        valid = [p for p in args.prefills if 0 <= p < length - 1]
        if 0 not in valid:
            valid = [0, *valid]
        for prefill in valid:
            path = arm_path(out_dir, length, prefill)
            if path.exists() and not args.recapture:
                print(f"  reuse {path.name}", flush=True)
                continue
            t0 = time.monotonic()
            routes: list[dict[str, int]] = []
            with observe_decode_routes(routes):
                logits = capture_chain(runner, prompt_ids, prefill, routes)
            secs = time.monotonic() - t0
            prov = dict(base_prov)
            prov["observed_routes"] = route_summary(routes)
            prov["prefill"] = prefill
            prov["prompt_tokens"] = length
            prov["sliding_window"] = args.sliding_window
            save_capture(path, logits, np.asarray(prompt_ids, dtype=np.int32), prov)
            timings.append(
                {"length": length, "prefill": prefill, "seconds": round(secs, 2),
                 "rows": int(logits.shape[0]), "npz_sha256": sha256_file(path)[:16]}
            )
            rs = prov["observed_routes"]
            print(
                f"  {path.name}: {logits.shape[0]} rows in {secs:.1f}s | "
                f"decode={rs['decode_launches']} split={rs['split_launches']} "
                f"selections={rs['selections']} split_keys={rs['split_key_range']} "
                f"head_dims={rs['head_dims']}",
                flush=True,
            )

    report = compare_dir(out_dir, args.lengths, args.prefills)
    report["captured"] = timings
    report["sliding_window"] = args.sliding_window
    out_json = Path(args.out) if args.out else out_dir / "v5_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print_summary(report)
    print(f"\nreport: {out_json}")
    return 0


def print_summary(report: dict[str, Any]) -> None:
    print("\n=== V5 chunked-vs-tokenwise at matched positions ===")
    for entry in report["lengths"]:
        if "skipped" in entry:
            print(f"  L={entry['length']}: SKIPPED {entry['skipped']}")
            continue
        print(f"  L={entry['length']}:")
        for arm in entry["arms"]:
            if "skipped" in arm:
                print(f"    prefill={arm['prefill']:>5}: {arm['skipped']}")
                continue
            v = arm["verdict"]
            print(
                f"    prefill={arm['prefill']:>5}: rows={v['rows']:>5} "
                f"kl_mean={v['kl_mean']:.3e} kl_p95={v['kl_p95']:.3e} "
                f"kl_max={v['kl_max']:.3e} top1={v['top1_rate']:.6f} "
                f"flips={v['top1_flips']} passed={v['passed']}"
            )
            print(
                f"                   pos={v['matched_position_range']} "
                f"max_block={v['max_block']}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="capture every arm in one process, then compare")
    run_p.add_argument("--lengths", type=int, nargs="+",
                       default=[511, 512, 513, 1023, 1024, 1025, 1537])
    run_p.add_argument("--prefills", type=int, nargs="+", default=[0, 256, 512, 1024])
    run_p.add_argument("--context", type=int, default=8192)
    run_p.add_argument("--artifact", type=str, default=None)
    run_p.add_argument("--dir", type=str, default=str(DEFAULT_DIR))
    run_p.add_argument("--out", type=str, default=None)
    run_p.add_argument("--sliding-window", type=int, default=512)
    run_p.add_argument("--recapture", action="store_true")

    cmp_p = sub.add_parser("compare", help="compare already-captured arms (no GPU)")
    cmp_p.add_argument("--lengths", type=int, nargs="+",
                       default=[511, 512, 513, 1023, 1024, 1025, 1537])
    cmp_p.add_argument("--prefills", type=int, nargs="+", default=[0, 256, 512, 1024])
    cmp_p.add_argument("--dir", type=str, default=str(DEFAULT_DIR))
    cmp_p.add_argument("--out", type=str, default=None)

    args = p.parse_args(argv)
    if args.command == "run":
        return run(args)
    report = compare_dir(Path(args.dir), args.lengths, args.prefills)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())