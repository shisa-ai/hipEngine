#!/usr/bin/env python3
"""Gemma 4 teacher-forced evaluator: freeze the incumbent chain, gate candidates.

Two subcommands implement the evaluator-freeze sub-item deferred out of the
G0 baseline (see ``docs/campaigns/GEMMA4-26B-A4B-OPTIMIZATION.md``):

``capture``
    Teacher-force the frozen campaign prompt chain on the current default
    path and store every position's full-vocabulary next-token distribution
    as float32. This is the *strict* arm: run it on the incumbent tree
    before any changed-arithmetic candidate lands, keep the ``.npz`` outside
    the repository at a pinned path, and record its sha256 in a manifest.

``gate``
    Recompute the same chain with the candidate path and compare against a
    frozen baseline: per-row KL (baseline || candidate) in float64 and top-1
    agreement, judged against the binding production limits from
    ``docs/EXECUTION-PROFILES.md`` (mean KL <= 1e-3, p95 <= 5e-3,
    p99 <= 2e-2, max <= 5e-2, top-1 >= 99% overall).

The chain is shared: the frozen prompt ids are forced into every arm, so rows
are paired positions rather than each arm's own sampled trajectory. A capture
run twice on the same tree must gate at KL == 0 and 100% top-1; that self-gate
is the smoke for the capture path.

The row-count standard applies: a ~60-row screening probe only passes with
zero top-1 flips, while promotion evidence needs 500-1000 paired rows. The
default chain (1023 rows from the 1024-token prompt) is inside that band.

Example (freeze, on the incumbent tree)::

    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1 PYTHONPATH=. \\
      .venv/bin/python scripts/gemma4_teacher_forced_gate.py capture \\
      --out /mnt/nvme1/gemma4-eval/baseline-<commit>.npz \\
      --manifest benchmarks/results/<freeze>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Binding production limits, docs/EXECUTION-PROFILES.md "Metric over
# full-vocabulary teacher-forced rows". These are admission limits, not
# tuning targets: every limit binds together.
THRESHOLDS: dict[str, float] = {
    "kl_mean": 1e-3,
    "kl_p95": 5e-3,
    "kl_p99": 2e-2,
    "kl_max": 5e-2,
    "top1_rate": 0.99,
}

# The shipped decode_slices, captured on first use by force_slices so an
# override can be undone inside one process.
_ORIGINAL_DECODE_SLICES: Any = None


def row_kl_divergence(baseline_row: np.ndarray, candidate_row: np.ndarray) -> float:
    """KL(baseline || candidate) for one full-vocabulary row, in float64."""

    b = np.asarray(baseline_row, dtype=np.float64)
    c = np.asarray(candidate_row, dtype=np.float64)
    log_p = b - b.max()
    log_p -= np.log(np.exp(log_p).sum())
    log_q = c - c.max()
    log_q -= np.log(np.exp(log_q).sum())
    p = np.exp(log_p)
    return float(np.sum(p * (log_p - log_q)))


def _nearest_rank(values: np.ndarray, q: float) -> float:
    """Nearest-rank percentile, matching the campaign harness's convention."""

    ordered = np.sort(values)
    rank = max(1, int(np.ceil(q * ordered.size)))
    return float(ordered[min(rank, ordered.size) - 1])


def evaluate(baseline_logits: np.ndarray, candidate_logits: np.ndarray) -> dict[str, Any]:
    """Compare a candidate chain against the frozen baseline and judge it."""

    baseline = np.asarray(baseline_logits)
    candidate = np.asarray(candidate_logits)
    if baseline.ndim != 2 or candidate.ndim != 2:
        raise ValueError(
            f"logits must be (rows, vocab); got {baseline.shape} vs {candidate.shape}"
        )
    if baseline.shape[0] != candidate.shape[0]:
        raise ValueError(
            f"row count mismatch: baseline {baseline.shape[0]} vs "
            f"candidate {candidate.shape[0]}"
        )
    if baseline.shape[1] != candidate.shape[1]:
        raise ValueError(
            f"vocab mismatch: baseline {baseline.shape[1]} vs "
            f"candidate {candidate.shape[1]}"
        )
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("logits must be finite; a non-finite value was found")

    rows = baseline.shape[0]
    kl = np.empty(rows, dtype=np.float64)
    flips: list[int] = []
    base_top1 = np.argmax(baseline, axis=1)
    cand_top1 = np.argmax(candidate, axis=1)
    for i in range(rows):
        kl[i] = row_kl_divergence(baseline[i], candidate[i])
        if base_top1[i] != cand_top1[i]:
            flips.append(i)

    top1_rate = (rows - len(flips)) / rows
    verdict: dict[str, Any] = {
        "rows": rows,
        "vocab": int(baseline.shape[1]),
        "kl_mean": float(kl.mean()),
        "kl_p95": _nearest_rank(kl, 0.95),
        "kl_p99": _nearest_rank(kl, 0.99),
        "kl_max": float(kl.max()),
        "top1_rate": top1_rate,
        "top1_flips": len(flips),
        "top1_flip_rows": flips[:32],
        "thresholds": dict(THRESHOLDS),
        "direction": "kl(baseline || candidate)",
        "percentile_method": "nearest-rank",
    }
    failed = [
        key
        for key, limit in THRESHOLDS.items()
        if (verdict[key] > limit if key != "top1_rate" else verdict[key] < limit)
    ]
    verdict["failed"] = failed
    verdict["passed"] = not failed
    return verdict


def capture_chain(
    runner: Any, prompt_ids: Sequence[int], prefill: int = 0
) -> np.ndarray:
    """Teacher-force ``prompt_ids`` and return the scored (rows, vocab) chain.

    Row ``t`` is the distribution after consuming ``prompt_ids[prefill + t]``
    and predicting the next id; the runner contract is last-row logits, so one
    forward per forced token produces one paired row.

    ``prefill`` ids are pushed through the cache in a single forward before
    scoring starts and are not themselves scored. That is what makes the chain
    exercise the decode path the model actually serves. Without it every row
    runs at key counts ``1..len(ids)-1``, which at a 1024-token prompt stops one
    key below the decode split's 1024-key entry threshold: the split never runs,
    and the gate reports ``kl_max`` of exactly 0.0 because it is comparing the
    single-kernel path with itself. Measured directly - 30 ``decode_slices``
    calls per forward, one per layer, keys ``[1]`` at position 0 through ``[5]``
    at position 4, all returning 1 slice - and the 2026-09-25 split gate result
    was this artifact. With a prefill, scoring starts at key count
    ``prefill + 1`` and every row above the threshold engages the split.
    """

    ids = [int(t) for t in prompt_ids]
    if len(ids) < 2:
        raise ValueError(f"prompt must carry at least 2 ids, got {len(ids)}")
    if not 0 <= prefill < len(ids) - 1:
        raise ValueError(
            f"prefill must leave at least one scored row, got {prefill} of "
            f"{len(ids)} ids"
        )
    runner.reset()
    if prefill:
        runner.forward(ids[:prefill])
    rows: list[np.ndarray] = []
    for position in range(prefill, len(ids) - 1):
        logits = runner.forward([ids[position]])
        rows.append(np.asarray(logits, dtype=np.float32).reshape(-1))
    return np.stack(rows)


def force_slices(slices: int | None) -> None:
    """Pin the decode split's slice count for one capture or gate arm.

    A harness override, not a product control: nothing in the engine reads it.
    It exists so a baseline can be frozen at the slice count it actually shipped
    with, and so the strict arm - 1, which selects the single-kernel path - can
    be captured on the same tree as the split arms. An arm captured this way
    records the value in its manifest, and ``gate`` refuses to compare two arms
    whose chain geometry differs.

    ``None`` restores the shipped policy, so the override is reversible within
    one process rather than only per-invocation.
    """

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention

    global _ORIGINAL_DECODE_SLICES
    if _ORIGINAL_DECODE_SLICES is None:
        _ORIGINAL_DECODE_SLICES = gemma4_attention.decode_slices
    if slices is None:
        gemma4_attention.decode_slices = _ORIGINAL_DECODE_SLICES
        return
    gemma4_attention.decode_slices = (
        lambda keys, head_dim: 1 if slices <= 1 else int(slices)
    )


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def chain_sha256(prompt_ids: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(prompt_ids, dtype=np.int32).tobytes()).hexdigest()


def save_capture(
    path: Path,
    logits: np.ndarray,
    prompt_ids: Sequence[int],
    provenance: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        logits=np.asarray(logits, dtype=np.float32),
        prompt_ids=np.asarray(prompt_ids, dtype=np.int32),
        provenance=np.frombuffer(
            json.dumps(provenance, sort_keys=True).encode("utf-8"), dtype=np.uint8
        ),
    )


def load_capture(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path) as data:
        missing = {"logits", "prompt_ids"} - set(data.files)
        if missing:
            raise ValueError(f"capture is missing arrays: {sorted(missing)}")
        logits = np.array(data["logits"], dtype=np.float32, copy=True)
        prompt_ids = np.array(data["prompt_ids"], dtype=np.int32, copy=True)
        provenance: dict[str, Any] = {}
        if "provenance" in data.files:
            provenance = json.loads(bytes(np.array(data["provenance"]).tobytes()))
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D, got shape {logits.shape}")
    prefill = int(provenance.get("prefill", 0))
    if logits.shape[0] != len(prompt_ids) - 1 - prefill:
        raise ValueError(
            f"capture rows {logits.shape[0]} does not match chain length "
            f"{len(prompt_ids)} - 1 - {prefill} prefill"
        )
    if not np.isfinite(logits).all():
        raise ValueError("capture logits are not finite")
    return logits, prompt_ids, provenance


def _load_chain(artifact: Path, prompt_tokens: int, context: int):
    from scripts.gemma4_campaign_bench import exact_prompt_ids, _resolve_generator

    llm, runner, loading = _resolve_generator(artifact, context)
    generator = llm._get_text_generator()
    prompt_ids = exact_prompt_ids(generator.tokenize, prompt_tokens)
    return runner, prompt_ids, loading


def _provenance(artifact: Path, loading: dict[str, Any]) -> dict[str, Any]:
    from scripts.gemma4_campaign_bench import _provenance as harness_provenance

    provenance = harness_provenance(artifact)
    provenance["context_length"] = loading.get("context_length")
    provenance["max_block"] = loading.get("max_block")
    return provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        from scripts.gemma4_campaign_bench import DEFAULT_ARTIFACT, DEFAULT_CONTEXT

        p.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
        p.add_argument("--prompt", type=int, default=1024)
        p.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
        p.add_argument(
            "--prefill",
            type=int,
            default=0,
            help="ids pushed through the cache before scoring; the chain must "
            "clear the decode split's 1024-key entry threshold to exercise it",
        )
        p.add_argument(
            "--slices",
            type=int,
            default=None,
            help="pin decode_slices for this arm (1 = strict single-kernel path)",
        )

    capture = sub.add_parser("capture", help="freeze the incumbent chain")
    common(capture)
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--manifest", type=Path, help="JSON freeze record to write")

    gate = sub.add_parser("gate", help="judge the current path against a baseline")
    common(gate)
    gate.add_argument("--baseline", type=Path, required=True)
    gate.add_argument("--out", type=Path, help="verdict JSON to write")

    args = parser.parse_args(argv)

    if args.command == "capture":
        started = time.time()
        force_slices(args.slices)
        runner, prompt_ids, loading = _load_chain(args.artifact, args.prompt, args.context)
        logits = capture_chain(runner, prompt_ids, args.prefill)
        provenance = _provenance(args.artifact, loading)
        provenance["prefill"] = int(args.prefill)
        provenance["forced_slices"] = None if args.slices is None else int(args.slices)
        save_capture(args.out, logits, prompt_ids, provenance)
        elapsed = time.time() - started
        digest = sha256_file(args.out)
        record = {
            "kind": "gemma4_teacher_forced_freeze",
            "performance_claim": False,
            "npz_path": str(args.out),
            "npz_sha256": digest,
            "npz_bytes": args.out.stat().st_size,
            "rows": int(logits.shape[0]),
            "vocab": int(logits.shape[1]),
            "prompt_tokens": len(prompt_ids),
            "prefill": int(args.prefill),
            "forced_slices": None if args.slices is None else int(args.slices),
            "scored_key_range": [int(args.prefill) + 1, len(prompt_ids) - 1],
            "chain_sha256": chain_sha256(prompt_ids),
            "thresholds": dict(THRESHOLDS),
            "threshold_source": "docs/EXECUTION-PROFILES.md production table "
            "(mean/p95/p99/max KL, top-1)",
            "direction": "kl(baseline || candidate)",
            "capture_seconds": round(elapsed, 1),
            "provenance": provenance,
        }
        print(json.dumps(record, indent=1, sort_keys=True))
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        return 0

    baseline, base_ids, base_provenance = load_capture(args.baseline)
    force_slices(args.slices)
    runner, prompt_ids, _ = _load_chain(args.artifact, args.prompt, args.context)
    if [int(x) for x in base_ids] != [int(x) for x in prompt_ids]:
        raise SystemExit(
            "chain mismatch: the candidate chain differs from the frozen "
            "baseline's prompt ids; both arms must teacher-force the same ids"
        )
    if int(base_provenance.get("prefill", 0)) != int(args.prefill):
        raise SystemExit(
            "prefill mismatch: the baseline scored a different key range "
            f"(baseline {base_provenance.get('prefill', 0)}, candidate "
            f"{args.prefill}); the comparison would not be paired"
        )
    candidate = capture_chain(runner, prompt_ids, args.prefill)
    verdict = evaluate(baseline, candidate)
    verdict["prefill"] = int(args.prefill)
    verdict["scored_key_range"] = [int(args.prefill) + 1, len(prompt_ids) - 1]
    verdict["baseline_forced_slices"] = base_provenance.get("forced_slices")
    verdict["candidate_forced_slices"] = None if args.slices is None else int(args.slices)
    verdict["baseline_path"] = str(args.baseline)
    verdict["baseline_sha256"] = sha256_file(args.baseline)
    verdict["chain_sha256"] = chain_sha256(prompt_ids)
    text = json.dumps(verdict, indent=1, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())