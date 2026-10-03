#!/usr/bin/env python3
"""Gemma 4 teacher-forced evaluator: freeze the incumbent chain, gate candidates.

Two subcommands implement the evaluator-freeze sub-item deferred out of the
G0 baseline (see ``docs/campaigns/GEMMA4-26B-A4B-OPTIMIZATION.md``):

``capture``
    Teacher-force the frozen campaign prompt chain on the current default
    path and store every position's full-vocabulary next-token distribution
    as float32. Use ``--slices 1`` to capture the strict single-kernel arm;
    without that override, capture uses the ordinary shipping policy. Keep the
    ``.npz`` outside the repository and record its sha256 in a manifest.

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
zero top-1 flips, while promotion evidence needs 500-1000 paired rows plus
category, isolation and task gates not supplied by this single-chain evaluator.
Gate requires observed split-family launches (the incumbent split or
flash-decoding) unless ``--slices 1`` explicitly requests
single-kernel evaluation. The default 1024-token chain never reaches the split:
use, for example, ``--prompt 2048 --prefill 1024`` to score 1023 decode rows.

Example (freeze, on the incumbent tree)::

    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1 PYTHONPATH=. \\
      .venv/bin/python scripts/gemma4_teacher_forced_gate.py capture \\
      --out /mnt/nvme1/gemma4-eval/baseline-<commit>.npz \\
      --manifest benchmarks/results/<freeze>.json
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
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

# Margin bands for the per-row breakdown, ascending and exclusive of the upper
# edge. "Close" is the band a decision can actually move in; the top-1 bar is
# nearly blind there because a near-tie that stays a tie is not a flip.
MARGIN_BANDS: tuple[tuple[str, float], ...] = (
    ("margin_lt_0.01", 0.01),
    ("margin_0.01_to_0.05", 0.05),
    ("margin_0.05_to_0.20", 0.20),
    ("margin_ge_0.20", float("inf")),
)

# The upper edge of the bands a changed decision can live in.
CLOSE_MARGIN = 0.05

# Rows per block when forming the full-vocabulary denominator.
_MARGIN_CHUNK = 64

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


def row_top2_margin(baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row top-1 probability margin and top-1 logit gap.

    The margin is ``p1 - p2`` and therefore needs the full-vocabulary
    denominator; it is formed in row blocks with a float64 accumulator so a
    million-row capture does not allocate a second float64 copy of itself. The
    logit gap is exact and free, and is reported alongside because it is what a
    numerical divergence directly perturbs.
    """

    x = np.asarray(baseline, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError(f"baseline must be (rows, vocab>=2); got {x.shape}")
    rows = x.shape[0]
    margin = np.empty(rows, dtype=np.float64)
    gap = np.empty(rows, dtype=np.float64)
    for start in range(0, rows, _MARGIN_CHUNK):
        block = x[start:start + _MARGIN_CHUNK]
        top2 = np.sort(np.partition(block, -2, axis=1)[:, -2:], axis=1)
        second = top2[:, 0].astype(np.float64)
        first = top2[:, 1].astype(np.float64)
        stop = start + block.shape[0]
        gap[start:stop] = first - second
        total = np.exp(block - top2[:, 1:2], dtype=np.float64).sum(axis=1)
        margin[start:stop] = (1.0 - np.exp(second - first)) / total
    return margin, gap


def margin_report(
    baseline: np.ndarray,
    candidate: np.ndarray,
    kl: np.ndarray,
    flips: Sequence[int],
) -> dict[str, Any]:
    """Break a verdict down by how close the frozen row's decision was.

    A single ``kl_max`` cannot distinguish a probability-tail difference at a
    near-one-hot row from a changed decision at a near-tie, and the frozen chain
    is almost entirely the former. This reports both, so a breach can be read
    against where it actually happened.
    """

    margin, gap = row_top2_margin(baseline)
    flipped = np.zeros(margin.shape[0], dtype=bool)
    for index in flips:
        flipped[int(index)] = True
    close = margin < CLOSE_MARGIN

    bands: list[dict[str, Any]] = []
    lower = 0.0
    for name, upper in MARGIN_BANDS:
        in_band = (margin >= lower) & (margin < upper)
        count = int(in_band.sum())
        bands.append({
            "band": name,
            "rows": count,
            "share": count / margin.shape[0],
            "flips": int(np.count_nonzero(in_band & flipped)),
            "kl_mean": float(kl[in_band].mean()) if count else None,
            "kl_max": float(kl[in_band].max()) if count else None,
        })
        lower = upper

    return {
        "band_definition": "margin is p1 - p2 within the frozen baseline row",
        "bands": bands,
        "margin_median": float(np.median(margin)),
        "margin_min": float(margin.min()),
        "logit_gap_median": float(np.median(gap)),
        "close_margin_definition": f"margin < {CLOSE_MARGIN}",
        "close_margin_rows": int(np.count_nonzero(close)),
        "close_margin_flips": int(np.count_nonzero(close & flipped)),
        "close_margin_kl_max": float(kl[close].max()) if close.any() else None,
    }


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
    if rows == 0 or baseline.shape[1] == 0:
        raise ValueError("logits must have nonempty rows and vocabulary")
    kl = np.empty(rows, dtype=np.float64)
    flips: list[int] = []
    base_top1 = np.argmax(baseline, axis=1)
    cand_top1 = np.argmax(candidate, axis=1)
    for i in range(rows):
        kl[i] = row_kl_divergence(baseline[i], candidate[i])
        if base_top1[i] != cand_top1[i]:
            flips.append(i)

    top1_rate = (rows - len(flips)) / rows
    # ``kl_max`` alone cannot be acted on: with a passing p99 a single row can
    # dominate it, and the summary says nothing about where that row sits in
    # the chain. Report the worst offenders and how many clear the limit so a
    # failure points at a key range instead of a single number.
    worst = np.argsort(kl)[::-1][:8]
    kl_limit = THRESHOLDS["kl_max"]
    verdict: dict[str, Any] = {
        "rows": rows,
        "vocab": int(baseline.shape[1]),
        "kl_mean": float(kl.mean()),
        "kl_p95": _nearest_rank(kl, 0.95),
        "kl_p99": _nearest_rank(kl, 0.99),
        "kl_max": float(kl.max()),
        "kl_rows_over_limit": int((kl > kl_limit).sum()),
        "kl_worst_rows": [[int(i), float(kl[i])] for i in worst],
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
    verdict["margin_report"] = margin_report(baseline, candidate, kl, flips)
    verdict["top1_flips_close_margin"] = verdict["margin_report"]["close_margin_flips"]
    verdict["evidence_level"] = "screen" if rows < 500 else "numerical_rows"
    verdict["promotion_qualified"] = False  # Category/task/isolation gates are separate.
    if rows < 500 and flips:
        failed.append("screen_top1_flips")
    verdict["failed"] = failed
    verdict["passed"] = not failed
    return verdict


@contextmanager
def observe_decode_routes(routes: list[dict[str, int]]):
    """Record actual launcher selections, not merely the requested slice policy."""
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention

    original = attention._launch_prefill

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["tokens"] == 1:
            routes.append({
                "keys": int(kwargs["keys"] or 1),
                "head_dim": int(kwargs["head_dim"]),
                "selection": int(attention.decode_selection(kwargs.get("library"))),
            })
        return result

    attention._launch_prefill = observed
    try:
        yield
    finally:
        attention._launch_prefill = original


def route_summary(routes: list[dict[str, int]]) -> dict[str, Any]:
    # The split family is the multi-slice partials-plus-combine route: the
    # incumbent split (selection 2) and flash-decoding (selection 3, the
    # negative-slices split request) both exercise it.
    split = [row for row in routes if row["selection"] == 2]
    flash = [row for row in routes if row["selection"] == 3]
    family = split + flash
    return {
        "decode_launches": len(routes),
        "split_launches": len(split),
        "flash_launches": len(flash),
        "split_family_launches": len(family),
        "selections": sorted({row["selection"] for row in routes}),
        "split_key_range": [min(row["keys"] for row in family),
                            max(row["keys"] for row in family)] if family else None,
        "head_dims": sorted({row["head_dim"] for row in routes}),
    }


def require_candidate_route(verdict: dict[str, Any], routes: dict[str, Any],
                            forced_slices: int | None) -> None:
    if not routes["decode_launches"]:
        verdict["failed"].append("no_decode_launches_observed")
    elif forced_slices != 1 and not routes["split_family_launches"]:
        verdict["failed"].append("split_not_exercised")
    verdict["passed"] = not verdict["failed"]


def capture_chain(
    runner: Any, prompt_ids: Sequence[int], prefill: int = 0,
    routes: list[dict[str, int]] | None = None,
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
    with observe_decode_routes(routes) if routes is not None else nullcontext():
        for position in range(prefill, len(ids) - 1):
            logits = runner.forward([ids[position]])
            rows.append(np.array(logits, dtype=np.float32, copy=True).reshape(-1))
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


def _load_chain(
    artifact: Path, prompt_tokens: int, context: int,
    *, corpus: str = "frozen", corpus_seed: int | None = None,
):
    """Resolve the chain a capture or gate arm teacher-forces.

    ``frozen`` is the campaign corpus, cycled: eight sentences repeated to the
    target length, so its rows are near one-hot. ``probe`` is generated from a
    fixed seed and never repeats, so one chain carries rows across the whole
    margin range. Both are deterministic, which is what makes two arms paired.
    """

    from scripts.gemma4_campaign_bench import (
        PROBE_CORPUS_SEED,
        exact_prompt_ids,
        probe_corpus,
        _resolve_generator,
    )

    llm, runner, loading = _resolve_generator(artifact, context)
    generator = llm._get_text_generator()
    if corpus == "frozen":
        prompt_ids = exact_prompt_ids(generator.tokenize, prompt_tokens)
        seed = None
    elif corpus == "probe":
        seed = PROBE_CORPUS_SEED if corpus_seed is None else int(corpus_seed)
        prompt_ids = exact_prompt_ids(
            generator.tokenize, prompt_tokens,
            corpus=probe_corpus(seed=seed),
            require_single_pass=True,
        )
    else:
        raise ValueError(f"unknown corpus {corpus!r}")
    return runner, prompt_ids, loading, {"corpus": corpus, "corpus_seed": seed}


def _provenance(artifact: Path, loading: dict[str, Any]) -> dict[str, Any]:
    from scripts.gemma4_campaign_bench import _provenance as harness_provenance

    provenance = harness_provenance(artifact)
    provenance["context_length"] = loading.get("context_length")
    provenance["max_block"] = loading.get("max_block")
    root = Path(__file__).resolve().parents[1]
    provenance["evaluator_source_sha256"] = sha256_file(Path(__file__))
    provenance["attention_source_sha256"] = {
        suffix: sha256_file(root / "hipengine/kernels/hip_gfx1100/gemma4" /
                            f"gemma4_attention.{suffix}")
        for suffix in ("hip", "py")
    }
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
        p.add_argument(
            "--corpus",
            choices=("frozen", "probe"),
            default="frozen",
            help="frozen = the campaign's cycled prose chain; probe = a "
            "seeded never-repeating chain that carries low-margin rows, which "
            "the cycled chain cannot",
        )
        p.add_argument(
            "--corpus-seed",
            type=int,
            default=None,
            help="override the probe corpus seed (both arms must match)",
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
    if args.slices is not None and args.slices < 1:
        parser.error("--slices must be positive")
    if not 0 <= args.prefill < args.prompt - 1 or args.prompt > args.context:
        parser.error("require 0 <= prefill < prompt - 1 and prompt <= context")

    if args.command == "capture":
        started = time.time()
        force_slices(args.slices)
        runner, prompt_ids, loading, chain_kind = _load_chain(
            args.artifact, args.prompt, args.context,
            corpus=args.corpus, corpus_seed=args.corpus_seed,
        )
        routes = []
        try:
            logits = capture_chain(runner, prompt_ids, args.prefill, routes=routes)
        finally:
            runner.close()
            force_slices(None)
        provenance = _provenance(args.artifact, loading)
        provenance["observed_routes"] = route_summary(routes)
        provenance["prefill"] = int(args.prefill)
        provenance["forced_slices"] = None if args.slices is None else int(args.slices)
        provenance["chain_kind"] = chain_kind
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
            "chain_kind": chain_kind,
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
    runner, prompt_ids, loading, chain_kind = _load_chain(
        args.artifact, args.prompt, args.context,
        corpus=args.corpus, corpus_seed=args.corpus_seed,
    )
    if [int(x) for x in base_ids] != [int(x) for x in prompt_ids]:
        runner.close()
        force_slices(None)
        raise SystemExit(
            "chain mismatch: the candidate chain differs from the frozen "
            "baseline's prompt ids; both arms must teacher-force the same ids"
        )
    base_kind = base_provenance.get(
        "chain_kind", {"corpus": "frozen", "corpus_seed": None}
    )
    if base_kind != chain_kind:
        runner.close()
        force_slices(None)
        raise SystemExit(
            f"chain kind mismatch: baseline {base_kind} vs candidate {chain_kind}; "
            "a probe baseline cannot be gated against a frozen chain"
        )
    if int(base_provenance.get("prefill", 0)) != int(args.prefill):
        runner.close()
        force_slices(None)
        raise SystemExit(
            "prefill mismatch: the baseline scored a different key range "
            f"(baseline {base_provenance.get('prefill', 0)}, candidate "
            f"{args.prefill}); the comparison would not be paired"
        )
    routes = []
    try:
        candidate = capture_chain(runner, prompt_ids, args.prefill, routes=routes)
    finally:
        runner.close()
        force_slices(None)
    verdict = evaluate(baseline, candidate)
    verdict["candidate_provenance"] = _provenance(args.artifact, loading)
    verdict["baseline_provenance"] = base_provenance
    verdict["observed_routes"] = route_summary(routes)
    require_candidate_route(verdict, verdict["observed_routes"], args.slices)
    verdict["prefill"] = int(args.prefill)
    verdict["chain_kind"] = chain_kind
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