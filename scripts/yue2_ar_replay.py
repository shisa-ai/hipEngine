#!/usr/bin/env python3
"""YuE2 AR replay gate: run the frozen oracle matrix through the HIP runtime.

Loads the committed ``tests/fixtures/yue2/ar_replay`` matrix (real prefixes and
forced tokens, per-branch reference logits) and replays it through
``Yue2ArRuntime`` on the local GPU, then reports the production numerical
envelope against the reference: full-vocabulary KL on the rows where the fixture
kept full logits, top-1 agreement and top-8 overlap everywhere, and a
deterministic repeat.

    python3 scripts/yue2_ar_replay.py --limit-cases 4
    python3 scripts/yue2_ar_replay.py --json benchmarks/results/<artifact>.json

The reference logits are BF16, so every comparison upcasts both sides to FP32
exactly and never re-rounds the reference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import platform
import shlex
import subprocess
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/yue2/ar_replay"
VOCAB = 184704


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def softmax_f32(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(values.max())
    weights = np.exp(shifted)
    return (weights / weights.sum()).astype(np.float32)


def kl_divergence(reference: np.ndarray, candidate: np.ndarray) -> float:
    """KL(reference || candidate) over one full-vocabulary row.

    The sum is non-negative in exact arithmetic; fp32 accumulation can return a
    value a few 1e-11 below zero for identical rows, which is clamped so a gate
    report never shows a negative divergence.
    """
    p = softmax_f32(reference)
    q = softmax_f32(candidate)
    mask = p > 0
    value = float(np.sum(p[mask].astype(np.float64) * (np.log(p[mask]) - np.log(q[mask]))))
    return max(value, 0.0)


def summarize(rows: list[float]) -> dict:
    if not rows:
        return {}
    values = np.asarray(rows, dtype=np.float64)
    return {
        "rows": int(values.size),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def replay_case(runtime, fixture: dict, *, max_steps: int | None = None) -> dict:
    """Replay one case; returns per-row reference/candidate logit pairs.

    Candidate rows are FP32 logit values, never BF16 bit patterns: the reference
    side is upcast from stored BF16 bits, so a bit-pattern candidate would be
    compared as integers and every metric would be meaningless.
    """
    import numpy as np

    from hipengine.runtime.yue2_ar import bf16_bits_to_f32

    prefix_positive = fixture["prefix_positive"].astype(np.int64)
    prefix_negative = fixture["prefix_negative"].astype(np.int64) if "prefix_negative" in fixture else None
    tokens = fixture["tokens"].astype(np.int64)
    steps = len(tokens) if max_steps is None else min(max_steps, len(tokens))
    runtime.reset()
    runtime.prefill_host_rows([runtime.embed_row(int(t)) for t in prefix_positive], branch=0)
    if prefix_negative is not None:
        runtime.prefill_host_rows([runtime.embed_row(int(t)) for t in prefix_negative], branch=1)
    rows = {
        "prefill": (bf16_bits_to_f32(fixture["prefill_logits"][0]), runtime.logits(0, as_bf16=False)),
        "steps": [],
    }
    if prefix_negative is not None:
        rows["prefill_negative"] = (
            bf16_bits_to_f32(fixture["prefill_logits_negative"][0]),
            runtime.logits(1, as_bf16=False),
        )
    for step in range(steps):
        # Each branch keeps its own position counter: the negative prefix may be
        # shorter (the reference cache advances per branch), so branch 1 decodes
        # at its own context length, not the positive branch's.
        token = int(tokens[step])
        position_positive = len(prefix_positive) + step
        runtime.push_token(runtime.embed_row(token), position_positive, branch=0)
        runtime.forward_layers(position_positive, branch=0)
        candidate = [runtime.logits(0, as_bf16=False)]
        if prefix_negative is not None:
            position_negative = len(prefix_negative) + step
            runtime.push_token(runtime.embed_row(token), position_negative, branch=1)
            runtime.forward_layers(position_negative, branch=1)
            candidate.append(runtime.logits(1, as_bf16=False))
        rows["steps"].append(candidate)
    return rows


def full_logit_rows(fixture: dict) -> list[tuple[int, int]]:
    """Pairs with a full-vocabulary row actually present in this fixture file.

    ``full_logits_steps`` lists the rows the full-precision artifact recorded;
    the committed compact fixture keeps only some of them. Probing the file keeps
    the gate honest about how many KL rows it really scored.
    """
    recorded = fixture["full_logits_steps"].reshape(-1, 2)
    return [
        (int(index), int(branch))
        for index, branch in recorded
        if f"full_logits_{index}_{branch}" in fixture
    ]


def evaluate(case_name: str, manifest_entry: dict, fixture: dict, rows: dict) -> dict:
    """Compare replay rows against the fixture's recorded reference summaries."""
    reference_argmax = fixture["step_argmax"].astype(np.int64)
    reference_top_ids = fixture["step_top_ids"].astype(np.int64)
    reference_top_vals = fixture["step_top_vals"].astype(np.float32)
    full_steps = full_logit_rows(fixture)
    full_rows = {
        (index, branch): bf16_bits_to_f32(fixture[f"full_logits_{index}_{branch}"][0])
        for index, branch in full_steps
    }

    kl_rows: list[float] = []
    kl_scope: dict[str, list[float]] = {}
    top1_hits = 0
    top1_total = 0
    top1_scope: dict[str, list[int]] = {}
    top8_hits = 0
    top8_total = 0
    top8_mass: list[float] = []
    argmax_flips: list[dict] = []
    worst: list[tuple[float, str]] = []

    def record(scope: str, kl: float | None) -> None:
        if kl is None:
            return
        kl_rows.append(kl)
        kl_scope.setdefault(scope, []).append(kl)
        worst.append((kl, scope))

    for label in ("prefill", "prefill_negative"):
        if label not in rows:
            continue
        reference, candidate = rows[label]
        record(f"{label}:L{manifest_entry['prefix_lengths'][0]}", kl_divergence(reference, candidate))

    for step, candidate_rows in enumerate(rows["steps"]):
        for branch, candidate in enumerate(candidate_rows):
            reference = full_rows.get((step, branch))
            scope = f"{manifest_entry['phase']}:{'cfg' if manifest_entry['cfg'] else 'nocfg'}:L{manifest_entry['prefix_lengths'][0]}"
            if reference is not None:
                record(scope, kl_divergence(reference, candidate))
            reference_ids = reference_top_ids[step, branch]
            reference_values = reference_top_vals[step, branch]
            # Reference top-8 membership must survive in the candidate's own top-8.
            candidate_top = np.argpartition(candidate, -8)[-8:]
            hits = len(set(int(v) for v in reference_ids) & set(int(v) for v in candidate_top))
            top8_hits += hits
            top8_total += len(reference_ids)
            reference_mass = softmax_f32(reference_values)
            candidate_mass = softmax_f32(candidate[reference_ids])
            top8_mass.append(float(np.sum(reference_mass * np.log(reference_mass / candidate_mass))))
            hit = int(candidate.argmax()) == int(reference_argmax[step, branch])
            top1_hits += int(hit)
            top1_total += 1
            bucket = top1_scope.setdefault(scope, [0, 0])
            bucket[0] += int(hit)
            bucket[1] += 1
            if not hit:
                argmax_flips.append(
                    {
                        "step": step,
                        "branch": branch,
                        "reference": int(reference_argmax[step, branch]),
                        "candidate": int(candidate.argmax()),
                        "reference_top1_value": float(reference_top_vals[step, branch, 0]),
                        "candidate_top1_value": float(candidate.max()),
                    }
                )

    worst.sort(reverse=True)
    return {
        "case": case_name,
        "phase": manifest_entry["phase"],
        "cfg": manifest_entry["cfg"],
        "prefix_lengths": manifest_entry["prefix_lengths"],
        "steps": len(rows["steps"]),
        "full_vocab_rows": len(kl_rows),
        "full_vocab_steps": [list(step) for step in full_steps],
        "kl": summarize(kl_rows),
        "kl_by_scope": {scope: summarize(values) for scope, values in sorted(kl_scope.items())},
        "top1_agreement": (top1_hits / top1_total) if top1_total else None,
        "top1_hits": top1_hits,
        "top1_rows": top1_total,
        "top1_by_scope": {
            scope: {"hits": hits, "rows": total, "agreement": hits / total}
            for scope, (hits, total) in sorted(top1_scope.items())
        },
        "top8_recall": (top8_hits / top8_total) if top8_total else None,
        "top8_hits": top8_hits,
        "top8_rows": top8_total,
        "top8_kl": summarize(top8_mass),
        "argmax_flips": argmax_flips[:8],
        "worst_rows": [{"scope": scope, "kl": kl} for kl, scope in worst[:5]],
    }


def aggregate(results: list[dict]) -> dict:
    kl_rows: list[float] = []
    scope_rows: dict[str, list[float]] = {}
    pooled_weighted = 0.0
    pooled_rows = 0
    hits = 0
    total = 0
    scope_top1: dict[str, list[int]] = {}
    top8_hits = 0
    top8_total = 0
    for result in results:
        # Pool the raw hit counts; re-deriving them from the per-case ratios
        # would round twice. Only the per-scope means are approximate here.
        case_kl = result["kl"]
        if case_kl.get("rows"):
            pooled_weighted += float(case_kl["mean"]) * int(case_kl["rows"])
            pooled_rows += int(case_kl["rows"])
        hits += result["top1_hits"]
        total += result["top1_rows"]
        top8_hits += result["top8_hits"]
        top8_total += result["top8_rows"]
        for scope, values in result["kl_by_scope"].items():
            scope_rows.setdefault(scope, []).extend([values["mean"]] * values["rows"])
        for scope, entry in result["top1_by_scope"].items():
            bucket = scope_top1.setdefault(scope, [0, 0])
            bucket[0] += entry["hits"]
            bucket[1] += entry["rows"]
    return {
        "cases": len(results),
        "rows": total,
        "top1_agreement": (hits / total) if total else None,
        "top1_by_scope": {
            scope: {"hits": v[0], "rows": v[1], "agreement": v[0] / v[1]}
            for scope, v in sorted(scope_top1.items())
        },
        "top8_recall": (top8_hits / top8_total) if top8_total else None,
        "scope_mean_kl": {scope: float(np.mean(values)) for scope, values in sorted(scope_rows.items())},
        # Row-weighted mean over every recorded full-vocabulary row: the headline
        # gate value, so it is computed once here rather than re-derived later.
        "pooled_mean_kl": (pooled_weighted / pooled_rows) if pooled_rows else None,
    }


def _gate_passes(summary: dict) -> bool:
    """The broad floor from AGENTS.md: mean KL <= 0.05 and top-1 >= 0.9."""

    pooled = summary.get("pooled_mean_kl")
    top1 = summary.get("top1_agreement")
    if pooled is None or top1 is None:
        return False
    return float(pooled) <= 0.05 and float(top1) >= 0.9


def _command_line(argv: list[str] | None) -> str:
    """The exact command this artifact came from, never hand-written."""

    args = sys.argv[1:] if argv is None else list(argv)
    return "python3 scripts/yue2_ar_replay.py" + "".join(
        f" {shlex.quote(str(value))}" for value in args
    )


def _host_identity() -> dict:
    """Physical host identity, read from the machine rather than typed."""

    name = ""
    try:
        name = Path("/etc/hostname").read_text().strip()
    except OSError:
        pass
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu = ""
    try:
        completed = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=60, check=False
        )
        # rocminfo lists the CPU agent with a "Marketing Name" too, and on this
        # host that line also contains "Radeon"; the GPU agent's line ends in
        # "Graphics", so prefer it over the first match.
        candidates = [
            line.split(":", 1)[1].strip()
            for line in completed.stdout.splitlines()
            if line.strip().startswith("Marketing Name:")
        ]
        radeon = [value for value in candidates if "Radeon" in value]
        graphics = [value for value in radeon if "Graphics" in value]
        gpu = (graphics or radeon or candidates or [""])[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return {"name": name, "cpu": cpu, "gpu": gpu}


def _provenance(argv: list[str] | None, fixtures: Path, repeat: str) -> dict:
    """Evidence-policy fields: model, quant, workload, host, hardware, command."""

    integrity = fixtures / "integrity.json"
    digest = ""
    if integrity.is_file():
        digest = hashlib.sha256(integrity.read_bytes()).hexdigest()
    return {
        "model": "m-a-p/YuE2-3B",
        "quant": "bf16",
        "measurement_basis": (
            "Frozen upstream-oracle fixtures replayed on this host; KL, top-1 and "
            "top-8 are measured against the reference's recorded full-vocabulary "
            "distributions, and the same runtime instance replays the first case "
            "after the whole matrix."
        ),
        "correctness_gate": {
            "broad_floor": {"mean_kl_max": 0.05, "top1_min": 0.9},
            "repeat": repeat,
        },
        "host": _host_identity(),
        "command": _command_line(argv),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fixtures_integrity_sha256": digest,
        "torch_imported": "torch" in sys.modules,
        "python": platform.python_version(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default=str(FIXTURES))
    parser.add_argument(
        "--full-artifacts",
        default="",
        help="directory of full-precision oracle artifacts; when set, full-vocabulary KL uses every recorded step",
    )
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--cases", default="", help="comma-separated case names to replay (default: all)")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--rope-table", choices=("reference", "f64"), default="reference")
    parser.add_argument("--prefill-variant", choices=("strict", "hipblaslt"), default="hipblaslt")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--repeat", type=int, default=0, help="repeat the first case and require bit equality")
    parser.add_argument(
        "--keep-host-weights",
        action="store_true",
        help="keep the host-side weight arrays resident after upload (debugging)",
    )
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    import os

    from hipengine.loading.yue2 import load_yue2_weights
    from hipengine.runtime.yue2_ar import Yue2ArRuntime

    fixtures = Path(args.fixtures)
    full_artifacts = Path(args.full_artifacts) if args.full_artifacts else None
    manifest = json.loads((fixtures / "manifest.json").read_text())
    names = sorted(manifest)
    if args.cases:
        requested = [name.strip() for name in args.cases.split(",") if name.strip()]
        unknown = [name for name in requested if name not in manifest]
        if unknown:
            raise SystemExit(f"unknown case(s): {unknown}")
        names = requested
    elif args.limit_cases:
        names = names[: args.limit_cases]
    model_dir = args.model_dir or os.environ.get("YUE2_MODEL_DIR") or _cached_model_dir()

    started = time.time()
    weights = load_yue2_weights(model_dir)
    print(f"[replay] loaded weights in {time.time() - started:.1f}s ({weights.bytes / 2**30:.2f} GiB)", flush=True)
    max_context = 0
    for name in names:
        with np.load(fixtures / f"{name}.npz") as handle:
            max_context = max(max_context, int(handle["prefix_positive"].shape[0]) + int(handle["tokens"].shape[0]))
    runtime = Yue2ArRuntime(
        weights,
        max_context=max_context,
        branches=2,
        backend=args.backend,
        prefill_variant=args.prefill_variant,
        rope_table=args.rope_table,
    )
    print(f"[replay] runtime ready: backend={runtime.backend} max_context={max_context} "
          f"rope={args.rope_table} prefill={args.prefill_variant}", flush=True)
    weights_bytes = int(weights.bytes)
    if not args.keep_host_weights:
        # The runtime holds only the embedding row table; dropping the loader's
        # host arrays returns ~6 GiB before the replay starts.
        del weights
        gc.collect()
        print("[replay] released host weight arrays", flush=True)
    results = []
    first_case_rows: dict | None = None
    repeat_state = "not-requested"
    try:
        for name in names:
            entry = manifest[name]
            fixture = load_fixture(fixtures, full_artifacts, name)
            case_started = time.time()
            rows = replay_case(runtime, fixture, max_steps=args.max_steps or None)
            if name == names[0]:
                first_case_rows = rows
            result = evaluate(name, entry, fixture, rows)
            result["seconds"] = time.time() - case_started
            results.append(result)
            kl = result["kl"]
            print(
                f"[replay] {name}: rows={result['full_vocab_rows']} "
                f"mean_kl={kl.get('mean', float('nan')):.3e} max_kl={kl.get('max', float('nan')):.3e} "
                f"top1={result['top1_agreement']:.4f} top8_recall={result['top8_recall']:.4f} "
                f"{result['seconds']:.1f}s",
                flush=True,
            )
            if result["argmax_flips"]:
                print(f"          flips: {result['argmax_flips'][:3]}", flush=True)
        if args.repeat:
            # Replay the first case again after every other case has run: the
            # comparison covers both repeatability and cross-case isolation.
            name = names[0]
            fixture = load_fixture(fixtures, full_artifacts, name)
            repeat_rows = replay_case(runtime, fixture, max_steps=args.max_steps or None)
            identical = first_case_rows is not None and rows_equal(first_case_rows, repeat_rows)
            print(
                f"[replay] repeat of {name} after {len(names)} cases: "
                f"{'bit-identical' if identical else 'DIFFERS'}",
                flush=True,
            )
            repeat_state = "bit-identical" if identical else "DIFFERS"
            if not identical:
                return 1
    finally:
        runtime.close()

    summary = aggregate(results)
    print("[replay] aggregate:", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.json:
        artifact = {
            "protocol": "yue2-ar-replay",
            **_provenance(argv, fixtures, repeat_state),
            "status": "pass" if _gate_passes(summary) else "fail",
            "fixtures": str(fixtures),
            "full_artifacts": str(full_artifacts) if full_artifacts else None,
            "model_dir": str(model_dir),
            "backend": runtime.backend,
            "rope_table": args.rope_table,
            "prefill_variant": args.prefill_variant,
            "max_context": max_context,
            "weights_bytes": weights_bytes,
            "aggregate": summary,
            "cases": results,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(f"[replay] wrote {args.json}", flush=True)
    return 0


def rows_equal(first: dict, second: dict) -> bool:
    """Bit equality of every replayed row (prefill and decode)."""
    if not np.array_equal(first["prefill"][1], second["prefill"][1]):
        return False
    for key in ("prefill_negative",):
        if (key in first) != (key in second):
            return False
        if key in first and not np.array_equal(first[key][1], second[key][1]):
            return False
    if len(first["steps"]) != len(second["steps"]):
        return False
    return all(
        np.array_equal(left, right)
        for first_step, second_step in zip(first["steps"], second["steps"])
        for left, right in zip(first_step, second_step)
    )


def load_fixture(fixtures: Path, full_artifacts: Path | None, name: str) -> dict:
    """Load one case, preferring the full-precision artifact when available."""
    path = fixtures / f"{name}.npz"
    if full_artifacts is not None:
        candidate = full_artifacts / f"{name}.npz"
        if candidate.is_file():
            path = candidate
    with np.load(path) as handle:
        return {key: handle[key] for key in handle.files}


def _cached_model_dir() -> str:
    cache = Path.home() / ".cache/huggingface/hub"
    for directory in sorted(cache.glob("models--m-a-p--YuE2-3B/snapshots/*")):
        if (directory / "model.safetensors").is_file():
            return str(directory)
    raise SystemExit("YuE2-3B checkpoint not found; set YUE2_MODEL_DIR")


if __name__ == "__main__":
    raise SystemExit(main())
