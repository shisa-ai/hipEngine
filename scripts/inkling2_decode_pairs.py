#!/usr/bin/env python3
"""Paired decode-rate comparison of Inkling 2 GGUF decode arms on one GPU.

Each arm is ``scripts/inkling2_decode_bench.py`` in a fresh process with its own Inkling
variables, so an arm is exactly what a user reaches by setting those variables. Every arm runs
once per pair, and the pairs run in the listed order and again in reverse order. A difference
between two arms must therefore appear in both orders, which separates it from anything that
drifts while the GPU warms.

    python scripts/inkling2_decode_pairs.py run MODEL --backend hip_gfx1100 --execution-profile fast \
        --quant UD-Q4_K_M --arms default routes resident graph --pairs 5 --reps 3 --output pairs.json
    python scripts/inkling2_decode_pairs.py trace KERNEL_CSV MARKER_CSV LABEL STEPS

``trace`` reduces one ROCTx range of a ``rocprofv3 --kernel-trace --marker-trace`` run to kernel
time and launches per step; the decode-trace command is in
``docs/campaigns/INKLING2-MINI-GGUF-EXECUTION.md``. ``run --kernel-trace ARM=FILE`` attaches such a
file to an arm as its GPU-busy figure.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).with_name("inkling2_decode_bench.py")
PROTOCOL = ("one greedy stream, prefix cache off, drafting off; each arm is the decode bench in a fresh "
            "process (one warmup request, then the repetitions); every arm runs once per pair and the "
            "pairs run in the listed order and again in reverse order; decode_ms_per_token is the "
            "median over a run's repetitions and the arm figure is the median over runs")
# Every Inkling decode variable at its default; an arm overrides some of them.
BASE = {"HIPENGINE_INKLING2_DEVICE_ROUTES": "0", "HIPENGINE_INKLING2_RESIDENT_DECODE": "0",
        "HIPENGINE_INKLING2_DECODE_GRAPH": "0"}
ARMS = {
    "default": {},
    "routes": {"HIPENGINE_INKLING2_DEVICE_ROUTES": "1"},
    "resident": {"HIPENGINE_INKLING2_DEVICE_ROUTES": "1", "HIPENGINE_INKLING2_RESIDENT_DECODE": "1"},
    "graph": {"HIPENGINE_INKLING2_DEVICE_ROUTES": "1", "HIPENGINE_INKLING2_RESIDENT_DECODE": "1",
              "HIPENGINE_INKLING2_DECODE_GRAPH": "1"},
}
ORDERS = ("forward", "reverse")
# A win must exceed this ratio in both orders.
BEAT = 1.03
# What each run keeps per depth in the artifact.
RUN_DEPTH_KEYS = ("prompt_tokens", "reps", "decode_tok_s", "decode_ms_per_token", "first_token_s",
                  "output_ids_sha256", "calls_per_token", "graph")


def variables(item):
    """Parse ``HIPENGINE_<name>=value``."""
    name, separator, value = item.partition("=")
    if not separator or not re.fullmatch(r"HIPENGINE_[A-Z0-9_]+", name) or not value:
        raise ValueError(f"variable must look like HIPENGINE_<name>=value, got {item!r}")
    return {name: value}


def resolve_arms(names, settings):
    """Arm name to its variables: a known arm, plus anything ``--set`` added or replaced."""
    arms = {name: dict(ARMS.get(name, {})) for name in names}
    for setting in settings:
        arm, separator, item = setting.partition(":")
        if not separator or arm not in arms:
            raise ValueError(f"--set expects an arm that runs, as ARM:VAR=VALUE, got {setting!r}")
        arms[arm].update(variables(item))
    return arms


def schedule(arms, pairs):
    """One (order, pair, arm) per bench process."""
    return [(order, pair, arm) for order in ORDERS for pair in range(pairs)
            for arm in (arms if order == "forward" else arms[::-1])]


def run_bench(model, args, arm_env, label, output):
    command = [sys.executable, "-B", str(BENCH), str(model), "--backend", args.backend,
               "--execution-profile", args.execution_profile, "--quant", args.quant,
               "--prompt-tokens", *map(str, args.prompt_tokens), "--output-tokens", str(args.output_tokens),
               "--max-sequence-length", str(args.max_sequence_length), "--reps", str(args.reps),
               "--warmup", "1", "--counts", "--label", label, "--output", str(output)]
    env = dict(os.environ)
    env.update(BASE)
    env.update(arm_env)
    subprocess.run(command, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(output.read_text(encoding="utf-8"))


def spread(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def average_calls(samples):
    """Mean HIP-call tallies over the runs that counted."""
    total, counted = {}, 0
    for calls in samples:
        if not calls:
            continue
        counted += 1
        for name, entry in calls.items():
            slot = total.setdefault(name, {"calls": 0.0, "seconds": 0.0, "bytes": 0.0})
            for field in slot:
                slot[field] += entry[field]
    for entry in total.values():
        for field in entry:
            entry[field] /= counted
    return total


def summarize_arm(runs, reference):
    """Figures of one arm per depth, overall, per order, and against the reference arm's runs."""
    depths = []
    for tokens in [depth["prompt_tokens"] for depth in runs[0]["depths"]]:
        rows = [next(depth for depth in run["depths"] if depth["prompt_tokens"] == tokens) for run in runs]
        reference_rows = [next(depth for depth in run["depths"] if depth["prompt_tokens"] == tokens)
                          for run in reference]
        by_order, reference_by_order = {}, {}
        for order in ORDERS:
            selected = [row for row, run in zip(rows, runs, strict=True) if run["order"] == order]
            by_order[order] = {"decode_tok_s": spread([row["decode_tok_s"]["median"] for row in selected]),
                               "first_token_s": spread([row["first_token_s"]["median"] for row in selected])}
            reference_by_order[order] = [row["decode_tok_s"]["median"]
                                         for row, run in zip(reference_rows, reference, strict=True)
                                         if run["order"] == order]
        digests = sorted({digest for row in rows for digest in row["output_ids_sha256"]})
        reference_digests = sorted({digest for row in reference_rows for digest in row["output_ids_sha256"]})
        graph = [row["graph"] for row in rows if row.get("graph")]
        depths.append({
            "prompt_tokens": tokens,
            "runs": len(rows),
            "decode_tok_s": spread([row["decode_tok_s"]["median"] for row in rows]),
            "decode_ms_per_token": statistics.median(row["decode_ms_per_token"] for row in rows),
            "first_token_s": spread([row["first_token_s"]["median"] for row in rows]),
            "by_order": by_order,
            "tok_s_ratio_to_reference": {
                order: by_order[order]["decode_tok_s"]["median"] / statistics.median(reference_by_order[order])
                for order in ORDERS},
            "identical_output": len(digests) == 1,
            "output_ids_sha256": digests,
            "output_identical_to_reference": digests == reference_digests,
            "calls_per_token": average_calls([row["calls_per_token"] for row in rows]),
            "graph": None if not graph else {
                "captures_per_request": statistics.mean(row["captures"] for row in graph),
                "capture_s": statistics.mean(row["capture_s"] for row in graph),
                "replays_per_token": statistics.mean(row["replays_per_token"] for row in graph)},
        })
    return depths


def decide(depths, reference_depths):
    """Per depth: is the arm faster than the reference in both orders, and is its output the same?"""
    decisions = []
    for depth, reference in zip(depths, reference_depths, strict=True):
        ratios = depth["tok_s_ratio_to_reference"]
        faster = all(ratios[order] > BEAT for order in ORDERS)
        same = depth["output_identical_to_reference"]
        decisions.append({"prompt_tokens": depth["prompt_tokens"], "tok_s_ratio_to_reference": ratios,
                          "faster_in_both_orders": faster, "output_identical_to_reference": same,
                          "promote": faster and same,
                          "reference_decode_tok_s": reference["decode_tok_s"]})
    return decisions


def family(name):
    """Kernel name without return type, namespace noise, template arguments and parameters."""
    name = name.replace("(anonymous namespace)::", "")
    name = re.sub(r"^void ", "", name)
    name = re.sub(r"[<(].*", "", name)
    return name.rsplit("::", 1)[-1] or name


def trace_range(kernel_csv, marker_csv, label, steps):
    """Kernel time and launches per step inside one ROCTx range."""
    begin = end = None
    with open(marker_csv, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Function"] == label:
                begin, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
    if begin is None:
        raise ValueError(f"marker range {label!r} not found in {marker_csv}")
    families = collections.defaultdict(lambda: [0, 0])
    with open(kernel_csv, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            start = int(row["Start_Timestamp"])
            if begin <= start <= end:
                entry = families[family(row["Kernel_Name"])]
                entry[0] += 1
                entry[1] += int(row["End_Timestamp"]) - start
    if steps <= 0 or not families:
        raise ValueError("steps must be positive and the range must hold kernels")
    return {"label": label, "steps": steps,
            "wall_ms_per_step": (end - begin) / steps / 1e6,
            "kernel_ms_per_step": sum(nanoseconds for _, nanoseconds in families.values()) / steps / 1e6,
            "launches_per_step": sum(count for count, _ in families.values()) / steps,
            "families": [{"family": name, "launches_per_step": count / steps, "ms_per_step": nanoseconds / steps / 1e6}
                         for name, (count, nanoseconds) in sorted(families.items(), key=lambda item: -item[1][1])]}


def provenance(model, args):
    from hipengine.benchmark.provenance import collect_artifact_provenance

    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    hipcc_version = None
    if version_file and Path(version_file).is_file():
        hipcc_version = Path(version_file).read_text(encoding="utf-8")
    return collect_artifact_provenance(
        repo_root=REPO_ROOT, configured_backend=args.backend, model_path=model, quant=args.quant,
        kv_dtype="fp32", command=[sys.argv[0], *sys.argv[1:]],
        environment={name: value for name, value in sorted(os.environ.items()) if name.startswith("HIPENGINE_")},
        timing_protocol=PROTOCOL, warmups=1, repetitions=args.pairs * args.reps, hipcc_version=hipcc_version)


def run(model, args):
    arms = resolve_arms(args.arms, args.set)
    traces = {}
    for item in args.kernel_trace:
        arm, separator, path = item.partition("=")
        if not separator or arm not in arms:
            raise ValueError(f"--kernel-trace expects an arm that runs, as ARM=FILE, got {item!r}")
        traces[arm] = Path(path)
    runs = []
    with tempfile.TemporaryDirectory(prefix="inkling2-decode-pairs-") as scratch:
        for order, pair, arm in schedule(list(arms), args.pairs):
            label = f"{order}-{pair}-{arm}"
            result = run_bench(model, args, arms[arm], label, Path(scratch) / f"{label}.json")
            # Per-run figures only; call counts and diagnostics are averaged into the arm summary.
            runs.append({"order": order, "pair": pair, "arm": arm, "resolved_profile": result["resolved_profile"],
                         "fell_back_to_strict": result["fell_back_to_strict"],
                         "depths": [{key: depth[key] for key in RUN_DEPTH_KEYS} for depth in result["depths"]]})
            print(f"{label}: " + "  ".join(f"{depth['prompt_tokens']} tokens {depth['decode_tok_s']['median']:.2f} tok/s"
                                           for depth in result["depths"]), file=sys.stderr, flush=True)
    reference = [run for run in runs if run["arm"] == args.arms[0]]
    reference_depths = summarize_arm(reference, reference)
    summaries, decisions = {}, {}
    for arm in arms:
        depths = summarize_arm([run for run in runs if run["arm"] == arm], reference)
        summaries[arm] = {"environment": arms[arm],
                          "kernel_trace": json.loads(traces[arm].read_text(encoding="utf-8"))
                          if arm in traces else None,
                          "depths": depths}
        decisions[arm] = None if arm == args.arms[0] else decide(depths, reference_depths)
    return {
        "kind": "inkling2-mini-gguf-decode-pairs",
        "status": "diagnostic",
        "recorded": date.today().isoformat(),
        "purpose": "Decode rate of the Inkling 2 GGUF decode arms against the first arm, in paired runs and "
                   "in both orders, with output identity, HIP call counts and graph accounting.",
        "scope": "Development lane. Not a Loom comparison; does not update a scoreboard row.",
        "tool": "scripts/inkling2_decode_pairs.py",
        "timing_protocol": PROTOCOL,
        "protocol": {"arms": list(arms), "orders": list(ORDERS), "pairs": args.pairs, "reps": args.reps,
                     "warmup": 1, "output_tokens": args.output_tokens, "beat_threshold": BEAT},
        "provenance": provenance(model, args),
        "model": {"path": str(model), "bytes": model.stat().st_size},
        "backend": args.backend,
        "requested_profile": args.execution_profile,
        "arms": summaries,
        "decision": decisions,
        "runs": runs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    paired = commands.add_parser("run", help="run the paired protocol and write one JSON artifact")
    paired.add_argument("model", type=Path)
    paired.add_argument("--backend", default="hip_gfx1100")
    paired.add_argument("--execution-profile", default="fast")
    paired.add_argument("--quant", default=None)
    paired.add_argument("--prompt-tokens", type=int, nargs="+", default=[47])
    paired.add_argument("--output-tokens", type=int, default=128)
    paired.add_argument("--max-sequence-length", type=int, default=4096)
    paired.add_argument("--pairs", type=int, default=5)
    paired.add_argument("--reps", type=int, default=3)
    paired.add_argument("--arms", nargs="+", default=list(ARMS),
                        help="arm names in run order; the first is the reference")
    paired.add_argument("--set", action="append", default=[], metavar="ARM:VAR=VALUE",
                        help="replace one variable of one arm that runs")
    paired.add_argument("--kernel-trace", action="append", default=[], metavar="ARM=FILE",
                        help="JSON written by the trace command")
    paired.add_argument("--output", type=Path, required=True)
    traced = commands.add_parser("trace", help="reduce one ROCTx range of a rocprofv3 CSV trace")
    traced.add_argument("kernel_csv", type=Path)
    traced.add_argument("marker_csv", type=Path)
    traced.add_argument("label")
    traced.add_argument("steps", type=int)
    args = parser.parse_args(argv)
    if args.command == "trace":
        print(json.dumps(trace_range(args.kernel_csv, args.marker_csv, args.label, args.steps), indent=1))
        return 0
    model = Path(os.path.abspath(args.model))
    if not model.is_file():
        parser.error(f"model file not found: {model}")
    if args.pairs < 1 or args.reps < 1 or args.output_tokens < 2 or min(args.prompt_tokens) < 1:
        parser.error("need at least one pair, one repetition, two output tokens and positive prompts")
    if len(set(args.arms)) != len(args.arms):
        parser.error("arm names must be distinct")
    if max(args.prompt_tokens) + args.output_tokens > args.max_sequence_length:
        parser.error("max-sequence-length must cover the longest prompt plus the output tokens")
    try:
        resolve_arms(args.arms, args.set)
    except ValueError as exc:
        parser.error(str(exc))
    result = run(model, args)
    text = json.dumps(result, indent=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
