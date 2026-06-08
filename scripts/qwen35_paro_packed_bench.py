#!/usr/bin/env python3
"""Side-by-side decode/prefill benchmark across one or more Qwen3.5/PARO checkpoints.

The motivating use case is comparing the canonical packed PARO format
(``mlp.shared_expert.{gate,up,down}_proj.{qweight,qzeros,scales,theta,pairs,
channel_scales}``) against any other variant once the checkpoint exists.
Today the upstream ``z-lab/Qwen3.5-35B-A3B-PARO`` ships only fp16
``shared_expert.*.weight`` and is *not* loadable by hipEngine — see
``--how-to-pack`` below for the paroquant invocation that produces a packed
artifact.

The script:
1. Validates each checkpoint up-front (packed PARO shared-expert present).
2. Runs ``scripts/qwen35_paro_bench.py`` as a subprocess once per checkpoint
   with identical decode/prefill settings, captures the JSON output.
3. Emits a side-by-side markdown table covering prefill tps, decode tps,
   peak VRAM (HIP + tracked), and any geometric mean speedups.

The harness does **not** fabricate results: if a checkpoint is missing the
packed shared-expert tensors the run is skipped with a clear error pointing
at paroquant. If only one valid checkpoint is provided the script still runs
and emits a single-row table.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qwen35_kv_policy_args import add_kv_policy_args, append_kv_policy_flags, kv_policy_json, resolve_args_kv_policy

BENCH_SCRIPT = REPO_ROOT / "scripts" / "qwen35_paro_bench.py"

HOW_TO_PACK = textwrap.dedent(
    """
    Producing a packed-shared-expert PARO checkpoint
    -----------------------------------------------
    The upstream z-lab Qwen3.5-35B-A3B-PARO snapshot ships only fp16
    mlp.shared_expert.{gate,up,down}_proj.weight. hipEngine now requires the
    packed PARO layout for the dense shared expert. To mint a packed
    checkpoint, re-run paroquant *without* mlp.shared_expert in the
    --skipped-modules list. From ~/amd-gpu-tuning/paroquant:

        # 1. Quantize (this is the expensive step; uses a single CUDA GPU)
        bash experiments/optimize/4bit_moe.sh \\
            --model <path-to-base-Qwen3.5-MoE> \\
            --skipped-modules "mlp.gate" "mlp.shared_expert_gate" \\
                              "linear_attn.in_proj_a" "linear_attn.in_proj_b"

        # 2. Convert the resulting per-module .pt files into a packed
        #    safetensors checkpoint:
        python -m paroquant.cli.convert \\
            --model <path-to-base-Qwen3.5-MoE> \\
            --result-dir <dir-with-per-module-.pt-files> \\
            --output-path <packed-output-dir> \\
            --mode real

        # 3. Strip the now-duplicate fp16 fallback tensors:
        python scripts/strip_paro_safetensors.py \\
            --input-dir <packed-output-dir> \\
            --output-dir <stripped-packed-output-dir>

    Then point this script at <stripped-packed-output-dir> with --checkpoint.
    """
).strip()


@dataclass
class CheckpointSpec:
    label: str
    path: Path


@dataclass
class RunResult:
    label: str
    path: Path
    skipped: bool
    skipped_reason: str | None
    raw: dict[str, Any] | None
    prefill_tps: float | None
    decode_tps: float | None
    decode_p50_ms: float | None
    decode_p95_ms: float | None
    peak_hip_used_gib: float | None
    peak_tracked_gib: float | None
    owned_session_peak_gib: float | None


def parse_checkpoint_spec(raw: str) -> CheckpointSpec:
    if "=" in raw:
        label, _, path = raw.partition("=")
        return CheckpointSpec(label=label.strip(), path=Path(path.strip()))
    path = Path(raw.strip())
    return CheckpointSpec(label=path.name, path=path)


def validate_packed_shared_expert(path: Path) -> str | None:
    """Return None if the checkpoint has packed shared-expert tensors, else a reason string."""
    if not path.exists():
        return f"checkpoint directory does not exist: {path}"
    from hipengine.loading import load_weight_index

    try:
        index = load_weight_index(path)
    except Exception as exc:  # pragma: no cover - depends on disk contents
        return f"failed to load weight index: {exc}"
    # Probe layer 0 for the packed shared-expert tensor family. We use raw
    # name matching against the normalized HF tensor names so the check works
    # for both `model.language_model.` and `model.` rooted layouts.
    required_packed = {
        f"layers.0.mlp.shared_expert.{proj}.{suffix}"
        for proj in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("qweight", "qzeros", "scales", "theta", "pairs", "channel_scales")
    }
    from hipengine.loading.qwen35_paro import normalize_qwen35_weight_name

    present = {normalize_qwen35_weight_name(name) for name in index.tensors}
    missing = sorted(required_packed - present)
    if missing:
        preview = ", ".join(missing[:6])
        more = "" if len(missing) <= 6 else f" (+{len(missing) - 6} more)"
        return f"missing packed shared-expert tensors: {preview}{more}"
    return None


def run_bench(spec: CheckpointSpec, common_args: list[str], json_out: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--model",
        str(spec.path),
        "--json",
        str(json_out),
        *common_args,
    ]
    print(f"[run] {spec.label}: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"bench run failed for {spec.label} (exit {completed.returncode})")
    if not json_out.exists():
        raise RuntimeError(f"bench did not produce JSON at {json_out}")
    return json.loads(json_out.read_text())


def extract_metrics(label: str, path: Path, raw: dict[str, Any]) -> RunResult:
    prefill = raw.get("prefill_native") or raw.get("prefill") or {}
    decode = raw.get("decode") or {}
    memory = raw.get("memory") or {}

    def _f(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return RunResult(
        label=label,
        path=path,
        skipped=False,
        skipped_reason=None,
        raw=raw,
        prefill_tps=_f(prefill.get("tokens_per_second")),
        decode_tps=_f(decode.get("tokens_per_second")),
        decode_p50_ms=_f(decode.get("p50_ms_per_token") or decode.get("median_ms_per_token")),
        decode_p95_ms=_f(decode.get("p95_ms_per_token")),
        peak_hip_used_gib=_f(memory.get("hip_peak_used_gib") or memory.get("hip_max_used_gib")),
        peak_tracked_gib=_f(memory.get("tracked_peak_gib") or memory.get("tracked_peak_allocated_gib")),
        owned_session_peak_gib=_f(memory.get("owned_session_peak_gib")),
    )


def format_metric(value: float | None, fmt: str = "{:.2f}") -> str:
    return "n/a" if value is None else fmt.format(value)


def emit_markdown_table(results: list[RunResult], baseline_label: str | None) -> str:
    columns = [
        ("label", "label"),
        ("prefill_tps", "prefill tok/s"),
        ("decode_tps", "decode tok/s"),
        ("decode_p50_ms", "decode p50 ms/tok"),
        ("decode_p95_ms", "decode p95 ms/tok"),
        ("peak_hip_used_gib", "HIP peak used (GiB)"),
        ("peak_tracked_gib", "tracked peak (GiB)"),
        ("owned_session_peak_gib", "session owned peak (GiB)"),
    ]
    show_speedup = baseline_label is not None and any(r.label == baseline_label and not r.skipped for r in results)
    header = "| " + " | ".join(name for _key, name in columns) + (" | vs baseline (decode tps) |" if show_speedup else " |")
    sep = "|" + "|".join("---" for _ in columns) + ("|---|" if show_speedup else "|")
    rows = [header, sep]
    baseline = next((r for r in results if r.label == baseline_label and not r.skipped), None) if show_speedup else None
    for r in results:
        if r.skipped:
            rows.append(f"| {r.label} | skipped: {r.skipped_reason} |" + " |" * (len(columns) - 1 + (1 if show_speedup else 0)))
            continue
        cells = []
        for key, _name in columns:
            if key == "label":
                cells.append(r.label)
            else:
                cells.append(format_metric(getattr(r, key)))
        if show_speedup:
            if baseline is None or baseline.decode_tps is None or r.decode_tps is None:
                speedup = "n/a"
            else:
                speedup = f"{r.decode_tps / baseline.decode_tps:.3f}x"
            cells.append(speedup)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help=(
            "Checkpoint to benchmark. Repeatable. Use 'LABEL=PATH' to control the row label, "
            "or just PATH to use the directory basename."
        ),
    )
    parser.add_argument("--prompt-length", type=int, default=16, help="Repeated-token prompt length")
    parser.add_argument("--decode-tokens", type=int, default=32, help="Generated tokens per bench")
    parser.add_argument("--warmup-decode-tokens", type=int, default=2)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--graph-steps-per-replay", type=int, default=1)
    parser.add_argument(
        "--baseline",
        default=None,
        help="Label of the run to use as the decode-tps baseline in the comparison table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "qwen35_paro_packed_compare",
        help="Directory to drop per-run JSON outputs and the consolidated comparison report.",
    )
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=None,
        help="Forwarded to qwen35_paro_bench.py to avoid spawning hipcc.",
    )
    parser.add_argument(
        "--require-cached-build",
        action="store_true",
        help="Forwarded to qwen35_paro_bench.py; fails if any kernel needs to be rebuilt.",
    )
    add_kv_policy_args(parser, help_prefix="KV storage forwarded to qwen35_paro_bench.py")
    parser.add_argument(
        "--how-to-pack",
        action="store_true",
        help="Print the paroquant runbook for producing a packed-shared-expert checkpoint and exit.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the upfront packed-shared-expert presence check (will probably crash the bench).",
    )
    args = parser.parse_args()

    if args.how_to_pack:
        print(HOW_TO_PACK)
        return 0

    if not args.checkpoint:
        parser.error("at least one --checkpoint is required (or pass --how-to-pack)")
    if not BENCH_SCRIPT.exists():
        parser.error(f"missing bench script {BENCH_SCRIPT}")

    specs = [parse_checkpoint_spec(raw) for raw in args.checkpoint]
    seen_labels: set[str] = set()
    for spec in specs:
        if spec.label in seen_labels:
            parser.error(f"duplicate label {spec.label!r}; use LABEL=PATH to disambiguate")
        seen_labels.add(spec.label)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    common_args = [
        "--prompt-length", str(args.prompt_length),
        "--decode-tokens", str(args.decode_tokens),
        "--warmup-decode-tokens", str(args.warmup_decode_tokens),
        "--token-id", str(args.token_id),
        "--max-layers", str(args.max_layers),
        "--graph-steps-per-replay", str(args.graph_steps_per_replay),
    ]
    if args.compiler_version_file is not None:
        common_args.extend(["--compiler-version-file", str(args.compiler_version_file)])
    if args.require_cached_build:
        common_args.append("--require-cached-build")
    kv_forward = append_kv_policy_flags("", args).strip().split()
    common_args.extend(kv_forward)

    results: list[RunResult] = []
    for spec in specs:
        if not args.skip_validation:
            reason = validate_packed_shared_expert(spec.path)
            if reason is not None:
                results.append(RunResult(
                    label=spec.label,
                    path=spec.path,
                    skipped=True,
                    skipped_reason=reason,
                    raw=None,
                    prefill_tps=None,
                    decode_tps=None,
                    decode_p50_ms=None,
                    decode_p95_ms=None,
                    peak_hip_used_gib=None,
                    peak_tracked_gib=None,
                    owned_session_peak_gib=None,
                ))
                continue
        json_out = args.output_dir / f"{spec.label}.json"
        try:
            raw = run_bench(spec, common_args, json_out)
        except Exception as exc:
            results.append(RunResult(
                label=spec.label,
                path=spec.path,
                skipped=True,
                skipped_reason=f"bench run failed: {exc}",
                raw=None,
                prefill_tps=None,
                decode_tps=None,
                decode_p50_ms=None,
                decode_p95_ms=None,
                peak_hip_used_gib=None,
                peak_tracked_gib=None,
                owned_session_peak_gib=None,
            ))
            continue
        results.append(extract_metrics(spec.label, spec.path, raw))

    markdown = emit_markdown_table(results, baseline_label=args.baseline)
    print()
    print(markdown)
    print()

    report = {
        "config": {
            "prompt_length": args.prompt_length,
            "decode_tokens": args.decode_tokens,
            "warmup_decode_tokens": args.warmup_decode_tokens,
            "token_id": args.token_id,
            "max_layers": args.max_layers,
            "graph_steps_per_replay": args.graph_steps_per_replay,
            "checkpoints": [{"label": s.label, "path": str(s.path)} for s in specs],
            "baseline": args.baseline,
            "kv_policy": kv_policy_json(resolve_args_kv_policy(args, block_size=256)),
        },
        "rows": [
            {
                "label": r.label,
                "path": str(r.path),
                "skipped": r.skipped,
                "skipped_reason": r.skipped_reason,
                "prefill_tps": r.prefill_tps,
                "decode_tps": r.decode_tps,
                "decode_p50_ms": r.decode_p50_ms,
                "decode_p95_ms": r.decode_p95_ms,
                "peak_hip_used_gib": r.peak_hip_used_gib,
                "peak_tracked_gib": r.peak_tracked_gib,
                "owned_session_peak_gib": r.owned_session_peak_gib,
            }
            for r in results
        ],
        "markdown": markdown,
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "comparison.md").write_text(markdown + "\n", encoding="utf-8")
    print(f"report: {args.output_dir / 'comparison.json'}")
    print(f"table:  {args.output_dir / 'comparison.md'}")

    if all(r.skipped for r in results):
        sys.stderr.write(
            "\nno successful runs.\n"
            "if every row is skipped with 'missing packed shared-expert tensors', the\n"
            "checkpoint is not in the packed PARO format hipEngine currently requires.\n"
            "see scripts/qwen35_paro_packed_bench.py --how-to-pack for the runbook.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
