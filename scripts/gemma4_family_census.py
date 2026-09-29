#!/usr/bin/env python3
"""Gemma 4 kernel-family census for hipEngine and llama.cpp ``rocprofv3`` traces.

Two subcommands:

``drive``  Run under ``rocprofv3 --kernel-trace --output-format csv``. Loads the
           public Gemma 4 surface, warms a full-shape prefill plus eight decode
           steps, then runs ``--prefill-repeats`` prefills of the frozen
           campaign corpus (realistic expert routing, not a repeated token) and
           ``--decode`` greedy single-token forwards. One-second idle sleeps
           separate the phases, so ``rollup`` splits the trace by timestamp.
           Warm the JIT cache first with ``--warm-only`` outside the profiler.

``rollup`` Bucket every kernel of a phase into a family (dense, MoE gate_up,
           MoE down, layer-29 experts, routing glue, attention sliding/global,
           norm, ...), print a table and optionally write compact JSON.

Recipe (from the gemma4 worktree; ``N`` is the physical GPU):

    hipcc --version > /tmp/hipcc_version.txt
    ROCR_VISIBLE_DEVICES=N .venv/bin/python scripts/gemma4_family_census.py drive --warm-only
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=N HIPENGINE_REQUIRE_CACHED_BUILD=1 \\
      HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipcc_version.txt PYTHONPATH=. \\
      rocprofv3 --kernel-trace --output-format csv -d <dir> -o run -- \\
      .venv/bin/python scripts/gemma4_family_census.py drive --prompt 1024 --decode 64
    python3 scripts/gemma4_family_census.py rollup hipengine <dir>/run_kernel_trace.csv \\
      --decode-steps 64 --json <out.json>

    # llama.cpp: trace two llama-bench runs with the same flags,
    #   A: -p <P> -n 0 -r 2 --no-warmup          (prefill only)
    #   B: -p 0 -n 64 -d <P> -r 1 --no-warmup    (depth prefill + 64 decode steps)
    python3 scripts/gemma4_family_census.py rollup llamacpp <B>/run_kernel_trace.csv \\
      --pp-trace <A>/run_kernel_trace.csv --pp-runs 2 --decode-steps 64

Without the compiler-version pair the profiled process spawns ``clang++
--version`` and deadlocks under the profiler.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# (family, regex) — first match wins. Layer 29 carries the artifact's only
# Q5_K gate_up / Q8_0 down experts and is reported separately.
ENGINE_FAMILIES: tuple[tuple[str, str], ...] = (
    # q8_0_selected_grouped_wmma_prefill is P3's newly-bound Q8_0 down owner,
    # which runs only for layer 29 (the sole Q8_0 expert down in the artifact);
    # without it the fix lands in `other` and layer 29 reads as improved by a
    # sum that no longer includes its own kernel.
    ("moe.l29_experts", r"gguf_q4_k_selected_dual_grouped_rowbatch|gguf_k_selected_prefill_out_kernel|q8_0_selected_grouped_wmma_prefill"),
    ("moe.gate_up", r"q4_k_selected"),
    ("moe.down", r"q5_1_selected"),
    ("moe.act_quant", r"q8_1_mmq_ds4_pack|q8_1_mmq_gather_ds4_pack"),
    ("moe.route_glue", r"qwen35_moe_|gemma4_moe_|gemma4_expert_weight_scale|qwen35_router_select"),
    ("moe.router_gemm", r"router_logits|gemma4_router_prescale"),
    ("attn.sliding", r"^attn_fwd|gemma4_attention_decode_class_kernel<unsigned short, 1|^dim_sliding:"),
    # flash_attn_tile<512 is P2's tiled head_dim-512 prefill port. It replaced
    # gemma4_attention_decode_class_kernel<...,2,...>, which is the only symbol
    # this rule used to match, so post-P2 runs reported attn.global as 0.00 ms
    # on zero launches while the kernel sat in `other`.
    ("attn.global", r"gemma4_attention_decode_class_kernel<unsigned short, 2|flash_attn_tile<512|^dim_global:"),
    ("lm_head", r"gguf_k_pack8_prefill_out_kernel<unsigned short, float"),
    ("dense.q8_0", r"gguf_q8_0|gguf_k_pack8_prefill_out_kernel|mmq128_prefill"),
    ("norm", r"rmsnorm"),
    ("elementwise", r"rotary|gelu|branch_add|gemma4_scale|embedding"),
    # The k_*_to_* converters are P2's dtype staging (Q bf16->f32, KV bf16->f16,
    # mask u8->f16, out f32->bf16). They are conversions, so runtime_copy_fill
    # is their honest bucket; leaving them in `other` hides the staging cost
    # that P2's row exists to weigh against the kernel's win.
    ("runtime_copy_fill", r"__amd_rocclr|k_kv_bf16_to_f16|k_q_bf16_to_f32|k_f32_to_bf16|k_mask_u8_to_f16"),
)
# ggml types: 7 Q5_1, 8 Q8_0, 12 Q4_K, 13 Q5_K.
LLAMACPP_FAMILIES: tuple[tuple[str, str], ...] = (
    ("lm_head", r"^lm_head:"),
    ("moe.l29_experts", r"<\(ggml_type\)13|mul_mat_q<\(ggml_type\)8, 64,"),
    ("moe.gate_up", r"<\(ggml_type\)12"),
    ("moe.down", r"<\(ggml_type\)7"),
    ("moe.act_quant", r"quantize_mmq_q8_1<\(mmq_q8_1_ds_layout\)1"),
    ("moe.route_glue", r"mm_ids_helper|moe_weighted_reduction|topk_moe|k_get_rows|op_repeat"),
    ("moe.router_gemm", r"Cijk_|mul_mat_vec_f"),
    ("attn.sliding", r"flash_attn_ext_f16<256|flash_attn_tile<256|flash_attn_combine_results<256|mask_to_KV_max<32"),
    ("attn.global", r"flash_attn_tile<512|flash_attn_combine_results<512|mask_to_KV_max<4"),
    ("attn.kv_write", r"k_set_rows|convert_unary"),
    ("dense.q8_0", r"<\(ggml_type\)8|quantize_mmq_q8_1<\(mmq_q8_1_ds_layout\)0|quantize_q8_1"),
    ("norm", r"rms_norm"),
    ("elementwise", r"unary_gated|bin_bcast|scale_f32|softcap"),
    ("runtime_copy_fill", r"__amd_rocclr"),
)

Kernel = tuple[int, int, str]


def _kernel_name(row: dict[str, str]) -> str:
    """Disambiguate same-named kernels whose role is only visible in the grid."""

    name, grid_x = row["Kernel_Name"], row["Grid_Size_X"]
    if "mul_mat_vec_q<(ggml_type)8" in name and grid_x == "8388608":
        return "lm_head:" + name  # 262144 vocab rows x 32 lanes
    if "gemma4_attention_decode_dimension_kernel" in name:
        return ("dim_sliding:" if grid_x == "4096" else "dim_global:") + name
    return name


def load_phases(path: Path, gap_s: float) -> list[list[Kernel]]:
    rows: list[Kernel] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["Start_Timestamp"]), int(row["End_Timestamp"]), _kernel_name(row)))
    rows.sort()
    phases: list[list[Kernel]] = [[]]
    last_end: int | None = None
    for start, end, name in rows:
        if last_end is not None and start - last_end > gap_s * 1e9:
            phases.append([])
        phases[-1].append((start, end, name))
        last_end = end if last_end is None else max(last_end, end)
    return [phase for phase in phases if phase]


def family_of(table: tuple[tuple[str, str], ...], name: str) -> str:
    for family, pattern in table:
        if re.search(pattern, name):
            return family
    return "other"


Totals = dict[str, tuple[float, float]]  # kernel name -> (launches, ms)


def totals(kernels: list[Kernel] | Any) -> Totals:
    out: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    for start, end, name in kernels:
        out[name][0] += 1
        out[name][1] += (end - start) / 1e6
    return {name: (count, ms) for name, (count, ms) in out.items()}


def rollup_totals(
    by_name: Totals, table: tuple[tuple[str, str], ...], divisor: float = 1.0, span_ms: float | None = None
) -> dict[str, Any]:
    families: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    other: collections.Counter[str] = collections.Counter()
    for name, (count, ms) in by_name.items():
        family = family_of(table, name)
        families[family][0] += count / divisor
        families[family][1] += ms / divisor
        if family == "other":
            other[name[:100]] += ms / divisor
    busy = sum(ms for _, ms in families.values())
    return {
        "busy_ms": round(busy, 4),
        "span_ms": None if span_ms is None else round(span_ms / divisor, 4),
        "families": {
            family: {"ms": round(ms, 4), "launches": round(count, 1)}
            for family, (count, ms) in sorted(families.items(), key=lambda item: -item[1][1])
        },
        "other_top": {name: round(ms, 4) for name, ms in other.most_common(8)},
    }


def print_rollup(title: str, result: dict[str, Any]) -> None:
    busy = result["busy_ms"]
    span = "" if result["span_ms"] is None else f", span {result['span_ms']:.3f} ms"
    print(f"## {title}: busy {busy:.3f} ms{span}")
    for family, row in result["families"].items():
        share = 100 * row["ms"] / busy if busy else 0.0
        print(f"  {family:20s} {row['ms']:10.3f} ms  {share:5.1f}%  launches {row['launches']:8.1f}")
    for name, ms in result["other_top"].items():
        print(f"     other: {ms:.3f} ms {name}")


def _span(kernels: list[Kernel]) -> float:
    return (kernels[-1][1] - kernels[0][0]) / 1e6


def cmd_rollup(args: argparse.Namespace) -> int:
    if args.engine == "hipengine":
        # The driver sleeps between phases: warm, prefill x repeats, decode.
        # Busy phases only (load leaves tiny copy/fill phases); the last
        # prefill is the steady-state one.
        phases = [p for p in load_phases(args.trace, 0.5) if sum(e - s for s, e, _ in p) > 50e6]
        prefill_k, decode_k = phases[-2], phases[-1]
        table = ENGINE_FAMILIES
        prefill = rollup_totals(totals(prefill_k), table, span_ms=_span(prefill_k))
        decode = rollup_totals(totals(decode_k), table, args.decode_steps, span_ms=_span(decode_k))
        source = {"trace": str(args.trace)}
    else:
        # llama.cpp runs a second HIP stream, and rocprofv3 timestamps from
        # different queues interleave, so a timestamp split leaks kernels
        # between forwards. Count instead: prefill is the -p trace divided by
        # its run count; decode is the -d trace (one depth prefill of the
        # same shape plus N steps) minus one prefill.
        if args.pp_trace is None:
            raise SystemExit("llamacpp needs --pp-trace (llama-bench -p P -n 0 -r R --no-warmup)")
        table = LLAMACPP_FAMILIES
        pp = totals(k for phase in load_phases(args.pp_trace, 0.05) for k in phase)
        per_run = {name: (count / args.pp_runs, ms / args.pp_runs) for name, (count, ms) in pp.items()}
        prefill = rollup_totals(per_run, table)
        decode = None
        if args.trace is not None:
            tg = totals(k for phase in load_phases(args.trace, 0.05) for k in phase)
            diff: Totals = {}
            for name, (count, ms) in tg.items():
                base_count, base_ms = per_run.get(name, (0.0, 0.0))
                if count - base_count > 0.5 and ms - base_ms > 0:
                    diff[name] = (count - base_count, ms - base_ms)
            decode = rollup_totals(diff, table, args.decode_steps)
        source = {"pp_trace": str(args.pp_trace), "pp_runs": args.pp_runs, "tg_trace": str(args.trace)}
    result = {"engine": args.engine, **source, "prefill": prefill, "decode_per_token": decode}
    print_rollup(f"{args.engine} prefill", prefill)
    if decode is not None:
        print_rollup(f"{args.engine} decode per token", decode)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    return 0


def cmd_drive(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO / "scripts"))
    from gemma4_campaign_bench import DEFAULT_ARTIFACT, exact_prompt_ids

    import hipengine
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    llm = hipengine.LLM(model=str(args.artifact or DEFAULT_ARTIFACT))
    generator = llm._get_text_generator()
    generator.context_length = int(args.context)
    runner = generator._ensure_runner()
    prompt_ids = exact_prompt_ids(generator.tokenize, args.prompt)

    def prefill() -> tuple[Any, float]:
        runner.reset()
        runtime.device_synchronize()
        start = time.perf_counter()
        logits = runner.forward(list(prompt_ids))
        runtime.device_synchronize()
        return logits, time.perf_counter() - start

    def decode(logits: Any, steps: int) -> float:
        token = int(runner.next_token(logits))
        start = time.perf_counter()
        for _ in range(steps):
            logits = runner.forward([token])
            token = int(runner.next_token(logits))
        runtime.device_synchronize()
        return time.perf_counter() - start

    logits, _ = prefill()
    decode(logits, 8)
    if args.warm_only:
        print("[family_census] warm-only complete", flush=True)
        return 0
    time.sleep(1.0)
    for index in range(args.prefill_repeats):
        logits, seconds = prefill()
        print(f"[family_census] prefill{index} tokens={args.prompt} tps={args.prompt / seconds:.1f}", flush=True)
        time.sleep(1.0)
    seconds = decode(logits, args.decode)
    print(f"[family_census] decode steps={args.decode} tps={args.decode / seconds:.2f}", flush=True)
    time.sleep(1.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    drive = sub.add_parser("drive", help="phase-separated driver to run under rocprofv3")
    drive.add_argument("--artifact", type=Path, default=None)
    drive.add_argument("--prompt", type=int, default=1024)
    drive.add_argument("--prefill-repeats", type=int, default=2)
    drive.add_argument("--decode", type=int, default=64)
    drive.add_argument("--context", type=int, default=8192)
    drive.add_argument("--warm-only", action="store_true")
    drive.set_defaults(func=cmd_drive)
    roll = sub.add_parser("rollup", help="bucket a kernel trace into families")
    roll.add_argument("engine", choices=("hipengine", "llamacpp"))
    roll.add_argument("trace", type=Path, nargs="?", default=None,
                      help="hipEngine: the drive trace; llama.cpp: the -p 0 -n N -d P trace")
    roll.add_argument("--pp-trace", type=Path, default=None, help="llama.cpp: the -p P -n 0 trace")
    roll.add_argument("--pp-runs", type=int, default=2, help="llama.cpp: -r of the -p trace")
    roll.add_argument("--decode-steps", type=int, default=64)
    roll.add_argument("--json", type=Path, default=None)
    roll.set_defaults(func=cmd_rollup)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
