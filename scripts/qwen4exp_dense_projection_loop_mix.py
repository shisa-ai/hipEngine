#!/usr/bin/env python3
"""Instruction-issue mix of the dense GGUF prefill projection's inner loop.

The projection family that carries every dense projection in a Qwen4Exp prefill
is a scalar-FP32-FMA kernel: it dequantizes each weight element on the fly and
multiplies it into one of ``COL_TILE * ROW_BATCH`` accumulators. That design
fixes an instruction mix, and the mix — not the memory system — is what caps the
kernel, because the arithmetic-intensity of the operation is in the thousands of
FLOPs per byte while the achieved rate is a tenth of peak.

This script compiles the kernel, finds the innermost backward-branch loop for a
requested template instantiation, and counts issue slots by instruction class.
The useful output is the ratio of FMA-class instructions to total instructions:
it is the fraction of the machine's issue bandwidth that can possibly do
arithmetic, and therefore the ceiling this design can reach.

It reports the ceiling; it does not claim the kernel reaches it. The gap between
the two is the latency-hiding headroom, and the distance from the ceiling to
peak is the headroom that requires a different execution mechanism.

Example:
    python3 scripts/qwen4exp_dense_projection_loop_mix.py \
        --arch gfx1151 --clock-ghz 2.9 --compute-units 40 \
        --json benchmarks/results/<dir>/loop-mix.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip"

# Template instantiations of gguf_k_prefill_out_coltile_rowbatch_kernel, keyed
# by the mangled-suffix fragment that identifies the arguments.
# Order is <scalar_t, out_t, qtype, COL_TILE, ROW_BATCH, WAVE_SCALE>.
INSTANTIATIONS = {
    "production_coltile8_rowbatch4_wave_scale": (
        "IffLi8ELi8ELi4ELb1EEEvPKT_PKhPT0_iii",
        "float, float, 8, 8, 4, true",
    ),
    "superseded_coltile8_rowbatch4": (
        "IffLi8ELi8ELi4ELb0EEEvPKT_PKhPT0_iii",
        "float, float, 8, 8, 4, false",
    ),
    "coltile4_rowbatch8": (
        "IffLi8ELi4ELi8ELb0EEEvPKT_PKhPT0_iii",
        "float, float, 8, 4, 8, false",
    ),
    "coltile16_rowbatch4": (
        "IffLi8ELi16ELi4ELb0EEEvPKT_PKhPT0_iii",
        "float, float, 8, 16, 4, false",
    ),
    "coltile32_rowbatch1": (
        "IffLi8ELi32ELi1ELb0EEEvPKT_PKhPT0_iii",
        "float, float, 8, 32, 1, false",
    ),
}

FMA_CLASS = re.compile(r"(fmac|dual_fma|fma_mix|v_fma)")
MNEMONIC = re.compile(r"\s+([a-z][a-z0-9_]*)[\s$]")
BACKWARD_BRANCH = re.compile(r"\s+s_branch\s+(\S+)")
LABEL = re.compile(r"^(\S+):")


def compile_assembly(source: Path, arch: str, outdir: Path) -> Path:
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        raise SystemExit("hipcc not found on PATH; source the ROCm environment")
    subprocess.run(
        [
            hipcc, "-save-temps=obj", "-O3", f"--offload-arch={arch}", "-mcumode",
            "-mllvm", "-amdgpu-unroll-threshold-local=600", "-c", str(source),
            "-o", str(outdir / "loopmix.o"),
        ],
        check=True,
        cwd=outdir,
        capture_output=True,
    )
    candidates = sorted(outdir.glob("*-gfx*.s"))
    if not candidates:
        raise SystemExit(f"no device assembly produced in {outdir}")
    return candidates[0]


def kernel_body(lines: list[str], tag: str) -> list[str]:
    prefix = (
        "_ZN12_GLOBAL__N_142gguf_k_prefill_out_coltile_rowbatch_kernel" + tag
    )
    start = next((i for i, l in enumerate(lines) if l.startswith(prefix)), None)
    if start is None:
        raise SystemExit(f"instantiation {tag} not found in assembly")
    end = next(
        (j for j in range(start + 1, len(lines)) if ".Lfunc_end" in lines[j]),
        len(lines),
    )
    return lines[start:end]


def innermost_loop(body: list[str]) -> list[str]:
    """Return the body of the largest backward-branching block.

    The k-loop is the only backward branch in this kernel, so the largest span
    between a branch and its target is the loop.
    """
    labels: dict[str, int] = {}
    for i, line in enumerate(body):
        m = LABEL.match(line)
        if m:
            labels.setdefault(m.group(1), i)
    best: tuple[int, int, int] | None = None
    for i, line in enumerate(body):
        m = BACKWARD_BRANCH.match(line)
        if m and m.group(1) in labels and labels[m.group(1)] < i:
            span = i - labels[m.group(1)] + 1
            if best is None or span > best[0]:
                best = (span, labels[m.group(1)], i)
    if best is None:
        raise SystemExit("no backward branch found; cannot isolate the k-loop")
    return body[best[1]:best[2] + 1]


def classify(loop: list[str]) -> dict[str, int]:
    ops: collections.Counter[str] = collections.Counter()
    for line in loop:
        m = MNEMONIC.match(line)
        if m:
            ops[m.group(1)] += 1
    return dict(ops.most_common())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--arch", default="gfx1151")
    parser.add_argument("--clock-ghz", type=float, default=2.9)
    parser.add_argument("--compute-units", type=int, default=40)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--in-features", type=int, default=2560)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--instantiation",
        default="production_coltile8_rowbatch4_wave_scale",
        choices=sorted(INSTANTIATIONS),
    )
    args = parser.parse_args()

    peak_gflops = args.compute_units * 128 * 2 * args.clock_ghz
    iterations = args.in_features // args.threads

    tmp = Path(tempfile.mkdtemp(prefix="loopmix-"))
    try:
        asm = compile_assembly(args.source, args.arch, tmp)
        lines = asm.read_text(errors="replace").splitlines()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    tag, template_args = INSTANTIATIONS[args.instantiation]
    loop = innermost_loop(kernel_body(lines, tag))
    ops = classify(loop)
    total = sum(ops.values())
    fma = sum(n for op, n in ops.items() if FMA_CLASS.search(op))

    # The loop body performs ROW_BATCH * COL_TILE accumulating FMAs per
    # iteration; everything else is dequantization, addressing or scheduling.
    accumulators = int(template_args.split(",")[3]) * int(template_args.split(",")[4])
    flops_per_wave_iteration = 32 * 2 * accumulators
    flops_per_cycle = flops_per_wave_iteration / total
    # One FMA per lane per cycle is 64 FLOP/cycle for a wave32.
    issue_ceiling_pct = 100.0 * flops_per_cycle / 64.0

    payload = {
        "schema": 1,
        "kind": "qwen4exp_dense_projection_loop_mix",
        "performance_claim": False,
        "status": "diagnostic",
        "source": str(args.source),
        "arch": args.arch,
        "instantiation": args.instantiation,
        "template_args": template_args,
        "hardware": {
            "compute_units": args.compute_units,
            "clock_ghz": args.clock_ghz,
            "fp32_peak_gflops": round(peak_gflops, 1),
        },
        "loop": {
            "instructions_per_iteration": total,
            "fma_class_instructions": fma,
            "fma_issue_share_pct": round(100.0 * fma / total, 1),
            "iterations_per_thread": iterations,
            "accumulators": accumulators,
        },
        "ceiling": {
            "flops_per_cycle_per_wave32": round(flops_per_cycle, 2),
            "issue_limited_share_of_peak_pct": round(issue_ceiling_pct, 1),
            "issue_limited_gflops": round(peak_gflops * issue_ceiling_pct / 100.0, 1),
            "note": (
                "Upper bound for this design at perfect issue. Real kernels lose "
                "more to memory-latency stalls; the gap between this and the "
                "measured rate is latency-hiding headroom, and the gap between "
                "this ceiling and peak is what needs a different mechanism."
            ),
        },
        "instruction_histogram": ops,
        "top_non_fma": {
            op: n for op, n in ops.items() if not FMA_CLASS.search(op)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"{args.instantiation}  <{template_args}>")
    print(f"  k-loop: {total} instructions/iteration, {fma} FMA-class "
          f"({100.0 * fma / total:.1f}% of issue slots)")
    print(f"  issue-limited ceiling: {issue_ceiling_pct:.1f}% of "
          f"{peak_gflops:.0f} GFLOP/s = {peak_gflops * issue_ceiling_pct / 100:.0f} GFLOP/s")
    print(f"  non-FMA instructions: {total - fma} "
          f"({100.0 * (total - fma) / total:.1f}%)")
    print("  top non-FMA:")
    for op, n in list(payload["top_non_fma"].items())[:10]:
        print(f"    {op:22s} {n:5d}  {100.0 * n / total:5.1f}%")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
