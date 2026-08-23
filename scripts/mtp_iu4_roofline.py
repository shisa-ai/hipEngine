#!/usr/bin/env python3
"""Build and run the gfx11 MTP arithmetic-ceiling screen.

The benchmark compares register-resident FP16/BF16/I8/I4 WMMA and the vector
DOT4/DOT8 instructions relevant to hipEngine's current verifier. It does not
include activation quantization, weight loads, reconstruction, epilogues, or a
model forward, so its result is diagnostic ceiling evidence only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "microbench" / "mtp_iu4_roofline.hip"
EXPECTED_ISA = (
    "v_wmma_f32_16x16x16_f16",
    "v_wmma_f32_16x16x16_bf16",
    "v_wmma_i32_16x16x16_iu8",
    "v_wmma_i32_16x16x16_iu4",
    "v_dot4_i32_iu8",
    "v_dot8_i32_iu4",
)
WMMA_THEORETICAL_GFX1151_TOPS = {
    "fp16_wmma": 59.392,
    "bf16_wmma": 59.392,
    "i8i8_wmma": 59.392,
    "u8s8_wmma": 59.392,
    "i4i4_wmma": 118.784,
    "u4s4_wmma": 118.784,
    "u4u4_wmma": 118.784,
}


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_arch(override: str | None) -> tuple[str, dict[str, Any]]:
    rocminfo = _run(["rocminfo"]).stdout
    if override:
        arch = override
    else:
        matches = re.findall(r"^\s*Name:\s+(gfx\d+)\s*$", rocminfo, re.MULTILINE)
        if not matches:
            raise RuntimeError("rocminfo did not expose a gfx architecture")
        arch = matches[0]
    segment_match = re.search(
        rf"^\s*Name:\s+{re.escape(arch)}\s*$([\s\S]*?)(?=^\*{{7}}|\Z)",
        rocminfo,
        re.MULTILINE,
    )
    segment = segment_match.group(1) if segment_match else ""

    def field(pattern: str, cast: type[int] | type[str] = str) -> Any:
        match = re.search(pattern, segment, re.MULTILINE)
        if match is None:
            return None
        value = match.group(1).strip()
        return cast(value)

    snapshot = {
        "arch": arch,
        "marketing_name": field(r"^\s*Marketing Name:\s+(.+?)\s*$"),
        "compute_units": field(r"^\s*Compute Unit:\s+(\d+)\s*$", int),
        "simds_per_cu": field(r"^\s*SIMDs per CU:\s+(\d+)\s*$", int),
        "max_clock_mhz": field(r"^\s*Max Clock Freq\. \(MHz\):\s+(\d+)\s*$", int),
        "wavefront_size": field(r"^\s*Wavefront Size:\s+(\d+)", int),
        "max_waves_per_cu": field(r"^\s*Max Waves Per CU:\s+(\d+)", int),
    }
    return arch, snapshot


def _git_state() -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=ROOT).stdout.strip()
    branch = _run(["git", "branch", "--show-current"], cwd=ROOT).stdout.strip()
    status = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT).stdout.splitlines()
    staged = [line for line in status if line and line[0] not in {" ", "?"}]
    unstaged = [line for line in status if len(line) > 1 and line[1] not in {" ", "?"}]
    untracked = [line for line in status if line.startswith("??")]
    return {
        "commit": commit,
        "branch": branch,
        "staged_dirty": bool(staged),
        "unstaged_dirty": bool(unstaged),
        "untracked_dirty": bool(untracked),
        "dirty": bool(status),
        "status_entry_count": len(status),
        "note": "Diagnostic-only run; dirty state is disclosed and cannot support a retained product performance claim.",
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {}
    for measurement in raw["measurements"]:
        milliseconds = [float(value) for value in measurement["milliseconds"]]
        total_ops = float(measurement["total_ops"])
        median_ms = statistics.median(milliseconds)
        tops_samples = [total_ops / (value / 1000.0) / 1.0e12 for value in milliseconds]
        lane = str(measurement["lane"])
        row = {
            "lane": lane,
            "chains": int(measurement["chains"]),
            "instruction_ops": int(measurement["instruction_ops"]),
            "total_ops": int(measurement["total_ops"]),
            "milliseconds": milliseconds,
            "timing": {
                "samples": len(milliseconds),
                "median_ms": median_ms,
                "p95_ms": _percentile(milliseconds, 0.95),
                "min_ms": min(milliseconds),
                "max_ms": max(milliseconds),
                "mean_ms": statistics.mean(milliseconds),
                "stdev_ms": statistics.stdev(milliseconds) if len(milliseconds) > 1 else 0.0,
            },
            "throughput": {
                "median_tops": total_ops / (median_ms / 1000.0) / 1.0e12,
                "best_tops": max(tops_samples),
                "min_tops": min(tops_samples),
            },
        }
        theoretical = WMMA_THEORETICAL_GFX1151_TOPS.get(lane)
        if theoretical is not None and raw["device"]["arch"] == "gfx1151":
            row["throughput"]["theoretical_tops"] = theoretical
            row["throughput"]["median_percent_of_theoretical"] = (
                100.0 * row["throughput"]["median_tops"] / theoretical
            )
        rows.append(row)
        current = best.get(lane)
        if current is None or row["throughput"]["median_tops"] > current["median_tops"]:
            best[lane] = {
                "chains": row["chains"],
                "median_tops": row["throughput"]["median_tops"],
                "best_tops": row["throughput"]["best_tops"],
                "median_ms": row["timing"]["median_ms"],
            }
    return rows, best


def _ratio(best: dict[str, dict[str, Any]], numerator: str, denominator: str) -> float:
    return best[numerator]["median_tops"] / best[denominator]["median_tops"]


def _parse_isa(cache_dir: Path) -> dict[str, Any]:
    assembly_files = sorted(cache_dir.glob("*-amdgcn-*-gfx*.s"))
    text = "\n".join(path.read_text(errors="replace") for path in assembly_files)
    counts = {mnemonic: len(re.findall(rf"\b{re.escape(mnemonic)}\b", text)) for mnemonic in EXPECTED_ISA}
    return {
        "assembly_files": [str(path) for path in assembly_files],
        "mnemonic_counts": counts,
        "all_expected_seen": all(count > 0 for count in counts.values()),
    }


def _build(args: argparse.Namespace, arch: str) -> tuple[Path, list[str], str, dict[str, Any]]:
    hipcc_candidate = str(Path(args.hipcc).expanduser())
    hipcc_resolved = shutil.which(hipcc_candidate)
    if hipcc_resolved is None:
        raise RuntimeError(f"hipcc was not found: {args.hipcc}")
    hipcc = str(Path(hipcc_resolved).resolve())
    version = _run([hipcc, "--version"]).stdout.strip()
    source_sha = _sha256(SOURCE)
    flags = ["-O3", "-std=c++17", f"--offload-arch={arch}", "--save-temps"]
    key_payload = json.dumps(
        {"source_sha256": source_sha, "hipcc": hipcc, "hipcc_version": version, "flags": flags},
        sort_keys=True,
    ).encode()
    key = hashlib.sha256(key_payload).hexdigest()[:20]
    cache_dir = Path(args.cache_root).expanduser().resolve() / key
    executable = cache_dir / "mtp_iu4_roofline"
    build_command = [hipcc, *flags, str(SOURCE), "-o", str(executable)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_hit = executable.exists()
    if args.require_cached and not cache_hit:
        raise RuntimeError(f"required cached executable is missing: {executable}")
    if not cache_hit or args.rebuild:
        _run(build_command, cwd=cache_dir)
        cache_hit = False
    isa = _parse_isa(cache_dir)
    if not isa["all_expected_seen"]:
        raise RuntimeError(f"compiled assembly missed an expected instruction: {isa['mnemonic_counts']}")
    build = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_sha,
        "hipcc": hipcc,
        "hipcc_version": version,
        "flags": flags,
        "cache_key": key,
        "cache_dir": str(cache_dir),
        "cache_hit": cache_hit,
        "command": _command_text(build_command),
    }
    return executable, build_command, version, {"build": build, "isa": isa}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="compact JSON artifact path")
    parser.add_argument("--arch", help="HIP offload architecture (default: detect with rocminfo)")
    parser.add_argument("--hipcc", default=os.environ.get("HIPCC", "hipcc"))
    parser.add_argument("--cache-root", default="~/.cache/hipengine/research/mtp_iu4_roofline")
    parser.add_argument("--iterations", type=int, default=65536)
    parser.add_argument("--blocks", type=int, default=0, help="0 uses 16x HIP-reported multiprocessors")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0 or args.blocks < 0 or args.samples <= 0 or args.warmups < 0:
        raise ValueError("iterations/samples must be positive; blocks/warmups must be non-negative")
    arch, hardware = _detect_arch(args.arch)
    if not arch.startswith("gfx11"):
        raise RuntimeError(f"this GFX11 benchmark does not support {arch}")

    executable, _build_command, _version, build_data = _build(args, arch)
    run_command = [
        str(executable),
        "--iterations", str(args.iterations),
        "--blocks", str(args.blocks),
        "--samples", str(args.samples),
        "--warmups", str(args.warmups),
    ]
    raw_result = json.loads(_run(run_command).stdout)
    rows, best = _summarize(raw_result)
    comparisons = {
        "u4s4_wmma_vs_fp16_wmma": _ratio(best, "u4s4_wmma", "fp16_wmma"),
        "u4s4_wmma_vs_bf16_wmma": _ratio(best, "u4s4_wmma", "bf16_wmma"),
        "u4s4_wmma_vs_u8s8_wmma": _ratio(best, "u4s4_wmma", "u8s8_wmma"),
        "u4s4_wmma_vs_u8s8_dot4": _ratio(best, "u4s4_wmma", "u8s8_dot4"),
        "u4s4_dot8_vs_u8s8_dot4": _ratio(best, "u4s4_dot8", "u8s8_dot4"),
        "i4i4_vs_u4s4_wmma": _ratio(best, "i4i4_wmma", "u4s4_wmma"),
        "u4u4_vs_u4s4_wmma": _ratio(best, "u4u4_wmma", "u4s4_wmma"),
    }
    artifact = {
        "schema_version": 1,
        "date": dt.datetime.now(dt.timezone.utc).date().isoformat(),
        "kind": "hipengine_mtp_iu4_instruction_roofline",
        "status": "diagnostic_only",
        "performance_claim": False,
        "verdict_scope": (
            "Register-resident GFX11 instruction-throughput ceiling only; excludes activation packing, "
            "weight traffic, scale/zero correction, epilogues, kernel composition, and model execution."
        ),
        "hardware": {**hardware, "hostname": os.uname().nodename},
        "software": {"repo": _git_state(), **build_data["build"]},
        "methodology": {
            "workgroup": "one wave32 (32 threads) per block",
            "dependency_chains": [2, 4, 8],
            "timing": "HIP events around one kernel launch; warmups excluded",
            "statistics": "median, linearly interpolated p95, min, max, mean, sample stdev",
            "operation_count": (
                "WMMA: 16*16*16 MAC * 2 ops = 8192 ops per wave instruction. "
                "DOT4/DOT8: 32 lanes * 4/8 MAC * 2 ops = 256/512 ops per wave instruction."
            ),
            "interpretation": "Select the strongest median chain count per lane; do not use best-of-sample as the headline.",
            "run_command": _command_text(run_command),
            "raw_parameters": raw_result["parameters"],
        },
        "isa_verification": build_data["isa"],
        "measurements": rows,
        "best_median_by_lane": best,
        "comparisons": comparisons,
        "limitations": [
            "Not an operation-complete verifier projection or GEMM.",
            "No activation quantization, weight decode/load, correction, scaling, conversion, SiLU, residual, or launch graph is included.",
            "HIP device clock_rate_khz is a property ceiling, not a measured sustained clock; clocks and package power were not pinned.",
            "The dirty shared worktree is disclosed; this diagnostic cannot promote a product path or update a topline model row.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "best_median_by_lane": best,
        "comparisons": comparisons,
        "isa_all_expected_seen": build_data["isa"]["all_expected_seen"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        raise SystemExit(2)
