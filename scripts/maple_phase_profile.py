#!/usr/bin/env python3
"""Profile corrected Maple public prefill, c1 decode, and c8 batch decode.

All libraries are prebuilt with a pinned compiler-version cache key before any
``rocprofv3`` child starts. Each phase runs in its own process and trace. The
artifact reports wall, kernel, host-gap, launch, and family attribution per
request/batch step and per useful token.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "deepgrove/maple-preview-2bit-mlx"
DEFAULT_VERSION_FILE = Path("/tmp/hipengine-maple-hipcc-version.txt")
DEFAULT_RAW_ROOT = Path("/tmp/hipengine-maple-phase-profile")
PINNED_REVISION = "361db5da5e74ff6fcdd852d478e1f266ce11013a"

FAMILY_RULES = (
    ("maple_selected_ternary_dual_grouped", "expert_gate_up"),
    ("maple_selected_ternary_grouped", "expert_down"),
    ("qwen35_moe_group_count_active_parallel", "moe_compaction"),
    ("qwen35_moe_group_prefix_active_parallel", "moe_compaction"),
    ("qwen35_moe_group_scatter_active_parallel", "moe_compaction"),
    ("maple_selected_ternary_dual_gemv_batched", "expert_gate_up"),
    ("maple_selected_ternary_dual_gemv", "expert_gate_up"),
    ("maple_selected_ternary_gemv_batched", "expert_down"),
    ("maple_selected_ternary_gemv", "expert_down"),
    ("maple_ternary_qkv_gemm", "qkv_proj"),
    ("maple_ternary_qkv_gemv", "qkv_proj"),
    ("maple_ternary_gemm", "o_proj"),
    ("maple_ternary_gemv", "o_proj"),
    ("maple_affine4_gemv_batched", "lm_head"),
    ("maple_affine4_gemv", "lm_head"),
    ("maple_affine4_embed_batched", "embed"),
    ("maple_affine4_embed", "embed"),
    ("maple_qknorm_rope_kv_write_batched_decode", "qknorm_kvwrite"),
    ("maple_qknorm_rope_kv_write_batched", "qknorm_kvwrite"),
    ("maple_qknorm_rope_kv_write", "qknorm_kvwrite"),
    ("maple_attention_prefill_ring", "attention"),
    ("maple_attention_decode_batched", "attention"),
    ("maple_attention_decode", "attention"),
    ("maple_router_logits_batched", "router_logits"),
    ("maple_router_softmax_topk_batched", "router_topk"),
    ("maple_router_logits", "router_logits"),
    ("maple_router_softmax_topk", "router_topk"),
    ("maple_clamped_swiglu", "swiglu"),
    ("maple_weighted_residual", "weighted_residual"),
    ("maple_kv_span_update", "kv_span_update"),
    ("paro_add_rmsnorm", "add_rmsnorm"),
    ("paro_rmsnorm", "rmsnorm"),
    ("argmax", "argmax"),
)

PHASES = {
    "prefill320": {"units": 2, "measured_units": 1, "useful_tokens_per_unit": 320},
    "decode_c1": {"units": 36, "measured_units": 32, "useful_tokens_per_unit": 1},
    "decode_c8": {"units": 40, "measured_units": 32, "useful_tokens_per_unit": 8},
}

CHILD_SOURCE = r'''
import json
import sys
import time
from hipengine.loading.maple import load_maple_checkpoint
from hipengine.runtime.maple import MapleBatchRunner, MapleRunner

phase, model, backend = sys.argv[1:4]
checkpoint = load_maple_checkpoint(model)
if phase == "prefill320":
    runner = MapleRunner.load(checkpoint, backend=backend, max_context=512)
    tokens = tuple(9000 + (index % 512) for index in range(320))
    times = []
    try:
        for repetition in range(2):
            runner.reset()
            started = time.perf_counter()
            runner.prefill_native(tokens)
            elapsed = time.perf_counter() - started
            if repetition == 1:
                times.append(elapsed)
    finally:
        runner.close()
elif phase == "decode_c1":
    runner = MapleRunner.load(checkpoint, backend=backend, max_context=64)
    token = 9707
    times = []
    try:
        for step in range(36):
            started = time.perf_counter()
            result = runner.step(token)
            elapsed = time.perf_counter() - started
            token = result.token_id
            if step >= 4:
                times.append(elapsed)
    finally:
        runner.close()
elif phase == "decode_c8":
    runner = MapleBatchRunner.load(
        checkpoint, backend=backend, batch_size=8, per_capacity=64
    )
    tokens = [9000 + index for index in range(8)]
    times = []
    try:
        for step in range(40):
            started = time.perf_counter()
            tokens = runner.batch_step(tokens)
            elapsed = time.perf_counter() - started
            if step >= 8:
                times.append(elapsed)
    finally:
        runner.close()
else:
    raise ValueError(phase)
print("MAPLE_PHASE_WALL=" + json.dumps({"phase": phase, "seconds": times}), flush=True)
'''


def _capture(command: list[str], *, timeout: float = 30.0) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": shlex.join(command),
        "returncode": int(completed.returncode),
        "output": (completed.stdout + completed.stderr).strip(),
    }


def family_of(name: str) -> str:
    for marker, family in FAMILY_RULES:
        if marker in name:
            return family
    return "other"


@contextlib.contextmanager
def _arch_environment(arch: str):
    previous_target = os.environ.get("HIP_TARGET_ARCHS")
    previous_offload = os.environ.get("HIP_OFFLOAD_ARCH")
    os.environ["HIP_TARGET_ARCHS"] = arch
    os.environ["HIP_OFFLOAD_ARCH"] = arch
    try:
        yield
    finally:
        if previous_target is None:
            os.environ.pop("HIP_TARGET_ARCHS", None)
        else:
            os.environ["HIP_TARGET_ARCHS"] = previous_target
        if previous_offload is None:
            os.environ.pop("HIP_OFFLOAD_ARCH", None)
        else:
            os.environ["HIP_OFFLOAD_ARCH"] = previous_offload


def write_version_file(path: Path) -> str:
    from hipengine.core.build import compiler_version_text

    text = compiler_version_text("hipcc")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    return text


def prebuild(*, backend: str, version_file: Path) -> list[str]:
    from hipengine.kernels.backends import (
        hip_target_arch_for_backend,
        load_backend_kernel_package,
    )
    from hipengine.kernels.hip_gfx1100.attention.maple_attention import build_maple_attention
    from hipengine.kernels.hip_gfx1100.linear.lm_head import build_lm_head
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        build_qwen35_moe_group_scatter,
    )
    from hipengine.kernels.hip_gfx1100.moe.maple_moe import build_maple_moe
    from hipengine.kernels.hip_gfx1100.norm.rmsnorm import build_qwen35_rmsnorm
    from hipengine.kernels.hip_gfx1100.quant.maple_ternary import build_maple_ternary

    compiler_version = version_file.read_text().strip()
    arch = hip_target_arch_for_backend(backend)
    load_backend_kernel_package(backend)
    outputs = []
    with _arch_environment(arch):
        for build in (
            build_maple_ternary,
            build_maple_attention,
            build_maple_moe,
            build_qwen35_moe_group_scatter,
            build_qwen35_rmsnorm,
            build_lm_head,
        ):
            artifact = build(load=False, compiler_version=compiler_version)
            if not artifact.output_path.is_file():
                raise FileNotFoundError(artifact.output_path)
            cached = build(
                load=False,
                compiler_version=compiler_version,
                require_cached=True,
            )
            outputs.append(str(cached.output_path))
    return outputs


def _parse_trace(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(directory.rglob("*kernel_trace.csv"))
    if not paths:
        raise FileNotFoundError(f"no kernel trace CSV under {directory}")
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                name = (raw.get("Kernel_Name") or raw.get("Name") or "").strip()
                if not name or name.startswith("__amd_rocclr"):
                    continue
                duration = raw.get("DurationNs") or raw.get("Duration")
                if duration:
                    duration_ns = int(duration)
                else:
                    duration_ns = int(raw["End_Timestamp"]) - int(raw["Start_Timestamp"])
                rows.append(
                    {
                        "name": name,
                        "duration_ns": duration_ns,
                        "family": family_of(name),
                        "vgpr": raw.get("VGPR_Count"),
                        "scratch": raw.get("Scratch_Size"),
                        "lds": raw.get("LDS_Block_Size"),
                    }
                )
    return rows


def _profile_phase(
    phase: str,
    *,
    model: str,
    backend: str,
    raw_root: Path,
    version_file: Path,
) -> dict[str, Any]:
    child = raw_root / f"{phase}_child.py"
    child.write_text(CHILD_SOURCE)
    output = raw_root / phase
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HIPENGINE_COMPILER_VERSION_FILE"] = str(version_file)
    env["HIPENGINE_HIP_ARCH"] = "gfx1151"
    env["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        "rocprofv3",
        "--kernel-trace",
        "--output-format",
        "csv",
        "--output-directory",
        str(output),
        "--",
        sys.executable,
        str(child),
        phase,
        model,
        backend,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"profile failed ({completed.returncode}):\n{combined}")
    marker_rows = [line for line in completed.stdout.splitlines() if line.startswith("MAPLE_PHASE_WALL=")]
    if len(marker_rows) != 1:
        raise RuntimeError(f"missing wall marker for {phase}:\n{combined}")
    wall = json.loads(marker_rows[0].split("=", 1)[1])["seconds"]
    trace = _parse_trace(output)
    config = PHASES[phase]
    units = config["units"]
    useful_tokens = config["useful_tokens_per_unit"]
    family_ns = collections.Counter()
    kernel_ns = collections.Counter()
    for row in trace:
        family_ns[row["family"]] += row["duration_ns"]
        kernel_ns[row["name"]] += row["duration_ns"]
    kernel_seconds_per_unit = sum(row["duration_ns"] for row in trace) / 1e9 / units
    wall_median = statistics.median(wall)
    host_gap = wall_median - kernel_seconds_per_unit
    families = [
        {
            "family": family,
            "milliseconds_per_unit": duration / 1e6 / units,
            "microseconds_per_useful_token": duration / 1e3 / units / useful_tokens,
            "kernel_share": duration / sum(family_ns.values()),
        }
        for family, duration in family_ns.most_common()
    ]
    return {
        "phase": phase,
        "command": shlex.join(command),
        "traced_units": units,
        "measured_units": config["measured_units"],
        "useful_tokens_per_unit": useful_tokens,
        "wall_samples_seconds": wall,
        "median_wall_seconds_per_unit": wall_median,
        "kernel_seconds_per_unit": kernel_seconds_per_unit,
        "host_gap_seconds_per_unit": host_gap,
        "host_gap_fraction": host_gap / wall_median,
        "kernel_launches_per_unit": len(trace) / units,
        "useful_tokens_per_second": useful_tokens / wall_median,
        "top_family": families[0]["family"],
        "families": families,
        "top_kernels": [
            {
                "kernel": name,
                "milliseconds_per_unit": duration / 1e6 / units,
            }
            for name, duration in kernel_ns.most_common(30)
        ],
    }


def _git_context() -> dict[str, Any]:
    head = _capture(["git", "rev-parse", "HEAD"])
    status = _capture(["git", "status", "--short", "--untracked-files=no"])
    return {
        "head": head["output"],
        "tracked_status": status["output"],
        "tracked_clean": status["returncode"] == 0 and not status["output"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--version-file", type=Path, default=DEFAULT_VERSION_FILE)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    git = _git_context()
    compiler_version = write_version_file(args.version_file)
    cached_libraries = prebuild(backend=args.backend, version_file=args.version_file)
    if args.raw_root.exists():
        shutil.rmtree(args.raw_root)
    args.raw_root.mkdir(parents=True)
    profiles = [
        _profile_phase(
            phase,
            model=args.model,
            backend=args.backend,
            raw_root=args.raw_root,
            version_file=args.version_file,
        )
        for phase in PHASES
    ]
    rocminfo = _capture(["bash", "-lc", "rocminfo | grep -E 'Name:|Marketing Name:|gfx' | head -8"])
    artifact = {
        "schema_version": 1,
        "date": date.today().isoformat(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_type": "maple_corrected_phase_profile",
        "status": "accepted_diagnostic" if git["tracked_clean"] else "rejected_dirty",
        "performance_claim": False,
        "model": {
            "id": args.model,
            "revision": PINNED_REVISION,
            "quant": "maple_ternary2",
        },
        "hardware": {
            "gpu": "AMD Radeon 8060S Graphics",
            "architecture": "gfx1151",
            "host": platform.node(),
            "rocminfo": rocminfo,
        },
        "software": {
            "python": platform.python_version(),
            "git": git,
            "compiler_version": compiler_version,
            "cached_libraries": cached_libraries,
        },
        "protocol": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "environment": {
                "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
                "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
                "HIPENGINE_COMPILER_VERSION_FILE": str(args.version_file),
                "HIPENGINE_REQUIRE_CACHED_BUILD": "1",
                "HIPENGINE_MAPLE_PREFILL_GQA4": os.environ.get(
                    "HIPENGINE_MAPLE_PREFILL_GQA4"
                ),
            },
            "raw_root": str(args.raw_root),
            "profiles": list(PHASES),
        },
        "profiles": profiles,
        "notes": [
            "Raw rocprof output stays outside the repository under raw_root.",
            "Prefill attribution averages one warm and one measured 320-token native request; wall throughput uses the measured request.",
            "Decode attribution includes warm and measured autoregressive steps with identical kernel structure; wall medians use measured steps only.",
            "This artifact attributes the corrected paths and does not itself retain a new throughput row.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": artifact["status"],
        "profiles": [
            {
                "phase": profile["phase"],
                "useful_tokens_per_second": profile["useful_tokens_per_second"],
                "median_wall_ms": profile["median_wall_seconds_per_unit"] * 1e3,
                "kernel_ms": profile["kernel_seconds_per_unit"] * 1e3,
                "host_gap_ms": profile["host_gap_seconds_per_unit"] * 1e3,
                "launches": profile["kernel_launches_per_unit"],
                "top_family": profile["top_family"],
                "top_families": profile["families"][:6],
            }
            for profile in profiles
        ],
        "artifact": str(args.out),
    }, indent=2, sort_keys=True))
    return 0 if git["tracked_clean"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
