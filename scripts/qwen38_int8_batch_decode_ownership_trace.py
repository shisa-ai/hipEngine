#!/usr/bin/env python3
"""Emit the IKV-C2 row-batched INT8 decode ownership trace.

The IKV-C2 primitive gate requires ``rocprofv3 --kernel-trace`` evidence that the
row-batched INT8 split-K producer and its strided gated reducer each launch once
for all rows, while the c1 leaf producer and reducer launch once per row. That
is what proves the packed consumer is genuinely row-batched rather than a loop of
c1 launches, which is the whole point of the campaign item.

This script runs that trace over the ``c4-ragged`` case of the primitive gate
and reduces the raw CSV to a compact artifact. The capability snapshot is
resolved live from the model registry, so the artifact cannot record an admitted
width that the registry has since moved past.

Profiling protocol (see ``docs/KERNELS.md``): the ``.so`` must already be built
before the profiled process starts, and the profiled process must not spawn
``hipcc``/``clang``. The script warms the JIT cache with an unprofiled gate run
and then profiles with ``HIPENGINE_REQUIRE_CACHED_BUILD=1`` set, which turns an
accidental compile inside the profiler into an error instead of a corrupt trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core import build as _hipengine_build  # noqa: E402
from hipengine.kernels.backends import hip_target_arch_for_backend  # noqa: E402
from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.models.kv_capabilities import KVCapabilityKey, model_artifact_identity  # noqa: E402
from hipengine.models.qwen35 import Qwen35GGUFModel  # noqa: E402

ARTIFACT_KIND = "qwen38_int8_row_batched_decode_ownership_trace"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_COMPILER_VERSION_FILE = Path("/tmp/hipengine-hipcc-version.txt")
DEFAULT_TRACE_ROOT = Path("/tmp/hipengine-ikv-c2-w7900-rocprof")

# Taken from the module that reads them. A hardcoded lookalike such as
# ``HIPENGINE_HIP_REQUIRE_CACHED_BUILD`` silently does nothing, which leaves a
# profiled run free to spawn hipcc and corrupt the trace while the artifact
# still claims it was cache-only.
REQUIRE_CACHED_BUILD_ENV = _hipengine_build._ENV_REQUIRE_CACHED_BUILD
COMPILER_VERSION_FILE_ENV = "HIPENGINE_COMPILER_VERSION_FILE"

# Primitive-gate cases, mirroring the parametrization of
# tests/test_gpu_qwen38_int8_batch_attention_gpu.py. A test asserts these stay in
# sync so the artifact's ``rows``/``live_counts`` cannot drift from the gate.
GATE_CASES: Mapping[str, tuple[int, tuple[int, ...]]] = {
    "c1-8k-page-tail": (1, (8193,)),
    "c2-page-boundary": (2, (255, 257)),
    "c4-ragged": (4, (1, 256, 257, 1025)),
    "c8-sparse": (8, (1, 2, 255, 256, 257, 513, 1023, 0)),
}
GATE_TEST = (
    "tests/test_gpu_qwen38_int8_batch_attention_gpu.py"
    "::test_qwen38_int8_batch_attention_matches_cpu_and_independent_c1"
)

# Substring -> launch kind. The batch names are matched first: the batch
# producer's name contains "_int8_batch_kernel", which does not contain the c1
# substring "_int8_kernel", but relying on that alone would be brittle.
_KERNEL_KINDS: tuple[tuple[str, str], ...] = (
    ("decode_split_k_ctx_tensor_gqa_int8_batch_kernel", "batch_producer"),
    ("decode_split_k_reduce_gate_batch_strided_kernel", "batch_reducer"),
    ("decode_split_k_ctx_tensor_gqa_int8_kernel", "c1_producer"),
    ("decode_split_k_reduce_gate_kernel", "c1_reducer"),
)
_KIND_ORDER = ("batch_producer", "batch_reducer", "c1_producer", "c1_reducer")


def classify_kernel(kernel_name: str) -> str | None:
    """Return the ownership kind for a traced kernel name, or None to ignore.

    Ignoring is the common case: the trace also contains runtime copies and
    unrelated dispatches that are not IKV-C2 evidence.
    """

    for needle, kind in _KERNEL_KINDS:
        if needle in kernel_name:
            return kind
    return None


def read_launches(csv_path: Path) -> list[dict[str, Any]]:
    """Reduce a rocprofv3 ``--kernel-trace`` CSV to the IKV-C2 dispatches."""

    launches: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("Kind") or "").strip() != "KERNEL_DISPATCH":
                continue
            name = row.get("Kernel_Name") or ""
            kind = classify_kernel(name)
            if kind is None:
                continue
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            if end < start:
                raise ValueError(
                    f"kernel {name!r} has an inverted timestamp window in {csv_path}"
                )
            launches.append(
                {
                    "kind": kind,
                    "kernel_name": name,
                    "duration_ns": end - start,
                    "grid": [
                        str(row["Grid_Size_X"]),
                        str(row["Grid_Size_Y"]),
                        str(row["Grid_Size_Z"]),
                    ],
                    "workgroup": [
                        str(row["Workgroup_Size_X"]),
                        str(row["Workgroup_Size_Y"]),
                        str(row["Workgroup_Size_Z"]),
                    ],
                    "vgpr_count": int(row["VGPR_Count"]),
                }
            )
    if not launches:
        raise ValueError(f"no IKV-C2 dispatches found in {csv_path}")
    return launches


def summarize_ownership(launches: Sequence[Mapping[str, Any]], rows: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive the ownership and kernel-time blocks from the traced dispatches."""

    if rows <= 1:
        raise ValueError("the ownership trace is only meaningful above c1")
    counts = {kind: 0 for kind in _KIND_ORDER}
    totals = {kind: 0 for kind in _KIND_ORDER}
    for launch in launches:
        kind = str(launch["kind"])
        if kind not in counts:
            raise ValueError(f"unexpected launch kind {kind!r}")
        counts[kind] += 1
        totals[kind] += int(launch["duration_ns"])

    batch_total = totals["batch_producer"] + totals["batch_reducer"]
    serial_total = totals["c1_producer"] + totals["c1_reducer"]
    if batch_total <= 0 or serial_total <= 0:
        raise ValueError(
            "the trace must contain both the packed path and the independent c1 leaf path"
        )
    if counts["batch_producer"] != 1 or counts["batch_reducer"] != 1:
        raise ValueError(
            "the packed producer and reducer must each launch exactly once for all rows; "
            f"observed producer={counts['batch_producer']} reducer={counts['batch_reducer']}"
        )
    if counts["c1_producer"] != rows or counts["c1_reducer"] != rows:
        raise ValueError(
            f"the c1 leaf must launch once per row ({rows}); observed "
            f"producer={counts['c1_producer']} reducer={counts['c1_reducer']}"
        )

    ownership = {
        "batch_launch_count": counts["batch_producer"] + counts["batch_reducer"],
        "c1_leaf_launch_count": counts["c1_producer"] + counts["c1_reducer"],
        "launches_per_kind": {kind: counts[kind] for kind in _KIND_ORDER},
        "single_batch_launch_covers_all_rows": True,
        "per_row_serial_c1_launches": counts["c1_producer"],
    }
    kernel_time_ns = {
        "batch_path_total": batch_total,
        "serial_c1_path_total": serial_total,
        "ratio_serial_over_batch": round(serial_total / batch_total, 3),
        "batch_producer": totals["batch_producer"],
        "batch_reducer": totals["batch_reducer"],
        "c1_producer": totals["c1_producer"],
        "c1_reducer": totals["c1_reducer"],
    }
    return ownership, kernel_time_ns


def _quant_key(info: Any) -> str:
    name = str(getattr(info, "file_type_name", "") or "").strip().lower()
    if name.startswith("mostly_"):
        name = name[len("mostly_") :]
    if not name:
        raise ValueError("GGUF metadata does not expose file_type_name")
    return f"gguf_{name}"


def capability_snapshot(model: Path, backend: str) -> dict[str, Any]:
    """Resolve the live INT8-KV capability so the artifact cannot go stale."""

    info = scan_gguf(model)
    identity = model_artifact_identity(model)
    if not identity.content_verified:
        raise ValueError(f"model identity unavailable: {identity.error}")
    key = KVCapabilityKey(
        artifact_sha256=identity.sha256,
        artifact_size_bytes=identity.size_bytes,
        backend=backend,
        target_arch=hip_target_arch_for_backend(backend),
        weight_quant=_quant_key(info),
        kv_storage="int8_per_token_head",
        storage_layout="uniform",
        scale_dtype="fp32",
        scale_granularity="per_token_head",
    )
    resolution = Qwen35GGUFModel().resolve_kv_capability(key=key, artifact=identity)
    evidence = resolution.as_dict().get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("qualified capability has no evidence payload")
    return {
        "capability_id": str(resolution.as_dict().get("capability_id", "")),
        "admitted_max_direct_rows": int(evidence.get("max_direct_rows", 0)),
        "decode_batch_variant": str(evidence.get("decode_batch_variant", "")),
    }


def _single_trace_csv(root: Path) -> Path:
    matches = sorted(root.rglob("*kernel_trace.csv"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one *kernel_trace.csv under {root}, found {len(matches)}"
        )
    return matches[0]


def _trace_environment(compiler_version_file: Path, backend: str) -> dict[str, str]:
    compiler_version = compiler_version_file.read_text(encoding="utf-8").strip()
    if not compiler_version:
        raise ValueError(f"compiler version file is empty: {compiler_version_file}")
    environment = dict(os.environ)
    environment.pop("ROCR_VISIBLE_DEVICES", None)
    environment.setdefault("HIP_VISIBLE_DEVICES", "0")
    environment["HIPENGINE_HIP_ARCH"] = hip_target_arch_for_backend(backend)
    environment[COMPILER_VERSION_FILE_ENV] = str(compiler_version_file)
    # The profiled process must not compile: a hipcc spawn inside the profiler
    # corrupts the trace rather than merely slowing it down.
    environment[REQUIRE_CACHED_BUILD_ENV] = "1"
    return environment


_CACHE_RELEVANT_ENV_KEYS = (
    "HIP_VISIBLE_DEVICES",
    "HIPENGINE_HIP_ARCH",
    COMPILER_VERSION_FILE_ENV,
    REQUIRE_CACHED_BUILD_ENV,
)


def _cache_relevant_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """The subset that decides whether the trace was genuinely cache-only.

    Recording this makes the cache-only claim checkable instead of asserted: a
    reader can see which switch was set and that it is the one the build module
    reads.
    """

    return {key: environment[key] for key in _CACHE_RELEVANT_ENV_KEYS if key in environment}


def _rocprofv3_version() -> str:
    executable = shutil.which("rocprofv3")
    if executable is None:
        return "unavailable"
    completed = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False
    )
    text = (completed.stdout or completed.stderr or "").strip()
    if not text:
        return "unavailable"
    first = text.splitlines()[0].strip()
    prefix = "version:"
    if first.lower().startswith(prefix):
        return first[len(prefix) :].strip() or "unavailable"
    return first


def _observed_device() -> str:
    executable = shutil.which("rocm-smi")
    if executable is None:
        return "unavailable"
    completed = subprocess.run(
        [executable, "--showproductname", "--json"], capture_output=True, text=True, check=False
    )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return "unavailable"
    for card in payload.values():
        if isinstance(card, Mapping):
            for key in ("Card Series", "Card Model", "Card SKU"):
                value = card.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return "unavailable"


def _run_gate(case: str, environment: Mapping[str, str]) -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"{GATE_TEST}[{case}]"],
        cwd=str(REPO_ROOT),
        env=dict(environment),
        check=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    case = str(args.case)
    if case not in GATE_CASES:
        raise ValueError(f"unknown gate case {case!r}; expected one of {sorted(GATE_CASES)}")
    rows, live_counts = GATE_CASES[case]
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    backend = str(args.backend)
    environment = _trace_environment(args.compiler_version_file.expanduser().resolve(), backend)

    trace_root = args.trace_dir.expanduser().resolve() if args.trace_dir else DEFAULT_TRACE_ROOT
    if args.trace_dir is None:
        if shutil.which("rocprofv3") is None:
            raise ValueError("rocprofv3 is not on PATH; pass --trace-dir to parse an existing trace")
        if trace_root.exists():
            shutil.rmtree(trace_root)
        trace_root.mkdir(parents=True, exist_ok=True)
        if args.warmup:
            # Warm the JIT cache outside the profiler so the traced run is
            # cache-only.
            _run_gate(case, environment)
        command = [
            "rocprofv3",
            "--kernel-trace",
            "--output-format",
            "csv",
            "-d",
            str(trace_root),
            "--",
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"{GATE_TEST}[{case}]",
        ]
        completed = subprocess.run(
            command, cwd=str(REPO_ROOT), env=dict(environment), check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"profiled gate run failed with exit {completed.returncode}; "
                "the artifact is not written"
            )
    else:
        command = None
        if not trace_root.is_dir():
            raise ValueError(f"trace directory does not exist: {trace_root}")

    launches = read_launches(_single_trace_csv(trace_root))
    ownership, kernel_time_ns = summarize_ownership(launches, rows)
    if command is not None:
        trace_command = " ".join(shlex.quote(part) for part in command)
        reduction = "profiled rocprofv3 run"
        trace_environment = _cache_relevant_environment(environment)
    else:
        trace_command = str(args.trace_command or f"unavailable (parsed from {trace_root})")
        reduction = f"parse-only {trace_root}"
        trace_environment = {}
    notes = [
        "kernel_time_ns is single-shot kernel duration from a correctness gate, not a "
        "benchmark; it is launch-count and sub-window evidence.",
        f"The batch producer and reducer each launch once for all {rows} rows; the c1 leaf "
        "producer and reducer each launch once per row.",
    ]
    if trace_environment:
        notes.insert(
            0,
            "Cache-only trace: the .so was built outside the profiler and "
            f"{REQUIRE_CACHED_BUILD_ENV}=1, with the compiler version pinned by "
            f"{COMPILER_VERSION_FILE_ENV}, prevented hipcc in the profiled process. "
            "Both are recorded in trace_environment.",
        )
    else:
        notes.insert(
            0,
            "Reduced from an existing trace; the cache-only property belongs to that "
            "run, which this parse-only path cannot observe, so no cache-only claim is "
            "made here.",
        )
    return {
        "schema": 1,
        "kind": ARTIFACT_KIND,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "host": {
            "observed_device": _observed_device(),
            "target_arch": hip_target_arch_for_backend(backend),
            "hip_visible_devices": environment.get("HIP_VISIBLE_DEVICES", ""),
            "rocprofv3_version": _rocprofv3_version(),
        },
        "capability": capability_snapshot(model, backend),
        "workload": {
            "gate": f"{GATE_TEST}[{case}]",
            "rows": rows,
            "live_counts": list(live_counts),
        },
        "ownership": ownership,
        "kernel_time_ns": kernel_time_ns,
        "launches": launches,
        "command": trace_command,
        "reduction": reduction,
        "trace_environment": trace_environment,
        "notes": notes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--case", default="c4-ragged", choices=sorted(GATE_CASES))
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="parse an existing rocprofv3 output directory instead of tracing",
    )
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=DEFAULT_COMPILER_VERSION_FILE,
    )
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="skip the unprofiled cache-warming gate run",
    )
    parser.set_defaults(warmup=True)
    parser.add_argument(
        "--trace-command",
        default=None,
        help=(
            "record the command that produced an existing --trace-dir, so a "
            "parse-only regeneration does not lose the trace's provenance"
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = run(args)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = json.dumps(artifact, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.expanduser().resolve().write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
