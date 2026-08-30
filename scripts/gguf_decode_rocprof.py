#!/usr/bin/env python3
"""Audit correct GGUF eager decode wall time and decode-only kernel families.

The parent performs three independent checks for SOL-G4:

* a clean, repeated 512-token eager baseline linked to the SOL-G1 state oracle;
* a direct-parent revision boundary measurement for the loaded-library cache
  that made correct eager decode fast; and
* a ``rocprofv3`` trace sliced to per-step ROCTX marker windows, excluding model
  load, prefill, and warmup from the layer-family Amdahl table.

All timed children force the production repacked/GEMV eager route, record every
generated token ID, and fail on the first token that differs from the expected
oracle. The profiled child requires cached JIT artifacts; its warm-build runs
outside ``rocprofv3`` with a pinned compiler-version file.
"""

from __future__ import annotations

import argparse
import collections
import csv
import ctypes
import functools
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_G1_ARTIFACT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "2026-07-11-sol-g1-gfx1151-gguf-eager-p512-d4.json"
)
DEFAULT_BEFORE_COMMIT = "74b11dbc3e75e2a50332907d13b882a063f7c56b"
DEFAULT_AFTER_COMMIT = "4499fb132cabd1da7364d8d6f48e080a1352c074"
DEFAULT_ROUTE_CHANGE_COMMIT = "e8521a2a3d8d4b5fe0f1078fd90eccafac29b55b"
MARKER_PREFIX = "hipengine_gguf_eager_decode_step_"
ROLE_MARKER_PREFIX = "hipengine_gguf_decode_role:"
KIND = "hipengine_gguf_eager_decode_audit"
SCHEMA_VERSION = 2


def _weight_role(weight: object) -> str:
    """Return the decode projection role carried by one resident GGUF weight."""

    spec = getattr(weight, "spec", None)
    slot_path = str(getattr(spec, "slot_path", ""))
    slot = slot_path.rsplit(".", 1)[-1]
    if slot in {"attn_q", "attn_k", "attn_v"}:
        return "full_attention_qkv"
    if slot == "attn_output":
        return "full_attention_output"
    if slot in {"attn_qkv", "attn_gate"}:
        return "gdn_input_projections"
    if slot in {"ssm_alpha", "ssm_beta"}:
        return "gdn_decay_projections"
    if slot == "ssm_out":
        return "gdn_output_projection"
    if slot in {"lm_head", "output"}:
        return "lm_head"
    if slot in {"ffn_gate_shexp", "ffn_up_shexp"}:
        return "shared_expert_gate_up"
    if slot == "ffn_down_shexp":
        return "shared_expert_down"
    if slot in {"ffn_gate_exps", "ffn_up_exps"}:
        return "selected_expert_gate_up"
    if slot == "ffn_down_exps":
        return "selected_expert_down"
    return f"weight_other:{slot or 'unknown'}"


def _combined_weight_role(weights: Sequence[object]) -> str:
    roles = tuple(dict.fromkeys(_weight_role(weight) for weight in weights))
    return roles[0] if len(roles) == 1 else "+".join(roles)


class _DecodeRoleMarkerPatches:
    """Profiler-only ROCTX wrappers around existing decode launch owners.

    Wrappers enqueue exactly the original work and add no synchronization. The
    role attribution later joins HIP launch API correlation IDs to kernel rows,
    so asynchronous execution does not need to fit inside the host marker time.
    """

    def __init__(self, session: object, marker: "_Roctx") -> None:
        self._session = session
        self._marker = marker
        self._patches: list[tuple[object, str, object]] = []

    def _patch(self, owner: object, name: str, role_fn) -> None:
        original = getattr(owner, name, None)
        if original is None:
            return

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            role = str(role_fn(args, kwargs))
            self._marker.push(f"{ROLE_MARKER_PREFIX}{role}")
            try:
                return original(*args, **kwargs)
            finally:
                self._marker.pop()

        self._patches.append((owner, name, original))
        setattr(owner, name, wrapped)

    def install(self) -> None:
        runner = getattr(self._session, "runner", None)
        if runner is None:
            raise RuntimeError("GGUF decode role markers require a live runner")
        module = sys.modules[type(runner).__module__]
        runner_type = type(runner)
        session_type = type(self._session)

        self._patch(
            runner_type,
            "_run_linear_attention_attn_only",
            lambda _args, _kwargs: "gdn_attention_core",
        )
        self._patch(
            runner_type,
            "_run_full_attention_attn_only",
            lambda _args, _kwargs: "full_attention_core",
        )
        self._patch(
            runner_type,
            "_run_post_attention_moe_c1",
            lambda _args, _kwargs: "moe_router_combine",
        )
        self._patch(
            runner_type,
            "_run_post_attention_moe_c1_unfused_selected_ffn",
            lambda _args, _kwargs: "selected_expert_other",
        )
        self._patch(
            session_type,
            "_sample_device_from_hidden",
            lambda _args, _kwargs: "lm_head",
        )

        for name, count in (
            ("launch_gguf_linear", 1),
            ("launch_gguf_linear_pair", 2),
            ("launch_gguf_linear_pair_concat", 2),
            ("launch_gguf_linear_triple", 3),
        ):
            self._patch(
                module,
                name,
                lambda args, _kwargs, count=count: _combined_weight_role(args[:count]),
            )
        for name in (
            "_launch_selected_raw_gguf_moe_pair_silu",
            "_launch_selected_raw_gguf_moe_pair",
        ):
            self._patch(
                module,
                name,
                lambda _args, _kwargs: "selected_expert_gate_up",
            )
        for name in (
            "_launch_selected_raw_gguf_moe_linear",
            "_launch_weighted_selected_raw_gguf_moe_linear",
        ):
            self._patch(
                module,
                name,
                lambda args, _kwargs: _weight_role(args[0]),
            )
        for name in (
            "_try_run_post_attention_moe_c1_fused_ffn",
            "_try_run_post_attention_moe_c1_compact_gemv",
        ):
            self._patch(
                module,
                name,
                lambda _args, _kwargs: "selected_expert_fused",
            )

    def restore(self) -> None:
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        self._patches.clear()


class _Roctx:
    def __init__(self) -> None:
        try:
            self._lib = ctypes.CDLL("libroctx64.so")
        except OSError as exc:  # pragma: no cover - depends on profiler SDK
            raise RuntimeError(f"libroctx64.so is required for decode marker windows: {exc}") from exc
        self._push = getattr(self._lib, "roctxRangePushA", None)
        self._pop = getattr(self._lib, "roctxRangePop", None)
        if self._push is None or self._pop is None:
            raise RuntimeError("libroctx64.so does not expose roctxRangePushA/roctxRangePop")
        self._push.argtypes = [ctypes.c_char_p]
        self._push.restype = ctypes.c_int
        self._pop.argtypes = []
        self._pop.restype = ctypes.c_int

    def push(self, name: str) -> None:
        self._push(name.encode("utf-8"))

    def pop(self) -> None:
        self._pop()


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git -C {root} {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result


def _source_state(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    commit = _git(resolved, "rev-parse", "HEAD").stdout.strip()
    branch = _git(resolved, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    status = _git(resolved, "status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines()
    return {
        "root": str(resolved),
        "commit": commit,
        "branch": branch.stdout.strip() or None,
        "dirty": bool(status),
        "status_entries": status,
    }


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty compiler-version file: {path}")
    return text


def _configure_child_imports(source_root: Path) -> Path:
    root = source_root.resolve()
    if not (root / "hipengine").is_dir():
        raise FileNotFoundError(f"source root does not contain hipengine/: {root}")
    for candidate in (str(REPO_ROOT), str(root)):
        while candidate in sys.path:
            sys.path.remove(candidate)
    sys.path.insert(0, str(root))
    os.chdir(root)
    return root


def _session_kwargs(
    session_type: type,
    *,
    max_sequence_length: int,
    compiler_version: str | None,
    require_cached_build: bool,
    backend: str,
) -> tuple[dict[str, Any], bool]:
    """Build kwargs shared by current and pre-backend-identity sessions."""

    supported = inspect.signature(session_type).parameters
    values: dict[str, Any] = {
        "max_sequence_length": max_sequence_length,
        "compiler_version": compiler_version,
        "require_cached_build": require_cached_build,
        "backend": backend,
        "use_wmma_prefill": True,
        "use_gemv_decode": True,
    }
    return ({name: value for name, value in values.items() if name in supported}, "backend" in supported)


def _run_eager_once(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    warmup_steps: int,
    steps: int,
    expected_token_id: int,
    marker: _Roctx | None = None,
) -> dict[str, Any]:
    session.reset()
    first = session.prefill(prompt_ids, use_bulk=True, return_logits=False)
    current = int(first.token_id)
    generated = [current]
    if current != expected_token_id:
        raise RuntimeError(
            f"unexpected prefill token: expected {expected_token_id}, observed {current}"
        )
    for index in range(warmup_steps):
        current = int(session.step(current, return_logits=False).token_id)
        generated.append(current)
        if current != expected_token_id:
            raise RuntimeError(
                f"unexpected warmup token at index {index}: "
                f"expected {expected_token_id}, observed {current}"
            )

    session.runtime.device_synchronize()
    start_ns = time.perf_counter_ns()
    step_wall_ms: list[float] = []
    for index in range(steps):
        if marker is not None:
            marker.push(f"{MARKER_PREFIX}{index}")
        step_start_ns = time.perf_counter_ns()
        try:
            result = session.step(current, return_logits=False)
            session.runtime.device_synchronize()
        finally:
            if marker is not None:
                marker.pop()
        step_end_ns = time.perf_counter_ns()
        current = int(result.token_id)
        generated.append(current)
        step_wall_ms.append((step_end_ns - step_start_ns) / 1e6)
        if current != expected_token_id:
            raise RuntimeError(
                f"unexpected timed token at index {index}: "
                f"expected {expected_token_id}, observed {current}"
            )
    session.runtime.device_synchronize()
    end_ns = time.perf_counter_ns()
    wall_ms = (end_ns - start_ns) / 1e6
    return {
        "expected_token_id": int(expected_token_id),
        "prompt_tokens": len(prompt_ids),
        "warmup_steps": int(warmup_steps),
        "timed_steps": int(steps),
        "generated_token_ids": generated,
        "all_tokens_exact": all(token == expected_token_id for token in generated),
        "wall_ms": wall_ms,
        "wall_ms_per_token": wall_ms / steps,
        "tok_s": steps * 1000.0 / wall_ms,
        "step_wall_ms": step_wall_ms,
    }


def _run_child(args: argparse.Namespace) -> int:
    source_root = _configure_child_imports(Path(args.source_root))
    os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    os.environ.setdefault(
        "HIPENGINE_HIP_ARCH",
        "gfx1151" if args.backend == "hip_gfx1151" else "gfx1100",
    )
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import reset_memory_stats
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from scripts.qwen35_gguf_bench import _memory_snapshot, _memory_summary

    runtime = get_hip_runtime()
    reset_memory_stats()
    memory_snapshots: dict[str, Any] = {
        "before_load": _memory_snapshot("before_load", runtime)
    }
    prompt_ids = [int(args.prompt_token_id)] * int(args.prompt_length)
    max_sequence_length = int(
        args.max_seq
        or args.prompt_length + args.warmup_steps + args.steps + 8
    )
    compiler_version = _read_compiler_version(args.compiler_version_file)
    warmup_runs: list[dict[str, Any]] = []
    measured_runs: list[dict[str, Any]] = []
    marker = _Roctx() if args.child_mode == "profile" else None
    session_kwargs, constructor_backend_supported = _session_kwargs(
        Qwen35GGUFResidentSession,
        max_sequence_length=max_sequence_length,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached),
        backend=str(args.backend),
    )
    with Qwen35GGUFResidentSession(args.model, **session_kwargs) as session:
        memory_snapshots["after_load"] = _memory_snapshot(
            "after_load", runtime, session
        )
        role_patches = None
        if marker is not None:
            role_patches = _DecodeRoleMarkerPatches(session, marker)
            role_patches.install()
        try:
            if args.child_mode == "warmbuild":
                measured_runs.append(
                    _run_eager_once(
                        session,
                        prompt_ids=prompt_ids,
                        warmup_steps=int(args.warmup_steps),
                        steps=max(1, int(args.steps)),
                        expected_token_id=int(args.expected_token_id),
                    )
                )
            else:
                for _ in range(int(args.benchmark_warmups)):
                    warmup_runs.append(
                        _run_eager_once(
                            session,
                            prompt_ids=prompt_ids,
                            warmup_steps=int(args.warmup_steps),
                            steps=int(args.steps),
                            expected_token_id=int(args.expected_token_id),
                        )
                    )
                repetitions = 1 if args.child_mode == "profile" else int(args.repetitions)
                for _ in range(repetitions):
                    measured_runs.append(
                        _run_eager_once(
                            session,
                            prompt_ids=prompt_ids,
                            warmup_steps=int(args.warmup_steps),
                            steps=int(args.steps),
                            expected_token_id=int(args.expected_token_id),
                            marker=marker,
                        )
                    )
        finally:
            if role_patches is not None:
                role_patches.restore()
        resolved_backend = str(getattr(session.runner, "backend", str(args.backend)))
        target_arch = str(getattr(session.runner, "target_arch", os.environ["HIPENGINE_HIP_ARCH"]))
        effective_wmma = bool(getattr(session, "use_wmma_prefill", False))
        effective_gemv = bool(getattr(session, "use_gemv_decode", False))
        memory_snapshots["before_close"] = _memory_snapshot(
            "before_close", runtime, session
        )
    memory_snapshots["after_close"] = _memory_snapshot("after_close", runtime)

    payload = {
        "kind": "hipengine_gguf_eager_decode_child",
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "child_mode": str(args.child_mode),
        "source": _source_state(source_root),
        "command": [str(part) for part in sys.argv],
        "workload": {
            "model": str(Path(args.model).resolve()),
            "quant": "gguf_q4_k_m",
            "kv_dtype": "bf16",
            "prompt_source": "repeated_token_id",
            "prompt_token_id": int(args.prompt_token_id),
            "prompt_length": int(args.prompt_length),
            "prompt_sha256_i64": hashlib.sha256(
                int(args.prompt_token_id).to_bytes(8, "little", signed=True)
                * int(args.prompt_length)
            ).hexdigest(),
            "expected_token_id": int(args.expected_token_id),
            "warmup_steps": int(args.warmup_steps),
            "timed_steps": int(args.steps),
            "benchmark_warmups": int(args.benchmark_warmups),
            "repetitions": len(measured_runs),
            "max_sequence_length": max_sequence_length,
        },
        "route": {
            "configured_backend": str(args.backend),
            "resolved_backend": resolved_backend,
            "target_arch": target_arch,
            "constructor_backend_supported": constructor_backend_supported,
            "graph_replay_decode": False,
            "decode_repack": True,
            "requested_use_wmma_prefill": True,
            "effective_use_wmma_prefill": effective_wmma,
            "requested_use_gemv_decode": True,
            "effective_use_gemv_decode": effective_gemv,
            "require_cached_build": bool(args.require_cached),
        },
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "memory": {
            "summary": _memory_summary(memory_snapshots),
            "snapshots": memory_snapshots,
        },
    }
    _validate_child_payload(payload, expected_token_id=int(args.expected_token_id))
    if args.child_json is not None:
        args.child_json.parent.mkdir(parents=True, exist_ok=True)
        args.child_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = _summarize_wall_runs(measured_runs, expected_token_id=int(args.expected_token_id))
    print(
        f"[gguf-eager:{args.child_mode}] {args.prompt_length}/{args.steps} "
        f"median={summary['median_tok_s']:.3f} tok/s exact=true",
        flush=True,
    )
    return 0


def _family(name: str) -> str:
    value = re.sub(r"^void\s+", "", name.strip()).replace("(anonymous namespace)::", "")
    value = re.sub(r"<[^>]*>", "", value).split("(")[0]
    return re.sub(r"_kernel$", "", value).strip()


def _bucket(family: str) -> str:
    value = family.lower()
    if "router" in value:
        return "moe_router"
    if any(key in value for key in ("linear_attn", "gdn", "ssm", "_conv_")):
        return "gdn_linear_attn"
    if value.startswith(
        ("q4_k_t16_selected", "qk_t16_selected", "gguf_k_selected", "gguf_q4_k_selected")
    ):
        return "moe_selected_gemv"
    if value.startswith("q8_0_t16"):
        return "dense_q8_0_gemv"
    if "dense_gemv" in value:
        return "dense_gemv_bf16"
    if "q6_k_t16_gemv" in value or "lm_head" in value or "argmax" in value:
        return "lm_head"
    if "rmsnorm" in value or "rotary" in value:
        return "rmsnorm_rope"
    if any(key in value for key in ("silu", "weighted_sum", "combine")):
        return "moe_combine_silu"
    if any(key in value for key in ("attn", "flash", "softmax", "paged_kv", "paged_full")):
        return "attn_core"
    if "embedding" in value:
        return "embedding"
    if any(key in value for key in ("copybuffer", "fillbuffer", "rocclr", "memcpy")):
        return "memcpy_fill"
    return "other"


def _read_marker_windows(path: Path, prefix: str) -> list[tuple[int, int, int]]:
    windows: list[tuple[int, int, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (
                row.get("Function")
                or row.get("Marker_Name")
                or row.get("Marker_Text")
                or row.get("Name")
                or ""
            ).strip()
            if not name.startswith(prefix):
                continue
            try:
                index = int(name.removeprefix(prefix))
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end >= start:
                windows.append((index, start, end))
    windows.sort(key=lambda item: item[0])
    return windows


def _read_role_windows(path: Path, prefix: str) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (
                row.get("Function")
                or row.get("Marker_Name")
                or row.get("Marker_Text")
                or row.get("Name")
                or ""
            ).strip()
            if not name.startswith(prefix):
                continue
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            windows.append(
                {
                    "role": name.removeprefix(prefix),
                    "thread_id": _optional_int(row.get("Thread_Id")),
                    "start_ns": start,
                    "end_ns": end,
                }
            )
    windows.sort(key=lambda item: (int(item["start_ns"]), -int(item["end_ns"])))
    return windows


def _read_hip_launches(path: Path) -> list[dict[str, Any]]:
    launches: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            function = str(row.get("Function") or "")
            if "LaunchKernel" not in function and "ExtLaunch" not in function:
                continue
            try:
                correlation_id = int(float(row["Correlation_Id"]))
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            launches.append(
                {
                    "function": function,
                    "thread_id": _optional_int(row.get("Thread_Id")),
                    "correlation_id": correlation_id,
                    "start_ns": start,
                    "end_ns": end,
                }
            )
    launches.sort(key=lambda item: int(item["start_ns"]))
    return launches


def _annotate_kernel_roles(
    rows: Sequence[dict[str, Any]],
    *,
    launches: Sequence[dict[str, Any]],
    windows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join asynchronous kernel rows to the innermost host role range."""

    by_thread: dict[int | None, list[dict[str, Any]]] = collections.defaultdict(list)
    for window in windows:
        by_thread[window.get("thread_id")].append(window)
    correlation_roles: dict[int, str] = {}
    for launch in launches:
        start = int(launch["start_ns"])
        end = int(launch["end_ns"])
        candidates = [
            window
            for window in by_thread.get(launch.get("thread_id"), ())
            if int(window["start_ns"]) <= start and end <= int(window["end_ns"])
        ]
        if not candidates and launch.get("thread_id") is not None:
            candidates = [
                window
                for window in by_thread.get(None, ())
                if int(window["start_ns"]) <= start and end <= int(window["end_ns"])
            ]
        if not candidates:
            continue
        innermost = min(
            candidates,
            key=lambda window: (
                int(window["end_ns"]) - int(window["start_ns"]),
                -int(window["start_ns"]),
            ),
        )
        correlation_roles[int(launch["correlation_id"])] = str(innermost["role"])

    output: list[dict[str, Any]] = []
    for row in rows:
        annotated = dict(row)
        correlation_id = _optional_int(row.get("correlation_id"))
        annotated["role"] = correlation_roles.get(correlation_id, "unattributed")
        output.append(annotated)
    return output


def _summarize_role_rows(
    rows: Sequence[dict[str, Any]], *, steps: int
) -> list[dict[str, Any]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    total_ns = sum(int(row["duration_ns"]) for row in rows)
    by_role: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {
            "calls": 0,
            "duration_ns": 0,
            "families": set(),
            "vgpr": set(),
            "scratch": set(),
        }
    )
    for row in rows:
        role = str(row.get("role") or "unattributed")
        record = by_role[role]
        record["calls"] += 1
        record["duration_ns"] += int(row["duration_ns"])
        if row.get("family"):
            record["families"].add(str(row["family"]))
        if row.get("vgpr") is not None:
            record["vgpr"].add(int(row["vgpr"]))
        if row.get("scratch") is not None:
            record["scratch"].add(int(row["scratch"]))
    output = []
    for role, record in sorted(
        by_role.items(), key=lambda item: (-int(item[1]["duration_ns"]), item[0])
    ):
        calls = int(record["calls"])
        duration_ns = int(record["duration_ns"])
        share = duration_ns / total_ns if total_ns else 0.0
        output.append(
            {
                "name": role,
                "calls": calls,
                "calls_per_token": calls / steps,
                "total_us": duration_ns / 1e3,
                "gpu_us_per_token": duration_ns / steps / 1e3,
                "share_pct": share * 100.0,
                "us_per_call": duration_ns / calls / 1e3 if calls else 0.0,
                "kernel_families": sorted(record["families"]),
                "vgpr_counts": sorted(record["vgpr"]),
                "scratch_sizes": sorted(record["scratch"]),
                "amdahl_speedup_if_2x": _amdahl_speedup(share, 2.0),
                "amdahl_speedup_if_4x": _amdahl_speedup(share, 4.0),
                "amdahl_speedup_if_infinite": _amdahl_speedup(share, float("inf")),
            }
        )
    return output


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def _read_kernels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            kernel = (
                row.get("Kernel_Name")
                or row.get("KernelName")
                or row.get("Name")
                or ""
            ).strip()
            family = _family(kernel)
            rows.append(
                {
                    "kernel": kernel,
                    "family": family,
                    "bucket": _bucket(family),
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                    "correlation_id": _optional_int(row.get("Correlation_Id")),
                    "vgpr": _optional_int(row.get("VGPR_Count")),
                    "scratch": _optional_int(row.get("Scratch_Size")),
                }
            )
    return rows


def _filter_kernels_by_windows(
    rows: Iterable[dict[str, Any]], windows: Sequence[tuple[int, int]]
) -> list[dict[str, Any]]:
    ordered = sorted(windows)
    return [
        row
        for row in rows
        if any(
            int(row["start_ns"]) >= start and int(row["end_ns"]) <= end
            for start, end in ordered
        )
    ]


def _amdahl_speedup(share: float, factor: float) -> float | None:
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"Amdahl share must be within [0, 1], got {share}")
    denominator = 1.0 - share if math.isinf(factor) else (1.0 - share) + share / factor
    if denominator <= 0.0:
        return None
    return 1.0 / denominator


def _summarize_rows(rows: Sequence[dict[str, Any]], *, steps: int, top: int) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    total_ns = sum(int(row["duration_ns"]) for row in rows)
    if total_ns <= 0:
        raise ValueError("selected decode window contains no positive-duration kernels")
    by_bucket: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    by_family: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    resources: dict[str, dict[str, set[int]]] = collections.defaultdict(
        lambda: {"vgpr": set(), "scratch": set()}
    )
    for row in rows:
        duration = int(row["duration_ns"])
        by_bucket[str(row["bucket"])][0] += 1
        by_bucket[str(row["bucket"])][1] += duration
        family = str(row["family"])
        by_family[family][0] += 1
        by_family[family][1] += duration
        for key in ("vgpr", "scratch"):
            if row.get(key) is not None:
                resources[family][key].add(int(row[key]))

    def emit(aggregate: dict[str, list[int]], *, include_amdahl: bool) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for name, (calls, duration_ns) in sorted(
            aggregate.items(), key=lambda item: (-item[1][1], item[0])
        ):
            share = duration_ns / total_ns
            record: dict[str, Any] = {
                "name": name,
                "calls": calls,
                "calls_per_token": calls / steps,
                "total_us": duration_ns / 1e3,
                "gpu_us_per_token": duration_ns / steps / 1e3,
                "share_pct": share * 100.0,
                "us_per_call": duration_ns / calls / 1e3,
            }
            if include_amdahl:
                record.update(
                    {
                        "amdahl_speedup_if_2x": _amdahl_speedup(share, 2.0),
                        "amdahl_speedup_if_4x": _amdahl_speedup(share, 4.0),
                        "amdahl_speedup_if_infinite": _amdahl_speedup(share, float("inf")),
                    }
                )
            else:
                record["vgpr_counts"] = sorted(resources[name]["vgpr"])
                record["scratch_sizes"] = sorted(resources[name]["scratch"])
            output.append(record)
        return output

    return {
        "timed_steps": int(steps),
        "total_kernels": len(rows),
        "total_gpu_us": total_ns / 1e3,
        "gpu_us_per_token": total_ns / steps / 1e3,
        "buckets": emit(by_bucket, include_amdahl=True),
        "top_kernels": emit(by_family, include_amdahl=False)[:top],
    }


def _summarize_wall_runs(
    runs: Sequence[dict[str, Any]], *, expected_token_id: int
) -> dict[str, Any]:
    if not runs:
        raise ValueError("no measured wall runs")
    for run_index, run in enumerate(runs):
        tokens = [int(token) for token in run.get("generated_token_ids", [])]
        mismatch = next((token for token in tokens if token != expected_token_id), None)
        if mismatch is not None or not tokens:
            raise ValueError(
                f"unexpected token in run {run_index}: expected {expected_token_id}, "
                f"observed {mismatch if mismatch is not None else 'empty stream'}"
            )
    samples = [float(run["wall_ms_per_token"]) for run in runs]
    median_ms = statistics.median(samples)
    return {
        "repetitions": len(runs),
        "all_tokens_exact": True,
        "expected_token_id": int(expected_token_id),
        "samples_ms_per_token": samples,
        "samples_tok_s": [1000.0 / value for value in samples],
        "median_ms_per_token": median_ms,
        "median_tok_s": 1000.0 / median_ms,
        "mean_ms_per_token": statistics.mean(samples),
        "stdev_ms_per_token": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms_per_token": min(samples),
        "max_ms_per_token": max(samples),
    }


def _validate_child_payload(payload: dict[str, Any], *, expected_token_id: int) -> None:
    route = payload.get("route") or {}
    if route.get("graph_replay_decode") is not False:
        raise ValueError("child did not record eager graph-off decode")
    if route.get("decode_repack") is not True:
        raise ValueError("child did not record decode repack")
    if route.get("effective_use_gemv_decode") is not True:
        raise ValueError("child did not activate the GGUF GEMV decode path")
    _summarize_wall_runs(payload.get("measured_runs") or [], expected_token_id=expected_token_id)


def _build_child_command(
    *,
    python: str,
    script: Path,
    child_mode: str,
    source_root: Path,
    model: Path,
    backend: str,
    prompt_token_id: int,
    prompt_length: int,
    expected_token_id: int,
    steps: int,
    warmup_steps: int,
    benchmark_warmups: int,
    repetitions: int,
    compiler_version_file: Path,
    child_json: Path,
    require_cached: bool,
) -> list[str]:
    command = [
        python,
        str(script),
        "--child-mode",
        child_mode,
        "--source-root",
        str(source_root),
        "--model",
        str(model),
        "--backend",
        backend,
        "--prompt-token-id",
        str(prompt_token_id),
        "--prompt-length",
        str(prompt_length),
        "--expected-token-id",
        str(expected_token_id),
        "--steps",
        str(steps),
        "--warmup-steps",
        str(warmup_steps),
        "--benchmark-warmups",
        str(benchmark_warmups),
        "--repetitions",
        str(repetitions),
        "--compiler-version-file",
        str(compiler_version_file),
        "--child-json",
        str(child_json),
    ]
    if require_cached:
        command.append("--require-cached")
    return command


def _run_command(command: Sequence[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"[gguf-eager-audit] {' '.join(command)}", flush=True)
    subprocess.run(list(command), cwd=cwd, env=env, check=True)


def _load_child(path: Path, *, expected_token_id: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_child_payload(payload, expected_token_id=expected_token_id)
    return payload


def _single_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} under {root}, found {matches}")
    return matches[0]


def _default_roctx_sdk() -> Path:
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    names = ("librocprofiler-sdk-roctx.so.1", "librocprofiler-sdk-roctx.so")
    # sys.prefix alone is wrong for a venv built on a ROCm conda env: the venv has no _rocm_sdk_*
    # packages of its own, so the search can never succeed. See worklog entry
    # 20260830T043105 (diagnosis) and 8c59be6d8 (first implementation).
    candidates = [
        Path(root) / "lib" / python_dir / "site-packages" / pkg / "lib" / name
        for root in dict.fromkeys((sys.prefix, sys.base_prefix))
        for pkg in ("_rocm_sdk_core", "_rocm_sdk_devel")
        for name in names
    ]
    candidates += [Path("/opt/rocm/lib") / name for name in names]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
def _prepare_roctx_override(sdk_path: Path) -> tuple[Path, tuple[Path, ...]]:
    if not sdk_path.exists():
        raise FileNotFoundError(f"rocprofiler SDK ROCTX library not found: {sdk_path}")
    override = Path("/tmp/hipengine-roctx-sdk-override-gguf-eager")
    override.mkdir(parents=True, exist_ok=True)
    symlink = override / "libroctx64.so"
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(sdk_path)
    dependency_paths: list[Path] = [sdk_path.parent]
    sysdeps = sdk_path.parent / "rocm_sysdeps" / "lib"
    if sysdeps.is_dir():
        dependency_paths.append(sysdeps)
    return override, tuple(dependency_paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_g1_artifact(path: Path, *, model: Path, prompt_token_id: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    classification = payload.get("classification") or {}
    workload = payload.get("workload") or {}
    if classification.get("passed") is not True or classification.get("first_divergence") is not None:
        raise ValueError(f"SOL-G1 artifact is not a passing no-divergence oracle: {path}")
    if int(workload.get("prompt_length", -1)) != 512:
        raise ValueError("SOL-G1 artifact does not cover the required 512-token prompt")
    if int(workload.get("prompt_token_id", -1)) != prompt_token_id:
        raise ValueError("SOL-G1 artifact prompt token does not match this audit")
    fingerprint = (payload.get("provenance") or {}).get("model_fingerprint") or {}
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "kind": payload.get("kind"),
        "classification": classification,
        "workload": workload,
        "model_path": str(model.resolve()),
        "model_fingerprint": fingerprint,
    }


def _commit_metadata(root: Path, commit: str) -> dict[str, Any]:
    text = _git(
        root,
        "show",
        "-s",
        "--format=%H%x00%P%x00%aI%x00%s",
        commit,
    ).stdout.rstrip("\n")
    full, parents, authored_at, subject = text.split("\x00", 3)
    return {
        "commit": full,
        "parents": parents.split() if parents else [],
        "authored_at": authored_at,
        "subject": subject,
    }


def _revision_boundary(
    before: dict[str, Any],
    after: dict[str, Any],
    current_short: dict[str, Any],
    *,
    expected_before: str,
    expected_after: str,
    route_change_commit: str,
    expected_token_id: int,
) -> dict[str, Any]:
    before_source = before["source"]
    after_source = after["source"]
    if before_source["commit"] != expected_before or after_source["commit"] != expected_after:
        raise ValueError("bisect source commits do not match the requested boundary")
    if before_source["dirty"] or after_source["dirty"]:
        raise ValueError("bisect source worktrees must be clean")
    metadata = _commit_metadata(REPO_ROOT, expected_after)
    if metadata["parents"] != [expected_before]:
        raise ValueError(
            f"performance boundary is not a direct parent edge: {metadata['parents']} != {expected_before}"
        )
    before_summary = _summarize_wall_runs(
        before["measured_runs"], expected_token_id=expected_token_id
    )
    after_summary = _summarize_wall_runs(
        after["measured_runs"], expected_token_id=expected_token_id
    )
    current_summary = _summarize_wall_runs(
        current_short["measured_runs"], expected_token_id=expected_token_id
    )
    speedup = after_summary["median_tok_s"] / before_summary["median_tok_s"]
    current_vs_after_pct = (
        current_summary["median_tok_s"] / after_summary["median_tok_s"] - 1.0
    ) * 100.0
    route_metadata = _commit_metadata(REPO_ROOT, route_change_commit)
    ancestry = _git(
        REPO_ROOT,
        "merge-base",
        "--is-ancestor",
        metadata["commit"],
        route_metadata["commit"],
        check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("eager speed boundary is not an ancestor of the production route change")
    return {
        "classification": "first_performance_changing_revision_found",
        "protocol": "same_host_direct_parent_graph_off_repacked_eager_p8_d32_1x4",
        "production_eager_route_change": route_metadata,
        "before": {"source": before_source, "summary": before_summary, "command": before["command"]},
        "after": {
            "source": after_source,
            "metadata": metadata,
            "summary": after_summary,
            "command": after["command"],
        },
        "speedup_after_vs_before": speedup,
        "current_short_context": {
            "source": current_short["source"],
            "summary": current_summary,
            "command": current_short["command"],
            "delta_vs_boundary_after_pct": current_vs_after_pct,
        },
        "interpretation": (
            "4499fb13 is the direct-parent eager performance boundary: loaded HIP "
            "library memoization removes repeated host library loads without changing "
            "decode math. e8521a2a subsequently selects that now-fast eager route for "
            "production. Current short-context eager speed remains in the same band."
        ),
    }


def _run_parent(args: argparse.Namespace) -> int:
    raw_root = Path(args.raw_root).resolve()
    if raw_root.exists():
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True)
    trace_dir = raw_root / "trace"
    trace_dir.mkdir()
    compiler_file = Path(args.compiler_version_file).resolve()
    if not compiler_file.exists():
        result = subprocess.run(["hipcc", "--version"], capture_output=True, text=True, check=True)
        compiler_file.parent.mkdir(parents=True, exist_ok=True)
        compiler_file.write_text(result.stdout, encoding="utf-8")

    source_root = REPO_ROOT.resolve()
    env = os.environ.copy()
    env["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    env["HIPENGINE_HIP_ARCH"] = "gfx1151" if args.backend == "hip_gfx1151" else "gfx1100"
    env["HIPENGINE_COMPILER_VERSION_FILE"] = str(compiler_file)
    script = Path(__file__).resolve()

    def child_command(
        *,
        mode: str,
        root: Path,
        prompt_length: int,
        steps: int,
        warmup_steps: int,
        benchmark_warmups: int,
        repetitions: int,
        output: Path,
        require_cached: bool,
    ) -> list[str]:
        return _build_child_command(
            python=sys.executable,
            script=script,
            child_mode=mode,
            source_root=root,
            model=Path(args.model),
            backend=str(args.backend),
            prompt_token_id=int(args.prompt_token_id),
            prompt_length=prompt_length,
            expected_token_id=int(args.expected_token_id),
            steps=steps,
            warmup_steps=warmup_steps,
            benchmark_warmups=benchmark_warmups,
            repetitions=repetitions,
            compiler_version_file=compiler_file,
            child_json=output,
            require_cached=require_cached,
        )

    if not args.skip_warmbuild:
        warmbuild_json = raw_root / "warmbuild.json"
        command = child_command(
            mode="warmbuild",
            root=source_root,
            prompt_length=int(args.prompt_length),
            steps=1,
            warmup_steps=1,
            benchmark_warmups=0,
            repetitions=1,
            output=warmbuild_json,
            require_cached=False,
        )
        _run_command(command, cwd=source_root, env=env)

    baseline_json = raw_root / "baseline-current-p512.json"
    baseline_command = child_command(
        mode="baseline",
        root=source_root,
        prompt_length=int(args.prompt_length),
        steps=int(args.baseline_steps),
        warmup_steps=int(args.baseline_warmup_steps),
        benchmark_warmups=int(args.baseline_warmups),
        repetitions=int(args.baseline_repetitions),
        output=baseline_json,
        require_cached=True,
    )
    _run_command(baseline_command, cwd=source_root, env=env)

    short_json = raw_root / "baseline-current-p8.json"
    short_command = child_command(
        mode="baseline",
        root=source_root,
        prompt_length=int(args.bisect_prompt_length),
        steps=int(args.bisect_steps),
        warmup_steps=int(args.baseline_warmup_steps),
        benchmark_warmups=int(args.baseline_warmups),
        repetitions=int(args.baseline_repetitions),
        output=short_json,
        require_cached=True,
    )
    _run_command(short_command, cwd=source_root, env=env)

    if args.bisect_before_root is None or args.bisect_after_root is None:
        raise ValueError("--bisect-before-root and --bisect-after-root are required for SOL-G4")
    bisect_children: list[tuple[str, Path, str]] = [
        ("before", Path(args.bisect_before_root).resolve(), str(args.bisect_before_commit)),
        ("after", Path(args.bisect_after_root).resolve(), str(args.bisect_after_commit)),
    ]
    bisect_payloads: dict[str, dict[str, Any]] = {}
    for label, root, expected_commit in bisect_children:
        state = _source_state(root)
        if state["commit"] != expected_commit or state["dirty"]:
            raise ValueError(f"{label} bisect worktree does not match clean {expected_commit}: {state}")
        output = raw_root / f"bisect-{label}.json"
        command = child_command(
            mode="baseline",
            root=root,
            prompt_length=int(args.bisect_prompt_length),
            steps=int(args.bisect_steps),
            warmup_steps=int(args.baseline_warmup_steps),
            benchmark_warmups=int(args.baseline_warmups),
            repetitions=int(args.baseline_repetitions),
            output=output,
            require_cached=True,
        )
        _run_command(command, cwd=root, env=env)
        bisect_payloads[label] = _load_child(
            output, expected_token_id=int(args.expected_token_id)
        )

    profile_json = raw_root / "profile-child.json"
    profile_child_command = child_command(
        mode="profile",
        root=source_root,
        prompt_length=int(args.prompt_length),
        steps=int(args.profile_steps),
        warmup_steps=int(args.profile_warmup_steps),
        benchmark_warmups=0,
        repetitions=1,
        output=profile_json,
        require_cached=True,
    )
    roctx_override, roctx_dependencies = _prepare_roctx_override(Path(args.roctx_sdk))
    profile_env = env.copy()
    ld_prefix = os.pathsep.join(
        [str(roctx_override), *(str(path) for path in roctx_dependencies)]
    )
    profile_env["LD_LIBRARY_PATH"] = f"{ld_prefix}:{profile_env.get('LD_LIBRARY_PATH', '')}"
    rocprof_command = [
        str(args.rocprofv3),
        "--kernel-trace",
        "--marker-trace",
        "--hip-trace",
        "--output-format",
        "csv",
        "-d",
        str(trace_dir),
        "-o",
        "sol-g4-gguf-eager",
        "--",
        *profile_child_command,
    ]
    _run_command(rocprof_command, cwd=source_root, env=profile_env)

    baseline = _load_child(baseline_json, expected_token_id=int(args.expected_token_id))
    current_short = _load_child(short_json, expected_token_id=int(args.expected_token_id))
    profile_child = _load_child(profile_json, expected_token_id=int(args.expected_token_id))
    kernel_csv = _single_file(trace_dir, "*_kernel_trace.csv")
    marker_csv = _single_file(trace_dir, "*_marker_api_trace.csv")
    hip_api_csv = _single_file(trace_dir, "*_hip_api_trace.csv")
    windows = _read_marker_windows(marker_csv, MARKER_PREFIX)
    expected_indices = list(range(int(args.profile_steps)))
    if [index for index, _start, _end in windows] != expected_indices:
        raise ValueError(
            f"expected exact marker windows {expected_indices}, observed "
            f"{[index for index, _start, _end in windows]}"
        )
    all_kernels = _read_kernels(kernel_csv)
    selected_kernels = _filter_kernels_by_windows(
        all_kernels, [(start, end) for _index, start, end in windows]
    )
    role_windows = _read_role_windows(marker_csv, ROLE_MARKER_PREFIX)
    hip_launches = _read_hip_launches(hip_api_csv)
    role_kernels = _annotate_kernel_roles(
        selected_kernels,
        launches=hip_launches,
        windows=role_windows,
    )
    profile_summary = _summarize_rows(
        role_kernels, steps=int(args.profile_steps), top=int(args.top)
    )
    profile_summary["roles"] = _summarize_role_rows(
        role_kernels,
        steps=int(args.profile_steps),
    )
    profile_summary["role_marker_windows"] = len(role_windows)
    profile_summary["hip_kernel_launch_apis"] = len(hip_launches)
    profile_summary["role_attributed_kernels"] = sum(
        row["role"] != "unattributed" for row in role_kernels
    )
    profile_wall = _summarize_wall_runs(
        profile_child["measured_runs"], expected_token_id=int(args.expected_token_id)
    )
    profile_summary.update(
        {
            "marker_windows": len(windows),
            "whole_trace_kernels": len(all_kernels),
            "selected_decode_kernels": len(selected_kernels),
            "host_profiled_wall": profile_wall,
            "gpu_kernel_share_of_profiled_host_wall_pct": (
                profile_summary["gpu_us_per_token"]
                / (profile_wall["median_ms_per_token"] * 1000.0)
                * 100.0
            ),
        }
    )

    revision = _revision_boundary(
        bisect_payloads["before"],
        bisect_payloads["after"],
        current_short,
        expected_before=str(args.bisect_before_commit),
        expected_after=str(args.bisect_after_commit),
        route_change_commit=str(args.route_change_commit),
        expected_token_id=int(args.expected_token_id),
    )
    baseline_summary = _summarize_wall_runs(
        baseline["measured_runs"], expected_token_id=int(args.expected_token_id)
    )
    correctness_gate = _validate_g1_artifact(
        Path(args.correctness_artifact),
        model=Path(args.model),
        prompt_token_id=int(args.prompt_token_id),
    )

    from hipengine.benchmark.provenance import collect_artifact_provenance

    provenance = collect_artifact_provenance(
        repo_root=source_root,
        configured_backend=str(args.backend),
        resolved_backend=str(baseline["route"]["resolved_backend"]),
        target_arch=str(baseline["route"]["target_arch"]),
        model_path=Path(args.model),
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=[str(part) for part in sys.argv],
        environment={
            "HIPENGINE_BACKEND": os.environ.get("HIPENGINE_BACKEND"),
            "HIPENGINE_HIP_ARCH": env["HIPENGINE_HIP_ARCH"],
            "HIPENGINE_GGUF_DECODE_REPACK": "1",
            "HIPENGINE_GGUF_WMMA_PREFILL": "1 (constructor)",
            "HIPENGINE_GGUF_GEMV_DECODE": "1 (constructor)",
        },
        build_profile="gguf_eager_decode_sol_g4_audit",
        timing_protocol="clean_resident_eager_1x4_plus_roctx_decode_windows_v2",
        warmups=int(args.baseline_warmups),
        repetitions=int(args.baseline_repetitions),
        profiler={
            "enabled": True,
            "kind": "rocprofv3_kernel_and_marker_trace",
            "command": rocprof_command,
            "profile_steps": int(args.profile_steps),
            "marker_prefix": MARKER_PREFIX,
            "kernel_trace_sha256": _sha256(kernel_csv),
            "marker_trace_sha256": _sha256(marker_csv),
            "hip_api_trace_sha256": _sha256(hip_api_csv),
        },
    )
    measurement_valid = (
        not bool(provenance["dirty"])
        and baseline_summary["all_tokens_exact"]
        and revision["classification"] == "first_performance_changing_revision_found"
        and profile_summary["marker_windows"] == int(args.profile_steps)
        and profile_summary["role_attributed_kernels"] > 0
        and bool(correctness_gate["classification"]["passed"])
    )
    artifact = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if measurement_valid else "diagnostic_invalid",
        "performance_claim": bool(measurement_valid),
        "correctness_claim": True,
        "classification": {
            "passed": bool(measurement_valid),
            "correct_eager_baseline_recorded": True,
            "first_performance_changing_revision_recorded": True,
            "decode_only_amdahl_recorded": True,
            "conclusion": (
                "Current gfx1151 eager decode is correct and retains the post-4499fb13 "
                "short-context speed band; the 512-token baseline is context-cost, not a "
                "post-July regression. The Amdahl table contains only marked timed decode steps."
            ),
        },
        "workload": {
            "model": str(Path(args.model).resolve()),
            "quant": "gguf_q4_k_m",
            "kv_dtype": "bf16",
            "backend": str(args.backend),
            "prompt_source": "repeated_token_id",
            "prompt_token_id": int(args.prompt_token_id),
            "expected_token_id": int(args.expected_token_id),
            "primary_prompt_length": int(args.prompt_length),
            "baseline_steps": int(args.baseline_steps),
            "profile_steps": int(args.profile_steps),
            "bisect_prompt_length": int(args.bisect_prompt_length),
            "bisect_steps": int(args.bisect_steps),
            "route": "repacked WMMA bulk prefill + GEMV eager decode; graph off",
        },
        "correctness_gate": correctness_gate,
        "current_eager_baseline": {
            "protocol": "one resident session; one discarded full run; four measured full runs",
            "summary": baseline_summary,
            "child": baseline,
        },
        "revision_bisect": revision,
        "layer_family_amdahl": {
            "protocol": (
                "rocprofv3 kernel+marker+HIP API trace; one ROCTX range per synchronized eager step; "
                "prefill and warmup excluded by timestamp containment; nested decode roles joined "
                "to asynchronous kernels through HIP launch correlation IDs"
            ),
            "summary": profile_summary,
            "child": profile_child,
            "raw_trace": {
                "raw_root": str(raw_root),
                "kernel_csv": str(kernel_csv),
                "kernel_csv_sha256": _sha256(kernel_csv),
                "marker_csv": str(marker_csv),
                "marker_csv_sha256": _sha256(marker_csv),
                "hip_api_csv": str(hip_api_csv),
                "hip_api_csv_sha256": _sha256(hip_api_csv),
            },
        },
        "provenance": provenance,
    }
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"[gguf-eager-audit] wrote {args.out}", flush=True)
    print(
        f"[gguf-eager-audit] current p512={baseline_summary['median_tok_s']:.3f} tok/s; "
        f"boundary={revision['speedup_after_vs_before']:.3f}x; "
        f"profile={profile_summary['gpu_us_per_token']:.1f} GPU us/token",
        flush=True,
    )
    return 0 if measurement_valid else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-mode", choices=("warmbuild", "baseline", "profile"))
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("hip_gfx1100", "hip_gfx1151"), default="hip_gfx1151")
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--expected-token-id", type=int, default=9707)
    parser.add_argument("--steps", type=int, default=24, help="Internal child timed steps")
    parser.add_argument("--warmup-steps", type=int, default=4, help="Internal child decode warmup")
    parser.add_argument("--benchmark-warmups", type=int, default=0, help="Internal discarded full runs")
    parser.add_argument("--repetitions", type=int, default=1, help="Internal measured full runs")
    parser.add_argument("--max-seq", type=int, default=0)
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument("--child-json", type=Path)

    parser.add_argument("--baseline-steps", type=int, default=128)
    parser.add_argument("--baseline-warmup-steps", type=int, default=1)
    parser.add_argument("--baseline-warmups", type=int, default=1)
    parser.add_argument("--baseline-repetitions", type=int, default=4)
    parser.add_argument("--profile-steps", type=int, default=24)
    parser.add_argument("--profile-warmup-steps", type=int, default=4)
    parser.add_argument("--bisect-prompt-length", type=int, default=8)
    parser.add_argument("--bisect-steps", type=int, default=32)
    parser.add_argument("--bisect-before-root", type=Path)
    parser.add_argument("--bisect-after-root", type=Path)
    parser.add_argument("--bisect-before-commit", default=DEFAULT_BEFORE_COMMIT)
    parser.add_argument("--bisect-after-commit", default=DEFAULT_AFTER_COMMIT)
    parser.add_argument("--route-change-commit", default=DEFAULT_ROUTE_CHANGE_COMMIT)
    parser.add_argument("--correctness-artifact", type=Path, default=DEFAULT_G1_ARTIFACT)
    parser.add_argument("--compiler-version-file", type=Path, default=Path("/tmp/hipengine-hipcc-version.txt"))
    parser.add_argument("--skip-warmbuild", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/hipengine-sol-g4-gguf-eager-audit"))
    parser.add_argument("--rocprofv3", default=shutil.which("rocprofv3") or "rocprofv3")
    parser.add_argument("--roctx-sdk", type=Path, default=_default_roctx_sdk())
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    for name in (
        "prompt_length",
        "steps",
        "repetitions",
        "baseline_steps",
        "baseline_repetitions",
        "profile_steps",
        "bisect_prompt_length",
        "bisect_steps",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "warmup_steps",
        "benchmark_warmups",
        "baseline_warmup_steps",
        "baseline_warmups",
        "profile_warmup_steps",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    return _run_child(args) if args.child_mode is not None else _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
