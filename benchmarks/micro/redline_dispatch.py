#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
# Derived from warpfront/redline@33683f3 dispatch-floor and C-API examples.
"""Measure hipEngine's dispatch/grid floor through direct retained Redline PM4.

This is the separate dispatch control for ``redline_matrix.py``. The native HIP
runner already captures an inner hipGraph, so timer substitution cannot lower it.
This runner instead compiles the same ``gmb_noop_kernel`` source through pinned
Radiowave, records single and burst retained PM4 IBs directly through
``redline-capi``, and returns every GPU/host timing sample plus exact output
correctness. There is no native-HIP fallback.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from hipengine.core.hip import HipMemcpyKind, get_hip_runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
MICRO_ROOT = REPO_ROOT / "benchmarks" / "micro"
TIMING_CONTRACT = MICRO_ROOT / "timing_contract.py"
REDLINE_MATRIX = MICRO_ROOT / "redline_matrix.py"
GMB_SOURCE = REPO_ROOT / "scripts" / "microbench" / "graph_node_microbench.hip"
PINNED_REDLINE_COMMIT = "33683f3d4f302a6c56bcc7a4c33ab8be3262dd2e"
RL_OK = 0
RL_ERR_COMPILE = -4
RL_QUEUE_AUTO = 0


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing_contract = _load_module(TIMING_CONTRACT, "micro_redline_dispatch_timing")
redline_matrix = _load_module(REDLINE_MATRIX, "micro_redline_dispatch_matrix")


def _copy_integer(buffer: bytearray, arg: dict[str, Any], value: int) -> None:
    offset = int(arg["offset"])
    size = int(arg["size"])
    if offset < 0 or size <= 0 or offset + size > len(buffer):
        raise ValueError("kernel argument metadata is out of range")
    maximum = (1 << (size * 8)) - 1
    if value < 0 or value > maximum:
        raise ValueError(f"kernel argument value {value} does not fit in {size} bytes")
    buffer[offset : offset + size] = value.to_bytes(size, "little")


def pack_gmb_kernarg(
    spec: dict[str, Any],
    *,
    output_pointer: int,
    n: int,
    grid_blocks: int,
    block_size: int,
) -> bytearray:
    """Pack explicit and hidden hipcc kernargs from inspected metadata."""

    size = int(spec.get("kernarg_size", 0))
    args = spec.get("args")
    if size <= 0 or not isinstance(args, list):
        raise ValueError("kernel specification is missing kernarg metadata")
    if min(output_pointer, n, grid_blocks, block_size) <= 0:
        raise ValueError("dispatch pointers and geometry must be positive")
    packed = bytearray(size)
    explicit_values = (output_pointer, n)
    explicit_index = 0
    hidden_values = {
        "hidden_block_count_x": grid_blocks,
        "hidden_block_count_y": 1,
        "hidden_block_count_z": 1,
        "hidden_group_size_x": block_size,
        "hidden_group_size_y": 1,
        "hidden_group_size_z": 1,
        "hidden_remainder_x": block_size,
        "hidden_remainder_y": 1,
        "hidden_remainder_z": 1,
        "hidden_grid_dims": 1,
        "hidden_dynamic_lds_size": 0,
    }
    for arg in args:
        kind = str(arg.get("value_kind") or "")
        if kind.startswith("hidden_"):
            if kind in hidden_values:
                _copy_integer(packed, arg, hidden_values[kind])
            continue
        if explicit_index >= len(explicit_values):
            raise ValueError("gmb_noop kernel exposes unexpected explicit arguments")
        _copy_integer(packed, arg, explicit_values[explicit_index])
        explicit_index += 1
    if explicit_index != len(explicit_values):
        raise ValueError("gmb_noop kernel does not expose pointer and length arguments")
    return packed


def make_dispatch_row(
    *,
    mode: str,
    sweep: str,
    count: int,
    grid_blocks: int,
    lane_count: int,
    warmup: int,
    gpu_single_us: Sequence[float],
    host_single_us: Sequence[float],
    gpu_burst_us: Sequence[float],
    host_burst_us: Sequence[float],
    single_correct: bool,
    burst_correct: bool,
) -> dict[str, Any]:
    mode = timing_contract.parse_timing_mode(mode)
    if sweep not in {"count", "grid"}:
        raise ValueError("dispatch sweep must be count or grid")
    if min(count, grid_blocks, lane_count) <= 0 or warmup < 0:
        raise ValueError("dispatch shape/lane values must be positive")
    correctness_pass = bool(single_correct and burst_correct)
    correctness = timing_contract.make_correctness(
        status="pass" if correctness_pass else "fail",
        oracle="exact every-element output after single and measured burst replay",
        logical_iterations=count,
        coverage=(
            "all_dispatches" if mode == "independent_throughput" else "chained_final_state"
        ),
        synchronization_method=(
            "redline_rmw" if mode == "serial_latency" else "disjoint_retained_pm4_lanes"
        ),
        barrier_count=count - 1 if mode == "serial_latency" else 0,
    )
    row = timing_contract.make_timed_row_contract(
        timing_mode=mode,
        backend="redline",
        repetitions=count,
        dispatches_per_iteration=1,
        dependency_validation_status="pass" if correctness_pass else "fail",
        submission=timing_contract.make_submission(
            strategy="retained_pm4_ib",
            queue_or_stream_count=lane_count,
            recording_in_timed_region=False,
        ),
        single_timing=timing_contract.make_timing_control(
            logical_iterations=1,
            dispatches_per_iteration=1,
            gpu_samples_us=gpu_single_us,
            host_samples_us=host_single_us,
            gpu_clock="redline_pm4_timestamp",
        ),
        burst_timing=timing_contract.make_timing_control(
            logical_iterations=count,
            dispatches_per_iteration=1,
            gpu_samples_us=gpu_burst_us,
            host_samples_us=host_burst_us,
            gpu_clock="redline_pm4_timestamp",
        ),
        correctness=correctness,
    )
    row["timing"]["single"]["retained_lane_count"] = 1
    row["timing"]["burst"]["retained_lane_count"] = (
        1 if mode == "serial_latency" else min(lane_count, count)
    )
    return {
        "sweep": sweep,
        "node_count": count,
        "dispatch_count": count,
        "grid_blocks": grid_blocks,
        "correctness_pass": correctness_pass,
        "single_correctness_pass": bool(single_correct),
        "burst_correctness_pass": bool(burst_correct),
        "warmup": warmup,
        **row,
    }


def dispatch_row_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    count = row.get("node_count", row.get("dispatch_count"))
    return (
        str(row.get("sweep")),
        int(count),
        int(row.get("grid_blocks")),
        str(row.get("timing_mode")),
    )


def build_dispatch_result(
    *,
    rows: list[dict[str, Any]],
    hardware: dict[str, Any],
    source: dict[str, Any],
    command: Sequence[str],
    environment_ref: str,
    redline_provenance: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("dispatch result requires timing rows")
    keys = [dispatch_row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("dispatch result contains duplicate rows")
    for row in rows:
        timing_contract.validate_timed_row(row)
    correctness_pass = all(bool(row.get("correctness_pass")) for row in rows)
    return {
        "schema_version": 2,
        "kind": "hipengine_micro_result",
        "bench": "dispatch_grid_floor",
        "backend": "redline",
        "hardware": hardware,
        "source": source,
        "command": list(command),
        "parameters": parameters,
        "correctness": {
            "status": "pass" if correctness_pass else "fail",
            "oracle": "exact every-element output after single and measured burst replay",
            "rows": len(rows),
        },
        "measurements": {"rows": rows},
        "classification": "runtime_dispatch",
        "environment_ref": environment_ref,
        "redline_provenance": redline_provenance,
        "notes": (
            "Direct profiled retained-PM4 dispatch/grid control. The code object is "
            "Radiowave-certified from hipEngine's gmb_noop source; no hipGraph preload "
            "or native-HIP fallback participates in timing."
        ),
    }


class _Capi:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.lib = ctypes.CDLL(str(self.path))
        self._configure()

    def _configure(self) -> None:
        lib = self.lib
        vp = ctypes.c_void_p
        lib.rl_abi_version.argtypes = []
        lib.rl_abi_version.restype = ctypes.c_uint32
        lib.rl_gpu_new.argtypes = [ctypes.c_int32]
        lib.rl_gpu_new.restype = vp
        lib.rl_gpu_free.argtypes = [vp]
        lib.rl_gpu_pm4_queue_count.argtypes = [vp, ctypes.c_int, ctypes.c_size_t]
        lib.rl_gpu_pm4_queue_count.restype = ctypes.c_size_t
        lib.rl_gpu_load_module_radiowave.argtypes = [
            vp,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(vp),
        ]
        lib.rl_gpu_load_module_radiowave.restype = ctypes.c_int32
        lib.rl_module_free.argtypes = [vp]
        lib.rl_module_kernarg_size.argtypes = [vp, ctypes.c_char_p]
        lib.rl_module_kernarg_size.restype = ctypes.c_int64
        lib.rl_module_radiowave_certified.argtypes = [vp]
        lib.rl_module_radiowave_certified.restype = ctypes.c_bool
        lib.rl_pm4_builder_new.argtypes = [vp]
        lib.rl_pm4_builder_new.restype = vp
        lib.rl_pm4_builder_free.argtypes = [vp]
        lib.rl_pm4_wait_rmw.argtypes = [vp, vp, ctypes.c_char_p]
        lib.rl_pm4_wait_rmw.restype = ctypes.c_int32
        lib.rl_pm4_dispatch.argtypes = [
            vp,
            vp,
            ctypes.c_char_p,
            *([ctypes.c_uint32] * 7),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        lib.rl_pm4_dispatch.restype = ctypes.c_int32
        lib.rl_pm4_finalize_profiled.argtypes = [vp, vp, ctypes.POINTER(vp)]
        lib.rl_pm4_finalize_profiled.restype = ctypes.c_int32
        lib.rl_pm4_replay_profiled.argtypes = [vp, ctypes.POINTER(ctypes.c_double)]
        lib.rl_pm4_replay_profiled.restype = ctypes.c_int32
        lib.rl_pm4_ib_free.argtypes = [vp]
        lib.rl_pm4_finalize_multi_profiled.argtypes = [
            vp,
            ctypes.POINTER(vp),
            ctypes.c_size_t,
            ctypes.POINTER(vp),
        ]
        lib.rl_pm4_finalize_multi_profiled.restype = ctypes.c_int32
        lib.rl_pm4_replay_multi_profiled.argtypes = [vp, ctypes.POINTER(ctypes.c_double)]
        lib.rl_pm4_replay_multi_profiled.restype = ctypes.c_int32
        lib.rl_pm4_multi_ib_lane_count.argtypes = [vp]
        lib.rl_pm4_multi_ib_lane_count.restype = ctypes.c_size_t
        lib.rl_pm4_multi_ib_free.argtypes = [vp]

    def check(self, status: int, what: str) -> None:
        if int(status) != RL_OK:
            raise RuntimeError(f"{what} failed with Redline status {int(status)}")


class _RetainedIb:
    def __init__(self, capi: _Capi, pointer: int, *, multi: bool, lanes: int):
        self.capi = capi
        self.pointer = pointer
        self.multi = multi
        self.lanes = lanes

    def replay(self) -> tuple[float, float]:
        elapsed = ctypes.c_double()
        started = time.perf_counter_ns()
        if self.multi:
            status = self.capi.lib.rl_pm4_replay_multi_profiled(
                ctypes.c_void_p(self.pointer), ctypes.byref(elapsed)
            )
        else:
            status = self.capi.lib.rl_pm4_replay_profiled(
                ctypes.c_void_p(self.pointer), ctypes.byref(elapsed)
            )
        host_us = (time.perf_counter_ns() - started) / 1000.0
        self.capi.check(status, "profiled retained-PM4 replay")
        return float(elapsed.value), host_us

    def close(self) -> None:
        if not self.pointer:
            return
        if self.multi:
            self.capi.lib.rl_pm4_multi_ib_free(ctypes.c_void_p(self.pointer))
        else:
            self.capi.lib.rl_pm4_ib_free(ctypes.c_void_p(self.pointer))
        self.pointer = 0


class _DispatchContext:
    def __init__(
        self,
        *,
        capi_path: Path,
        code_object: Path,
        radiowave_manifest: Path,
        kernel_spec: dict[str, Any],
        lane_cap: int,
    ):
        self.capi = _Capi(capi_path)
        if self.capi.lib.rl_abi_version() != 1:
            raise RuntimeError("unsupported Redline C ABI")
        self.gpu = self.capi.lib.rl_gpu_new(0)
        if not self.gpu:
            raise RuntimeError("rl_gpu_new failed")
        code = code_object.read_bytes()
        manifest = radiowave_manifest.read_bytes()
        self._code_storage = (ctypes.c_uint8 * len(code)).from_buffer_copy(code)
        self._manifest_storage = (ctypes.c_uint8 * len(manifest)).from_buffer_copy(manifest)
        module = ctypes.c_void_p()
        self.capi.check(
            self.capi.lib.rl_gpu_load_module_radiowave(
                self.gpu,
                self._code_storage,
                len(code),
                self._manifest_storage,
                len(manifest),
                ctypes.byref(module),
            ),
            "rl_gpu_load_module_radiowave",
        )
        self.module = int(module.value or 0)
        if not self.module or not self.capi.lib.rl_module_radiowave_certified(
            ctypes.c_void_p(self.module)
        ):
            raise RuntimeError("Redline module is not Radiowave-certified")
        self.spec = kernel_spec
        self.symbol = str(kernel_spec["symbol"])
        reported_size = int(
            self.capi.lib.rl_module_kernarg_size(
                ctypes.c_void_p(self.module), self.symbol.encode()
            )
        )
        if reported_size != int(kernel_spec["kernarg_size"]):
            raise RuntimeError(
                f"Redline kernarg size {reported_size} != manifest {kernel_spec['kernarg_size']}"
            )
        resolved = int(
            self.capi.lib.rl_gpu_pm4_queue_count(
                self.gpu, RL_QUEUE_AUTO, max(1, lane_cap)
            )
        )
        if resolved <= 0:
            raise RuntimeError("Redline queue policy resolved zero lanes")
        self.lane_cap = min(resolved, lane_cap)

    def close(self) -> None:
        if self.module:
            self.capi.lib.rl_module_free(ctypes.c_void_p(self.module))
            self.module = 0
        if self.gpu:
            self.capi.lib.rl_gpu_free(self.gpu)
            self.gpu = None

    def _dispatch(
        self,
        builder: int,
        *,
        output_pointer: int,
        n: int,
        grid_blocks: int,
        block_size: int,
    ) -> None:
        kernarg = pack_gmb_kernarg(
            self.spec,
            output_pointer=output_pointer,
            n=n,
            grid_blocks=grid_blocks,
            block_size=block_size,
        )
        storage = (ctypes.c_uint8 * len(kernarg)).from_buffer_copy(kernarg)
        self.capi.check(
            self.capi.lib.rl_pm4_dispatch(
                ctypes.c_void_p(builder),
                ctypes.c_void_p(self.module),
                self.symbol.encode(),
                grid_blocks * block_size,
                1,
                1,
                block_size,
                1,
                1,
                0,
                storage,
                len(storage),
            ),
            "rl_pm4_dispatch",
        )

    def build(
        self,
        *,
        mode: str,
        output_pointer: int,
        n: int,
        count: int,
        grid_blocks: int,
        block_size: int,
    ) -> _RetainedIb:
        mode = timing_contract.parse_timing_mode(mode)
        active_lanes = (
            1 if mode == "serial_latency" else min(self.lane_cap, count)
        )
        builders: list[int] = []
        builders_owned = True
        try:
            for _ in range(active_lanes):
                builder = self.capi.lib.rl_pm4_builder_new(self.gpu)
                if not builder:
                    raise RuntimeError("rl_pm4_builder_new failed")
                builders.append(int(builder))
            for rep in range(count):
                lane = 0 if mode == "serial_latency" else rep % active_lanes
                builder = builders[lane]
                if mode == "serial_latency" and rep > 0:
                    self.capi.check(
                        self.capi.lib.rl_pm4_wait_rmw(
                            ctypes.c_void_p(builder),
                            ctypes.c_void_p(self.module),
                            self.symbol.encode(),
                        ),
                        "rl_pm4_wait_rmw",
                    )
                pointer = (
                    output_pointer
                    if mode == "serial_latency"
                    else output_pointer + rep * n * ctypes.sizeof(ctypes.c_float)
                )
                self._dispatch(
                    builder,
                    output_pointer=pointer,
                    n=n,
                    grid_blocks=grid_blocks,
                    block_size=block_size,
                )
            out = ctypes.c_void_p()
            if active_lanes == 1:
                builders_owned = False
                self.capi.check(
                    self.capi.lib.rl_pm4_finalize_profiled(
                        self.gpu, ctypes.c_void_p(builders[0]), ctypes.byref(out)
                    ),
                    "rl_pm4_finalize_profiled",
                )
                return _RetainedIb(
                    self.capi, int(out.value or 0), multi=False, lanes=1
                )
            array = (ctypes.c_void_p * active_lanes)(
                *[ctypes.c_void_p(builder) for builder in builders]
            )
            status = int(
                self.capi.lib.rl_pm4_finalize_multi_profiled(
                    self.gpu, array, active_lanes, ctypes.byref(out)
                )
            )
            if status in {RL_OK, RL_ERR_COMPILE}:
                builders_owned = False
            self.capi.check(status, "rl_pm4_finalize_multi_profiled")
            observed = int(self.capi.lib.rl_pm4_multi_ib_lane_count(out))
            if observed != active_lanes:
                self.capi.lib.rl_pm4_multi_ib_free(out)
                raise RuntimeError(
                    f"retained multi-IB lane count {observed} != {active_lanes}"
                )
            return _RetainedIb(
                self.capi, int(out.value or 0), multi=True, lanes=active_lanes
            )
        except Exception:
            if builders_owned:
                for builder in builders:
                    self.capi.lib.rl_pm4_builder_free(ctypes.c_void_p(builder))
            raise


def _copy_output(runtime: Any, pointer: int, elements: int) -> list[float]:
    host = (ctypes.c_float * elements)()
    runtime.memcpy(
        ctypes.addressof(host),
        pointer,
        ctypes.sizeof(host),
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return list(host)


def _all_exact(values: Sequence[float], expected: float) -> bool:
    return all(value == expected and math.isfinite(value) for value in values)


def _measure_ib(
    runtime: Any,
    ib: _RetainedIb,
    *,
    output_pointer: int,
    output_elements: int,
    expected_increment: float,
    warmup: int,
    samples: int,
) -> tuple[list[float], list[float], bool]:
    output_bytes = output_elements * ctypes.sizeof(ctypes.c_float)
    runtime.memset(output_pointer, 0, output_bytes)
    runtime.device_synchronize()
    ib.replay()
    first_correct = _all_exact(
        _copy_output(runtime, output_pointer, output_elements), expected_increment
    )
    runtime.memset(output_pointer, 0, output_bytes)
    runtime.device_synchronize()
    for _ in range(warmup):
        ib.replay()
    gpu_samples: list[float] = []
    host_samples: list[float] = []
    for _ in range(samples):
        gpu_us, host_us = ib.replay()
        gpu_samples.append(gpu_us)
        host_samples.append(host_us)
    final_expected = expected_increment * (warmup + samples)
    final_correct = _all_exact(
        _copy_output(runtime, output_pointer, output_elements), final_expected
    )
    return gpu_samples, host_samples, first_correct and final_correct


def _run_command(command: Sequence[str], *, cwd: Path, env: dict[str, str]) -> str:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}"
        )
    return completed.stdout


def _compile_code_object(args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    build = args.build_dir.expanduser().resolve()
    build.mkdir(parents=True, exist_ok=True)
    bundle = build / "gmb_noop.redline.co"
    radiowave_manifest = build / "gmb_noop.redline.radiowave.json"
    hsaco = build / "gmb_noop.redline.hsaco"
    kernel_manifest = build / "gmb_noop.redline.manifest.json"
    radiowave = args.redline_root / "target" / "release" / "radiowave"
    compile_command = [
        str(radiowave),
        "compile",
        "--source",
        str(GMB_SOURCE),
        "--output",
        str(bundle),
        "--manifest",
        str(radiowave_manifest),
        "--arch",
        args.gfx_arch,
        "--scheduler-profile",
        "default",
        "--hipcc",
        str(args.hipcc),
        "--wave32",
        "--no-fast-math",
    ]
    _run_command(compile_command, cwd=REPO_ROOT, env=env)
    listed = _run_command(
        [
            str(args.bundler),
            "-type=o",
            f"-input={bundle}",
            "-list",
        ],
        cwd=REPO_ROOT,
        env=env,
    ).splitlines()
    target = next((line for line in listed if "amdgcn-amd-amdhsa" in line), None)
    if not target:
        raise RuntimeError("Radiowave output contains no AMDGPU offload target")
    unbundle_command = [
        str(args.bundler),
        "-type=o",
        f"-input={bundle}",
        f"-targets={target}",
        f"-output={hsaco}",
        "-unbundle",
    ]
    _run_command(unbundle_command, cwd=REPO_ROOT, env=env)
    manifest_tool = (
        args.redline_root / "examples" / "hipengine-6409" / "hsaco_manifest.py"
    )
    manifest_command = [
        sys.executable,
        str(manifest_tool),
        str(hsaco),
        "--readobj",
        str(args.llvm_readobj),
        "--out",
        str(kernel_manifest),
    ]
    _run_command(manifest_command, cwd=REPO_ROOT, env=env)
    kernels = json.loads(kernel_manifest.read_text(encoding="utf-8"))["kernels"]
    kernel = next(
        (item for item in kernels if item.get("name") == "gmb_noop_kernel"), None
    )
    if kernel is None:
        raise RuntimeError("compiled code object does not contain gmb_noop_kernel")
    certification = json.loads(radiowave_manifest.read_text(encoding="utf-8"))
    bundle_sha = redline_matrix._sha256(bundle)
    expected_sha = str(certification.get("output_sha256") or "")
    if expected_sha and expected_sha not in {bundle_sha, bundle_sha.removeprefix("sha256:")}:
        raise RuntimeError("Radiowave output hash does not match its manifest")
    return {
        "bundle": bundle,
        "radiowave_manifest": radiowave_manifest,
        "hsaco": hsaco,
        "kernel_manifest": kernel_manifest,
        "kernel": kernel,
        "commands": [compile_command, unbundle_command, manifest_command],
    }


def _source_record(environment: dict[str, Any], source_hash: str) -> dict[str, Any]:
    repo = environment.get("repo")
    if not isinstance(repo, dict):
        raise ValueError("environment is missing repository provenance")
    return {
        "repo": str(repo.get("root") or REPO_ROOT),
        "branch": str(repo.get("branch") or ""),
        "commit": str(repo.get("commit") or ""),
        "dirty": bool(repo.get("dirty")),
        "source_hash": source_hash,
    }


def _combined_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    evidence = redline_matrix.validate_redline_checkout(args.redline_root)
    library = args.redline_root / "target" / "release" / "libredline_dispatch.so"
    env = dict(os.environ)
    env.update(
        {
            "PATH": (
                f"{args.hipcc.parent}:{args.bundler.parent}:"
                f"{env.get('PATH', '')}"
            ),
            "ROCM_PATH": str(args.rocm_root),
            "HIP_PATH": str(args.rocm_root),
            "HIP_CLANG_PATH": str(args.bundler.parent),
            "HIPENGINE_HIP_ARCH": args.gfx_arch,
            "HIP_VISIBLE_DEVICES": args.visible_device,
            "ROCR_VISIBLE_DEVICES": args.visible_device,
            "REDLINE_REAL_HIPCC": str(args.hipcc),
        }
    )
    if args.build_redline:
        redline_matrix._build_redline(
            args.redline_root,
            env=env,
            log=args.build_dir / "build-redline.log",
        )
    if not library.is_file():
        raise ValueError(f"Redline C API library is missing: {library}")
    environment = json.loads(args.environment_json.read_text(encoding="utf-8"))
    compiled = _compile_code_object(args, env)
    runtime = get_hip_runtime()
    context = _DispatchContext(
        capi_path=library,
        code_object=compiled["bundle"],
        radiowave_manifest=compiled["radiowave_manifest"],
        kernel_spec=compiled["kernel"],
        lane_cap=args.independent_lanes,
    )
    rows: list[dict[str, Any]] = []
    try:
        shapes = [
            *( ("count", count, math.ceil(args.n / args.block_size)) for count in args.counts ),
            *( ("grid", args.grid_sweep_count, grid) for grid in args.grid_sweep ),
        ]
        for mode in args.modes:
            for sweep, count, grid_blocks in shapes:
                single_elements = args.n
                burst_elements = args.n * (
                    count if mode == "independent_throughput" else 1
                )
                single_pointer = runtime.malloc(
                    single_elements * ctypes.sizeof(ctypes.c_float)
                )
                burst_pointer = runtime.malloc(
                    burst_elements * ctypes.sizeof(ctypes.c_float)
                )
                single_ib = None
                burst_ib = None
                try:
                    single_ib = context.build(
                        mode=mode,
                        output_pointer=single_pointer,
                        n=args.n,
                        count=1,
                        grid_blocks=grid_blocks,
                        block_size=args.block_size,
                    )
                    burst_ib = context.build(
                        mode=mode,
                        output_pointer=burst_pointer,
                        n=args.n,
                        count=count,
                        grid_blocks=grid_blocks,
                        block_size=args.block_size,
                    )
                    single_gpu, single_host, single_correct = _measure_ib(
                        runtime,
                        single_ib,
                        output_pointer=single_pointer,
                        output_elements=single_elements,
                        expected_increment=1.0,
                        warmup=args.warmup,
                        samples=args.samples,
                    )
                    burst_gpu, burst_host, burst_correct = _measure_ib(
                        runtime,
                        burst_ib,
                        output_pointer=burst_pointer,
                        output_elements=burst_elements,
                        expected_increment=(
                            float(count) if mode == "serial_latency" else 1.0
                        ),
                        warmup=args.warmup,
                        samples=args.samples,
                    )
                    rows.append(
                        make_dispatch_row(
                            mode=mode,
                            sweep=sweep,
                            count=count,
                            grid_blocks=grid_blocks,
                            lane_count=burst_ib.lanes,
                            warmup=args.warmup,
                            gpu_single_us=single_gpu,
                            host_single_us=single_host,
                            gpu_burst_us=burst_gpu,
                            host_burst_us=burst_host,
                            single_correct=single_correct,
                            burst_correct=burst_correct,
                        )
                    )
                finally:
                    if burst_ib is not None:
                        burst_ib.close()
                    if single_ib is not None:
                        single_ib.close()
                    runtime.free(burst_pointer)
                    runtime.free(single_pointer)
    finally:
        context.close()
    source_paths = [Path(__file__).resolve(), GMB_SOURCE, TIMING_CONTRACT, REDLINE_MATRIX]
    source = _source_record(environment, _combined_hash(source_paths))
    adapters = redline_matrix._adapter_paths(args.redline_root)
    sidecars = [
        compiled["bundle"],
        compiled["radiowave_manifest"],
        compiled["hsaco"],
        compiled["kernel_manifest"],
    ]
    provenance = {
        "checkout": {
            "root": evidence["root"],
            "commit": evidence["commit"],
            "dirty": evidence["dirty"],
        },
        "library": redline_matrix._file_record(library),
        "adapters": [redline_matrix._file_record(path) for path in adapters],
        "sidecars": [redline_matrix._file_record(path) for path in sidecars],
        "compile_commands": compiled["commands"],
        "execution_proof": {
            "api": "redline-capi",
            "native_hip_fallback_available": False,
            "profiled_retained_pm4_required": True,
            "radiowave_manifest_verified": True,
        },
    }
    return build_dispatch_result(
        rows=rows,
        hardware={"gpu_name": args.gpu_name, "gfx_arch": args.gfx_arch},
        source=source,
        command=sys.argv.copy(),
        environment_ref=str(args.environment_ref),
        redline_provenance=provenance,
        parameters={
            "counts": args.counts,
            "grid_sweep": args.grid_sweep,
            "grid_sweep_count": args.grid_sweep_count,
            "n_elements": args.n,
            "local_size_x": args.block_size,
            "modes": args.modes,
            "samples": args.samples,
            "warmup": args.warmup,
            "independent_lanes_requested": args.independent_lanes,
            "independent_lanes_resolved": context.lane_cap,
            "method": "direct profiled retained PM4 with every-element correctness",
        },
    )


def _positive_csv(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redline-root", type=Path, default=Path("/home/lhl/redline"))
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path, required=True)
    parser.add_argument("--environment-ref", type=Path, required=True)
    parser.add_argument("--rocm-root", type=Path, required=True)
    parser.add_argument("--hipcc", type=Path, required=True)
    parser.add_argument("--bundler", type=Path, required=True)
    parser.add_argument("--llvm-readobj", type=Path, required=True)
    parser.add_argument("--build-redline", action="store_true")
    parser.add_argument("--gfx-arch", default="gfx1100")
    parser.add_argument("--gpu-name", default="AMD Radeon Pro W7900")
    parser.add_argument("--visible-device", default="0")
    parser.add_argument("--counts", type=_positive_csv, default=_positive_csv("1,50,200,941"))
    parser.add_argument("--grid-sweep", type=_positive_csv, default=_positive_csv("1,128,1024,8192"))
    parser.add_argument("--grid-sweep-count", type=int, default=941)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--modes", default=",".join(("serial_latency", "independent_throughput")))
    parser.add_argument("--independent-lanes", type=int, default=2)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args(argv)
    args.redline_root = args.redline_root.expanduser().resolve()
    args.rocm_root = args.rocm_root.expanduser().resolve()
    args.hipcc = args.hipcc.expanduser().resolve()
    args.bundler = args.bundler.expanduser().resolve()
    args.llvm_readobj = args.llvm_readobj.expanduser().resolve()
    args.environment_json = args.environment_json.expanduser().resolve()
    args.environment_ref = args.environment_ref.expanduser().resolve()
    args.build_dir = args.build_dir.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.modes = [item for item in args.modes.split(",") if item]
    if not args.modes or any(mode not in timing_contract.TIMING_MODES for mode in args.modes):
        parser.error("--modes must contain serial_latency and/or independent_throughput")
    if min(
        args.grid_sweep_count,
        args.n,
        args.block_size,
        args.independent_lanes,
        args.samples,
    ) <= 0 or args.warmup < 0:
        parser.error("shape/lane/sample values must be positive and warmup non-negative")
    if not args.rocm_root.is_dir():
        parser.error(f"ROCm root is missing: {args.rocm_root}")
    for path, label in (
        (args.hipcc, "hipcc"),
        (args.bundler, "clang-offload-bundler"),
        (args.llvm_readobj, "llvm-readobj"),
        (args.environment_json, "environment JSON"),
    ):
        if not path.is_file():
            parser.error(f"{label} is missing: {path}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_benchmark(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
