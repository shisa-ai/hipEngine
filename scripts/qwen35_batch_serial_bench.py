#!/usr/bin/env python3
"""Diagnostic benchmark for the Qwen3.5/PARO scheduler serial c>N bridge.

This measures the current resident batch scheduler path that uses
``step_batch_serial`` over batch-shaped state/KV slots.  It is intentionally
labelled as a blocked/non-retained throughput attempt until native compact
prefill and c-aware decode kernels replace the serial bridge and the standard
correctness/protocol gates in ``docs/BENCHMARK.md`` pass.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.generation import GeneratedToken, ResidentBatchScheduler
from hipengine.kvcache import ResolvedKVPolicy
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_batch_artifact_schema import _load_payload, validate_cn_diagnostic_artifact_payload
from scripts.qwen35_kv_policy_args import add_kv_policy_args, kv_policy_json, resolve_args_kv_policy

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)
DEFAULT_FIXTURE = "fixtures/qwen35_paro/parent_512_32_seed1234.json"
_COMMAND_ENV_KEYS = ("HIP_VISIBLE_DEVICES",)


def _payload_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)


def _command_env_prefix_parts() -> list[str]:
    assignments = [
        f"{key}={value}"
        for key in _COMMAND_ENV_KEYS
        if (value := os.environ.get(key)) is not None and value.strip()
    ]
    return ["env", *assignments] if assignments else []


def _load_prompt_slices(path: Path, *, prompt_length: int, batch_size: int) -> list[list[int]]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    fixture = _load_payload(path)
    tokens = [int(token) for token in fixture["prompt_ids"]]
    needed = int(prompt_length) * int(batch_size)
    if len(tokens) < needed:
        raise ValueError(f"fixture contains {len(tokens)} tokens, need at least {needed}")
    return [tokens[row * prompt_length : (row + 1) * prompt_length] for row in range(batch_size)]


def _shape_key_payload(key) -> dict[str, Any]:
    return {
        "mode": key.mode.value,
        "active_c": key.active_c,
        "context_bucket": key.context_bucket,
        "active_mask": list(key.active_mask),
        "kv_storage_dtype": key.kv_storage_dtype,
        "layer_plan": key.layer_plan,
        "top_k": key.top_k,
        "experts_per_token": key.experts_per_token,
        "replay_steps": key.replay_steps,
        "draft_depth": key.draft_depth,
        "tree_shape": list(key.tree_shape),
    }


def _result_payload(result) -> dict[str, Any]:
    return {
        "token_id": int(result.token_id),
        "token_text": result.token_text,
        "logit": float(result.logit),
    }


def _summarize_samples(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(sample) for sample in samples]
    if not values:
        return {"samples": [], "median": None, "p95": None, "min": None, "max": None, "stdev": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "samples": values,
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _all_finite(rows: Iterable[dict[str, Any]]) -> bool:
    return all(math.isfinite(float(row["logit"])) for row in rows)


def _run_capture(command: Sequence[str], *, timeout: float = 5.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {"command": " ".join(shlex.quote(part) for part in command), "returncode": proc.returncode, "output": proc.stdout.strip()}
    except Exception as exc:  # pragma: no cover - best-effort environment capture
        return {"command": " ".join(shlex.quote(part) for part in command), "returncode": None, "output": f"{type(exc).__name__}: {exc}"}


def _software_context() -> dict[str, Any]:
    commit = _run_capture(["git", "rev-parse", "--short", "HEAD"])
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO_ROOT, check=False).returncode != 0
    return {
        "python": sys.version.split()[0],
        "hipcc_version": _run_capture(["hipcc", "--version"], timeout=10.0)["output"],
        "hipengine_commit": commit["output"],
        "hipengine_dirty": dirty,
        "torch_rocm": _run_capture(
            ["python3", "-c", "import torch; print(torch.__version__, torch.version.hip)"],
            timeout=10.0,
        ),
    }


def _visible_hip_device_context() -> dict[str, Any]:
    env_keys = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
    visible_env: dict[str, str] = {}
    for key in env_keys:
        value = os.environ.get(key)
        if value is not None and value.strip():
            visible_env[key] = value
    context: dict[str, Any] = {"env": visible_env}
    try:
        hip = ctypes.CDLL("libamdhip64.so")
        count = ctypes.c_int()
        count_error = int(hip.hipGetDeviceCount(ctypes.byref(count)))
        context["hipGetDeviceCount_error"] = count_error
        context["visible_device_count"] = int(count.value)
        if count_error != 0 or count.value <= 0:
            return context
        device = ctypes.c_int()
        device_error = int(hip.hipGetDevice(ctypes.byref(device)))
        context["hipGetDevice_error"] = device_error
        context["current_device"] = int(device.value)
        if device_error != 0:
            return context
        name = ctypes.create_string_buffer(256)
        name_error = int(hip.hipDeviceGetName(name, len(name), device))
        context["hipDeviceGetName_error"] = name_error
        if name_error == 0:
            context["device_name"] = name.value.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - best-effort benchmark provenance.
        context["error"] = f"{type(exc).__name__}: {exc}"
    return context


def _hardware_context() -> dict[str, Any]:
    visible_device = _visible_hip_device_context()
    visible_device_name = visible_device.get("device_name")
    gpu_name = visible_device_name if isinstance(visible_device_name, str) and visible_device_name else "AMD Radeon Pro W7900"
    return {
        "gpu": gpu_name,
        "arch": "gfx1100",
        "default_hardware": gpu_name == "AMD Radeon Pro W7900",
        "visible_device": visible_device,
        "rocminfo": _run_capture(["bash", "-lc", "rocminfo | grep -E 'Name:|gfx' | head -4"], timeout=10.0),
        "rocm_smi": _run_capture(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp"], timeout=10.0),
    }


def _command(argv: Sequence[str] | None) -> str:
    parts = [*_command_env_prefix_parts(), "python3", "scripts/qwen35_batch_serial_bench.py"]
    parts.extend(sys.argv[1:] if argv is None else list(argv))
    return " ".join(shlex.quote(part) for part in parts)


def _compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text()


def _run_scheduler_serial_bench(
    runner: Qwen35ParoNextTokenRunner,
    prompts: list[list[int]],
    *,
    max_layers: int,
    warmup_decode_tokens: int,
    decode_tokens: int,
    compiler_version: str | None,
    require_cached_build: bool,
    kv_policy: ResolvedKVPolicy,
) -> dict[str, Any]:
    batch_size = len(prompts)
    prompt_lengths = {len(prompt) for prompt in prompts}
    if len(prompt_lengths) != 1:
        raise ValueError("current benchmark expects equal prompt lengths")
    prompt_length = prompt_lengths.pop()
    max_sequence_length = prompt_length + warmup_decode_tokens + decode_tokens + 1
    scheduler = ResidentBatchScheduler(capacity=batch_size)
    request_ids = [scheduler.submit(prompt, max_new_tokens=warmup_decode_tokens + decode_tokens) for prompt in prompts]
    admitted = scheduler.admit_pending()
    if admitted != tuple(request_ids):
        raise RuntimeError(f"unexpected admitted request ids {admitted!r}")

    seed_by_request: dict[int, Any] = {}
    generated_by_request: dict[int, list[dict[str, Any]]] = {request_id: [] for request_id in request_ids}
    measured_step_seconds: list[float] = []
    warmup_step_seconds: list[float] = []
    scheduler_metadata: dict[str, Any] = {
        "request_ids": list(request_ids),
        "admitted": list(admitted),
        "slot_to_request_after_admit": list(scheduler.active_batch.slot_to_request),
        "active_count_after_admit": scheduler.active_count,
        "prefill_work_items": 0,
        "prefill_request_order": [],
    }

    load_start = time.perf_counter()
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=max_layers,
        max_batch_size=batch_size,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    ) as session:
        load_seconds = time.perf_counter() - load_start
        batch_execution = session.batch_execution_metadata(scheduler_owned=True).to_json_dict()

        prefill_start = time.perf_counter()
        while True:
            work = scheduler.next_prefill_work(chunk_size=1)
            if work is None:
                break
            scheduler_metadata["prefill_work_items"] += 1
            request_id = work.request_ids[0]
            scheduler_metadata["prefill_request_order"].append(request_id)
            request = scheduler.active_batch.requests[request_id]
            token = work.token_rows[0][0]
            position = request.next_prompt_index - 1
            slot = scheduler.active_batch.slot_for(request_id)
            sample = request.remaining_prefill == 0
            result = session.step_batch_serial([token], positions=[position], slots=[slot], sample=sample)[0]
            if sample:
                if result is None:
                    raise RuntimeError("prefill did not produce a seed token")
                seed_by_request[request_id] = result
        prefill_seconds = time.perf_counter() - prefill_start

        if set(seed_by_request) != set(request_ids):
            raise RuntimeError("missing one or more prefill seed tokens")

        scheduler_metadata["slot_to_request_at_decode"] = list(scheduler.active_batch.slot_to_request)
        scheduler_metadata["active_count_at_decode"] = scheduler.active_count
        shape_key = scheduler.shape_key(
            mode="decode",
            top_k=8,
            experts_per_token=8,
            replay_steps=1,
            kv_storage_dtype=kv_policy.storage_dtype.value,
            layer_plan=f"max_layers={int(max_layers)}",
        )
        scheduler.graph_buckets.get_or_create(shape_key, _shape_key_payload)
        scheduler.graph_buckets.get(shape_key)
        stats = scheduler.graph_buckets.stats
        scheduler_metadata["decode_shape_key"] = _shape_key_payload(shape_key)
        scheduler_metadata["graph_bucket_stats"] = stats.to_json_dict()

        next_token_by_request = {request_id: seed_by_request[request_id].token_id for request_id in request_ids}
        warmup_start = time.perf_counter()
        for _ in range(warmup_decode_tokens):
            step_start = time.perf_counter()
            _decode_scheduler_step(session, scheduler, next_token_by_request, generated_by_request, count_output=False)
            warmup_step_seconds.append(time.perf_counter() - step_start)
        warmup_seconds = time.perf_counter() - warmup_start

        decode_start = time.perf_counter()
        for _ in range(decode_tokens):
            step_start = time.perf_counter()
            _decode_scheduler_step(session, scheduler, next_token_by_request, generated_by_request, count_output=True)
            measured_step_seconds.append(time.perf_counter() - step_start)
        decode_seconds = time.perf_counter() - decode_start
        completed = list(scheduler.completed.values())
        scheduler_metadata["active_count_after_completion"] = scheduler.active_count
        scheduler_metadata["slot_to_request_after_completion"] = list(scheduler.active_batch.slot_to_request)

    seed_rows = [_result_payload(seed_by_request[request_id]) for request_id in request_ids]
    generated_rows = [row for rows in generated_by_request.values() for row in rows]
    finite_logits = _all_finite(seed_rows) and _all_finite(generated_rows)
    return {
        "load_seconds": load_seconds,
        "prefill_seconds": prefill_seconds,
        "warmup_seconds": warmup_seconds,
        "decode_seconds": decode_seconds,
        "warmup_step_seconds": warmup_step_seconds,
        "decode_step_seconds": measured_step_seconds,
        "seed_tokens": {str(request_id): _result_payload(seed_by_request[request_id]) for request_id in request_ids},
        "generated_tokens": {str(request_id): generated_by_request[request_id] for request_id in request_ids},
        "scheduler_metadata": scheduler_metadata,
        "batch_execution": batch_execution,
        "completed": [
            {
                "request_id": done.request_id,
                "prompt_tokens": list(done.prompt_tokens),
                "generated_tokens": list(done.generated_tokens),
                "finished": done.finished,
            }
            for done in completed
        ],
        "finite_logits": finite_logits,
    }


def _decode_scheduler_step(
    session: Qwen35ParoResidentSession,
    scheduler: ResidentBatchScheduler,
    next_token_by_request: dict[int, int],
    generated_by_request: dict[int, list[dict[str, Any]]],
    *,
    count_output: bool,
) -> None:
    work = scheduler.next_decode_work()
    if work is None:
        raise RuntimeError("scheduler did not emit decode work")
    request_ids = work.request_ids
    results = session.step_batch_serial(
        [next_token_by_request[request_id] for request_id in request_ids],
        positions=[scheduler.active_batch.requests[request_id].context_len for request_id in request_ids],
        slots=[scheduler.active_batch.slot_for(request_id) for request_id in request_ids],
        sample=True,
    )
    generated: list[GeneratedToken] = []
    for request_id, result in zip(request_ids, results, strict=True):
        if result is None:
            raise RuntimeError("decode did not produce a token")
        next_token_by_request[request_id] = result.token_id
        if count_output:
            generated_by_request[request_id].append(_result_payload(result))
        generated.append(GeneratedToken(request_id, result.token_id))
    scheduler.record_generated(generated)


def _build_payload(args: argparse.Namespace, argv: Sequence[str] | None, bench: dict[str, Any], prompt_lengths: list[int]) -> dict[str, Any]:
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    aggregate_prefill_tokens = args.batch_size * args.prompt_length
    aggregate_decode_tokens = args.batch_size * args.decode_tokens
    prefill_tok_s = aggregate_prefill_tokens / bench["prefill_seconds"] if bench["prefill_seconds"] > 0 else None
    decode_tok_s = aggregate_decode_tokens / bench["decode_seconds"] if bench["decode_seconds"] > 0 and aggregate_decode_tokens else None
    throughput_claim_eligible = bool(bench["batch_execution"].get("throughput_claim_eligible"))
    accepted = bool(bench["finite_logits"] and throughput_claim_eligible and args.max_layers == 40 and args.prompt_length >= 512 and args.decode_tokens >= 128)
    blocked_reasons = []
    if not throughput_claim_eligible:
        blocked_reasons.append("batch_execution.throughput_claim_eligible=false: current c>N path is scheduler_serial_slot_bridge")
    if args.prompt_length < 512 or args.decode_tokens < 128:
        blocked_reasons.append("workload is a reduced diagnostic shape, not the docs/BENCHMARK.md c=N 512/128 protocol")
    if args.max_layers != 40:
        blocked_reasons.append("max_layers is not the full 40-layer Qwen3.5/PARO model")
    if not bench["finite_logits"]:
        blocked_reasons.append("non-finite seed or decode logits")
    payload = {
        "schema": 2,
        "status": "accepted" if accepted else "blocked",
        "artifact_path": str(args.json) if args.json is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_tag": f"qwen35-paro-c{args.batch_size}-scheduler-serial-bridge",
        "summary": "Qwen3.5/PARO scheduler serial c>N bridge diagnostic benchmark",
        "performance_claim": accepted,
        "hardware": _hardware_context(),
        "software": _software_context(),
        "workload": {
            "shape": f"c={args.batch_size} prompt={args.prompt_length} decode={args.decode_tokens}",
            "model": "Qwen3.5-35B-A3B-PARO",
            "model_path": str(Path(args.model)),
            "quant": "w4_paro",
            "prompt_tokens_per_request": args.prompt_length,
            "prompt_tokens_aggregate": aggregate_prefill_tokens,
            "gen_tokens_per_request": args.decode_tokens,
            "gen_tokens_aggregate": aggregate_decode_tokens,
            "warmup_decode_tokens": args.warmup_decode_tokens,
            "concurrency": args.batch_size,
            "prompt_lengths": prompt_lengths,
            "max_layers": args.max_layers,
            "kv_policy": kv_policy_json(kv_policy),
            "kv_storage_dtype": kv_policy.storage_dtype.value,
            "scheduler_path": "scheduler_serial_slot_bridge",
            "native_compact_prefill": False,
            "native_caware_decode": False,
        },
        "commands": {
            "environment": [
                "rocminfo | grep -E 'Name:|gfx' | head -4",
                "rocm-smi --showmeminfo vram --showuse --showtemp",
                "hipcc --version",
            ],
            "correctness_reference": "benchmarks/results/2026-05-15-hipengine-qwen35-cn-generated-equality-accepted.json",
            "benchmark": _command(argv),
            "profiler": None,
        },
        "correctness": {
            "passed": bool(bench["finite_logits"]),
            "oracle": "generated-token finite-logit smoke plus prior scheduler generated-equality artifact",
            "finite_logits": bool(bench["finite_logits"]),
            "generated_equality_artifact": "benchmarks/results/2026-05-15-hipengine-qwen35-cn-generated-equality-accepted.json",
            "kl_mean": None,
            "top1_agreement": None,
        },
        "execution": {
            "batch_execution": bench["batch_execution"],
            "scheduler_metadata": bench["scheduler_metadata"],
            "completed": bench["completed"],
            "seed_tokens": bench["seed_tokens"],
            "generated_tokens": bench["generated_tokens"],
        },
        "measurements": {
            "load_seconds": bench["load_seconds"],
            "prefill_seconds": bench["prefill_seconds"],
            "warmup_decode_seconds": bench["warmup_seconds"],
            "decode_seconds": bench["decode_seconds"],
            "prefill_tok_s": prefill_tok_s,
            "decode_tok_s_aggregate": decode_tok_s,
            "decode_tok_s_per_request": decode_tok_s / args.batch_size if decode_tok_s is not None else None,
            "decode_step_seconds": _summarize_samples(bench["decode_step_seconds"]),
            "warmup_step_seconds": _summarize_samples(bench["warmup_step_seconds"]),
        },
        "memory": {
            "max_batch_size": args.batch_size,
            "max_sequence_length": args.prompt_length + args.warmup_decode_tokens + args.decode_tokens + 1,
            "kv_policy": kv_policy_json(kv_policy),
            "kv_storage_dtype": kv_policy.storage_dtype.value,
            "allocator_reserved_peak_bytes": None,
        },
        "profiler": {"status": "not_captured", "notes": "No kernel port or retained performance claim in this diagnostic iteration."},
        "decision": {
            "accepted": accepted,
            "reason": "correctness/protocol passed" if accepted else "; ".join(blocked_reasons),
        },
        "notes": [
            "Diagnostic benchmark only; timings are recorded to characterize the current bridge but are not retained as throughput claims unless decision.accepted=true.",
            "The c>N path uses step_batch_serial over batch-shaped state/KV slots and is blocked on native compact prefill plus c-aware decode kernels.",
        ],
    }
    validate_cn_diagnostic_artifact_payload(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--warmup-decode-tokens", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    add_kv_policy_args(parser, help_prefix="Resident KV storage for scheduler serial benchmark")
    parser.add_argument("--json", type=Path, help="Optional path to write JSON output")
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.decode_tokens < 0 or args.warmup_decode_tokens < 0:
        raise ValueError("decode token counts must be non-negative")
    if args.decode_tokens == 0 and args.warmup_decode_tokens == 0:
        raise ValueError("at least one decode or warmup decode token is required")
    if args.max_layers <= 0:
        raise ValueError("--max-layers must be positive")

    prompts = _load_prompt_slices(Path(args.fixture), prompt_length=args.prompt_length, batch_size=args.batch_size)
    runner = Qwen35ParoNextTokenRunner(Path(args.model))
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    bench = _run_scheduler_serial_bench(
        runner,
        prompts,
        max_layers=args.max_layers,
        warmup_decode_tokens=args.warmup_decode_tokens,
        decode_tokens=args.decode_tokens,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached_build=args.require_cached_build,
        kv_policy=kv_policy,
    )
    payload = _build_payload(args, argv, bench, [len(prompt) for prompt in prompts])
    text = _payload_json(payload)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0 if payload["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
