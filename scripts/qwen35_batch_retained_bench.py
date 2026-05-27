#!/usr/bin/env python3
"""Retained Qwen3.5/PARO compact c>N benchmark.

This is the accepted-path companion to ``qwen35_batch_serial_bench.py``.  It
uses scheduler-owned compact native prefill plus ``step_batch_native`` decode,
then (unless skipped) compares generated token ids against independent c=1
resident runs before marking a row accepted.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
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
from scripts.qwen35_batch_artifact_schema import validate_cn_diagnostic_artifact_payload
from scripts.qwen35_kv_policy_args import add_kv_policy_args, kv_policy_json, resolve_args_kv_policy

DEFAULT_MODEL = "/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16"
DEFAULT_FIXTURE = "fixtures/qwen35_paro/parent_512_32_seed1234.json"


def _load_prompt_slices(path: Path, *, prompt_length: int, batch_size: int) -> list[list[int]]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    fixture = json.loads(path.read_text())
    tokens = [int(token) for token in fixture["prompt_ids"]]
    needed = int(prompt_length) * int(batch_size)
    if len(tokens) < needed:
        raise ValueError(f"fixture contains {len(tokens)} tokens, need at least {needed}")
    return [tokens[row * prompt_length : (row + 1) * prompt_length] for row in range(batch_size)]


def _result_payload(result) -> dict[str, Any]:
    return {"token_id": int(result.token_id), "token_text": result.token_text, "logit": float(result.logit)}


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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not (_is_number(numerator) and _is_number(denominator)):
        return None
    denom = float(denominator)
    if denom <= 0.0:
        return None
    return float(numerator) / denom


def _extract_decode_rates(payload: Mapping[str, Any]) -> tuple[float | None, float | None]:
    measurements = payload.get("measurements")
    aggregate = None
    per_request = None
    if isinstance(measurements, Mapping):
        if _is_number(measurements.get("decode_tok_s_aggregate")):
            aggregate = float(measurements["decode_tok_s_aggregate"])
        if _is_number(measurements.get("decode_tok_s_per_request")):
            per_request = float(measurements["decode_tok_s_per_request"])
    throughput = payload.get("throughput")
    if isinstance(throughput, Mapping) and _is_number(throughput.get("warmed_decode_tok_s")):
        aggregate = float(throughput["warmed_decode_tok_s"])
        per_request = float(throughput["warmed_decode_tok_s"])
    workload = payload.get("workload")
    if aggregate is not None and per_request is None and isinstance(workload, Mapping):
        concurrency = workload.get("concurrency")
        if isinstance(concurrency, int) and concurrency > 0:
            per_request = aggregate / concurrency
    return aggregate, per_request


def _scaling_reference(path: Path | None, *, default_workload_concurrency: int | None = None) -> dict[str, Any]:
    if path is None:
        return {
            "artifact_path": None,
            "status": "missing",
            "decode_tok_s_aggregate": None,
            "decode_tok_s_per_request": None,
            "reason": "no artifact path provided",
        }
    path = Path(path)
    if not path.exists():
        return {
            "artifact_path": str(path),
            "status": "missing",
            "decode_tok_s_aggregate": None,
            "decode_tok_s_per_request": None,
            "reason": "artifact path does not exist",
        }
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return {
            "artifact_path": str(path),
            "status": "invalid_json",
            "decode_tok_s_aggregate": None,
            "decode_tok_s_per_request": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, Mapping):
        return {
            "artifact_path": str(path),
            "status": "invalid_json",
            "decode_tok_s_aggregate": None,
            "decode_tok_s_per_request": None,
            "reason": "artifact root is not an object",
        }
    aggregate, per_request = _extract_decode_rates(payload)
    workload = payload.get("workload")
    workload_concurrency = None
    prompt_tokens_per_request = None
    gen_tokens_per_request = None
    if isinstance(workload, Mapping):
        concurrency = workload.get("concurrency")
        if isinstance(concurrency, int) and not isinstance(concurrency, bool):
            workload_concurrency = concurrency
        prompt_tokens = workload.get("prompt_tokens_per_request", workload.get("prompt_length"))
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
            prompt_tokens_per_request = prompt_tokens
        gen_tokens = workload.get("gen_tokens_per_request", workload.get("decode_tokens"))
        if isinstance(gen_tokens, int) and not isinstance(gen_tokens, bool):
            gen_tokens_per_request = gen_tokens
    if prompt_tokens_per_request is None:
        prompt_tokens = payload.get("prompt_length")
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
            prompt_tokens_per_request = prompt_tokens
    if gen_tokens_per_request is None:
        gen_tokens = payload.get("decode_tokens")
        if isinstance(gen_tokens, int) and not isinstance(gen_tokens, bool):
            gen_tokens_per_request = gen_tokens
    if workload_concurrency is None and default_workload_concurrency is not None:
        workload_concurrency = int(default_workload_concurrency)
    status = str(payload.get("status") or "loaded")
    reason = None if aggregate is not None and per_request is not None else "decode throughput fields missing"
    return {
        "artifact_path": str(path),
        "status": status,
        "run_tag": payload.get("run_tag"),
        "workload_concurrency": workload_concurrency,
        "prompt_tokens_per_request": prompt_tokens_per_request,
        "gen_tokens_per_request": gen_tokens_per_request,
        "decode_tok_s_aggregate": aggregate,
        "decode_tok_s_per_request": per_request,
        "reason": reason,
    }


def _primitive_correctness_reference(path: Path | None, *, rows: int) -> dict[str, Any]:
    if path is None:
        return {
            "artifact_path": None,
            "status": "missing",
            "passed": False,
            "reason": "no primitive correctness artifact path provided",
        }
    path = Path(path)
    if not path.exists():
        return {
            "artifact_path": str(path),
            "status": "missing",
            "passed": False,
            "reason": "artifact path does not exist",
        }
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return {
            "artifact_path": str(path),
            "status": "invalid_json",
            "passed": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, Mapping):
        return {
            "artifact_path": str(path),
            "status": "invalid_json",
            "passed": False,
            "reason": "artifact root is not an object",
        }
    reasons: list[str] = []
    artifact_rows = payload.get("rows")
    if not isinstance(artifact_rows, int) or isinstance(artifact_rows, bool) or artifact_rows != int(rows):
        reasons.append(f"artifact rows={artifact_rows!r} does not match batch_size={rows}")
    if payload.get("passed") is not True:
        reasons.append("primitive correctness payload did not pass")
    if payload.get("append_key_mismatch") != 0:
        reasons.append("append_key_mismatch is non-zero")
    if payload.get("append_value_mismatch") != 0:
        reasons.append("append_value_mismatch is non-zero")
    attn_vs_c1 = payload.get("attn_batch_vs_c1_max_abs")
    if not _is_number(attn_vs_c1) or float(attn_vs_c1) > 1e-6:
        reasons.append("attn_batch_vs_c1_max_abs is missing or above 1e-6")
    return {
        "artifact_path": str(path),
        "status": "loaded",
        "rows": payload.get("rows"),
        "passed": not reasons,
        "append_key_mismatch": payload.get("append_key_mismatch"),
        "append_value_mismatch": payload.get("append_value_mismatch"),
        "attn_batch_vs_c1_max_abs": attn_vs_c1,
        "attn_batch_vs_numpy_max_abs": payload.get("attn_batch_vs_numpy_max_abs"),
        "reason": None if not reasons else "; ".join(reasons),
    }


def _build_scaling_comparison(
    args: argparse.Namespace,
    *,
    native_decode_tok_s_aggregate: float | None,
    native_decode_tok_s_per_request: float | None,
) -> dict[str, Any]:
    c1 = _scaling_reference(getattr(args, "c1_baseline_json", None), default_workload_concurrency=1)
    serial = _scaling_reference(getattr(args, "serial_bridge_json", None))
    ratios = {
        "aggregate_vs_c1": _safe_ratio(native_decode_tok_s_aggregate, c1.get("decode_tok_s_aggregate")),
        "per_request_vs_c1": _safe_ratio(native_decode_tok_s_per_request, c1.get("decode_tok_s_per_request")),
        "aggregate_vs_serial_bridge": _safe_ratio(native_decode_tok_s_aggregate, serial.get("decode_tok_s_aggregate")),
        "per_request_vs_serial_bridge": _safe_ratio(native_decode_tok_s_per_request, serial.get("decode_tok_s_per_request")),
    }
    complete = all(value is not None for value in ratios.values())
    return {
        "complete": complete,
        "native": {
            "decode_tok_s_aggregate": native_decode_tok_s_aggregate,
            "decode_tok_s_per_request": native_decode_tok_s_per_request,
        },
        "c1_baseline": c1,
        "serial_bridge_baseline": serial,
        "ratios": ratios,
    }


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
        return {
            "command": " ".join(shlex.quote(part) for part in command),
            "returncode": proc.returncode,
            "output": proc.stdout.strip(),
        }
    except Exception as exc:  # pragma: no cover - best-effort environment capture
        return {
            "command": " ".join(shlex.quote(part) for part in command),
            "returncode": None,
            "output": f"{type(exc).__name__}: {exc}",
        }


def _software_context() -> dict[str, Any]:
    commit = _run_capture(["git", "rev-parse", "HEAD"])
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


def _hardware_context() -> dict[str, Any]:
    return {
        "gpu": "AMD Radeon Pro W7900",
        "arch": "gfx1100",
        "default_hardware": True,
        "rocminfo": _run_capture(["bash", "-lc", "rocminfo | grep -E 'Name:|gfx' | head -4"], timeout=10.0),
        "rocm_smi": _run_capture(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp"], timeout=10.0),
    }


def _command(argv: Sequence[str] | None) -> str:
    parts = ["python3", "scripts/qwen35_batch_retained_bench.py"]
    parts.extend(sys.argv[1:] if argv is None else list(argv))
    return " ".join(shlex.quote(part) for part in parts)


def _primitive_correctness_command(path: Path | None, *, rows: int) -> str:
    parts = ["python3", "scripts/qwen35_batch_correctness.py", "--rows", str(rows), "--json"]
    parts.append(str(path) if path is not None else "<primitive-correctness-json>")
    return " ".join(shlex.quote(part) for part in parts)


def _compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text()


def _decode_scheduler_step_native(
    session: Qwen35ParoResidentSession,
    scheduler: ResidentBatchScheduler,
    next_token_by_request: dict[int, int],
    generated_by_request: dict[int, list[dict[str, Any]]],
    *,
    count_output: bool,
) -> tuple[int, bool]:
    work = scheduler.next_decode_work()
    if work is None:
        raise RuntimeError("scheduler did not emit decode work")
    request_ids = tuple(request_id for request_id in work.request_ids if request_id in next_token_by_request)
    slots = [scheduler.active_batch.slot_for(request_id) for request_id in request_ids]
    if tuple(slots) != tuple(range(len(slots))):
        raise RuntimeError(f"native retained benchmark requires compact slots, got {slots!r}")
    results = session.step_batch_native(
        [next_token_by_request[request_id] for request_id in request_ids],
        positions=[scheduler.active_batch.requests[request_id].context_len for request_id in request_ids],
        slots=slots,
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
    return len(results), True


def _run_native_bench(
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
        "prefill_slabs": [],
        "decode_native_steps": 0,
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

        prefill_start = time.perf_counter()
        slabs = scheduler.next_compact_prefill_slabs(chunk_size=prompt_length, block_size=session.block_size)
        for slab in slabs:
            scheduler_metadata["prefill_slabs"].append(
                {
                    "request_ids": list(slab.request_ids),
                    "slot_ids": list(slab.physical_slot_ids),
                    "rows": slab.rows,
                    "request_count": slab.request_count,
                    "block_count": slab.block_count,
                }
            )
            results = session.prefill_native_packed(slab, sample=True)
            for request_id, result in zip(slab.request_ids, results, strict=True):
                if result is None:
                    raise RuntimeError("prefill did not produce a seed token")
                seed_by_request[request_id] = result
        prefill_seconds = time.perf_counter() - prefill_start

        if set(seed_by_request) != set(request_ids):
            raise RuntimeError("missing one or more prefill seed tokens")

        next_token_by_request = {request_id: seed_by_request[request_id].token_id for request_id in request_ids}
        warmup_start = time.perf_counter()
        for _ in range(warmup_decode_tokens):
            step_start = time.perf_counter()
            _count, native = _decode_scheduler_step_native(
                session,
                scheduler,
                next_token_by_request,
                generated_by_request,
                count_output=False,
            )
            scheduler_metadata["decode_native_steps"] += int(native)
            warmup_step_seconds.append(time.perf_counter() - step_start)
        warmup_seconds = time.perf_counter() - warmup_start

        decode_start = time.perf_counter()
        for _ in range(decode_tokens):
            step_start = time.perf_counter()
            _count, native = _decode_scheduler_step_native(
                session,
                scheduler,
                next_token_by_request,
                generated_by_request,
                count_output=True,
            )
            scheduler_metadata["decode_native_steps"] += int(native)
            measured_step_seconds.append(time.perf_counter() - step_start)
        decode_seconds = time.perf_counter() - decode_start
        completed = list(scheduler.completed.values())
        scheduler_metadata["active_count_after_completion"] = scheduler.active_count
        scheduler_metadata["slot_to_request_after_completion"] = list(scheduler.active_batch.slot_to_request)
        batch_execution = session.batch_execution_metadata(scheduler_owned=True, native_decode=True).to_json_dict()

    completed_payload = [done.to_json_dict() for done in completed]
    request_observability = {
        str(done.request_id): done.observability.to_json_dict()
        for done in completed
    }
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
        "completed": completed_payload,
        "request_observability": request_observability,
        "finite_logits": finite_logits,
    }


def _run_c1_reference_tokens(
    runner: Qwen35ParoNextTokenRunner,
    prompts: list[list[int]],
    *,
    total_decode_tokens: int,
    max_layers: int,
    max_sequence_length: int,
    compiler_version: str | None,
    require_cached_build: bool,
    kv_policy: ResolvedKVPolicy,
) -> list[list[int]]:
    rows: list[list[int]] = []
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=max_layers,
        max_batch_size=1,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=kv_policy.create_policy(),
        kv_scale_dtype=kv_policy.scale_dtype,
        kv_scale_granularity=kv_policy.scale_granularity,
    ) as session:
        for prompt in prompts:
            scheduler = ResidentBatchScheduler(capacity=1)
            request_id = scheduler.submit(prompt, max_new_tokens=total_decode_tokens)
            admitted = scheduler.admit_pending()
            if admitted != (request_id,):
                raise RuntimeError(f"unexpected c=1 admitted request ids {admitted!r}")
            slabs = scheduler.next_compact_prefill_slabs(chunk_size=len(prompt), block_size=session.block_size)
            if len(slabs) != 1:
                raise RuntimeError("c=1 reference expected one compact prefill slab")
            seed = session.prefill_native_packed(slabs[0], sample=True)[0]
            if seed is None:
                raise RuntimeError("c=1 prefill did not produce a seed token")
            token_ids = [int(seed.token_id)]
            next_token = int(seed.token_id)
            for offset in range(total_decode_tokens):
                result = session.step_batch_native(
                    [next_token],
                    positions=[len(prompt) + offset],
                    slots=[0],
                    sample=True,
                )[0]
                if result is None:
                    raise RuntimeError("c=1 decode did not produce a token")
                next_token = int(result.token_id)
                token_ids.append(next_token)
            rows.append(token_ids)
            session.reset()
    return rows


def _generated_sequences_from_bench(bench: dict[str, Any], request_ids: Sequence[int]) -> list[list[int]]:
    rows: list[list[int]] = []
    completed_by_id = {int(row["request_id"]): row for row in bench.get("completed", [])}
    for request_id in request_ids:
        seed = int(bench["seed_tokens"][str(request_id)]["token_id"])
        if request_id in completed_by_id:
            generated = [int(token) for token in completed_by_id[request_id]["generated_tokens"]]
        else:
            generated = [int(item["token_id"]) for item in bench["generated_tokens"][str(request_id)]]
        rows.append([seed, *generated])
    return rows


def _build_payload(
    args: argparse.Namespace,
    argv: Sequence[str] | None,
    bench: dict[str, Any],
    prompt_lengths: list[int],
    equality: dict[str, Any],
) -> dict[str, Any]:
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    aggregate_prefill_tokens = args.batch_size * args.prompt_length
    aggregate_decode_tokens = args.batch_size * args.decode_tokens
    prefill_tok_s = aggregate_prefill_tokens / bench["prefill_seconds"] if bench["prefill_seconds"] > 0 else None
    decode_tok_s = aggregate_decode_tokens / bench["decode_seconds"] if bench["decode_seconds"] > 0 and aggregate_decode_tokens else None
    decode_tok_s_per_request = decode_tok_s / args.batch_size if decode_tok_s is not None else None
    scaling = _build_scaling_comparison(
        args,
        native_decode_tok_s_aggregate=decode_tok_s,
        native_decode_tok_s_per_request=decode_tok_s_per_request,
    )
    primitive_correctness_path = getattr(args, "primitive_correctness_json", None)
    primitive_correctness = _primitive_correctness_reference(
        primitive_correctness_path,
        rows=args.batch_size,
    )
    correctness_reference_command = _primitive_correctness_command(primitive_correctness_path, rows=args.batch_size)
    profiler = {"status": "not_captured", "notes": "E2E retained c>N row; profiler trace not captured in this iteration."}
    profiler_captured = profiler.get("status") == "captured" and profiler.get("expected_kernels_present") is True
    batch_execution = dict(bench["batch_execution"])
    throughput_claim_eligible = bool(batch_execution.get("throughput_claim_eligible"))
    native_caware_decode = bool(batch_execution.get("native_caware_decode"))
    equality_passed = bool(equality.get("passed"))
    protocol_shape = args.max_layers == 40 and args.prompt_length >= 512 and args.decode_tokens >= 128
    scaling_complete = bool(scaling["complete"])
    primitive_passed = bool(primitive_correctness["passed"])
    accepted = bool(
        bench["finite_logits"]
        and throughput_claim_eligible
        and equality_passed
        and primitive_passed
        and protocol_shape
        and scaling_complete
        and profiler_captured
    )
    primitive_loaded = primitive_correctness.get("status") == "loaded"
    correctness_rejected = bool(bench["finite_logits"] and (not equality_passed or (primitive_loaded and not primitive_passed)))
    status = "accepted" if accepted else ("rejected_correctness" if correctness_rejected else "blocked")
    blocked_reasons: list[str] = []
    if not throughput_claim_eligible:
        blocked_reasons.append("batch_execution.throughput_claim_eligible=false")
    if not equality_passed:
        blocked_reasons.append("generated-token equality vs independent c=1 did not pass")
    if not primitive_passed:
        blocked_reasons.append(f"primitive c>N correctness gate did not pass: {primitive_correctness.get('reason')}")
    if args.prompt_length < 512 or args.decode_tokens < 128:
        blocked_reasons.append("workload is a reduced diagnostic shape, not the docs/BENCHMARK.md c=N 512/128 protocol")
    if args.max_layers != 40:
        blocked_reasons.append("max_layers is not the full 40-layer Qwen3.5/PARO model")
    if not scaling_complete:
        blocked_reasons.append("scaling comparison vs c=1 and serial bridge baselines is incomplete")
    if not profiler_captured:
        blocked_reasons.append("profiler trace was not captured with expected kernels present")
    if not bench["finite_logits"]:
        blocked_reasons.append("non-finite seed or decode logits")
    per_request_observability = dict(bench.get("request_observability", {}))
    admission_timestamps = {
        request_id: row.get("admitted_timestamp")
        for request_id, row in per_request_observability.items()
        if isinstance(row, dict)
    }
    completion_timestamps = {
        request_id: row.get("completion_timestamp")
        for request_id, row in per_request_observability.items()
        if isinstance(row, dict)
    }
    request_latencies = [
        float(row["completion_timestamp"]) - float(row["submitted_timestamp"])
        for row in per_request_observability.values()
        if isinstance(row, dict)
        and row.get("completion_timestamp") is not None
        and row.get("submitted_timestamp") is not None
    ]
    latency_summary = _summarize_samples(request_latencies)
    payload = {
        "schema": 3,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_tag": f"qwen35-paro-c{args.batch_size}-native-retained",
        "summary": "Qwen3.5/PARO scheduler compact native c>N benchmark",
        "performance_claim": accepted,
        "hardware": _hardware_context(),
        "software": _software_context(),
        "workload": {
            "shape": f"c={args.batch_size} prompt={args.prompt_length} decode={args.decode_tokens}",
            "model": "Qwen3.5/3.6-35B-A3B-PARO",
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
            "scheduler_path": "scheduler_native_compact_batch",
            "native_compact_prefill": True,
            "native_caware_decode": native_caware_decode,
        },
        "commands": {
            "environment": [
                "rocminfo | grep -E 'Name:|gfx' | head -4",
                "rocm-smi --showmeminfo vram --showuse --showtemp",
                "hipcc --version",
            ],
            "correctness_reference": f"inline generated-token equality vs independent c=1 plus {correctness_reference_command}",
            "benchmark": _command(argv),
            "profiler": None,
        },
        "correctness": {
            "passed": bool(bench["finite_logits"] and equality_passed and primitive_passed),
            "oracle": "generated-token ids equal independent c=1 resident runs through the same native packed prefill/decode path plus scripts/qwen35_batch_correctness.py primitive GPU correctness for the same c>N row count",
            "finite_logits": bool(bench["finite_logits"]),
            "generated_token_equality": equality,
            "primitive_batch_correctness": primitive_correctness,
            "kl_mean": None,
            "top1_agreement": None,
        },
        "execution": {
            "batch_execution": batch_execution,
            "scheduler_metadata": bench["scheduler_metadata"],
            "completed": bench["completed"],
            "seed_tokens": bench["seed_tokens"],
            "generated_tokens": bench["generated_tokens"],
        },
        "observability": {
            "admission_timestamps": admission_timestamps,
            "completion_timestamps": completion_timestamps,
            "request_latency_seconds": {
                "p50": latency_summary["median"],
                "p95": latency_summary["p95"],
                "samples": latency_summary["samples"],
            },
            "per_request": per_request_observability,
        },
        "measurements": {
            "load_seconds": bench["load_seconds"],
            "prefill_seconds": bench["prefill_seconds"],
            "warmup_decode_seconds": bench["warmup_seconds"],
            "decode_seconds": bench["decode_seconds"],
            "prefill_tok_s": prefill_tok_s,
            "decode_tok_s_aggregate": decode_tok_s,
            "decode_tok_s_per_request": decode_tok_s_per_request,
            "decode_step_seconds": _summarize_samples(bench["decode_step_seconds"]),
            "warmup_step_seconds": _summarize_samples(bench["warmup_step_seconds"]),
        },
        "scaling": scaling,
        "memory": {
            "max_batch_size": args.batch_size,
            "max_sequence_length": args.prompt_length + args.warmup_decode_tokens + args.decode_tokens + 1,
            "kv_policy": kv_policy_json(kv_policy),
            "kv_storage_dtype": kv_policy.storage_dtype.value,
            "allocator_reserved_peak_bytes": None,
            "dynamic_pool": {
                "enabled": False,
                "evidence": "resident retained bench still uses fixed session allocation; C4 pool counters are unavailable here",
                "pool_counters": {
                    "current_bytes": 0,
                    "high_water_observed_bytes": 0,
                    "grow_events": 0,
                    "grow_failures": 0,
                    "shrink_events": 0,
                    "free_pages": 0,
                    "refcounted_pages": 0,
                },
            },
            "stable_block_id": {"passed": False, "audit": "not captured in retained bench"},
            "prefix_sharing": {"enabled": False, "savings_bytes": 0},
        },
        "profiler": profiler,
        "decision": {
            "accepted": accepted,
            "reason": "correctness/protocol passed" if accepted else "; ".join(blocked_reasons),
        },
        "notes": [
            "Native retained c>N path uses packed prompt slabs and step_batch_native for decode.",
            "Batch split-K decode remains out of scope; this accepted protocol keeps context < 1024.",
        ],
    }
    validate_cn_diagnostic_artifact_payload(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--skip-generated-equality", action="store_true")
    parser.add_argument("--c1-baseline-json", type=Path, help="c=1 baseline artifact used for retained scaling ratios")
    parser.add_argument("--serial-bridge-json", type=Path, help="scheduler serial-bridge artifact for retained scaling ratios")
    parser.add_argument("--primitive-correctness-json", type=Path, help="scripts/qwen35_batch_correctness.py JSON for this c>N row count")
    add_kv_policy_args(parser, help_prefix="Resident KV storage for retained native c>N benchmark")
    parser.add_argument("--json", type=Path, help="Optional path to write JSON output")
    args = parser.parse_args(argv)

    if args.batch_size <= 1:
        raise ValueError("--batch-size must be greater than 1 for retained c>N")
    if args.decode_tokens <= 0 or args.warmup_decode_tokens < 0:
        raise ValueError("decode token counts must be positive/non-negative")
    if args.max_layers <= 0:
        raise ValueError("--max-layers must be positive")

    prompts = _load_prompt_slices(Path(args.fixture), prompt_length=args.prompt_length, batch_size=args.batch_size)
    runner = Qwen35ParoNextTokenRunner(Path(args.model))
    kv_policy = resolve_args_kv_policy(args, block_size=256)
    compiler_version = _compiler_version(args.compiler_version_file)
    os.environ["HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE"] = "1"
    bench = _run_native_bench(
        runner,
        prompts,
        max_layers=args.max_layers,
        warmup_decode_tokens=args.warmup_decode_tokens,
        decode_tokens=args.decode_tokens,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        kv_policy=kv_policy,
    )

    request_ids = list(range(args.batch_size))
    batch_sequences = _generated_sequences_from_bench(bench, request_ids)
    if args.skip_generated_equality:
        equality = {
            "passed": False,
            "skipped": True,
            "reason": "--skip-generated-equality was provided",
            "batch_sequences": batch_sequences,
            "c1_sequences": None,
        }
    else:
        c1_sequences = _run_c1_reference_tokens(
            runner,
            prompts,
            total_decode_tokens=args.warmup_decode_tokens + args.decode_tokens,
            max_layers=args.max_layers,
            max_sequence_length=args.prompt_length + args.warmup_decode_tokens + args.decode_tokens + 1,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            kv_policy=kv_policy,
        )
        equality = {
            "passed": batch_sequences == c1_sequences,
            "skipped": False,
            "batch_sequences": batch_sequences,
            "c1_sequences": c1_sequences,
            "mismatches": [
                {"row": row, "batch": batch_sequences[row], "c1": c1_sequences[row]}
                for row in range(args.batch_size)
                if batch_sequences[row] != c1_sequences[row]
            ],
        }

    payload = _build_payload(args, argv, bench, [len(prompt) for prompt in prompts], equality)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0 if payload["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
