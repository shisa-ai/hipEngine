#!/usr/bin/env python3
"""Recertify Maple native prefill on a registered public backend path.

The correctness phase compares serial and native prefill over the complete
code/general-English/general-Japanese/mixed prompt suite plus heldouts. It checks
byte-exact hidden/KV/span state, full-logit KL/top-1 at the prefill seed, and
subsequent decode positions. The performance phase measures repeated native and
serial rows at fixed 128/320/512 shapes derived from those natural prompts and
records exact tracked lifecycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import (  # noqa: E402
    DeviceBuffer,
    copy_device_to_host,
    host_array_ptr,
    memory_stats,
    reset_memory_stats,
)
from hipengine.kernels.backends import (  # noqa: E402
    backend_package_capability,
    cuda_target_arch_for_backend,
)
from hipengine.loading.maple import load_maple_checkpoint  # noqa: E402
from hipengine.runtime.maple import PREFILL_CHUNK, MapleRunner  # noqa: E402
from hipengine.tokenization.maple import MapleTokenizer  # noqa: E402

DEFAULT_MODEL = "deepgrove/maple-preview-2bit-mlx"
DEFAULT_SUITE = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
DEFAULT_HELDOUT = REPO_ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"
REQUIRED_CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")
PINNED_REVISION = "361db5da5e74ff6fcdd852d478e1f266ce11013a"


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


def _load_suite(path: Path, *, heldout: bool) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages") or []
        users = [message["content"] for message in messages if message.get("role") == "user"]
        if len(users) != 1:
            raise ValueError(f"{path}: prompt {row.get('id')!r} must have one user message")
        category = str(row["category"])
        if category not in REQUIRED_CATEGORIES:
            raise ValueError(f"{path}: unsupported category {category!r}")
        prompts.append(
            {
                "id": str(row["id"]),
                "category": category,
                "text": str(users[0]),
                "heldout": bool(heldout),
            }
        )
    return prompts


def _tokenizer(checkpoint) -> MapleTokenizer:
    spec = checkpoint.spec
    return MapleTokenizer.from_model_path(
        checkpoint.index.model_path,
        model_vocab_size=spec.vocab_size,
        eos_token_id=spec.eos_token_id,
        bos_token_id=spec.bos_token_id,
    )


def _kl_divergence(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64, copy=False)
    cand = candidate.astype(np.float64, copy=False)
    ref_log_z = float(np.max(ref)) + math.log(float(np.exp(ref - np.max(ref)).sum()))
    cand_log_z = float(np.max(cand)) + math.log(float(np.exp(cand - np.max(cand)).sum()))
    ref_log_p = ref - ref_log_z
    cand_log_p = cand - cand_log_z
    probability = np.exp(ref_log_p)
    return float(np.sum(probability * (ref_log_p - cand_log_p)))


def _maple_runner_type(backend: str):
    capability = backend_package_capability(
        backend,
        "maple_runner_type",
        default=None,
    )
    return MapleRunner if capability is None else capability()


def _copy_native_final_logits(runner: MapleRunner) -> np.ndarray:
    return runner.copy_logits()


def _copy_device_bytes(
    runner: MapleRunner,
    source: DeviceBuffer,
    *,
    offset: int = 0,
    nbytes: int | None = None,
) -> np.ndarray:
    size = source.nbytes - int(offset) if nbytes is None else int(nbytes)
    if offset < 0 or size < 0 or offset + size > source.nbytes:
        raise ValueError("Maple state copy exceeds device buffer bounds")
    host = np.empty(size, dtype=np.uint8)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(ptr=source.ptr + int(offset), nbytes=size),
        nbytes=size,
        runtime=runner.runtime,
    )
    return host


def _prefill_state_gate(
    serial: MapleRunner,
    native: MapleRunner,
    *,
    prompt_length: int,
) -> dict[str, Any]:
    spec = serial.checkpoint.spec
    if native.checkpoint.spec != spec:
        raise ValueError("Maple state comparison requires the same model spec")
    serial_digest = hashlib.sha256()
    native_digest = hashlib.sha256()
    mismatches: list[str] = []
    components = 0

    def compare(
        name: str,
        serial_buffer: DeviceBuffer,
        native_buffer: DeviceBuffer,
        *,
        serial_offset: int = 0,
        native_offset: int = 0,
        nbytes: int | None = None,
    ) -> None:
        nonlocal components
        serial_bytes = _copy_device_bytes(
            serial, serial_buffer, offset=serial_offset, nbytes=nbytes
        )
        native_bytes = _copy_device_bytes(
            native, native_buffer, offset=native_offset, nbytes=nbytes
        )
        serial_digest.update(name.encode("utf-8") + b"\0" + serial_bytes.tobytes())
        native_digest.update(name.encode("utf-8") + b"\0" + native_bytes.tobytes())
        components += 1
        if not np.array_equal(serial_bytes, native_bytes):
            mismatches.append(name)

    hidden_bytes = spec.hidden_size * np.dtype(np.uint16).itemsize
    final_row = (int(prompt_length) - 1) % PREFILL_CHUNK
    compare(
        "final_hidden",
        serial.buffers.hidden,
        native.buffers.pf.hidden,
        native_offset=final_row * hidden_bytes,
        nbytes=hidden_bytes,
    )
    compare(
        "final_normalized",
        serial.buffers.normalized,
        native.buffers.normalized,
        nbytes=hidden_bytes,
    )

    live_cache_bytes = int(prompt_length) * spec.kv_size * np.dtype(np.uint16).itemsize
    for layer_id, (serial_layer, native_layer) in enumerate(
        zip(serial.buffers.layers, native.buffers.layers, strict=True)
    ):
        compare(
            f"layer_{layer_id}.key_cache",
            serial_layer.key_cache,
            native_layer.key_cache,
            nbytes=live_cache_bytes,
        )
        compare(
            f"layer_{layer_id}.value_cache",
            serial_layer.value_cache,
            native_layer.value_cache,
            nbytes=live_cache_bytes,
        )

    for owner_name in ("sliding_span_owner", "global_span_owner"):
        serial_owner = getattr(serial.buffers, owner_name)
        native_owner = getattr(native.buffers, owner_name)
        for field in (
            "base_offsets",
            "live_counts",
            "token_positions",
            "evict_mask",
            "row_positions",
        ):
            compare(
                f"{owner_name}.{field}",
                getattr(serial_owner, field),
                getattr(native_owner, field),
            )

    return {
        "passed": not mismatches,
        "component_count": components,
        "mismatches": mismatches,
        "serial_sha256": serial_digest.hexdigest(),
        "native_sha256": native_digest.hexdigest(),
    }


def _quality_gate(
    checkpoint,
    tokenizer: MapleTokenizer,
    prompts: list[dict[str, Any]],
    *,
    backend: str,
    continuation_steps: int,
) -> dict[str, Any]:
    runner_type = _maple_runner_type(backend)
    serial = runner_type.load(checkpoint, backend=backend, max_context=512)
    native = runner_type.load(checkpoint, backend=backend, max_context=512)
    rows: list[dict[str, Any]] = []
    try:
        for prompt in prompts:
            tokens = tokenizer.encode_chat(prompt["text"])
            if len(tokens) > 512:
                raise ValueError(f"prompt {prompt['id']} has {len(tokens)} tokens; native cap is 512")
            serial.reset()
            native.reset()
            serial_result = serial.prefill(tokens)
            serial_logits = serial.copy_logits()
            native_result = native.prefill_native(tokens)
            native_logits = _copy_native_final_logits(native)
            state = _prefill_state_gate(
                serial,
                native,
                prompt_length=len(tokens),
            )
            position_rows = [
                {
                    "position": len(tokens) - 1,
                    "kl": _kl_divergence(serial_logits, native_logits),
                    "serial_top1": int(np.argmax(serial_logits)),
                    "native_top1": int(np.argmax(native_logits)),
                    "token_equal": native_result.token_id == serial_result.token_id,
                }
            ]
            serial_token = serial_result.token_id
            native_token = native_result.token_id
            for offset in range(continuation_steps):
                serial_step = serial.step(serial_token)
                native_step = native.step(native_token)
                serial_logits = serial.copy_logits()
                native_logits = native.copy_logits()
                position_rows.append(
                    {
                        "position": len(tokens) + offset,
                        "kl": _kl_divergence(serial_logits, native_logits),
                        "serial_top1": int(np.argmax(serial_logits)),
                        "native_top1": int(np.argmax(native_logits)),
                        "token_equal": native_step.token_id == serial_step.token_id,
                    }
                )
                serial_token = serial_step.token_id
                native_token = native_step.token_id
            rows.append(
                {
                    **prompt,
                    "prompt_tokens": len(tokens),
                    "state": state,
                    "positions": position_rows,
                }
            )
    finally:
        native.close()
        serial.close()

    positions = [position for row in rows for position in row["positions"]]
    top1_matches = sum(
        position["serial_top1"] == position["native_top1"] for position in positions
    )
    token_matches = sum(position["token_equal"] for position in positions)
    category_counts = Counter(row["category"] for row in rows)
    heldout_counts = Counter(row["category"] for row in rows if row["heldout"])
    max_kl = max(position["kl"] for position in positions)
    mean_kl = statistics.fmean(position["kl"] for position in positions)
    top1_agreement = top1_matches / len(positions)
    state_matches = sum(row["state"]["passed"] for row in rows)
    passed = (
        len(rows) == 18
        and all(category_counts[category] > 0 for category in REQUIRED_CATEGORIES)
        and all(heldout_counts[category] > 0 for category in REQUIRED_CATEGORIES)
        and max_kl <= 0.05
        and top1_agreement >= 0.90
        and token_matches == len(positions)
        and state_matches == len(rows)
    )
    return {
        "passed": passed,
        "prompt_count": len(rows),
        "position_count": len(positions),
        "category_counts": dict(sorted(category_counts.items())),
        "heldout_category_counts": dict(sorted(heldout_counts.items())),
        "max_kl": max_kl,
        "mean_kl": mean_kl,
        "top1_matches": top1_matches,
        "top1_agreement": top1_agreement,
        "token_matches": token_matches,
        "state_matches": state_matches,
        "rows": rows,
    }


def _shape_tokens(tokenizer: MapleTokenizer, prompt: dict[str, Any], length: int) -> tuple[int, ...]:
    chat = tokenizer.encode_chat(prompt["text"])
    content = tokenizer.encode(prompt["text"])
    if not content:
        raise ValueError(f"prompt {prompt['id']} tokenized to empty content")
    values = list(chat[:length])
    while len(values) < length:
        values.extend(content[: length - len(values)])
    return tuple(values)


def _selected_shape_prompts(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category in REQUIRED_CATEGORIES:
        natural = next(row for row in prompts if row["category"] == category and not row["heldout"])
        heldout = next(row for row in prompts if row["category"] == category and row["heldout"])
        selected.extend((natural, heldout))
    return selected


def _time_prefill_samples(
    runner,
    shaped_rows: list[tuple[dict[str, Any], tuple[int, ...]]],
    repetitions: int,
    *,
    native: bool,
) -> list[dict[str, Any]]:
    prefill = runner.prefill_native if native else runner.prefill
    samples: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for prompt, tokens in shaped_rows:
            runner.reset()
            started = time.perf_counter()
            result = prefill(tokens)
            elapsed = time.perf_counter() - started
            samples.append(
                {
                    "repetition": repetition,
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "heldout": prompt["heldout"],
                    "seconds": elapsed,
                    "tokens_per_second": len(tokens) / elapsed,
                    "next_token": result.token_id,
                }
            )
    return samples


def _performance_rows(
    checkpoint,
    tokenizer: MapleTokenizer,
    prompts: list[dict[str, Any]],
    *,
    backend: str,
    lengths: tuple[int, ...],
    repetitions: int,
    warmups: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    selected = _selected_shape_prompts(prompts)
    shaped = {
        length: [(prompt, _shape_tokens(tokenizer, prompt, length)) for prompt in selected]
        for length in lengths
    }
    runner_type = _maple_runner_type(backend)
    reset_memory_stats()
    runner = runner_type.load(checkpoint, backend=backend, max_context=max(lengths))
    resident = memory_stats()
    native_rows: list[dict[str, Any]] = []
    try:
        for length in lengths:
            for _ in range(warmups):
                runner.reset()
                runner.prefill_native(shaped[length][0][1])
            samples = _time_prefill_samples(
                runner,
                shaped[length],
                repetitions,
                native=True,
            )
            total_tokens = length * len(samples)
            total_seconds = sum(sample["seconds"] for sample in samples)
            native_rows.append(
                {
                    "prompt_tokens": length,
                    "samples": len(samples),
                    "aggregate_tokens_per_second": total_tokens / total_seconds,
                    "median_sample_tokens_per_second": statistics.median(
                        sample["tokens_per_second"] for sample in samples
                    ),
                    "min_sample_tokens_per_second": min(
                        sample["tokens_per_second"] for sample in samples
                    ),
                    "max_sample_tokens_per_second": max(
                        sample["tokens_per_second"] for sample in samples
                    ),
                    "sample_rows": samples,
                }
            )
    finally:
        runner.close()
    after_close = memory_stats()

    serial = runner_type.load(checkpoint, backend=backend, max_context=max(lengths))
    try:
        for row in native_rows:
            length = row["prompt_tokens"]
            for _ in range(warmups):
                serial.reset()
                serial.prefill(shaped[length][0][1])
            samples = _time_prefill_samples(
                serial,
                shaped[length],
                repetitions,
                native=False,
            )
            total_tokens = length * len(samples)
            total_seconds = sum(sample["seconds"] for sample in samples)
            aggregate = total_tokens / total_seconds
            row["serial_reference"] = {
                "samples": len(samples),
                "aggregate_tokens_per_second": aggregate,
                "median_sample_tokens_per_second": statistics.median(
                    sample["tokens_per_second"] for sample in samples
                ),
                "min_sample_tokens_per_second": min(
                    sample["tokens_per_second"] for sample in samples
                ),
                "max_sample_tokens_per_second": max(
                    sample["tokens_per_second"] for sample in samples
                ),
                "sample_rows": samples,
                "native_over_serial": row["aggregate_tokens_per_second"] / aggregate,
            }
    finally:
        serial.close()
    if memory_stats()["current_allocated_bytes"] != 0:
        raise RuntimeError("Maple prefill benchmark leaked tracked device allocations")
    return native_rows, resident, after_close


def _hardware_context(backend: str) -> dict[str, Any]:
    if backend.startswith("cuda_"):
        nvidia_smi = _capture(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
        fields = [part.strip() for part in nvidia_smi["output"].split(",")]
        return {
            "gpu": fields[0] if fields and fields[0] else "NVIDIA GPU0",
            "architecture": cuda_target_arch_for_backend(backend),
            "physical_gpu": 0,
            "nvidia_smi": nvidia_smi,
            "nvcc": _capture(["nvcc", "--version"]),
        }
    return {
        "gpu": "AMD Radeon 8060S Graphics",
        "architecture": "gfx1151",
        "rocminfo": _capture(
            [
                "bash",
                "-lc",
                "rocminfo | grep -E 'Name:|Marketing Name:|gfx' | head -8",
            ]
        ),
        "rocm_smi": _capture(
            ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--showtemp"]
        ),
        "hipcc": _capture(["hipcc", "--version"]),
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
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--lengths", default="128,320,512")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--continuation-steps", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lengths = tuple(int(part) for part in args.lengths.split(",") if part)
    if not lengths or any(length <= 0 or length > 512 for length in lengths):
        raise ValueError("--lengths must contain values in [1, 512]")
    if args.repetitions <= 0 or args.warmups < 0 or args.continuation_steps < 0:
        raise ValueError("invalid repetition/warmup/continuation count")

    git = _git_context()
    checkpoint = load_maple_checkpoint(args.model)
    tokenizer = _tokenizer(checkpoint)
    prompts = _load_suite(args.suite, heldout=False) + _load_suite(
        args.heldout, heldout=True
    )
    quality = _quality_gate(
        checkpoint,
        tokenizer,
        prompts,
        backend=args.backend,
        continuation_steps=args.continuation_steps,
    )
    rows, resident, after_close = _performance_rows(
        checkpoint,
        tokenizer,
        prompts,
        backend=args.backend,
        lengths=lengths,
        repetitions=args.repetitions,
        warmups=args.warmups,
    )
    lifecycle_passed = (
        after_close["current_allocated_bytes"] == 0
        and after_close["active_allocations"] == 0
    )
    hardware = _hardware_context(args.backend)
    protocol_qualified = (
        lengths == (128, 320, 512)
        and args.repetitions >= 3
        and args.warmups >= 1
        and args.continuation_steps >= 4
    )
    status = (
        "accepted"
        if quality["passed"]
        and lifecycle_passed
        and git["tracked_clean"]
        and protocol_qualified
        else "rejected"
    )
    resolved_path = str(Path(checkpoint.index.model_path).resolve())
    artifact = {
        "schema_version": 1,
        "date": date.today().isoformat(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_type": "maple_native_prefill_recertification",
        "status": status,
        "performance_claim": status == "accepted",
        "model": {
            "id": args.model,
            "revision": PINNED_REVISION,
            "resolved_path": resolved_path,
            "quant": "maple_ternary2",
            "exact_weight_bytes": checkpoint.validation.exact_weight_bytes,
        },
        "hardware": {**hardware, "host": platform.node()},
        "software": {"python": platform.python_version(), "git": git},
        "protocol": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
                "HIPENGINE_CUDA_ARCH": os.environ.get("HIPENGINE_CUDA_ARCH"),
                "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
                "HIPENGINE_COMPILER_VERSION_FILE": os.environ.get(
                    "HIPENGINE_COMPILER_VERSION_FILE"
                ),
                "HIPENGINE_REQUIRE_CACHED_BUILD": os.environ.get(
                    "HIPENGINE_REQUIRE_CACHED_BUILD"
                ),
                "HIPENGINE_MAPLE_PREFILL_GROUPED_MOE": os.environ.get(
                    "HIPENGINE_MAPLE_PREFILL_GROUPED_MOE"
                ),
                "HIPENGINE_NVCC_VERSION_FILE": os.environ.get(
                    "HIPENGINE_NVCC_VERSION_FILE"
                ),
            },
            "backend": args.backend,
            "suite": str(args.suite),
            "heldout": str(args.heldout),
            "lengths": list(lengths),
            "shape_prompts_per_length": 8,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "continuation_steps": args.continuation_steps,
            "qualified_shape_max_tokens": max(lengths),
            "public_native_prefill_up_to_context": True,
            "serial_fallback_above_qualified_shapes": False,
            "qualified_protocol": protocol_qualified,
        },
        "correctness": quality,
        "performance": {"native_prefill": rows},
        "memory": {
            "resident_tracked": resident,
            "after_close": after_close,
            "lifecycle_passed": lifecycle_passed,
            "scope": "hipEngine-owned device allocations; excludes HIP/CUDA runtime internals",
        },
        "notes": [
            "The retained performance protocol remains fixed at 128/320/512 tokens; public native prefill may continue beyond the qualified 512-token shapes only under a backend-specific long-context gate.",
            "Fixed-shape timing rows are derived from two natural/heldout prompts per category; they are not output-quality scores.",
            "Correctness is evaluated on unmodified natural prompts with byte-exact final hidden/normalized state, live K/V and KVLiveSpans metadata, full-logit KL, and continuation parity.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": status,
        "quality": {key: quality[key] for key in ("prompt_count", "position_count", "max_kl", "mean_kl", "top1_agreement", "token_matches", "state_matches", "passed")},
        "rows": [{key: row[key] for key in ("prompt_tokens", "aggregate_tokens_per_second", "median_sample_tokens_per_second", "min_sample_tokens_per_second", "max_sample_tokens_per_second")} | {"serial_tokens_per_second": row["serial_reference"]["aggregate_tokens_per_second"], "native_over_serial": row["serial_reference"]["native_over_serial"]} for row in rows],
        "resident_tracked_bytes": resident["current_allocated_bytes"],
        "lifecycle_passed": lifecycle_passed,
        "protocol_qualified": protocol_qualified,
        "artifact": str(args.out),
    }, indent=2, sort_keys=True))
    return 0 if status == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
