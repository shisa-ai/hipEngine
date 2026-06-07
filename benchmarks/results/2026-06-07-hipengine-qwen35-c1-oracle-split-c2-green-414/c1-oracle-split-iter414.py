#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Sequence

from hipengine.core.dtype import DType
from hipengine.generation import ResidentBatchScheduler
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_batch_retained_bench import DEFAULT_FIXTURE, DEFAULT_MODEL, _compiler_version, _load_prompt_slices

MODEL = "/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16"
FIXTURE = "/tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json"
PROMPT_LENGTH = 512
BATCH_SIZE = 2
WARMUP_DECODE_TOKENS = 8
DECODE_TOKENS = 128
TOTAL_DECODE_TOKENS = WARMUP_DECODE_TOKENS + DECODE_TOKENS
MAX_LAYERS = 40
MAX_SEQUENCE_LENGTH = PROMPT_LENGTH + TOTAL_DECODE_TOKENS + 1
COMPILER_VERSION_FILE = pathlib.Path("/tmp/hipengine-retained/hipcc-version.txt")
OUT = pathlib.Path("/tmp/hipengine-c1-oracle-split-iter414.json")


def _equal_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    n = 0
    for x, y in zip(left, right):
        if int(x) != int(y):
            break
        n += 1
    return n


def _window(seq: Sequence[int], center: int, radius: int = 5) -> list[int]:
    start = max(0, int(center) - int(radius))
    end = min(len(seq), int(center) + int(radius) + 1)
    return [int(x) for x in seq[start:end]]


def _run_serial_c1(runner: Qwen35ParoNextTokenRunner, prompts: list[list[int]], *, compiler_version: str | None) -> list[list[int]]:
    rows: list[list[int]] = []
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        max_layers=MAX_LAYERS,
        max_batch_size=1,
        compiler_version=compiler_version,
        require_cached_build=True,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        for prompt in prompts:
            seed = session.prefill_native(prompt, sample=True)
            if seed is None:
                raise RuntimeError("serial c1 prefill did not produce a seed token")
            seq = [int(seed.token_id)]
            next_token = int(seed.token_id)
            for offset in range(TOTAL_DECODE_TOKENS):
                position = len(prompt) + offset
                session._set_token_embedding(next_token, stream=0)
                session._set_position(position, stream=0)
                hidden = session._run_layers(position=position, stream=0)
                result = session._sample_from_hidden(hidden)
                if result is None:
                    raise RuntimeError("serial c1 decode did not produce a token")
                next_token = int(result.token_id)
                seq.append(next_token)
            rows.append(seq)
            session.reset()
    return rows


def _run_native_batch_c1(runner: Qwen35ParoNextTokenRunner, prompts: list[list[int]], *, compiler_version: str | None) -> list[list[int]]:
    rows: list[list[int]] = []
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        max_layers=MAX_LAYERS,
        max_batch_size=1,
        compiler_version=compiler_version,
        require_cached_build=True,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        for prompt in prompts:
            scheduler = ResidentBatchScheduler(capacity=1)
            request_id = scheduler.submit(prompt, max_new_tokens=TOTAL_DECODE_TOKENS)
            admitted = scheduler.admit_pending()
            if admitted != (request_id,):
                raise RuntimeError(f"unexpected native c1 admitted ids {admitted!r}")
            slabs = scheduler.next_compact_prefill_slabs(chunk_size=len(prompt), block_size=session.block_size)
            if len(slabs) != 1:
                raise RuntimeError("native c1 expected one compact prefill slab")
            seed = session.prefill_native_packed(slabs[0], sample=True)[0]
            if seed is None:
                raise RuntimeError("native c1 prefill did not produce a seed token")
            seq = [int(seed.token_id)]
            next_token = int(seed.token_id)
            for offset in range(TOTAL_DECODE_TOKENS):
                result = session.step_batch_native(
                    [next_token],
                    positions=[len(prompt) + offset],
                    slots=[0],
                    sample=True,
                )[0]
                if result is None:
                    raise RuntimeError("native c1 decode did not produce a token")
                next_token = int(result.token_id)
                seq.append(next_token)
            rows.append(seq)
            session.reset()
    return rows


def _device_payload() -> dict[str, object]:
    payload: dict[str, object] = {"env": {"HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES")}}
    try:
        import ctypes
        hip = ctypes.CDLL("libamdhip64.so")
        count = ctypes.c_int()
        payload["hipGetDeviceCount_error"] = int(hip.hipGetDeviceCount(ctypes.byref(count)))
        payload["visible_device_count"] = int(count.value)
        current = ctypes.c_int()
        payload["hipGetDevice_error"] = int(hip.hipGetDevice(ctypes.byref(current)))
        payload["current_device"] = int(current.value)
        name = ctypes.create_string_buffer(256)
        payload["hipDeviceGetName_error"] = int(hip.hipDeviceGetName(name, ctypes.sizeof(name), current))
        payload["device_name"] = name.value.decode(errors="replace")
    except Exception as exc:
        payload["error"] = repr(exc)
    return payload


def main() -> int:
    started = time.perf_counter()
    prompts = _load_prompt_slices(pathlib.Path(FIXTURE), prompt_length=PROMPT_LENGTH, batch_size=BATCH_SIZE)
    compiler_version = _compiler_version(COMPILER_VERSION_FILE)
    runner = Qwen35ParoNextTokenRunner(MODEL)
    serial_started = time.perf_counter()
    serial = _run_serial_c1(runner, prompts, compiler_version=compiler_version)
    serial_seconds = time.perf_counter() - serial_started
    native_started = time.perf_counter()
    native = _run_native_batch_c1(runner, prompts, compiler_version=compiler_version)
    native_seconds = time.perf_counter() - native_started
    prefixes = [_equal_prefix(s, n) for s, n in zip(serial, native, strict=True)]
    mismatch_indices = [None if p == len(serial[row]) == len(native[row]) else p for row, p in enumerate(prefixes)]
    mismatch_summaries = []
    for row, idx in enumerate(mismatch_indices):
        if idx is None:
            continue
        mismatch_summaries.append({
            "row": row,
            "first_mismatch_index": idx,
            "serial_token_at_mismatch": int(serial[row][idx]),
            "native_batch_token_at_mismatch": int(native[row][idx]),
            "serial_window": _window(serial[row], idx),
            "native_batch_window": _window(native[row], idx),
        })
    payload = {
        "schema": 1,
        "mode": "qwen35_paro_c1_serial_vs_native_batch_oracle_split",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": "RX 7900 XTX via HIP_VISIBLE_DEVICES=1",
        "performance_claim": False,
        "workload": {
            "model": MODEL,
            "fixture": FIXTURE,
            "prompt_length": PROMPT_LENGTH,
            "batch_size": BATCH_SIZE,
            "rows": BATCH_SIZE,
            "warmup_decode_tokens": WARMUP_DECODE_TOKENS,
            "decode_tokens": DECODE_TOKENS,
            "total_decode_tokens": TOTAL_DECODE_TOKENS,
            "tokens_per_sequence": 1 + TOTAL_DECODE_TOKENS,
            "max_layers": MAX_LAYERS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "kv_storage_dtype": "bf16",
        },
        "device": _device_payload(),
        "timing": {"serial_c1_seconds": serial_seconds, "native_batch_c1_seconds": native_seconds, "total_seconds": time.perf_counter() - started},
        "correctness": {
            "passed": serial == native,
            "comparison": "serial_c1_vs_native_batch_c1",
            "prefix_lengths": prefixes,
            "min_equal_prefix_tokens": min(prefixes),
            "first_mismatch_indices": mismatch_indices,
            "mismatch_summaries": mismatch_summaries,
        },
        "sequences": {"serial_c1": serial, "native_batch_c1": native},
        "command": "HIP_VISIBLE_DEVICES=1 python3 /tmp/hipengine-c1-oracle-split-iter414.py",
        "notes": ["serial c1 uses prefill_native/_run_layers/_sample_from_hidden", "native_batch c1 uses packed prefill plus step_batch_native, matching retained-bench c1 reference"],
    }
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(OUT), "prefix_lengths": prefixes, "mismatch_summaries": mismatch_summaries, "timing": payload["timing"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
