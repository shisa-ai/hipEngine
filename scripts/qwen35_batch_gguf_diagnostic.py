#!/usr/bin/env python3
"""GGUF c>N generated-token equality diagnostic and command template.

This is the GGUF counterpart to the Qwen/PARO c>N correctness harnesses.  The
default mode emits a no-GPU command template for sweep planning.  ``--execute``
runs the production packed AR route against independent c1 generations and
emits a correctness-only artifact.  Neither mode makes a throughput claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from scripts.qwen35_batch_artifact_schema import _load_payload
from scripts.qwen35_batch_constants import (
    RETAINED_ARTIFACT_GGUF_DIAGNOSTIC_SCRIPT,
    RETAINED_ARTIFACT_GGUF_E2E_CORRECTNESS_SCRIPT,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json"
GGUF_QUANTS = (
    "gguf_q4_k_m",
    "gguf_q5_k_m",
    "gguf_q6_k",
    "gguf_q8_0",
    "gguf_ud_q3_k_m",
)
_GGUF_DIAGNOSTIC_SCRIPT = RETAINED_ARTIFACT_GGUF_DIAGNOSTIC_SCRIPT
_GGUF_E2E_CORRECTNESS_SCRIPT = RETAINED_ARTIFACT_GGUF_E2E_CORRECTNESS_SCRIPT
_COMMAND_ENV_KEYS = (
    "HIP_VISIBLE_DEVICES",
    "HIPENGINE_HIP_ARCH",
    "HIPENGINE_COMPILER_VERSION_FILE",
    "HIPENGINE_REQUIRE_CACHED_BUILD",
)


def _command_env_prefix_parts() -> list[str]:
    assignments = [
        f"{key}={value}"
        for key in _COMMAND_ENV_KEYS
        if (value := os.environ.get(key))
    ]
    return ["env", *assignments] if assignments else []


def _payload_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, allow_nan=False)


def _load_fixture(path: Path) -> dict[str, Any]:
    fixture = _load_payload(path)
    required = {"model", "prompt", "prompt_ids", "sampling", "acceptance"}
    missing = sorted(required - set(fixture))
    if missing:
        raise ValueError(f"fixture {path} missing required keys: {', '.join(missing)}")
    return fixture


def _canonical_command(args: argparse.Namespace) -> str:
    argv = [
        *_command_env_prefix_parts(),
        "python3",
        _GGUF_DIAGNOSTIC_SCRIPT,
        "--fixture",
        str(args.fixture),
        "--rows",
        str(args.rows),
        "--backend",
        str(args.backend),
        "--quant",
        str(args.quant),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.model:
        argv.extend(["--model", str(args.model)])
    if getattr(args, "prompt_suite", None) is not None:
        argv.extend(["--prompt-suite", str(args.prompt_suite)])
    if bool(getattr(args, "execute", False)):
        argv.extend(["--repeat-runs", str(int(getattr(args, "repeat_runs", 1))), "--execute"])
    if getattr(args, "json", None) is not None:
        argv.extend(["--json", str(args.json)])
    return shlex.join(argv)


def _single_row_command(args: argparse.Namespace, *, model: str, row: int) -> str:
    argv = [
        *_command_env_prefix_parts(),
        "python3",
        _GGUF_E2E_CORRECTNESS_SCRIPT,
        "--fixture",
        str(args.fixture),
        "--model",
        model,
        "--backend",
        str(args.backend),
        "--quant",
        str(args.quant),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--json",
        f"/tmp/hipengine-gguf-c1-row{row}.json",
    ]
    return shlex.join(argv)


def _prompt_tokens(fixture: dict[str, Any], prompt_length: int) -> tuple[int, ...]:
    source = tuple(int(token) for token in fixture["prompt_ids"])
    if not source:
        raise ValueError("fixture prompt_ids must be non-empty")
    length = len(source) if int(prompt_length) <= 0 else int(prompt_length)
    if length <= 0:
        raise ValueError("prompt_length must be positive")
    return tuple(source[index % len(source)] for index in range(length))


def _native_batch_once(
    session: Qwen35GGUFResidentSession,
    prompts: tuple[tuple[int, ...], ...],
    *,
    rows: int,
    max_new_tokens: int,
) -> tuple[list[list[int]], float, float, list[float], dict[str, str]]:
    if len(prompts) != rows:
        raise ValueError("native prompts must match rows")
    session.reset()
    first_tokens: list[int] = []
    prefill_start = time.perf_counter()
    for slot, prompt in enumerate(prompts):
        result = session.prefill_slot(prompt, slot=slot, return_logits=False)
        first_tokens.append(int(result.token_id))
    prefill_s = time.perf_counter() - prefill_start

    sequences = [[token] for token in first_tokens]
    current = list(first_tokens)
    execution_paths: dict[str, str] = {}
    step_latencies_s: list[float] = []
    with session.capture_native_rows_graph(
        rows=rows,
        max_context_len=session.target_layout.max_sequence_length,
    ) as graph:
        decode_start = time.perf_counter()
        for _ in range(max(0, max_new_tokens - 1)):
            step_start = time.perf_counter()
            result = graph.step(current)
            step_latencies_s.append(time.perf_counter() - step_start)
            current = [int(token) for token in result.token_ids]
            execution_paths = dict(result.execution_paths)
            for row, token in enumerate(current):
                sequences[row].append(token)
        decode_s = time.perf_counter() - decode_start
    return sequences, prefill_s, decode_s, step_latencies_s, execution_paths


def _independent_c1_sequences(
    runner,
    model: str,
    prompts: tuple[tuple[int, ...], ...],
    *,
    rows: int,
    max_new_tokens: int,
    quant: str,
    max_sequence_length: int,
) -> list[list[int]]:
    if len(prompts) != rows:
        raise ValueError("independent c1 prompts must match rows")
    sequences: list[list[int]] = []
    for prompt in prompts:
        with Qwen35GGUFResidentSession(
            model,
            max_sequence_length=max_sequence_length,
            shared_runner=runner,
        ) as session:
            session.select_prefill_quant(quant)
            result = session.prefill(prompt, return_logits=False)
            tokens = [int(result.token_id)]
            remaining = max(0, max_new_tokens - 1)
            if remaining:
                with session.capture_decode_graph(
                    position=len(prompt),
                    steps_per_replay=1,
                    max_replay_steps=remaining,
                    record_steps=remaining,
                ) as graph:
                    graph.replay(remaining)
                    tokens.extend(int(token) for token in graph.read_generated_token_ids(remaining))
            sequences.append(tokens)
    return sequences


def _run_native_q3(
    args: argparse.Namespace,
    fixture: dict[str, Any],
    *,
    model: str,
    quant: str,
    backend: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    rows = int(args.rows)
    prompt = _prompt_tokens(fixture, int(args.prompt_length))
    prompts = tuple(
        tuple(prompt[(index + row) % len(prompt)] for index in range(len(prompt)))
        for row in range(rows)
    )
    max_sequence_length = max(len(row) for row in prompts) + max(1, max_new_tokens) + 1
    prefill_samples: list[float] = []
    decode_samples: list[float] = []
    decode_step_latencies_s: list[float] = []
    native_sequences: list[list[int]] = []
    execution_paths: dict[str, str] = {}
    reset_memory_stats()
    with Qwen35GGUFResidentSession(
        model,
        max_sequence_length=max_sequence_length,
        max_batch_size=rows,
    ) as session:
        session.select_prefill_quant(quant)
        for _ in range(int(args.warmup)):
            _native_batch_once(
                session,
                prompts,
                rows=rows,
                max_new_tokens=max_new_tokens,
            )
        for _ in range(int(args.repeats)):
            (
                native_sequences,
                prefill_s,
                decode_s,
                step_latencies,
                execution_paths,
            ) = _native_batch_once(
                session,
                prompts,
                rows=rows,
                max_new_tokens=max_new_tokens,
            )
            prefill_samples.append(prefill_s)
            decode_samples.append(decode_s)
            decode_step_latencies_s.extend(step_latencies)
        native_memory = memory_stats()
        runtime = session.runtime
        native_hip_memory: dict[str, int] = {}
        if runtime is not None:
            free_bytes, total_bytes = runtime.mem_get_info()
            native_hip_memory = {
                "free_bytes": int(free_bytes),
                "used_bytes": int(total_bytes - free_bytes),
                "total_bytes": int(total_bytes),
            }
        independent = _independent_c1_sequences(
            session.runner,
            model,
            prompts,
            rows=rows,
            max_new_tokens=max_new_tokens,
            quant=quant,
            max_sequence_length=max_sequence_length,
        )
        harness_memory = memory_stats()

    equality = [native == scalar for native, scalar in zip(native_sequences, independent, strict=True)]
    status = "eq_ok" if all(equality) else "rejected_correctness"
    prefill_tokens = sum(len(row) for row in prompts)
    decode_tokens = rows * max(0, max_new_tokens - 1)
    prefill_median = statistics.median(prefill_samples)
    decode_median = statistics.median(decode_samples)
    sorted_step_latencies = sorted(decode_step_latencies_s)
    latency_p50_s = (
        statistics.median(sorted_step_latencies) if sorted_step_latencies else 0.0
    )
    latency_p95_s = (
        sorted_step_latencies[
            min(
                len(sorted_step_latencies) - 1,
                math.ceil(0.95 * len(sorted_step_latencies)) - 1,
            )
        ]
        if sorted_step_latencies
        else 0.0
    )
    normalized = argparse.Namespace(**vars(args))
    return {
        "schema": 2,
        "mode": "gguf_cN_native_equality",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "performance_claim": status == "eq_ok",
        "rows": rows,
        "model": model,
        "backend": backend,
        "quant": quant,
        "hardware": str(args.hardware_label or "unspecified"),
        "fixture": str(Path(args.fixture)),
        "prompt_token_count": len(prompt),
        "prompt_token_counts": [len(row) for row in prompts],
        "prompt_row_rotation_offsets": list(range(rows)),
        "max_new_tokens": max_new_tokens,
        "command": _canonical_command(normalized),
        "workload": {
            "prompt_rows": rows,
            "prompt_tokens_per_row": [len(row) for row in prompts],
            "generated_tokens_per_row": max_new_tokens,
            "native_compact_prefill": False,
            "prefill_path": "resident_slot_serial_bulk",
            "native_caware_decode": True,
            "serial_decode_fallback": False,
        },
        "scheduler": {
            "mode": "fixed_all_active",
            "capacity": rows,
            "peak_active_rows": rows,
            "active_c_histogram": {str(rows): max(0, max_new_tokens - 1)},
            "active_mask": [True] * rows,
            "graph_bucket_key": (
                f"decode:c={rows}:ctx={max_sequence_length}:"
                f"mask={'1' * rows}:top_k=8:experts=8"
            ),
            "compaction_occurred": False,
            "continuous_prompt_list_gate": (
                "tests/test_generation_qwen35_gguf_sampling.py::"
                "test_gguf_native_scheduler_reclaims_compacts_and_readmits"
            ),
        },
        "measurements": {
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "prefill_wall_s": prefill_samples,
            "prefill_wall_s_median": prefill_median,
            "prefill_tok_s_aggregate": prefill_tokens / prefill_median,
            "decode_wall_s": decode_samples,
            "decode_wall_s_median": decode_median,
            "decode_tok_s_aggregate": (
                decode_tokens / decode_median if decode_tokens > 0 else 0.0
            ),
            "decode_tok_s_per_request": (
                decode_tokens / decode_median / rows if decode_tokens > 0 else 0.0
            ),
            "decode_step_latency_ms": [
                1000.0 * sample for sample in decode_step_latencies_s
            ],
            "decode_step_latency_p50_ms": 1000.0 * latency_p50_s,
            "decode_step_latency_p95_ms": 1000.0 * latency_p95_s,
        },
        "memory": {
            **native_memory,
            "tracked_peak_allocated_gib": (
                int(native_memory.get("peak_allocated_bytes", 0)) / 1024**3
            ),
            "hip": native_hip_memory,
            "correctness_harness_peak_allocated_bytes": int(
                harness_memory.get("peak_allocated_bytes", 0)
            ),
            "correctness_harness_peak_allocated_gib": (
                int(harness_memory.get("peak_allocated_bytes", 0)) / 1024**3
            ),
        },
        "execution_paths": execution_paths,
        "generated_token_ids": {
            "native_cN": native_sequences,
            "independent_c1": independent,
            "row_equal": equality,
        },
        "correctness": {
            "generated_ids_equal": all(equality),
            "full_logits_gate": (
                "tests/test_qwen35_gguf_target_rows.py C=2/4/8, C=2 split-attention, "
                "and reclaim/compact/readmit"
            ),
            "kl_threshold": 0.05,
            "top1_threshold": 0.90,
        },
        "notes": [
            "Prompt rows are cyclically rotated so slot aliasing cannot self-validate.",
            "Prefill is serial across resident slots but shares one weight set; only decode throughput is native c>N.",
            "Native decode provenance must contain indexed Conv/GDN, KVLiveSpans batch attention, selected-row MoE, and row argmax.",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.rows <= 0:
        raise ValueError("rows must be positive")
    fixture = _load_fixture(Path(args.fixture))
    model = str(args.model or fixture["model"].get("path", ""))
    quant = str(args.quant or fixture["acceptance"].get("quant", ""))
    backend = str(args.backend or fixture["acceptance"].get("backend", ""))
    requested_max_new_tokens = getattr(args, "max_new_tokens", None)
    max_new_tokens = int(
        fixture["sampling"].get("max_new_tokens", 0)
        if requested_max_new_tokens is None
        else requested_max_new_tokens
    )
    normalized = argparse.Namespace(
        fixture=Path(args.fixture),
        rows=int(args.rows),
        model=str(args.model or ""),
        backend=backend,
        quant=quant,
        max_new_tokens=max_new_tokens,
        repeat_runs=int(getattr(args, "repeat_runs", 1)),
        execute=bool(getattr(args, "execute", False)),
        prompt_suite=getattr(args, "prompt_suite", None),
        json=getattr(args, "json", None),
    )

    if normalized.execute:
        return _run_equality(normalized, fixture=fixture, model=model)

    blockers = [
        "native GGUF c>N was not executed in template mode; rerun with --execute",
        "diagnostic records template commands only; no c>N correctness or performance claim is allowed",
    ]
    if int(args.rows) < 2:
        blockers.append("c>N diagnostic requires rows >= 2")
    if quant not in GGUF_QUANTS:
        blockers.append(f"quant {quant!r} is not in the supported GGUF template set {GGUF_QUANTS!r}")
    if model and not Path(model).exists():
        blockers.append(f"model path is not present on this host: {model}")

    independent_c1 = [
        _single_row_command(normalized, model=model, row=row)
        for row in range(int(args.rows))
    ]
    payload = {
        "schema": 1,
        "mode": "gguf_cN_equality_template",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "rows": int(args.rows),
        "model": model,
        "backend": backend,
        "quant": quant,
        "fixture": str(Path(args.fixture)),
        "prompt_token_count": len(fixture["prompt_ids"]),
        "max_new_tokens": max_new_tokens,
        "command": _canonical_command(normalized),
        "independent_c1_commands": independent_c1,
        "native_cN_command": _canonical_command(normalized),
        "expected_terminal_statuses": ["eq_ok", "blocked", "rejected_correctness"],
        "blockers": blockers,
        "notes": [
            "Template mode is CPU-safe sweep planning; --execute runs the production packed AR route.",
            "An eq_ok artifact compares generated token ids from native c>N against independent c=1 rows.",
            "No benchmark rollup or retained performance claim should consume this blocked template artifact.",
        ],
    }
    return payload


def _run_equality(
    args: argparse.Namespace,
    *,
    fixture: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    if args.rows < 2:
        blockers.append("c>N diagnostic requires rows >= 2")
    if args.repeat_runs <= 0:
        blockers.append("repeat-runs must be positive")
    if args.max_new_tokens <= 0:
        blockers.append("max-new-tokens must be positive")
    if args.quant not in GGUF_QUANTS:
        blockers.append(f"quant {args.quant!r} is not in the supported GGUF set {GGUF_QUANTS!r}")
    if not model or not Path(model).exists():
        blockers.append(f"model path is not present on this host: {model}")

    prompt_rows: tuple[Any, ...]
    prompt_metadata: list[dict[str, str]]
    if args.prompt_suite is not None:
        try:
            suite_rows = _load_prompt_suite(Path(args.prompt_suite))
        except (OSError, ValueError) as exc:
            blockers.append(f"prompt suite could not be loaded: {exc}")
            suite_rows = []
        if len(suite_rows) < int(args.rows):
            blockers.append(
                f"prompt suite contains {len(suite_rows)} row(s), need at least {args.rows}"
            )
        selected = suite_rows[: int(args.rows)]
        prompt_rows = tuple(row["prompt"] for row in selected)
        prompt_metadata = [
            {"id": str(row["id"]), "category": str(row["category"])}
            for row in selected
        ]
    else:
        prompt_ids = tuple(int(token) for token in fixture["prompt_ids"])
        if not prompt_ids:
            blockers.append("fixture prompt_ids must be non-empty")
        if len(prompt_ids) + int(args.max_new_tokens) >= 1024:
            blockers.append("production packed GGUF AR currently requires context < 1024")
        prompt_rows = tuple(prompt_ids for _row in range(int(args.rows)))
        prompt_metadata = [
            {"id": f"fixture_row_{row}", "category": "fixture"}
            for row in range(int(args.rows))
        ]
    if blockers:
        return _equality_payload(
            args,
            fixture=fixture,
            model=model,
            status="blocked",
            blockers=blockers,
            independent_c1_token_ids=[],
            runs=[],
            prepared_context_tokens=None,
            prompt_metadata=prompt_metadata,
        )

    sampling = SamplingParams(
        max_tokens=int(args.max_new_tokens),
        temperature=float(fixture["sampling"].get("temperature", 0.0)),
        top_p=float(fixture["sampling"].get("top_p", 1.0)),
        ignore_eos=bool(fixture["sampling"].get("ignore_eos", False)),
    )
    llm = LLM(model, backend=str(args.backend), quant=str(args.quant))
    prepared_context_tokens = llm.prepare(
        max_sequence_length=1024,
        sampling_params=sampling,
    )

    independent_c1_token_ids: list[list[int]] = []
    for prompt in prompt_rows:
        output = llm.generate_detailed((prompt,), sampling)[0]
        if output.generated_token_ids is None:
            raise RuntimeError("GGUF independent c1 generation did not expose generated_token_ids")
        independent_c1_token_ids.append([int(token) for token in output.generated_token_ids])

    runs: list[dict[str, Any]] = []
    for repeat_index in range(int(args.repeat_runs)):
        outputs = llm.generate_detailed(prompt_rows, sampling)
        if len(outputs) != int(args.rows):
            raise RuntimeError(
                f"GGUF c>N generation returned {len(outputs)} rows for requested c={args.rows}"
            )
        token_rows: list[list[int]] = []
        for output in outputs:
            if output.generated_token_ids is None:
                raise RuntimeError("GGUF c>N generation did not expose generated_token_ids")
            token_rows.append([int(token) for token in output.generated_token_ids])
        row_equal = [
            tokens == independent_c1_token_ids[row]
            for row, tokens in enumerate(token_rows)
        ]
        batch_execution = _last_batch_generation(llm)
        runs.append(
            {
                "repeat_index": repeat_index,
                "generated_token_ids": token_rows,
                "row_equal": row_equal,
                "all_rows_equal": all(row_equal),
                "execution_path": batch_execution.get("path"),
                "native_caware_decode": bool(batch_execution.get("native_caware_decode", False)),
                "serial_decode_fallback": bool(batch_execution.get("serial_decode_fallback", True)),
                "batch_execution": batch_execution,
            }
        )

    equality_ok = all(bool(item["all_rows_equal"]) for item in runs)
    native_ok = all(
        bool(item["native_caware_decode"]) and not bool(item["serial_decode_fallback"])
        for item in runs
    )
    if not equality_ok:
        status = "rejected_correctness"
        blockers.append("native GGUF c>N generated tokens differ from independent c1")
    elif not native_ok:
        status = "blocked"
        blockers.append("generation did not stay on the native c-aware GGUF decode route")
    else:
        status = "eq_ok"
    return _equality_payload(
        args,
        fixture=fixture,
        model=model,
        status=status,
        blockers=blockers,
        independent_c1_token_ids=independent_c1_token_ids,
        runs=runs,
        prepared_context_tokens=prepared_context_tokens,
        prompt_metadata=prompt_metadata,
    )


def _load_prompt_suite(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_no}: prompt row must be an object")
            prompt_id = raw.get("id")
            category = raw.get("category")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(f"{path}:{line_no}: prompt id must be non-empty text")
            if prompt_id in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate prompt id {prompt_id!r}")
            seen_ids.add(prompt_id)
            if not isinstance(category, str) or not category:
                raise ValueError(f"{path}:{line_no}: category must be non-empty text")
            prompt = raw.get("prompt")
            if prompt is None:
                messages = raw.get("messages")
                if not isinstance(messages, list) or len(messages) != 1:
                    raise ValueError(f"{path}:{line_no}: expected prompt or one user message")
                message = messages[0]
                if not isinstance(message, dict) or message.get("role") != "user":
                    raise ValueError(f"{path}:{line_no}: expected one user message")
                prompt = message.get("content")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{path}:{line_no}: prompt must be non-empty text")
            rows.append({"id": prompt_id, "category": category, "prompt": prompt})
    if not rows:
        raise ValueError(f"{path}: prompt suite is empty")
    return rows


def _last_batch_generation(llm: Any) -> dict[str, Any]:
    generator = getattr(llm, "_text_generator", None)
    visited: set[int] = set()
    while generator is not None and id(generator) not in visited:
        visited.add(id(generator))
        payload = getattr(generator, "last_batch_generation", None)
        if isinstance(payload, dict):
            compact_keys = (
                "path",
                "batch_size",
                "request_ids",
                "prompt_lengths",
                "decode_steps",
                "native_decode_steps",
                "serial_decode_fallback",
                "native_compact_prefill",
                "native_caware_decode",
                "native_sampler_rows",
                "throughput_claim_eligible",
                "group_rows",
            )
            return {key: payload[key] for key in compact_keys if key in payload}
        generator = getattr(generator, "inner", None)
    return {}


def _equality_payload(
    args: argparse.Namespace,
    *,
    fixture: dict[str, Any],
    model: str,
    status: str,
    blockers: list[str],
    independent_c1_token_ids: list[list[int]],
    runs: list[dict[str, Any]],
    prepared_context_tokens: int | None,
    prompt_metadata: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema": 1,
        "mode": "gguf_cN_generated_token_equality",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "passed": status == "eq_ok",
        "performance_claim": False,
        "rows": int(args.rows),
        "repeat_runs": int(args.repeat_runs),
        "model": model,
        "backend": str(args.backend),
        "quant": str(args.quant),
        "fixture": str(Path(args.fixture)),
        "prompt_suite": None if args.prompt_suite is None else str(Path(args.prompt_suite)),
        "prompt_rows": prompt_metadata,
        "prompt_token_count": (
            len(fixture["prompt_ids"])
            if args.prompt_suite is None
            else None
        ),
        "max_new_tokens": int(args.max_new_tokens),
        "prepared_context_tokens": prepared_context_tokens,
        "command": _canonical_command(args),
        "independent_c1_commands": (
            [
                _single_row_command(args, model=model, row=row)
                for row in range(int(args.rows))
            ]
            if args.prompt_suite is None
            else []
        ),
        "independent_c1_token_ids": independent_c1_token_ids,
        "runs": runs,
        "blockers": blockers,
        "notes": [
            "Generated-token equality is checked against independent width-1 generations.",
            "This correctness diagnostic does not establish hidden/state/KV equality or a performance claim.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model", default="", help="Override fixture model path")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", choices=GGUF_QUANTS, default="gguf_q4_k_m")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument(
        "--prompt-suite",
        type=Path,
        help=(
            "Optional JSONL prompt suite; execute mode uses its first --rows prompts "
            "instead of repeating fixture prompt_ids"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run native c>N versus independent c1 generated-token equality instead of emitting a blocked template",
    )
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = _payload_json(payload)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0 if payload["status"] in {"eq_ok", "blocked", "rejected_correctness"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
