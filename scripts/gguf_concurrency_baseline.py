#!/usr/bin/env python3
"""Record Phase-A GGUF c1 and explicit serial c2/c4 controls.

The serial control disables packed and per-slot-stream prompt/decode routes so a
multi-prompt request executes complete c1 rows in order.  An optional package-c4
probe records the current automatic route separately; its timing is diagnostic
and is not mixed into the serial-control rates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from hipengine.benchmark.provenance import collect_artifact_provenance


REPO_ROOT = Path(__file__).resolve().parents[1]
_ROUTE_ENV_KEYS = (
    "HIPENGINE_GGUF_AR_PACKED_PREFILL",
    "HIPENGINE_GGUF_AR_PACKED_DECODE",
    "HIPENGINE_GGUF_AR_STREAM_DECODE",
)
_PROVENANCE_ENV_KEYS = (
    "HIPENGINE_BACKEND",
    "HIPENGINE_HIP_ARCH",
    "HIPENGINE_COMPILER_VERSION_FILE",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_MAX_HW_QUEUES",
)
_ROUTE_MANIFEST_KEYS = (
    "path",
    "batch_size",
    "prompt_lengths",
    "decode_steps",
    "native_decode_steps",
    "serial_decode_fallback",
    "native_compact_prefill",
    "native_caware_decode",
    "native_sampler_rows",
    "throughput_claim_eligible",
    "group_rows",
    "timing_scope",
    "timing_owner",
    "batch_id",
)


def _parse_concurrencies(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            raise ValueError("concurrencies must not contain blank entries")
        try:
            value = int(text, 10)
        except ValueError as exc:
            raise ValueError("concurrencies must contain integers") from exc
        if value <= 0:
            raise ValueError("concurrencies must be positive")
        if value in values:
            raise ValueError("concurrencies must be unique")
        values.append(value)
    if not values:
        raise ValueError("concurrencies must not be empty")
    return tuple(values)


def _execution_environment(mode: str) -> dict[str, str]:
    if mode == "serial":
        return {
            "HIPENGINE_GGUF_AR_PACKED_PREFILL": "0",
            "HIPENGINE_GGUF_AR_PACKED_DECODE": "0",
            "HIPENGINE_GGUF_AR_STREAM_DECODE": "0",
        }
    if mode == "package":
        return {
            "HIPENGINE_GGUF_AR_PACKED_PREFILL": "1",
            "HIPENGINE_GGUF_AR_PACKED_DECODE": "1",
            "HIPENGINE_GGUF_AR_STREAM_DECODE": "1",
        }
    raise ValueError(f"unknown GGUF concurrency execution mode: {mode!r}")


@contextmanager
def _temporary_environment(updates: Mapping[str, str]) -> Iterator[None]:
    prior = {key: os.environ.get(key) for key in updates}
    os.environ.update({str(key): str(value) for key, value in updates.items()})
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _trajectory_fingerprint(tokens: Sequence[int]) -> dict[str, Any]:
    digest = hashlib.sha256()
    token_list = [int(token) for token in tokens]
    for token in token_list:
        digest.update(int(token).to_bytes(8, "little", signed=True))
    return {
        "length": len(token_list),
        "token_ids_sha256": digest.hexdigest(),
        "first_token_ids": token_list[:8],
        "last_token_ids": token_list[-8:],
        "final_token_id": token_list[-1] if token_list else None,
    }


def _compact_route_manifest(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {key: payload[key] for key in _ROUTE_MANIFEST_KEYS if key in payload}


def _last_batch_generation(llm: Any) -> dict[str, Any]:
    generator = getattr(llm, "_text_generator", None)
    visited: set[int] = set()
    while generator is not None and id(generator) not in visited:
        visited.add(id(generator))
        payload = getattr(generator, "last_batch_generation", None)
        if isinstance(payload, Mapping):
            return dict(payload)
        generator = getattr(generator, "inner", None)
    return {}


def _measurement_sample(
    outputs: Sequence[Any],
    *,
    wall_seconds: float,
    prompt_length: int,
    last_batch_generation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row_trajectories: list[dict[str, Any]] = []
    owned_timing_ms: dict[str, float] = {}
    timed_rows = 0
    owned_timing_rows = 0
    timing_scopes: set[str] = set()
    batch_ids: set[str] = set()
    generated_tokens = 0
    continuation_tokens = 0

    for output in outputs:
        raw_tokens = getattr(output, "generated_token_ids", None)
        if raw_tokens is None:
            raise RuntimeError("GGUF baseline output did not expose generated_token_ids")
        tokens = tuple(int(token) for token in raw_tokens)
        row_trajectories.append(_trajectory_fingerprint(tokens))
        generated_tokens += len(tokens)
        continuation_tokens += max(0, len(tokens) - 1)

        telemetry = getattr(output, "telemetry", None)
        timing = getattr(telemetry, "timing", None)
        if not isinstance(timing, Mapping):
            continue
        timed_rows += 1
        scope = getattr(telemetry, "timing_scope", None)
        if scope is not None:
            timing_scopes.add(str(scope))
        batch_id = getattr(telemetry, "batch_id", None)
        if batch_id is not None:
            batch_ids.add(str(batch_id))
        if getattr(telemetry, "timing_owner", None) is False:
            continue
        owned_timing_rows += 1
        for key, value in timing.items():
            owned_timing_ms[str(key)] = owned_timing_ms.get(str(key), 0.0) + float(value)

    rows = len(outputs)
    prompt_tokens = rows * int(prompt_length)
    prefill_seconds = float(owned_timing_ms.get("prefill_ms", 0.0)) / 1000.0
    decode_seconds = float(owned_timing_ms.get("decode_ms", 0.0)) / 1000.0
    rates = {
        "prefill_tok_s": (prompt_tokens / prefill_seconds) if prefill_seconds > 0.0 else None,
        "decode_tok_s": (continuation_tokens / decode_seconds) if decode_seconds > 0.0 else None,
        "generated_wall_tok_s": (generated_tokens / wall_seconds) if wall_seconds > 0.0 else None,
    }
    trajectory_hashes = [row["token_ids_sha256"] for row in row_trajectories]
    return {
        "wall_seconds": float(wall_seconds),
        "accounting": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "continuation_decode_tokens": continuation_tokens,
            "rows": rows,
        },
        "timing_ownership": {
            "timed_rows": timed_rows,
            "owned_timing_rows": owned_timing_rows,
            "timing_scopes": sorted(timing_scopes),
            "batch_ids": sorted(batch_ids),
        },
        "owned_timing_ms": dict(sorted(owned_timing_ms.items())),
        "rates": rates,
        "row_trajectories": row_trajectories,
        "all_rows_same_trajectory": len(set(trajectory_hashes)) <= 1,
        "route": _compact_route_manifest(last_batch_generation),
    }


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    return {
        "samples": samples,
        "median": float(statistics.median(samples)),
        "min": min(samples),
        "max": max(samples),
    }


def _summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one measured sample is required")
    rate_names = ("prefill_tok_s", "decode_tok_s", "generated_wall_tok_s")
    rates: dict[str, Any] = {}
    for name in rate_names:
        values = [float(sample["rates"][name]) for sample in samples if sample["rates"].get(name) is not None]
        rates[name] = _metric_summary(values) if values else None
    signatures = [
        tuple(str(row["token_ids_sha256"]) for row in sample["row_trajectories"])
        for sample in samples
    ]
    routes = [dict(sample.get("route") or {}) for sample in samples]
    return {
        "sample_count": len(samples),
        "wall_seconds": _metric_summary([float(sample["wall_seconds"]) for sample in samples]),
        "rates": rates,
        "repeatable_trajectories": len(set(signatures)) == 1,
        "stable_route_manifest": all(route == routes[0] for route in routes[1:]),
        "route": routes[0],
        "samples": [dict(sample) for sample in samples],
    }


def _run_generation(
    llm: Any,
    *,
    prompts: tuple[tuple[int, ...], ...],
    sampling: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    outputs = llm.generate_detailed(prompts, sampling)
    wall_seconds = time.perf_counter() - started
    return _measurement_sample(
        outputs,
        wall_seconds=wall_seconds,
        prompt_length=len(prompts[0]),
        last_batch_generation=_last_batch_generation(llm),
    )


def _run_serial_control(
    llm: Any,
    *,
    concurrency: int,
    prompt: tuple[int, ...],
    sampling: Any,
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, Any]:
    prompts = tuple(prompt for _ in range(int(concurrency)))
    with _temporary_environment(_execution_environment("serial")):
        for _ in range(int(warmup_runs)):
            _run_generation(llm, prompts=prompts, sampling=sampling)
        samples = [
            _run_generation(llm, prompts=prompts, sampling=sampling)
            for _ in range(int(measured_runs))
        ]
    summary = _summarize_samples(samples)
    expected_serial_fallback = int(concurrency) > 1
    route_ok = all(
        sample["route"].get("path") == "gguf_serial_greedy_decode"
        and sample["route"].get("native_caware_decode") is False
        and sample["route"].get("serial_decode_fallback") is expected_serial_fallback
        for sample in samples
    )
    exact_accounting = all(
        sample["accounting"]["rows"] == int(concurrency)
        and sample["accounting"]["prompt_tokens"] == int(concurrency) * len(prompt)
        and sample["accounting"]["generated_tokens"] == int(concurrency) * int(sampling.max_tokens)
        and sample["accounting"]["continuation_decode_tokens"]
        == int(concurrency) * max(0, int(sampling.max_tokens) - 1)
        for sample in samples
    )
    return {
        "mode": "serial",
        "concurrency": int(concurrency),
        "environment": _execution_environment("serial"),
        "route_ok": route_ok,
        "exact_accounting": exact_accounting,
        **summary,
    }


def _run_package_inventory(
    llm: Any,
    *,
    concurrency: int,
    prompt: tuple[int, ...],
    sampling: Any,
) -> dict[str, Any]:
    prompts = tuple(prompt for _ in range(int(concurrency)))
    with _temporary_environment(_execution_environment("package")):
        sample = _run_generation(llm, prompts=prompts, sampling=sampling)
    return {
        "mode": "package_inventory",
        "concurrency": int(concurrency),
        "environment": _execution_environment("package"),
        "timing_eligible": False,
        "timing_note": (
            "Route inventory only: packed timing ownership may intentionally collapse "
            "batch-shared fields and is not compared with serial controls."
        ),
        "sample": sample,
    }


def _reference_trajectory_hash(control: Mapping[str, Any]) -> str:
    samples = control.get("samples") or []
    if not samples:
        raise ValueError("c1 control has no measured samples")
    rows = samples[0].get("row_trajectories") or []
    if len(rows) != 1:
        raise ValueError("c1 control must contain exactly one row")
    return str(rows[0]["token_ids_sha256"])


def _all_control_rows_match(control: Mapping[str, Any], reference_hash: str) -> bool:
    return all(
        str(row["token_ids_sha256"]) == str(reference_hash)
        for sample in control.get("samples") or []
        for row in sample.get("row_trajectories") or []
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.prompt_length) <= 0:
        raise ValueError("prompt-length must be positive")
    if int(args.max_new_tokens) <= 0:
        raise ValueError("max-new-tokens must be positive")
    if int(args.warmup_runs) < 0 or int(args.measured_runs) <= 0:
        raise ValueError("warmup-runs must be non-negative and measured-runs must be positive")
    concurrencies = _parse_concurrencies(args.concurrencies)
    if 1 not in concurrencies:
        raise ValueError("concurrencies must include c1 as the trajectory and rate reference")
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    if int(args.prompt_length) + int(args.max_new_tokens) >= int(args.prepare_context_tokens):
        raise ValueError("prepare-context-tokens must exceed prompt-length + max-new-tokens")

    from hipengine import LLM, SamplingParams
    from hipengine.core.memory import memory_stats, reset_memory_stats

    sampling = SamplingParams(
        max_tokens=int(args.max_new_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )
    prompt = tuple(int(args.prompt_token_id) for _ in range(int(args.prompt_length)))
    llm = LLM(str(model), backend=str(args.backend), quant=str(args.quant))
    prepared_context_tokens = llm.prepare(
        max_sequence_length=int(args.prepare_context_tokens),
        sampling_params=sampling,
    )
    reset_memory_stats()
    memory_after_prepare = memory_stats()

    controls = [
        _run_serial_control(
            llm,
            concurrency=concurrency,
            prompt=prompt,
            sampling=sampling,
            warmup_runs=int(args.warmup_runs),
            measured_runs=int(args.measured_runs),
        )
        for concurrency in concurrencies
    ]
    c1_control = next(control for control in controls if int(control["concurrency"]) == 1)
    c1_hash = _reference_trajectory_hash(c1_control)
    for control in controls:
        control["matches_c1_trajectory"] = _all_control_rows_match(control, c1_hash)

    package_inventory = None
    if bool(args.include_package_c4):
        package_inventory = _run_package_inventory(
            llm,
            concurrency=4,
            prompt=prompt,
            sampling=sampling,
        )
        package_inventory["matches_c1_trajectory"] = all(
            str(row["token_ids_sha256"]) == c1_hash
            for row in package_inventory["sample"]["row_trajectories"]
        )

    memory_final = memory_stats()
    controls_ok = all(
        bool(control["route_ok"])
        and bool(control["exact_accounting"])
        and bool(control["repeatable_trajectories"])
        and bool(control["stable_route_manifest"])
        and bool(control["matches_c1_trajectory"])
        for control in controls
    )
    command = [sys.executable, "scripts/gguf_concurrency_baseline.py", *sys.argv[1:]]
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=str(args.backend),
        target_arch="gfx1100" if str(args.backend) == "hip_gfx1100" else None,
        model_path=model,
        quant=str(args.quant),
        kv_dtype="bf16",
        command=command,
        environment={
            **{key: os.environ.get(key) for key in _PROVENANCE_ENV_KEYS},
            **{key: "controlled per artifact row" for key in _ROUTE_ENV_KEYS},
        },
        build_profile="public GGUF package defaults; explicit serial route controls",
        timing_protocol=(
            "one model load/prepare; per-width discarded warmups; measured c1/serial-c2/serial-c4 "
            "choice-owned prefill/decode timing; package c4 is route inventory only"
        ),
        warmups=int(args.warmup_runs),
        repetitions=int(args.measured_runs),
        profiler={"used": False, "reason": "Phase A control; profiler begins at Phase B1/C1"},
    )
    return {
        "schema": 1,
        "kind": "gfx1100_gguf_concurrency_phase_a_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_complete" if controls_ok else "failed",
        "passed": controls_ok,
        "performance_claim": False,
        "provenance": provenance,
        "workload": {
            "model": str(model),
            "backend": str(args.backend),
            "quant": str(args.quant),
            "kv_dtype": "bf16",
            "prompt_token_id": int(args.prompt_token_id),
            "prompt_length": int(args.prompt_length),
            "max_new_tokens": int(args.max_new_tokens),
            "continuation_decode_tokens_per_row": int(args.max_new_tokens) - 1,
            "sampling": "greedy_top1_ignore_eos",
            "prepare_context_tokens": int(args.prepare_context_tokens),
            "prepared_context_tokens": prepared_context_tokens,
            "concurrencies": list(concurrencies),
            "warmup_runs": int(args.warmup_runs),
            "measured_runs": int(args.measured_runs),
        },
        "controls": controls,
        "package_c4_route_inventory": package_inventory,
        "correctness": {
            "c1_trajectory_sha256": c1_hash,
            "all_serial_controls_exact_accounting": all(bool(row["exact_accounting"]) for row in controls),
            "all_serial_controls_repeatable": all(bool(row["repeatable_trajectories"]) for row in controls),
            "all_serial_controls_match_c1": all(bool(row["matches_c1_trajectory"]) for row in controls),
            "all_serial_routes_explicit": all(bool(row["route_ok"]) for row in controls),
        },
        "memory": {
            "scope": "hipengine tracked allocator; excludes driver/AOTriton internal allocations",
            "after_prepare": memory_after_prepare,
            "final": memory_final,
        },
        "notes": [
            "Serial controls disable packed and per-slot-stream prefill/decode routes.",
            "Decode rate counts only continuation steps after the prompt-final token: C*(max_new_tokens-1).",
            "Generated-wall rate counts all returned tokens and includes public generation overhead, but excludes model load/prepare.",
            "The repeated-token workload is a Phase A control, not prompt-diversity or promotion evidence.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--concurrencies", default="1,2,4")
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prepare-context-tokens", type=int, default=1024)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--include-package-c4", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
