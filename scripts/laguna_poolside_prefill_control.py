#!/usr/bin/env python3
"""Capture matched Poolside llama.cpp Laguna prefill controls.

The harness sends the exact deterministic token stream used by
``laguna_long_context_profile.py`` to one resident Poolside llama.cpp server.
Only llama.cpp's native ``timings.prompt_ms`` interval is used for throughput;
model load, HTTP transport, and the endpoint's sampled-but-not-decoded token are
reported separately.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from hipengine.loading.gguf import GGUFReader
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_poolside_ar_bench import (
    _post_json,
    _process_rss_bytes,
    _read_optional_int,
    _terminate,
    _wait_for_server,
)
from scripts.laguna_prefill_profile import _profile_token_stream
from scripts.laguna_target_ar_bench import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    _load_prompts,
    _repo_state,
    _sha256_bytes,
    _sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = Path(
    "/home/lhl/models/hipengine_sources/poolside-llama.cpp-laguna/"
    "build-hip-gfx1151/bin/llama-server"
)
DEFAULT_HIPENGINE_ARTIFACT = Path(
    "benchmarks/results/2026-07-23-gfx1151-laguna-prefill-current-main-all-family-profile.json"
)
DEFAULT_LENGTHS = (128, 512, 1024, 4096)
DEFAULT_SERVER_LOG = Path("/tmp/laguna-poolside-prefill-control-server.log")
DEFAULT_OUTPUT = Path("/tmp/laguna-poolside-prefill-control.json")
DEFAULT_GTT_PATH = Path("/sys/class/drm/card1/device/mem_info_gtt_used")


def _parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item) for item in value.split(",") if item.strip())
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be positive integers")
    if len(set(lengths)) != len(lengths):
        raise argparse.ArgumentTypeError("lengths must be distinct")
    return lengths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--hipengine-artifact", type=Path, default=DEFAULT_HIPENGINE_ARTIFACT
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--context-length", type=int, default=4097)
    parser.add_argument("--lengths", type=_parse_lengths, default=DEFAULT_LENGTHS)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument("--kv-dtype", choices=("f16", "bf16"), default="bf16")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-rows", type=int, default=128)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--server-log", type=Path, default=DEFAULT_SERVER_LOG)
    parser.add_argument("--gtt-path", type=Path, default=DEFAULT_GTT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    revision = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=path, text=True
    ).strip()
    tracked = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=path,
        text=True,
    ).strip()
    return {
        "path": str(path.resolve()),
        "revision": revision,
        "tracked_clean": not bool(tracked),
        "tracked_status": tracked.splitlines(),
    }


def _command_capture(command: Sequence[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            tuple(command), capture_output=True, text=True, timeout=20.0, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": list(command), "available": False, "error": str(exc)}
    return {
        "command": list(command),
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _completion_payload(token_ids: Sequence[int]) -> dict[str, Any]:
    return {
        "prompt": [int(token) for token in token_ids],
        # This Poolside endpoint samples one token but performs no generated-token
        # decode when n_predict=0. The native prompt timer remains independent.
        "n_predict": 0,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "repeat_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "dry_multiplier": 0.0,
        "seed": 4242,
        "return_tokens": True,
        "cache_prompt": False,
        "ignore_eos": True,
        "stream": False,
    }


def _response_row(
    *,
    length: int,
    repetition: int,
    response: Mapping[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    timings = response.get("timings") or {}
    prompt_n = int(timings.get("prompt_n", 0) or 0)
    prompt_seconds = float(timings.get("prompt_ms", 0.0) or 0.0) / 1000.0
    predicted_n = int(
        timings.get("predicted_n", response.get("tokens_predicted", 0)) or 0
    )
    predicted_seconds = float(timings.get("predicted_ms", 0.0) or 0.0) / 1000.0
    tokens = [int(token) for token in (response.get("tokens") or ())]
    return {
        "length": int(length),
        "repetition": int(repetition),
        "prompt_n": prompt_n,
        "prompt_seconds": prompt_seconds,
        "prompt_tok_s": (
            prompt_n / prompt_seconds if prompt_seconds > 0.0 else 0.0
        ),
        "wall_seconds": float(wall_seconds),
        "valid_prompt_count": prompt_n == int(length),
        "sampled_token_ids": tokens,
        "predicted_n": predicted_n,
        "predicted_seconds": predicted_seconds,
        "no_post_prompt_decode": predicted_n <= 1 and predicted_seconds <= 0.01,
        "stop_type": response.get("stop_type"),
        "native_timings": dict(timings),
    }


def _timing_order(lengths: Sequence[int], repetition: int) -> tuple[int, ...]:
    ordered = tuple(int(length) for length in lengths)
    return ordered if int(repetition) % 2 == 0 else tuple(reversed(ordered))


def _summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    lengths: Sequence[int],
    repetitions: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for length in lengths:
        selected = sorted(
            (row for row in rows if int(row["length"]) == int(length)),
            key=lambda row: int(row["repetition"]),
        )
        if len(selected) != int(repetitions):
            raise ValueError(
                f"length {length} has {len(selected)} rows, expected {repetitions}"
            )
        if not all(bool(row["valid_prompt_count"]) for row in selected):
            raise ValueError(f"length {length} has an inexact native prompt count")
        if not all(bool(row["no_post_prompt_decode"]) for row in selected):
            raise ValueError(f"length {length} performed a post-prompt decode")
        prompt_seconds = [float(row["prompt_seconds"]) for row in selected]
        wall_seconds = [float(row["wall_seconds"]) for row in selected]
        if any(not math.isfinite(value) or value <= 0.0 for value in prompt_seconds):
            raise ValueError(f"length {length} has invalid native prompt timing")
        median_seconds = statistics.median(prompt_seconds)
        result[str(length)] = {
            "length": int(length),
            "samples_prompt_seconds": prompt_seconds,
            "samples_prompt_tok_s": [
                int(length) / seconds for seconds in prompt_seconds
            ],
            "median_prompt_seconds": median_seconds,
            "median_prompt_tok_s": int(length) / median_seconds,
            "samples_wall_seconds": wall_seconds,
            "median_wall_seconds": statistics.median(wall_seconds),
            "all_prompt_counts_exact": True,
            "all_prompt_only": True,
            "sampled_token_ids": [
                list(row.get("sampled_token_ids") or ()) for row in selected
            ],
        }
    return result


def _load_hipengine_reference(
    path: Path,
    *,
    model_sha256: str,
    token_stream_sha256: str,
) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not artifact.get("pass"):
        raise ValueError("hipEngine current-main prefill artifact did not pass")
    if artifact["model"]["sha256"] != model_sha256:
        raise ValueError("hipEngine comparison uses a different model hash")
    if artifact["protocol"]["token_stream_sha256"] != token_stream_sha256:
        raise ValueError("hipEngine comparison uses a different token stream")
    timings = artifact["balanced_timing"]["timings"]
    return {
        "path": str(path.resolve()),
        "artifact_repo_revision": artifact["repo"]["revision"],
        "model_sha256_matches": True,
        "token_stream_sha256_matches": True,
        "timing_scope": artifact["protocol"]["timing_scope"],
        "chunk_size": artifact["protocol"]["chunk_size"],
        "median_tok_s": {
            str(length): float(row["median_tok_s"])
            for length, row in timings.items()
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    lengths = tuple(int(length) for length in args.lengths)
    if lengths != DEFAULT_LENGTHS:
        raise ValueError(f"retained Poolside control requires lengths {DEFAULT_LENGTHS}")
    if args.repetitions < 3:
        raise ValueError("retained Poolside control requires at least three repetitions")
    if args.context_length <= max(lengths):
        raise ValueError("Poolside endpoint requires one context slot beyond the prompt")
    if args.batch_size < max(lengths):
        raise ValueError("batch size must admit the complete longest prompt")
    if args.ubatch_size <= 0 or args.ubatch_size > args.batch_size:
        raise ValueError("ubatch size must be positive and fit the batch")
    if args.warmup_rows <= 0 or args.warmup_rows > min(lengths):
        raise ValueError("warmup rows must fit the shortest retained prompt")
    if not args.server_bin.is_file() or not os.access(args.server_bin, os.X_OK):
        raise FileNotFoundError(f"Poolside server is not executable: {args.server_bin}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required")

    harness_repo = _repo_state()
    if not harness_repo["tracked_clean"]:
        raise RuntimeError("retained Poolside control requires a clean tracked hipEngine tree")
    source_root = args.server_bin.resolve().parents[2]
    source_repo = _git_state(source_root)
    if not source_repo["tracked_clean"]:
        raise RuntimeError("retained Poolside control requires clean Poolside source")

    prompt_payload = args.prompts.read_bytes()
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    token_stream, token_source = _profile_token_stream(prompts, max(lengths))
    token_stream_sha256 = _sha256_json(token_stream)
    hipengine_reference = _load_hipengine_reference(
        args.hipengine_artifact,
        model_sha256=args.model_sha256,
        token_stream_sha256=token_stream_sha256,
    )

    server_command = [
        str(args.server_bin.resolve()),
        "-m",
        str(args.model.resolve()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "-c",
        str(args.context_length),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "-ngl",
        "999",
        "-fa",
        "off",
        "-ctk",
        args.kv_dtype,
        "-ctv",
        args.kv_dtype,
        "--parallel",
        "1",
        "--no-warmup",
        "--no-repack",
        "--no-mmap",
        "--cache-ram",
        "0",
        "--metrics",
    ]
    binary_sha256 = _file_sha256(args.server_bin)
    binary_version = _command_capture((str(args.server_bin.resolve()), "--version"))
    clocks_before = _command_capture(
        ("rocm-smi", "--showproductname", "--showclocks", "--showpower", "--json")
    )

    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("GPU_MAX_HW_QUEUES", "1")
    rows: list[dict[str, Any]] = []
    gtt_samples: list[int] = []
    rss_samples: list[int] = []
    started_at = time.perf_counter()
    with args.server_log.open("wb") as log:
        process = subprocess.Popen(
            server_command,
            cwd=source_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        startup_seconds = _wait_for_server(
            host=args.host,
            port=args.port,
            timeout=args.startup_timeout,
            process=process,
        )
        ready_gtt = _read_optional_int(args.gtt_path)
        ready_rss = _process_rss_bytes(process.pid)
        if ready_gtt is not None:
            gtt_samples.append(ready_gtt)
        if ready_rss is not None:
            rss_samples.append(ready_rss)

        warmup = _post_json(
            f"http://{args.host}:{args.port}/completion",
            _completion_payload(token_stream[: args.warmup_rows]),
            args.request_timeout,
        )
        warmup_row = _response_row(
            length=args.warmup_rows,
            repetition=-1,
            response=warmup,
            wall_seconds=0.0,
        )
        if not warmup_row["valid_prompt_count"] or not warmup_row["no_post_prompt_decode"]:
            raise RuntimeError("Poolside prefill warmup did not remain prompt-only")

        for repetition in range(args.repetitions):
            for length in _timing_order(lengths, repetition):
                started = time.perf_counter()
                response = _post_json(
                    f"http://{args.host}:{args.port}/completion",
                    _completion_payload(token_stream[:length]),
                    args.request_timeout,
                )
                wall_seconds = time.perf_counter() - started
                row = _response_row(
                    length=length,
                    repetition=repetition,
                    response=response,
                    wall_seconds=wall_seconds,
                )
                rows.append(row)
                gtt = _read_optional_int(args.gtt_path)
                rss = _process_rss_bytes(process.pid)
                if gtt is not None:
                    gtt_samples.append(gtt)
                if rss is not None:
                    rss_samples.append(rss)
                print(
                    f"rep={repetition} length={length} "
                    f"prompt={row['prompt_tok_s']:.3f} tok/s "
                    f"native={row['prompt_seconds']:.6f}s",
                    flush=True,
                )
    finally:
        _terminate(process)

    clocks_after = _command_capture(
        ("rocm-smi", "--showproductname", "--showclocks", "--showpower", "--json")
    )
    summary = _summarize_rows(rows, lengths=lengths, repetitions=args.repetitions)
    ratios = {
        length: summary[length]["median_prompt_tok_s"]
        / hipengine_reference["median_tok_s"][length]
        for length in summary.keys() & hipengine_reference["median_tok_s"].keys()
    }
    result = {
        "schema": 1,
        "kind": "hipengine_poolside_laguna_prefill_control",
        "status": "accepted_external_control",
        "pass": True,
        "performance_claim": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "same-model same-token-stream Poolside llama.cpp prefill control",
        "command": (str(Path(sys.executable).resolve()), *sys.argv),
        "server_command": server_command,
        "timing": {
            "primary": "llama.cpp response timings.prompt_ms",
            "includes": "prompt decode including final prompt logits",
            "excludes": "model load, HTTP transport, and sampled-token decode",
            "endpoint_note": "n_predict=0 samples one token but performs no generated-token forward",
            "order": "ascending then alternating direction by repetition",
            "warmup_rows": int(args.warmup_rows),
            "repetitions": int(args.repetitions),
        },
        "protocol": {
            "lengths": list(lengths),
            "context_capacity": int(args.context_length),
            "batch_size": int(args.batch_size),
            "ubatch_size": int(args.ubatch_size),
            "flash_attention": False,
            "kv_dtype": args.kv_dtype,
            "gpu_layers": 999,
            "parallel": 1,
            "mmap": False,
            "repack": False,
            "prompt_cache": False,
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompt_payload),
            "token_stream_sha256": token_stream_sha256,
            "token_source": token_source,
        },
        "model": {
            "path": str(args.model.resolve()),
            "size_bytes": args.model.stat().st_size,
            "sha256": args.model_sha256,
            "quant": "Q4_K_M mixed GGUF v3",
        },
        "poolside": {
            "source": source_repo,
            "binary": {
                "path": str(args.server_bin.resolve()),
                "sha256": binary_sha256,
                "version": binary_version,
            },
            "startup_seconds_excluded": startup_seconds,
            "server_log_path": str(args.server_log.resolve()),
            "server_log_sha256": _file_sha256(args.server_log),
        },
        "harness_repo": harness_repo,
        "hardware": {
            "target_arch": os.environ.get("HIPENGINE_HIP_ARCH", "gfx1151"),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "gpu_max_hw_queues": env.get("GPU_MAX_HW_QUEUES"),
            "clocks_before": clocks_before,
            "clocks_after": clocks_after,
            "gtt_path": str(args.gtt_path),
            "gtt_samples_bytes": gtt_samples,
            "gtt_peak_bytes": max(gtt_samples) if gtt_samples else None,
            "rss_samples_bytes": rss_samples,
            "rss_peak_bytes": max(rss_samples) if rss_samples else None,
        },
        "warmup": warmup_row,
        "rows": rows,
        "summary": summary,
        "hipengine_reference": hipengine_reference,
        "diagnostic_llamacpp_over_hipengine": ratios,
        "comparison_eligibility": {
            "cross_engine_speed_ratio_eligible": False,
            "same_model_hash": True,
            "same_token_stream_hash": True,
            "same_kv_dtype": args.kv_dtype == "bf16",
            "same_microbatch_rows": args.ubatch_size
            == hipengine_reference["chunk_size"],
            "reason": "llama.cpp prompt_ms excludes HTTP and sampling but does not include hipEngine's final argmax bookkeeping; ratios are diagnostic controls",
        },
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = _parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
