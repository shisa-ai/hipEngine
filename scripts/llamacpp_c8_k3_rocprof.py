#!/usr/bin/env python3
"""Capture one prewarmed llama.cpp C8/K3 request with delayed rocprof tracing.

The Python client remains outside the profiled process. ``llama-server`` starts
under rocprofv3, warms before the collection period, and receives one measured
eight-lane request while collection is active. Raw CSV remains under
``--raw-root``; ``--output`` records commands, source identity, request timing,
acceptance counters, output hashes, and raw-file hashes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"


def read_prompt(path: Path, prompt_id: str | None) -> dict[str, str]:
    """Return one suite prompt rendered with the benchmark's ChatML boundary."""

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if prompt_id is not None and row["id"] != prompt_id:
            continue
        content = row["messages"][0]["content"]
        return {
            "id": str(row["id"]),
            "category": str(row["category"]),
            "rendered": (
                f"<|im_start|>user\n{content}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        }
    raise ValueError(f"prompt suite does not contain {prompt_id!r}")


def build_server_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.server.resolve()),
        "-m",
        str(args.model.resolve()),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "-c",
        str(args.ctx_size),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "-ngl",
        str(args.gpu_layers),
        "-np",
        str(args.width),
        "--no-mmap",
        "-t",
        str(args.threads),
        "-tb",
        str(args.threads),
        "-fa",
        "on",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        str(args.draft_max),
        "--spec-draft-p-min",
        "0.0",
    ]


def build_rocprof_command(
    args: argparse.Namespace, *, trace_dir: Path, server_command: list[str]
) -> list[str]:
    period = f"{args.collection_delay_s}:{args.collection_duration_s}:1"
    return [
        str(args.rocprofv3),
        "--kernel-trace",
        "--hip-runtime-trace",
        "--memory-copy-trace",
        "--output-format",
        "csv",
        "--collection-period",
        period,
        "-d",
        str(trace_dir),
        "--",
        *server_command,
    ]


def _get(url: str, *, timeout_s: float = 2.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read()


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode())


def wait_for_health(url: str, *, timeout_s: float, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"profiled llama-server exited with {process.returncode}")
        try:
            _get(url)
            return
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(1.0)
    raise TimeoutError(f"llama-server health timeout: {last_error!r}")


def request_payload(prompt: str, *, n_predict: int) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": int(n_predict),
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 1,
        "cache_prompt": False,
        "stream": False,
    }


def run_concurrent_request(
    *,
    url: str,
    prompt: str,
    n_predict: int,
    width: int,
    timeout_s: float,
) -> tuple[list[dict[str, Any]], float]:
    """Issue a barrier-synchronized request per lane and return complete wall."""

    barrier = threading.Barrier(width + 1)
    epoch = [0.0]
    payload = request_payload(prompt, n_predict=n_predict)

    def run_lane(lane: int) -> dict[str, Any]:
        barrier.wait(timeout=30.0)
        while epoch[0] == 0.0:
            time.sleep(0)
        start = time.perf_counter()
        response = _post_json(url, payload, timeout_s=timeout_s)
        end = time.perf_counter()
        return summarize_response(response, lane=lane, start=start - epoch[0], end=end - epoch[0])

    with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
        futures = [pool.submit(run_lane, lane) for lane in range(width)]
        barrier.wait(timeout=30.0)
        epoch[0] = time.perf_counter()
        rows = [future.result() for future in futures]

    wall_s = max(row["end_offset_s"] for row in rows) - min(
        row["start_offset_s"] for row in rows
    )
    return rows, float(wall_s)


def summarize_response(
    response: dict[str, Any], *, lane: int, start: float, end: float
) -> dict[str, Any]:
    return {
        "lane": int(lane),
        "start_offset_s": float(start),
        "end_offset_s": float(end),
        "tokens_predicted": int(response.get("tokens_predicted", 0)),
        "tokens_evaluated": int(response.get("tokens_evaluated", 0)),
        "content_sha256": hashlib.sha256(
            str(response.get("content", "")).encode()
        ).hexdigest(),
        "timings": response.get("timings") or {},
    }


def terminate_process_group(
    process: subprocess.Popen[bytes], *, graceful_s: float = 45.0
) -> str:
    if process.poll() is not None:
        return "already_exited"
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=graceful_s)
        return "interrupted"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return "killed_after_timeout"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def raw_file_records(trace_dir: Path, pattern: str) -> list[dict[str, Any]]:
    return [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(trace_dir.rglob(pattern))
    ]


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = args.raw_root.resolve()
    if raw_root.exists():
        shutil.rmtree(raw_root)
    trace_dir = raw_root / "trace"
    trace_dir.mkdir(parents=True)
    log_path = raw_root / "rocprof-server.log"

    prompt = read_prompt(args.prompts, args.prompt_id)
    server_command = build_server_command(args)
    rocprof_command = build_rocprof_command(
        args, trace_dir=trace_dir, server_command=server_command
    )
    artifact: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen38_llamacpp_c8_k3_delayed_profile",
        "status": "running",
        "performance_claim": False,
        "label": str(args.label),
        "host": platform.node(),
        "source": str(args.source.resolve()),
        "source_commit": git_value(args.source, "rev-parse", "HEAD"),
        "source_dirty": bool(git_value(args.source, "status", "--short")),
        "server": str(args.server.resolve()),
        "server_command": server_command,
        "rocprof_command": rocprof_command,
        "model": str(args.model.resolve()),
        "model_size_bytes": args.model.stat().st_size,
        "prompt": prompt,
        "protocol": {
            "width": int(args.width),
            "candidate_depth": int(args.draft_max),
            "visible_tokens_per_lane": int(args.max_tokens),
            "kv_type": "f16",
            "flash_attention": True,
            "sampling": "raw greedy; temperature=0, top_k=1, top_p=1",
        },
        "profile_mode": (
            "llama-server starts under delayed rocprof collection; C8 is prewarmed "
            "before collection; this separate unprofiled client drives one C8/K3 request"
        ),
    }

    environment = os.environ.copy()
    environment["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    environment["ROCR_VISIBLE_DEVICES"] = str(args.gpu)
    process: subprocess.Popen[bytes] | None = None
    launch = time.monotonic()
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                rocprof_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        wait_for_health(
            f"http://{args.host}:{args.port}/health",
            timeout_s=args.server_start_timeout_s,
            process=process,
        )
        completion_url = f"http://{args.host}:{args.port}/completion"
        warmup_rows, warmup_wall = run_concurrent_request(
            url=completion_url,
            prompt=prompt["rendered"],
            n_predict=args.warmup_tokens,
            width=args.width,
            timeout_s=args.request_timeout_s,
        )
        artifact["warmup"] = {"wall_s": warmup_wall, "rows": warmup_rows}

        collection_start = launch + args.collection_delay_s
        if time.monotonic() >= collection_start:
            raise RuntimeError(
                "warmup overran collection start; increase --collection-delay-s"
            )
        time.sleep(max(0.0, collection_start + 1.0 - time.monotonic()))
        rows, wall_s = run_concurrent_request(
            url=completion_url,
            prompt=prompt["rendered"],
            n_predict=args.max_tokens,
            width=args.width,
            timeout_s=args.request_timeout_s,
        )
        visible_tokens = sum(row["tokens_predicted"] for row in rows)
        expected_visible_tokens = args.width * args.max_tokens
        artifact["measured"] = {
            "wall_s": wall_s,
            "visible_tokens": visible_tokens,
            "expected_visible_tokens": expected_visible_tokens,
            "all_lanes_full_length": visible_tokens == expected_visible_tokens,
            "complete_wall_tok_s": visible_tokens / wall_s,
            "lane_content_exact": len({row["content_sha256"] for row in rows}) == 1,
            "draft_generated": sum(
                int(row["timings"].get("draft_n", 0)) for row in rows
            ),
            "draft_accepted": sum(
                int(row["timings"].get("draft_n_accepted", 0)) for row in rows
            ),
            "rows": rows,
        }
        collection_end = launch + args.collection_delay_s + args.collection_duration_s
        time.sleep(max(0.0, collection_end + 2.0 - time.monotonic()))
    finally:
        artifact["termination"] = (
            terminate_process_group(process) if process is not None else "not_started"
        )
        artifact["rocprof_return_code"] = process.returncode if process is not None else None
        artifact["raw"] = {
            "root": str(raw_root),
            "server_log": str(log_path),
            "kernel_csv": raw_file_records(trace_dir, "*_kernel_trace.csv"),
            "hip_api_csv": raw_file_records(trace_dir, "*_hip_api_trace.csv"),
            "memory_copy_csv": raw_file_records(trace_dir, "*_memory_copy_trace.csv"),
        }
        measured = artifact.get("measured") or {}
        passed = (
            bool(artifact["raw"]["kernel_csv"])
            and measured.get("all_lanes_full_length") is True
            and measured.get("lane_content_exact") is True
            and int(measured.get("draft_generated", 0)) > 0
        )
        artifact["status"] = "diagnostic_retained" if passed else "diagnostic_incomplete"
        artifact["passed"] = passed
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-id")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--draft-max", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--warmup-tokens", type=int, default=2)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--collection-delay-s", type=float, default=20.0)
    parser.add_argument("--collection-duration-s", type=float, default=12.0)
    parser.add_argument("--server-start-timeout-s", type=float, default=600.0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--rocprofv3", default="rocprofv3")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_profile(args)
    measured = artifact.get("measured") or {}
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "label": artifact["label"],
                "source_commit": artifact["source_commit"],
                "complete_wall_tok_s": measured.get("complete_wall_tok_s"),
                "draft_generated": measured.get("draft_generated"),
                "draft_accepted": measured.get("draft_accepted"),
                "kernel_csv": artifact["raw"]["kernel_csv"],
            },
            indent=2,
        )
    )
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
