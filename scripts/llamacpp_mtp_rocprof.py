#!/usr/bin/env python3
"""Profile a bounded llama.cpp HIP MTP server request with rocprofv3.

This is a diagnostic bridge for the GGUF MTP parity work.  llama.cpp's MTP
stage JSONL shows draft/verify wall buckets, but not kernel families.  This
wrapper starts ``llama-server`` under ``rocprofv3 --kernel-trace``, sends a
small deterministic ``/completion`` request, then summarizes any kernel CSV
that survives profiler finalization with ``scripts/llamacpp_kernel_trace_summary``.

The trace is intentionally whole-process, not a draft-window-only marker trace.
Use it as a kernel-family proxy alongside ``LLAMA_MTP_STAGE_TIMINGS`` rather
than as a retained throughput row.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.llamacpp_kernel_trace_summary import build_summary as build_kernel_summary
from scripts.llamacpp_mtp_bench import _summarize_stage_timings

SCHEMA = "hipengine.llamacpp_mtp_rocprof.v1"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_SERVER_BIN = Path("/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-server")


def build_server_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.server_bin),
        "-m",
        str(args.model),
        "-ngl",
        str(args.gpu_layers),
        "-fa",
        str(args.flash_attn),
        "-ctk",
        str(args.cache_type_k),
        "-ctv",
        str(args.cache_type_v),
        "-c",
        str(args.ctx_size),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "--alias",
        str(args.alias),
        "--no-cache-prompt",
        "--reasoning",
        str(args.reasoning),
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        str(args.draft_max),
    ]
    cmd.extend(str(item) for item in args.server_extra_arg)
    return cmd


def build_completion_payload(args: argparse.Namespace) -> dict[str, Any]:
    prompt = [int(args.token_id)] * int(args.prompt_tokens) if args.token_repeat else str(args.prompt)
    return {
        "prompt": prompt,
        "n_predict": int(args.max_tokens),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "min_p": float(args.min_p),
        "seed": int(args.seed),
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
    }


def build_rocprof_command(args: argparse.Namespace, *, trace_dir: Path, server_command: list[str]) -> list[str]:
    return [
        str(args.rocprofv3),
        "--kernel-trace",
        "--output-format",
        "csv",
        "-d",
        str(trace_dir),
        "--",
        *server_command,
    ]


def wait_for_health(host: str, port: int, timeout_s: float) -> None:
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + float(timeout_s)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(2.0)
    raise TimeoutError(f"server did not become healthy within {timeout_s}s: {last_error!r}")


def post_json(host: str, port: int, path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        return json.loads(resp.read().decode("utf-8"))


def terminate_process_group(proc: subprocess.Popen[bytes], *, graceful_s: float, kill_s: float) -> str:
    if proc.poll() is not None:
        return "already_exited"
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_exited"
    try:
        proc.wait(timeout=float(graceful_s))
        return "terminated"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=float(kill_s))
        return "killed_after_finalize_timeout"


def find_kernel_csv(trace_dir: Path) -> Path | None:
    files = sorted(trace_dir.glob("**/*_kernel_trace.csv"))
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    return max(files, key=lambda path: path.stat().st_size)


def _git_rev_parse(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_dirty(repo: Path) -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return None


def _llamacpp_repo_from_bin(server_bin: Path) -> Path | None:
    path = server_bin.resolve()
    for parent in path.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _request_summary(response: dict[str, Any], wall_s: float) -> dict[str, Any]:
    timings = response.get("timings") or {}
    predicted = int(timings.get("predicted_n") or 0)
    draft_n = int(timings.get("draft_n") or 0)
    draft_acc = int(timings.get("draft_n_accepted") or 0)
    return {
        "wall_s": float(wall_s),
        "predicted_n": predicted,
        "predicted_per_second": timings.get("predicted_per_second"),
        "predicted_ms": timings.get("predicted_ms"),
        "draft_n": draft_n,
        "draft_n_accepted": draft_acc,
        "draft_acceptance": (draft_acc / draft_n) if draft_n else None,
        "accepted_per_output": (draft_acc / predicted) if predicted else None,
        "stop_type": response.get("stop_type"),
        "truncated": response.get("truncated"),
        "timings": timings,
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = Path(args.raw_root)
    trace_dir = raw_root / "trace"
    log_path = raw_root / "llama-server-rocprof.log"
    response_path = raw_root / "response.json"
    stage_path = raw_root / "stage-timings.jsonl"
    if raw_root.exists():
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    stage_path.unlink(missing_ok=True)

    server_command = build_server_command(args)
    rocprof_command = build_rocprof_command(args, trace_dir=trace_dir, server_command=server_command)
    env = os.environ.copy()
    env["LLAMA_MTP_STAGE_TIMINGS"] = str(stage_path)

    response: dict[str, Any] | None = None
    request_error: str | None = None
    start_error: str | None = None
    terminate_status = "not_started"
    return_code: int | None = None
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            rocprof_command,
            cwd=Path(args.server_bin).resolve().parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            wait_for_health(str(args.host), int(args.port), float(args.server_start_timeout))
            payload = build_completion_payload(args)
            t0 = time.perf_counter()
            response = post_json(str(args.host), int(args.port), "/completion", payload, timeout_s=float(args.request_timeout))
            wall_s = time.perf_counter() - t0
            response_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
            request_summary = _request_summary(response, wall_s)
        except Exception as exc:
            request_summary = {}
            if proc.poll() is not None:
                start_error = f"server exited early with rc={proc.returncode}"
            request_error = repr(exc)
        finally:
            terminate_status = terminate_process_group(
                proc,
                graceful_s=float(args.profiler_finalize_timeout),
                kill_s=20.0,
            )
            return_code = proc.returncode

    kernel_csv = find_kernel_csv(trace_dir)
    kernel_summary = (
        build_kernel_summary(
            kernel_csv,
            label=str(args.label),
            command=" ".join(rocprof_command),
            top=int(args.top),
        )
        if kernel_csv is not None
        else None
    )
    llama_repo = _llamacpp_repo_from_bin(Path(args.server_bin))
    return {
        "schema": SCHEMA,
        "date": date.today().isoformat(),
        "status": "diagnostic_retained" if kernel_summary is not None and request_error is None else "diagnostic_incomplete",
        "performance_claim": False,
        "purpose": "Whole-process llama.cpp HIP MTP kernel-family proxy under rocprofv3.",
        "label": str(args.label),
        "model": str(args.model),
        "hardware": str(args.hardware),
        "software": {
            "hipengine_commit": _git_rev_parse(REPO_ROOT),
            "hipengine_dirty": _git_dirty(REPO_ROOT),
            "llama_cpp_repo": str(llama_repo) if llama_repo is not None else None,
            "llama_cpp_commit": _git_rev_parse(llama_repo) if llama_repo is not None else None,
            "llama_cpp_dirty": _git_dirty(llama_repo) if llama_repo is not None else None,
        },
        "config": {
            "draft_max": int(args.draft_max),
            "max_tokens": int(args.max_tokens),
            "prompt": str(args.prompt),
            "token_repeat": bool(args.token_repeat),
            "prompt_tokens": int(args.prompt_tokens),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "min_p": float(args.min_p),
            "seed": int(args.seed),
        },
        "server_command": server_command,
        "rocprof_command": rocprof_command,
        "terminate_status": terminate_status,
        "rocprof_return_code": return_code,
        "start_error": start_error,
        "request_error": request_error,
        "raw_root": str(raw_root),
        "server_log": str(log_path),
        "response_json": str(response_path) if response is not None else None,
        "stage_timings_jsonl": str(stage_path),
        "stage_timing_summary": _summarize_stage_timings(stage_path),
        "kernel_trace_csv": str(kernel_csv) if kernel_csv is not None else None,
        "kernel_summary": kernel_summary,
        "request_summary": request_summary,
        "notes": [
            "Whole-process trace: includes model load, prompt prefill, target verify, draft, sampling, and shutdown kernels.",
            "Use stage_timing_summary for llama.cpp MTP stage windows; use kernel_summary only as a family proxy unless ROCTX markers are added upstream.",
        ],
    }


def _print_summary(artifact: dict[str, Any]) -> None:
    req = artifact.get("request_summary") or {}
    kernel = artifact.get("kernel_summary") or {}
    print(
        "[llamacpp-mtp-rocprof] "
        f"status={artifact.get('status')} "
        f"terminate={artifact.get('terminate_status')} "
        f"tps={req.get('predicted_per_second')} "
        f"acc/out={req.get('accepted_per_output')} "
        f"kernel_csv={artifact.get('kernel_trace_csv')}"
    )
    if kernel:
        print("\n=== KERNEL BUCKETS ===")
        print(f"{'bucket':28s} {'dispatches':>10s} {'total_ms':>10s} {'share':>8s}")
        for row in kernel.get("buckets", [])[: int(artifact.get("top") or 20)]:
            print(
                f"{row['bucket'][:28]:28s} "
                f"{int(row['dispatches']):10d} "
                f"{float(row['total_ms']):10.3f} "
                f"{float(row['share_of_total']) * 100.0:7.1f}%"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER_BIN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--alias", default="qwen36-35b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--flash-attn", default="on")
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--reasoning", default="off")
    parser.add_argument("--draft-max", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="Write a Python function add(a, b).")
    parser.add_argument("--token-repeat", action="store_true")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--server-extra-arg", action="append", default=[])
    parser.add_argument("--rocprofv3", default="rocprofv3")
    parser.add_argument("--server-start-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--profiler-finalize-timeout", type=float, default=90.0)
    parser.add_argument("--hardware", default="AMD Radeon 8060S / Ryzen AI Max+ 395 (gfx1151)")
    parser.add_argument("--label", default="llamacpp-hip-mtp-whole-run")
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/hipengine-llamacpp-mtp-rocprof"))
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / f"{date.today().isoformat()}-llamacpp-mtp-rocprof.json",
    )
    args = parser.parse_args(argv)

    artifact = run_profile(args)
    artifact["top"] = int(args.top)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"[llamacpp-mtp-rocprof] wrote {args.out}")
    _print_summary(artifact)
    return 0 if artifact.get("request_error") is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
