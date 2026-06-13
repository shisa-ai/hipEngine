#!/usr/bin/env python3
"""llama.cpp Vulkan server concurrency sweep for Qwen3.6 35B.

Runs llama-server once per (concurrency, rep), sends concurrent /completion
requests with exact 512-token prompt-id rows from the hipEngine fixture, and
summarizes aggregate decode tok/s across the batch.

The primary aggregate decode metric is sum(predicted tokens) divided by the
maximum llama.cpp per-request predicted_ms.  This mirrors a batched decode wall
window more closely than end-to-end HTTP wall time, which includes prompt eval
and client/server scheduling.  End-to-end aggregate tok/s is recorded too.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REPO = Path("/home/lhl/llama.cpp/llama.cpp-vulkan")
DEFAULT_SERVER_BIN = DEFAULT_REPO / "build" / "bin" / "llama-server"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf")
DEFAULT_FIXTURE = Path("/tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json")
DEFAULT_ICD = "/usr/share/vulkan/icd.d/radeon_icd.json"


def _load_prompt_rows(path: Path, prompt_length: int) -> list[list[int]]:
    data = json.loads(path.read_text())
    ids = data.get("prompt_ids")
    if not isinstance(ids, list):
        raise ValueError(f"{path} does not contain list prompt_ids")
    row_lengths = data.get("row_lengths")
    if isinstance(row_lengths, list) and row_lengths:
        rows = []
        offset = 0
        for n in row_lengths:
            n_int = int(n)
            rows.append([int(x) for x in ids[offset : offset + n_int]])
            offset += n_int
        return [row[:prompt_length] for row in rows if len(row) >= prompt_length]
    if len(ids) % prompt_length != 0:
        raise ValueError(f"flat prompt_ids length {len(ids)} is not divisible by prompt length {prompt_length}")
    return [list(map(int, ids[i : i + prompt_length])) for i in range(0, len(ids), prompt_length)]


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        parsed = json.loads(response.read())
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object response, got {type(parsed).__name__}")
    return parsed


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_from_response(index: int, response: dict[str, Any], wall_s: float) -> dict[str, Any]:
    timings = response.get("timings") or {}
    if not isinstance(timings, dict):
        timings = {}
    pred_n = _number(timings.get("predicted_n"))
    if pred_n is None:
        pred_n = _number(response.get("tokens_predicted")) or 0.0
    pred_ms = _number(timings.get("predicted_ms"))
    pred_per_s = _number(timings.get("predicted_per_second"))
    prompt_n = _number(timings.get("prompt_n"))
    if prompt_n is None:
        prompt_n = _number(response.get("tokens_evaluated")) or 0.0
    prompt_ms = _number(timings.get("prompt_ms"))
    return {
        "index": index,
        "wall_s": wall_s,
        "predicted_n": int(pred_n),
        "predicted_ms": pred_ms,
        "predicted_per_second": pred_per_s,
        "prompt_n": int(prompt_n),
        "prompt_ms": prompt_ms,
        "stop": response.get("stop"),
    }


def _wait_ready(port: int, proc: subprocess.Popen[str], log_path: Path, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    health_url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
            raise RuntimeError(f"llama-server exited with {proc.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if 200 <= response.status < 500:
                    return
        except Exception:
            pass
        time.sleep(0.25)
    tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
    raise TimeoutError(f"llama-server did not become ready on port {port}\n{tail}")


def _start_server(args: argparse.Namespace, c: int, rep: int, log_path: Path) -> subprocess.Popen[str]:
    ctx_size = args.ctx_per_seq * c
    port = args.port_base + c * 10 + rep
    cmd = [
        str(args.server_bin),
        "-m",
        str(args.model),
        "-ngl",
        str(args.n_gpu_layers),
        "-fa",
        args.flash_attn,
        "-ctk",
        args.cache_type_k,
        "-ctv",
        args.cache_type_v,
        "-c",
        str(ctx_size),
        "-np",
        str(c),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-webui",
        "--no-cache-prompt",
        "--cache-ram",
        "0",
        "--ctx-checkpoints",
        "0",
        "--metrics",
    ]
    if args.no_warmup:
        cmd.append("--no-warmup")
    if args.extra_server_arg:
        cmd.extend(args.extra_server_arg)

    env = os.environ.copy()
    env["DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1"] = "1"
    env["VK_DRIVER_FILES"] = args.vk_driver_files
    env["GGML_VK_VISIBLE_DEVICES"] = str(args.gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=args.repo,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    proc._hipengine_log_file = log_f  # type: ignore[attr-defined]
    proc._hipengine_command = cmd  # type: ignore[attr-defined]
    proc._hipengine_port = port  # type: ignore[attr-defined]
    _wait_ready(port, proc, log_path, args.server_ready_timeout)
    return proc


def _stop_server(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=20)
    log_f = getattr(proc, "_hipengine_log_file", None)
    if log_f is not None:
        log_f.close()


def _run_client(args: argparse.Namespace, rows: list[list[int]], c: int, port: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/completion"
    barrier = threading.Barrier(c + 1)

    def one(i: int) -> dict[str, Any]:
        payload = {
            "prompt": rows[i],
            "n_predict": args.decode_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "seed": args.seed + i,
            "ignore_eos": True,
            "cache_prompt": False,
            "stream": False,
        }
        barrier.wait()
        t0 = time.perf_counter()
        response = _post_json(url, payload, args.request_timeout)
        return _record_from_response(i, response, time.perf_counter() - t0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
        futures = [pool.submit(one, i) for i in range(c)]
        t0 = time.perf_counter()
        barrier.wait()
        records = [f.result() for f in futures]
        wall_s = time.perf_counter() - t0

    total_pred = sum(int(r["predicted_n"]) for r in records)
    predicted_ms_values = [float(r["predicted_ms"]) for r in records if r.get("predicted_ms") is not None]
    max_pred_ms = max(predicted_ms_values) if predicted_ms_values else None
    aggregate_decode_tok_s_timing = total_pred / (max_pred_ms / 1000.0) if max_pred_ms and max_pred_ms > 0 else None
    return {
        "concurrency": c,
        "wall_s": wall_s,
        "total_predicted": total_pred,
        "max_predicted_ms": max_pred_ms,
        "aggregate_decode_tok_s_timing": aggregate_decode_tok_s_timing,
        "aggregate_tok_s_e2e": total_pred / wall_s if wall_s > 0 else None,
        "per_request_decode_tok_s_timing": (aggregate_decode_tok_s_timing / c) if aggregate_decode_tok_s_timing else None,
        "records": sorted(records, key=lambda r: r["index"]),
    }


def _median(values: list[float | None]) -> float | None:
    real = [float(v) for v in values if v is not None]
    return statistics.median(real) if real else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER_BIN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--vk-driver-files", default=DEFAULT_ICD)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--ctx-per-seq", type=int, default=1024)
    parser.add_argument("--concurrencies", default="1,2,4,8")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--port-base", type=int, default=18100)
    parser.add_argument("--n-gpu-layers", default="99")
    parser.add_argument("--flash-attn", default="on")
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--extra-server-arg", action="append")
    parser.add_argument("--server-ready-timeout", type=float, default=240.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/llamacpp-vulkan-concurrency-sweep"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    rows = _load_prompt_rows(args.fixture, args.prompt_length)
    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    if max(concurrencies) > len(rows):
        raise ValueError(f"fixture only has {len(rows)} rows, need {max(concurrencies)}")

    summary_rows: dict[str, Any] = {}
    for c in concurrencies:
        reps: list[dict[str, Any]] = []
        for rep in range(args.reps):
            log_path = args.work_dir / f"server-c{c}-r{rep}.log"
            proc = _start_server(args, c, rep, log_path)
            command = " ".join(str(x) for x in getattr(proc, "_hipengine_command"))
            port = int(getattr(proc, "_hipengine_port"))
            try:
                result = _run_client(args, rows, c, port)
                result["server_log"] = str(log_path)
                result["server_command"] = command
                reps.append(result)
                agg = result.get("aggregate_decode_tok_s_timing")
                e2e = result.get("aggregate_tok_s_e2e")
                print(f"c{c} rep{rep}: decode={agg:.2f} e2e={e2e:.2f} wall={result['wall_s']:.2f}s", flush=True)
            finally:
                _stop_server(proc)
        aggregate_runs = [r.get("aggregate_decode_tok_s_timing") for r in reps]
        e2e_runs = [r.get("aggregate_tok_s_e2e") for r in reps]
        agg_median = _median(aggregate_runs)
        summary_rows[str(c)] = {
            "concurrency": c,
            "decode_tok_s_aggregate_median": agg_median,
            "decode_tok_s_per_request_median": (agg_median / c) if agg_median is not None else None,
            "decode_tok_s_aggregate_runs": aggregate_runs,
            "e2e_tok_s_aggregate_median": _median(e2e_runs),
            "e2e_tok_s_aggregate_runs": e2e_runs,
            "path": "llama-server /completion continuous batching",
            "rep_artifacts": reps,
        }
        print(
            f"== c{c}: agg_median={summary_rows[str(c)]['decode_tok_s_aggregate_median']:.2f} "
            f"per={summary_rows[str(c)]['decode_tok_s_per_request_median']:.2f}",
            flush=True,
        )

    summary = {
        "kind": "llamacpp_vulkan_concurrency_decode_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "host": f"GGML_VK_VISIBLE_DEVICES={args.gpu}",
        "model": str(args.model),
        "fixture": str(args.fixture),
        "shape": {
            "prompt_length": args.prompt_length,
            "decode_tokens": args.decode_tokens,
            "ctx_per_seq": args.ctx_per_seq,
            "reps": args.reps,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
        },
        "methodology": {
            "server": "llama-server restarted per (concurrency, rep) with -np concurrency and -c ctx_per_seq*concurrency",
            "client": "concurrent POST /completion with exact prompt token-id rows",
            "primary_metric": "sum(predicted_n) / max(predicted_ms) across the concurrent responses",
            "secondary_metric": "sum(predicted_n) / client wall time including prompt eval and HTTP",
            "aggregate_across_runs": "median",
        },
        "rows": summary_rows,
    }
    print("FINAL " + json.dumps(summary), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
