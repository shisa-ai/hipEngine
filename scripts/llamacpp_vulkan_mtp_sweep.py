#!/usr/bin/env python3
"""Run a llama.cpp Vulkan draft-MTP sweep with the shared MTP prompt suite.

This script starts one llama-server per mode, drives it with scripts/mtp-bench.py,
and writes a compact summary next to the raw server/client logs. It is intended
for external comparison diagnostics against hipEngine MTP rows.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from urllib.error import URLError


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLAMA_DIR = Path("/home/lhl/llama.cpp/llama.cpp-vulkan")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "fixtures" / "llamacpp_mtp_bench_prompts.json"
DEFAULT_ICD = Path("/usr/share/vulkan/icd.d/radeon_icd.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-dir", type=Path, default=DEFAULT_LLAMA_DIR)
    parser.add_argument("--server-bin", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpu", default="0", help="GGML_VK_VISIBLE_DEVICES value")
    parser.add_argument("--vulkan-icd", type=Path, default=DEFAULT_ICD)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", default="99")
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--draft-max-values", default="1,2,3,4")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-names", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--server-start-timeout", type=float, default=720.0)
    parser.add_argument("--alias", default="llama")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-base", type=int, default=0, help="0 chooses a free port per run")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--no-ignore-eos", action="store_true")
    parser.add_argument("--extra-server-arg", action="append", default=[])
    parser.add_argument("--extra-client-arg", action="append", default=[])
    args = parser.parse_args()

    llama_dir = args.llama_dir.resolve()
    server_bin = args.server_bin or (llama_dir / "build" / "bin" / "llama-server")
    out_dir = args.out_dir or Path("/tmp") / f"llamacpp-vulkan-mtp-sweep-{timestamp_slug()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1"] = "1"
    env["VK_ICD_FILENAMES"] = str(args.vulkan_icd)
    env["GGML_VK_VISIBLE_DEVICES"] = str(args.gpu)

    modes: list[tuple[str, int | None]] = []
    if not args.skip_base:
        modes.append(("base", None))
    for value in parse_csv_ints(args.draft_max_values):
        modes.append((f"b{value}", value))

    run_meta: dict[str, Any] = {
        "schema": 1,
        "kind": "llamacpp_vulkan_mtp_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "llama_dir": str(llama_dir),
        "server_bin": str(server_bin),
        "model": str(args.model),
        "gpu": str(args.gpu),
        "vulkan_icd": str(args.vulkan_icd),
        "max_tokens": int(args.max_tokens),
        "ctx_size": int(args.ctx_size),
        "gpu_layers": str(args.gpu_layers),
        "cache_type_k": str(args.cache_type_k),
        "cache_type_v": str(args.cache_type_v),
        "prompts_file": str(args.prompts_file),
        "prompt_names": args.prompt_names,
        "limit": args.limit,
        "llama_version": capture_version(server_bin, llama_dir, env),
        "git": capture_git(llama_dir),
        "runs": {},
    }

    for idx, (mode, draft_max) in enumerate(modes):
        port = choose_port(args.host) if args.port_base == 0 else args.port_base + idx
        result = run_one(args, env, llama_dir, server_bin, out_dir, mode, draft_max, port)
        run_meta["runs"][mode] = result

    summary = summarize(out_dir, [mode for mode, _ in modes])
    run_meta["summary"] = summary
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"out_dir {out_dir}")
    return 0


def run_one(
    args: argparse.Namespace,
    env: dict[str, str],
    llama_dir: Path,
    server_bin: Path,
    out_dir: Path,
    mode: str,
    draft_max: int | None,
    port: int,
) -> dict[str, Any]:
    server_log = out_dir / f"server-{mode}.log"
    client_log = out_dir / f"client-{mode}.log"
    result_json = out_dir / f"{mode}.json"

    server_cmd = [
        str(server_bin),
        "-m", str(args.model),
        "-ngl", str(args.gpu_layers),
        "-fa", "on",
        "-ctk", str(args.cache_type_k),
        "-ctv", str(args.cache_type_v),
        "-c", str(args.ctx_size),
        "--host", str(args.host),
        "--port", str(port),
        "--alias", str(args.alias),
        "--no-cache-prompt",
        "--cache-ram", "0",
    ]
    if draft_max is not None:
        server_cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_max)]
    server_cmd += list(args.extra_server_arg)

    (out_dir / f"cmd-server-{mode}.txt").write_text(shell_join(server_cmd) + "\n", encoding="utf-8")
    print(f"=== {mode} port={port} ===", flush=True)
    with server_log.open("wb") as log:
        proc = subprocess.Popen(server_cmd, cwd=llama_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_for_health(args.host, port, args.server_start_timeout, proc, server_log)
        client_cmd = [
            sys.executable,
            "scripts/mtp-bench.py",
            "--url", f"http://{args.host}:{port}",
            "--prompts-file", str(args.prompts_file),
            "--model", str(args.alias),
            "--seed", str(args.seed),
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--no-cache-prompt",
            "--max-tokens", str(args.max_tokens),
            "--timeout", str(args.timeout),
            "--out", str(result_json),
        ]
        if not args.no_ignore_eos:
            client_cmd.append("--ignore-eos")
        if args.prompt_names:
            client_cmd += ["--prompt-names", str(args.prompt_names)]
        if args.limit is not None:
            client_cmd += ["--limit", str(args.limit)]
        client_cmd += list(args.extra_client_arg)
        (out_dir / f"cmd-client-{mode}.txt").write_text(shell_join(client_cmd) + "\n", encoding="utf-8")
        client_env = os.environ.copy()
        client_env["PYTHONPATH"] = f"{REPO_ROOT}:{client_env.get('PYTHONPATH', '')}"
        with client_log.open("w", encoding="utf-8") as log:
            completed = subprocess.run(client_cmd, cwd=REPO_ROOT, env=client_env, text=True, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            tail = tail_text(client_log, 120)
            raise RuntimeError(f"client failed for {mode} with exit {completed.returncode}\n{tail}")
    finally:
        terminate(proc)

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    return {
        "server_command": server_cmd,
        "client_command": client_cmd,
        "server_log": str(server_log),
        "client_log": str(client_log),
        "result_json": str(result_json),
        "aggregate": payload.get("aggregate"),
    }


def summarize(out_dir: Path, modes: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    base_mean: float | None = None
    base_wall: float | None = None
    for mode in modes:
        payload = json.loads((out_dir / f"{mode}.json").read_text(encoding="utf-8"))
        results = payload.get("results") or []
        total_pred = sum(int(row.get("predicted_n") or 0) for row in results)
        total_wall = sum(float(row.get("wall_s") or 0.0) for row in results)
        per_request = [float(row.get("predicted_per_second") or 0.0) for row in results]
        total_draft = sum(int(row.get("draft_n") or 0) for row in results)
        total_accepted = sum(int(row.get("draft_n_accepted") or 0) for row in results)
        wall_tps = total_pred / total_wall if total_wall > 0.0 else None
        mean_tps = statistics.fmean(per_request) if per_request else None
        median_tps = statistics.median(per_request) if per_request else None
        accept_rate = total_accepted / total_draft if total_draft else None
        accepted_per_output = total_accepted / total_pred if total_pred else None
        draft_per_output = total_draft / total_pred if total_pred else None
        if mode == "base":
            base_mean = mean_tps
            base_wall = wall_tps
        rows.append({
            "mode": mode,
            "total_predicted": total_pred,
            "total_wall_s": total_wall,
            "wall_tps": wall_tps,
            "mean_tps": mean_tps,
            "median_tps": median_tps,
            "total_draft": total_draft,
            "total_accepted": total_accepted,
            "accept_rate": accept_rate,
            "accepted_per_output": accepted_per_output,
            "draft_per_output": draft_per_output,
            "speedup_mean_vs_base": (mean_tps / base_mean) if mean_tps and base_mean else None,
            "speedup_wall_vs_base": (wall_tps / base_wall) if wall_tps and base_wall else None,
        })
    return {"out_dir": str(out_dir), "rows": rows}


def print_summary(summary: dict[str, Any]) -> None:
    print("\nSUMMARY")
    print(
        f"{'mode':<6} {'mean_tps':>10} {'wall_tps':>10} {'speedup':>8} "
        f"{'draft':>8} {'accepted':>9} {'acc/draft':>10} {'acc/output':>10}"
    )
    for row in summary.get("rows", []):
        accept = row.get("accept_rate")
        print(
            f"{row['mode']:<6} "
            f"{fmt(row.get('mean_tps')):>10} "
            f"{fmt(row.get('wall_tps')):>10} "
            f"{fmt(row.get('speedup_mean_vs_base'), 3):>8} "
            f"{int(row.get('total_draft') or 0):>8} "
            f"{int(row.get('total_accepted') or 0):>9} "
            f"{fmt(accept, 3):>10} "
            f"{fmt(row.get('accepted_per_output'), 3):>10}"
        )


def parse_csv_ints(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one draft max value")
    return values


def choose_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_health(host: str, port: int, timeout: float, proc: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://{host}:{port}/health"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before ready, tail of {log_path}:\n{tail_text(log_path, 160)}")
        try:
            with urlopen(url, timeout=2.0) as resp:
                if resp.status < 500:
                    return
        except URLError as exc:
            last_error = exc
        time.sleep(2.0)
    raise TimeoutError(f"server did not become ready within {timeout}s: {last_error}\n{tail_text(log_path, 160)}")


def terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20.0)


def capture_version(server_bin: Path, cwd: Path, env: dict[str, str]) -> str | None:
    try:
        return subprocess.check_output([str(server_bin), "--version"], cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except Exception:
        return None


def capture_git(repo: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=repo, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        except Exception:
            return None
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "describe": run(["git", "describe", "--tags", "--always", "--dirty"]),
        "status_short": run(["git", "status", "--short"]),
    }


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def tail_text(path: Path, lines: int) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(data[-lines:])
    except Exception as exc:
        return f"<failed to read {path}: {exc}>"


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
