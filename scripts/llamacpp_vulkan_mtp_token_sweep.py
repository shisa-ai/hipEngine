#!/usr/bin/env python3
"""Run llama.cpp Vulkan MTP with exact prompt token IDs.

Prompts are rendered/tokenized with the same code path used by hipEngine's MTP
prompt-suite harness, then sent to llama-server's /completion endpoint as token
ID arrays. This avoids llama-server applying a different chat template and makes
prompt rendering comparable to hipEngine runs.
"""

from __future__ import annotations

import argparse
import importlib.util
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
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLAMA_DIR = Path("/home/lhl/llama.cpp/llama.cpp-vulkan")
DEFAULT_LLAMA_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf")
DEFAULT_TOKENIZER_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "fixtures" / "llamacpp_mtp_bench_prompts.json"
DEFAULT_ICD = Path("/usr/share/vulkan/icd.d/radeon_icd.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-dir", type=Path, default=DEFAULT_LLAMA_DIR)
    parser.add_argument("--server-bin", type=Path)
    parser.add_argument("--llama-model", type=Path, default=DEFAULT_LLAMA_MODEL)
    parser.add_argument("--tokenizer-model", type=Path, default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-render", choices=("raw", "qwen_chat_thinking_off", "qwen_chat_thinking_on"), default="qwen_chat_thinking_on")
    parser.add_argument("--prompt-names")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu", default="0", help="GGML_VK_VISIBLE_DEVICES value")
    parser.add_argument("--vulkan-icd", type=Path, default=DEFAULT_ICD)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", default="99")
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--draft-max-values", default="1,2,3,4")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-base", type=int, default=0, help="0 chooses a free port per run")
    parser.add_argument("--alias", default="llama")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--server-start-timeout", type=float, default=720.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--no-ignore-eos", action="store_true")
    parser.add_argument("--extra-server-arg", action="append", default=[])
    args = parser.parse_args()

    economics = load_economics_module()
    _load_prompt_encoder = economics._load_prompt_encoder
    _load_prompt_suite = economics._load_prompt_suite
    _select_prompts = economics._select_prompts

    llama_dir = args.llama_dir.resolve()
    server_bin = args.server_bin or (llama_dir / "build" / "bin" / "llama-server")
    out_dir = args.out_dir or Path("/tmp") / f"llamacpp-vulkan-mtp-token-sweep-{timestamp_slug()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    suite = _load_prompt_suite(args.prompts_file)
    prompts = _select_prompts(suite, names_csv=args.prompt_names, limit=args.limit)
    encoder = _load_prompt_encoder(args.tokenizer_model, args.prompt_render)
    encoded_prompts = []
    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for prompt in prompts:
        enc = encoder.encode(prompt["prompt"])
        safe = safe_name(prompt["name"])
        (prompt_dir / f"{safe}.tokens.txt").write_text(",".join(str(x) for x in enc.token_ids), encoding="utf-8")
        (prompt_dir / f"{safe}.txt").write_text(enc.rendered_text, encoding="utf-8")
        encoded_prompts.append({
            "name": str(prompt["name"]),
            "source_prompt": str(prompt["prompt"]),
            "rendered_prompt": enc.rendered_text,
            "token_ids": enc.token_ids,
            "token_count": len(enc.token_ids),
        })

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
        "kind": "llamacpp_vulkan_mtp_token_prompt_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "llama_dir": str(llama_dir),
        "server_bin": str(server_bin),
        "llama_model": str(args.llama_model),
        "tokenizer_model": str(args.tokenizer_model),
        "prompt_render": str(args.prompt_render),
        "tokenization": encoder.tokenization,
        "prompts_file": str(args.prompts_file),
        "prompt_count": len(encoded_prompts),
        "prompt_token_counts": {p["name"]: p["token_count"] for p in encoded_prompts},
        "gpu": str(args.gpu),
        "vulkan_icd": str(args.vulkan_icd),
        "max_tokens": int(args.max_tokens),
        "ctx_size": int(args.ctx_size),
        "llama_version": capture_version(server_bin, llama_dir, env),
        "git": capture_git(llama_dir),
        "runs": {},
    }

    for idx, (mode, draft_max) in enumerate(modes):
        port = choose_port(args.host) if args.port_base == 0 else args.port_base + idx
        run_meta["runs"][mode] = run_one(args, env, llama_dir, server_bin, out_dir, encoded_prompts, mode, draft_max, port)

    summary = summarize(out_dir, [mode for mode, _ in modes])
    run_meta["summary"] = summary
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"out_dir {out_dir}")
    return 0


def run_one(
    args: argparse.Namespace,
    env: dict[str, str],
    llama_dir: Path,
    server_bin: Path,
    out_dir: Path,
    prompts: list[dict[str, Any]],
    mode: str,
    draft_max: int | None,
    port: int,
) -> dict[str, Any]:
    server_log = out_dir / f"server-{mode}.log"
    result_json = out_dir / f"{mode}.json"
    server_cmd = [
        str(server_bin),
        "-m", str(args.llama_model),
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
        rows = []
        for prompt in prompts:
            payload = completion_payload(args, prompt["token_ids"])
            started = time.perf_counter()
            response = post_json(f"http://{args.host}:{port}/completion", payload, args.timeout)
            wall_s = time.perf_counter() - started
            row = record_from_response(prompt["name"], prompt["token_count"], response, wall_s)
            rows.append(row)
            print(format_result_line(row), flush=True)
        artifact = {"results": rows, "aggregate": aggregate(rows)}
        result_json.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print("Aggregate:", json.dumps(artifact["aggregate"], indent=2), flush=True)
    finally:
        terminate(proc)
    return {
        "server_command": server_cmd,
        "server_log": str(server_log),
        "result_json": str(result_json),
        "aggregate": json.loads(result_json.read_text(encoding="utf-8")).get("aggregate"),
    }


def completion_payload(args: argparse.Namespace, token_ids: list[int]) -> dict[str, Any]:
    payload = {
        "prompt": [int(x) for x in token_ids],
        "n_predict": int(args.max_tokens),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "min_p": float(args.min_p),
        "seed": int(args.seed),
        "cache_prompt": False,
        "stream": False,
    }
    if not args.no_ignore_eos:
        payload["ignore_eos"] = True
    return payload


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def record_from_response(name: str, prompt_tokens: int, response: dict[str, Any], wall_s: float) -> dict[str, Any]:
    timings = response.get("timings") or {}
    predicted_n = first_number(timings.get("predicted_n"), response.get("tokens_predicted"), response.get("tokens_evaluated"))
    predicted_per_second = first_number(timings.get("predicted_per_second"))
    if predicted_per_second is None and predicted_n is not None and timings.get("predicted_ms"):
        predicted_per_second = 1000.0 * float(predicted_n) / float(timings["predicted_ms"])
    if predicted_per_second is None and predicted_n is not None and wall_s > 0:
        predicted_per_second = float(predicted_n) / wall_s
    draft_n = int(first_number(timings.get("draft_n"), 0) or 0)
    draft_n_accepted = int(first_number(timings.get("draft_n_accepted"), 0) or 0)
    row = {
        "name": name,
        "prompt_tokens": int(prompt_tokens),
        "wall_s": round(wall_s, 3),
        "predicted_n": int(predicted_n or 0),
        "predicted_per_second": round(float(predicted_per_second or 0.0), 2),
        "prompt_per_second": first_number(timings.get("prompt_per_second")),
        "tokens_evaluated": response.get("tokens_evaluated"),
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "timings": timings,
    }
    row["accept_rate"] = round(draft_n_accepted / draft_n, 4) if draft_n else None
    row["accepted_per_output_token"] = round(draft_n_accepted / row["predicted_n"], 4) if row["predicted_n"] else None
    return row


def first_number(*values: Any) -> int | float | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def format_result_line(row: dict[str, Any]) -> str:
    accept = "n/a" if row["accept_rate"] is None else f"{row['accept_rate']:.3f}"
    density = "n/a" if row["accepted_per_output_token"] is None else f"{row['accepted_per_output_token']:.3f}"
    return (
        f"  {row['name']:<18} prompt={row['prompt_tokens']:>4} pred={row['predicted_n']:>4} "
        f"draft={row['draft_n']:>4} acc={row['draft_n_accepted']:>4} "
        f"rate={accept} density={density} tok/s={row['predicted_per_second']:.1f}"
    )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_pred = sum(int(row.get("predicted_n") or 0) for row in rows)
    total_draft = sum(int(row.get("draft_n") or 0) for row in rows)
    total_accepted = sum(int(row.get("draft_n_accepted") or 0) for row in rows)
    total_wall = sum(float(row.get("wall_s") or 0.0) for row in rows)
    return {
        "n_requests": len(rows),
        "total_predicted": total_pred,
        "total_draft": total_draft,
        "total_draft_accepted": total_accepted,
        "aggregate_accept_rate": round(total_accepted / total_draft, 4) if total_draft else None,
        "accepted_per_output_token": round(total_accepted / total_pred, 4) if total_pred else None,
        "wall_s_total": round(total_wall, 3),
    }


def summarize(out_dir: Path, modes: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    base_mean = None
    for mode in modes:
        payload = json.loads((out_dir / f"{mode}.json").read_text(encoding="utf-8"))
        results = payload.get("results") or []
        per_request = [float(row.get("predicted_per_second") or 0.0) for row in results]
        total_pred = sum(int(row.get("predicted_n") or 0) for row in results)
        total_draft = sum(int(row.get("draft_n") or 0) for row in results)
        total_acc = sum(int(row.get("draft_n_accepted") or 0) for row in results)
        mean_tps = statistics.fmean(per_request) if per_request else None
        if mode == "base":
            base_mean = mean_tps
        rows.append({
            "mode": mode,
            "mean_tps": mean_tps,
            "median_tps": statistics.median(per_request) if per_request else None,
            "total_predicted": total_pred,
            "total_draft": total_draft,
            "total_accepted": total_acc,
            "accept_rate": total_acc / total_draft if total_draft else None,
            "accepted_per_output_token": total_acc / total_pred if total_pred else None,
            "speedup_mean_vs_base": (mean_tps / base_mean) if mean_tps and base_mean else None,
        })
    return {"out_dir": str(out_dir), "rows": rows}


def print_summary(summary: dict[str, Any]) -> None:
    print("\nSUMMARY")
    print(f"{'mode':<6} {'mean_tps':>10} {'speedup':>8} {'draft':>8} {'accepted':>9} {'accept':>8} {'density':>8}")
    for row in summary.get("rows", []):
        print(
            f"{row['mode']:<6} {fmt(row.get('mean_tps')):>10} {fmt(row.get('speedup_mean_vs_base'), 3):>8} "
            f"{int(row.get('total_draft') or 0):>8} {int(row.get('total_accepted') or 0):>9} "
            f"{fmt(row.get('accept_rate'), 3):>8} {fmt(row.get('accepted_per_output_token'), 3):>8}"
        )


def load_economics_module() -> Any:
    path = REPO_ROOT / "scripts" / "mtp_prompt_suite_economics.py"
    spec = importlib.util.spec_from_file_location("hipengine_mtp_prompt_suite_economics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
            with request.urlopen(url, timeout=2.0) as resp:
                if resp.status < 500:
                    return
        except error.URLError as exc:
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


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def tail_text(path: Path, lines: int) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
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
