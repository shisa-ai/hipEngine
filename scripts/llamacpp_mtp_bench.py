#!/usr/bin/env python3
"""Run llama.cpp base-vs-MTP external comparison diagnostics.

This script is intentionally outside hipEngine's runtime path. It starts a
local llama-server, runs deterministic requests with and without bundled MTP,
and writes one compact JSON artifact that can be compared against hipEngine
diagnostic rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl"
DEFAULT_MODEL = "/models/gguf/Qwen3.6-27B-Q4_K_M.gguf"
DEFAULT_SERVER_BIN = "/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-server"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", default=DEFAULT_SERVER_BIN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--alias", default="qwen36")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--flash-attn", default="on")
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--draft-max", type=int, default=2)
    parser.add_argument("--mode", choices=("base", "mtp", "both"), default="both")
    parser.add_argument("--protocol", choices=("natural", "token-repeat", "both"), default="both")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--shapes", nargs="+", default=["512/128", "4096/128"])
    parser.add_argument("--server-start-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for server logs. Defaults to a sibling directory next to --output.",
    )
    args = parser.parse_args()

    modes = ["base", "mtp"] if args.mode == "both" else [args.mode]
    protocols = (
        ["natural", "token_repeat"]
        if args.protocol == "both"
        else [args.protocol.replace("-", "_")]
    )
    logs_dir = args.log_dir or (
        args.output.with_suffix("").parent / (args.output.with_suffix("").name + "-logs")
    )
    logs_dir.mkdir(parents=True, exist_ok=True)

    artifact: dict[str, Any] = {
        "schema": 1,
        "status": "diagnostic_retained",
        "performance_claim": False,
        "kind": "llamacpp_mtp_external_comparison",
        "timestamp": _utc_timestamp(),
        "config": _config_json(args),
        "software": {
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "hipengine_commit": _git_rev_parse(REPO_ROOT),
            "hipengine_dirty": _git_dirty(REPO_ROOT),
            "llama_cpp_commit": _git_rev_parse(Path(args.server_bin).resolve().parents[2]),
            "llama_cpp_dirty": _git_dirty(Path(args.server_bin).resolve().parents[2]),
        },
        "runs": {},
        "summary": {},
        "notes": [
            "External comparison diagnostic; no hipEngine correctness gate is implied.",
            "Natural prompt MTP can produce different output hashes from base even at "
            "temperature=0.",
            "Token-repeat prompts are artificial and can overstate MTP acceptance versus "
            "natural prompts.",
        ],
    }

    server_process: subprocess.Popen[bytes] | None = None
    try:
        for mode in modes:
            log_path = logs_dir / f"server-{mode}.log"
            command = _server_command(args, mode)
            with log_path.open("wb") as log:
                server_process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                _wait_for_health(args.host, args.port, args.server_start_timeout)
                mode_payload: dict[str, Any] = {
                    "server_command": command,
                    "server_log": str(log_path),
                    "protocols": {},
                }
                if "natural" in protocols:
                    mode_payload["protocols"]["natural"] = _run_natural(args)
                if "token_repeat" in protocols:
                    mode_payload["protocols"]["token_repeat"] = _run_token_repeat(args)
                artifact["runs"][mode] = mode_payload
            finally:
                _terminate(server_process)
                server_process = None
    finally:
        if server_process is not None:
            _terminate(server_process)

    artifact["summary"] = _summarize_artifact(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(_summary_text(artifact))
    return 0


def _server_command(args: argparse.Namespace, mode: str) -> list[str]:
    cmd = [
        args.server_bin,
        "-m",
        args.model,
        "-ngl",
        str(args.gpu_layers),
        "-fa",
        args.flash_attn,
        "-ctk",
        args.cache_type_k,
        "-ctv",
        args.cache_type_v,
        "-c",
        str(args.ctx_size),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--alias",
        args.alias,
        "--no-cache-prompt",
    ]
    if mode == "mtp":
        cmd.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.draft_max)])
    return cmd


def _run_natural(args: argparse.Namespace) -> dict[str, Any]:
    prompts = _read_prompts(args.prompts)
    _post_json(
        args,
        "/v1/chat/completions",
        {
            "model": args.alias,
            "messages": [{"role": "user", "content": "Write a Python function add(a, b)."}],
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "max_tokens": 32,
            "seed": args.seed,
            "stream": False,
            "cache_prompt": False,
        },
        timeout=args.request_timeout,
    )
    rows = []
    for prompt in prompts:
        payload = {
            "model": args.alias,
            "messages": prompt["messages"],
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "stream": False,
            "cache_prompt": False,
        }
        t0 = time.perf_counter()
        resp = _post_json(args, "/v1/chat/completions", payload, timeout=args.request_timeout)
        wall_s = time.perf_counter() - t0
        content = _chat_content(resp)
        timings = resp.get("timings") or {}
        rows.append(
            {
                "id": prompt["id"],
                "category": prompt.get("category", "uncategorized"),
                "wall_s": wall_s,
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "content_chars": len(content),
                "timings": timings,
                "draft_acceptance": _draft_acceptance(timings),
            }
        )
    return {
        "prompt_file": str(args.prompts),
        "request_count": len(rows),
        "rows": rows,
        "summary": _summarize_rows(rows),
        "category_summary": _summarize_by_category(rows),
    }


def _run_token_repeat(args: argparse.Namespace) -> dict[str, Any]:
    _post_json(
        args,
        "/completion",
        _completion_payload(args, prompt_len=32, decode_tokens=8),
        timeout=args.request_timeout,
    )
    rows = []
    for shape in args.shapes:
        prompt_len, decode_tokens = _parse_shape(shape)
        payload = _completion_payload(args, prompt_len=prompt_len, decode_tokens=decode_tokens)
        t0 = time.perf_counter()
        resp = _post_json(args, "/completion", payload, timeout=args.request_timeout)
        wall_s = time.perf_counter() - t0
        timings = resp.get("timings") or {}
        rows.append(
            {
                "shape": shape,
                "prompt_len_requested": prompt_len,
                "decode_tokens_requested": decode_tokens,
                "tokens_evaluated": resp.get("tokens_evaluated"),
                "tokens_predicted": timings.get("predicted_n"),
                "prompt_per_second": timings.get("prompt_per_second"),
                "predicted_per_second": timings.get("predicted_per_second"),
                "prompt_ms": timings.get("prompt_ms"),
                "predicted_ms": timings.get("predicted_ms"),
                "draft_n": timings.get("draft_n"),
                "draft_n_accepted": timings.get("draft_n_accepted"),
                "draft_acceptance": _draft_acceptance(timings),
                "wall_s": wall_s,
                "stop_type": resp.get("stop_type"),
                "truncated": resp.get("truncated"),
                "timings": timings,
            }
        )
    return {
        "token_id": args.token_id,
        "rows": rows,
        "summary": _summarize_token_repeat(rows),
    }


def _completion_payload(
    args: argparse.Namespace,
    *,
    prompt_len: int,
    decode_tokens: int,
) -> dict[str, Any]:
    return {
        "prompt": [args.token_id] * prompt_len,
        "n_predict": decode_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "seed": args.seed,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
    }


def _read_prompts(path: Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" not in item or "messages" not in item:
                raise ValueError(f"{path}:{line_no}: expected id and messages")
            prompts.append(item)
    if not prompts:
        raise ValueError(f"{path} did not contain prompts")
    return prompts


def _post_json(
    args: argparse.Namespace,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{args.host}:{args.port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_health(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://{host}:{port}/health"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status < 500:
                    return
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(2.0)
    raise TimeoutError(f"server did not become healthy within {timeout}s: {last_error}")


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20.0)


def _chat_content(resp: dict[str, Any]) -> str:
    choice = resp["choices"][0]
    msg = choice.get("message") or {}
    return (msg.get("reasoning_content") or "") + (msg.get("content") or "")


def _draft_acceptance(timings: dict[str, Any]) -> float | None:
    draft_n = timings.get("draft_n") or 0
    draft_accepted = timings.get("draft_n_accepted") or 0
    return (draft_accepted / draft_n) if draft_n else None


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred_n = sum((row.get("timings", {}).get("predicted_n") or 0) for row in rows)
    pred_ms = sum((row.get("timings", {}).get("predicted_ms") or 0.0) for row in rows)
    pred = [
        row.get("timings", {}).get("predicted_per_second")
        for row in rows
        if row.get("timings", {}).get("predicted_per_second")
    ]
    draft_n = sum((row.get("timings", {}).get("draft_n") or 0) for row in rows)
    draft_acc = sum((row.get("timings", {}).get("draft_n_accepted") or 0) for row in rows)
    return {
        "requests": len(rows),
        "predicted_per_second_median": statistics.median(pred) if pred else None,
        "predicted_per_second_weighted": (1000.0 * pred_n / pred_ms) if pred_ms else None,
        "draft_n": draft_n,
        "draft_n_accepted": draft_acc,
        "draft_acceptance": (draft_acc / draft_n) if draft_n else None,
    }


def _summarize_by_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("category", "uncategorized"), []).append(row)
    return {category: _summarize_rows(items) for category, items in sorted(grouped.items())}


def _summarize_token_repeat(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "weighted_predicted_per_second": _weighted_tps(rows, "tokens_predicted", "predicted_ms"),
        "weighted_prompt_per_second": _weighted_tps(rows, "tokens_evaluated", "prompt_ms"),
        "draft_acceptance": _weighted_draft_acceptance(rows),
    }


def _weighted_tps(rows: list[dict[str, Any]], n_key: str, ms_key: str) -> float | None:
    total_n = sum((row.get(n_key) or 0) for row in rows)
    total_ms = sum((row.get(ms_key) or 0.0) for row in rows)
    return (1000.0 * total_n / total_ms) if total_ms else None


def _weighted_draft_acceptance(rows: list[dict[str, Any]]) -> float | None:
    draft_n = sum((row.get("draft_n") or 0) for row in rows)
    draft_acc = sum((row.get("draft_n_accepted") or 0) for row in rows)
    return (draft_acc / draft_n) if draft_n else None


def _summarize_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    runs = artifact.get("runs", {})
    for protocol in ("natural", "token_repeat"):
        if "base" in runs and "mtp" in runs:
            base = runs["base"]["protocols"].get(protocol)
            mtp = runs["mtp"]["protocols"].get(protocol)
            if base and mtp:
                base_tps = _summary_tps(base["summary"], protocol)
                mtp_tps = _summary_tps(mtp["summary"], protocol)
                summary[protocol] = {
                    "base_weighted_predicted_per_second": base_tps,
                    "mtp_weighted_predicted_per_second": mtp_tps,
                    "speedup": (mtp_tps / base_tps) if base_tps and mtp_tps else None,
                    "mtp_draft_acceptance": mtp["summary"].get("draft_acceptance"),
                }
    return summary


def _summary_tps(summary: dict[str, Any], protocol: str) -> float | None:
    if protocol == "token_repeat":
        return summary.get("weighted_predicted_per_second")
    return summary.get("predicted_per_second_weighted")


def _summary_text(artifact: dict[str, Any]) -> str:
    lines = ["llama.cpp MTP diagnostic complete"]
    for protocol, row in artifact.get("summary", {}).items():
        speedup = row.get("speedup")
        acc = _fmt(row.get("mtp_draft_acceptance"), 3) if speedup else "-"
        lines.append(
            f"{protocol}: base={_fmt(row.get('base_weighted_predicted_per_second'))} "
            f"mtp={_fmt(row.get('mtp_weighted_predicted_per_second'))} "
            f"speedup={speedup:.3f}x acc={acc}"
        )
    return "\n".join(lines)


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _parse_shape(text: str) -> tuple[int, int]:
    def parse_part(part: str) -> int:
        part = part.strip().lower()
        if part.endswith("k"):
            return int(part[:-1]) * 1024
        return int(part)

    try:
        left, right = text.split("/", 1)
        return parse_part(left), parse_part(right)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}; expected prompt/decode") from exc


def _config_json(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "server_bin": args.server_bin,
        "model": args.model,
        "alias": args.alias,
        "host": args.host,
        "port": args.port,
        "ctx_size": args.ctx_size,
        "gpu_layers": args.gpu_layers,
        "flash_attn": args.flash_attn,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "draft_max": args.draft_max,
        "protocol": args.protocol,
        "prompts": str(args.prompts),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "token_id": args.token_id,
        "shapes": args.shapes,
    }


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


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
