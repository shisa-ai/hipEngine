#!/usr/bin/env python3
"""Run the standardized llama.cpp HIP C1-C8 complete-wall engine matrix.

This is the in-tree version of the external comparator harness that produced
``benchmarks/results/2026-08-30-w7900-qwen38-q4km-c1c8-cross-engine.json``. That
packet recorded only the external script SHA-256, so the llama.cpp side of the
published matrix could not be re-run from this repository. This harness keeps the
identical protocol and makes the comparator reproducible:

* ten-prompt ``mtpbench-code-general-ja`` suite rendered as one user turn;
* greedy ``temperature=0, top_k=1, seed=1`` with ``cache_prompt=false``;
* one wave per (arm, width, prompt): every lane releases on a barrier and the
  cell wall is ``max(end) - min(start)``, so prompt processing, first token, and
  decode all sit inside a single boundary;
* a ``prefill`` arm at ``--prefill-tokens`` (default 1) plus an ``ar`` arm at
  ``--decode-tokens`` (default 24). Pass ``--decode-tokens 4`` for the short arm
  that separates steady-state decode from first-token cost, or 0 for a
  comparator-prefill-only packet;
* an optional built-in MTP arm (``--spec-draft-n-max N``) timed with the same
  boundary;
* per-lane anti-repetition guards and cross-lane content exactness;
* server parallel capacity fixed independently by ``--parallel`` (default 8),
  so a narrow ``--widths`` repeat preserves the published ``-np 8`` protocol.

The server runs as a child process and is terminated on exit. Output packets are
comparator evidence: they rank engines against each other and never become a
hipEngine topline row by themselves.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
SCHEMA = "hipengine.llamacpp_c1c8_engine_matrix.v1"
TIMING_BOUNDARY = (
    "per (arm,width,prompt) wave: sum of max(lane_end) - min(lane_start) over prompts"
)


def post(url: str, payload: dict[str, Any], *, timeout: float = 600.0) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error


def get(url: str, *, timeout: float = 2.0) -> bytes:
    with urlopen(url, timeout=timeout) as response:
        return response.read()


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        content = row["messages"][0]["content"]
        prompts.append(
            {
                "id": str(row["id"]),
                "category": str(row.get("category", "")),
                "rendered": (
                    "<|im_start|>user\n"
                    + content
                    + "<|im_end|>\n<|im_start|>assistant\n"
                ),
            }
        )
    return prompts


def _widths(raw: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in raw.split(",") if part.strip())
    if not values or any(value < 1 or value > 32 for value in values):
        raise argparse.ArgumentTypeError("widths must be positive integers <= 32")
    return values


def _parallel_capacity(widths: tuple[int, ...], parallel: int) -> int:
    """Keep server capacity fixed while selecting a measured width subset."""

    parallel = int(parallel)
    if parallel < max(int(width) for width in widths):
        raise ValueError("parallel capacity must be at least the largest measured width")
    return parallel


def _workload_arms(
    prefill_tokens: int,
    decode_tokens: int,
    spec_draft_n_max: int,
) -> tuple[tuple[str, int], ...]:
    """Resolve measured arms; zero disables an unrelated non-spec arm."""

    prefill_tokens = int(prefill_tokens)
    decode_tokens = int(decode_tokens)
    spec_draft_n_max = int(spec_draft_n_max)
    if prefill_tokens < 0 or decode_tokens < 0 or spec_draft_n_max < 0:
        raise ValueError("token counts and speculative depth must be non-negative")
    if spec_draft_n_max:
        if decode_tokens <= 0:
            raise ValueError("speculative measurement requires positive decode tokens")
        return (("mtp", decode_tokens),)
    arms: list[tuple[str, int]] = []
    if prefill_tokens:
        arms.append(("prefill", prefill_tokens))
    if decode_tokens:
        arms.append(("ar", decode_tokens))
    if not arms:
        raise ValueError("at least one measured arm must be enabled")
    return tuple(arms)


def guard(text: str) -> dict[str, Any]:
    text = str(text)
    head = text[:30]
    unique_fraction = 0.0 if not head else len(set(head)) / len(head)
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[index : index + 3]) for index in range(max(0, len(words) - 2))]
    repeats = max(([grams.count(gram) for gram in set(grams)]), default=0)
    return {
        "nonempty": bool(text.strip()),
        "unique_char30_fraction": round(unique_fraction, 6),
        "max_word_trigram_repeats": repeats,
        "passed": bool(text.strip()) and unique_fraction >= 0.5 and repeats <= 3,
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def server_command(args: argparse.Namespace, port: int) -> list[str]:
    if args.base_url:
        return ["<external>", str(args.base_url)]
    if args.server is None:
        raise ValueError("--server is required unless --base-url is given")
    command = [
        str(Path(args.server).resolve()),
        "-m",
        str(Path(args.model).resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(args.context),
        "-b",
        str(args.batch),
        "-ub",
        str(args.ubatch),
        "-ngl",
        "999",
        "-np",
        str(_parallel_capacity(tuple(args.widths), int(args.parallel))),
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
    ]
    if args.spec_draft_n_max:
        command += [
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            str(args.spec_draft_n_max),
            "--spec-draft-p-min",
            "0.0",
        ]
    return command


def start_server(command: list[str], log_path: Path, port: int) -> subprocess.Popen:
    log = log_path.open("wb")
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), start_new_session=True
    )
    deadline = time.time() + float(args_health_timeout())
    try:
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exit {process.returncode}: "
                    + log_path.read_text(errors="replace")[-12000:]
                )
            try:
                get(f"http://127.0.0.1:{port}/health")
                return process
            except (OSError, URLError):
                time.sleep(1.0)
    finally:
        log.close()
    raise TimeoutError(f"server health timeout after {args_health_timeout()}s")


def args_health_timeout() -> int:
    return int(os.environ.get("HIPENGINE_LLAMACPP_HEALTH_TIMEOUT", "600"))


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def completion_request(
    flavor: str,
    *,
    model: str,
    prompt: str,
    n_predict: int,
    sampling: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Build one request for the named wire flavor.

    ``llamacpp`` is llama.cpp's native ``/completion`` with ``n_predict``;
    ``openai`` is ``/v1/completions`` with ``max_tokens``, which is what
    hipEngine's own server speaks. Both flavors keep the same sampling,
    denominator, and guard semantics so a wave is comparable across engines.
    """

    if flavor == "openai":
        return (
            "/v1/completions",
            {"model": model, "prompt": prompt, "max_tokens": int(n_predict), **sampling},
        )
    return "/completion", dict(sampling, prompt=prompt, n_predict=int(n_predict))


def completion_response_fields(flavor: str, payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or {}
    if flavor == "openai":
        choices = payload.get("choices") or [{}]
        text = str(choices[0].get("text", ""))
        return {
            "tokens_predicted": int(usage.get("completion_tokens", 0)),
            "tokens_evaluated": int(usage.get("prompt_tokens", 0)),
            "content_sha256": sha256_text(text),
            "guard": guard(text),
            "server_predicted_tok_s": None,
            "server_prompt_tok_s": None,
            "stop_type": choices[0].get("finish_reason"),
        }
    timings = payload.get("timings") or {}
    return {
        "tokens_predicted": int(payload.get("tokens_predicted", 0)),
        "tokens_evaluated": int(payload.get("tokens_evaluated", 0)),
        "content_sha256": sha256_text(str(payload.get("content", ""))),
        "guard": guard(str(payload.get("content", ""))),
        "server_predicted_tok_s": timings.get("predicted_per_second"),
        "server_prompt_tok_s": timings.get("prompt_per_second"),
        "stop_type": payload.get("stop_type"),
    }


def run_wave(
    base_url: str,
    sampling: dict[str, Any],
    prompt: str,
    *,
    width: int,
    n_predict: int,
    flavor: str = "llamacpp",
    model: str = "",
) -> list[dict[str, Any]]:
    barrier = threading.Barrier(width)
    epoch = [0.0]

    def lane(_: int) -> dict[str, Any]:
        barrier.wait(60)
        if epoch[0] == 0.0:
            epoch[0] = time.perf_counter()
        while epoch[0] == 0.0:
            pass
        route, body = completion_request(
            flavor, model=model, prompt=prompt, n_predict=n_predict, sampling=sampling
        )
        started = time.perf_counter()
        payload = post(f"{base_url}{route}", body)
        ended = time.perf_counter()
        row = {"started": started, "ended": ended, "wall_seconds": ended - started}
        row.update(completion_response_fields(flavor, payload))
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
        rows = list(pool.map(lane, range(width)))
    wall = max(row["ended"] for row in rows) - min(row["started"] for row in rows)
    for row in rows:
        row["cell_wall_seconds"] = wall
    return rows


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(Path(args.prompts).resolve())
    widths = tuple(sorted({int(width) for width in args.widths}))
    base_url = (
        args.base_url.rstrip("/") + "/v1" if args.base_url else f"http://127.0.0.1:{int(args.port)}/v1"
    )
    sampling = {"temperature": 0.0, "top_p": 1.0}
    if args.flavor == "llamacpp":
        sampling.update({"top_k": 1, "seed": int(args.seed), "cache_prompt": False, "stream": False})
    arms = _workload_arms(
        int(args.prefill_tokens),
        int(args.decode_tokens),
        int(args.spec_draft_n_max),
    )
    process: subprocess.Popen | None = None
    log_path = Path(args.output).with_suffix(".server.log")
    started = time.perf_counter()
    cells: list[dict[str, Any]] = []
    try:
        if args.base_url:
            external = args.base_url.rstrip("/")
            deadline = time.time() + float(args_health_timeout())
            while time.time() < deadline:
                try:
                    get(f"{external}/health")
                    break
                except (OSError, URLError, RuntimeError):
                    time.sleep(1.0)
            else:
                raise TimeoutError(f"external server health timeout: {external}/health")
        else:
            process = start_server(server_command(args, int(args.port)), log_path, int(args.port))
        warm_route, warm_body = completion_request(
            args.flavor,
            model=args.served_model_name,
            prompt=prompts[0]["rendered"],
            n_predict=2,
            sampling=sampling,
        )
        post(f"{base_url.removesuffix('/v1')}{warm_route}", warm_body)
        for arm, tokens in arms:
            for width in widths:
                for prompt in prompts:
                    rows = run_wave(
                        base_url.removesuffix("/v1"),
                        sampling,
                        prompt["rendered"],
                        width=width,
                        n_predict=tokens,
                        flavor=args.flavor,
                        model=args.served_model_name,
                    )
                    numerator = sum(
                        row["tokens_evaluated"] if arm == "prefill" else row["tokens_predicted"]
                        for row in rows
                    )
                    wall = rows[0]["cell_wall_seconds"]
                    hashes = {row["content_sha256"] for row in rows}
                    cells.append(
                        {
                            "arm": arm,
                            "width": width,
                            "prompt": prompt["id"],
                            "category": prompt["category"],
                            "n_predict": tokens,
                            "wall_seconds": wall,
                            "numerator_tokens": numerator,
                            "tok_s": numerator / wall if wall else None,
                            "lane_exact": len(hashes) == 1,
                            "guards_passed": all(row["guard"]["passed"] for row in rows),
                            "rows": rows,
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "label": args.label,
                                "arm": arm,
                                "width": width,
                                "prompt": prompt["id"],
                                "tok_s": round(numerator / wall, 3) if wall else None,
                                "exact": len(hashes) == 1,
                            }
                        ),
                        flush=True,
                    )
    finally:
        stop_server(process)
    return finalize(args, prompts, widths, cells, started, log_path)


def summarize_arm(cells: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for width in sorted({int(cell["width"]) for cell in cells if cell["arm"] == arm}):
        rows = [cell for cell in cells if cell["arm"] == arm and int(cell["width"]) == width]
        wall = sum(float(cell["wall_seconds"]) for cell in rows)
        numerator = sum(int(cell["numerator_tokens"]) for cell in rows)
        out[str(width)] = {
            "prompts": len(rows),
            "numerator_tokens": numerator,
            "wall_seconds": wall,
            "complete_wall_tok_s": numerator / wall if wall else None,
            "all_lane_exact": all(bool(cell["lane_exact"]) for cell in rows),
            "all_guards_passed": all(bool(cell["guards_passed"]) for cell in rows),
        }
    return out


def finalize(
    args: argparse.Namespace,
    prompts: list[dict[str, str]],
    widths: tuple[int, ...],
    cells: list[dict[str, Any]],
    started: float,
    log_path: Path,
) -> dict[str, Any]:
    model = Path(args.model).resolve()
    payload = {
        "schema": SCHEMA,
        "kind": "llamacpp_c1c8_engine_matrix",
        "generated": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if cells else "failed",
        "label": args.label,
        "host": platform.node(),
        "source_commit": args.source_commit,
        "server": str(Path(args.server).resolve()) if args.server else None,
        "base_url": args.base_url,
        "server_version": args.server_version,
        "model": str(model),
        "model_sha256": sha256_file(model),
        "model_size_bytes": model.stat().st_size,
        "command": server_command(args, int(args.port)),
        "protocol": {
            "prompts": str(Path(args.prompts).resolve()),
            "prompt_ids": [prompt["id"] for prompt in prompts],
            "suite_sha256": sha256_file(Path(args.prompts).resolve()),
            "widths": list(widths),
            "parallel_capacity": int(args.parallel),
            "sampling": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": int(args.seed)},
            "cache_prompt": False,
            "prefill_tokens": int(args.prefill_tokens),
            "decode_tokens": int(args.decode_tokens),
            "spec_draft_n_max": int(args.spec_draft_n_max),
            "timing_boundary": TIMING_BOUNDARY,
            "guards": {"unique_char30_fraction_min": 0.5, "word_trigram_repeats_max": 3},
        },
        "summary": {arm: summarize_arm(cells, arm) for arm in sorted({str(c["arm"]) for c in cells})},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "server_log": str(log_path),
        "cells": cells,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=None)
    parser.add_argument("--server-version", default=None)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--widths", type=_widths, default=tuple(range(1, 9)))
    parser.add_argument("--decode-tokens", type=int, default=24)
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=1,
        help="one-token arm that isolates admission + prefill + first token; 0 skips it",
    )
    parser.add_argument("--spec-draft-n-max", type=int, default=0)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "drive an already-running OpenAI-compatible server (for example a "
            "hipEngine http server) instead of launching --server; the wave "
            "protocol is unchanged so transports stay comparable"
        ),
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--served-model-name", default="qwen38-q4km")
    parser.add_argument(
        "--flavor",
        choices=("llamacpp", "openai"),
        default="llamacpp",
        help="wire protocol of the target server; openai targets hipEngine",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_matrix(args)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
