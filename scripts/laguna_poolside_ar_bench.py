#!/usr/bin/env python3
"""Run a matched raw-token Poolside llama.cpp baseline for Laguna target AR."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any
from urllib import error, request

from hipengine.loading.gguf import GGUFReader
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_target_ar_bench import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    EXPECTED_CATEGORIES,
    EXPECTED_PROMPT_COUNT,
    RETAINED_HORIZONS,
    _load_prompts,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = Path(
    "/home/lhl/models/hipengine_sources/poolside-llama.cpp-laguna/"
    "build-hip-gfx1151/bin/llama-server"
)
DEFAULT_HIPENGINE_ARTIFACT = Path(
    "/tmp/laguna-target-ar-category.json"
)
DEFAULT_SERVER_LOG = Path("/tmp/laguna-poolside-target-ar-server.log")
DEFAULT_GTT_PATH = Path("/sys/class/drm/card1/device/mem_info_gtt_used")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--hipengine-artifact",
        type=Path,
        default=DEFAULT_HIPENGINE_ARTIFACT,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument(
        "--output-horizons",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=RETAINED_HORIZONS,
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--server-log", type=Path, default=DEFAULT_SERVER_LOG)
    parser.add_argument("--gtt-path", type=Path, default=DEFAULT_GTT_PATH)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    call = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(call, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _wait_for_server(
    *,
    host: str,
    port: int,
    timeout: float,
    process: subprocess.Popen,
) -> float:
    started = time.perf_counter()
    deadline = started + float(timeout)
    url = f"http://{host}:{port}/health"
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Poolside server exited during startup with {process.returncode}")
        try:
            with request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return time.perf_counter() - started
        except (error.URLError, TimeoutError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"Poolside server did not become ready within {timeout:.1f}s")


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def _completion_payload(token_ids: tuple[int, ...], horizon: int) -> dict[str, Any]:
    return {
        "prompt": [int(token) for token in token_ids],
        "n_predict": int(horizon),
        "temperature": 0.0,
        "top_k": 0,
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


def _matching_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for lhs, rhs in zip(left, right):
        if int(lhs) != int(rhs):
            break
        count += 1
    return count


def _response_row(
    *,
    prompt: dict[str, Any],
    horizon: int,
    repetition: int,
    response: dict[str, Any],
    wall_seconds: float,
    hipengine_ids: list[int],
) -> dict[str, Any]:
    timings = response.get("timings") or {}
    tokens = [int(token) for token in (response.get("tokens") or ())]
    predicted_n = int(timings.get("predicted_n", response.get("tokens_predicted", 0)) or 0)
    prompt_n = int(timings.get("prompt_n", 0) or 0)
    prompt_seconds = float(timings.get("prompt_ms", 0.0) or 0.0) / 1000.0
    predicted_seconds = float(timings.get("predicted_ms", 0.0) or 0.0) / 1000.0
    valid_count = bool(len(tokens) == int(horizon) == predicted_n)
    valid_prompt_count = bool(prompt_n == int(prompt["prompt_tokens"]))
    return {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt_tokens": prompt["prompt_tokens"],
        "prompt_token_ids_sha256": prompt["token_ids_sha256"],
        "horizon": int(horizon),
        "repetition": int(repetition),
        "generated_token_ids": tokens,
        "generated_ids_sha256": _sha256(
            json.dumps(tokens, separators=(",", ":")).encode("utf-8")
        ),
        "valid_token_count": valid_count,
        "valid_prompt_count": valid_prompt_count,
        "prompt_n": prompt_n,
        "prompt_seconds": prompt_seconds,
        "prompt_tok_s": prompt_n / prompt_seconds if prompt_seconds > 0 else 0.0,
        "predicted_n": predicted_n,
        "predicted_seconds": predicted_seconds,
        "predicted_tok_s": (
            predicted_n / predicted_seconds if predicted_seconds > 0 else 0.0
        ),
        "wall_seconds": float(wall_seconds),
        "wall_output_tok_s": predicted_n / wall_seconds if wall_seconds > 0 else 0.0,
        "matches_hipengine": tokens == hipengine_ids,
        "matching_hipengine_prefix_tokens": _matching_prefix(tokens, hipengine_ids),
        "stop_type": response.get("stop_type"),
        "timings": timings,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Poolside aggregate requires at least one row")
    prompt_tokens = sum(int(row["prompt_n"]) for row in rows)
    prompt_seconds = sum(float(row["prompt_seconds"]) for row in rows)
    predicted_tokens = sum(int(row["predicted_n"]) for row in rows)
    predicted_seconds = sum(float(row["predicted_seconds"]) for row in rows)
    wall_seconds = sum(float(row["wall_seconds"]) for row in rows)
    return {
        "runs": len(rows),
        "prompt_tokens": prompt_tokens,
        "prompt_seconds": prompt_seconds,
        "prompt_tok_s": prompt_tokens / prompt_seconds,
        "predicted_tokens": predicted_tokens,
        "predicted_seconds": predicted_seconds,
        "predicted_tok_s": predicted_tokens / predicted_seconds,
        "wall_seconds": wall_seconds,
        "wall_output_tok_s": predicted_tokens / wall_seconds,
        "wall_median_seconds": statistics.median(float(row["wall_seconds"]) for row in rows),
        "valid_token_counts": all(bool(row["valid_token_count"]) for row in rows),
        "valid_prompt_counts": all(bool(row["valid_prompt_count"]) for row in rows),
        "hipengine_exact_runs": sum(bool(row["matches_hipengine"]) for row in rows),
    }


def _rollups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_horizon = {
        str(horizon): _aggregate([row for row in rows if row["horizon"] == horizon])
        for horizon in sorted({int(row["horizon"]) for row in rows})
    }
    by_category: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        by_category[category] = {
            str(horizon): _aggregate(
                [row for row in selected if row["horizon"] == horizon]
            )
            for horizon in sorted({int(row["horizon"]) for row in selected})
        }
    return {"horizons": by_horizon, "categories": by_category}


def _read_optional_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _process_rss_bytes(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _git_state(path: Path) -> dict[str, Any]:
    root = path.resolve()
    revision = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=root, text=True
    ).strip()
    tracked = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=root,
        text=True,
    ).strip()
    return {
        "path": str(root),
        "revision": revision,
        "tracked_clean": not bool(tracked),
        "tracked_status": tracked.splitlines(),
    }


def _hipengine_oracle(path: Path, horizons: tuple[int, ...]) -> dict[tuple[str, int], list[int]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not artifact.get("pass"):
        raise ValueError("hipEngine Laguna target artifact did not pass")
    oracle: dict[tuple[str, int], list[int]] = {}
    for row in artifact["prompt_runs"]:
        if row["mode"] != "bulk" or int(row["repetition"]) != 0:
            continue
        for horizon in horizons:
            oracle[(str(row["prompt_id"]), int(horizon))] = [
                int(token)
                for token in row["checkpoints"][str(horizon)]["generated_token_ids"]
            ]
    expected = EXPECTED_PROMPT_COUNT * len(horizons)
    if len(oracle) != expected:
        raise ValueError(f"hipEngine oracle has {len(oracle)} prompt/horizon rows, expected {expected}")
    return oracle


def run(args: argparse.Namespace) -> dict[str, Any]:
    horizons = tuple(sorted(set(int(value) for value in args.output_horizons)))
    if horizons != RETAINED_HORIZONS:
        raise ValueError(f"Poolside retained horizons must be {RETAINED_HORIZONS}")
    if args.repetitions < 2:
        raise ValueError("Poolside retained benchmark requires at least two repetitions")
    if not args.server_bin.is_file() or not os.access(args.server_bin, os.X_OK):
        raise FileNotFoundError(f"Poolside llama-server is not executable: {args.server_bin}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")

    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    if len(prompts) != EXPECTED_PROMPT_COUNT or {
        prompt["category"] for prompt in prompts
    } != EXPECTED_CATEGORIES:
        raise ValueError("Poolside benchmark requires the canonical Laguna prompt suite")
    oracle = _hipengine_oracle(args.hipengine_artifact, horizons)

    server_root = args.server_bin.resolve().parents[2]
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
        "-ngl",
        "999",
        "-fa",
        "off",
        "--parallel",
        "1",
        "--no-warmup",
        "--no-repack",
        "--no-mmap",
        "--metrics",
    ]
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("GPU_MAX_HW_QUEUES", "1")
    rows: list[dict[str, Any]] = []
    gtt_samples: list[int] = []
    rss_samples: list[int] = []
    with args.server_log.open("wb") as log:
        process = subprocess.Popen(
            server_command,
            cwd=server_root,
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
        _post_json(
            f"http://{args.host}:{args.port}/completion",
            _completion_payload(prompts[0]["token_ids"], max(horizons)),
            args.request_timeout,
        )
        for repetition in range(args.repetitions):
            for prompt_index, prompt in enumerate(prompts):
                order = horizons if (repetition + prompt_index) % 2 == 0 else horizons[::-1]
                for horizon in order:
                    payload = _completion_payload(prompt["token_ids"], horizon)
                    started = time.perf_counter()
                    response = _post_json(
                        f"http://{args.host}:{args.port}/completion",
                        payload,
                        args.request_timeout,
                    )
                    wall_seconds = time.perf_counter() - started
                    row = _response_row(
                        prompt=prompt,
                        horizon=horizon,
                        repetition=repetition,
                        response=response,
                        wall_seconds=wall_seconds,
                        hipengine_ids=oracle[(prompt["id"], horizon)],
                    )
                    rows.append(row)
                    gtt = _read_optional_int(args.gtt_path)
                    rss = _process_rss_bytes(process.pid)
                    if gtt is not None:
                        gtt_samples.append(gtt)
                    if rss is not None:
                        rss_samples.append(rss)
                    print(
                        f"rep={repetition} prompt={prompt['id']} h={horizon} "
                        f"prefill={row['prompt_tok_s']:.2f} tok/s "
                        f"predicted={row['predicted_tok_s']:.2f} tok/s "
                        f"match={row['matches_hipengine']}",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        _terminate(process)

    rollups = _rollups(rows)
    passed = bool(
        all(row["valid_token_count"] and row["valid_prompt_count"] for row in rows)
    )
    server_binary = args.server_bin.read_bytes()
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "poolside_llamacpp_laguna_target_ar_category_baseline",
        "status": "complete" if passed else "rejected",
        "pass": passed,
        "performance_claim": False,
        "comparison_eligibility": {
            "prompt_ids_and_output_horizons_matched": True,
            "generated_ids_match_is_reported_not_required": True,
            "cross_engine_speed_ratio_eligible": False,
            "reason": (
                "llama.cpp predicted_ms owns all generated tokens while hipEngine decode "
                "owns horizon-1 post-TTFT forward calls; HTTP wall also differs from the "
                "resident in-process hipEngine timing boundary"
            ),
        },
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": "Q4_K_M mixed GGUF v3",
        },
        "prompt_suite": {
            "path": str(args.prompts.resolve()),
            "sha256": _sha256(args.prompts.read_bytes()),
            "count": len(prompts),
            "categories": sorted(EXPECTED_CATEGORIES),
            "prompt_tokens_min": min(prompt["prompt_tokens"] for prompt in prompts),
            "prompt_tokens_max": max(prompt["prompt_tokens"] for prompt in prompts),
        },
        "protocol": {
            "context_length": args.context_length,
            "horizons": list(horizons),
            "repetitions": args.repetitions,
            "warmups": 1,
            "sampling": "temperature=0 with neutral penalties and fixed seed 4242",
            "cache_prompt": False,
            "flash_attention": False,
            "mmap": False,
            "repack": False,
            "timing_scope": "llama.cpp native prompt_ms/predicted_ms plus complete HTTP wall",
        },
        "harness_repo": _git_state(ROOT),
        "server": {
            "command": server_command,
            "startup_seconds": startup_seconds,
            "binary": str(args.server_bin.resolve()),
            "binary_sha256": _sha256(server_binary),
            "source": _git_state(server_root),
            "log": str(args.server_log.resolve()),
        },
        "hipengine_oracle_artifact": str(args.hipengine_artifact.resolve()),
        "rows": rows,
        "aggregate": rollups,
        "memory": {
            "gtt_path": str(args.gtt_path),
            "gtt_peak_sampled_bytes": max(gtt_samples) if gtt_samples else None,
            "server_rss_peak_sampled_bytes": max(rss_samples) if rss_samples else None,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
