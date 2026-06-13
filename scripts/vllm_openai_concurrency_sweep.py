#!/usr/bin/env python3
"""OpenAI-compatible vLLM concurrency sweep using exact prompt token IDs."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_FIXTURE = Path("/tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json")


class SweepError(RuntimeError):
    pass


def parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SweepError(f"HTTP {exc.code}: {body}") from exc
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise SweepError(f"expected object response, got {type(parsed).__name__}")
    return parsed


def get_json(url: str, timeout: float = 10.0) -> dict[str, Any] | None:
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            parsed = json.loads(resp.read())
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def get_metrics(url: str, timeout: float = 10.0) -> dict[str, float]:
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return {}
    values: dict[str, float] = {}
    wanted = (
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:e2e_request_latency_seconds_count",
        "vllm:inter_token_latency_seconds_sum",
        "vllm:inter_token_latency_seconds_count",
    )
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in wanted:
            continue
        try:
            values[name] = values.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
    return values


def metric_delta(after: dict[str, float], before: dict[str, float], key: str) -> float | None:
    if key not in after or key not in before:
        return None
    return after[key] - before[key]


def load_rows(path: Path, prompt_length: int | None, count: int | None) -> list[list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = data.get("prompt_ids")
    if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
        raise SweepError(f"{path} missing flat integer prompt_ids")
    plen = int(prompt_length or data.get("prompt_length") or 0)
    if plen <= 0:
        raise SweepError("prompt_length must be positive")
    n = int(count or data.get("prompt_count") or (len(ids) // plen))
    rows = [ids[i * plen : (i + 1) * plen] for i in range(n)]
    rows = [row for row in rows if len(row) == plen]
    if not rows:
        raise SweepError("no complete prompt rows in fixture")
    return rows


def choose_rows(rows: list[list[int]], c: int) -> list[list[int]]:
    return [rows[i % len(rows)] for i in range(c)]


def request_one(
    *,
    url: str,
    model: str,
    prompt_ids: list[int],
    decode_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    ignore_eos: bool,
    timeout: float,
    barrier: threading.Barrier,
    request_index: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": [int(x) for x in prompt_ids],
        "max_tokens": int(decode_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": int(seed + request_index),
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    barrier.wait()
    start = time.perf_counter()
    resp = post_json(url, payload, timeout)
    wall_s = time.perf_counter() - start
    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    choices = resp.get("choices") if isinstance(resp.get("choices"), list) else []
    return {
        "request_index": request_index,
        "wall_s": wall_s,
        "prompt_tokens": int(usage.get("prompt_tokens") or len(prompt_ids)),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "finish_reason": choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None,
    }


def run_rep(args: argparse.Namespace, rows: list[list[int]], c: int, rep: int) -> dict[str, Any]:
    selected = choose_rows(rows, c)
    url = args.url.rstrip("/") + "/v1/completions"
    metrics_url = args.url.rstrip("/") + "/metrics"
    before = get_metrics(metrics_url)
    barrier = threading.Barrier(c + 1)
    futures = []
    batch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as ex:
        for i, row in enumerate(selected):
            futures.append(
                ex.submit(
                    request_one,
                    url=url,
                    model=args.model,
                    prompt_ids=row,
                    decode_tokens=args.decode_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed + rep * 1000,
                    ignore_eos=args.ignore_eos,
                    timeout=args.request_timeout,
                    barrier=barrier,
                    request_index=i,
                )
            )
        barrier.wait()
        launched = time.perf_counter()
        results = [future.result() for future in as_completed(futures)]
    batch_wall_s = time.perf_counter() - launched
    submit_wall_s = time.perf_counter() - batch_start
    after = get_metrics(metrics_url)

    total_completion = sum(r["completion_tokens"] for r in results)
    max_request_wall = max((r["wall_s"] for r in results), default=batch_wall_s)
    e2e_aggregate_tok_s = total_completion / batch_wall_s if batch_wall_s > 0 else None
    max_wall_aggregate_tok_s = total_completion / max_request_wall if max_request_wall > 0 else None
    gen_metric = metric_delta(after, before, "vllm:generation_tokens_total")
    prompt_metric = metric_delta(after, before, "vllm:prompt_tokens_total")
    ttft_sum = metric_delta(after, before, "vllm:time_to_first_token_seconds_sum")
    ttft_count = metric_delta(after, before, "vllm:time_to_first_token_seconds_count")
    e2e_sum = metric_delta(after, before, "vllm:e2e_request_latency_seconds_sum")
    e2e_count = metric_delta(after, before, "vllm:e2e_request_latency_seconds_count")
    ttft_mean = (ttft_sum / ttft_count) if ttft_sum is not None and ttft_count else None
    e2e_mean = (e2e_sum / e2e_count) if e2e_sum is not None and e2e_count else None
    post_ttft_tok_s = None
    if ttft_mean is not None and e2e_mean is not None and e2e_mean > ttft_mean:
        post_ttft_tok_s = total_completion / (e2e_mean - ttft_mean)

    return {
        "concurrency": c,
        "rep": rep,
        "batch_wall_s": batch_wall_s,
        "submit_wall_s": submit_wall_s,
        "max_request_wall_s": max_request_wall,
        "total_completion_tokens": total_completion,
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "e2e_tok_s_aggregate": e2e_aggregate_tok_s,
        "max_wall_tok_s_aggregate": max_wall_aggregate_tok_s,
        "post_ttft_tok_s_aggregate": post_ttft_tok_s,
        "per_request_tok_s": (max_wall_aggregate_tok_s / c) if max_wall_aggregate_tok_s is not None else None,
        "metrics_delta": {
            "prompt_tokens_total": prompt_metric,
            "generation_tokens_total": gen_metric,
            "ttft_sum_s": ttft_sum,
            "ttft_count": ttft_count,
            "ttft_mean_s": ttft_mean,
            "e2e_latency_sum_s": e2e_sum,
            "e2e_latency_count": e2e_count,
            "e2e_latency_mean_s": e2e_mean,
        },
        "requests": sorted(results, key=lambda r: r["request_index"]),
    }


def median(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    return statistics.median(filtered) if filtered else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8008")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int)
    parser.add_argument("--prompt-count", type=int)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--concurrencies", type=parse_csv_ints, default=[1, 2, 4, 8])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.fixture, args.prompt_length, args.prompt_count)
    model_info = get_json(args.url.rstrip("/") + "/v1/models")

    if args.warmup_decode_tokens > 0:
        warm_args = argparse.Namespace(**vars(args))
        warm_args.decode_tokens = args.warmup_decode_tokens
        print("warmup c=1", flush=True)
        run_rep(warm_args, rows, 1, -1)

    reps_by_c: dict[str, list[dict[str, Any]]] = {}
    summary_rows: dict[str, Any] = {}
    for c in args.concurrencies:
        reps: list[dict[str, Any]] = []
        for rep in range(args.reps):
            started = time.perf_counter()
            rec = run_rep(args, rows, c, rep)
            reps.append(rec)
            print(
                f"c{c} rep{rep}: agg={rec['max_wall_tok_s_aggregate']:.2f} "
                f"per={rec['per_request_tok_s']:.2f} wall={rec['max_request_wall_s']:.2f}s "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
            time.sleep(0.5)
        reps_by_c[str(c)] = reps
        agg_runs = [r.get("max_wall_tok_s_aggregate") for r in reps]
        e2e_runs = [r.get("e2e_tok_s_aggregate") for r in reps]
        ttft_runs = [((r.get("metrics_delta") or {}).get("ttft_mean_s")) for r in reps]
        e2e_lat_runs = [((r.get("metrics_delta") or {}).get("e2e_latency_mean_s")) for r in reps]
        post_ttft_runs = [r.get("post_ttft_tok_s_aggregate") for r in reps]
        agg_median = median(agg_runs)
        summary_rows[str(c)] = {
            "concurrency": c,
            "decode_tok_s_aggregate_median": agg_median,
            "decode_tok_s_per_request_median": (agg_median / c) if agg_median is not None else None,
            "decode_tok_s_aggregate_runs": agg_runs,
            "e2e_tok_s_aggregate_median": median(e2e_runs),
            "e2e_tok_s_aggregate_runs": e2e_runs,
            "post_ttft_tok_s_aggregate_median": median(post_ttft_runs),
            "post_ttft_tok_s_aggregate_runs": post_ttft_runs,
            "ttft_mean_s_median": median(ttft_runs),
            "e2e_latency_mean_s_median": median(e2e_lat_runs),
            "rep_artifacts": reps,
            "path": "vLLM OpenAI /v1/completions with prompt token IDs",
        }

    out = {
        "kind": "vllm_openai_concurrency_decode_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "url": args.url,
        "model": args.model,
        "server_model_info": model_info,
        "fixture": str(args.fixture),
        "shape": {
            "prompt_length": args.prompt_length or len(rows[0]),
            "decode_tokens": args.decode_tokens,
            "concurrencies": args.concurrencies,
            "reps": args.reps,
        },
        "methodology": {
            "client": "concurrent POST /v1/completions with exact prompt token-id rows",
            "primary_metric": "sum(completion_tokens) / max(request wall time) across concurrent responses",
            "note": "OpenAI responses do not expose pure decode timings; primary metric includes prompt prefill and HTTP wall time.",
        },
        "summary_rows": summary_rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
