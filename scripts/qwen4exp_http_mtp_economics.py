#!/usr/bin/env python3
"""Measure served MTP economics against a true AR arm at a fixed request width.

One resident server, one prompt set, two arms: requests with speculative MTP
left off, then the same requests with it requested. Each wave issues `--width`
requests concurrently and the wall clock covers first send to last completion,
so the aggregate rate reflects what the serving path actually delivers.

The MTP arm is only valid if the server reports that MTP executed: a request
that silently fell back to autoregressive decode is counted as a failed arm,
not as an MTP measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl"


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exact_completion_count(body: dict[str, Any]) -> int:
    """Accept only the response-owned generated IDs, never retokenized text."""
    accounting = (body.get("hipengine") or {}).get("token_accounting") or {}
    rows = accounting.get("choice_generated_token_ids")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise ValueError("response lacks exact single-choice token accounting")
    if any(type(token) is not int or token < 0 for token in rows[0]):
        raise ValueError("generated token IDs must be non-negative integers")
    count = len(rows[0])
    total = accounting.get("total_generated_tokens")
    if (type(total) is not int or total != count
            or accounting.get("choice_generated_tokens") != [count]
            or (body.get("usage") or {}).get("completion_tokens") != count):
        raise ValueError("generated IDs, counts and usage disagree")
    return count


def _one(client: Any, url: str, model: str, row: dict[str, Any], mtp: bool, max_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": row["messages"],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "speculative_mtp": mtp,
        "include_hipengine": True,
    }
    started = time.perf_counter()
    response = client.post(f"{url}/v1/chat/completions", json=payload)
    wall = time.perf_counter() - started
    body = response.json()
    if response.status_code != 200:
        return {"id": row["id"], "status": response.status_code, "wall_s": wall, "error": json.dumps(body)[:400]}
    choice = (body.get("choices") or [{}])[0]
    usage = body.get("usage") or {}
    spec = (body.get("hipengine") or {}).get("speculative_mtp") or {}
    return {
        "id": row["id"],
        "category": row["category"],
        "status": 200,
        "wall_s": wall,
        "completion_tokens": exact_completion_count(body),
        "token_accounting": (body.get("hipengine") or {}).get("token_accounting"),
        "prompt_token_accounting": (body.get("hipengine") or {}).get("prompt_token_accounting"),
        "generation_shape": (body.get("hipengine") or {}).get("generation_shape"),
        "message": choice.get("message"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "speculative_requested": mtp,
        "speculative_used": spec.get("used"),
        "speculative_reason": spec.get("decision_reason"),
        "accepted_per_cycle": spec.get("accepted_per_cycle"),
        "acceptance": spec.get("acceptance_rate"),
        "route": ((body.get("hipengine") or {}).get("generation_shape") or {}).get("route"),
        "finish_reason": choice.get("finish_reason"),
    }


def route_matches_request(case: dict[str, Any]) -> bool:
    """Require a per-request witness for both true AR and actual MTP."""
    requested = case.get("speculative_requested")
    return (case.get("status") == 200 and isinstance(requested, bool)
            and case.get("speculative_used") is requested)


def _wave(
    rows: list[dict[str, Any]],
    url: str,
    model: str,
    mtp: bool | Sequence[bool],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    import httpx

    results: list[dict[str, Any]] = [{} for _ in rows]
    flags = list(mtp) if isinstance(mtp, (list, tuple)) else [bool(mtp)] * len(rows)
    if len(flags) != len(rows):
        raise ValueError("one MTP flag per request is required")

    def worker(index: int) -> None:
        client = httpx.Client(timeout=timeout)
        try:
            results[index] = _one(client, url, model, rows[index], flags[index], max_tokens)
        except Exception as exc:
            results[index] = {"id": rows[index]["id"], "status": 0, "error": str(exc)}
        finally:
            client.close()

    started = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(index,)) for index in range(len(rows))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    tokens = sum(int(item.get("completion_tokens") or 0) for item in results)
    return {
        "wall_s": round(wall, 3),
        "requests": len(rows),
        "completion_tokens": tokens,
        "aggregate_tok_s": round(tokens / wall, 3) if wall > 0 else 0.0,
        "per_request_tok_s": round(
            sum(
                (int(item.get("completion_tokens") or 0) / item["wall_s"])
                for item in results
                if item.get("wall_s")
            )
            / max(1, len(results)),
            3,
        ),
        "mean_wall_s": round(
            sum(item.get("wall_s") or 0.0 for item in results) / max(1, len(results)), 3
        ),
        "cases": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18400")
    parser.add_argument("--model", default="qwen38fn-capon")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument(
        "--arms",
        default="ar,mtp",
        help="Comma-separated arms to measure: ar, mtp, mixed (one MTP request in the wave).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = _load(args.prompts)
    if not rows or args.width <= 0 or args.max_tokens <= 0:
        parser.error("prompts must be nonempty and width/max-tokens must be positive")
    selected = [item.strip() for item in args.arms.split(",") if item.strip()]
    if not selected or len(set(selected)) != len(selected) or set(selected) - {"ar", "mtp", "mixed"}:
        parser.error("arms must be unique names from ar,mtp,mixed")
    if args.warmup:
        _wave(rows[:1], args.url, args.model, False, 8, args.timeout)
        _wave(rows[:1], args.url, args.model, True, 8, args.timeout)

    waves = [rows[start : start + args.width] for start in range(0, len(rows), args.width)]
    arms: dict[str, list[dict[str, Any]]] = {label: [] for label in selected}
    for label in selected:
        for wave in waves:
            if label == "ar":
                flags: bool | list[bool] = False
            elif label == "mtp":
                flags = True
            elif label == "mixed":
                # The first request of the wave asks for MTP, the rest stay AR.
                flags = [True] + [False] * (len(wave) - 1)
            else:
                parser.error(f"unknown arm: {label}")
            arms[label].append(
                _wave(wave, args.url, args.model, flags, args.max_tokens, args.timeout)
            )

    def totals(label: str) -> dict[str, Any]:
        wall = sum(item["wall_s"] for item in arms[label])
        tokens = sum(item["completion_tokens"] for item in arms[label])
        cases = [case for item in arms[label] for case in item["cases"]]
        return {
            "waves": len(arms[label]),
            "wall_s": round(wall, 3),
            "completion_tokens": tokens,
            "aggregate_tok_s": round(tokens / wall, 3) if wall else 0.0,
            "requests": len(cases),
            "errors": sum(1 for case in cases if case.get("status") != 200),
            "speculative_used": sum(1 for case in cases if case.get("speculative_used")),
            "mean_acceptance": round(
                sum(float(case.get("acceptance") or 0.0) for case in cases)
                / max(1, sum(1 for case in cases if case.get("acceptance") is not None)),
                4,
            ),
        }

    measured = {label: totals(label) for label in selected}
    valid_routes = all(
        route_matches_request(case)
        for label in selected for wave in arms[label] for case in wave["cases"]
    )
    artifact = {
        "schema": "hipengine.qwen4exp.http_mtp_economics.v2",
        "performance_claim": False,
        "status": "measured_diagnostic" if valid_routes else "invalid",
        "checks": {"every_request_used_requested_route": valid_routes},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": args.url,
        "model": args.model,
        "prompts": str(args.prompts),
        "width": args.width,
        "max_tokens": args.max_tokens,
        "arms": measured,
        "ratios": {
            label: round(
                measured[label]["aggregate_tok_s"] / measured["ar"]["aggregate_tok_s"], 4
            )
            for label in selected
            if valid_routes and label != "ar" and measured.get("ar", {}).get("aggregate_tok_s")
        },
        "waves": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "width": args.width,
                "tok_s": {label: measured[label]["aggregate_tok_s"] for label in selected},
                "ratios_vs_ar": artifact["ratios"],
                "mtp_used": {
                    label: f"{measured[label]['speculative_used']}/{measured[label]['requests']}"
                    for label in selected
                },
                "acceptance": {
                    label: measured[label]["mean_acceptance"] for label in selected
                },
            }
        )
    )
    # Every request must execute its named route, including the AR peers in a
    # mixed wave. One successful MTP request cannot mask another's fallback.
    return 0 if valid_routes else 1


if __name__ == "__main__":
    raise SystemExit(main())
