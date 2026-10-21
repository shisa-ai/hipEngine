#!/usr/bin/env python3
"""Measure served MTP economics against a true AR arm at a fixed request width.

One resident server and prompt set, with explicit AR and MTP request controls.
Report whole-cohort tokens/wall separately from the sum of native request decode
rates. --prepared renders each prompt, prefills all but its final token before
the timed wave, and requires a matching cache hit during measurement. This is
an explicit one-token-suffix protocol, not Gufo's identical-prompt preparation.

The MTP arm is only valid if the server reports that MTP executed: a request
that silently fell back to autoregressive decode is counted as a failed arm,
not as an MTP measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def native_phase_metrics(body: dict[str, Any]) -> dict[str, Any] | None:
    """Keep Gufo's output-count rate and canonical transition-count rate distinct."""
    choices = body.get("choices") or []
    phases = ((choices[0].get("hipengine") or {}).get("diagnostics") or {}).get("request_phases") if choices else None
    if phases is None:
        return None
    count = exact_completion_count(body)
    usage = body["usage"]
    counts = {key: phases.get(key) for key in (
        "prompt_tokens", "cached_prompt_tokens", "new_prompt_tokens", "generated_tokens", "decode_transitions",
    )}
    if (phases.get("schema") != 1 or phases.get("complete") is not True
            or any(type(value) is not int or value < 0 for value in counts.values())
            or counts["generated_tokens"] != count
            or counts["decode_transitions"] != max(0, count - 1)
            or counts["prompt_tokens"] != usage.get("prompt_tokens")
            or counts["cached_prompt_tokens"] != (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            or counts["new_prompt_tokens"] + counts["cached_prompt_tokens"] != counts["prompt_tokens"]):
        raise ValueError("native phase counts disagree with exact response accounting")
    times = {key: phases.get(key) for key in ("prefill_wall_ms", "decode_wall_ms", "request_wall_ms", "cache_restore_ms")}
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in times.values())
            or times["prefill_wall_ms"] <= 0
            or (count > 1 and times["decode_wall_ms"] <= 0)
            or times["prefill_wall_ms"] + times["decode_wall_ms"] > times["request_wall_ms"] + 1e-6):
        raise ValueError("native phase durations are invalid or exceed request wall")
    decode = times["decode_wall_ms"]
    return {
        "counts": counts, "times_ms": times,
        "prefill_new_tok_s": counts["new_prompt_tokens"] * 1000 / times["prefill_wall_ms"],
        "decode_output_tok_s": count * 1000 / decode if decode > 0 else None,
        "decode_transition_tok_s": max(0, count - 1) * 1000 / decode if decode > 0 else None,
    }


def summarize_phases(cases: Sequence[dict[str, Any]]) -> dict[str, float] | None:
    phases = [case.get("phase_metrics") for case in cases]
    if (not phases or any(case.get("status", 200) != 200 for case in cases)
            or any(phase is None or phase.get("decode_output_tok_s") is None for phase in phases)):
        return None
    return {
        "sum_request_decode_output_tok_s": sum(phase["decode_output_tok_s"] for phase in phases),
        "sum_request_decode_transition_tok_s": sum(phase["decode_transition_tok_s"] for phase in phases),
    }


def _token_ids(ids: Any, *, minimum: int = 1) -> list[int]:
    if (not isinstance(ids, list) or len(ids) < minimum
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError(f"prompt needs at least {minimum} non-negative integer token IDs")
    return list(ids)


def prepare_wave_rows(client, url, model, rows, flags):
    """Prefill every request before measuring any request in the cohort."""
    if len(rows) != len(flags):
        raise ValueError("one preparation route per request is required")
    prepared, observations = [], []
    for row, flag in zip(rows, flags, strict=True):
        if "prompt_token_ids" in row:
            ids = _token_ids(row["prompt_token_ids"], minimum=2)
        else:
            response = client.post(f"{url}/v1/hipengine/count_tokens", json={
                "messages": row["messages"], "enable_thinking": False,
            })
            response.raise_for_status()
            rendered = response.json()
            response = client.post(f"{url}/v1/hipengine/tokenize", json={"text": rendered["text"]})
            response.raise_for_status()
            tokens = response.json()
            ids = _token_ids(tokens["token_ids"], minimum=2)
            if len(ids) != rendered["token_count"] or len(ids) != tokens["token_count"]:
                raise ValueError("prepared prompt tokenization disagrees with server rendering")
        measured = {**row, "prompt_token_ids": ids}
        observation = _one(client, url, model, {**measured, "prompt_token_ids": ids[:-1]}, flag, 1)
        if not route_matches_request(observation):
            raise ValueError(f"preparation did not execute its requested route: {observation}")
        prepared.append(measured)
        observations.append(observation)
    return prepared, observations


def require_prepared_hit(case, *, expected_prompt_tokens):
    counts = (case.get("phase_metrics") or {}).get("counts") or {}
    if (counts.get("prompt_tokens") != expected_prompt_tokens
            or counts.get("cached_prompt_tokens") != expected_prompt_tokens - 1
            or counts.get("new_prompt_tokens") != 1):
        raise ValueError("prepared request did not reuse its declared prefix-minus-one checkpoint")


def _one(client: Any, url: str, model: str, row: dict[str, Any], mtp: bool, max_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model, "temperature": 0.0, "top_p": 1.0,
        "max_tokens": max_tokens, "ignore_eos": True, "speculative_mtp": mtp,
    }
    if "prompt_token_ids" in row:
        payload["prompt"] = _token_ids(row["prompt_token_ids"])
        endpoint = "/v1/completions"
    else:
        payload.update(messages=row["messages"], enable_thinking=False)
        endpoint = "/v1/chat/completions"
    started = time.perf_counter()
    response = client.post(f"{url}{endpoint}", json=payload)
    wall = time.perf_counter() - started
    body = response.json()
    if response.status_code != 200:
        return {"id": row["id"], "status": response.status_code, "wall_s": wall, "error": json.dumps(body)[:400]}
    count = exact_completion_count(body)
    if count != max_tokens:
        raise ValueError(f"requested {max_tokens} output tokens, received {count}")
    choice = (body.get("choices") or [{}])[0]
    usage = body.get("usage") or {}
    if "prompt" in payload and usage.get("prompt_tokens") != len(payload["prompt"]):
        raise ValueError("raw prompt token count differs from requested IDs")
    spec = (body.get("hipengine") or {}).get("speculative_mtp") or {}
    return {
        "id": row["id"], "category": row.get("category"),
        "status": 200, "wall_s": wall, "completion_tokens": count,
        "request_payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "prompt_token_ids": payload.get("prompt"),
        "phase_metrics": native_phase_metrics(body),
        "token_accounting": (body.get("hipengine") or {}).get("token_accounting"),
        "prompt_token_accounting": (body.get("hipengine") or {}).get("prompt_token_accounting"),
        "generation_shape": (body.get("hipengine") or {}).get("generation_shape"),
        "message": choice.get("message"), "text": choice.get("text"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "speculative_requested": mtp, "speculative_used": spec.get("used"),
        "speculative_reason": spec.get("decision_reason"),
        "accepted_per_cycle": spec.get("accepted_per_cycle"), "acceptance": spec.get("acceptance_rate"),
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
    *,
    prepared: bool = False,
) -> dict[str, Any]:
    import httpx

    results: list[dict[str, Any]] = [{} for _ in rows]
    flags = list(mtp) if isinstance(mtp, (list, tuple)) else [bool(mtp)] * len(rows)
    if len(flags) != len(rows):
        raise ValueError("one MTP flag per request is required")

    preparation = []
    if prepared:
        with httpx.Client(timeout=timeout) as client:
            rows, preparation = prepare_wave_rows(client, url, model, rows, flags)
    clients = [httpx.Client(timeout=timeout) for _ in rows]
    barrier = threading.Barrier(len(rows) + 1)

    def worker(index: int) -> None:
        barrier.wait()
        try:
            results[index] = _one(clients[index], url, model, rows[index], flags[index], max_tokens)
            if prepared:
                require_prepared_hit(results[index], expected_prompt_tokens=len(rows[index]["prompt_token_ids"]))
        except Exception as exc:
            results[index] = {**results[index], "id": rows[index]["id"], "status": 0, "error": str(exc)}

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(len(rows))]
    for thread in threads:
        thread.start()
    started = time.perf_counter()
    barrier.wait()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    for client in clients:
        client.close()
    tokens = sum(int(item.get("completion_tokens") or 0) for item in results)
    return {
        "wall_s": wall,
        "preparation": preparation,
        "phase_rates": summarize_phases(results),
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
    parser.add_argument("--prepared", action="store_true", help="Prefill every prompt except its final token before each timed wave; require exact prefix reuse.")
    parser.add_argument("--repetitions", type=int, default=3, help="Measured repetitions; rotate arm order between repetitions.")
    parser.add_argument("--server-manifest", type=Path, help="JSON with the separately captured resident server source, model and host identities.")
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
    if not rows or args.width <= 0 or args.max_tokens <= 0 or args.repetitions <= 0 or args.timeout <= 0:
        parser.error("prompts must be nonempty and width/max-tokens/repetitions/timeout must be positive")
    if args.prepared and args.max_tokens < 2:
        parser.error("prepared decode needs at least two output tokens")
    selected = [item.strip() for item in args.arms.split(",") if item.strip()]
    if not selected or len(set(selected)) != len(selected) or set(selected) - {"ar", "mtp", "mixed"}:
        parser.error("arms must be unique names from ar,mtp,mixed")
    from scripts.qwen4exp_mtp_inprocess_bench import source_snapshot

    source_before = source_snapshot()
    manifest = None if args.server_manifest is None else json.loads(args.server_manifest.read_text())
    waves = [rows[start : start + args.width] for start in range(0, len(rows), args.width)]
    arms: dict[str, list[dict[str, Any]]] = {label: [] for label in selected}

    def measure(label, wave):
        flags = [True] + [False] * (len(wave) - 1) if label == "mixed" else label == "mtp"
        options = {"prepared": True} if args.prepared else {}
        try:
            return _wave(wave, args.url, args.model, flags, args.max_tokens, args.timeout, **options)
        except Exception as exc:
            # Retain preparation failures rather than losing the run's artifact.
            return {"wall_s": 0.0, "completion_tokens": 0, "phase_rates": None,
                    "cases": [{"id": row["id"], "status": 0, "error": str(exc)} for row in wave]}

    warmups = [measure(label, waves[0]) for label in selected] if args.warmup else []
    for repetition in range(args.repetitions):
        offset = repetition % len(selected)
        order = selected[offset:] + selected[:offset]
        for label in order:
            for wave_index, wave in enumerate(waves):
                arms[label].append({**measure(label, wave), "repetition": repetition,
                                    "wave_index": wave_index, "arm_order": order})
    source_after = source_snapshot()

    def totals(label: str) -> dict[str, Any]:
        wall = sum(item["wall_s"] for item in arms[label])
        tokens = sum(item["completion_tokens"] for item in arms[label])
        cases = [case for item in arms[label] for case in item["cases"]]
        phase_rates = [item.get("phase_rates") for item in arms[label]]
        phase_means = None if not all(phase_rates) else {
            f"mean_{key}": sum(row[key] for row in phase_rates) / len(phase_rates)
            for key in phase_rates[0]
        }
        return {
            "phase_rate_means": phase_means,
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
        for wave in [*warmups, *(wave for label in selected for wave in arms[label])]
        for case in wave["cases"]
    )
    valid = valid_routes and source_before == source_after
    artifact = {
        "schema": "hipengine.qwen4exp.http_mtp_economics.v3",
        "performance_claim": False,
        "status": "measured_diagnostic" if valid else "invalid",
        "checks": {"every_request_used_requested_route": valid_routes, "harness_source_unchanged": source_before == source_after},
        "harness_source": source_before,
        "harness_source_after": source_after,
        "server_manifest": manifest,
        "command": [str(Path(__file__).relative_to(REPO_ROOT)), *(sys.argv[1:] if argv is None else argv)],
        "prompt_file_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "repetitions": args.repetitions,
        "preparation": "prefix_minus_one" if args.prepared else "none",
        "metric_definitions": {
            "aggregate_tok_s": "native output tokens / client cohort wall, excluding preparation and client construction",
            "sum_request_decode_output_tok_s": "sum(outputs / backend decode wall); Gufo published output-count convention",
            "sum_request_decode_transition_tok_s": "sum((outputs - 1) / backend decode wall); first output belongs to prefill",
            "backend_wall": "includes scheduler waits; not isolated GPU time",
        },
        "warmups": warmups,
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
            if valid and label != "ar" and measured.get("ar", {}).get("aggregate_tok_s")
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
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
