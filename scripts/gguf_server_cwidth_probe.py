#!/usr/bin/env python3
"""C-width server decode baseline for the IKV-C2 campaign (roadmap P5).

Measures the CURRENT C>1 state on the repaired INT8 route before any
row-batched consumer work: N concurrent completions with the same fixed
token-ID fixture through the real server, plus a REAL same-server serial
control that fires the same N requests one at a time. Both phases report
complete-request throughput (every prompt and completion token of every lane
over that phase's wall) and per-lane walls.

Measured, not inferred:

- per-lane ``completion_tokens`` / ``prompt_tokens`` come from each request's
  own ``usage`` block, not from one lane or from ``--max-tokens``;
- decode and prefill model-step counts come from the server's Prometheus
  counters (``hipengine_resident_work_decode_total`` /
  ``hipengine_resident_work_prefill_total``), read before and after each
  phase;
- the route/fallback counters (``hipengine_resident_route_total``,
  ``hipengine_resident_fallback_total``) and the last execution manifest
  (``hipengine_resident_route_manifest_info``) record the physical width and
  any serial fallback, so a width that never grouped cannot be reported as a
  batched win.

There is deliberately NO decode-only tok/s field. An earlier version derived
one by subtracting a single prefill wall from the concurrent wall and compared
it against N x C1; the 2026-09-10 review withdrew those fields as invalid (a
rate one GPU cannot multiply). Decode efficiency is reported only as measured
model steps over the measured phase wall, labeled as such.

Output: JSON artifact with per-width concurrent and serial records.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROM_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[^\s]+)\s*$"
)
_PROM_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> dict[str, float]:
    """Flatten Prometheus text into ``name`` and ``name{labels}`` keys.

    Only finite numeric samples are kept; ``NaN``/``+Inf`` and comment/HELP
    lines are dropped so a caller never propagates a non-JSON float.
    """

    out: dict[str, float] = {}
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue
        labels = match.group("labels")
        if labels:
            parts = [
                f'{name}="{value}"'
                for name, value in _PROM_LABEL.findall(labels)
            ]
            key = f"{match.group('name')}{{{','.join(sorted(parts))}}}"
        else:
            key = match.group("name")
        out[key] = value
    return out


def labeled_counter(
    metrics: dict[str, float],
    name: str,
) -> dict[str, float]:
    """Return ``{label_value: value}`` for a single-label Prometheus counter."""

    prefix = f"{name}{{"
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if not key.startswith(prefix):
            continue
        body = key[len(prefix) :].rstrip("}")
        # Single-label series only: the label name is fixed by the exporter.
        if "=" not in body or "," in body:
            continue
        _, _, raw = body.partition("=")
        out[raw.strip('"')] = float(value)
    return out


def _metric_delta(
    before: dict[str, float],
    after: dict[str, float],
    name: str,
) -> float | None:
    """Delta of a counter, or None when the exporter did not report it."""

    if name not in before or name not in after:
        return None
    return float(after[name]) - float(before[name])


def _usage_counts(payload: object) -> tuple[int, int]:
    """Return ``(prompt_tokens, completion_tokens)`` from a response payload.

    Missing or malformed usage is reported as ``(-1, -1)`` so a caller can tell
    "the server did not say" from "the server said zero".
    """

    if not isinstance(payload, dict):
        return -1, -1
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return -1, -1
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    try:
        return int(prompt), int(completion)
    except (TypeError, ValueError):
        return -1, -1


def build_phase_record(
    *,
    width: int,
    phase: str,
    wall_s: float,
    lane_results: list[tuple[float, dict]],
    lane_errors: list[str],
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
) -> dict[str, object]:
    """Assemble one concurrent or serial phase record.

    Pure: no device, server, or filesystem access, so the artifact shape and
    the accounting rules are testable without a GPU.
    """

    record: dict[str, object] = {
        "phase": phase,
        "wall_s": round(float(wall_s), 3),
    }
    if lane_errors or len(lane_results) != int(width):
        record["status"] = "request_failed"
        record["errors"] = list(lane_errors)[:4]
        record["completed_lanes"] = len(lane_results)
        record["expected_lanes"] = int(width)
        return record

    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    per_request_wall: list[float] = []
    for wall, payload in lane_results:
        prompt, completion = _usage_counts(payload)
        prompt_tokens.append(prompt)
        completion_tokens.append(completion)
        per_request_wall.append(round(float(wall), 3))

    if any(count < 0 for count in prompt_tokens + completion_tokens):
        record["status"] = "usage_missing"
        record["per_request_wall_s"] = per_request_wall
        record["prompt_tokens_reported"] = prompt_tokens
        record["completion_tokens_reported"] = completion_tokens
        return record

    total_tokens = sum(prompt_tokens) + sum(completion_tokens)
    record.update(
        {
            "status": "pass",
            "per_request_wall_s": per_request_wall,
            "prompt_tokens_reported": prompt_tokens,
            "completion_tokens_reported": completion_tokens,
            "prompt_tokens_total": int(sum(prompt_tokens)),
            "completion_tokens_total": int(sum(completion_tokens)),
            "complete_request_throughput_tok_s": round(
                total_tokens / max(float(wall_s), 1e-9), 3
            ),
        }
    )
    decode_steps = _metric_delta(
        metrics_before, metrics_after, "hipengine_resident_work_decode_total"
    )
    prefill_steps = _metric_delta(
        metrics_before, metrics_after, "hipengine_resident_work_prefill_total"
    )
    record["measured_decode_steps"] = (
        None if decode_steps is None else int(decode_steps)
    )
    record["measured_prefill_steps"] = (
        None if prefill_steps is None else int(prefill_steps)
    )
    record["measured_decode_steps_per_s"] = (
        None
        if decode_steps is None
        else round(float(decode_steps) / max(float(wall_s), 1e-9), 3)
    )
    record["measured_decode_steps_note"] = (
        "measured model decode steps over the phase wall; a scheduling-rate "
        "observation, not isolated decode efficiency"
    )
    fallbacks = {
        key: value
        for key, value in labeled_counter(
            metrics_after, "hipengine_resident_fallback_total"
        ).items()
    }
    if metrics_before:
        before_fallbacks = labeled_counter(
            metrics_before, "hipengine_resident_fallback_total"
        )
        fallbacks = {
            key: value - float(before_fallbacks.get(key, 0.0))
            for key, value in fallbacks.items()
        }
    record["fallback_reasons_delta"] = fallbacks
    record["route_counts_delta"] = {
        key: (
            value
            - float(
                labeled_counter(
                    metrics_before, "hipengine_resident_route_total"
                ).get(key, 0.0)
            )
        )
        for key, value in labeled_counter(
            metrics_after, "hipengine_resident_route_total"
        ).items()
    }
    manifest = manifest_from_metrics(metrics_after)
    if manifest:
        record["execution_manifest"] = manifest
    return record


def manifest_from_metrics(metrics: dict[str, float]) -> dict[str, str]:
    """Extract the last resident execution manifest's labels, if present."""

    for key in metrics:
        if not key.startswith("hipengine_resident_route_manifest_info{"):
            continue
        body = key[len("hipengine_resident_route_manifest_info{") :].rstrip("}")
        out: dict[str, str] = {}
        for name, value in _PROM_LABEL.findall(body):
            out[name] = value
        return out
    return {}


def serial_reference(
    serial_record: dict[str, object],
    concurrent_record: dict[str, object],
) -> dict[str, object]:
    """Compare a width's concurrent phase against its measured serial control."""

    out: dict[str, object] = {}
    serial_rate = serial_record.get("complete_request_throughput_tok_s")
    concurrent_rate = concurrent_record.get("complete_request_throughput_tok_s")
    if isinstance(serial_rate, (int, float)) and serial_rate:
        out["serial_complete_request_rate_tok_s"] = serial_rate
    if (
        isinstance(serial_rate, (int, float))
        and isinstance(concurrent_rate, (int, float))
        and serial_rate > 0
    ):
        out["concurrent_over_serial_ratio"] = round(
            float(concurrent_rate) / float(serial_rate), 4
        )
        out["concurrent_over_serial_note"] = (
            "same server, same card, same prompts: concurrent complete-request "
            "rate divided by the measured serial rate. A ratio near 1 means the "
            "width buys no aggregate throughput on this route."
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    ap.add_argument("--widths", default="1,2,4")
    ap.add_argument("--prompt-rows", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-context-tokens", type=int, default=16384)
    ap.add_argument("--vocab-span", type=int, default=32000)
    ap.add_argument("--port", type=int, default=18271)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--no-serial-control",
        action="store_true",
        help="skip the measured same-server serial control phase",
    )
    args = ap.parse_args()

    import numpy as np

    rng = np.random.default_rng(20260909)
    # Distinct-but-deterministic prompts per lane so lanes do not share a
    # prefix accidentally.
    prompts = [
        [int(t) for t in rng.integers(1000, args.vocab_span, size=args.prompt_rows)]
        for _ in range(max(int(w) for w in str(args.widths).split(",")))
    ]
    model_id = Path(args.model).name

    def launch_server(width: int) -> subprocess.Popen:
        env = os.environ.copy()
        env.setdefault("HIP_VISIBLE_DEVICES", "0")
        env.setdefault("GPU_MAX_HW_QUEUES", "1")
        env.setdefault("HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG", "1")
        env.setdefault("HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS", "none")
        cmd = [
            sys.executable, "-m", "hipengine.server",
            "--model", str(args.model),
            "--backend", "hip_gfx1100",
            "--quant", "gguf_q4_k_m",
            "--kv-storage", "int8_per_token_head",
            "--kv-scale-dtype", "fp32",
            "--kv-scale-granularity", "per_token_head",
            "--max-context-tokens", str(int(args.max_context_tokens)),
            "--max-active-requests", str(int(width)),
            "--speculative-mtp-serving", "off",
            "--prefix-cache", "off",
            "--metrics", "prometheus",
            "--port", str(int(args.port)),
        ]
        return subprocess.Popen(
            cmd, env=env, cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def wait_ready(srv: subprocess.Popen, timeout: float = 600.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if srv.poll() is not None:
                return False
            time.sleep(1.0)
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{args.port}/health", timeout=2
                ).read()
                return True
            except Exception:
                continue
        return False

    def scrape_metrics() -> dict[str, float]:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/metrics", timeout=10
            ) as response:
                return parse_prometheus(response.read().decode("utf-8", "replace"))
        except Exception:
            return {}

    def one_request(prompt: list[int], max_tokens: int) -> tuple[float, dict]:
        body = json.dumps({
            "model": model_id,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/completions",
            data=body, headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=900) as r:
            payload = json.loads(r.read())
        return time.perf_counter() - t0, payload

    def run_concurrent(width: int) -> tuple[float, list, list]:
        results: list[tuple[float, dict]] = []
        errors: list[str] = []
        threads: list[threading.Thread] = []

        def worker(i: int) -> None:
            try:
                results.append(one_request(prompts[i], args.max_tokens))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"lane {i}: {exc}")

        t_start = time.perf_counter()
        for i in range(width):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return time.perf_counter() - t_start, results, errors

    def run_serial(width: int) -> tuple[float, list, list]:
        results: list[tuple[float, dict]] = []
        errors: list[str] = []
        t_start = time.perf_counter()
        for i in range(width):
            try:
                results.append(one_request(prompts[i], args.max_tokens))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"lane {i}: {exc}")
        return time.perf_counter() - t_start, results, errors

    out: dict[str, object] = {
        "kind": "gguf_server_cwidth_decode_baseline",
        "protocol_tier": 1,
        "model": str(args.model),
        "kv": "int8_per_token_head + fp32 scales",
        "route": "repaired C1 route (slot-local prefill, packed batch decode above 1 slot, per-width pool lease)",
        "prompt_rows": int(args.prompt_rows),
        "max_tokens": int(args.max_tokens),
        "prompt_kind": "deterministic_varied_rng20260909_lanes",
        "withdrawn": (
            "aggregate_decode_tok_s_est and the 31%/13%-of-serial conclusions "
            "from the 2026-09-10 artifact are withdrawn as invalid: they "
            "subtracted one isolated prefill from a wall containing width "
            "prefills and compared against N x C1, a rate one GPU cannot "
            "multiply."
        ),
        "widths": {},
    }

    widths = [int(w) for w in str(args.widths).split(",") if w.strip()]
    for width in widths:
        srv = launch_server(width)
        try:
            if not wait_ready(srv):
                out["widths"][str(width)] = {"status": "server_failed"}
                continue
            before = scrape_metrics()
            wall, results, errors = run_concurrent(width)
            after = scrape_metrics()
            concurrent = build_phase_record(
                width=width,
                phase="concurrent",
                wall_s=wall,
                lane_results=results,
                lane_errors=errors,
                metrics_before=before,
                metrics_after=after,
            )
            serial: dict[str, object] = {"status": "skipped"}
            if not args.no_serial_control:
                s_before = scrape_metrics()
                s_wall, s_results, s_errors = run_serial(width)
                s_after = scrape_metrics()
                serial = build_phase_record(
                    width=width,
                    phase="serial",
                    wall_s=s_wall,
                    lane_results=s_results,
                    lane_errors=s_errors,
                    metrics_before=s_before,
                    metrics_after=s_after,
                )
            record: dict[str, object] = {
                "status": concurrent.get("status"),
                "concurrent": concurrent,
                "serial_control": serial,
            }
            record.update(serial_reference(serial, concurrent))
            record["teardown"] = "pending"
            out["widths"][str(width)] = record
            print(
                f"[C{width}] concurrent wall "
                f"{concurrent.get('wall_s')} s, complete-request "
                f"{concurrent.get('complete_request_throughput_tok_s')} tok/s; "
                f"serial wall {serial.get('wall_s')} s, complete-request "
                f"{serial.get('complete_request_throughput_tok_s')} tok/s; "
                f"ratio {record.get('concurrent_over_serial_ratio')}",
                flush=True,
            )
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=30)
                if str(width) in out["widths"]:
                    out["widths"][str(width)]["teardown"] = "clean"
            except Exception:
                srv.kill()

    payload = json.dumps(out, indent=2, default=str)
    print(payload)
    if args.json:
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
