#!/usr/bin/env python3
"""Server-faithful allocation probe: the real server binary, real pool, real route.

`scripts/gguf_capacity_probe.py` proves the *direct session's* memory envelope
(scalar bulk prefill). It cannot answer the server's: the server binds
requests to the shared pool with slot views, leases a packed-execution
workspace, and runs every ``int8_direct`` request through the packed slot-local
prefill entry (multi-slab prompts take per-layer BF16 oracles and the AOTriton
route). This probe drives ONE request through the actual server process and
scrapes the owner-deduplicated observability gauges (P0) at high cadence
during the request, so per-domain peaks are captured *while live*, not
inferred from post-request counters.

Per the two-tier capacity protocol (benchmarks/HARNESSES.md) this is still a
Tier-1 allocation probe: it never searches a bound with full prompts. Its
prompt is deliberately multi-slab (default 2,048 rows = two 1,024-row slabs on
the 27B dense geometry) so the executor's real working set - per-layer oracles,
packed workspace, hidden owners - is actually allocated; pass a boundary
prompt with --prompt-rows to cross a specific slab/page threshold.

Domains reported separately (never summed into one number):
  pool            canonical KV pool backing (current / high-water / pinned)
  workspace       packed execution scratch (owner-deduplicated unique bytes)
  workspace lease pool-plane pages pinned by the workspace lease
  prefill oracle  live BF16 oracle pair bytes while the request runs
  hidden/bulk     bulk prefill hidden/scratch/token owners
  persistent KV   request-owned payload/scale bytes after the request

Example:

    HIP_VISIBLE_DEVICES=0 python3 scripts/gguf_server_allocation_probe.py \
        --max-context-tokens 16384 --prompt-rows 2048 --json /tmp/probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROMPT_FIXTURE = REPO_ROOT / "benchmarks" / "parity" / "r0r4-c1-fixture.json"

_TRACKED_GAUGES = (
    "hipengine_kv_pool_current_bytes",
    "hipengine_kv_pool_high_water_observed_bytes",
    "hipengine_kv_pool_pinned_pages",
    "hipengine_resident_packed_workspace_current_bytes",
    "hipengine_resident_packed_workspace_leased_pool_bytes",
    "hipengine_resident_prefill_oracle_owner_bytes",
    "hipengine_resident_prefill_oracle_owners",
    "hipengine_resident_prefill_oracle_observed_peak_bytes",
    "hipengine_resident_prefill_oracle_observed_peak_owners",
    "hipengine_resident_prefill_hidden_owner_bytes",
    "hipengine_resident_kv_total_bytes",
)


def _parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        name, raw = parts
        try:
            values[name] = float(raw)
        except ValueError:
            continue
    return values


def _scrape(port: int, timeout: float = 5.0) -> dict[str, float] | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=timeout
        ) as response:
            return _parse_metrics(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--max-context-tokens", type=int, required=True)
    parser.add_argument("--prompt-rows", type=int, default=2048,
                        help="Request prompt rows (default 2048 = two 1,024-row slabs on the 27B)")
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--max-active-requests", type=int, default=1)
    parser.add_argument("--port", type=int, default=18430)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    # Exact fixture for the parity manifest's 2,048-row case; the same
    # deterministic spec otherwise (never a fresh RNG across shapes).
    if args.prompt_rows == 2048 and PROMPT_FIXTURE.exists():
        fixture = json.loads(PROMPT_FIXTURE.read_text(encoding="utf-8"))
        prompt_ids = [int(t) for t in fixture["case"]["prompt_token_ids"]]
        fixture_sha = fixture["case"]["prompt_token_ids_sha256"]
    else:
        import numpy as np

        rng = np.random.default_rng(20260910)
        prompt_ids = [
            int(t) for t in rng.integers(1000, 32000, size=int(args.prompt_rows))
        ]
        import hashlib

        fixture_sha = hashlib.sha256(
            __import__("numpy").asarray(prompt_ids, dtype="<i8").tobytes()
        ).hexdigest()

    server_env = os.environ.copy()
    server_env.setdefault("HIP_VISIBLE_DEVICES", "0")
    server_env.setdefault("GPU_MAX_HW_QUEUES", "1")
    server_env.setdefault("HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG", "1")
    server_env.setdefault("HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS", "none")
    command = [
        sys.executable, "-m", "hipengine.server",
        "--model", str(args.model),
        "--backend", "hip_gfx1100",
        "--quant", "gguf_q4_k_m",
        "--kv-storage", "int8_per_token_head",
        "--kv-scale-dtype", "fp32",
        "--kv-scale-granularity", "per_token_head",
        "--max-context-tokens", str(int(args.max_context_tokens)),
        "--max-active-requests", str(int(args.max_active_requests)),
        "--speculative-mtp-serving", "off",
        "--prefix-cache", "off",
        "--metrics", "prometheus",
        "--port", str(int(args.port)),
    ]
    server_log = (
        Path("/tmp") / f"server-alloc-probe-{int(time.time())}.log"
    )
    result: dict[str, object] = {
        "kind": "gguf_server_allocation_probe",
        "protocol_tier": 1,
        "server_route": "packed slot-local int8_direct (AOTriton default on)",
        "model": str(args.model),
        "max_context_tokens": int(args.max_context_tokens),
        "prompt_rows": int(args.prompt_rows),
        "prompt_token_ids_sha256": fixture_sha,
        "decode_tokens": int(args.decode_tokens),
        "max_active_requests": int(args.max_active_requests),
        "command": command,
        "server_log": str(server_log),
    }

    started = time.perf_counter()
    with server_log.open("w") as log_handle:
        server = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=server_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        try:
            baseline = None
            deadline = time.perf_counter() + float(args.startup_timeout)
            while time.perf_counter() < deadline:
                if server.poll() is not None:
                    result["status"] = "server_exited_during_startup"
                    result["server_exit_code"] = int(server.returncode)
                    print(json.dumps(result, indent=2, default=str))
                    return 1
                baseline = _scrape(args.port)
                if baseline is not None:
                    break
                time.sleep(1.0)
            if baseline is None:
                result["status"] = "startup_timeout"
                print(json.dumps(result, indent=2, default=str))
                return 1
            startup_seconds = time.perf_counter() - started

            peaks: dict[str, float] = {}
            stop = threading.Event()

            def poller() -> None:
                while not stop.is_set():
                    snapshot = _scrape(args.port, timeout=2.0)
                    if snapshot:
                        for name in _TRACKED_GAUGES:
                            value = snapshot.get(name)
                            if value is not None and value > peaks.get(name, float("-inf")):
                                peaks[name] = value
                    stop.wait(float(args.poll_interval))

            thread = threading.Thread(target=poller, daemon=True)
            thread.start()
            request_started = time.perf_counter()
            body = json.dumps(
                {
                    "prompt": prompt_ids,
                    "max_tokens": int(args.decode_tokens),
                    "temperature": 0.0,
                    "stream": False,
                }
            ).encode("utf-8")
            request_error = None
            response_text: str | None = None
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/v1/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=3600) as response:
                    response_text = response.read().decode("utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001 - probe reports, not raises
                request_error = f"{type(exc).__name__}: {exc}"[:400]
            request_wall = time.perf_counter() - request_started
            time.sleep(0.5)  # one final scrape window
            stop.set()
            thread.join(timeout=5.0)
            final = _scrape(args.port) or {}

            completed = response_text is not None and request_error is None
            usage = {}
            generated = None
            if completed:
                payload = json.loads(response_text)
                choices = payload.get("choices") or []
                generated = (choices[0].get("text") if choices else None)
                usage = payload.get("usage") or {}

            result.update(
                {
                    "startup_seconds": round(startup_seconds, 3),
                    "request_wall_seconds": round(request_wall, 3),
                    "status": "pass" if completed else "request_failed",
                    "request_error": request_error,
                    "usage": usage,
                    "generated_chars": None if generated is None else len(generated),
                    "baseline": {
                        name: baseline.get(name) for name in _TRACKED_GAUGES
                    },
                    "during_request_peaks": dict(sorted(peaks.items())),
                    "final": {name: final.get(name) for name in _TRACKED_GAUGES},
                    "domains_gib": {
                        "pool_high_water": round(
                            peaks.get(
                                "hipengine_kv_pool_high_water_observed_bytes", 0
                            )
                            / 2**30,
                            4,
                        ),
                        "pool_pinned_pages": peaks.get(
                            "hipengine_kv_pool_pinned_pages", 0
                        ),
                        "workspace_owner_deduped": round(
                            peaks.get(
                                "hipengine_resident_packed_workspace_current_bytes", 0
                            )
                            / 2**30,
                            4,
                        ),
                        "workspace_leased_pool": round(
                            peaks.get(
                                "hipengine_resident_packed_workspace_leased_pool_bytes",
                                0,
                            )
                            / 2**30,
                            4,
                        ),
                        "prefill_oracle_live": round(
                            peaks.get(
                                "hipengine_resident_prefill_oracle_owner_bytes", 0
                            )
                            / 2**30,
                            4,
                        ),
                        "prefill_oracle_observed_peak": round(
                            final.get(
                                "hipengine_resident_prefill_oracle_observed_peak_bytes",
                                peaks.get(
                                    "hipengine_resident_prefill_oracle_observed_peak_bytes",
                                    0,
                                ),
                            )
                            / 2**30,
                            4,
                        ),
                        "prefill_oracle_observed_peak_owners": final.get(
                            "hipengine_resident_prefill_oracle_observed_peak_owners",
                            peaks.get(
                                "hipengine_resident_prefill_oracle_observed_peak_owners",
                                0,
                            ),
                        ),
                        "prefill_oracle_owners": peaks.get(
                            "hipengine_resident_prefill_oracle_owners", 0
                        ),
                        "hidden_bulk_owners": round(
                            peaks.get(
                                "hipengine_resident_prefill_hidden_owner_bytes", 0
                            )
                            / 2**30,
                            4,
                        ),
                        "request_owned_kv_after": round(
                            final.get("hipengine_resident_kv_total_bytes", 0) / 2**30,
                            4,
                        ),
                    },
                }
            )
        finally:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=60)
                result["server_teardown"] = "clean"
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=30)
                result["server_teardown"] = "killed_after_timeout"

    payload = json.dumps(result, indent=2, default=str)
    print(payload)
    if args.json:
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0 if result.get("status") == "pass" and result.get("server_teardown") == "clean" else 1


if __name__ == "__main__":
    raise SystemExit(main())
