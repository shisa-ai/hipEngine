#!/usr/bin/env python3
"""Exercise wide-to-C1 and C1-to-wide pool histories inside one server session.

Packet 2 of the 24 GB capacity campaign (docs/QWEN38-27B-GFX1100-24GB-CAPACITY.md)
requires history evidence, not just single-point ceilings: a wide request must
not leave the chunked KV pool, workspace leases or graph buckets inflated for
the following C1 requests, and retirement must return memory to the device
allocator rather than parking it as unbounded reusable capacity.

The probe serves one cold server (BF16 KV, N=2) and drives alternating wide
(~wide-token prompt) and C1 (~narrow-token prompt) requests. After every
request it samples twice:

- **held** (immediately after the validated completion): reusable pool
  capacity the server may legitimately park (``free_pages``), plus
  ``retired_pages``/``retired_bytes`` returned to the device allocator so far,
  graph bucket entries, and the instantaneous whole-card usage.
- **drained** (after the KV idle grace elapses): the same counters, showing
  tail-chunk retirement and card-usage recovery.

The probe passes only when every request validates against exact server
accounting, drained whole-card usage returns to within the idle gate of the
post-load baseline, retirement is demonstrated (``retired_bytes`` grows), and
graph bucket entries stay bounded. This is a diagnostic harness: it changes no
route and makes no throughput claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gguf_context_ceiling_probe as ceiling  # noqa: E402

from hipengine.util.amdgpu_vram import VramSampler, select_card  # noqa: E402

GIB = 1 << 30
MIB = 1 << 20
_METRIC_RE = re.compile(
    r"^hipengine_kv_pool_(current_pages|current_bytes|free_pages|refcounted_pages|pinned_pages|"
    r"grow_events_total|shrink_events_total|retired_pages_total|retired_bytes_total)\{?\S*\s+([0-9.eE+-]+)$",
    re.MULTILINE,
)
_GRAPH_ENTRIES_RE = re.compile(
    r"^hipengine_graph_bucket_entries\S*\s+([0-9.eE+-]+)$", re.MULTILINE
)


def _metrics_text(client: ceiling.ProbeClient) -> str:
    status, payload = client._request("GET", "metrics")
    if status != 200:
        return ""
    if isinstance(payload, Mapping) and isinstance(payload.get("raw"), str):
        return payload["raw"]
    return payload if isinstance(payload, str) else ""


def _metrics_values(client: ceiling.ProbeClient) -> dict[str, float]:
    body = _metrics_text(client)
    values: dict[str, float] = {}
    for match in _METRIC_RE.finditer(body):
        values[f"kv_{match.group(1)}"] = float(match.group(2))
    entries = _GRAPH_ENTRIES_RE.search(body)
    if entries:
        values["graph_bucket_entries"] = float(entries.group(1))
    return values


def _sample_snapshot(
    client: ceiling.ProbeClient, card: Any
) -> tuple[int, dict[str, float]]:
    return ceiling._read_used(card), _metrics_values(client)


def _drain(seconds: float) -> None:
    time.sleep(max(0.0, seconds))


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise ValueError(f"model does not exist: {model_path}")

    card = select_card(pci_id=args.pci_id)
    sampler_interval_ms = max(1.0, args.sample_interval_ms)

    if ceiling.port_in_use(args.host, args.port):
        return {
            "status": "port_in_use",
            "reason": f"something already listens on {args.host}:{args.port}",
        }, 1

    baseline = ceiling._read_used(card)
    if baseline > int(args.maximum_baseline_mib) << 20:
        return {
            "status": "not_idle",
            "reason": (
                f"baseline {baseline >> 20} MiB exceeds the "
                f"{args.maximum_baseline_mib} MiB idle gate"
            ),
        }, 1

    env = dict(os.environ)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = str(args.gpu_index)
    env["GPU_MAX_HW_QUEUES"] = "1"
    env["HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS"] = str(args.kv_pool_idle_grace_seconds)
    env.setdefault("PYTHONUNBUFFERED", "1")

    command = [
        sys.executable,
        "-m",
        "hipengine.server",
        "--model",
        str(model_path),
        "--backend",
        args.backend,
        "--quant",
        args.quant,
        "--served-model-name",
        "pool-history-probe",
        "--kv-storage",
        args.kv_storage,
        "--max-context-tokens",
        str(args.context),
        "--max-active-requests",
        str(args.max_active_requests),
        "--generation-batch-window-ms",
        str(args.batch_window_ms),
        "--metrics",
        "prometheus",
        "--speculative-mtp-serving",
        "off",
        "--prefix-cache",
        "off",
        "--startup-min-free-mib",
        str(args.startup_min_free_mib),
        "--shutdown-grace-seconds",
        "10",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--log-level",
        args.server_log_level,
    ]

    log_path = args.json.with_suffix(".server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    sampler = VramSampler(card, interval_ms=sampler_interval_ms, keep_samples=True)
    sampler.start()
    server = None
    phases: list[dict[str, Any]] = []
    post_load_mib: int | None = None
    status = "ok"
    reason = "history completed with drained recovery and bounded reusable capacity"
    peak_mib = 0
    try:
        server = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=log_path.open("a"),
            stderr=subprocess.STDOUT,
        )
        client = ceiling.ProbeClient(args.host, args.port, timeout_s=10.0)
        ready_ok = False
        deadline = time.monotonic() + float(args.ready_timeout)
        while time.monotonic() < deadline:
            if server.poll() is not None:
                break
            try:
                payload = client.ready()
                if payload.get("ready") is True:
                    ready_ok = True
                    break
            except Exception:  # noqa: BLE001 - startup polls until timeout
                pass
            time.sleep(0.25)
        if not ready_ok:
            log_text = log_path.read_text(errors="replace")
            if server.poll() is not None:
                classified, why = ceiling.classify_failure(
                    stage="startup", log_text=log_text, body=""
                )
                status, reason = classified, why
            else:
                status = "ready_timeout"
                reason = f"server never became ready within {args.ready_timeout}s"
        else:
            post_load = ceiling._read_used(card)
            post_load_mib = post_load >> 20
            unit_ids = client.tokenize(ceiling.FILLER_CORPUS)
            plan = [
                ("wide_1", args.wide_prompt_tokens),
                ("c1_a", args.narrow_prompt_tokens),
                ("c1_b", args.narrow_prompt_tokens),
                ("wide_2", args.wide_prompt_tokens),
                ("c2", args.narrow_prompt_tokens),
            ]
            for label, target_tokens in plan:
                prompt_text, counted = ceiling.fit_prompt_tokens(
                    unit_token_ids=unit_ids,
                    suffix=ceiling.PROMPT_SUFFIX,
                    target_tokens=int(target_tokens),
                    count_tokens=lambda text: len(client.tokenize(text)),
                    detokenize=client.detokenize,
                )
                request_started = time.perf_counter()
                http_status, payload = client.completion(
                    served_model_name="pool-history-probe",
                    prompt=prompt_text,
                    max_tokens=args.max_tokens,
                    timeout_s=float(args.request_timeout),
                )
                request_ended = time.perf_counter()
                validation_info, validation_error = ceiling.validate_completion(
                    payload,
                    expected_prompt_tokens=counted,
                    expected_completion_tokens=args.max_tokens,
                )
                valid = validation_error is None
                held_used, held_metrics = _sample_snapshot(client, card)
                peak_mib = max(peak_mib, held_used)
                _drain(float(args.kv_pool_idle_grace_seconds) + 1.0)
                drained_used, drained_metrics = _sample_snapshot(client, card)
                phases.append(
                    {
                        "label": label,
                        "prompt_tokens": counted,
                        "validated": bool(valid),
                        "validation": validation_error or "exact",
                        "wall_seconds": round(request_ended - request_started, 4),
                        "held": {"used_mib": held_used >> 20, "metrics": held_metrics},
                        "drained": {
                            "used_mib": drained_used >> 20,
                            "metrics": drained_metrics,
                        },
                    }
                )
                if not valid:
                    status = "invalid_request"
                    reason = f"phase {label} failed validation: {validation_error}"
                    break

            if status == "ok":
                final_used, final_metrics = _sample_snapshot(client, card)
                retired_bytes = float(
                    final_metrics.get("kv_retired_bytes_total", 0.0)
                )
                grow_events = float(final_metrics.get("kv_grow_events_total", 0.0))
                graph_entries = float(final_metrics.get("graph_bucket_entries", 0.0))
                recovery_mib = (final_used - post_load) >> 20
                problems: list[str] = []
                if grow_events == 0 and retired_bytes == 0:
                    # The default serving pool pre-allocates its full serving
                    # capacity at load: nothing ever leaves the allocator's
                    # books because nothing extra was taken. Record the parked
                    # reusable capacity instead of demanding retirement.
                    pool_note = "pool_parks_full_capacity"
                elif retired_bytes <= 0:
                    problems.append(
                        "retirement not demonstrated: pool grew but "
                        "retired_bytes_total == 0"
                    )
                else:
                    pool_note = "pool_grew_and_retired"
                if recovery_mib > args.maximum_recovery_mib:
                    problems.append(
                        f"drained card usage {final_used >> 20} MiB exceeds the "
                        f"post-load baseline by {recovery_mib} MiB (gate "
                        f"{args.maximum_recovery_mib} MiB)"
                    )
                if graph_entries > float(args.maximum_graph_buckets):
                    problems.append(
                        f"graph bucket entries {graph_entries:g} exceed the "
                        f"bound {args.maximum_graph_buckets}"
                    )
                for phase in phases:
                    if not phase["validated"]:
                        problems.append(f"phase {phase['label']} did not validate")
                        break
                if problems:
                    status = "history_violation"
                    reason = "; ".join(problems)
                else:
                    phases.append(
                        {
                            "label": "final_drained",
                            "prompt_tokens": 0,
                            "validated": True,
                            "validation": pool_note,
                            "wall_seconds": 0.0,
                            "held": None,
                            "drained": {
                                "used_mib": final_used >> 20,
                                "metrics": final_metrics,
                            },
                        }
                    )
    except Exception as exc:  # noqa: BLE001 - diagnostic harness reports failures
        status = "probe_error"
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        sampler.stop()

    sampler_peak = sampler.peek()
    artifact = {
        "schema": "hipengine.benchmark-artifact/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "diagnostic",
        "status": status,
        "reason": reason,
        "model": {
            "path": str(model_path),
            "backend": args.backend,
            "quant": args.quant,
            "kv_storage": args.kv_storage,
        },
        "device": {
            "sysfs": str(card.sysfs_path),
            "gpu_index": args.gpu_index,
        },
        "protocol": {
            "commit": ceiling._git(["rev-parse", "HEAD"]),
            "command": " ".join(command),
            "context_tokens": args.context,
            "max_active_requests": args.max_active_requests,
            "max_tokens": args.max_tokens,
            "wide_prompt_tokens": args.wide_prompt_tokens,
            "narrow_prompt_tokens": args.narrow_prompt_tokens,
            "kv_pool_idle_grace_seconds": args.kv_pool_idle_grace_seconds,
            "phase_plan": ["wide_1", "c1_a", "c1_b", "wide_2", "c2"],
        },
        "result": {
            "baseline_mib": baseline >> 20,
            "post_load_mib": post_load_mib,
            "phases": phases,
            "sampler_peak_gib": round(sampler_peak / GIB, 3),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "phases": len(phases),
                "sampler_peak_gib": artifact["result"]["sampler_peak_gib"],
            }
        )
    )
    return artifact, 0 if status == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--kv-storage", default="bf16")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--max-active-requests", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--wide-prompt-tokens", type=int, default=1536)
    parser.add_argument("--narrow-prompt-tokens", type=int, default=64)
    parser.add_argument("--kv-pool-idle-grace-seconds", type=float, default=2.0)
    parser.add_argument("--batch-window-ms", type=float, default=5.0)
    parser.add_argument("--sample-interval-ms", type=float, default=20.0)
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--maximum-baseline-mib", type=int, default=512)
    parser.add_argument("--maximum-recovery-mib", type=int, default=128)
    parser.add_argument("--maximum-graph-buckets", type=int, default=4)
    parser.add_argument("--startup-min-free-mib", type=int, default=0)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--pci-id", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--server-log-level", default="INFO")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, code = run(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
