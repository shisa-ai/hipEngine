#!/usr/bin/env python3
"""Probe the single-request context ceiling of one GGUF model on one GPU.

A capacity claim is the largest declared context that **starts a server from
cold and completes one request**, not the largest that allocates. This probe
enforces that definition: it launches a dedicated single-request server pinned
to one device, samples whole-card VRAM from ``rocm-smi`` while it runs, issues
one completion, and records the outcome with provenance.

Whole-card sampling is deliberate. Tracked allocator high-water understates
what the card must supply by roughly 0.2-0.9 GiB, which is enough to turn a
"does not fit" into a "fits" if used for a capacity decision.

Device selection is explicit because several hipEngine benchmark harnesses
override ``HIP_VISIBLE_DEVICES``; the resolved device is recorded in the
artifact so a run that landed on the wrong card is visible rather than silent.

Diagnostic harness: it changes no route and makes no throughput claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_VRAM_USED_RE = re.compile(
    r"GPU\[(?P<index>\d+)\][^\n]*VRAM Total Used Memory \(B\)\s*:\s*(?P<used>\d+)"
)
_VRAM_TOTAL_RE = re.compile(
    r"GPU\[(?P<index>\d+)\][^\n]*VRAM Total Memory \(B\)\s*:\s*(?P<total>\d+)"
)


def parse_vram(text: str, gpu: int) -> tuple[int | None, int | None]:
    """Return ``(used_bytes, total_bytes)`` for ``gpu`` from rocm-smi output."""

    used = total = None
    for match in _VRAM_USED_RE.finditer(text):
        if int(match.group("index")) == gpu:
            used = int(match.group("used"))
    for match in _VRAM_TOTAL_RE.finditer(text):
        if int(match.group("index")) == gpu:
            total = int(match.group("total"))
    return used, total


def sample_vram(gpu: int, rocm_smi: str) -> tuple[int | None, int | None]:
    try:
        out = subprocess.run(
            [rocm_smi, "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return parse_vram(out, gpu)


def classify(server_alive: bool, http_ok: bool, body: str) -> tuple[str, str]:
    """Return ``(status, reason)`` for one probe point."""

    if not server_alive:
        return "server_died", "server exited before serving a request"
    if http_ok:
        return "ok", "server started and completed one request"
    lowered = body.lower()
    if "out of memory" in lowered or "hiperror" in lowered:
        return "oom", "request failed with an out-of-memory error"
    return "request_failed", body.strip()[:400] or "request failed with no body"


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument(
        "--kv-storage",
        default="bf16",
        help="server --kv-storage value; record it, since BF16 and INT8 may be "
        "indistinguishable by footprint",
    )
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--max-active-requests", default=1, type=int)
    parser.add_argument("--max-tokens", default=16, type=int)
    parser.add_argument("--port", default=8077, type=int)
    parser.add_argument("--ready-timeout", default=1200, type=int)
    parser.add_argument("--request-timeout", default=900, type=int)
    parser.add_argument("--sample-interval", default=5.0, type=float)
    parser.add_argument("--rocm-smi", default=shutil.which("rocm-smi") or "rocm-smi")
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)

    env = dict(os.environ)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")

    _, total_bytes = sample_vram(args.gpu, args.rocm_smi)
    log_path = args.server_log or args.json.with_suffix(".server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "hipengine.server",
        "--model", str(args.model),
        "--backend", args.backend,
        "--quant", args.quant,
        "--served-model-name", "ceiling-probe",
        "--kv-storage", args.kv_storage,
        "--max-context-tokens", str(args.context),
        "--max-active-requests", str(args.max_active_requests),
        "--port", str(args.port),
    ]
    peak = 0
    started = time.monotonic()
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=log)
        try:
            ready = False
            while time.monotonic() - started < args.ready_timeout:
                time.sleep(args.sample_interval)
                used, _ = sample_vram(args.gpu, args.rocm_smi)
                if used:
                    peak = max(peak, used)
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/v1/models", timeout=5
                    ):
                        ready = True
                        break
                except (urllib.error.URLError, OSError):
                    continue

            http_ok = False
            body = ""
            if ready and proc.poll() is None:
                payload = json.dumps({
                    "model": "ceiling-probe",
                    "prompt": "a " * max(1, args.context // 2),
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                }).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/v1/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(
                        req, timeout=args.request_timeout
                    ) as resp:
                        body = resp.read().decode()
                        http_ok = '"text"' in body
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode(errors="replace")
                except (urllib.error.URLError, OSError) as exc:
                    body = str(exc)
                used, _ = sample_vram(args.gpu, args.rocm_smi)
                if used:
                    peak = max(peak, used)
            elif not ready:
                body = "server never became ready"

            status, reason = classify(proc.poll() is None, http_ok, body)
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=60)

    log_text = log_path.read_text(errors="replace")
    warmup_oom = "out of memory" in log_text.lower() and "WARMUP" in log_text
    if status != "ok" and warmup_oom:
        status, reason = "oom_during_warmup", (
            "HIP out of memory raised during eager server warmup, before any "
            "request was issued; the boundary is a startup reservation limit at "
            "the declared context, not a live-token limit"
        )

    artifact = {
        "schema": 1,
        "kind": "gguf_context_ceiling_probe_point",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "performance_claim": False,
        "diagnostic": True,
        "definition": (
            "A point passes only if the server starts from cold and completes one "
            "request at the declared context."
        ),
        "source": {"commit": _git(["rev-parse", "HEAD"]),
                   "tracked_clean": _git(["status", "--porcelain"]) == ""},
        "model": {"path": str(args.model), "quant": args.quant,
                  "size_bytes": args.model.stat().st_size if args.model.exists() else None},
        "device": {"gpu_index": args.gpu, "backend": args.backend,
                   "vram_total_bytes": total_bytes,
                   "hip_visible_devices": env["HIP_VISIBLE_DEVICES"]},
        "config": {"kv_storage": args.kv_storage,
                   "max_context_tokens": args.context,
                   "max_active_requests": args.max_active_requests,
                   "max_tokens": args.max_tokens},
        "result": {"status": status, "reason": reason,
                   "whole_card_peak_bytes": peak or None,
                   "whole_card_peak_gib": round(peak / 2**30, 3) if peak else None},
        "metric_note": (
            "Peak is whole-card used VRAM sampled from rocm-smi, not tracked "
            "allocator high-water, which understates the requirement."
        ),
        "server_log": str(log_path),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"context": args.context, "kv_storage": args.kv_storage,
                      "status": status,
                      "peak_gib": artifact["result"]["whole_card_peak_gib"]}))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
