#!/usr/bin/env python3
"""Probe the single-request context ceiling of one GGUF model on one GPU.

A capacity claim is the largest declared context that **starts a server from
cold and completes one validated request**, not the largest that allocates.
This probe enforces that definition with the repairs the 24 GB capacity
campaign requires (docs/QWEN38-27B-GFX1100-24GB-CAPACITY.md Packet 1):

- **Live tokens, not declared tokens.** The prompt is fitted to an explicit
  token target through the server tokenizer, and the response usage is
  validated against the counted prompt tokens and the requested output
  horizon. Configured context is never presented as live context.
- **Continuous whole-card sampling.** A sysfs sampler polls the resolved DRM
  card at a short interval through startup, warmup, prefill and decode, so
  samples cover the request instead of bracketing it. Peaks remain lower
  bounds if the interval misses a transient.
- **Verified device identity.** The card is resolved by PCI id (not an
  ordinal), its unique_id and total bytes are recorded, and the server's own
  readiness payload must report the same selected device.
- **Evidence-based failure classification.** OOM is declared only for
  out-of-memory evidence (``out of memory``, ``hipErrorOutOfMemory``, or
  ``HIP error 2``); other HIP errors, process exits, readiness and request
  timeouts, malformed responses, and early stops each get their own status.
- **Effective-state capture.** The readiness payload's KV capacity/pool,
  prefix-cache and graph state, plus the response's MTP summary, are recorded
  so requested KV and engaged MTP can be audited rather than assumed.
- **Clean-teardown ownership check.** After the request the probe reads the
  server ownership counters and reports a leak instead of silently passing.

This is a diagnostic harness: it changes no route and makes no throughput
claim. It is the direct route; the service-owner route remains
``scripts/qwen38_int8_server_context_soak.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hipengine.benchmark.agentic_live import final_ownership_from_server  # noqa: E402
from hipengine.util.amdgpu_vram import VramSampler, select_card  # noqa: E402

GIB = 1 << 30

# Out-of-memory evidence, matched narrowly. "HIP error 2" is
# hipErrorOutOfMemory; any other "HIP error N" is a different failure and must
# not be reported as OOM.
_OOM_PATTERNS = (
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"hipErrorOutOfMemory"),
    re.compile(r"\bHIP error 2\b"),
)
_HIP_ERROR_PATTERN = re.compile(r"\bHIP error (\d+)", re.IGNORECASE)


def parse_vram(text: str, gpu: int) -> tuple[int | None, int | None]:
    """Legacy rocm-smi parsing kept only for offline note reconstruction."""
    used = total = None
    used_re = re.compile(
        r"GPU\[(?P<index>\d+)\][^\n]*VRAM Total Used Memory \(B\)\s*:\s*(?P<used>\d+)"
    )
    total_re = re.compile(
        r"GPU\[(?P<index>\d+)\][^\n]*VRAM Total Memory \(B\)\s*:\s*(?P<total>\d+)"
    )
    for match in used_re.finditer(text):
        if int(match.group("index")) == gpu:
            used = int(match.group("used"))
    for match in total_re.finditer(text):
        if int(match.group("index")) == gpu:
            total = int(match.group("total"))
    return used, total


def read_card_unique_id(sysfs_path: Path) -> str | None:
    try:
        return (sysfs_path / "unique_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def port_in_use(host: str, port: int) -> bool:
    """Return True when something already accepts connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
        except OSError:
            return False
    return True


# Packed-execution workspace lease constants mirrored from
# hipengine/generation/qwen35_gguf.py: the shared global KV pool permanently
# leases max(1, N) slots x max(ceil(ctx/256), 1024/256) pages for packed
# verification. These pages are pinned for the process lifetime and are a
# persistent baseline, not a request leak.
_WORKSPACE_LEASE_MIN_SEQUENCE = 1024


def expected_workspace_lease_pages(max_active_requests: int, context_tokens: int) -> int:
    """Packed-verification workspace pages the server permanently leases.

    Mirrors the capacity-honest server lease: the serving cap bounds resident
    slots, so the lease follows ``max(1, max_active_requests)`` slots rather
    than the historical 8-slot floor (see
    ``configure_engine_loop`` in ``hipengine/generation/qwen35_gguf.py``).
    """

    pages_per_request = max(1, (int(context_tokens) + 255) // 256)
    per_slot = max(pages_per_request, _WORKSPACE_LEASE_MIN_SEQUENCE // 256)
    slots = max(1, int(max_active_requests))
    return slots * per_slot


def classify_failure(
    *,
    stage: str,
    log_text: str,
    body: str,
) -> tuple[str, str]:
    """Classify one failure from its stage and matching evidence.

    ``stage`` is one of ``startup`` or ``request``. OOM is declared only when
    the log or body carries out-of-memory evidence; other HIP errors keep
    their numeric code in the status. Unrecognized evidence keeps a trimmed
    excerpt as the reason.
    """

    haystack = f"{log_text}\n{body}"
    for pattern in _OOM_PATTERNS:
        if pattern.search(haystack):
            return f"oom_{stage}", (
                f"out-of-memory evidence during {stage}: "
                f"{_excerpt(haystack, pattern.pattern)}"
            )
    match = _HIP_ERROR_PATTERN.search(haystack)
    if match:
        return f"hip_error_{match.group(1)}_{stage}", (
            f"non-OOM HIP error {match.group(1)} during {stage}: "
            f"{_excerpt(haystack, match.group(0))}"
        )
    if stage == "startup":
        return "server_died_startup", _excerpt(log_text or body, "")
    return "request_failed", _excerpt(body or log_text, "")


def _excerpt(text: str, marker: str, limit: int = 300) -> str:
    cleaned = " ".join(text.split())
    if marker:
        position = cleaned.lower().find(marker.lower())
        if position >= 0:
            cleaned = cleaned[max(0, position - 80) : position + 200]
    return cleaned[:limit]


def validate_completion(
    payload: Mapping[str, Any],
    *,
    expected_prompt_tokens: int,
    expected_completion_tokens: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one completion response against exact accounting.

    Returns ``(info, error)``. ``info`` carries the authoritative generated
    token IDs, finish reason, usage and MTP summary when validation passes.
    """

    choices = payload.get("choices")
    usage = payload.get("usage")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], Mapping)
        or not isinstance(usage, Mapping)
    ):
        return None, "response is missing a single choice or usage object"
    accounting = payload.get("hipengine")
    accounting = accounting if isinstance(accounting, Mapping) else {}
    rows = (accounting.get("token_accounting") or {}).get(
        "choice_generated_token_ids"
    )
    if not (
        isinstance(rows, list)
        and len(rows) == 1
        and all(isinstance(token, int) and not isinstance(token, bool) for token in rows[0])
    ):
        return None, "response is missing authoritative choice_generated_token_ids"
    generated = [int(token) for token in rows[0]]
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens != int(expected_prompt_tokens):
        return None, (
            f"usage.prompt_tokens {prompt_tokens} != counted prompt tokens "
            f"{expected_prompt_tokens}"
        )
    if completion_tokens != len(generated):
        return None, (
            f"usage.completion_tokens {completion_tokens} != generated ids {len(generated)}"
        )
    if len(generated) != int(expected_completion_tokens):
        return None, (
            f"generated {len(generated)} tokens != requested horizon "
            f"{expected_completion_tokens}"
        )
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason != "length":
        return None, (
            f"finish_reason {finish_reason!r} is not 'length'; the intended "
            "output horizon was not completed"
        )
    mtp_summary = accounting.get("speculative_mtp")
    return {
        "generated_token_ids": generated,
        "generated_token_ids_sha256": token_ids_sha256(generated),
        "generated_text": str(choice.get("text") or ""),
        "finish_reason": finish_reason,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "speculative_mtp": mtp_summary,
        "exact_accounting": True,
    }, None


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([int(token) for token in token_ids], separators=(",", ":")).encode())
    return digest.hexdigest()


def fit_prompt_tokens(
    *,
    unit_token_ids: Sequence[int],
    suffix: str,
    target_tokens: int,
    count_tokens: Callable[[str], int],
    detokenize: Callable[[Sequence[int]], str],
    tolerance: int = 2,
    max_iterations: int = 8,
) -> tuple[str, int]:
    """Fit token-level filler plus ``suffix`` to a token target.

    The filler is built from repeated ``unit_token_ids`` with a token-exact
    pad, so the fit lands within ``tolerance`` tokens of the target even when
    the tokenizer is not additive across copy boundaries. Returns
    ``(prompt_text, counted_tokens)``; raises ``ValueError`` when the target
    is unreachable or the fit does not settle.
    """

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    unit = [int(token) for token in unit_token_ids]
    if not unit:
        raise ValueError("unit_token_ids must not be empty")
    suffix_count = count_tokens(suffix)
    if suffix_count >= target_tokens:
        raise ValueError(
            f"suffix alone tokenizes to {suffix_count}, above target {target_tokens}"
        )
    copies = max(0, (target_tokens - suffix_count) // len(unit))
    pad = target_tokens - suffix_count - copies * len(unit)
    drift = 0
    for _ in range(max_iterations):
        filler_ids = unit * copies + unit[: max(0, pad)]
        prompt = detokenize(filler_ids) + suffix
        counted = count_tokens(prompt)
        deficit = target_tokens - counted
        if 0 <= deficit <= tolerance:
            return prompt, counted
        # Retokenizing the detokenized filler can merge or split a token at
        # the seam. Measure that drift and recompute the filler size so the
        # next iteration lands on the target.
        filler_count = copies * len(unit) + max(0, pad)
        drift = counted - filler_count - suffix_count
        needed = target_tokens - suffix_count - drift
        copies = max(0, needed // len(unit))
        pad = needed - copies * len(unit)
    raise ValueError(
        f"prompt fit did not converge: {counted} vs target {target_tokens}"
    )


@dataclass(frozen=True)
class StageMarks:
    """Monotonic-clock marks for the stages a probe point moves through."""

    server_start: float
    ready: float
    request_start: float
    request_end: float


class ProbeClient:
    """Small blocking HTTP client for the probe's control-plane calls."""

    def __init__(self, host: str, port: int, *, timeout_s: float) -> None:
        self.host = host
        self.port = port
        self.timeout_s = float(timeout_s)

    def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> tuple[int, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        try:
            connection.request(
                method,
                "/" + path.lstrip("/"),
                body=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            response = connection.getresponse()
            raw = response.read()
            status = int(response.status)
        finally:
            connection.close()
        try:
            decoded: Any = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = {"raw": raw.decode("utf-8", errors="replace")}
        return status, decoded

    def ready(self) -> dict[str, Any]:
        status, payload = self._request("GET", "ready")
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError(f"/ready returned HTTP {status}")
        return payload

    def capabilities(self) -> dict[str, Any]:
        status, payload = self._request("GET", "v1/hipengine/capabilities")
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError(f"/v1/hipengine/capabilities returned HTTP {status}")
        return payload

    def tokenize(self, text: str) -> list[int]:
        status, payload = self._request("POST", "v1/hipengine/tokenize", {"text": text})
        tokens = payload.get("token_ids") if isinstance(payload, Mapping) else None
        if status != 200 or not isinstance(tokens, list):
            raise RuntimeError(f"tokenize returned HTTP {status}")
        return [int(token) for token in tokens]

    def detokenize(self, token_ids: Sequence[int]) -> str:
        status, payload = self._request(
            "POST",
            "v1/hipengine/detokenize",
            {"token_ids": [int(token) for token in token_ids], "skip_special": False},
        )
        text = payload.get("text") if isinstance(payload, Mapping) else None
        if status != 200 or not isinstance(text, str):
            raise RuntimeError(f"detokenize returned HTTP {status}")
        return text

    def completion(
        self,
        *,
        served_model_name: str,
        prompt: str,
        max_tokens: int,
        timeout_s: float,
    ) -> tuple[int, Any]:
        self.timeout_s = float(timeout_s)
        return self._request(
            "POST",
            "v1/completions",
            {
                "model": served_model_name,
                "prompt": prompt,
                "max_tokens": int(max_tokens),
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "stream": False,
                "seed": 0,
            },
        )


# Fixed natural filler: enough variety that tokenization is realistic, small
# enough that the fit loop converges quickly. It is probe plumbing, not a task
# prompt; capacity claims based on it are labeled as repeated-corpus probes.
FILLER_CORPUS = (
    "The harbourmaster records each arriving vessel in a ledger: name, "
    "tonnage, berth, and the tide at docking. Clerks copy the entries into a "
    "second book for the archive every evening, and the two volumes are "
    "compared monthly for transcription errors. "
)
PROMPT_SUFFIX = "\n\nRepeat the word 'ledger' once as your entire answer."


def sampled_model_hash(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        for offset in sorted({0, max(0, size // 2 - (1 << 19)), max(0, size - (1 << 20))}):
            handle.seek(offset)
            digest.update(handle.read(1 << 20))
    return digest.hexdigest()


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _stage_peaks(
    samples: Sequence[tuple[float, int]], marks: StageMarks
) -> dict[str, int | None]:
    def peak_between(start: float, end: float) -> int | None:
        window = [value for t, value in samples if start <= t <= end]
        return max(window) if window else None

    return {
        "startup_peak_bytes": peak_between(0.0, marks.ready - marks.server_start),
        "request_peak_bytes": peak_between(
            marks.request_start - marks.server_start, marks.request_end - marks.server_start
        ),
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise ValueError(f"model does not exist: {model_path}")

    card = select_card(pci_id=args.pci_id)
    unique_id = read_card_unique_id(card.sysfs_path)
    sampler_interval_ms = max(1.0, args.sample_interval_ms)

    if port_in_use(args.host, args.port):
        artifact = _artifact(
            args,
            model_path=model_path,
            card=card,
            unique_id=unique_id,
            status="port_in_use",
            reason=f"something already listens on {args.host}:{args.port}",
            sampler=None,
            marks=None,
            readiness=None,
            capabilities=None,
            prompt=None,
            completion=None,
            ownership=None,
            server_exit=None,
        )
        return artifact, 1

    baseline = _read_used(card)
    if baseline > int(args.maximum_baseline_mib) << 20:
        artifact = _artifact(
            args,
            model_path=model_path,
            card=card,
            unique_id=unique_id,
            status="not_idle",
            reason=(
                f"baseline {baseline >> 20} MiB exceeds the "
                f"{args.maximum_baseline_mib} MiB idle gate"
            ),
            sampler=None,
            marks=None,
            readiness=None,
            capabilities=None,
            prompt=None,
            completion=None,
            ownership=None,
            server_exit=None,
        )
        return artifact, 1

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
        "ceiling-probe",
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
        args.speculative_mtp_serving,
        "--prefix-cache",
        args.prefix_cache,
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
    if args.kv_storage != "bf16":
        command.extend(
            [
                "--kv-scale-dtype",
                args.kv_scale_dtype,
                "--kv-scale-granularity",
                args.kv_scale_granularity,
            ]
        )

    log_path = args.server_log or args.json.with_suffix(".server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    server_start = time.perf_counter()
    sampler = VramSampler(card, interval_ms=sampler_interval_ms, keep_samples=True)
    sampler.start()
    readiness: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None
    marks: StageMarks | None = None
    prompt_info: dict[str, Any] | None = None
    completion_info: dict[str, Any] | None = None
    ownership: dict[str, Any] | None = None
    server_exit: int | None = None
    status = "ok"
    reason = "server started and completed one validated request"

    proc = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        stdout=log_path.open("a"),
        stderr=subprocess.STDOUT,
    )
    client = ProbeClient(args.host, args.port, timeout_s=10.0)
    try:
        # Startup: wait for readiness, sampling the whole time. A warmup OOM
        # leaves the process alive returning 503, so classify from matching
        # log evidence once readiness has been failing for a grace period.
        deadline = time.monotonic() + float(args.ready_timeout)
        last_error: Exception | None = None
        not_ready_since: float | None = None
        ready_ok = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                server_exit = int(proc.returncode)
                break
            body_text = ""
            try:
                payload = client.ready()
                if payload.get("ready") is True:
                    readiness = payload
                    ready_ok = True
                    break
                body_text = json.dumps(payload)
            except Exception as exc:  # noqa: BLE001 - startup polls until timeout
                last_error = exc
                body_text = str(exc)
            if not_ready_since is None:
                not_ready_since = time.monotonic()
            elif time.monotonic() - not_ready_since > 10.0:
                log_text = log_path.read_text(errors="replace")[-20000:]
                candidate, candidate_reason = classify_failure(
                    stage="startup", log_text=log_text, body=body_text
                )
                if candidate.startswith(("oom_", "hip_error")):
                    if readiness is None and body_text.startswith("{"):
                        try:
                            readiness = json.loads(body_text)
                        except json.JSONDecodeError:
                            readiness = None
                    status, reason = candidate, candidate_reason
                    break
            time.sleep(0.25)
        ready_at = time.perf_counter()

        if not ready_ok:
            if status == "ok":
                log_text = log_path.read_text(errors="replace")
                if proc.poll() is not None:
                    server_exit = int(proc.returncode)
                    status, reason = classify_failure(
                        stage="startup", log_text=log_text, body=""
                    )
                else:
                    status, reason = "ready_timeout", (
                        f"server never became ready within {args.ready_timeout}s: {last_error}"
                    )
        else:
            marks = StageMarks(
                server_start=server_start,
                ready=ready_at,
                request_start=ready_at,
                request_end=ready_at,
            )
            # Device identity: the server must report the requested ordinal.
            device = readiness.get("device")
            selected = (
                device.get("selected_visible_device")
                if isinstance(device, Mapping)
                else None
            )
            if selected is not None and int(selected) != int(args.gpu_index):
                status, reason = "device_mismatch", (
                    f"server reports device {selected!r}, requested {args.gpu_index}"
                )
            elif _kv_fell_back(readiness, args.kv_storage):
                status, reason = "kv_fallback", (
                    f"requested kv_storage {args.kv_storage} was not engaged; "
                    "effective route fell back (see effective_state.model.kv_capability)"
                )
            else:
                try:
                    capabilities = client.capabilities()
                except Exception as exc:  # noqa: BLE001
                    capabilities = {"error": f"{type(exc).__name__}: {exc}"}

                # Fit the prompt to the live token target through the server
                # tokenizer, then validate the response against it.
                target_prompt_tokens = int(
                    args.prompt_tokens
                    if args.prompt_tokens is not None
                    else args.context - args.max_tokens
                )
                try:
                    prompt_text, counted = fit_prompt_tokens(
                        unit_token_ids=client.tokenize(FILLER_CORPUS),
                        suffix=PROMPT_SUFFIX,
                        target_tokens=target_prompt_tokens,
                        count_tokens=lambda text: len(client.tokenize(text)),
                        detokenize=client.detokenize,
                    )
                except (ValueError, RuntimeError) as exc:
                    status, reason = "prompt_fit_failed", str(exc)
                    prompt_text, counted = None, None
                if prompt_text is not None:
                    prompt_info = {
                        "target_prompt_tokens": target_prompt_tokens,
                        "counted_prompt_tokens": counted,
                        "filler_corpus_chars": len(FILLER_CORPUS),
                        "suffix_chars": len(PROMPT_SUFFIX),
                    }
                    request_started = time.perf_counter()
                    try:
                        http_status, payload = client.completion(
                            served_model_name="ceiling-probe",
                            prompt=prompt_text,
                            max_tokens=args.max_tokens,
                            timeout_s=float(args.request_timeout),
                        )
                        request_ended = time.perf_counter()
                        marks = StageMarks(
                            server_start=server_start,
                            ready=ready_at,
                            request_start=request_started,
                            request_end=request_ended,
                        )
                    except Exception as exc:  # noqa: BLE001
                        request_ended = time.perf_counter()
                        marks = StageMarks(
                            server_start=server_start,
                            ready=ready_at,
                            request_start=request_started,
                            request_end=request_ended,
                        )
                        status, reason = "request_timeout", (
                            f"completion raised {type(exc).__name__}: {exc}"
                        )
                        payload = None
                    if payload is not None:
                        if http_status != 200:
                            body = payload.get("raw") if isinstance(payload, Mapping) else str(payload)
                            log_text = log_path.read_text(errors="replace")
                            status, reason = classify_failure(
                                stage="request", log_text=log_text, body=str(body)
                            )
                        else:
                            info, error = validate_completion(
                                payload,
                                expected_prompt_tokens=counted,
                                expected_completion_tokens=args.max_tokens,
                            )
                            if info is None:
                                status, reason = "invalid_response", str(error)
                                if error and "finish_reason" in str(error):
                                    status = "early_stop"
                            else:
                                completion_info = info
        # Post-request ownership and clean teardown, only meaningful when a
        # request actually completed. The pool may reclaim retained pages only
        # after its idle grace, so poll for the drain instead of checking once.
        if completion_info is not None:
            lease_pages = expected_workspace_lease_pages(
                args.max_active_requests, args.context
            )
            deadline = time.monotonic() + float(args.ownership_timeout_s)
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                try:
                    ready_after = client.ready()
                    _status, sessions = client._request(
                        "GET", "v1/hipengine/sessions"
                    )
                    ownership = final_ownership_from_server(
                        ready_after,
                        sessions if isinstance(sessions, Mapping) else {},
                        cache_mode=args.prefix_cache,
                        persistent_refcounted_pages=lease_pages,
                        persistent_pinned_pages=lease_pages,
                    )
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    time.sleep(0.5)
            if last_error is not None:
                status, reason = "ownership_unknown", f"{type(last_error).__name__}: {last_error}"
            else:
                nonzero = {k: v for k, v in (ownership or {}).items() if v}
                if nonzero:
                    status, reason = "ownership_leak", f"nonzero ownership: {nonzero}"
            if ownership is not None:
                ownership = {
                    **ownership,
                    "expected_workspace_lease_pages": lease_pages,
                }
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        server_exit = int(proc.returncode)
    finally:
        sampler.stop()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)

    artifact = _artifact(
        args,
        model_path=model_path,
        card=card,
        unique_id=unique_id,
        status=status,
        reason=reason,
        sampler=sampler,
        marks=marks,
        readiness=readiness,
        capabilities=capabilities,
        prompt=prompt_info,
        completion=completion_info,
        ownership=ownership,
        server_exit=server_exit,
    )
    return artifact, 0 if status == "ok" else 1


def _kv_fell_back(readiness: Mapping[str, Any], requested: str) -> bool:
    """True when the server engaged a different KV storage than requested."""

    model = readiness.get("model")
    capability = model.get("kv_capability") if isinstance(model, Mapping) else None
    if isinstance(capability, Mapping):
        effective = capability.get("effective_kv_storage")
        if isinstance(effective, str) and effective and effective != requested:
            return True
        action = capability.get("runtime_action")
        if isinstance(action, str) and action.startswith("fallback"):
            return True
    kv_capacity = readiness.get("kv_capacity")
    if isinstance(kv_capacity, Mapping):
        storage = kv_capacity.get("storage")
        if isinstance(storage, str) and storage and storage != requested:
            return True
    return False


def _read_used(card: Any) -> int:
    return int(card.vram_used_path.read_text().strip())


def _artifact(
    args: argparse.Namespace,
    *,
    model_path: Path,
    card: Any,
    unique_id: str | None,
    status: str,
    reason: str,
    sampler: VramSampler | None,
    marks: StageMarks | None,
    readiness: Mapping[str, Any] | None,
    capabilities: Mapping[str, Any] | None,
    prompt: Mapping[str, Any] | None,
    completion: Mapping[str, Any] | None,
    ownership: Mapping[str, Any] | None,
    server_exit: int | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "reason": reason,
        "server_exit_code": server_exit,
    }
    if sampler is not None:
        summary = sampler.result()
        result["whole_card"] = summary.to_dict(include_samples=True)
        if marks is not None:
            result["stage_peaks_bytes"] = _stage_peaks(sampler.result().samples, marks)
    kv_capacity = readiness.get("kv_capacity") if readiness else None
    result["effective_state"] = {
        "kv_capacity": kv_capacity,
        "prefix_cache": readiness.get("prefix_cache") if readiness else None,
        "graph_cache": readiness.get("graph_cache") if readiness else None,
        "context": readiness.get("context") if readiness else None,
        "queue": readiness.get("queue") if readiness else None,
        "startup": readiness.get("startup") if readiness else None,
        "model": readiness.get("model") if readiness else None,
        "speculative_mtp_capability": (capabilities or {}).get("speculative_mtp"),
    }
    artifact = {
        "schema": 2,
        "kind": "gguf_context_ceiling_probe_point",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "performance_claim": False,
        "diagnostic": True,
        "definition": (
            "A point passes only if the server starts from cold on the verified "
            "card and completes one request whose prompt and output tokens match "
            "exact server accounting at the intended horizon, with clean "
            "post-request ownership."
        ),
        "source": {
            "commit": _git(["rev-parse", "HEAD"]),
            "tracked_clean": _git(["status", "--porcelain"]) == "",
            "command": " ".join(_server_command_repr(args, model_path)),
            "environment": {
                "HIP_VISIBLE_DEVICES": str(args.gpu_index),
                "GPU_MAX_HW_QUEUES": "1",
                "HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS": str(
                    args.kv_pool_idle_grace_seconds
                ),
                "ROCR_VISIBLE_DEVICES": None,
            },
        },
        "model": {
            "path": str(model_path),
            "quant": args.quant,
            "size_bytes": model_path.stat().st_size,
            "sampled_fingerprint_sha256": sampled_model_hash(model_path),
        },
        "device": {
            "gpu_index_requested": args.gpu_index,
            "pci_id": card.pci_id,
            "drm_card": card.card_name,
            "unique_id": unique_id,
            "vram_total_bytes": card.vram_total_bytes,
            "hip_visible_devices": str(args.gpu_index),
            "selection_note": (
                "card resolved by pci_id through hipengine.util.amdgpu_vram; "
                "server-reported ordinal cross-checked against the request"
            ),
        },
        "config": {
            "kv_storage": args.kv_storage,
            "kv_scale_dtype": args.kv_scale_dtype if args.kv_storage != "bf16" else None,
            "kv_scale_granularity": (
                args.kv_scale_granularity if args.kv_storage != "bf16" else None
            ),
            "max_context_tokens": args.context,
            "max_active_requests": args.max_active_requests,
            "max_tokens": args.max_tokens,
            "prompt_tokens": args.prompt_tokens,
            "speculative_mtp_serving": args.speculative_mtp_serving,
            "prefix_cache": args.prefix_cache,
            "startup_min_free_mib": args.startup_min_free_mib,
            "batch_window_ms": args.batch_window_ms,
            "sample_interval_ms": max(1.0, args.sample_interval_ms),
        },
        "result": result,
        "prompt": prompt,
        "completion": completion,
        "ownership": ownership,
        "metric_note": (
            "Whole-card used VRAM sampled from sysfs mem_info_vram_used during "
            "startup and the request. Samples are lower bounds on transients "
            "shorter than the sampling interval."
        ),
        "server_log": str(
            args.server_log or args.json.with_suffix(".server.log")
        ),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "context": args.context,
                "kv_storage": args.kv_storage,
                "status": status,
                "peak_gib": artifact["result"].get("whole_card", {}).get("peak_gib"),
            }
        )
    )
    return artifact


def _server_command_repr(args: argparse.Namespace, model_path: Path) -> list[str]:
    return [
        "python", "-m", "hipengine.server",
        "--model", str(model_path),
        "--backend", args.backend,
        "--quant", args.quant,
        "--kv-storage", args.kv_storage,
        "--max-context-tokens", str(args.context),
        "--max-active-requests", str(args.max_active_requests),
        "--speculative-mtp-serving", args.speculative_mtp_serving,
        "--prefix-cache", args.prefix_cache,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument(
        "--kv-storage",
        default="bf16",
        help="server --kv-storage value; the effective route is validated from /ready",
    )
    parser.add_argument(
        "--kv-scale-dtype",
        default="fp32",
        help="scale dtype for non-BF16 KV; fp32/per_token_head is the qualified contract",
    )
    parser.add_argument(
        "--kv-scale-granularity",
        default="per_token_head",
        help="scale granularity for non-BF16 KV",
    )
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="live prompt token target; default context - max_tokens",
    )
    parser.add_argument("--max-tokens", default=16, type=int)
    parser.add_argument("--gpu-index", default=1, type=int,
                        help="ordinal passed via HIP_VISIBLE_DEVICES; identity is verified")
    parser.add_argument("--pci-id", default="0000:10:00.0",
                        help="PCI id of the card to sample (default: the RX 7900 XTX)")
    parser.add_argument("--max-active-requests", default=1, type=int)
    parser.add_argument("--port", default=8077, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--batch-window-ms", default=5.0, type=float)
    parser.add_argument("--speculative-mtp-serving", default="off")
    parser.add_argument("--prefix-cache", default="off")
    parser.add_argument("--startup-min-free-mib", default=0, type=int)
    parser.add_argument(
        "--kv-pool-idle-grace-seconds",
        default=0,
        type=int,
        help=(
            "HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS for the server; 0 makes the "
            "shared pool release retained pages as soon as the card idles, so "
            "the teardown ownership gate can require exact reclaim"
        ),
    )
    parser.add_argument(
        "--ready-timeout", default=1200.0, type=float
    )
    parser.add_argument("--request-timeout", default=900.0, type=float)
    parser.add_argument(
        "--ownership-timeout-s",
        default=90.0,
        type=float,
        help="how long to wait for post-request ownership to drain",
    )
    parser.add_argument(
        "--sample-interval-ms",
        default=20.0,
        type=float,
        help="sysfs VRAM sampling interval in milliseconds",
    )
    parser.add_argument("--maximum-baseline-mib", default=128, type=int)
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--server-log-level", default="info")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, code = run(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
