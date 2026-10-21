#!/usr/bin/env python3
"""Server-faithful allocation probe: the real server binary, real pool, real route.

`scripts/gguf_capacity_probe.py` proves the *direct session's* memory envelope.
It cannot answer the server's: the server binds requests to the shared pool with
slot views, leases a packed-execution workspace, and runs ``int8_direct``
requests through the packed slot-local prefill entry. This probe drives ONE
request through the actual server process and scrapes the owner-deduplicated
observability gauges at high cadence during the request, so per-domain peaks are
captured *while live*, not inferred from post-request counters.

Per the two-tier capacity protocol (benchmarks/HARNESSES.md) this is a Tier-1
allocation probe: it never searches a bound with full prompts. The default
prompt is deliberately multi-slab (2,048 rows = two 1,024-row slabs on the 27B
dense geometry) so the executor's real working set is allocated; --prompt-rows
crosses a specific slab/page threshold.

Axes: --kv-storage bf16|int8_per_token_head|auto, --speculative-mtp-serving
off|opt_in|auto|enabled, --dms-metadata/--dms-metadata-path, --max-context-tokens
N|auto, --max-active-requests N|auto, --prefix-cache default|off|radix.

``auto``/``default`` reach the server as an omission, never as a literal. The
probe refuses to run when an inherited environment variable would silently pin
an axis it was told to leave automatic. It injects no environment by default:
the caller's environment is the measurement's environment, and every override it
does make is recorded with its effect. Which axis combinations are executable is
the server's decision, not this script's; a refusal arrives as a server error and
is recorded in the artifact.
Recorded per run: policy, exact command, environment overrides and inherited
values; host and git identity; the raw ``/ready`` and
``/v1/hipengine/capabilities`` documents before and after the request; the
request's raw ``hipengine`` metadata, usage and generated text; per-domain gauge
peaks (pool, workspace, lease, prefill oracle, hidden/bulk, persistent KV - never
summed). The artifact is written on every exit path, and a non-passing run
carries the server log tail.

Examples:

    # Historical INT8 configuration, pinned context, one slot.
    python3 scripts/gguf_server_allocation_probe.py \
        --legacy-int8-diagnostic-overrides \
        --max-context-tokens 16384 --prompt-rows 2048 --json /tmp/probe.json

    # Production BF16 configuration with automatic context and capacity.
    python3 scripts/gguf_server_allocation_probe.py --kv-storage bf16 \
        --speculative-mtp-serving auto --prefix-cache default \
        --max-context-tokens auto --max-active-requests auto --json /tmp/bf16.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_FIXTURE = REPO_ROOT / "benchmarks" / "parity" / "r0r4-c1-fixture.json"
_PROBE_VERSION = 3

# Gauge -> (artifact field, scale, source). "peak" reads the polling high-water,
# "final" prefers the settled post-request sample.
_DOMAIN_GAUGES = (
    ("pool_high_water", "hipengine_kv_pool_high_water_observed_bytes", "gib", "peak"),
    ("pool_pinned_pages", "hipengine_kv_pool_pinned_pages", "count", "peak"),
    ("workspace_owner_deduped", "hipengine_resident_packed_workspace_current_bytes", "gib", "peak"),
    ("workspace_leased_pool", "hipengine_resident_packed_workspace_leased_pool_bytes", "gib", "peak"),
    ("prefill_oracle_live", "hipengine_resident_prefill_oracle_owner_bytes", "gib", "peak"),
    ("prefill_oracle_observed_peak", "hipengine_resident_prefill_oracle_observed_peak_bytes", "gib", "final"),
    ("prefill_oracle_observed_peak_owners", "hipengine_resident_prefill_oracle_observed_peak_owners", "count", "final"),
    ("prefill_oracle_owners", "hipengine_resident_prefill_oracle_owners", "count", "peak"),
    ("hidden_bulk_owners", "hipengine_resident_prefill_hidden_owner_bytes", "gib", "peak"),
    ("request_owned_kv_after", "hipengine_resident_kv_total_bytes", "gib", "final"),
)
_TRACKED_GAUGES = ("hipengine_kv_pool_current_bytes",) + tuple(
    gauge for _, gauge, _, _ in _DOMAIN_GAUGES
)

# The server reads these axes from the environment too, so an inherited value
# can pin an axis the probe was told to leave automatic.
_CONTEXT_PIN_ENV = "HIPENGINE_MAX_CONTEXT_TOKENS"
_CAPACITY_PIN_ENV = "HIPENGINE_MAX_ACTIVE_REQUESTS"
_PREFIX_PIN_ENV = "HIPENGINE_PREFIX_CACHE"
_AUTO_CONTEXT_ENV = "HIPENGINE_GGUF_AUTO_CONTEXT"
_INT8_ALLOW_UNVERIFIED_LONG_ENV = "HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG"
_INT8_BF16_FULL_LAYERS_ENV = "HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS"

# Inherited values recorded with every artifact: an unrecorded queue policy,
# scratch limit, packed-route switch or KV policy is an unattributable result.
_CAPTURED_ENV_KEYS = (
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_MAX_HW_QUEUES",
    "HSA_SCRATCH_SINGLE_LIMIT", "HIPENGINE_HIP_ARCH", _AUTO_CONTEXT_ENV,
    "HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB", _CONTEXT_PIN_ENV, _CAPACITY_PIN_ENV,
    _PREFIX_PIN_ENV, "HIPENGINE_SPECULATIVE_MTP_SERVING", "HIPENGINE_KV_STORAGE",
    "HIPENGINE_KV_SCALE_DTYPE", "HIPENGINE_KV_SCALE_GRANULARITY",
    "HIPENGINE_STARTUP_MIN_FREE_MIB", _INT8_ALLOW_UNVERIFIED_LONG_ENV,
    _INT8_BF16_FULL_LAYERS_ENV, "HIPENGINE_GGUF_AR_PACKED_DECODE",
    "HIPENGINE_GGUF_PACKED_LAYER_OUTER", "HIPENGINE_GGUF_DECODE_GRAPH",
)

_FALSEY_ENV_VALUES = {"0", "false", "no", "off"}


class PolicyError(ValueError):
    """A request the probe cannot honour as asked."""


def _parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and not line.startswith("#"):
            try:
                values[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return values


def _scrape(port: int, timeout: float = 5.0) -> dict[str, float] | None:
    try:
        url = f"http://127.0.0.1:{port}/metrics"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return _parse_metrics(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError):
        return None


def _http_json(url: str, timeout: float = 30.0) -> tuple[int | None, Any, str | None]:
    """GET a JSON document, keeping the body even when the status is not 2xx.

    ``/ready`` answers 503 with the readiness document that names the failure, so
    the body is decoded from ``HTTPError`` rather than discarded.
    """

    status: int | None = None
    raw: bytes | None = None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status, raw = int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            raw = exc.read()
        except OSError:
            raw = None
    except (urllib.error.URLError, OSError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"[:300]
    if raw is None:
        return status, None, "empty_response_body"
    try:
        return status, json.loads(raw.decode("utf-8", errors="replace")), None
    except json.JSONDecodeError as exc:
        return status, None, f"JSONDecodeError: {exc}"[:300]


def document(status: int | None, payload: Any, error: str | None) -> dict[str, Any]:
    """One endpoint read: transport facts plus the whole raw document.

    Everything the server publishes about the resident pool, KV capacity
    estimate, device and admission state is evidence for an allocation question,
    and a hand-picked subset cannot be re-read later.
    """
    return {"http_status": status, "error": error, "payload": payload}


def readiness_accepts_requests(status: int | None, payload: Any) -> bool:
    """Whether a ``/ready`` answer means the server will serve the probe's request."""
    return status == 200 and isinstance(payload, Mapping) and payload.get("ready") is True


def _positive_int_or_auto(raw: str) -> int | None:
    text = str(raw).strip().lower()
    if text == "auto":
        return None
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer or 'auto', got {raw!r}"
        ) from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {raw!r}")
    return value

def resolve_policy(args: argparse.Namespace, base_env: Mapping[str, str]) -> dict[str, Any]:
    """Resolve the requested axes, refusing only what this probe cannot honour.

    Mutually exclusive arguments are a probe-level contradiction. The environment
    checks exist because the server reads the same axes from the environment: if
    the caller asked for an automatic axis while an inherited variable pins it,
    the run would silently measure a pinned configuration. Whether a *combination*
    of axes is executable is the server's decision, so its own refusal is what the
    artifact records.
    """

    if args.dms_metadata is not None and args.dms_metadata_path is not None:
        raise PolicyError(
            "pass one of --dms-metadata (compact DMS) or --dms-metadata-path "
            "(online eviction), not both"
        )

    def refuse_if_pinned(env_name: str, axis: str, automatic: bool) -> None:
        pinned = str(base_env.get(env_name) or "").strip()
        if automatic and pinned:
            raise PolicyError(
                f"{axis} requested but {env_name}={pinned} pins it in the environment; "
                "unset it or pass an explicit value"
            )

    context, capacity = args.max_context_tokens, args.max_active_requests
    refuse_if_pinned(_CONTEXT_PIN_ENV, "--max-context-tokens auto", context is None)
    refuse_if_pinned(_CAPACITY_PIN_ENV, "--max-active-requests auto", capacity is None)
    refuse_if_pinned(_PREFIX_PIN_ENV, "--prefix-cache default", args.prefix_cache == "default")
    auto_context = str(base_env.get(_AUTO_CONTEXT_ENV) or "").strip().lower()
    if context is None and auto_context in _FALSEY_ENV_VALUES:
        raise PolicyError(
            f"automatic context requested but {_AUTO_CONTEXT_ENV}={auto_context} "
            "disables automatic context sizing; unset it or pass --max-context-tokens N"
        )

    return {
        "kv_storage": str(args.kv_storage),
        "kv_scale_dtype": str(args.kv_scale_dtype),
        "kv_scale_granularity": str(args.kv_scale_granularity),
        "speculative_mtp_serving": str(args.speculative_mtp_serving),
        "prefix_cache": str(args.prefix_cache),
        "max_context_tokens": None if context is None else int(context),
        "automatic_context": context is None,
        "max_active_requests": None if capacity is None else int(capacity),
        "automatic_serving_capacity": capacity is None,
        "dms_metadata": None if args.dms_metadata is None else str(args.dms_metadata),
        "dms_metadata_path": (
            None if args.dms_metadata_path is None else str(args.dms_metadata_path)
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the live hipEngine server's resident allocation for one "
                    "configuration. Tier-1 capacity probe: it never searches a bound."
    )
    parser.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--kv-storage", choices=("auto", "bf16", "int8_per_token_head"),
                        default="int8_per_token_head",
                        help="Server KV storage (server default bf16; historical default int8)")
    parser.add_argument("--kv-scale-dtype", default="fp32", help="INT8 KV scale dtype")
    parser.add_argument("--kv-scale-granularity", default="per_token_head",
                        help="INT8 KV scale granularity")
    parser.add_argument("--speculative-mtp-serving", choices=("off", "opt_in", "auto", "enabled"),
                        default="off",
                        help="MTP serving policy (server default auto; historical default off)")
    parser.add_argument("--prefix-cache", choices=("default", "off", "radix"), default="off",
                        help="'default' omits the flag; 'off'/'radix' pin it (historical: off)")
    parser.add_argument("--max-context-tokens", type=_positive_int_or_auto, default="auto",
                        help="Resident context tokens, or 'auto' to omit the flag")
    parser.add_argument("--max-active-requests", type=_positive_int_or_auto, default="1",
                        help="Active request cap, or 'auto' to omit the flag")
    parser.add_argument("--dms-metadata", type=Path, default=None,
                        help="Compact DMS sidecar metadata; the server validates it")
    parser.add_argument("--dms-metadata-path", type=Path, default=None,
                        help="DMS online-eviction metadata")
    parser.add_argument("--dms-prefill-mode", choices=("dense_pool", "layer_outer"),
                        default="dense_pool", help="DMS prefill mode, passed with --dms-metadata")
    parser.add_argument("--legacy-int8-diagnostic-overrides", action="store_true",
                        help="Historical INT8 replay: export "
                             f"{_INT8_ALLOW_UNVERIFIED_LONG_ENV}=1 and "
                             f"{_INT8_BF16_FULL_LAYERS_ENV}=none for an int8_per_token_head "
                             "run. Both change the measured envelope, so production rows "
                             "leave this off")
    parser.add_argument("--prompt-rows", type=int, default=2048,
                        help="Request prompt rows (default 2048 = two 1,024-row slabs on "
                             "the 27B). Pass the configured context length to measure a "
                             "full-depth prompt")
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--port", type=int, default=18430)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--json", type=Path, default=None,
                        help="Artifact path, always written; omit for a timestamped temp file")
    return parser.parse_args(argv)


def build_server_command(args: argparse.Namespace, policy: Mapping[str, Any]) -> list[str]:
    """The exact server argv for one configuration.

    A flag is emitted only when the axis is pinned, so ``auto``/``default`` reach
    the server as an omission rather than as a literal it must interpret.
    ``--kv-storage`` is always passed, including ``auto``, so an inherited
    ``HIPENGINE_KV_STORAGE`` cannot decide the KV policy instead.
    """

    command = [
        sys.executable, "-m", "hipengine.server",
        "--model", str(args.model),
        "--backend", "hip_gfx1100",
        "--quant", "gguf_q4_k_m",
        "--kv-storage", str(policy["kv_storage"]),
    ]
    if policy["kv_storage"] == "int8_per_token_head":
        command += [
            "--kv-scale-dtype", str(policy["kv_scale_dtype"]),
            "--kv-scale-granularity", str(policy["kv_scale_granularity"]),
        ]
    if policy["max_context_tokens"] is not None:
        command += ["--max-context-tokens", str(int(policy["max_context_tokens"]))]
    if policy["max_active_requests"] is not None:
        command += ["--max-active-requests", str(int(policy["max_active_requests"]))]
    command += ["--speculative-mtp-serving", str(policy["speculative_mtp_serving"])]
    if policy["prefix_cache"] != "default":
        command += ["--prefix-cache", str(policy["prefix_cache"])]
    if policy["dms_metadata"] is not None:
        command += ["--dms-metadata", str(policy["dms_metadata"]),
                    "--dms-prefill-mode", str(args.dms_prefill_mode)]
    if policy["dms_metadata_path"] is not None:
        command += ["--dms-metadata-path", str(policy["dms_metadata_path"])]
    command += ["--metrics", "prometheus", "--port", str(int(args.port))]
    return command


def build_server_env(
    args: argparse.Namespace, policy: Mapping[str, Any], base_env: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Server environment, plus a record of everything the probe sets.

    Nothing is injected by default: the caller's environment is the measurement's
    environment. The one opt-in override pair is recorded with its effect.
    """

    env = dict(base_env)
    overrides: dict[str, dict[str, str]] = {}
    if args.legacy_int8_diagnostic_overrides:
        if policy["kv_storage"] == "int8_per_token_head":
            env[_INT8_ALLOW_UNVERIFIED_LONG_ENV] = "1"
            env[_INT8_BF16_FULL_LAYERS_ENV] = "none"
            overrides[_INT8_ALLOW_UNVERIFIED_LONG_ENV] = {
                "value": "1",
                "effect": "historical replay: no BF16 full-attention prefix mirror above "
                          "8,192 positions, and an INT8 configuration the capability "
                          "registry would otherwise refuse is admitted as diagnostic_override",
            }
            overrides[_INT8_BF16_FULL_LAYERS_ENV] = {
                "value": "none",
                "effect": "historical replay: uniform INT8 layout, which is what lets packed "
                          "decode keep its standard width instead of one row",
            }
        else:
            overrides["<legacy-int8-diagnostic-overrides>"] = {
                "value": "ignored",
                "effect": f"only applies to int8_per_token_head, not {policy['kv_storage']}",
            }
    inherited = {name: base_env[name] for name in _CAPTURED_ENV_KEYS if name in base_env}
    return env, {"inherited": inherited, "overrides": overrides}


def build_prompt(args: argparse.Namespace) -> tuple[list[int], str]:
    """Deterministic prompt ids and their digest.

    The 2,048-row case uses the parity manifest's exact fixture; every other shape
    uses the same seeded generator, never a fresh RNG.
    """

    if int(args.prompt_rows) == 2048 and PROMPT_FIXTURE.exists():
        fixture = json.loads(PROMPT_FIXTURE.read_text(encoding="utf-8"))
        case = fixture["case"]
        return [int(token) for token in case["prompt_token_ids"]], str(
            case["prompt_token_ids_sha256"]
        )

    import numpy as np

    rng = np.random.default_rng(20260910)
    ids = [int(token) for token in rng.integers(1000, 32000, size=int(args.prompt_rows))]
    return ids, hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _git(*argv: str) -> str | None:
    try:
        out = subprocess.run(
            ("git", *argv), cwd=str(REPO_ROOT), capture_output=True, text=True,
            timeout=20, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def identity() -> dict[str, Any]:
    """Identify the host and the tree this measurement ran against."""

    try:
        digest: str | None = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        digest = None
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "head": _git("rev-parse", "--short", "HEAD"),
        "dirty": None if dirty is None else dirty[:400],
        "probe_script_sha256": digest,
    }

def completion_outcome(http_status: int | None, payload: Any) -> dict[str, Any]:
    """Decide whether one completion response is a usable measurement.

    HTTP 200 is not success by itself: the OpenAI shape lets a server answer 200
    with an ``error`` object, and a rejected or truncated generation still carries
    a ``choices`` list. Either would attribute allocation peaks to a request that
    never completed.
    """

    def fail(reason: str, status: int | None, error: Any = None) -> dict[str, Any]:
        return {"ok": False, "reason": reason, "http_status": status, "error": error}

    if http_status is None:
        return fail("no_http_status", None)
    if int(http_status) != 200:
        error = payload.get("error") if isinstance(payload, Mapping) else None
        return fail("http_error_status", int(http_status), error)
    if not isinstance(payload, Mapping):
        return fail("response_not_an_object", 200)
    if payload.get("error"):
        return fail("error_payload", 200, payload.get("error"))
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return fail("no_choices", 200)
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return fail("choice_not_an_object", 200)
    if not isinstance(choice.get("text"), str):
        return fail("choice_missing_text", 200)
    if not choice.get("finish_reason"):
        return fail("choice_incomplete_finish_reason", 200)
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return fail("usage_missing", 200)
    if not isinstance(usage.get("completion_tokens"), int) or usage["completion_tokens"] <= 0:
        return fail("usage_missing_completion_tokens", 200)
    return {"ok": True, "reason": None, "http_status": 200}


def _log_tail(path: Path, max_lines: int = 120, max_chars: int = 16000) -> list[str] | None:
    """Bounded tail of the server log, so a failed run is diagnosable."""

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except OSError:
        return None
    kept: list[str] = []
    budget = max_chars
    for line in reversed(lines):
        if kept and budget - len(line) < 0:
            break
        kept.append(line)
        budget -= len(line)
    return kept[::-1]


def finish_probe(
    result: dict[str, Any], json_path: Path | None, *, stdout: Any = None, stderr: Any = None
) -> int:
    """Write the artifact, print it, and return the process exit code.

    Every exit path goes through here, so a run that failed during policy
    resolution, log setup, launch, startup or the request still leaves evidence.
    """

    out = sys.stdout if stdout is None else stdout
    stamp = str(result.get("probe_stamp") or time.strftime("%Y%m%dT%H%M%S"))
    path = Path(json_path) if json_path is not None else (
        Path(tempfile.gettempdir()) / f"gguf-server-allocation-probe-{stamp}.json"
    )
    result["artifact_path"] = str(path)
    payload = json.dumps(result, indent=2, default=str)
    print(payload, file=out)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"failed to write artifact {path}: {type(exc).__name__}: {exc}",
              file=sys.stderr if stderr is None else stderr)
        return 1
    passed = result.get("status") == "pass" and result.get("server_teardown") == "clean"
    return 0 if passed else 1


def _base_result(args: argparse.Namespace, stamp: str) -> dict[str, Any]:
    return {
        "kind": "gguf_server_allocation_probe",
        "probe_version": _PROBE_VERSION,
        "protocol_tier": 1,
        "server_route": "one request through the live server process: shared pool with "
                        "slot views and the packed slot-local prefill entry",
        "probe_stamp": stamp,
        "model": str(args.model),
        "prompt_rows": int(args.prompt_rows),
        "decode_tokens": int(args.decode_tokens),
        "port": int(args.port),
        "identity": identity(),
    }


def _await_ready(
    server: subprocess.Popen, args: argparse.Namespace, result: dict[str, Any]
) -> bool:
    """Poll ``/ready`` until the server accepts requests, recording each read."""

    deadline = time.perf_counter() + float(args.startup_timeout)
    polls = 0
    last: dict[str, Any] | None = None
    while time.perf_counter() < deadline:
        if server.poll() is not None:
            result["status"] = "server_exited_during_startup"
            result["server_exit_code"] = int(server.returncode or 0)
            break
        read = _http_json(f"http://127.0.0.1:{args.port}/ready", timeout=5.0)
        polls += 1
        last = document(*read)
        if readiness_accepts_requests(read[0], read[1]):
            result["readiness_before"], result["ready_polls"] = last, polls
            return True
        time.sleep(1.0)
    else:
        result["status"] = "startup_timeout"
    result["readiness_before"], result["ready_polls"] = last, polls
    return False


def _teardown(server: subprocess.Popen, result: dict[str, Any]) -> None:
    """Stop the server, recording how far shutdown got. Never raises."""

    try:
        server.send_signal(signal.SIGTERM)
        server.wait(timeout=60)
        result["server_teardown"] = "clean"
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception as exc:  # noqa: BLE001 - teardown reports, it does not raise
        result["server_teardown"] = f"teardown_failed: {type(exc).__name__}: {exc}"[:200]
        return
    try:
        server.kill()
        server.wait(timeout=30)
        result["server_teardown"] = "killed_after_timeout"
    except Exception as exc:  # noqa: BLE001
        result["server_teardown"] = f"kill_failed: {type(exc).__name__}: {exc}"[:200]


def _request_once(
    port: int, prompt_ids: Sequence[int], decode_tokens: int
) -> tuple[int | None, str | None, str | None]:
    body = json.dumps({
        "prompt": [int(token) for token in prompt_ids],
        "max_tokens": int(decode_tokens),
        "temperature": 0.0,
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            return int(response.status), response.read().decode("utf-8", "replace"), None
    except urllib.error.HTTPError as exc:
        try:
            text = exc.read().decode("utf-8", "replace")
        except OSError:
            text = None
        return int(exc.code), text, f"HTTPError: {exc}"[:400]
    except Exception as exc:  # noqa: BLE001 - the probe reports, it does not raise
        return None, None, f"{type(exc).__name__}: {exc}"[:400]


def run_probe(
    args: argparse.Namespace,
    server_env: Mapping[str, str],
    prompt_ids: Sequence[int],
    result: dict[str, Any],
    log_path: Path,
) -> None:
    """Drive one request through the real server, recording what it cost.

    Never raises: a log that cannot be opened, a launch failure, an exception
    inside the probe and a teardown failure each become a recorded status, and
    the log tail is attached on the way out.
    """

    started = time.perf_counter()
    try:
        try:
            log_handle = log_path.open("w")
        except OSError as exc:
            result["status"] = "server_log_unavailable"
            result["error"] = f"{type(exc).__name__}: {exc}"[:400]
            return
        with log_handle:
            try:
                server = subprocess.Popen(
                    result["command"], cwd=str(REPO_ROOT), env=dict(server_env),
                    stdout=log_handle, stderr=subprocess.STDOUT,
                )
            except OSError as exc:
                result["status"] = "server_launch_failed"
                result["error"] = f"{type(exc).__name__}: {exc}"[:400]
                return
            try:
                _probe_running_server(args, prompt_ids, result, server, started)
            except Exception as exc:  # noqa: BLE001 - artifact must still be written
                result["status"] = "probe_error"
                result["error"] = f"{type(exc).__name__}: {exc}"[:800]
            finally:
                _teardown(server, result)
    finally:
        if result.get("status") != "pass" or result.get("server_teardown") != "clean":
            result["server_log_tail"] = _log_tail(log_path)


def _probe_running_server(
    args: argparse.Namespace,
    prompt_ids: Sequence[int],
    result: dict[str, Any],
    server: subprocess.Popen,
    started: float,
) -> None:
    if not _await_ready(server, args, result):
        return
    result["startup_seconds"] = round(time.perf_counter() - started, 3)
    base = f"http://127.0.0.1:{args.port}"
    result["capabilities_before"] = document(*_http_json(f"{base}/v1/hipengine/capabilities"))

    baseline = _scrape(args.port)
    if baseline is None:
        result["status"] = "metrics_unavailable"
        return
    result["baseline"] = {name: baseline.get(name) for name in _TRACKED_GAUGES}

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
    try:
        request_started = time.perf_counter()
        http_status, response_text, request_error = _request_once(
            args.port, prompt_ids, int(args.decode_tokens)
        )
        request_wall = time.perf_counter() - request_started
        time.sleep(0.5)  # one final scrape window
    finally:
        stop.set()
        thread.join(timeout=5.0)
    final = _scrape(args.port) or {}

    payload: Any = None
    parse_error = None
    if response_text is not None:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            parse_error = f"JSONDecodeError: {exc}"[:300]
    outcome = completion_outcome(http_status, payload)
    ok = bool(outcome["ok"])
    body = payload if isinstance(payload, Mapping) else {}
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], Mapping) else {}
    generated = choice.get("text") if isinstance(choice.get("text"), str) else None
    result.update({
        "request_wall_seconds": round(request_wall, 3),
        "status": "pass" if ok else "request_failed",
        "request_error": request_error,
        "response": {
            "http_status": http_status,
            "outcome": outcome,
            "parse_error": parse_error,
            # Raw server metadata: routing, generation shape, token ids, MTP
            # accounting and prompt accounting stay whole.
            "hipengine": body.get("hipengine"),
            "usage": body.get("usage"),
            "finish_reason": choice.get("finish_reason"),
            "generated_chars": None if generated is None else len(generated),
            "generated_text_head": None if generated is None else generated[:64],
            "body_head": None if ok or response_text is None else response_text[:2000],
        },
        "capabilities_after": document(*_http_json(f"{base}/v1/hipengine/capabilities")),
        "readiness_after": document(*_http_json(f"{base}/ready")),
        "during_request_peaks": dict(sorted(peaks.items())),
        "final": {name: final.get(name) for name in _TRACKED_GAUGES},
        "domains_gib": _domain_peaks(peaks, final),
    })


def _domain_peaks(peaks: Mapping[str, float], final: Mapping[str, float]) -> dict[str, Any]:
    """Per-domain peaks, kept separate because they are never additive."""

    out: dict[str, Any] = {}
    for field, gauge, scale, source in _DOMAIN_GAUGES:
        value = peaks.get(gauge, 0.0)
        if source == "final":
            value = final.get(gauge, value)
        out[field] = round(float(value) / 2**30, 4) if scale == "gib" else value
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    result = _base_result(args, stamp)

    try:
        policy = resolve_policy(args, os.environ)
    except PolicyError as exc:
        result["status"] = "policy_rejected"
        result["error"] = str(exc)
        return finish_probe(result, args.json)
    result["policy"] = policy

    try:
        prompt_ids, prompt_sha = build_prompt(args)
    except Exception as exc:  # noqa: BLE001 - report, then write the artifact
        result["status"] = "prompt_unavailable"
        result["error"] = f"{type(exc).__name__}: {exc}"[:400]
        return finish_probe(result, args.json)
    result["prompt_token_ids_sha256"] = prompt_sha

    result["command"] = build_server_command(args, policy)
    server_env, env_record = build_server_env(args, policy, os.environ)
    result["environment"] = env_record
    log_path = Path(tempfile.gettempdir()) / f"server-alloc-probe-{stamp}.log"
    result["server_log"] = str(log_path)

    run_probe(args, server_env, prompt_ids, result, log_path)
    return finish_probe(result, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
