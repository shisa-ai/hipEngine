#!/usr/bin/env python3
"""Find safe Qwen3.8-27B INT8-KV server context at c=1/2/4/8.

The harness starts a real OpenAI-compatible hipEngine server on an otherwise
idle RX 7900 XTX, requests enough dynamic KV for the offered client width, and
drives deterministic multi-turn ShareGPT conversations.  It records the lower
registry-selected physical residency when the public width queues in waves.  A
natural corpus fills each request to the same near-capacity token shape; as
conversation history grows, only the filler prefix is shortened.  This keeps
every turn at the memory frontier without using repeated-token prompts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.agentic_live import final_ownership_from_server  # noqa: E402
from hipengine.tokenization.identity import token_ids_sha256  # noqa: E402
from hipengine.util.amdgpu_vram import VramSampler, read_memory_used, select_card  # noqa: E402

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks/prompts/qwen38-sharegpt-soak-v1.json"
GIB = 1 << 30
MIB = 1 << 20


class SoakError(RuntimeError):
    """Raised when the safety protocol cannot be completed."""


@dataclass(frozen=True)
class PromptLane:
    lane: int
    source_row_index: int
    source_id: str
    user_turns: tuple[str, ...]


@dataclass(frozen=True)
class PromptFixture:
    path: str
    sha256: str
    source_dataset: str
    source_commit: str
    source_url: str
    lanes: tuple[PromptLane, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_fixture(path: str | Path) -> PromptFixture:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("kind") != "pinned_sharegpt_server_soak_prompts":
        raise ValueError("unsupported ShareGPT soak fixture")
    source = payload.get("source")
    rows = payload.get("lanes")
    if not isinstance(source, Mapping) or not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("ShareGPT soak fixture must contain exactly eight lanes")
    lanes: list[PromptLane] = []
    for expected, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or int(raw.get("lane", -1)) != expected:
            raise ValueError("ShareGPT lane IDs must be contiguous from zero")
        turns = raw.get("user_turns")
        if (
            not isinstance(turns, list)
            or len(turns) < 2
            or not all(isinstance(turn, str) and turn.strip() for turn in turns)
        ):
            raise ValueError(f"ShareGPT lane {expected} needs at least two user turns")
        lanes.append(
            PromptLane(
                lane=expected,
                source_row_index=int(raw["source_row_index"]),
                source_id=str(raw["source_id"]),
                user_turns=tuple(str(turn) for turn in turns),
            )
        )
    return PromptFixture(
        path=str(resolved),
        sha256=_file_sha256(resolved),
        source_dataset=str(source["dataset"]),
        source_commit=str(source["dataset_commit"]),
        source_url=str(source["url"]),
        lanes=tuple(lanes),
    )


def pool_pages(context_tokens: int, concurrency: int) -> int:
    if int(context_tokens) <= 0 or int(concurrency) <= 0:
        raise ValueError("context and concurrency must be positive")
    return int(concurrency) * math.ceil(int(context_tokens) / 256)


def selected_lane_indices(lane_count: int, *, concurrency: int, cycle: int) -> tuple[int, ...]:
    if lane_count <= 0 or concurrency <= 0 or cycle < 0:
        raise ValueError("lane count/concurrency must be positive and cycle non-negative")
    if concurrency > lane_count:
        raise ValueError("concurrency exceeds available prompt lanes")
    start = (int(cycle) * int(concurrency)) % int(lane_count)
    return tuple((start + offset) % int(lane_count) for offset in range(int(concurrency)))


def effective_resident_capacity(
    *,
    current_pool_pages: int,
    pages_per_request: int,
    offered_concurrency: int,
) -> int:
    if min(current_pool_pages, pages_per_request, offered_concurrency) <= 0:
        raise ValueError("pool pages, per-request pages, and concurrency must be positive")
    if int(current_pool_pages) % int(pages_per_request):
        raise ValueError("initial pool pages do not cover an integral resident width")
    capacity = int(current_pool_pages) // int(pages_per_request)
    if capacity <= 0 or capacity > int(offered_concurrency):
        raise ValueError("effective resident capacity is outside the offered width")
    return capacity


def extract_chat_response(
    response: Mapping[str, Any],
    *,
    expected_prompt_tokens: int,
    expected_completion_tokens: int,
) -> dict[str, Any]:
    choices = response.get("choices")
    usage = response.get("usage")
    root = response.get("hipengine")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], Mapping)
        or not isinstance(usage, Mapping)
        or not isinstance(root, Mapping)
    ):
        raise ValueError("chat response is missing choices, usage, or hipengine accounting")
    accounting = root.get("token_accounting")
    rows = accounting.get("choice_generated_token_ids") if isinstance(accounting, Mapping) else None
    generation_shape = root.get("generation_shape")
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], list)
        or not all(isinstance(token, int) and not isinstance(token, bool) for token in rows[0])
    ):
        raise ValueError("chat response is missing authoritative generated token IDs")
    if not isinstance(generation_shape, Mapping):
        raise ValueError("chat response is missing authoritative generation shape")
    generated = [int(token) for token in rows[0]]
    prompt_tokens = int(usage.get("prompt_tokens", -1))
    completion_tokens = int(usage.get("completion_tokens", -1))
    if prompt_tokens != int(expected_prompt_tokens):
        raise ValueError(
            f"server prompt accounting drifted: {prompt_tokens} != {expected_prompt_tokens}"
        )
    if completion_tokens != len(generated) or len(generated) != int(expected_completion_tokens):
        raise ValueError(
            "server completion accounting drifted: "
            f"usage={completion_tokens}, ids={len(generated)}, expected={expected_completion_tokens}"
        )
    choice = choices[0]
    message = choice.get("message")
    text = (
        str(message.get("content") or "")
        if isinstance(message, Mapping)
        else str(choice.get("text") or "")
    )
    return {
        "generated_token_ids": generated,
        "generated_token_ids_sha256": token_ids_sha256(generated),
        "generated_text": text,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "exact_accounting": True,
        "generation_shape": dict(generation_shape),
    }


def summarize_generation_shapes(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    queue_group_ids: list[str] = []
    queue_group_request_counts: list[int] = []
    backend_input_rows: list[int] = []
    backend_actual_group_rows: list[int] = []
    for request in requests:
        shape = request.get("generation_shape")
        if not isinstance(shape, Mapping):
            continue
        queue = shape.get("queue_group")
        if isinstance(queue, Mapping):
            group_id = str(queue.get("id") or "")
            if group_id:
                queue_group_ids.append(group_id)
            count = queue.get("request_count")
            if isinstance(count, int) and not isinstance(count, bool):
                queue_group_request_counts.append(int(count))
        groups = shape.get("backend_groups")
        if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes, bytearray)):
            for group in groups:
                if not isinstance(group, Mapping):
                    continue
                input_rows = group.get("input_rows")
                if isinstance(input_rows, int) and not isinstance(input_rows, bool):
                    backend_input_rows.append(int(input_rows))
                actual = group.get("actual_group_rows")
                if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes, bytearray)):
                    backend_actual_group_rows.extend(
                        int(value)
                        for value in actual
                        if isinstance(value, int) and not isinstance(value, bool)
                    )
    return {
        "request_count": len(requests),
        "shape_count": sum(isinstance(row.get("generation_shape"), Mapping) for row in requests),
        "queue_group_count": len(set(queue_group_ids)),
        "queue_group_request_counts": sorted(set(queue_group_request_counts)),
        "backend_input_rows": sorted(set(backend_input_rows)),
        "backend_actual_group_rows": sorted(set(backend_actual_group_rows)),
        "maximum_backend_group_rows": max(backend_actual_group_rows, default=None),
    }


def evaluate_safety_gate(
    *,
    requests: Sequence[Mapping[str, Any]],
    ready_after: Mapping[str, Any],
    ownership: Mapping[str, Any],
    total_vram_bytes: int,
    peak_vram_bytes: int,
    minimum_headroom_mib: int,
) -> dict[str, Any]:
    zero_fields = (
        "pending_requests",
        "active_requests",
        "stream_producers",
        "model_active_requests",
        "session_count",
        "kv_refcounted_pages",
        "kv_pinned_pages",
        "graph_owners",
        "workspace_owners",
        "cache_resident_entries",
        "cache_resident_pages",
        "cache_resident_bytes",
    )
    headroom = int(total_vram_bytes) - int(peak_vram_bytes)
    checks = {
        "all_requests": bool(requests) and all(row.get("passed") is True for row in requests),
        "ready_after": ready_after.get("ready") is True,
        "idle_ownership": all(int(ownership.get(field, -1)) == 0 for field in zero_fields),
        "minimum_headroom": headroom >= int(minimum_headroom_mib) * MIB,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "headroom_bytes": headroom,
        "headroom_gib": headroom / GIB,
        "minimum_headroom_mib": int(minimum_headroom_mib),
    }


class LiveHTTPClient:
    def __init__(self, base_url: str, *, timeout_s: float) -> None:
        parsed = urllib.parse.urlparse(str(base_url))
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("soak client currently requires an http base URL")
        self.host = parsed.hostname
        self.port = int(parsed.port or 80)
        self.timeout_s = float(timeout_s)

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> tuple[int, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        try:
            connection.request(
                str(method),
                "/" + str(path).lstrip("/"),
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

    def get(self, path: str) -> tuple[int, Any]:
        return self._request("GET", path)

    def post(self, path: str, payload: Mapping[str, Any]) -> tuple[int, Any]:
        return self._request("POST", path, payload)

    def ready(self) -> dict[str, Any]:
        status, payload = self.get("ready")
        if status != 200 or not isinstance(payload, dict):
            raise SoakError(f"/ready failed with HTTP {status}: {payload}")
        return payload

    def sessions(self) -> dict[str, Any]:
        status, payload = self.get("v1/hipengine/sessions")
        if status != 200 or not isinstance(payload, dict):
            raise SoakError(f"sessions endpoint failed with HTTP {status}: {payload}")
        return payload

    def tokenize(self, text: str) -> list[int]:
        status, payload = self.post("v1/hipengine/tokenize", {"text": str(text)})
        tokens = payload.get("token_ids") if isinstance(payload, Mapping) else None
        if status != 200 or not isinstance(tokens, list) or not all(isinstance(x, int) for x in tokens):
            raise SoakError(f"tokenize failed with HTTP {status}: {payload}")
        return [int(token) for token in tokens]

    def detokenize(self, token_ids: Sequence[int]) -> str:
        status, payload = self.post(
            "v1/hipengine/detokenize",
            {"token_ids": [int(token) for token in token_ids], "skip_special": False},
        )
        text = payload.get("text") if isinstance(payload, Mapping) else None
        if status != 200 or not isinstance(text, str):
            raise SoakError(f"detokenize failed with HTTP {status}: {payload}")
        return text

    def count_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        status, payload = self.post(
            "v1/hipengine/count_tokens",
            {"messages": [dict(message) for message in messages], "enable_thinking": False},
        )
        count = payload.get("token_count") if isinstance(payload, Mapping) else None
        if status != 200 or not isinstance(count, int):
            raise SoakError(f"count_tokens failed with HTTP {status}: {payload}")
        return int(count)


@dataclass
class LaneState:
    lane: PromptLane
    history: list[dict[str, str]]
    corpus_tokens: list[int]


def _natural_corpus(fixture: PromptFixture, *, lane_index: int) -> str:
    ordered = [
        fixture.lanes[(lane_index + offset) % len(fixture.lanes)]
        for offset in range(len(fixture.lanes))
    ]
    sections = []
    for lane in ordered:
        sections.append(
            f"[ShareGPT source row {lane.source_row_index} id {lane.source_id}]\n"
            + "\n".join(lane.user_turns)
        )
    return (
        "The following is a pinned natural-language reference corpus used only as context. "
        "Answer the final user message, not the corpus.\n\n" + "\n\n".join(sections)
    )


def _repeat_prefix(values: Sequence[int], count: int) -> list[int]:
    if count < 0 or not values:
        raise ValueError("natural corpus tokens must be non-empty and count non-negative")
    repeats, tail = divmod(int(count), len(values))
    return list(values) * repeats + list(values[:tail])


def _fit_messages(
    client: LiveHTTPClient,
    state: LaneState,
    *,
    target_prompt_tokens: int,
    user_text: str,
) -> tuple[list[dict[str, str]], int, int]:
    dialogue = [*state.history, {"role": "user", "content": str(user_text)}]
    base_count = client.count_tokens(dialogue)
    if base_count > int(target_prompt_tokens):
        raise SoakError(
            f"lane {state.lane.lane} dialogue alone has {base_count} tokens, "
            f"above target {target_prompt_tokens}"
        )
    budget = max(0, int(target_prompt_tokens) - base_count)
    observed = base_count
    messages: list[dict[str, str]] = dialogue
    for _ in range(8):
        filler = client.detokenize(_repeat_prefix(state.corpus_tokens, budget))
        messages = [{"role": "system", "content": filler}, *dialogue]
        observed = client.count_tokens(messages)
        delta = int(target_prompt_tokens) - observed
        if delta == 0:
            break
        budget = max(0, budget + delta)
    if observed > int(target_prompt_tokens) or int(target_prompt_tokens) - observed > 8:
        raise SoakError(
            f"lane {state.lane.lane} could not fit near target: {observed} vs {target_prompt_tokens}"
        )
    return messages, observed, budget


def _one_chat_request(
    client: LiveHTTPClient,
    *,
    served_model_name: str,
    lane_state: LaneState,
    messages: Sequence[Mapping[str, Any]],
    prompt_tokens: int,
    max_tokens: int,
    cycle: int,
    turn: int,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    barrier.wait(timeout=60.0)
    started = time.perf_counter()
    status = 0
    payload: Any = None
    error: str | None = None
    normalized: dict[str, Any] | None = None
    try:
        status, payload = client.post(
            "v1/chat/completions",
            {
                "model": str(served_model_name),
                "messages": [dict(message) for message in messages],
                "max_tokens": int(max_tokens),
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "enable_thinking": False,
                "stream": False,
            },
        )
        if status != 200 or not isinstance(payload, Mapping):
            raise SoakError(f"HTTP {status}: {payload}")
        normalized = extract_chat_response(
            payload,
            expected_prompt_tokens=int(prompt_tokens),
            expected_completion_tokens=int(max_tokens),
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    row = {
        "cycle": int(cycle),
        "turn": int(turn),
        "lane": int(lane_state.lane.lane),
        "source_row_index": int(lane_state.lane.source_row_index),
        "source_id": str(lane_state.lane.source_id),
        "http_status": int(status),
        "elapsed_seconds": elapsed,
        "prompt_tokens": int(prompt_tokens),
        "max_tokens": int(max_tokens),
        "error": error,
        "passed": error is None and normalized is not None,
        **({} if normalized is None else normalized),
    }
    if normalized is not None:
        lane_state.history.extend(
            (
                {"role": "user", "content": str(messages[-1]["content"])},
                {"role": "assistant", "content": str(normalized["generated_text"])},
            )
        )
    return row


def _wait_ready(
    client: LiveHTTPClient,
    process: subprocess.Popen[str],
    log_path: Path,
    *,
    timeout_s: float,
) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    deadline = started + float(timeout_s)
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            raise SoakError(f"server exited during startup with {process.returncode}\n{tail}")
        try:
            ready = client.ready()
            if ready.get("ready") is True:
                return time.perf_counter() - started, ready
        except Exception as exc:
            last_error = exc
        time.sleep(0.25)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
    raise SoakError(f"server readiness timed out: {last_error}\n{tail}")


def _wait_idle(client: LiveHTTPClient, *, timeout_s: float) -> tuple[dict[str, Any], dict[str, int]]:
    deadline = time.monotonic() + float(timeout_s)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            ready = client.ready()
            ownership = final_ownership_from_server(ready, client.sessions(), cache_mode="off")
            return ready, ownership
        except Exception as exc:
            last_error = exc
        time.sleep(0.05)
    raise SoakError(f"server did not drain request ownership: {last_error}")


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15.0)


def _capture(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"command": list(command), "returncode": completed.returncode, "output": completed.stdout}


def _server_command(args: argparse.Namespace) -> list[str]:
    return [
        str(Path(sys.executable)),
        "-m",
        "hipengine.server",
        "--model",
        str(args.model.resolve()),
        "--backend",
        "hip_gfx1100",
        "--quant",
        "gguf_q4_k_m",
        "--served-model-name",
        str(args.served_model_name),
        "--max-context-tokens",
        str(args.context_tokens),
        "--max-active-requests",
        str(args.concurrency),
        "--kv-storage",
        "int8_per_token_head",
        "--kv-scale-dtype",
        "fp32",
        "--kv-scale-granularity",
        "per_token_head",
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
        str(args.host),
        "--port",
        str(args.port),
        "--log-level",
        str(args.server_log_level),
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model.is_file():
        raise ValueError(f"model does not exist: {args.model}")
    if args.concurrency not in {1, 2, 4, 8}:
        raise ValueError("concurrency must be one of 1,2,4,8")
    if min(args.context_tokens, args.cycles, args.turns, args.max_tokens) <= 0:
        raise ValueError("context, cycles, turns, and max-tokens must be positive")
    target_prompt_tokens = int(args.context_tokens) - int(args.max_tokens) - 1
    if target_prompt_tokens <= 0:
        raise ValueError("context does not leave room for prompt and completion")
    fixture = load_prompt_fixture(args.prompts)
    card = select_card(card_name=str(args.drm_card))
    baseline = read_memory_used(card, domain="vram")
    if baseline > int(args.maximum_baseline_mib) * MIB:
        raise SoakError(
            f"target GPU is not idle: baseline {baseline / MIB:.1f} MiB exceeds "
            f"{args.maximum_baseline_mib} MiB"
        )

    pages = pool_pages(int(args.context_tokens), int(args.concurrency))
    per_request_pages = math.ceil(int(args.context_tokens) / 256)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1",
            "HIP_VISIBLE_DEVICES": str(args.hip_device),
            "HIPENGINE_HIP_ARCH": "gfx1100",
            "GPU_MAX_HW_QUEUES": "1",
            "HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG": "1",
            "HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS": "none",
            "HIPENGINE_KV_POOL_INITIAL_PAGES": str(pages),
            "HIPENGINE_KV_POOL_LOW_WATER_PAGES": str(pages),
            "HIPENGINE_KV_POOL_HIGH_WATER_PAGES": str(pages),
            "HIPENGINE_KV_POOL_CHUNK_PAGES": str(per_request_pages),
            "HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS": "0",
        }
    )
    if args.compiler_version_file is not None:
        env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())

    args.work_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.work_dir / f"c{args.concurrency}-ctx{args.context_tokens}.server.log"
    command = _server_command(args)
    invocation = [str(Path(sys.executable)), str(Path(__file__).resolve()), *sys.argv[1:]]
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "kind": "qwen38_27b_int8_kv_dedicated_xtx_context_soak",
        "status": "running",
        "passed": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "context_tokens_per_request": int(args.context_tokens),
            "target_prompt_tokens": target_prompt_tokens,
            "max_tokens": int(args.max_tokens),
            "reserved_context_slots": 1,
            "offered_client_concurrency": int(args.concurrency),
            "aggregate_offered_context_tokens": int(args.context_tokens) * int(args.concurrency),
            "cycles": int(args.cycles),
            "turns_per_lane_per_cycle": int(args.turns),
            "prefix_cache": "off",
            "speculative_mtp": "off",
            "sampling": "greedy temperature=0, ignore_eos=true",
            "kv_storage": "int8_per_token_head",
            "kv_scale_dtype": "fp32",
            "kv_scale_granularity": "per_token_head",
            "requested_pool_initial_low_high_pages": pages,
            "pool_chunk_pages": per_request_pages,
            "whole_card_poll_ms": float(args.memory_poll_ms),
            "minimum_headroom_mib": int(args.minimum_headroom_mib),
            "maximum_idle_baseline_mib": int(args.maximum_baseline_mib),
        },
        "prompt_fixture": asdict(fixture),
        "model": {
            "path": str(args.model.resolve()),
            "size_bytes": args.model.stat().st_size,
            "sampled_fingerprint_sha256": _sampled_model_hash(args.model),
        },
        "source": {
            "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "repo_status": subprocess.check_output(["git", "status", "-sb"], cwd=REPO_ROOT, text=True),
            "command": invocation,
            "command_shell": shlex.join(invocation),
            "server_command": command,
            "server_command_shell": shlex.join(command),
            "environment": {key: env[key] for key in sorted(env) if key.startswith("HIPENGINE_") or key in {"HIP_VISIBLE_DEVICES", "GPU_MAX_HW_QUEUES"}},
        },
        "hardware": {
            "card": card.to_dict(),
            "baseline_vram_bytes": baseline,
            "baseline_vram_gib": baseline / GIB,
            "rocminfo": _capture(["bash", "-lc", "rocminfo | grep -E 'Name:|gfx' | head -12"]),
            "rocm_smi": _capture(["rocm-smi", "--showpids", "--showuse", "--showmemuse"]),
            "hipcc": _capture(["hipcc", "--version"]),
        },
        "server": {"log_path": str(log_path)},
        "cycles": [],
    }
    _write_json(args.output, artifact)

    sampler = VramSampler(card, interval_ms=float(args.memory_poll_ms), keep_samples=False)
    process: subprocess.Popen[str] | None = None
    log_handle = log_path.open("w", encoding="utf-8")
    requests: list[dict[str, Any]] = []
    ready_after: dict[str, Any] = {}
    ownership: dict[str, int] = {}
    error: str | None = None
    sampler.start()
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        client = LiveHTTPClient(f"http://{args.host}:{args.port}", timeout_s=float(args.request_timeout_s))
        startup_seconds, ready_before = _wait_ready(
            client,
            process,
            log_path,
            timeout_s=float(args.startup_timeout_s),
        )
        ready_pool = ready_before.get("kv_capacity", {}).get("pool", {})
        current_pool_pages = int(ready_pool.get("current_pages", 0))
        resident_capacity = effective_resident_capacity(
            current_pool_pages=current_pool_pages,
            pages_per_request=per_request_pages,
            offered_concurrency=int(args.concurrency),
        )
        artifact["protocol"].update(
            {
                "effective_pool_initial_pages": current_pool_pages,
                "effective_physical_resident_capacity": resident_capacity,
                "maximum_simultaneously_resident_context_tokens": (
                    resident_capacity * int(args.context_tokens)
                ),
                "offered_width_requires_queue_waves": resident_capacity < int(args.concurrency),
            }
        )
        artifact["server"].update(
            {
                "pid": process.pid,
                "startup_seconds": startup_seconds,
                "ready_before": ready_before,
                "vram_at_ready_bytes": read_memory_used(card, domain="vram"),
            }
        )
        _write_json(args.output, artifact)

        corpus_tokens = [
            client.tokenize(_natural_corpus(fixture, lane_index=index))
            for index in range(len(fixture.lanes))
        ]
        for cycle in range(int(args.cycles)):
            lane_indices = selected_lane_indices(
                len(fixture.lanes), concurrency=int(args.concurrency), cycle=cycle
            )
            states = [
                LaneState(fixture.lanes[index], [], corpus_tokens[index])
                for index in lane_indices
            ]
            cycle_row: dict[str, Any] = {
                "cycle": cycle,
                "lane_indices": list(lane_indices),
                "turns": [],
            }
            artifact["cycles"].append(cycle_row)
            for turn in range(int(args.turns)):
                prepared: list[tuple[LaneState, list[dict[str, str]], int, int]] = []
                for state in states:
                    user_text = state.lane.user_turns[turn % len(state.lane.user_turns)]
                    messages, observed, filler_budget = _fit_messages(
                        client,
                        state,
                        target_prompt_tokens=target_prompt_tokens,
                        user_text=user_text,
                    )
                    prepared.append((state, messages, observed, filler_budget))
                barrier = threading.Barrier(len(prepared) + 1)
                released_at = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(prepared)) as executor:
                    futures = [
                        executor.submit(
                            _one_chat_request,
                            client,
                            served_model_name=str(args.served_model_name),
                            lane_state=state,
                            messages=messages,
                            prompt_tokens=observed,
                            max_tokens=int(args.max_tokens),
                            cycle=cycle,
                            turn=turn,
                            barrier=barrier,
                        )
                        for state, messages, observed, _budget in prepared
                    ]
                    barrier.wait(timeout=60.0)
                    rows = [future.result() for future in futures]
                burst_wall = time.perf_counter() - released_at
                requests.extend(rows)
                ready_turn, ownership_turn = _wait_idle(
                    client, timeout_s=float(args.idle_timeout_s)
                )
                turn_row = {
                    "turn": turn,
                    "burst_wall_seconds": burst_wall,
                    "filler_token_budgets": [budget for _state, _messages, _observed, budget in prepared],
                    "requests": rows,
                    "ready_after": ready_turn,
                    "ownership_after": ownership_turn,
                    "vram_after_bytes": read_memory_used(card, domain="vram"),
                    "passed": all(row["passed"] is True for row in rows),
                }
                cycle_row["turns"].append(turn_row)
                print(
                    f"c{args.concurrency} ctx={args.context_tokens} cycle={cycle} turn={turn}: "
                    f"{sum(row['passed'] is True for row in rows)}/{len(rows)} passed, "
                    f"wall={burst_wall:.3f}s",
                    flush=True,
                )
                _write_json(args.output, artifact)
                if not turn_row["passed"]:
                    raise SoakError(f"cycle {cycle} turn {turn} request gate failed")
        ready_after, ownership = _wait_idle(client, timeout_s=float(args.idle_timeout_s))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            _stop_server(process)
        log_handle.close()
        time.sleep(max(2.0, 2.0 * float(args.memory_poll_ms) / 1000.0))
        sampler.stop()

    memory = sampler.result().to_dict()
    gate = evaluate_safety_gate(
        requests=requests,
        ready_after=ready_after,
        ownership=ownership,
        total_vram_bytes=int(memory["memory_total_bytes"]),
        peak_vram_bytes=int(memory["peak_bytes"]),
        minimum_headroom_mib=int(args.minimum_headroom_mib),
    )
    if error is not None:
        gate["passed"] = False
        gate["checks"]["run_completed"] = False
    else:
        gate["checks"]["run_completed"] = True
    artifact.update(
        {
            "status": "passed_safe_context_soak" if gate["passed"] else "failed_context_soak",
            "passed": bool(gate["passed"]),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
            "request_count": len(requests),
            "execution_shape": summarize_generation_shapes(requests),
            "ready_after": ready_after,
            "final_ownership": ownership,
            "memory": memory,
            "gate": gate,
            "server_returncode": None if process is None else process.returncode,
            "server_log_sha256": _file_sha256(log_path),
            "server_log_tail": log_path.read_text(encoding="utf-8", errors="replace")[-12000:],
        }
    )
    _write_json(args.output, artifact)
    print(
        json.dumps(
            {
                "passed": artifact["passed"],
                "context_tokens_per_request": args.context_tokens,
                "offered_client_concurrency": args.concurrency,
                "aggregate_offered_context_tokens": args.context_tokens * args.concurrency,
                "effective_physical_resident_capacity": artifact["protocol"].get(
                    "effective_physical_resident_capacity"
                ),
                "requests": len(requests),
                "peak_vram_gib": memory["peak_gib"],
                "headroom_gib": gate["headroom_gib"],
                "error": error,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return artifact


def _sampled_model_hash(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        for offset in sorted({0, max(0, size // 2 - (1 << 19)), max(0, size - (1 << 20))}):
            handle.seek(offset)
            digest.update(handle.read(1 << 20))
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--served-model-name", default="qwen38-27b-q4km")
    parser.add_argument("--hip-device", default="1")
    parser.add_argument("--drm-card", default="card0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8038)
    parser.add_argument("--batch-window-ms", type=float, default=5.0)
    parser.add_argument("--startup-min-free-mib", type=int, default=1024)
    parser.add_argument("--minimum-headroom-mib", type=int, default=512)
    parser.add_argument("--maximum-baseline-mib", type=int, default=128)
    parser.add_argument("--memory-poll-ms", type=float, default=20.0)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--request-timeout-s", type=float, default=1200.0)
    parser.add_argument("--idle-timeout-s", type=float, default=60.0)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--server-log-level", default="info")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = run(args)
    except (OSError, ValueError, SoakError, json.JSONDecodeError) as exc:
        print(f"Qwen3.8 INT8 server context soak rejected: {exc}", file=sys.stderr)
        return 2
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
