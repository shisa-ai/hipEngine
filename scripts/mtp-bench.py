#!/usr/bin/env python3
"""llama.cpp-compatible MTP prompt-suite benchmark.

This is a protocol-compatible hipEngine copy of the ad-hoc ``mtp-bench.py``
script used in llama.cpp MTP PR discussions, including PR #23287.  It keeps the
same default prompt suite and request shape so we can run the same benchmark
against a llama.cpp server and a hipEngine OpenAI-compatible server.  It can
also wrap hipEngine's existing prompt-suite verifier-economics harness via
``--mode hipengine-current`` so old/current diagnostic artifacts and new
llama.cpp-compatible server numbers share one entry point.

Defaults intentionally mirror the upstream gist as of raw revision
``0bee1e2b88904e62670d0df1cf0991883b0815d7``:

* POST ``/v1/chat/completions``
* ``model="llama"``
* one user message per prompt
* ``max_tokens=192``
* ``seed=42``
* no explicit temperature/top_p/cache_prompt unless requested by CLI flags

Source reference:
https://gist.github.com/am17an/228edfb84ed082aa88e3865d6fa27090
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance

DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "fixtures" / "llamacpp_mtp_bench_prompts.json"
DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_ENGINE_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")
DEFAULT_HIPENGINE_RAW_ROOT = Path("/tmp/hipengine-mtp-llamacpp-prompt-suite-economics")
SOURCE_GIST = "https://gist.github.com/am17an/228edfb84ed082aa88e3865d6fa27090"
SOURCE_RAW = (
    "https://gist.githubusercontent.com/am17an/228edfb84ed082aa88e3865d6fa27090/raw/"
    "0bee1e2b88904e62670d0df1cf0991883b0815d7/mtp-bench.py"
)
DEFAULT_HELDOUT_PROMPT_NAMES = frozenset(
    {
        "code_markdown_table",
        "general_en_explain",
        "general_ja_explain",
        "mixed_ja_en_review",
    }
)


class BenchError(RuntimeError):
    """Raised for benchmark setup or response-shape errors."""


def server_artifact_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Collect canonical identity for a server-mode benchmark artifact."""

    explicit_model_path = getattr(args, "artifact_model_path", None)
    request_model_path = Path(str(getattr(args, "model", ""))).expanduser()
    model_path = explicit_model_path if explicit_model_path is not None else request_model_path
    return collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(getattr(args, "artifact_configured_backend", "auto")),
        resolved_backend=getattr(args, "artifact_resolved_backend", None),
        target_arch=getattr(args, "artifact_target_arch", None),
        device_name=getattr(args, "artifact_device_name", None),
        model_path=model_path,
        model_revision=getattr(args, "artifact_model_revision", None),
        quant=getattr(args, "artifact_quant", None),
        kv_dtype=getattr(args, "artifact_kv_dtype", None),
        command=(sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]),
        build_profile=getattr(args, "artifact_build_profile", None),
        timing_protocol="client_makespan",
        warmups=int(getattr(args, "artifact_warmups", 0)),
        repetitions=int(getattr(args, "artifact_repetitions", 1)),
    )


def _prompt_text_from_messages(messages: Any, *, path: Path, name: str) -> str:
    if not isinstance(messages, list) or len(messages) != 1:
        raise BenchError(
            f"{path} prompt {name!r} must contain exactly one user message"
        )
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise BenchError(f"{path} prompt {name!r} must contain one user message")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise BenchError(f"{path} prompt {name!r} user content must be non-empty text")
    return content


def _normalize_prompt_entry(item: Any, *, path: Path) -> dict[str, str]:
    if not isinstance(item, dict):
        raise BenchError(f"invalid prompt entry in {path}: {item!r}")
    name = str(item.get("name") or item.get("id") or "")
    if not name:
        raise BenchError(f"prompt entries require a non-empty name/id: {item!r}")
    prompt = item.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        prompt = _prompt_text_from_messages(item.get("messages"), path=path, name=name)

    normalized = {"name": name, "prompt": prompt}
    category = item.get("category")
    if category is not None:
        if not isinstance(category, str) or not category:
            raise BenchError(f"{path} prompt {name!r} category must be non-empty text")
        normalized["category"] = category
    split = item.get("split")
    if split is not None:
        if split not in {"train", "heldout"}:
            raise BenchError(f"{path} prompt {name!r} split must be train or heldout")
        normalized["split"] = str(split)
    elif category is not None:
        normalized["split"] = "heldout" if name in DEFAULT_HELDOUT_PROMPT_NAMES else "train"
    return normalized


def load_prompt_suite(path: Path) -> dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(raw_text)
    except json.JSONDecodeError:
        raw_prompts: list[Any] = []
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw_prompts.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise BenchError(f"invalid JSONL in {path} at line {line_number}: {exc}") from exc
        suite: dict[str, Any] = {"prompts": raw_prompts, "source_format": "jsonl"}
    else:
        if not isinstance(loaded, dict):
            raise BenchError(f"{path} must contain a JSON object or JSONL prompt rows")
        suite = dict(loaded)
        suite["source_format"] = "json"

    prompts = suite.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise BenchError(f"{path} does not contain a non-empty 'prompts' list")

    seen: set[str] = set()
    normalized_prompts: list[dict[str, str]] = []
    for item in prompts:
        normalized = _normalize_prompt_entry(item, path=path)
        name = normalized["name"]
        if name in seen:
            raise BenchError(f"duplicate prompt name in {path}: {name}")
        seen.add(name)
        normalized_prompts.append(normalized)
    suite["prompts"] = normalized_prompts
    return suite


def split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def select_prompts(
    suite: dict[str, Any],
    *,
    names_csv: str | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    prompts = [dict(prompt) for prompt in suite["prompts"]]
    names = split_csv(names_csv)
    if names:
        by_name = {p["name"]: p for p in prompts}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise BenchError(f"unknown prompt name(s): {', '.join(missing)}")
        prompts = [by_name[name] for name in names]
    if limit is not None:
        prompts = prompts[: max(0, limit)]
    if not prompts:
        raise BenchError("prompt selection is empty")
    return prompts


def prompt_suite_metadata(
    path: Path,
    suite: dict[str, Any],
    prompts: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        source = str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        source = str(path.resolve())
    category_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for prompt in prompts:
        category = prompt.get("category")
        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
        split = prompt.get("split")
        if split:
            split_counts[split] = split_counts.get(split, 0) + 1
    return {
        "schema_version": 1,
        "source": source,
        "source_format": str(suite.get("source_format") or "unknown"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selected_prompt_count": len(prompts),
        "selected_prompt_names": [prompt["name"] for prompt in prompts],
        "category_counts": dict(sorted(category_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
    }


def make_payload(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.cache_prompt is not None:
        payload["cache_prompt"] = args.cache_prompt
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if args.extra_payload:
        try:
            extra = json.loads(args.extra_payload)
        except json.JSONDecodeError as exc:
            raise BenchError(f"--extra-payload must be a JSON object: {exc}") from exc
        if not isinstance(extra, dict):
            raise BenchError("--extra-payload must decode to a JSON object")
        payload.update(extra)
    return payload


def post_json(url: str, payload: dict[str, Any], *, timeout: float, api_key: str | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = response.read()
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise BenchError(f"HTTP {exc.code} from {url}: {body}") from exc
    except error.URLError as exc:
        raise BenchError(f"failed to POST {url}: {exc}") from exc
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise BenchError(f"expected JSON object response from {url}, got {type(parsed).__name__}")
    return parsed


def first_number(*values: Any) -> int | float | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def choice_hipengine_payloads(response: dict[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list):
        return []
    payloads: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        payload = choice.get("hipengine")
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def exact_token_accounting(response: dict[str, Any]) -> dict[str, Any] | None:
    hipengine = response.get("hipengine")
    if not isinstance(hipengine, dict):
        return None
    accounting = hipengine.get("token_accounting")
    if not isinstance(accounting, dict):
        return None
    raw_rows = accounting.get("choice_generated_token_ids")
    raw_counts = accounting.get("choice_generated_tokens")
    raw_total = accounting.get("total_generated_tokens")
    if not isinstance(raw_rows, list) or not isinstance(raw_counts, list):
        raise BenchError("hipEngine token_accounting choice fields must be lists")
    rows: list[list[int]] = []
    for row in raw_rows:
        if not isinstance(row, list):
            raise BenchError("hipEngine choice_generated_token_ids rows must be lists")
        normalized: list[int] = []
        for token_id in row:
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise BenchError("hipEngine generated token IDs must be non-negative integers")
            normalized.append(int(token_id))
        rows.append(normalized)
    counts = []
    for value in raw_counts:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BenchError("hipEngine choice_generated_tokens must contain non-negative integers")
        counts.append(int(value))
    expected_counts = [len(row) for row in rows]
    if counts != expected_counts:
        raise BenchError("hipEngine choice_generated_tokens do not match exact ID rows")
    if isinstance(raw_total, bool) or not isinstance(raw_total, int) or raw_total < 0:
        raise BenchError("hipEngine total_generated_tokens must be a non-negative integer")
    total = int(raw_total)
    if total != sum(counts):
        raise BenchError("hipEngine total_generated_tokens does not match choice totals")
    normalized_accounting: dict[str, Any] = {
        "choice_generated_token_ids": rows,
        "choice_generated_tokens": counts,
        "total_generated_tokens": total,
    }
    retokenized = accounting.get("retokenized_visible_tokens")
    if retokenized is not None:
        if isinstance(retokenized, bool) or not isinstance(retokenized, int) or retokenized < 0:
            raise BenchError("hipEngine retokenized_visible_tokens must be a non-negative integer")
        normalized_accounting["retokenized_visible_tokens"] = int(retokenized)
    return normalized_accounting


def numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, bool) or raw is None:
            continue
        if not isinstance(raw, (int, float)):
            continue
        out[str(key)] = float(raw)
    return out


def _shape_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchError(f"{label} must be an integer >= {minimum}")
    return int(value)


def generation_shape_from_response(response: dict[str, Any]) -> dict[str, Any] | None:
    root = response.get("hipengine")
    if not isinstance(root, dict) or "generation_shape" not in root:
        return None
    raw = root.get("generation_shape")
    if not isinstance(raw, dict):
        raise BenchError("hipengine.generation_shape must be an object")
    if raw.get("schema_version") != 1:
        raise BenchError("hipengine.generation_shape.schema_version must be 1")
    route = raw.get("route")
    if not isinstance(route, str) or not route.strip():
        raise BenchError("hipengine.generation_shape.route must be a non-empty string")

    raw_cap = raw.get("route_cap")
    if not isinstance(raw_cap, dict) or raw_cap.get("scope") != "queue_requests":
        raise BenchError("hipengine.generation_shape.route_cap must use queue_requests scope")
    cap_value = raw_cap.get("value")
    if cap_value is not None:
        cap_value = _shape_int(cap_value, label="route_cap.value", minimum=1)
    if not isinstance(raw_cap.get("applied"), bool):
        raise BenchError("hipengine.generation_shape.route_cap.applied must be a bool")
    route_cap = {
        "scope": "queue_requests",
        "value": cap_value,
        "applied": raw_cap["applied"],
    }

    raw_queue = raw.get("queue_group")
    if not isinstance(raw_queue, dict):
        raise BenchError("hipengine.generation_shape.queue_group must be an object")
    queue_id = raw_queue.get("id")
    if not isinstance(queue_id, str) or not queue_id.strip():
        raise BenchError("hipengine.generation_shape.queue_group.id must be non-empty")
    request_count = _shape_int(
        raw_queue.get("request_count"),
        label="queue_group.request_count",
        minimum=1,
    )
    prompt_rows = _shape_int(
        raw_queue.get("prompt_rows"),
        label="queue_group.prompt_rows",
        minimum=1,
    )
    item_index = _shape_int(
        raw_queue.get("item_index"),
        label="queue_group.item_index",
    )
    item_prompt_offset = _shape_int(
        raw_queue.get("item_prompt_offset"),
        label="queue_group.item_prompt_offset",
    )
    item_prompt_rows = _shape_int(
        raw_queue.get("item_prompt_rows"),
        label="queue_group.item_prompt_rows",
        minimum=1,
    )
    if item_index >= request_count:
        raise BenchError("queue_group.item_index must be smaller than request_count")
    if item_prompt_offset + item_prompt_rows > prompt_rows:
        raise BenchError("queue_group item prompt slice exceeds prompt_rows")

    raw_backend_groups = raw.get("backend_groups")
    if not isinstance(raw_backend_groups, list) or not raw_backend_groups:
        raise BenchError("hipengine.generation_shape.backend_groups must be non-empty")
    backend_groups: list[dict[str, Any]] = []
    for group_index, raw_group in enumerate(raw_backend_groups):
        if not isinstance(raw_group, dict):
            raise BenchError(f"backend_groups[{group_index}] must be an object")
        group_id = raw_group.get("id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise BenchError(f"backend_groups[{group_index}].id must be non-empty")
        call_index = _shape_int(
            raw_group.get("call_index"),
            label=f"backend_groups[{group_index}].call_index",
        )
        prompt_offset = _shape_int(
            raw_group.get("prompt_offset"),
            label=f"backend_groups[{group_index}].prompt_offset",
        )
        input_rows = _shape_int(
            raw_group.get("input_rows"),
            label=f"backend_groups[{group_index}].input_rows",
            minimum=1,
        )
        raw_actual_rows = raw_group.get("actual_group_rows")
        if not isinstance(raw_actual_rows, list) or not raw_actual_rows:
            raise BenchError(f"backend_groups[{group_index}].actual_group_rows must be non-empty")
        actual_rows = [
            _shape_int(
                value,
                label=f"backend_groups[{group_index}].actual_group_rows",
                minimum=1,
            )
            for value in raw_actual_rows
        ]
        if sum(actual_rows) != input_rows:
            raise BenchError(f"backend_groups[{group_index}] actual rows must sum to input_rows")
        max_actual_rows = _shape_int(
            raw_group.get("max_actual_group_rows"),
            label=f"backend_groups[{group_index}].max_actual_group_rows",
            minimum=1,
        )
        if max_actual_rows != max(actual_rows):
            raise BenchError(f"backend_groups[{group_index}] max_actual_group_rows is inconsistent")
        verifier_rows = _shape_int(
            raw_group.get("verifier_rows"),
            label=f"backend_groups[{group_index}].verifier_rows",
        )
        backend_groups.append(
            {
                "id": group_id,
                "call_index": call_index,
                "prompt_offset": prompt_offset,
                "input_rows": input_rows,
                "actual_group_rows": actual_rows,
                "max_actual_group_rows": max_actual_rows,
                "verifier_rows": verifier_rows,
            }
        )
    if sorted(group["call_index"] for group in backend_groups) != list(range(len(backend_groups))):
        raise BenchError("backend group call_index values must be contiguous from zero")
    if len({group["id"] for group in backend_groups}) != len(backend_groups):
        raise BenchError("backend group ids must be unique within a queue group")
    backend_cursor = 0
    for group in sorted(backend_groups, key=lambda item: item["call_index"]):
        if group["prompt_offset"] != backend_cursor:
            raise BenchError("backend group prompt slices must be contiguous from zero")
        backend_cursor += group["input_rows"]
    if backend_cursor != prompt_rows:
        raise BenchError("backend group prompt slices must cover queue_group.prompt_rows")
    verifier_rows = _shape_int(raw.get("verifier_rows"), label="generation_shape.verifier_rows")
    if verifier_rows != sum(group["verifier_rows"] for group in backend_groups):
        raise BenchError("generation_shape.verifier_rows must equal the backend-group sum")
    return {
        "schema_version": 1,
        "route": route,
        "route_cap": route_cap,
        "queue_group": {
            "id": queue_id,
            "request_count": request_count,
            "prompt_rows": prompt_rows,
            "item_index": item_index,
            "item_prompt_offset": item_prompt_offset,
            "item_prompt_rows": item_prompt_rows,
        },
        "backend_groups": backend_groups,
        "verifier_rows": verifier_rows,
    }


def aggregate_generation_shapes(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    shapes = [row.get("generation_shape") for row in results]
    present = [shape for shape in shapes if isinstance(shape, dict)]
    if not present:
        return None
    if len(present) != len(results):
        raise BenchError("generation_shape is missing from part of a shaped server run")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for shape in present:
        queue = shape["queue_group"]
        grouped.setdefault(str(queue["id"]), []).append(shape)

    summaries: list[dict[str, Any]] = []
    backend_group_rows: list[int] = []
    route_cap_values: set[int] = set()
    for queue_id, group_shapes in sorted(grouped.items()):
        first = group_shapes[0]
        first_queue = first["queue_group"]
        request_count = int(first_queue["request_count"])
        prompt_rows = int(first_queue["prompt_rows"])
        invariant = {
            "route": first["route"],
            "route_cap": first["route_cap"],
            "backend_groups": first["backend_groups"],
            "verifier_rows": first["verifier_rows"],
        }
        for shape in group_shapes[1:]:
            queue = shape["queue_group"]
            if int(queue["request_count"]) != request_count or int(queue["prompt_rows"]) != prompt_rows:
                raise BenchError(f"queue group {queue_id!r} has inconsistent counts")
            candidate = {
                "route": shape["route"],
                "route_cap": shape["route_cap"],
                "backend_groups": shape["backend_groups"],
                "verifier_rows": shape["verifier_rows"],
            }
            if candidate != invariant:
                raise BenchError(f"queue group {queue_id!r} has inconsistent backend shape")
        if len(group_shapes) != request_count:
            raise BenchError(
                f"queue group {queue_id!r} expected {request_count} response items; found {len(group_shapes)}"
            )
        item_indices = sorted(int(shape["queue_group"]["item_index"]) for shape in group_shapes)
        if item_indices != list(range(request_count)):
            raise BenchError(f"queue group {queue_id!r} item indices are incomplete or duplicated")
        slices = sorted(
            (
                int(shape["queue_group"]["item_prompt_offset"]),
                int(shape["queue_group"]["item_prompt_rows"]),
            )
            for shape in group_shapes
        )
        cursor = 0
        for offset, rows in slices:
            if offset != cursor:
                raise BenchError(f"queue group {queue_id!r} prompt slices are not contiguous")
            cursor += rows
        if cursor != prompt_rows:
            raise BenchError(f"queue group {queue_id!r} prompt slices do not cover prompt_rows")
        cap_value = first["route_cap"]["value"]
        if cap_value is not None:
            route_cap_values.add(int(cap_value))
        for backend_group in first["backend_groups"]:
            backend_group_rows.extend(int(rows) for rows in backend_group["actual_group_rows"])
        summaries.append(
            {
                "id": queue_id,
                "route": first["route"],
                "route_cap": first["route_cap"],
                "request_count": request_count,
                "prompt_rows": prompt_rows,
                "backend_groups": first["backend_groups"],
                "verifier_rows": int(first["verifier_rows"]),
            }
        )
    return {
        "queue_group_count": len(summaries),
        "queue_group_request_counts": [row["request_count"] for row in summaries],
        "queue_group_prompt_rows": [row["prompt_rows"] for row in summaries],
        "route_cap_values": sorted(route_cap_values),
        "backend_group_rows": backend_group_rows,
        "max_backend_group_rows": max(backend_group_rows),
        "verifier_rows_total": sum(row["verifier_rows"] for row in summaries),
        "queue_groups": summaries,
    }


def backend_timing_records(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize per-choice timing payloads without losing batch ownership."""

    records: list[dict[str, Any]] = []
    for choice_index, payload in enumerate(payloads):
        timing = numeric_mapping(payload.get("timing"))
        if not timing:
            continue
        scope_explicit = "timing_scope" in payload
        scope = payload.get("timing_scope", "choice")
        if not isinstance(scope, str) or scope not in {"choice", "batch", "request", "client"}:
            raise BenchError(f"choice {choice_index} has invalid timing_scope {scope!r}")
        raw_group_rows = payload.get("group_rows", 1)
        if isinstance(raw_group_rows, bool) or not isinstance(raw_group_rows, int) or raw_group_rows <= 0:
            raise BenchError(f"choice {choice_index} has invalid group_rows {raw_group_rows!r}")
        raw_owner = payload.get("timing_owner", True)
        if not isinstance(raw_owner, bool):
            raise BenchError(f"choice {choice_index} has invalid timing_owner {raw_owner!r}")
        raw_batch_id = payload.get("batch_id")
        batch_id = None if raw_batch_id is None else str(raw_batch_id).strip()
        if raw_batch_id is not None and not batch_id:
            raise BenchError(f"choice {choice_index} has an empty batch_id")
        if scope == "batch":
            if not batch_id:
                raise BenchError(f"choice {choice_index} has batch timing without batch_id")
            if scope_explicit and "timing_owner" not in payload:
                raise BenchError(f"choice {choice_index} has batch timing without timing_owner")
            if scope_explicit and "group_rows" not in payload:
                raise BenchError(f"choice {choice_index} has batch timing without group_rows")

        record: dict[str, Any] = {
            "choice_index": choice_index,
            "timing": timing,
            "timing_scope": scope,
            "group_rows": int(raw_group_rows),
            "timing_owner": raw_owner,
        }
        if batch_id is not None:
            record["batch_id"] = batch_id
        if not scope_explicit:
            record["legacy_scope_defaulted"] = True
        records.append(record)
    return records


def selected_timing_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select locally owned records; cross-response batch validation is aggregate's job."""

    selected: list[dict[str, Any]] = []
    seen_owned_batch_ids: set[str] = set()
    for record in records:
        if record.get("timing_scope") != "batch":
            selected.append(record)
            continue
        if not record.get("timing_owner"):
            continue
        batch_id = str(record["batch_id"])
        if batch_id in seen_owned_batch_ids:
            raise BenchError(f"multiple timing owners for batch_id {batch_id!r} in one response")
        seen_owned_batch_ids.add(batch_id)
        selected.append(record)
    return selected


def merged_timing(records: list[dict[str, Any]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for record in records:
        timing = record.get("timing")
        if not isinstance(timing, dict):
            continue
        for key, value in timing.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            merged[str(key)] = merged.get(str(key), 0.0) + float(value)
    return merged


def record_from_response(name: str, response: dict[str, Any], wall_s: float) -> dict[str, Any]:
    usage = response.get("usage") or {}
    timings = response.get("timings") or {}
    if not isinstance(usage, dict):
        usage = {}
    if not isinstance(timings, dict):
        timings = {}

    token_accounting = exact_token_accounting(response)
    exact_generated = None if token_accounting is None else token_accounting["total_generated_tokens"]
    predicted_n = first_number(exact_generated, usage.get("completion_tokens"), timings.get("predicted_n"))
    predicted_per_second = (
        float(exact_generated) / wall_s
        if exact_generated is not None and wall_s > 0
        else first_number(timings.get("predicted_per_second"))
    )
    if predicted_per_second is None and predicted_n is not None and wall_s > 0:
        predicted_per_second = float(predicted_n) / wall_s
    if predicted_per_second is None:
        predicted_per_second = 0.0

    draft_n = first_number(timings.get("draft_n"), 0) or 0
    draft_n_accepted = first_number(timings.get("draft_n_accepted"), 0) or 0

    record = {
        "name": name,
        "wall_s": round(wall_s, 3),
        "predicted_n": int(predicted_n) if predicted_n is not None else 0,
        "predicted_per_second": round(float(predicted_per_second), 2),
        "draft_n": int(draft_n),
        "draft_n_accepted": int(draft_n_accepted),
    }
    if token_accounting is not None:
        record.update(token_accounting)
    generation_shape = generation_shape_from_response(response)
    if generation_shape is not None:
        record["generation_shape"] = generation_shape
    hipengine_payloads = choice_hipengine_payloads(response)
    if hipengine_payloads:
        record["hipengine"] = hipengine_payloads
        timing_records = backend_timing_records(hipengine_payloads)
        if timing_records:
            record["backend_timing_records"] = timing_records
        timing = merged_timing(selected_timing_records(timing_records))
        if timing:
            record["backend_timing_ms"] = {key: round(value, 3) for key, value in timing.items()}
            if record["draft_n"] == 0 and "mtp_generated_draft_tokens" in timing:
                record["draft_n"] = int(timing["mtp_generated_draft_tokens"])
            if record["draft_n_accepted"] == 0 and "mtp_accepted_draft_tokens" in timing:
                record["draft_n_accepted"] = int(timing["mtp_accepted_draft_tokens"])
        decode_state = hipengine_payloads[0].get("decode_state")
        if isinstance(decode_state, dict):
            record["backend_decode_state"] = {
                key: decode_state[key]
                for key in (
                    "execution_path",
                    "serial_decode_fallback",
                    "generated_tokens",
                    "prompt_tokens",
                    "sampler_mode",
                )
                if key in decode_state
            }
            backend_generated = first_number(decode_state.get("generated_tokens"))
            if backend_generated is not None:
                record["backend_generated_tokens"] = int(backend_generated)
                record["backend_generated_per_second"] = round(
                    float(backend_generated) / wall_s,
                    2,
                ) if wall_s > 0 else 0.0
            backend_prompt = first_number(decode_state.get("prompt_tokens"))
            if backend_prompt is not None:
                record["backend_prompt_tokens"] = int(backend_prompt)
    if exact_generated is not None:
        record["backend_generated_tokens"] = int(exact_generated)
        record["backend_generated_per_second"] = (
            round(float(exact_generated) / wall_s, 2) if wall_s > 0 else 0.0
        )
    record["accept_rate"] = (
        round(record["draft_n_accepted"] / record["draft_n"], 4) if record["draft_n"] else None
    )
    return record


def format_result_line(record: dict[str, Any]) -> str:
    accept_rate = f"{record['accept_rate']:.3f}" if record["accept_rate"] is not None else "n/a"
    backend_generated = record.get("backend_generated_tokens")
    backend_tok_s = record.get("backend_generated_per_second")
    backend_part = (
        f" gen={int(backend_generated):>4} gen_tok/s={float(backend_tok_s):.1f}"
        if isinstance(backend_generated, int) and isinstance(backend_tok_s, (int, float))
        else ""
    )
    return (
        f"  {record['name']:<18} pred={record['predicted_n']:>4} "
        f"draft={record['draft_n']:>4} acc={record['draft_n_accepted']:>4} "
        f"rate={accept_rate} tok/s={record['predicted_per_second']:.1f}{backend_part}"
    )


def aggregate(
    results: list[dict[str, Any]],
    *,
    client_wall_s: float | None = None,
    concurrency: int = 1,
) -> dict[str, Any]:
    total_draft = sum(int(x.get("draft_n") or 0) for x in results)
    total_accepted = sum(int(x.get("draft_n_accepted") or 0) for x in results)
    total_predicted = sum(int(x.get("predicted_n") or 0) for x in results)
    total_backend_generated = sum(int(x.get("backend_generated_tokens") or 0) for x in results)
    request_wall = sum(float(x.get("wall_s") or 0.0) for x in results)
    aggregate_wall = float(client_wall_s) if client_wall_s is not None else request_wall
    timing_records: list[dict[str, Any]] = []
    for row in results:
        row_records = row.get("backend_timing_records")
        if isinstance(row_records, list):
            timing_records.extend(record for record in row_records if isinstance(record, dict))
            continue
        timing = numeric_mapping(row.get("backend_timing_ms"))
        if timing:
            timing_records.append(
                {
                    "timing": timing,
                    "timing_scope": "choice",
                    "group_rows": 1,
                    "timing_owner": True,
                    "legacy_scope_defaulted": True,
                }
            )

    selected_records: list[dict[str, Any]] = []
    batch_records: dict[str, list[dict[str, Any]]] = {}
    choice_payloads_counted = 0
    non_owner_copies_ignored = 0
    for timing_record in timing_records:
        if timing_record.get("timing_scope") != "batch":
            selected_records.append(timing_record)
            choice_payloads_counted += 1
            continue
        batch_id = str(timing_record.get("batch_id") or "").strip()
        if not batch_id:
            raise BenchError("batch timing record is missing batch_id")
        batch_records.setdefault(batch_id, []).append(timing_record)

    for batch_id, records in sorted(batch_records.items()):
        group_rows = {int(record.get("group_rows", 0)) for record in records}
        if len(group_rows) != 1 or next(iter(group_rows)) <= 0:
            raise BenchError(f"inconsistent group_rows for batch_id {batch_id!r}")
        owners = [record for record in records if record.get("timing_owner") is True]
        if len(owners) != 1:
            raise BenchError(
                f"batch_id {batch_id!r} requires exactly one timing owner; found {len(owners)}"
            )
        selected_records.append(owners[0])
        non_owner_copies_ignored += len(records) - 1

    backend_timing_totals: dict[str, float] = {}
    backend_timing_counts: dict[str, int] = {}
    for timing_record in selected_records:
        timing = numeric_mapping(timing_record.get("timing"))
        for key, raw_value in timing.items():
            backend_timing_totals[key] = backend_timing_totals.get(key, 0.0) + raw_value
            backend_timing_counts[key] = backend_timing_counts.get(key, 0) + 1
    payload = {
        "n_requests": len(results),
        "concurrency": int(concurrency),
        "total_predicted": total_predicted,
        "total_draft": total_draft,
        "total_draft_accepted": total_accepted,
        "aggregate_accept_rate": round(total_accepted / total_draft, 4) if total_draft else None,
        "wall_s_total": round(aggregate_wall, 2),
        "request_wall_s_total": round(request_wall, 2),
        "aggregate_predicted_per_second": round(total_predicted / aggregate_wall, 2) if aggregate_wall > 0 else None,
    }
    if results and all("total_generated_tokens" in row for row in results):
        total_generated = sum(int(row["total_generated_tokens"]) for row in results)
        payload["total_generated_tokens"] = total_generated
        payload["aggregate_generated_per_second"] = (
            round(total_generated / aggregate_wall, 2) if aggregate_wall > 0 else None
        )
    if total_backend_generated > 0:
        payload["total_backend_generated"] = total_backend_generated
        payload["aggregate_backend_generated_per_second"] = (
            round(total_backend_generated / aggregate_wall, 2)
            if aggregate_wall > 0
            else None
        )
    if backend_timing_totals:
        payload["backend_timing_totals_ms"] = {
            key: round(value, 3)
            for key, value in sorted(backend_timing_totals.items())
        }
        payload["backend_timing_mean_ms"] = {
            key: round(backend_timing_totals[key] / max(1, backend_timing_counts[key]), 3)
            for key in sorted(backend_timing_totals)
        }
        payload["backend_timing_dedup"] = {
            "batch_ids": sorted(batch_records),
            "batch_payloads_counted": len(batch_records),
            "choice_payloads_counted": choice_payloads_counted,
            "non_owner_copies_ignored": non_owner_copies_ignored,
        }
    generation_shape = aggregate_generation_shapes(results)
    if generation_shape is not None:
        payload["generation_shape"] = generation_shape
    return payload


def iter_batches(items: list[dict[str, str]], batch_size: int) -> list[list[tuple[int, dict[str, str]]]]:
    indexed = list(enumerate(items))
    return [indexed[i : i + batch_size] for i in range(0, len(indexed), batch_size)]


def run_prompt_request(
    prompt: dict[str, str],
    args: argparse.Namespace,
    url: str,
    *,
    barrier: threading.Barrier | None = None,
) -> dict[str, Any]:
    payload = make_payload(prompt["prompt"], args)
    if barrier is not None:
        barrier.wait()
    start = time.perf_counter()
    response = post_json(url, payload, timeout=args.timeout, api_key=args.api_key)
    wall_s = time.perf_counter() - start
    record = record_from_response(prompt["name"], response, wall_s)
    for field in ("category", "split"):
        if field in prompt:
            record[field] = prompt[field]
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    suite = load_prompt_suite(args.prompts_file)
    prompts = select_prompts(suite, names_csv=args.prompt_names, limit=args.limit)
    base_url = args.url.rstrip("/")
    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    url = f"{base_url}{endpoint}"

    if args.concurrency < 1:
        raise BenchError("--concurrency must be >= 1")
    if args.artifact_warmups < 0:
        raise BenchError("--artifact-warmups must be >= 0")
    if args.artifact_repetitions < 1:
        raise BenchError("--artifact-repetitions must be >= 1")

    out: dict[str, Any] = {
        "results": [],
        "concurrency": int(args.concurrency),
        "prompt_suite": prompt_suite_metadata(args.prompts_file, suite, prompts),
    }
    if args.print_payload:
        for prompt in prompts:
            payload = make_payload(prompt["prompt"], args)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return out

    out["provenance"] = server_artifact_provenance(args)

    client_wall_s = 0.0
    for batch in iter_batches(prompts, args.concurrency):
        batch_size = len(batch)
        if batch_size == 1:
            batch_start = time.perf_counter()
            _, prompt = batch[0]
            record = run_prompt_request(prompt, args, url)
            client_wall_s += time.perf_counter() - batch_start
            out["results"].append(record)
            print(format_result_line(record))
            continue

        barrier = threading.Barrier(batch_size + 1)
        batch_start = time.perf_counter()
        records_by_index: list[tuple[int, dict[str, Any]]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = [
                (index, pool.submit(run_prompt_request, prompt, args, url, barrier=barrier))
                for index, prompt in batch
            ]
            barrier.wait()
            for index, future in futures:
                records_by_index.append((index, future.result()))
        client_wall_s += time.perf_counter() - batch_start
        for _, record in sorted(records_by_index, key=lambda item: item[0]):
            out["results"].append(record)
            print(format_result_line(record))

    out["aggregate"] = aggregate(out["results"], client_wall_s=client_wall_s, concurrency=args.concurrency)
    print("\nAggregate:", json.dumps(out["aggregate"], indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print("Wrote", args.out)
    return out


def default_hipengine_out_path() -> Path:
    return (
        REPO_ROOT
        / "benchmarks"
        / "results"
        / f"{date.today().isoformat()}-hipengine-mtp-llamacpp-prompt-suite-economics.json"
    )


def hipengine_output_path(args: argparse.Namespace) -> Path:
    return args.out if args.out is not None else default_hipengine_out_path()


def build_hipengine_current_command(args: argparse.Namespace) -> list[str]:
    """Build the existing hipEngine prompt-suite economics command.

    This deliberately shells out to ``scripts/mtp_prompt_suite_economics.py`` so
    the JSON remains compatible with the artifacts we already use for MTP
    verifier economics, while this tool owns the shared llama.cpp prompt-suite
    entry point.
    """

    cmd = [
        sys.executable,
        "scripts/mtp_prompt_suite_economics.py",
        "--model",
        str(args.engine_model),
        "--prompts-file",
        str(args.prompts_file),
        "--prompt-render",
        str(args.prompt_render),
        "--decode-tokens",
        str(args.max_tokens),
        "--candidate-budgets",
        str(args.candidate_budgets),
        "--runs",
        str(args.runs),
        "--proposal-impl",
        str(args.proposal_impl),
        "--backend",
        str(args.backend),
        "--hip-arch",
        str(args.hip_arch),
        "--chain-attn-mode",
        str(args.chain_attn_mode),
        "--graph-mode",
        str(args.graph_mode),
        "--raw-root",
        str(args.raw_root),
        "--out",
        str(hipengine_output_path(args)),
    ]
    if args.prompt_names:
        cmd += ["--prompt-names", str(args.prompt_names)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.small_batch_decode_threshold is not None:
        cmd += ["--small-batch-decode-threshold", str(args.small_batch_decode_threshold)]
    if args.verify_gpu_accept is not None:
        cmd += ["--verify-gpu-accept", str(args.verify_gpu_accept)]
    if args.llama_target_cycle_cost is not None:
        cmd += ["--llama-target-cycle-cost", str(args.llama_target_cycle_cost)]
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def quote_command(cmd: list[str]) -> str:
    return " ".join(_shell_quote(part) for part in cmd)


def _shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "@%_+=:,./-" for ch in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def run_hipengine_current(args: argparse.Namespace) -> None:
    cmd = build_hipengine_current_command(args)
    if args.print_command:
        print(quote_command(cmd))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    print("[hipengine-current] running existing prompt-suite economics:")
    print("  " + quote_command(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True)
    if completed.returncode != 0:
        raise BenchError(f"hipEngine current economics exited with status {completed.returncode}")

    out_path = hipengine_output_path(args)
    if not args.dry_run and out_path.exists():
        print_hipengine_summary(out_path)


def print_hipengine_summary(path: Path) -> None:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    aggregate_by_budget = artifact.get("aggregate_by_budget") or {}
    if not aggregate_by_budget:
        return
    print("\nHipEngine current economics summary:")
    for budget, row in sorted(aggregate_by_budget.items(), key=lambda item: int(item[0])):
        prompts = row.get("prompts")
        exact = row.get("all_exact_ar_match")
        cycle = row.get("cycle_cost_ar_tokens_mean_across_prompts_mean")
        visible = row.get("avg_visible_tokens_per_cycle_mean_across_prompts_mean")
        observed = row.get("observed_cycle_speedup_vs_ar_mean_across_prompts_mean")
        actual = row.get("actual_decode_speedup_vs_ar_mean_across_prompts_mean")
        accept = row.get("acceptance_rate_mean_across_prompts_mean")
        print(
            f"  B={budget:<3} prompts={prompts} exact={exact} "
            f"cycle_cost={format_optional(cycle)} AR-tok "
            f"visible/cycle={format_optional(visible)} "
            f"cycle_speedup={format_optional(observed)}x "
            f"actual_mtp/ar={format_optional(actual)}x "
            f"accept={format_optional(accept)}"
        )


def format_optional(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def diff(path_a: Path, path_b: Path) -> None:
    data_a = json.loads(path_a.read_text(encoding="utf-8"))
    data_b = json.loads(path_b.read_text(encoding="utf-8"))
    print(f"{'metric':<24} {'A':>14} {'B':>14} {'delta':>10}")
    for key in (
        "aggregate_accept_rate",
        "total_predicted",
        "total_draft",
        "total_draft_accepted",
        "wall_s_total",
    ):
        val_a = data_a["aggregate"].get(key)
        val_b = data_b["aggregate"].get(key)
        if val_a is None or val_b is None:
            print(f"{key:<24} {str(val_a):>14} {str(val_b):>14}")
            continue
        delta = val_b - val_a
        delta_str = f"{delta:>+10.4f}" if isinstance(delta, float) else f"{delta:>+10}"
        print(f"{key:<24} {val_a:>14} {val_b:>14} {delta_str}")

    by_a = {row["name"]: row for row in data_a.get("results", [])}
    print("\n{:<20} {:>8} {:>8} {:>8}".format("prompt", "A", "B", "delta"))
    for row_b in data_b.get("results", []):
        row_a = by_a.get(row_b["name"]) or {}
        accept_a = row_a.get("accept_rate") or 0
        accept_b = row_b.get("accept_rate") or 0
        print(f"{row_b['name']:<20} {accept_a:>8.3f} {accept_b:>8.3f} {accept_b - accept_a:>+8.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("server", "hipengine-current"),
        default="server",
        help="server = llama.cpp/OpenAI-compatible requests; hipengine-current = existing verifier economics wrapper",
    )

    # Shared prompt/output controls.
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-names", help="comma-separated prompt names to run")
    parser.add_argument("--limit", type=int, help="run only the first N selected prompts")
    parser.add_argument("--list-prompts", action="store_true", help="list selected prompt names and exit")
    parser.add_argument("--max-tokens", type=int, default=192, help="server max_tokens / hipEngine decode_tokens")
    parser.add_argument("--out", type=Path, help="write JSON results")
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("A", "B"), help="diff two server-mode JSON outputs and exit")

    # llama.cpp / OpenAI-compatible server mode.
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="OpenAI-compatible server base URL")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="completion endpoint path")
    parser.add_argument("--model", default="llama", help="model field sent in each server request")
    parser.add_argument("--seed", type=int, default=42, help="seed field sent in each server request")
    parser.add_argument("--temperature", type=float, default=None, help="optional temperature field; omitted by default for gist parity")
    parser.add_argument("--top-p", type=float, default=None, help="optional top_p field; omitted by default for gist parity")
    parser.add_argument("--ignore-eos", action="store_true", help="send ignore_eos=true")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--cache-prompt", dest="cache_prompt", action="store_true", default=None, help="send cache_prompt=true")
    cache_group.add_argument("--no-cache-prompt", dest="cache_prompt", action="store_false", help="send cache_prompt=false")
    parser.add_argument("--extra-payload", help="JSON object merged into each server request payload")
    parser.add_argument("--api-key", help="Bearer token for servers requiring OpenAI-style auth")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request timeout in seconds")
    parser.add_argument("--concurrency", type=int, default=1, help="server-mode concurrent request count")
    parser.add_argument("--print-payload", action="store_true", help="print server request payloads instead of posting")
    parser.add_argument("--artifact-model-path", type=Path, help="local served-model path used for canonical artifact fingerprinting")
    parser.add_argument("--artifact-model-revision", help="served-model revision when it cannot be inferred from a snapshots/<revision> path")
    parser.add_argument("--artifact-quant", help="resolved server quant recorded in canonical provenance")
    parser.add_argument("--artifact-kv-dtype", help="resolved server KV dtype recorded in canonical provenance")
    parser.add_argument("--artifact-configured-backend", default="auto", help="configured server backend selector")
    parser.add_argument("--artifact-resolved-backend", help="resolved server backend; auto-detected locally when omitted")
    parser.add_argument("--artifact-target-arch", help="resolved target architecture; auto-detected locally when omitted")
    parser.add_argument("--artifact-device-name", help="selected device name; queried from the local HIP runtime when omitted")
    parser.add_argument("--artifact-build-profile", help="build profile name recorded in canonical provenance")
    parser.add_argument("--artifact-warmups", type=int, default=0, help="discarded server warmup repetitions")
    parser.add_argument("--artifact-repetitions", type=int, default=1, help="measured server repetitions")

    # hipEngine current verifier-economics mode.  Defaults are the W7900/gfx1100
    # path used by current local M12 artifacts.
    parser.add_argument("--engine-model", type=Path, default=DEFAULT_ENGINE_MODEL, help="hipEngine model path for hipengine-current mode")
    parser.add_argument("--candidate-budgets", default="3", help="comma-separated candidate budgets for hipengine-current mode")
    parser.add_argument("--runs", type=int, default=1, help="runs per prompt/budget for hipengine-current mode")
    parser.add_argument(
        "--prompt-render",
        choices=("raw", "qwen_chat_thinking_off", "qwen_chat_thinking_on"),
        default="raw",
        help="hipengine-current prompt rendering before tokenization",
    )
    parser.add_argument(
        "--proposal-impl",
        choices=("persistent_device", "persistent_device_b1", "reload_d2h"),
        default="persistent_device",
    )
    parser.add_argument("--backend", default="hip_gfx1100", help="hipEngine backend for hipengine-current mode")
    parser.add_argument("--hip-arch", default="gfx1100", help="HIP arch for hipengine-current mode")
    parser.add_argument("--chain-attn-mode", choices=("c1_loop", "batched", "decode_batched"), default="batched")
    parser.add_argument("--graph-mode", choices=("off", "auto", "validate"), default="off")
    parser.add_argument("--small-batch-decode-threshold", type=int, default=7)
    parser.add_argument("--verify-gpu-accept", default=None)
    parser.add_argument("--llama-target-cycle-cost", type=float, default=2.0)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_HIPENGINE_RAW_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="pass --dry-run to hipengine-current mode")
    parser.add_argument("--print-command", action="store_true", help="print hipengine-current command and exit")

    parser.epilog = (
        "Examples:\n"
        "  python3 scripts/mtp-bench.py --url http://127.0.0.1:8080 --out llama-mtp.json\n"
        "  python3 scripts/mtp-bench.py --diff llama-master.json llama-pr23287.json\n"
        "  python3 scripts/mtp-bench.py --temperature 0 --no-cache-prompt\n"
        "  python3 scripts/mtp-bench.py --mode hipengine-current --candidate-budgets 3 --runs 3 --out hipengine-current.json\n\n"
        f"Source gist: {SOURCE_GIST}\nRaw revision: {SOURCE_RAW}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.diff:
            diff(*args.diff)
            return 0
        suite = load_prompt_suite(args.prompts_file)
        prompts = select_prompts(suite, names_csv=args.prompt_names, limit=args.limit)
        if args.list_prompts:
            for prompt in prompts:
                print(f"{prompt['name']}\t{len(prompt['prompt'])} chars")
            return 0
        if args.mode == "hipengine-current":
            run_hipengine_current(args)
        else:
            run(args)
        return 0
    except BenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
