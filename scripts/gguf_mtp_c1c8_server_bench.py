#!/usr/bin/env python3
"""Compare AR and explicitly enabled MTP over the canonical C1-C8 prompt suite."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

from fastapi.testclient import TestClient

from hipengine import LLM
from hipengine.benchmark.provenance import collect_model_identity
from hipengine.core.memory import memory_stats
from hipengine.server.api import ServerConfig, create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
FULL_PROMPT_IDS = (
    "code_merge_intervals",
    "code_topological_sort",
    "code_lru_cache",
    "code_markdown_table",
    "general_en_plan",
    "general_en_explain",
    "general_ja_plan",
    "general_ja_explain",
    "mixed_ja_en_translate",
    "mixed_ja_en_review",
)
ARMS = ("ar", "mtp")
_CORRECTNESS_CONTRACTS = ("ar_exact", "mtp_self_exact")


def _render_messages(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt messages must be a non-empty list")
    rendered: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"prompt message {index} must be a mapping")
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"prompt message {index} has unsupported role {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"prompt message {index} content must be text")
        rendered.append(
            f"<|im_start|>{'system' if role == 'developer' else role}\n"
            f"{content}<|im_end|>"
        )
    rendered.append("<|im_start|>assistant\n")
    return "\n".join(rendered)


def load_prompt_suite(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        prompt_id = payload.get("id")
        category = payload.get("category")
        if not isinstance(prompt_id, str) or not isinstance(category, str):
            raise ValueError(f"prompt line {line_number} has invalid id/category")
        rendered = _render_messages(payload.get("messages"))
        rows.append(
            {
                "id": prompt_id,
                "category": category,
                "heldout": prompt_id.endswith(("markdown_table", "_explain", "_review")),
                "rendered_prompt": rendered,
                "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            }
        )
    if tuple(row["id"] for row in rows) != FULL_PROMPT_IDS:
        raise ValueError("canonical MTP prompt IDs/order are incomplete")
    return tuple(rows)


def _parse_widths(raw: str) -> tuple[int, ...]:
    widths = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not widths or len(set(widths)) != len(widths) or any(width not in range(1, 9) for width in widths):
        raise argparse.ArgumentTypeError("widths must be a unique subset of 1..8")
    return widths


def _parse_expected_mtp_widths(raw: str) -> tuple[int, ...]:
    normalized = str(raw).strip().lower()
    if normalized in {"", "none", "k0"}:
        return ()
    return _parse_widths(normalized)


def _generated_ids(payload: Mapping[str, Any]) -> list[int]:
    root = payload.get("hipengine")
    root = root if isinstance(root, Mapping) else {}
    accounting = root.get("token_accounting")
    accounting = accounting if isinstance(accounting, Mapping) else {}
    rows = accounting.get("choice_generated_token_ids")
    if isinstance(rows, Sequence) and rows and not isinstance(rows, (str, bytes, bytearray)):
        candidate = rows[0]
    else:
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, Sequence) and choices else {}
        choice = choice if isinstance(choice, Mapping) else {}
        details = choice.get("hipengine")
        details = details if isinstance(details, Mapping) else {}
        candidate = details.get("generated_token_ids")
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes, bytearray)):
        raise ValueError("response omitted authoritative generated token IDs")
    if any(not isinstance(token, int) or isinstance(token, bool) for token in candidate):
        raise ValueError("generated token IDs must be integers")
    return [int(token) for token in candidate]


def _response_mtp(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    root = payload.get("hipengine")
    root = root if isinstance(root, Mapping) else {}
    shape = root.get("generation_shape")
    shape = shape if isinstance(shape, Mapping) else {}
    summary = root.get("speculative_mtp")
    return str(shape.get("route") or "default"), dict(summary) if isinstance(summary, Mapping) else {}


def _mtp_engaged(route: str, summary: Mapping[str, Any]) -> bool:
    """Use committed-cycle truth, not the frontend's pre-cycle route label."""

    del route
    return bool(
        summary.get("used") is True
        and int(summary.get("draft_tokens", 0) or 0) > 0
        and int(summary.get("draft_cycles", 0) or 0) > 0
    )


def _mtp_budget_conformed(summary: Mapping[str, Any], *, budget: int) -> bool:
    generated = int(summary.get("draft_tokens", 0) or 0)
    cycles = int(summary.get("draft_cycles", 0) or 0)
    return bool(cycles > 0 and generated > 0 and generated <= int(budget) * cycles)


def _cell_correctness(
    ar_ids: Sequence[Sequence[int]],
    mtp_ids: Sequence[Sequence[int]],
    *,
    contract: str,
) -> dict[str, bool]:
    if contract not in _CORRECTNESS_CONTRACTS:
        raise ValueError(f"unsupported correctness contract: {contract!r}")
    ar_self_exact = bool(ar_ids and all(list(row) == list(ar_ids[0]) for row in ar_ids))
    mtp_self_exact = bool(mtp_ids and all(list(row) == list(mtp_ids[0]) for row in mtp_ids))
    ar_mtp_equal = bool(ar_self_exact and mtp_self_exact and list(mtp_ids[0]) == list(ar_ids[0]))
    return {
        "ar_self_exact": ar_self_exact,
        "mtp_self_exact": mtp_self_exact,
        "ar_mtp_equal": ar_mtp_equal,
        "passed": ar_mtp_equal if contract == "ar_exact" else ar_self_exact and mtp_self_exact,
    }


def _backend_mtp_telemetry(llm: LLM) -> dict[str, Any]:
    generator = llm._get_text_generator()
    payload = getattr(generator, "last_batch_generation", None)
    return copy.deepcopy(payload) if isinstance(payload, Mapping) else {}


def _resident_observability(llm: LLM, *, recent: int) -> dict[str, Any]:
    pending = [llm._get_text_generator()]
    seen: set[int] = set()
    while pending:
        owner = pending.pop(0)
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        snapshot = getattr(owner, "observability_snapshot", None)
        if callable(snapshot):
            payload = snapshot()
            if not isinstance(payload, Mapping):
                return {}
            result = copy.deepcopy(dict(payload))
            adapter = getattr(owner, "_mtp2_adapter", None)
            cycle_contract = getattr(adapter, "cycle_workspace_contract", None)
            if adapter is not None:
                result["mtp2_adapter"] = {
                    "cycle_workspace": (
                        copy.deepcopy(cycle_contract())
                        if callable(cycle_contract)
                        else None
                    ),
                    "active_states": len(getattr(adapter, "_states", {})),
                    "provider_groups": len(getattr(adapter, "_provider_groups", {})),
                    "prompt_streaming_sinks": len(
                        getattr(adapter, "_prompt_streaming_sinks", {})
                    ),
                    "batch_accept_workspace_allocated": bool(
                        getattr(adapter, "_batch_accept_workspace", None) is not None
                    ),
                }
            routes = result.get("routes")
            if isinstance(routes, Mapping):
                compact_routes = copy.deepcopy(dict(routes))
                completed = compact_routes.get("recent_completed")
                if isinstance(completed, Sequence) and not isinstance(
                    completed, (str, bytes, bytearray)
                ):
                    compact_routes["recent_completed"] = list(completed)[-int(recent) :]
                result["routes"] = compact_routes
            return result
        pending.extend(
            getattr(owner, name, None)
            for name in ("_driver", "_runner", "_inner")
        )
    return {}


def _memory_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(key): int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        for key in (
            "current_allocated_bytes",
            "peak_allocated_bytes",
            "total_allocated_bytes",
            "total_freed_bytes",
            "active_allocations",
            "peak_allocations",
        )
    }


def _backend_mtp_engaged(payload: Mapping[str, Any], *, width: int) -> bool:
    summary = payload.get("speculative_mtp")
    summary = summary if isinstance(summary, Mapping) else {}
    cycles = int(summary.get("direct_cycles", 0) or 0)
    if cycles <= 0:
        by_request = summary.get("cycles_by_request")
        if isinstance(by_request, Mapping):
            cycles = sum(
                len(value)
                for value in by_request.values()
                if isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
            )
    return bool(
        str(payload.get("path", "")).startswith("gguf_")
        and int(payload.get("batch_size", 0) or 0) == int(width)
        and int(summary.get("total_draft_tokens", 0) or 0) > 0
        and cycles > 0
    )


def _diagnostic_plan(**kwargs: Any) -> dict[str, Any]:
    rows = int(kwargs["realized_group_rows"])
    budget = int(kwargs.get("candidate_budget", 2))
    # M1 protocol: the gfx1151 production physical adapter is qualified one
    # cycle through C8/R32; static grouping intent is advertised at width 8 and
    # clamped per profile by the capability-owned bound.
    diagnostic_max_rows = 8
    admitted = bool(
        1 <= rows <= diagnostic_max_rows
        and budget in {1, 2, 3}
        and kwargs["sampling_mode"] == "greedy_fast"
        and int(kwargs["context_tokens"]) <= 95
        and int(kwargs["output_horizon_tokens"]) == 24
        and kwargs["memory_fit"]
    )
    key = {
        "realized_group_rows": rows,
        "context_tokens": int(kwargs["context_tokens"]),
        "output_horizon_tokens": int(kwargs["output_horizon_tokens"]),
        "candidate_budget": budget,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    reason = "diagnostic_physical_gguf_mtp" if admitted else "diagnostic_scope_miss"
    evidence_key = "gguf-c1-c8-generation2-diagnostic"
    static_key = {
        "candidate_budget": budget,
        "sampling_mode": str(kwargs["sampling_mode"]),
        "context_tokens": int(kwargs["context_tokens"]),
        "output_horizon_tokens": int(kwargs["output_horizon_tokens"]),
        "memory_fit": bool(kwargs["memory_fit"]),
        "max_realized_group_rows": diagnostic_max_rows,
    }
    static_digest = hashlib.sha256(
        json.dumps(static_key, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "plan_fingerprint": f"sha256:{digest}",
        "key": key,
        "admitted": admitted,
        "selected_route": "speculative_mtp" if admitted else "default",
        "selected_candidate_count": budget if admitted else 0,
        "reason": reason,
        "strict_fallback_key": "gguf_target_ar",
        "evidence_key": evidence_key,
        "evidence_fingerprint": f"sha256:{static_digest}",
        "evidence_artifacts": [],
        "automatic_eligible": False,
        "static_eligibility": {
            "state": "speculative_capable" if admitted else "permanent_ar",
            "eligible": admitted,
            "reason": reason,
            "max_candidate_count": budget if admitted else 0,
            "max_realized_group_rows": diagnostic_max_rows if admitted else 0,
            "automatic_eligible": False,
            "strict_fallback_key": "gguf_target_ar",
            "evidence_key": evidence_key,
            "evidence_fingerprint": f"sha256:{static_digest}",
            "evidence_artifacts": [],
        },
    }


def _install_diagnostic_plan(llm: LLM) -> None:
    def resolve(self: LLM, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault(
            "candidate_budget",
            int(getattr(self, "speculative_candidate_budget", 2)),
        )
        return _diagnostic_plan(**kwargs)

    llm.resolve_speculative_mtp_serving_plan = MethodType(resolve, llm)


def _request(
    client: TestClient,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    mtp: bool,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    barrier.wait(timeout=30.0)
    started = time.perf_counter()
    response = client.post(
        "/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
            "top_p": 1.0,
            "speculative_mtp": bool(mtp),
        },
    )
    completed = time.perf_counter()
    payload = response.json()
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {payload}")
    route, summary = _response_mtp(payload)
    return {
        "started": started,
        "completed": completed,
        "wall_seconds": completed - started,
        "generated_ids": _generated_ids(payload),
        "route": route,
        "mtp": summary,
        "usage": payload.get("usage"),
    }


def _run_arm(
    client: TestClient,
    *,
    llm: LLM,
    model: str,
    prompt: str,
    width: int,
    max_tokens: int,
    arm: str,
) -> dict[str, Any]:
    memory_before = memory_stats()
    barrier = threading.Barrier(int(width) + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(width)) as executor:
        futures = [
            executor.submit(
                _request,
                client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                mtp=arm == "mtp",
                barrier=barrier,
            )
            for _ in range(int(width))
        ]
        barrier.wait(timeout=30.0)
        rows = [future.result() for future in futures]
    wall = max(row["completed"] for row in rows) - min(row["started"] for row in rows)
    generated = sum(len(row["generated_ids"]) for row in rows)
    backend_telemetry = _backend_mtp_telemetry(llm)
    resident_observability = _resident_observability(llm, recent=int(width))
    memory_after = memory_stats()
    return {
        "arm": arm,
        "width": int(width),
        "wall_seconds": wall,
        "generated_tokens": generated,
        "tok_s": generated / wall,
        "rows": rows,
        "backend_telemetry": backend_telemetry,
        "resident_observability": resident_observability,
        "tracked_memory_before": memory_before,
        "tracked_memory_after": memory_after,
        "tracked_memory_delta": _memory_delta(memory_before, memory_after),
    }


def _acceptance_cycle_rows(
    cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        mtp = cell.get("mtp")
        mtp = mtp if isinstance(mtp, Mapping) else {}
        resident = mtp.get("resident_observability")
        resident = resident if isinstance(resident, Mapping) else {}
        routes = resident.get("routes")
        routes = routes if isinstance(routes, Mapping) else {}
        completed = routes.get("recent_completed")
        if not isinstance(completed, Sequence) or isinstance(
            completed, (str, bytes, bytearray)
        ):
            continue
        for request in completed:
            if not isinstance(request, Mapping):
                continue
            candidates = request.get("specdec2_mtp2_candidate_counts")
            accepted = request.get("specdec2_mtp2_accepted_counts")
            if candidates is None and accepted is None:
                continue
            if (
                not isinstance(candidates, Sequence)
                or isinstance(candidates, (str, bytes, bytearray))
                or not isinstance(accepted, Sequence)
                or isinstance(accepted, (str, bytes, bytearray))
            ):
                raise ValueError("MTP acceptance telemetry must contain cycle-count sequences")
            if len(candidates) != len(accepted):
                raise ValueError("MTP candidate and accepted cycle counts must have equal length")
            for candidate_count, accepted_count in zip(candidates, accepted, strict=True):
                if (
                    not isinstance(candidate_count, int)
                    or isinstance(candidate_count, bool)
                    or int(candidate_count) < 0
                    or not isinstance(accepted_count, int)
                    or isinstance(accepted_count, bool)
                    or int(accepted_count) < 0
                ):
                    raise ValueError("MTP candidate and accepted cycle counts must be nonnegative integers")
                if int(accepted_count) > int(candidate_count):
                    raise ValueError(
                        f"accepted count {int(accepted_count)} exceeds candidate count {int(candidate_count)}"
                    )
                rows.append(
                    {
                        "width": int(cell["width"]),
                        "category": str(cell["category"]),
                        "heldout": bool(cell["heldout"]),
                        "candidate_count": int(candidate_count),
                        "accepted_count": int(accepted_count),
                    }
                )
    return rows


def _acceptance_rollup(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    proposed = sum(int(row["candidate_count"]) for row in rows)
    accepted = sum(int(row["accepted_count"]) for row in rows)
    max_position = max((int(row["candidate_count"]) for row in rows), default=0)
    positions: list[dict[str, Any]] = []
    for position in range(1, max_position + 1):
        proposed_cycles = sum(
            int(row["candidate_count"]) >= position for row in rows
        )
        accepted_cycles = sum(
            int(row["accepted_count"]) >= position for row in rows
        )
        conditional_opportunities = sum(
            int(row["candidate_count"]) >= position
            and int(row["accepted_count"]) >= position - 1
            for row in rows
        )
        positions.append(
            {
                "position": position,
                "proposed_cycles": proposed_cycles,
                "accepted_cycles": accepted_cycles,
                "position_acceptance": (
                    accepted_cycles / proposed_cycles if proposed_cycles else None
                ),
                "conditional_opportunities": conditional_opportunities,
                "conditional_position_acceptance": (
                    accepted_cycles / conditional_opportunities
                    if conditional_opportunities
                    else None
                ),
            }
        )
    return {
        "cycles": len(rows),
        "proposed_draft_tokens": proposed,
        "accepted_draft_tokens": accepted,
        "draft_acceptance": accepted / proposed if proposed else None,
        "positions": positions,
    }


def _acceptance_scopes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "all": _acceptance_rollup(rows),
        "train": _acceptance_rollup([row for row in rows if not bool(row["heldout"])]),
        "heldout": _acceptance_rollup([row for row in rows if bool(row["heldout"])]),
    }


def _acceptance_categories(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        category: _acceptance_rollup(
            [row for row in rows if str(row["category"]) == category]
        )
        for category in sorted({str(row["category"]) for row in rows})
    }


def summarize_acceptance(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Roll up request-local MTP acceptance without changing its denominator."""

    rows = _acceptance_cycle_rows(cells)
    widths = sorted({int(row["width"]) for row in rows})
    return {
        "denominators": {
            "draft_acceptance": "accepted draft tokens / proposed draft tokens",
            "position_acceptance": "cycles accepting through this position / cycles proposing this position",
            "conditional_position_acceptance": "cycles accepting through this position / cycles accepting the preceding positions while proposing this position",
        },
        "scopes": _acceptance_scopes(rows),
        "categories": _acceptance_categories(rows),
        "by_width": {
            str(width): {
                "scopes": _acceptance_scopes(
                    [row for row in rows if int(row["width"]) == width]
                ),
                "categories": _acceptance_categories(
                    [row for row in rows if int(row["width"]) == width]
                ),
            }
            for width in widths
        },
    }


def summarize(
    cells: Sequence[Mapping[str, Any]],
    *,
    widths: Sequence[int],
    expected_mtp_widths: Sequence[int] | None = None,
) -> dict[str, Any]:
    expected = set(widths if expected_mtp_widths is None else expected_mtp_widths)
    result: dict[str, Any] = {}
    for width in widths:
        selected = [cell for cell in cells if int(cell["width"]) == int(width)]
        arms: dict[str, Any] = {}
        for arm in ARMS:
            rows = [cell[arm] for cell in selected]
            wall = sum(float(row["wall_seconds"]) for row in rows)
            generated = sum(int(row["generated_tokens"]) for row in rows)
            arms[arm] = {
                "wall_seconds": wall,
                "generated_tokens": generated,
                "tok_s": generated / wall,
            }
        exact_cells = sum(bool(cell["exact"]) for cell in selected)
        engaged_cells = sum(bool(cell["mtp_engaged"]) for cell in selected)
        budget_conformed_cells = sum(
            bool(cell.get("mtp_budget_conformed", True)) for cell in selected
        )
        mtp_expected = int(width) in expected
        result[str(width)] = {
            "ar": arms["ar"],
            "mtp": arms["mtp"],
            "mtp_vs_ar_ratio": arms["mtp"]["tok_s"] / arms["ar"]["tok_s"],
            "mtp_vs_ar_percent": 100.0 * (arms["mtp"]["tok_s"] / arms["ar"]["tok_s"] - 1.0),
            "exact_cells": exact_cells,
            "engaged_cells": engaged_cells,
            "budget_conformed_cells": budget_conformed_cells,
            "cells": len(selected),
            "mtp_expected": mtp_expected,
            "route_expectation_passed": bool(
                engaged_cells == len(selected) if mtp_expected else engaged_cells == 0
            ),
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompt_suite(Path(args.prompts).resolve())
    widths = tuple(args.widths)
    resident_capacity = (
        max(widths)
        if args.resident_capacity is None
        else int(args.resident_capacity)
    )
    if resident_capacity < max(widths):
        raise ValueError("resident capacity cannot be smaller than a measured width")
    budget_env = "HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET"
    existing_budget = os.environ.get(budget_env)
    if existing_budget is not None and int(existing_budget) != int(args.candidate_budget):
        raise ValueError(
            f"{budget_env}={existing_budget!r} conflicts with --candidate-budget "
            f"{int(args.candidate_budget)}"
        )
    os.environ[budget_env] = str(int(args.candidate_budget))
    expected_mtp_widths = (
        widths
        if args.expected_mtp_widths is None
        else tuple(args.expected_mtp_widths)
    )
    if not set(expected_mtp_widths).issubset(widths):
        raise ValueError("expected MTP widths must be a subset of measured widths")
    repo_status = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    source_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True
    ).strip()
    if repo_status:
        raise RuntimeError("benchmark requires tracked-clean source")
    if args.generation2_diagnostic:
        import hipengine.kernels.hip_gfx1100 as backend_package

        backend_package.GGUF_SPECDEC2_MTP2_PHYSICAL = True
    started = time.perf_counter()
    initial_memory = memory_stats()
    llm = LLM(
        str(args.model),
        backend=str(args.backend),
        execution_profile=str(args.execution_profile),
        max_active_requests=resident_capacity,
        max_sequence_length=int(args.max_sequence_length),
        speculative_candidate_budget=int(args.candidate_budget),
    )
    runtime_profile: dict[str, Any] = {}
    try:
        llm.prepare(max_sequence_length=int(args.max_sequence_length))
        runtime_profile = {
            "requested": str(args.execution_profile),
            "resolved": getattr(llm, "resolved_execution_profile", None),
            "manifest_sha256": getattr(llm, "execution_profile_manifest_sha256", None),
            "strict_manifest_sha256": getattr(
                llm, "execution_profile_strict_manifest_sha256", None
            ),
        }
        if args.generation2_diagnostic:
            _install_diagnostic_plan(llm)
        app = create_app(
            ServerConfig(
                model=str(args.model),
                backend=str(args.backend),
                quant=str(args.quant),
                served_model_name="qwen36-mtp-c1c8",
                eager_load=False,
                metrics="prometheus",
                generation_batch_window_ms=float(args.batch_window_ms),
                max_context_tokens=int(args.max_sequence_length),
                max_active_requests=resident_capacity,
                speculative_mtp_serving="opt_in",
                speculative_candidate_budget=int(args.candidate_budget),
                shutdown_grace_seconds=5.0,
            ),
            llm=llm,
        )
        cells: list[dict[str, Any]] = []
        with TestClient(app) as client:
            for width in widths:
                for arm in ARMS:
                    _run_arm(
                        client,
                        llm=llm,
                        model="qwen36-mtp-c1c8",
                        prompt=str(prompts[0]["rendered_prompt"]),
                        width=width,
                        max_tokens=int(args.max_tokens),
                        arm=arm,
                    )
                for prompt_index, prompt in enumerate(prompts):
                    order = ARMS if (prompt_index + width) % 2 else tuple(reversed(ARMS))
                    measured: dict[str, Any] = {}
                    for arm in order:
                        measured[arm] = _run_arm(
                            client,
                            llm=llm,
                            model="qwen36-mtp-c1c8",
                            prompt=str(prompt["rendered_prompt"]),
                            width=width,
                            max_tokens=int(args.max_tokens),
                            arm=arm,
                        )
                    ar_ids = [row["generated_ids"] for row in measured["ar"]["rows"]]
                    mtp_ids = [row["generated_ids"] for row in measured["mtp"]["rows"]]
                    correctness = _cell_correctness(
                        ar_ids,
                        mtp_ids,
                        contract=str(args.correctness_contract),
                    )
                    response_engaged = all(
                        _mtp_engaged(row["route"], row["mtp"])
                        for row in measured["mtp"]["rows"]
                    )
                    backend_engaged = _backend_mtp_engaged(
                        measured["mtp"]["backend_telemetry"],
                        width=width,
                    )
                    engaged = bool(response_engaged or backend_engaged)
                    budget_conformed = bool(
                        not engaged
                        or all(
                            _mtp_budget_conformed(
                                row["mtp"], budget=int(args.candidate_budget)
                            )
                            for row in measured["mtp"]["rows"]
                        )
                    )
                    cell = {
                        "prompt_id": prompt["id"],
                        "category": prompt["category"],
                        "heldout": prompt["heldout"],
                        "prompt_sha256": prompt["prompt_sha256"],
                        "width": width,
                        "order": list(order),
                        "ar": measured["ar"],
                        "mtp": measured["mtp"],
                        "correctness": correctness,
                        "exact": correctness["passed"],
                        "mtp_engaged": engaged,
                        "mtp_budget_conformed": budget_conformed,
                    }
                    cells.append(cell)
                    print(
                        json.dumps(
                            {
                                "width": width,
                                "prompt": prompt["id"],
                                "ar_tok_s": measured["ar"]["tok_s"],
                                "mtp_tok_s": measured["mtp"]["tok_s"],
                                "correctness": correctness,
                                "engaged": engaged,
                                "budget_conformed": budget_conformed,
                            }
                        ),
                        flush=True,
                    )
        # TestClient owns current-server shutdown. Older servers leave close idempotent.
    finally:
        llm.close()
    summary = summarize(
        cells,
        widths=widths,
        expected_mtp_widths=expected_mtp_widths,
    )
    passed = all(
        int(row["exact_cells"])
        == int(row["budget_conformed_cells"])
        == int(row["cells"])
        == len(prompts)
        and bool(row["route_expectation_passed"])
        for row in summary.values()
    )
    payload = {
        "schema": 2,
        "kind": "gguf_mtp_c1c8_server_bench",
        "date": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if passed else "failed",
        "passed": passed,
        "source": {"commit": source_commit, "dirty": False},
        "model": collect_model_identity(args.model),
        "hardware": {
            "backend": args.backend,
            "hip_visible_devices": __import__("os").environ.get("HIP_VISIBLE_DEVICES"),
            "gpu_max_hw_queues": __import__("os").environ.get("GPU_MAX_HW_QUEUES"),
        },
        "protocol": {
            "prompts": str(Path(args.prompts).resolve()),
            "prompt_ids": list(FULL_PROMPT_IDS),
            "widths": list(widths),
            "resident_capacity": resident_capacity,
            "expected_mtp_widths": list(expected_mtp_widths),
            "max_tokens": int(args.max_tokens),
            "candidate_budget": int(args.candidate_budget),
            "batch_window_ms": float(args.batch_window_ms),
            "warmup_arms_per_width": 2,
            "runs": 1,
            "sampling": "raw greedy, no processed target, natural stop/EOS",
            "execution_profile": str(args.execution_profile),
            "correctness_contract": str(args.correctness_contract),
            "generated_id_equality": "diagnostic; production promotion binds the complete execution-profile numerical/task gate",
            "generation2_diagnostic_plan": bool(args.generation2_diagnostic),
            "timing": "blocking OpenAI barrier-to-last-completion complete wall",
        },
        "runtime_profile": runtime_profile,
        "summary": summary,
        "acceptance": summarize_acceptance(cells),
        "cells": cells,
        "initial_memory": initial_memory,
        "final_memory": memory_stats(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument(
        "--execution-profile",
        choices=("strict", "production"),
        default="production",
    )
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--widths", type=_parse_widths, default=tuple(range(1, 9)))
    parser.add_argument(
        "--resident-capacity",
        type=int,
        default=None,
        help="resident owner capacity; defaults to the maximum measured width",
    )
    parser.add_argument(
        "--expected-mtp-widths",
        type=_parse_expected_mtp_widths,
        default=None,
        help="Expected engaged MTP widths; use 'none' for typed K0-only cells",
    )
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--candidate-budget", type=int, default=2)
    parser.add_argument("--batch-window-ms", type=float, default=20.0)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--generation2-diagnostic", action="store_true")
    parser.add_argument(
        "--correctness-contract",
        choices=_CORRECTNESS_CONTRACTS,
        default="ar_exact",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
