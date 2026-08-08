#!/usr/bin/env python3
"""Natural-prompt, context, rebuild, cancellation, and teardown gate for PM4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from hipengine.core.pm4.transport import create_graph_submission_context  # noqa: E402
from hipengine.loading import load_gguf_index  # noqa: E402
from hipengine.runtime.gguf_decode_graph import (  # noqa: E402
    capture_qwen35_gguf_decode_graph,
)
from hipengine.runtime.qwen35_gguf_runner import (  # noqa: E402
    Qwen35GGUFResidentSession,
)
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer  # noqa: E402
from scripts.gguf_decode_graph_g5 import _prefill  # noqa: E402
from scripts.gguf_mtp_bench import build_chat_prompt  # noqa: E402
from scripts.pm4_graph_bench import (  # noqa: E402
    _Roctx,
    _logits_sha256,
    _read_compiler_version,
    _state_sha256,
    _timed_replay,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_SUITES = (
    REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
    REPO_ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl",
)
DEFAULT_MEMORY_RECOVERY_TOLERANCE = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PromptCase:
    name: str
    category: str
    source: str
    content_sha256: str
    token_ids: tuple[int, ...]


def _message_content(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt record must contain a non-empty messages list")
    if any(
        not isinstance(message, dict)
        or message.get("role") != "user"
        or not isinstance(message.get("content"), str)
        or not message["content"]
        for message in messages
    ):
        raise ValueError("promotion prompt records must contain non-empty user messages only")
    return "\n\n".join(str(message["content"]) for message in messages)


def _load_prompt_cases(
    paths: Sequence[Path], tokenizer: Qwen35GGUFTokenizer
) -> tuple[PromptCase, ...]:
    cases: list[PromptCase] = []
    names: set[str] = set()
    for path_value in paths:
        path = Path(path_value).expanduser().resolve(strict=True)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            name = str(record.get("id", "")).strip()
            category = str(record.get("category", "")).strip()
            if not name or not category:
                raise ValueError(f"{path}:{line_number}: prompt id/category is empty")
            if name in names:
                raise ValueError(f"duplicate promotion prompt id {name!r}")
            content = _message_content(record)
            token_ids = tuple(
                int(token) for token in build_chat_prompt(tokenizer, content, reasoning="off")
            )
            if not token_ids:
                raise ValueError(f"{path}:{line_number}: prompt tokenization is empty")
            names.add(name)
            try:
                source = str(path.relative_to(REPO_ROOT))
            except ValueError:
                source = str(path)
            cases.append(
                PromptCase(
                    name=name,
                    category=category,
                    source=source,
                    content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    token_ids=token_ids,
                )
            )
    if not cases:
        raise ValueError("promotion suite contains no prompts")
    return tuple(cases)


def _repeat_to_length(values: Sequence[int], length: int) -> tuple[int, ...]:
    items = tuple(int(value) for value in values)
    if not items or length <= 0:
        raise ValueError("4K context construction requires tokens and a positive length")
    repeats = (int(length) + len(items) - 1) // len(items)
    return (items * repeats)[: int(length)]


def _row(session: Any, graph: Any, tokens: Sequence[int], *, steps: int, label: str) -> dict:
    prefill_start_ns = time.perf_counter_ns()
    seed_token = _prefill(session, list(tokens))
    prefill_ms = (time.perf_counter_ns() - prefill_start_ns) / 1e6
    graph.rearm_replay_window()
    session.runtime.device_synchronize()
    row = _timed_replay(graph, steps=steps, roctx=_Roctx(False), label=label)
    final = session._read_sample(return_logits=False)
    row.update(
        {
            "prefill_ms": prefill_ms,
            "seed_token_id": int(seed_token),
            "final_token_id": int(final.token_id),
            "state_sha256": _state_sha256(
                session,
                input_token_id=int(seed_token),
                predicted_token_id=int(final.token_id),
            ),
            "final_logits_sha256": _logits_sha256(session),
        }
    )
    return row


def _compare_rows(hipgraph: dict[str, Any], pm4: dict[str, Any]) -> dict[str, bool]:
    return {
        "seed_token_exact": hipgraph["seed_token_id"] == pm4["seed_token_id"],
        "final_token_exact": hipgraph["final_token_id"] == pm4["final_token_id"],
        "state_exact": hipgraph["state_sha256"] == pm4["state_sha256"],
        "final_logits_exact": hipgraph["final_logits_sha256"] == pm4["final_logits_sha256"],
    }


def _capture_pm4(
    session: Qwen35GGUFResidentSession,
    context: Any,
    *,
    position: int,
    steps: int,
):
    graph = capture_qwen35_gguf_decode_graph(
        session,
        position=position,
        steps_per_replay=1,
        max_replay_steps=steps,
        attention_max_context_len=position + steps,
        submission_transport="pm4",
        submission_context=context,
    )
    session._pin_device_kv_graph(graph)
    return graph


def _run_case(
    session: Qwen35GGUFResidentSession,
    context: Any,
    case: PromptCase,
    *,
    steps: int,
) -> dict[str, Any]:
    _prefill(session, list(case.token_ids))
    capture_start_ns = time.perf_counter_ns()
    hipgraph = session.capture_decode_graph(
        position=len(case.token_ids),
        steps_per_replay=1,
        max_replay_steps=steps,
        attention_max_context_len=len(case.token_ids) + steps,
        submission_transport="hipgraph",
    )
    hip_capture_ms = (time.perf_counter_ns() - capture_start_ns) / 1e6
    pm4 = None
    try:
        capture_start_ns = time.perf_counter_ns()
        pm4 = _capture_pm4(
            session,
            context,
            position=len(case.token_ids),
            steps=steps,
        )
        pm4_capture_ms = (time.perf_counter_ns() - capture_start_ns) / 1e6
        hip_row = _row(
            session,
            hipgraph,
            case.token_ids,
            steps=steps,
            label=f"pm4-promotion:{case.name}:hipgraph",
        )
        pm4_row = _row(
            session,
            pm4,
            case.token_ids,
            steps=steps,
            label=f"pm4-promotion:{case.name}:pm4",
        )
        comparison = _compare_rows(hip_row, pm4_row)
        live = pm4.transport_provenance()
        native = live.get("executable", {})
        transport_ok = bool(
            live.get("transport") == "pm4"
            and live.get("stateful_registers") is True
            and live.get("local_cache_dependencies") is True
            and live.get("native_fallbacks") == 0
            and native.get("retired") is True
            and native.get("local_cache_dependencies") is True
        )
        passed = bool(all(comparison.values()) and transport_ok)
        return {
            "name": case.name,
            "category": case.category,
            "source": case.source,
            "content_sha256": case.content_sha256,
            "prompt_tokens": len(case.token_ids),
            "capture_ms": {"hipgraph": hip_capture_ms, "pm4": pm4_capture_ms},
            "hipgraph": hip_row,
            "pm4": pm4_row,
            "comparison": comparison,
            "transport_proof": transport_ok,
            "pm4_dwords": int(native.get("pm4_dwords", 0)),
            "passed": passed,
        }
    finally:
        hipgraph.close()
        if pm4 is not None:
            pm4.close()


def _cancellation_gates(
    session: Qwen35GGUFResidentSession,
    context: Any,
    tokens: Sequence[int],
) -> dict[str, Any]:
    _prefill(session, list(tokens))
    graph = _capture_pm4(session, context, position=len(tokens), steps=1)
    before = graph.transport_provenance()
    graph.close()
    no_submit_closed = graph.transport_provenance()

    _prefill(session, list(tokens))
    graph = _capture_pm4(session, context, position=len(tokens), steps=1)
    try:
        graph.rearm_replay_window()
        _timed_replay(
            graph,
            steps=1,
            roctx=_Roctx(False),
            label="pm4-promotion:cancel-after-submit",
        )
        after_submit_live = graph.transport_provenance()
    finally:
        graph.close()
    after_submit_closed = graph.transport_provenance()
    return {
        "before_submit": {
            "submission_started_before_close": bool(before.get("submission_started")),
            "closed": bool(no_submit_closed.get("closed")),
            "passed": bool(
                before.get("submission_started") is False and no_submit_closed.get("closed") is True
            ),
        },
        "after_submit": {
            "submission_started_before_close": bool(after_submit_live.get("submission_started")),
            "retired_before_close": bool(after_submit_live.get("executable", {}).get("retired")),
            "closed": bool(after_submit_closed.get("closed")),
            "passed": bool(
                after_submit_live.get("submission_started") is True
                and after_submit_live.get("executable", {}).get("retired") is True
                and after_submit_closed.get("closed") is True
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--suite-files", type=Path, nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--context-stress-length", type=int, default=4096)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument(
        "--memory-recovery-tolerance-bytes",
        type=int,
        default=DEFAULT_MEMORY_RECOVERY_TOLERANCE,
    )
    parser.add_argument("--json", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.steps) <= 0 or int(args.context_stress_length) <= 0:
        raise ValueError("steps and context stress length must be positive")
    model = args.model.expanduser().resolve(strict=True)
    compiler_version = _read_compiler_version(args.compiler_version_file)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(model))
    natural_cases = _load_prompt_cases(args.suite_files, tokenizer)
    stress_tokens = _repeat_to_length(natural_cases[-1].token_ids, int(args.context_stress_length))
    cases = (
        *natural_cases,
        PromptCase(
            name=f"context_stress_{int(args.context_stress_length)}",
            category="context_stress",
            source=natural_cases[-1].source,
            content_sha256=natural_cases[-1].content_sha256,
            token_ids=stress_tokens,
        ),
    )
    max_sequence_length = max(len(case.token_ids) for case in cases) + int(args.steps) + 8
    old_stateful = os.environ.get("HIPENGINE_PM4_STATEFUL_REGISTERS")
    old_local = os.environ.get("HIPENGINE_PM4_LOCAL_CACHE_DEPENDENCIES")
    os.environ["HIPENGINE_PM4_STATEFUL_REGISTERS"] = "1"
    os.environ["HIPENGINE_PM4_LOCAL_CACHE_DEPENDENCIES"] = "1"
    rows: list[dict[str, Any]] = []
    cancellation: dict[str, Any] = {}
    context_before_close: dict[str, Any] = {}
    context_after_close: dict[str, Any] = {}
    internal_contexts: dict[str, Any] = {}
    free_before = free_after = total_bytes = 0
    try:
        with Qwen35GGUFResidentSession(
            model,
            max_sequence_length=max_sequence_length,
            compiler_version=compiler_version,
            require_cached_build=bool(args.require_cached),
            backend="hip_gfx1100",
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as session:
            # Establish the memory baseline only after the declared maximum shape
            # has initialized its lazy prefill workspaces. Otherwise a later 4K
            # prefill is misclassified as a graph/context teardown leak.
            _prefill(session, list(cases[-1].token_ids))
            free_before, total_bytes = session.runtime.mem_get_info()
            context = create_graph_submission_context(
                backend=str(session.runner.backend),
                gfx_arch=str(session.runner.target_arch),
                runtime=session.runtime,
                transport="pm4",
            )
            if context is None:
                raise RuntimeError("PM4 promotion context was not created")
            try:
                cancellation = _cancellation_gates(session, context, cases[0].token_ids)
                for case in cases:
                    row = _run_case(session, context, case, steps=int(args.steps))
                    rows.append(row)
                    if context.provenance().get("children") != 0:
                        raise RuntimeError("PM4 context retained a closed graph generation")
                context_before_close = context.provenance()
            finally:
                internal_contexts = session.close_decode_graph_submission_contexts()
                if context.provenance().get("children") == 0:
                    context.close()
                context_after_close = context.provenance()
            session.runtime.device_synchronize()
            free_after, total_after = session.runtime.mem_get_info()
            if total_after != total_bytes:
                raise RuntimeError("device total memory changed during promotion gate")
            pci_bdf = session.runtime.device_pci_bus_id()
    finally:
        if old_stateful is None:
            os.environ.pop("HIPENGINE_PM4_STATEFUL_REGISTERS", None)
        else:
            os.environ["HIPENGINE_PM4_STATEFUL_REGISTERS"] = old_stateful
        if old_local is None:
            os.environ.pop("HIPENGINE_PM4_LOCAL_CACHE_DEPENDENCIES", None)
        else:
            os.environ["HIPENGINE_PM4_LOCAL_CACHE_DEPENDENCIES"] = old_local

    categories: dict[str, dict[str, int | bool]] = {}
    for row in rows:
        category = str(row["category"])
        entry = categories.setdefault(category, {"cases": 0, "passed": True})
        entry["cases"] = int(entry["cases"]) + 1
        entry["passed"] = bool(entry["passed"] and row["passed"])
    cancellation_passed = bool(
        cancellation.get("before_submit", {}).get("passed")
        and cancellation.get("after_submit", {}).get("passed")
    )
    memory_delta = int(free_after) - int(free_before)
    memory_recovered = bool(memory_delta >= -int(args.memory_recovery_tolerance_bytes))
    context_passed = bool(
        context_before_close.get("children") == 0
        and context_before_close.get("generations") == len(rows) + 2
        and context_before_close.get("stateful_registers") is True
        and context_before_close.get("local_cache_dependencies") is True
        and context_after_close.get("closed") is True
        and context_after_close.get("native_context_closed") is True
    )
    passed = bool(
        rows
        and all(row["passed"] for row in rows)
        and cancellation_passed
        and context_passed
        and memory_recovered
    )
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend="hip_gfx1100",
        resolved_backend="hip_gfx1100",
        target_arch="gfx1100",
        model_path=model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=[str(part) for part in sys.argv],
        environment={
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_PM4_STATEFUL_REGISTERS": "1",
            "HIPENGINE_PM4_LOCAL_CACHE_DEPENDENCIES": "1",
        },
        build_profile="pm4_promotion_gate",
        timing_protocol=(
            "one resident model; one HIP and PM4 graph per prompt; exact prefill reset"
        ),
        warmups=0,
        repetitions=1,
    )
    return {
        "schema_version": 1,
        "kind": "hipengine_pm4_promotion_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "performance_claim": False,
        "model": {"path": str(model), "size_bytes": model.stat().st_size},
        "hardware": {
            "backend": "hip_gfx1100",
            "gfx_arch": "gfx1100",
            "pci_bdf": pci_bdf,
        },
        "workload": {
            "suite_files": [str(Path(path).resolve()) for path in args.suite_files],
            "natural_cases": len(natural_cases),
            "total_cases": len(cases),
            "steps_per_transport": int(args.steps),
            "context_stress_length": int(args.context_stress_length),
        },
        "categories": categories,
        "cases": rows,
        "cancellation": cancellation,
        "context_lifecycle": {
            "before_close": context_before_close,
            "after_close": context_after_close,
            "internal_contexts": internal_contexts,
            "passed": context_passed,
        },
        "memory": {
            "free_before": int(free_before),
            "free_after": int(free_after),
            "free_delta_bytes": memory_delta,
            "total_bytes": int(total_bytes),
            "recovery_tolerance_bytes": int(args.memory_recovery_tolerance_bytes),
            "recovered": memory_recovered,
        },
        "provenance": provenance,
        "notes": [
            "HIP graph is the exact oracle for every natural and context-stress case.",
            "No submit-plus-queue-recreate lifecycle arm is performed.",
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
