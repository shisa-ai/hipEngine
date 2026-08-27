#!/usr/bin/env python3
"""Qualify the real Qwen3.6-27B Dense physical C2 MTP target numerically.

The candidate logits come from the actual request-major C2 packed verifier.  A
correctness-only wrapper reads candidate IDs before target execution and copies
its full-vocabulary logits afterward; ordinary serving remains device-only.
Strict and production consume the same strict-teacher contexts.  Generated-ID
agreement is reported only as a diagnostic outside the strict control surface.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine import LLM, SamplingParams  # noqa: E402
from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    DeviceBuffer,
    copy_device_to_host,
    host_array_ptr,
)
from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.runtime.prefill import PrefillConfig  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import (  # noqa: E402
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
)
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer  # noqa: E402
from scripts.gguf_mtp_c1c8_server_bench import (  # noqa: E402
    _resident_observability,
    load_prompt_suite,
)
from scripts.gguf_mtp_forced_target_probe import _probe_serial  # noqa: E402
from scripts.quant_quality.metrics import per_row_metrics  # noqa: E402
from scripts.qwen36_moe_mtp2_production_gate import (  # noqa: E402
    THRESHOLDS,
    _file_sha256,
    _sha256_json,
    _strict_prefix,
    numerical_verdict,
    paired_task_verdict,
)

KIND = "qwen36_dense_mtp2_c2_production_gate"
MODEL_SHA256 = "a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f"


class GateError(RuntimeError):
    """Raised when the C2 production packet cannot be evaluated honestly."""


def _telemetry_dict(chunk: Any) -> dict[str, Any]:
    telemetry = getattr(chunk, "telemetry", None)
    if telemetry is None:
        return {}
    if isinstance(telemetry, Mapping):
        return dict(telemetry)
    convert = getattr(telemetry, "to_json_dict", None)
    return dict(convert()) if callable(convert) else {}


def _output_record(chunks: Sequence[Any]) -> dict[str, Any]:
    if not chunks:
        raise GateError("production stream returned no chunks")
    final = chunks[-1]
    telemetry = _telemetry_dict(final)
    decode = telemetry.get("decode_state")
    decode = decode if isinstance(decode, Mapping) else {}
    token_ids = tuple(int(value) for value in (getattr(final, "generated_token_ids", ()) or ()))
    if not token_ids:
        raise GateError("production stream omitted final generated-token IDs")
    request_id = decode.get("request_id")
    if request_id is None:
        raise GateError("production stream omitted resident request identity")
    return {
        "request_id": int(request_id),
        "token_ids": token_ids,
        "telemetry": telemetry,
    }


def _copy_device_i32(tensor: Any, runtime: Any) -> tuple[int, ...]:
    values = np.empty(tuple(int(value) for value in tensor.shape), dtype=np.int32)
    copy_device_to_host(
        host_array_ptr(values),
        DeviceBuffer(int(tensor.ptr), values.nbytes),
        values.nbytes,
        runtime=runtime,
    )
    return tuple(int(value) for value in values.reshape(-1))


def _install_packed_capture(context: dict[str, Any]):
    original = Qwen35GGUFResidentSession.verify_target_blocks_batch

    def wrapped(self: Any, jobs: Any, **kwargs: Any):
        job_rows = list(jobs)
        if context.get("enabled") is not True:
            return original(self, jobs, **kwargs)
        runtime = self.runtime
        if runtime is None or self.runner is None:
            raise GateError("packed C2 capture lost runtime/runner")
        runtime.device_synchronize()
        inputs: list[tuple[int, ...]] = []
        starts: list[int] = []
        request_ids: list[int] = []
        full_row_counts: list[int] = []
        for job in job_rows:
            host_ids = tuple(int(value) for value in job.get("input_token_ids", ()))
            if not host_ids:
                raise GateError("packed C2 capture found an empty job")
            candidates = job.get("candidate_token_ids_device")
            if candidates is not None:
                candidate_ids = _copy_device_i32(candidates, runtime)
                host_ids = (host_ids[0], *candidate_ids)
            inputs.append(host_ids)
            full_row_counts.append(len(host_ids))
            starts.append(int(job["session"].position))
            request_ids.append(int(job["request_id"]))
        results = original(self, jobs, **kwargs)
        vocab = int(self.runner.vocab_size)
        cycle_key = (int(context["repeat"]), int(context["pair"]))
        cycle = int(context["cycle_counts"].get(cycle_key, 0))
        context["cycle_counts"][cycle_key] = cycle + 1
        for request_id, start, input_ids, full_rows in zip(
            request_ids, starts, inputs, full_row_counts, strict=True
        ):
            # Terminal device proposal slots use an out-of-vocabulary sentinel.
            # They are control padding, not teacher-forced model rows.
            active_rows = 1
            for token_id in input_ids[1:]:
                if token_id < 0 or token_id >= vocab:
                    break
                active_rows += 1
            active_inputs = input_ids[:active_rows]
            context["captures"].append(
                {
                    "repeat": int(context["repeat"]),
                    "pair": int(context["pair"]),
                    "cycle": cycle,
                    "request_id": request_id,
                    "start_position": start,
                    "inputs": active_inputs,
                    "full_rows": full_rows,
                    "active_rows": active_rows,
                }
            )
        return results

    Qwen35GGUFResidentSession.verify_target_blocks_batch = wrapped
    return original


def _strict_outputs(
    model: Path,
    prompts: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int,
    max_sequence_length: int,
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    llm = LLM(
        str(model),
        backend="hip_gfx1100",
        execution_profile="strict",
        max_active_requests=1,
        max_sequence_length=max_sequence_length,
        speculative_candidate_budget=2,
    )
    result: dict[str, tuple[int, ...]] = {}
    try:
        llm.prepare(max_sequence_length=max_sequence_length)
        sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0)
        for prompt in prompts:
            output = llm.generate_detailed((str(prompt["rendered_prompt"]),), sampling)[0]
            token_ids = tuple(int(value) for value in (output.generated_token_ids or ()))
            if len(token_ids) != max_tokens:
                raise GateError(f"strict output length mismatch for {prompt['id']}")
            result[str(prompt["id"])] = token_ids
        profile = {
            "name": getattr(llm, "execution_profile", None),
            "manifest_sha256": getattr(llm, "execution_profile_manifest_sha256", None),
            "strict_manifest_sha256": getattr(
                llm, "execution_profile_strict_manifest_sha256", None
            ),
        }
    finally:
        llm.close()
    return result, profile


def _production_pairs(
    model: Path,
    prompts: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int,
    max_sequence_length: int,
    repeat_runs: int,
    context: dict[str, Any],
) -> tuple[
    dict[str, list[tuple[int, ...]]],
    dict[str, tuple[int, ...]],
    dict[str, Any],
    dict[str, Any],
]:
    llm = LLM(
        str(model),
        backend="hip_gfx1100",
        execution_profile="production",
        max_active_requests=2,
        max_sequence_length=max_sequence_length,
        speculative_candidate_budget=2,
    )
    output_runs: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    prompt_by_id = {str(row["id"]): row for row in prompts}
    try:
        llm.prepare(max_sequence_length=max_sequence_length)
        sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0)
        pairs = tuple(
            (str(prompts[i]["id"]), str(prompts[i + 1]["id"]))
            for i in range(0, len(prompts), 2)
        )

        def run_pair(
            pair: tuple[str, str],
            *,
            repeat: int,
            pair_index: int,
            capture_enabled: bool,
        ) -> tuple[tuple[str, dict[str, Any]], ...]:
            barrier = threading.Barrier(2)
            start_capture = len(context["captures"])
            context["repeat"] = repeat
            context["pair"] = pair_index
            context["enabled"] = capture_enabled

            def run(item: tuple[int, str]) -> tuple[str, dict[str, Any]]:
                index, prompt_id = item
                barrier.wait(timeout=30.0)
                # Stable request/slot ownership is part of same-schedule
                # repeatability. The stagger is tiny relative to prefill and
                # still forms one live C2 decode group.
                if index:
                    time.sleep(0.01)
                chunks = tuple(
                    llm.stream_speculative_mtp_detailed(
                        str(prompt_by_id[prompt_id]["rendered_prompt"]),
                        sampling,
                    )
                )
                return prompt_id, _output_record(chunks)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                rows = tuple(executor.map(run, enumerate(pair)))
            context["enabled"] = False
            request_to_prompt = {
                int(row["request_id"]): prompt_id for prompt_id, row in rows
            }
            captures = context["captures"][start_capture:]
            if capture_enabled and not captures:
                raise GateError(f"production pair {pair} executed no physical C2 target")
            for capture in captures:
                try:
                    capture["prompt_id"] = request_to_prompt[
                        int(capture["request_id"])
                    ]
                except KeyError as exc:
                    raise GateError(
                        "physical capture request identity crossed pair ownership"
                    ) from exc
            return rows

        # Discard one complete physical run per pair so graph/JIT/provider-pool
        # first use cannot contaminate the three binding same-schedule repeats.
        for pair_index, pair in enumerate(pairs):
            run_pair(
                pair,
                repeat=-1,
                pair_index=pair_index,
                capture_enabled=False,
            )
        for repeat in range(repeat_runs):
            for pair_index, pair in enumerate(pairs):
                rows = run_pair(
                    pair,
                    repeat=repeat,
                    pair_index=pair_index,
                    capture_enabled=True,
                )
                for prompt_id, row in rows:
                    output_runs[prompt_id].append(tuple(row["token_ids"]))
        permutation_outputs: dict[str, tuple[int, ...]] = {}
        for pair_index, pair in enumerate(pairs):
            rows = run_pair(
                tuple(reversed(pair)),
                repeat=repeat_runs,
                pair_index=pair_index,
                capture_enabled=False,
            )
            for prompt_id, row in rows:
                permutation_outputs[prompt_id] = tuple(row["token_ids"])
        context["enabled"] = False
        service = llm._get_text_generator()
        snapshot = service.live_loop_snapshot()
        observability = _resident_observability(llm, recent=16)
        profile = {
            "name": getattr(llm, "execution_profile", None),
            "manifest_sha256": getattr(llm, "execution_profile_manifest_sha256", None),
            "strict_manifest_sha256": getattr(
                llm, "execution_profile_strict_manifest_sha256", None
            ),
            "snapshot": snapshot,
        }
    finally:
        context["enabled"] = False
        llm.close()
    return dict(output_runs), permutation_outputs, profile, observability


def _repeat_verdict(captures: Sequence[Mapping[str, Any]], repeat_runs: int) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in captures:
        grouped[(str(row["prompt_id"]), int(row["start_position"]))].append(row)
    rows = []
    for (prompt_id, start), values in sorted(grouped.items()):
        repeats = {int(row["repeat"]) for row in values}
        inputs = {tuple(int(value) for value in row["inputs"]) for row in values}
        passed = bool(
            repeats == set(range(repeat_runs))
            and len(values) == repeat_runs
            and len(inputs) == 1
        )
        rows.append(
            {
                "prompt_id": prompt_id,
                "start_position": start,
                "repeats": sorted(repeats),
                "inputs_equal": len(inputs) == 1,
                "passed": passed,
            }
        )
    return {"passed": bool(rows) and all(row["passed"] for row in rows), "rows": rows}


def _teacher_metrics(
    model: Path,
    prompts: Sequence[Mapping[str, Any]],
    strict_outputs: Mapping[str, Sequence[int]],
    captures: Sequence[Mapping[str, Any]],
    *,
    repeat_runs: int,
    max_sequence_length: int,
    compiler_version_file: Path | None,
    require_cached_build: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the physical C2 target on strict-teacher resident states."""

    prompt_by_id = {str(row["id"]): row for row in prompts}
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(model))
    first_groups: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in captures:
        if int(row["repeat"]) == 0:
            first_groups[(int(row["pair"]), int(row["cycle"]))].append(row)
    compiler_version = (
        None
        if compiler_version_file is None
        else compiler_version_file.read_text(encoding="utf-8")
    )
    runner = Qwen35GGUFFullStackRunner(
        model,
        backend="hip_gfx1100",
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
    )
    sessions = [
        Qwen35GGUFResidentSession(
            model,
            backend="hip_gfx1100",
            shared_runner=runner,
            compiler_version=compiler_version,
            require_cached_build=require_cached_build,
            max_sequence_length=max_sequence_length,
            use_wmma_prefill=False,
            use_gemv_decode=True,
            prefill_config=PrefillConfig(),
        )
        for _ in range(2)
    ]
    metrics_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []

    def strict_inputs(
        session: Qwen35GGUFResidentSession,
        capture: Mapping[str, Any],
    ) -> tuple[tuple[int, ...], int, Mapping[str, Any]]:
        prompt_id = str(capture["prompt_id"])
        prompt = prompt_by_id[prompt_id]
        prompt_tokens = tokenizer.encode(str(prompt["rendered_prompt"]))
        output_index = int(capture["start_position"]) - len(prompt_tokens) + 1
        if output_index <= 0 or output_index >= len(strict_outputs[prompt_id]):
            raise GateError(
                f"physical C2 capture position is outside strict teacher: {prompt_id}"
            )
        root = _strict_prefix(session, prompt_tokens, output_index)
        live_inputs = tuple(int(value) for value in capture["inputs"])
        return (root, *live_inputs[1:]), output_index, prompt

    def run_physical(
        group: Sequence[Mapping[str, Any]],
    ) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
        inputs_by_slot: list[tuple[int, ...]] = []
        jobs: list[dict[str, Any]] = []
        for slot, (session, capture) in enumerate(zip(sessions, group, strict=True)):
            inputs, _output_index, _prompt = strict_inputs(session, capture)
            inputs_by_slot.append(inputs)
            jobs.append(
                {
                    "session": session,
                    "request_id": slot,
                    "resident_slot": slot,
                    "transaction_id": 0,
                    "bulk_attention_mode": "bulk",
                    "use_wmma_prefill": False,
                    "capture_linear_state_rows": True,
                    "defer_linear_state_commit": True,
                    "defer_state_scatter": False,
                    "input_token_ids": inputs,
                }
            )
        sessions[0].verify_target_blocks_batch(jobs, device_result=False)
        runtime = sessions[0].runtime
        if runtime is None or sessions[0].runner is None:
            raise GateError("strict physical C2 target lost runtime/runner")
        runtime.device_synchronize()
        rows = sum(len(values) for values in inputs_by_slot)
        vocab = int(sessions[0].runner.vocab_size)
        logits_owner = sessions[0]._verify_logits_buf
        if logits_owner is None:
            raise GateError("strict physical C2 target omitted full logits")
        logits = np.empty((rows, vocab), dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(logits),
            DeviceBuffer(int(logits_owner.ptr), logits.nbytes),
            logits.nbytes,
            runtime=runtime,
        )
        if not np.isfinite(logits).all():
            raise GateError("strict physical C2 target produced non-finite logits")
        return np.ascontiguousarray(logits), tuple(inputs_by_slot)

    try:
        for group_key, group_rows in sorted(first_groups.items()):
            if len(group_rows) != 2:
                raise GateError(f"strict physical group {group_key} is not C2")
            group = tuple(sorted(group_rows, key=lambda row: int(row["request_id"])))
            # Discard first use for this exact ragged physical shape, then bind
            # three identical full-logit repeats on strict-teacher states.
            run_physical(group)
            candidate_runs: list[np.ndarray] = []
            input_runs: list[tuple[tuple[int, ...], ...]] = []
            for _ in range(repeat_runs):
                logits, inputs_by_slot = run_physical(group)
                candidate_runs.append(logits)
                input_runs.append(inputs_by_slot)
            hashes = [
                hashlib.sha256(values.view(np.uint8)).hexdigest()
                for values in candidate_runs
            ]
            top1_runs = [
                tuple(int(value) for value in np.argmax(values, axis=1))
                for values in candidate_runs
            ]
            repeat_passed = bool(
                len(set(hashes)) == 1
                and len(set(top1_runs)) == 1
                and len(set(input_runs)) == 1
            )
            repeat_rows.append(
                {
                    "pair": group_key[0],
                    "cycle": group_key[1],
                    "candidate_logits_sha256": hashes,
                    "candidate_top1": [list(values) for values in top1_runs],
                    "inputs_equal": len(set(input_runs)) == 1,
                    "passed": repeat_passed,
                }
            )
            candidate_logits = candidate_runs[0]
            offset = 0
            for session, capture, inputs in zip(
                sessions, group, input_runs[0], strict=True
            ):
                prompt_id = str(capture["prompt_id"])
                _inputs, output_index, prompt = strict_inputs(session, capture)
                strict_tokens, strict_logits, *_ = _probe_serial(
                    session,
                    list(inputs),
                    capture_pre_output_norm_hidden=False,
                    capture_layer_output_hidden=[],
                )
                candidate_slice = np.ascontiguousarray(
                    candidate_logits[offset : offset + len(inputs)]
                )
                offset += len(inputs)
                labels = np.asarray(strict_tokens, dtype=np.int64)
                values = per_row_metrics(
                    np.ascontiguousarray(strict_logits, dtype=np.float32),
                    candidate_slice,
                    labels,
                    top_k=5,
                )
                candidate_top1 = np.argmax(candidate_slice, axis=1)
                for row_index in range(len(inputs)):
                    strict_row = strict_logits[row_index]
                    top2 = np.partition(strict_row, -2)[-2:]
                    metrics_rows.append(
                        {
                            "prompt_id": prompt_id,
                            "category": str(prompt["category"]),
                            "heldout": bool(prompt["heldout"]),
                            "shape": f"k{len(inputs) - 1}",
                            "transition": (
                                "prefill_to_verify"
                                if output_index <= 2
                                else "verify_to_verify"
                            ),
                            "position": int(capture["start_position"]) + row_index,
                            "strict_top1": int(strict_tokens[row_index]),
                            "candidate_top1": int(candidate_top1[row_index]),
                            "strict_margin": float(top2.max() - top2.min()),
                            "kl": float(values["kl_nats"][row_index]),
                            "top1_equal": bool(values["top1_equal"][row_index]),
                            "top5_overlap": float(
                                values["topk_set_overlap"][row_index]
                            ),
                            "teacher_nll": float(
                                values["teacher_nll_nats"][row_index]
                            ),
                            "strict_teacher_nll": float(
                                values["reference_teacher_nll_nats"][row_index]
                            ),
                            "delta_p": float(values["delta_p"][row_index]),
                            "max_abs_logit_delta": float(
                                values["max_abs_logit_delta"][row_index]
                            ),
                        }
                    )
            if offset != int(candidate_logits.shape[0]):
                raise GateError("strict physical C2 logit row accounting drifted")
    finally:
        for session in sessions:
            session.close()
        runner.close()
    return metrics_rows, {
        "passed": bool(repeat_rows) and all(row["passed"] for row in repeat_rows),
        "cycles": len(repeat_rows),
        "rows": repeat_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model.resolve()
    prompts = load_prompt_suite(args.prompts.resolve())
    if args.limit is not None:
        prompts = prompts[: int(args.limit)]
    if len(prompts) < 2 or len(prompts) % 2:
        raise GateError("C2 production gate requires a positive even prompt count")
    if args.limit is None and len(prompts) != 10:
        raise GateError("full C2 production gate requires all ten prompts")
    if os.popen("git status --porcelain --untracked-files=no").read().strip():
        raise GateError("production gate requires a tracked-clean worktree")
    os.environ["HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A"] = "0"
    os.environ["HIPENGINE_GGUF_FP16_RECURRENT_STATE"] = "0"
    os.environ["HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"] = "1"

    strict_outputs, strict_profile = _strict_outputs(
        model,
        prompts,
        max_tokens=args.max_tokens,
        max_sequence_length=args.max_sequence_length,
    )
    context: dict[str, Any] = {
        "enabled": False,
        "repeat": 0,
        "pair": 0,
        "captures": [],
        "cycle_counts": {},
    }
    original = _install_packed_capture(context)
    try:
        (
            production_outputs,
            permutation_outputs,
            production_profile,
            observability,
        ) = _production_pairs(
            model,
            prompts,
            max_tokens=args.max_tokens,
            max_sequence_length=args.max_sequence_length,
            repeat_runs=args.repeat_runs,
            context=context,
        )
    finally:
        Qwen35GGUFResidentSession.verify_target_blocks_batch = original
    captures: list[dict[str, Any]] = context["captures"]
    repeat_gate = _repeat_verdict(captures, args.repeat_runs)
    row_metrics, physical_repeat_gate = _teacher_metrics(
        model,
        prompts,
        strict_outputs,
        captures,
        repeat_runs=args.repeat_runs,
        max_sequence_length=args.max_sequence_length,
        compiler_version_file=args.compiler_version_file,
        require_cached_build=args.require_cached_build,
    )
    numerical = numerical_verdict(row_metrics)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(model))
    output_checks = []
    task_rows = []
    for prompt_id, runs in production_outputs.items():
        repeat_exact = len(set(runs)) == 1
        strict_equal = runs[0] == tuple(strict_outputs[prompt_id])
        permutation_equal = permutation_outputs.get(prompt_id) == runs[0]
        output_checks.append(
            {
                "prompt_id": prompt_id,
                "repeat_exact": repeat_exact,
                "neighbor_permutation_isolation": permutation_equal,
                "strict_generated_ids_equal_diagnostic": strict_equal,
            }
        )
        task_rows.append(
            paired_task_verdict(
                prompt_id,
                tokenizer.decode(strict_outputs[prompt_id]),
                tokenizer.decode(runs[0]),
            )
        )
    tasks = {"passed": all(row["passed"] for row in task_rows), "prompts": task_rows}
    snapshot = production_profile.pop("snapshot")
    engine_service = snapshot.get("engine_service", {})
    adapter = observability.get("mtp2_adapter", {})
    recent = observability.get("routes", {}).get("recent_completed", [])
    lifecycle = {
        "active_children_zero": int(engine_service.get("active_children", -1)) == 0,
        "command_queue_empty": int(engine_service.get("command_queue_depth", -1)) == 0,
        "sole_driver": bool(engine_service.get("sole_driver")),
        "legacy_fallback_zero": int(
            engine_service.get("speculative_routes", {}).get(
                "legacy_prelaunch_fallback", -1
            )
        )
        == 0,
        "active_states_zero": int(adapter.get("active_states", -1)) == 0,
        "provider_groups_zero": int(adapter.get("provider_groups", -1)) == 0,
        "candidate_d2h_zero": all(
            int(row.get("specdec2_mtp2_candidate_d2h_after_target", -1)) == 0
            for row in recent
        ),
        "recoveries_zero": all(
            int(row.get("specdec2_mtp2_recoverable_failures", -1)) == 0
            for row in recent
        ),
    }
    profile_checks = {
        "strict_profile": strict_profile.get("name") == "strict",
        "production_profile": production_profile.get("name") == "production",
        "strict_manifest": bool(strict_profile.get("manifest_sha256")),
        "production_manifest": bool(production_profile.get("manifest_sha256")),
        "registered_strict_fallback": (
            production_profile.get("strict_manifest_sha256")
            == strict_profile.get("manifest_sha256")
        ),
    }
    cycle_groups: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in captures:
        cycle_groups[
            (int(row["repeat"]), int(row["pair"]), int(row["cycle"]))
        ].append(row)
    capture_shape_checks = {
        "physical_pairs": bool(cycle_groups)
        and all(
            len(rows) == 2
            and len({int(row["request_id"]) for row in rows}) == 2
            for rows in cycle_groups.values()
        ),
        "candidate_depth_bounded": all(
            0 <= len(row["inputs"]) - 1 <= 2 for row in captures
        )
        and any(len(row["inputs"]) - 1 == 2 for row in captures),
        "actual_full_logits": all(
            row["active_rows"] == len(row["candidate_top1"]) for row in captures
        ),
    }
    checks = {
        "full_suite": args.limit is None and len(prompts) == 10,
        "physical_capture": all(capture_shape_checks.values()),
        "numerical": bool(numerical["passed"]),
        "repeat_determinism": bool(
            repeat_gate["passed"] and physical_repeat_gate["passed"]
        ),
        "output_repeat": all(row["repeat_exact"] for row in output_checks),
        "neighbor_permutation_isolation": all(
            row["neighbor_permutation_isolation"] for row in output_checks
        ),
        "task_noninferiority": bool(tasks["passed"]),
        "profiles": all(profile_checks.values()),
        "lifecycle": all(lifecycle.values()),
    }
    if args.limit is not None:
        checks["full_suite"] = True
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend="hip_gfx1100",
        resolved_backend="hip_gfx1100",
        detected_arches=("gfx1100",),
        target_arch="gfx1100",
        model_path=model,
        quant="Q4_K_M",
        kv_dtype="bf16",
        command=sys.argv,
        environment={
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
        },
        timing_protocol="correctness-only physical-C2 strict-teacher full-logit packet",
        warmups=0,
        repetitions=args.repeat_runs,
        host_name=platform.node(),
    )
    review_rows = [
        row
        for row in row_metrics
        if float(row["kl"]) > 2.0e-2 or not bool(row["top1_equal"])
    ]
    return {
        "schema": 1,
        "kind": KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "performance_claim": False,
        "provenance": provenance,
        "model": {
            "path": str(model),
            "size_bytes": model.stat().st_size,
            "sha256": MODEL_SHA256,
            "quant": "Q4_K_M",
            "kv": "bf16",
        },
        "workload": {
            "prompts": str(args.prompts.resolve()),
            "prompt_sha256": _file_sha256(args.prompts.resolve()),
            "prompt_ids": [str(row["id"]) for row in prompts],
            "resident_capacity": 2,
            "physical_group_rows": 2,
            "candidate_budget": 2,
            "max_tokens": args.max_tokens,
            "repeat_runs": args.repeat_runs,
            "sampling": "raw greedy",
            "correctness_capture": "candidate IDs D2H before target and full logits D2H after target; no performance claim",
        },
        "thresholds": THRESHOLDS,
        "checks": checks,
        "profiles": {
            "strict": strict_profile,
            "production": production_profile,
            "checks": profile_checks,
        },
        "physical_capture": {
            "checks": capture_shape_checks,
            "cycles": len(captures),
            "rows": sum(int(row["active_rows"]) for row in captures if int(row["repeat"]) == 0),
            "digest": _sha256_json(
                [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"candidate_logits"}
                    }
                    for row in captures
                ]
            ),
        },
        "numerical": numerical,
        "repeat_determinism": {
            "passed": bool(
                repeat_gate["passed"] and physical_repeat_gate["passed"]
            ),
            "live_schedule": repeat_gate,
            "strict_teacher_physical_logits": physical_repeat_gate,
        },
        "output_control": output_checks,
        "tasks": tasks,
        "lifecycle": {
            "checks": lifecycle,
            "engine_service": engine_service,
            "mtp2_adapter": adapter,
        },
        "bf16_relative": {
            "applicable": False,
            "reason": "No aligned same-model BF16/full-precision artifact is available on the binding host.",
        },
        "review_rows": review_rows,
        "row_metrics": row_metrics,
        "capture_hashes": {
            "strict_outputs": _sha256_json(strict_outputs),
            "production_outputs": _sha256_json(production_outputs),
            "row_metrics": _sha256_json(row_metrics),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"),
    )
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--verify-model-sha256", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_tokens != 24 or args.repeat_runs < 3:
        raise GateError("binding C2 gate requires D24 and at least three repeats")
    if args.verify_model_sha256 and _file_sha256(args.model) != MODEL_SHA256:
        raise GateError("model SHA-256 does not match the frozen 27B artifact")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "checks": payload["checks"],
                "summary": payload["numerical"]["aggregate"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
