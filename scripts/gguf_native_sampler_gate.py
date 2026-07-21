#!/usr/bin/env python3
"""Correctness/lifecycle gate for the scoped GGUF native GPU sampler.

The gate exercises supported stochastic rows through the resident model owner,
repeats one dense physical c4 trajectory at fixed seeds, and rebuilds every
selected trajectory through the forced-token host fallback. It compares every
Conv/GDN and logical live-KV byte, checks stop/EOS/logprob telemetry, and drains
all request/KV ownership. Timings are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

try:
    from scripts.gguf_prefix_reuse_gate import _capture_state, _compare_states
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from gguf_prefix_reuse_gate import _capture_state, _compare_states


DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
_HARDWARE_LABELS = {
    "hip_gfx1100": "AMD Radeon Pro W7900 (gfx1100)",
    "hip_gfx1151": "AMD Radeon 8060S (gfx1151)",
}


def _prefill_work(request_ids: tuple[int, ...], prompts: tuple[tuple[int, ...], ...]):
    from hipengine.dispatch import WorkItem, WorkKind

    return WorkItem(
        kind=WorkKind.PREFILL,
        request_ids=request_ids,
        row_to_request=request_ids,
        token_rows=prompts,
    )


def _decode_work(request_ids: tuple[int, ...]):
    from hipengine.dispatch import WorkItem, WorkKind

    return WorkItem(
        kind=WorkKind.DECODE,
        request_ids=request_ids,
        row_to_request=request_ids,
    )


def _run_rows(
    runner: Any,
    *,
    request_ids: tuple[int, ...],
    request: Any | tuple[Any, ...],
    prompts: tuple[tuple[int, ...], ...],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if len(request_ids) != len(prompts):
        raise ValueError("request ids and prompts must align")
    request_rows = request if isinstance(request, tuple) else (request,) * len(request_ids)
    if len(request_rows) != len(request_ids):
        raise ValueError("individual requests must align with request ids")
    states = {
        int(request_id): SimpleNamespace(request_id=int(request_id))
        for request_id in request_ids
    }
    if isinstance(request, tuple):
        for request_id, row_request, prompt in zip(
            request_ids,
            request_rows,
            prompts,
            strict=True,
        ):
            runner.register_batch(
                (int(request_id),),
                row_request,
                prompt_rows=(prompt,),
            )
    else:
        runner.register_batch(request_ids, request, prompt_rows=prompts)
    for request_id in request_ids:
        runner.reserve_admission(states[int(request_id)])
    runner.prefill_batch(_prefill_work(request_ids, prompts), commit=True)

    generated: dict[int, list[int]] = {int(request_id): [] for request_id in request_ids}
    active = request_ids
    transitions = 0
    while active:
        transitions += 1
        if transitions > max(int(row.max_tokens) for row in request_rows) + 1:
            raise RuntimeError("native sampler gate exceeded its decode transition budget")
        outputs = runner.decode_batch(_decode_work(active), commit=True)
        for output in outputs:
            generated[int(output.request_id)].append(int(output.token_id))
        active = tuple(
            int(request_id)
            for request_id in active
            if not bool(runner._rows[int(request_id)].slot.done)
        )

    # Packed c>N keeps canonical state in the owner workspace between steps.
    # Scatter it once before byte-level same-physical-shape comparisons.
    runner._flush_all_packed_owners()
    row_payloads: list[dict[str, Any]] = []
    captured: dict[int, dict[str, Any]] = {}
    for request_id in request_ids:
        row = runner._rows[int(request_id)]
        if row.slot is None or row.lease is None or row.sampling_state is None:
            raise RuntimeError("native sampler row lost resident state")
        captured[int(request_id)] = _capture_state(row.lease.session)
        chunk = runner._native_stream_chunk(row)
        detailed = runner._native_output(
            row,
            SimpleNamespace(finish_reason="length", finish_details=None),
        )
        decode_state = detailed.telemetry.to_json_dict()["decode_state"]
        allocation = row.kv_allocation
        if allocation is None:
            raise RuntimeError("native sampler row lost device KV allocation")
        block_ids = tuple(int(block_id) for block_id in allocation.block_ids)
        row_payloads.append(
            {
                "request_id": int(request_id),
                "row_index": int(row.row_index),
                "generated_token_ids": list(generated[int(request_id)]),
                "samples": [
                    {
                        "token_id": int(sample.token_id),
                        "logit": float(sample.logit),
                        "logprob": (
                            None if sample.logprob is None else float(sample.logprob)
                        ),
                        "top_logprobs": [
                            [int(token_id), float(logprob)]
                            for token_id, logprob in sample.top_logprobs
                        ],
                        "mode": str(sample.mode.value),
                        "candidate_count": int(sample.candidate_count),
                    }
                    for sample in row.samples
                ],
                "sampling_state": {
                    "step_index": int(row.sampling_state.step_index),
                    "generated_tokens": [
                        int(token) for token in row.sampling_state.generated_tokens
                    ],
                    "forced_tokens": [
                        int(token) for token in row.sampling_state.forced_tokens
                    ],
                },
                "finish_details": (
                    None
                    if detailed.finish_details is None
                    else detailed.finish_details.to_json_dict()
                ),
                "decode_state": decode_state,
                "token_logprobs": [
                    {
                        "token_id": int(token.token_id),
                        "token_text": str(token.token_text),
                        "logprob": (
                            None if token.logprob is None else float(token.logprob)
                        ),
                        "top_logprobs": [
                            {
                                "token_id": int(candidate_id),
                                "token_text": str(candidate_text),
                                "logprob": float(candidate_logprob),
                            }
                            for candidate_id, candidate_text, candidate_logprob in token.top_logprobs
                        ],
                    }
                    for token in detailed.token_logprobs
                ],
                "stream_token_logprob_count": len(chunk.token_logprobs),
                "kv_block_ids": list(block_ids),
                "kv_refcounts_before_release": [
                    int(runner.kv_pool.refcount(block_id)) for block_id in block_ids
                ],
            }
        )

    before_release = runner.kv_pool.stats.to_json_dict()
    for request_id in request_ids:
        runner.rollback_admission(states[int(request_id)])
        runner._rows.pop(int(request_id))
    after_release = runner.kv_pool.stats.to_json_dict()
    payload = {
        "request_ids": list(request_ids),
        "transitions": transitions,
        "rows": row_payloads,
        "pool_before_release": before_release,
        "pool_after_release": after_release,
    }
    return payload, captured


def _native_request(
    prompts: tuple[tuple[int, ...], ...],
    *,
    max_tokens: int,
    row_seeds: tuple[int, ...],
):
    from hipengine.generation.registry import GenerationRequest

    return GenerationRequest(
        prompts=prompts,
        max_tokens=int(max_tokens),
        temperature=0.85,
        top_k=8,
        top_p=0.82,
        min_p=0.08,
        seed=17,
        row_seeds=row_seeds,
        repetition_penalty=1.05,
        presence_penalty=0.1,
        frequency_penalty=0.02,
        logit_bias=((9709, 0.2), (9710, -0.15)),
        suppress_token_ids=(0, 1),
        min_tokens=2,
        eos_token_id=248044,
        ignore_eos=True,
        logprobs=True,
        top_logprobs=3,
    )


def _forced_request(prompt: tuple[int, ...], token_ids: tuple[int, ...]):
    from hipengine.generation.registry import GenerationRequest

    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=len(token_ids),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        forced_tokens_pending=token_ids,
        forced_token_reason="native_sampler_teacher_forced_oracle",
    )


def _row_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["row_index"]): row for row in payload["rows"]}


def _native_row_telemetry_exact(row: dict[str, Any]) -> bool:
    decode = row["decode_state"]
    samples = row["samples"]
    return bool(
        decode.get("sampler_mode") == "gpu_sample"
        and decode.get("execution_path") == "gguf_packed_ar_native_sampler_decode"
        and decode.get("native_sampler_rows") is True
        and decode.get("full_vocab_logits_d2h") is False
        and int(decode.get("logits_d2h_bytes", -1)) == 0
        and "sampler_fallback_reason" not in decode
        and row["finish_details"] is not None
        and row["finish_details"].get("sampler_mode") == "gpu_sample"
        and len(row["token_logprobs"]) == len(row["generated_token_ids"])
        and int(row["stream_token_logprob_count"]) == 1
        and all(sample["mode"] == "gpu_sample" for sample in samples)
        and all(sample["logprob"] is not None for sample in samples)
        and all(len(sample["top_logprobs"]) <= 3 for sample in samples)
        and row["sampling_state"]["generated_tokens"]
        == row["generated_token_ids"]
        and row["sampling_state"]["step_index"]
        == len(row["generated_token_ids"])
        and not row["sampling_state"]["forced_tokens"]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine import LLM

    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    prompt_tokens = int(args.prompt_tokens)
    max_tokens = int(args.max_tokens)
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    if max_tokens < 3:
        raise ValueError("max_tokens must be at least three for stop/EOS gates")
    if prompt_tokens + max_tokens - 1 > int(args.max_sequence_length):
        raise ValueError("max_sequence_length does not cover the gate trajectory")

    prompts = tuple(
        (int(args.prompt_token_id) + row_index,) * prompt_tokens
        for row_index in range(4)
    )
    row_seeds = (17, 29, 43, 71)
    native_request = _native_request(
        prompts,
        max_tokens=max_tokens,
        row_seeds=row_seeds,
    )
    os.environ["HIPENGINE_QWEN35_NATIVE_SAMPLER"] = "1"
    llm = LLM(
        str(model),
        backend=str(args.backend),
        quant=str(args.quant),
        max_active_requests=4,
        prefix_cache="off",
    )
    runner = None
    payload: dict[str, Any] | None = None
    try:
        llm.prepare(max_sequence_length=int(args.max_sequence_length))
        runner = llm._get_text_generator()._runner
        first, first_states = _run_rows(
            runner,
            request_ids=(1001, 1002, 1003, 1004),
            request=native_request,
            prompts=prompts,
        )
        second, second_states = _run_rows(
            runner,
            request_ids=(1101, 1102, 1103, 1104),
            request=native_request,
            prompts=prompts,
        )
        first_rows = _row_map(first)
        second_rows = _row_map(second)
        repeat_mismatches: list[dict[str, Any]] = []
        for row_index in range(4):
            if (
                first_rows[row_index]["generated_token_ids"]
                != second_rows[row_index]["generated_token_ids"]
            ):
                repeat_mismatches.append(
                    {"row_index": row_index, "kind": "generated_token_ids"}
                )
            state_mismatches = _compare_states(
                first_states[1001 + row_index],
                second_states[1101 + row_index],
            )
            repeat_mismatches.extend(
                {
                    "row_index": row_index,
                    "kind": "state",
                    **mismatch,
                }
                for mismatch in state_mismatches
            )

        oracle_mismatches: list[dict[str, Any]] = []
        expected_rows = tuple(
            tuple(
                int(token)
                for token in first_rows[row_index]["generated_token_ids"]
            )
            for row_index in range(4)
        )
        oracle, oracle_states = _run_rows(
            runner,
            request_ids=(1201, 1202, 1203, 1204),
            request=tuple(
                _forced_request(prompt, token_ids)
                for prompt, token_ids in zip(prompts, expected_rows, strict=True)
            ),
            prompts=prompts,
        )
        oracle_rows = oracle["rows"]
        for row_index, oracle_row in enumerate(oracle_rows):
            if oracle_row["generated_token_ids"] != list(expected_rows[row_index]):
                oracle_mismatches.append(
                    {"row_index": row_index, "kind": "generated_token_ids"}
                )
            state_mismatches = _compare_states(
                first_states[1001 + row_index],
                oracle_states[1201 + row_index],
            )
            oracle_mismatches.extend(
                {
                    "row_index": row_index,
                    "kind": "state",
                    **mismatch,
                }
                for mismatch in state_mismatches
            )

        stop_token = int(first_rows[0]["generated_token_ids"][1])
        stop_length = first_rows[0]["generated_token_ids"].index(stop_token) + 1
        stop_request = replace(
            native_request,
            prompts=(prompts[0],),
            row_seeds=(row_seeds[0],),
            stop_token_ids=(stop_token,),
        )
        stop_run, _ = _run_rows(
            runner,
            request_ids=(1301,),
            request=stop_request,
            prompts=(prompts[0],),
        )
        stop_row = stop_run["rows"][0]
        stop_exact = bool(
            stop_row["generated_token_ids"]
            == first_rows[0]["generated_token_ids"][:stop_length]
            and stop_row["finish_details"].get("reason") == "stop"
            and stop_row["finish_details"].get("stop_sequence") == [stop_token]
            and _native_row_telemetry_exact(stop_row)
        )

        simple_control_request = replace(
            native_request,
            prompts=(prompts[0],),
            row_seeds=(row_seeds[0],),
            min_tokens=0,
            eos_token_id=248044,
            ignore_eos=True,
        )
        simple_control, _ = _run_rows(
            runner,
            request_ids=(1401,),
            request=simple_control_request,
            prompts=(prompts[0],),
        )
        simple_ids = simple_control["rows"][0]["generated_token_ids"]
        eos_token = int(simple_ids[2])
        eos_length = simple_ids.index(eos_token) + 1
        eos_request = replace(
            simple_control_request,
            ignore_eos=False,
            eos_token_id=eos_token,
        )
        eos_run, _ = _run_rows(
            runner,
            request_ids=(1402,),
            request=eos_request,
            prompts=(prompts[0],),
        )
        eos_row = eos_run["rows"][0]
        eos_exact = bool(
            eos_row["generated_token_ids"] == simple_ids[:eos_length]
            and eos_row["finish_details"].get("reason") == "eos"
            and int(eos_row["finish_details"].get("eos_token_id", -1)) == eos_token
            and _native_row_telemetry_exact(eos_row)
        )

        observability = runner.observability_snapshot()
        final_pool = runner.kv_pool.stats.to_json_dict()
        native_rows = [*first["rows"], *second["rows"], stop_row, eos_row]
        native_telemetry_exact = all(
            _native_row_telemetry_exact(row) for row in native_rows
        )
        host_oracle_fallback_exact = all(
            row["decode_state"].get("sampler_mode") == "processed_argmax"
            and row["decode_state"].get("native_sampler_rows") is False
            and row["decode_state"].get("full_vocab_logits_d2h") is True
            and int(row["decode_state"].get("logits_d2h_bytes", 0)) > 0
            and row["decode_state"].get("sampler_fallback_reason")
            == "processed_logits_required"
            for row in oracle_rows
        )
        ownership_exact = bool(
            int(final_pool["refcounted_pages"]) == 0
            and int(final_pool["pinned_pages"]) == 0
            and int(final_pool["cow_fork_events"]) == 0
            and not runner.active_request_ids
            and int(runner.available_session_count) == 4
            and all(
                int(run_payload["pool_after_release"]["refcounted_pages"]) == 0
                for run_payload in (
                    first,
                    second,
                    oracle,
                    stop_run,
                    simple_control,
                    eos_run,
                )
            )
        )
        counts = observability["routes"]["counts"]
        batch_route_exact = bool(
            int(counts["native_sampler_batch_launches"])
            >= 2 * (max_tokens - 1)
            and int(counts["native_sampler_requests"]) >= 11
            and int(counts["native_sampler_row_launches"])
            >= 4 * 2 + 3
        )
        passed = bool(
            not repeat_mismatches
            and not oracle_mismatches
            and native_telemetry_exact
            and host_oracle_fallback_exact
            and stop_exact
            and eos_exact
            and ownership_exact
            and batch_route_exact
        )
        payload = {
            "schema": 1,
            "kind": "gguf_native_sampler_correctness_gate",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if passed else "failed",
            "passed": passed,
            "correctness_claim": True,
            "performance_claim": False,
            "timing_claim": False,
            "model": str(model),
            "quant": str(args.quant),
            "backend": str(args.backend),
            "target_arch": str(runner.generator.target_arch),
            "hardware": _HARDWARE_LABELS.get(str(args.backend), str(args.backend)),
            "workload": {
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
                "physical_rows": 4,
                "row_seeds": list(row_seeds),
                "temperature": 0.85,
                "top_k": 8,
                "top_p": 0.82,
                "min_p": 0.08,
                "processors": [
                    "logit_bias",
                    "repetition_penalty",
                    "presence_penalty",
                    "frequency_penalty",
                    "suppress_token_ids",
                    "min_tokens",
                ],
                "logprobs": True,
                "top_logprobs": 3,
            },
            "fixed_seed_repeat": {
                "exact": not repeat_mismatches,
                "mismatches": repeat_mismatches,
                "first": first,
                "second": second,
            },
            "forced_host_oracle": {
                "exact": not oracle_mismatches,
                "fallback_exact": host_oracle_fallback_exact,
                "mismatches": oracle_mismatches,
                "rows": oracle_rows,
            },
            "finish_routes": {
                "stop_exact": stop_exact,
                "stop": stop_row,
                "eos_exact": eos_exact,
                "eos": eos_row,
            },
            "telemetry": {
                "native_exact": native_telemetry_exact,
                "batch_route_exact": batch_route_exact,
                "observability": observability,
            },
            "ownership": {
                "exact": ownership_exact,
                "final_pool": final_pool,
                "active_request_ids": list(runner.active_request_ids),
                "available_sessions": int(runner.available_session_count),
            },
            "mechanical_checks": {
                "gpu_sampler_test": (
                    "HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 "
                    "HIPENGINE_COMPILER_VERSION_FILE=/tmp/agentic-w7900-hipcc-version.txt "
                    "uv run pytest -q "
                    "tests/test_gpu_sampler_kernel.py::"
                    "test_gguf_native_sampler_workspace_batches_supported_rows_and_fails_closed"
                ),
                "runtime_tests": (
                    "uv run pytest -q tests/test_generation_qwen35_gguf_sampling.py "
                    "tests/test_gpu_sampler_kernel.py"
                ),
            },
            "notes": [
                "Native admission is exactly supports_native_gpu_sampling; forced and dynamic queues stay host-backed.",
                "Every Conv/GDN byte and logical block-table-ordered live BF16 K/V byte is compared.",
                "Native rows read back selected IDs/logprobs only; no full-vocabulary logits row is copied to host.",
                "No timing from this correctness gate is a performance claim.",
            ],
        }
    finally:
        llm.close()
    assert payload is not None
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    command_args = list(sys.argv[1:] if argv is None else argv)
    payload["command"] = " ".join(
        [shlex.quote(sys.executable), shlex.quote(str(Path(__file__).resolve()))]
        + [shlex.quote(str(value)) for value in command_args]
    )
    payload["environment"] = {
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "HIPENGINE_COMPILER_VERSION_FILE": os.environ.get(
            "HIPENGINE_COMPILER_VERSION_FILE"
        ),
        "HIPENGINE_QWEN35_NATIVE_SAMPLER": os.environ.get(
            "HIPENGINE_QWEN35_NATIVE_SAMPLER"
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
