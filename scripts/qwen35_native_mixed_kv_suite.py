#!/usr/bin/env python3
# ruff: noqa: E402
"""Native BF16-vs-compressed KV fidelity suite for GGUF and PARO.

The harness runs the committed ten-prompt mtpbench category corpus plus the
``mixed_v1`` control at one exact prompt/decode shape. Reference tokens are
teacher-forced into the candidate so every compared logit row has an identical
token history. Timings are diagnostic only; performance claims use the
dedicated matched benchmark harnesses.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.kvcache import ResolvedKVPolicy, resolve_kv_policy
from hipengine.loading import load_gguf_index
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
)
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoNextTokenRunner,
    Qwen35ParoResidentSession,
)
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.gguf_mtp_category_bench import DEFAULT_HELDOUT_PROMPT_IDS, load_prompt_rows
from scripts.qwen35_gguf_kv_asymmetric_suite import PromptCase, _build_prompt_cases
from scripts.qwen35_kv_policy_args import kv_policy_json

DEFAULT_GGUF_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_PARO_MODEL = Path(
    "/home/lhl/.cache/huggingface/hub/"
    "models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/"
    "snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1"
)
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl"


@dataclass
class PromptRun:
    prompt_id: str
    logits: np.ndarray
    generated_token_ids: list[int]
    decode_input_ids: list[int]
    elapsed_seconds: float


@dataclass
class PolicyRun:
    policy: ResolvedKVPolicy
    prompts: dict[str, PromptRun]
    elapsed_seconds: float
    layout_audit: dict[str, Any]


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"compiler version file {path} is empty")
    return text


def _parse_layer_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError("layer-index expression must not be empty")
    indices: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"invalid empty layer-index term in {value!r}")
        if "-" in token:
            bounds = token.split("-")
            if len(bounds) != 2 or not all(bound.isdigit() for bound in bounds):
                raise ValueError(f"invalid layer-index range {token!r}")
            start, end = (int(bound) for bound in bounds)
            if end < start:
                raise ValueError(f"descending layer-index range {token!r}")
            indices.extend(range(start, end + 1))
        elif token.isdigit():
            indices.append(int(token))
        else:
            raise ValueError(f"invalid layer index {token!r}")
    if len(indices) != len(set(indices)):
        raise ValueError(f"layer-index expression contains duplicates: {value!r}")
    return sorted(indices)


def _checked_logits(logits: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise FloatingPointError(f"{label} logits are empty or non-finite")
    return values.copy()


def _read_paro_logits(session: Qwen35ParoResidentSession) -> np.ndarray:
    values = np.empty((session.vocab_size,), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(values),
        DeviceBuffer(session.lm_logits.ptr, values.nbytes),
        runtime=session.runtime,
    )
    return _checked_logits(values, "PARO")


def _teacher_inputs(run: PromptRun, decode_steps: int) -> list[int]:
    if decode_steps <= 0:
        return []
    if len(run.generated_token_ids) < decode_steps:
        raise ValueError("reference run does not contain enough teacher tokens")
    return [int(token) for token in run.generated_token_ids[:decode_steps]]


def _gguf_layout_audit(
    session: Qwen35GGUFResidentSession,
    *,
    policy: ResolvedKVPolicy,
    expected_bf16_layers: Sequence[int] | None = None,
    expected_int8_layers: Sequence[int] | None = None,
    require_no_bf16_mirror: bool = False,
) -> dict[str, Any]:
    scratch = session.scratch
    if scratch is None:
        raise RuntimeError("GGUF mixed-KV audit requires resident scratch")
    bf16_indices: list[int] = []
    int8_indices: list[int] = []
    primary_bytes = 0
    scale_bytes = 0
    mirror_bytes = 0
    full_index = 0
    for key, value, metadata in zip(
        scratch.full_key_caches,
        scratch.full_value_caches,
        scratch.full_kv_scale_metadata,
        strict=True,
    ):
        if key is None or value is None:
            continue
        primary_bytes += int(key.nbytes) + int(value.nbytes)
        if metadata is None:
            bf16_indices.append(full_index)
        else:
            int8_indices.append(full_index)
        full_index += 1
    for key, value in zip(
        scratch.full_k_scale_caches,
        scratch.full_v_scale_caches,
        strict=True,
    ):
        if key is not None:
            scale_bytes += int(key.nbytes)
        if value is not None:
            scale_bytes += int(value.nbytes)
    for key, value in zip(
        scratch.full_bf16_mirror_key_caches,
        scratch.full_bf16_mirror_value_caches,
        strict=True,
    ):
        if key is not None:
            mirror_bytes += int(key.nbytes)
        if value is not None:
            mirror_bytes += int(value.nbytes)

    if expected_bf16_layers is None and policy.requested_storage == "tail4_hadamard_group32":
        expected_bf16_layers = list(range(max(0, full_index - 4)))
        expected_int8_layers = list(range(max(0, full_index - 4), full_index))
    expected_bf16 = (
        None
        if expected_bf16_layers is None
        else [int(value) for value in expected_bf16_layers]
    )
    expected_int8 = (
        None
        if expected_int8_layers is None
        else [int(value) for value in expected_int8_layers]
    )
    fixed_layer_policy_passed = bool(
        expected_bf16 is None
        or (bf16_indices == expected_bf16 and int8_indices == expected_int8)
    )
    complete_partition = bool(
        sorted((*bf16_indices, *int8_indices)) == list(range(full_index))
        and not set(bf16_indices).intersection(int8_indices)
    )
    expected_layout = (
        "tail4_hadamard_group32"
        if policy.requested_storage == "tail4_hadamard_group32"
        else "uniform"
    )
    expected_granularity = str(policy.scale_granularity)
    scale_granularity_passed = all(
        metadata is None or metadata.granularity == expected_granularity
        for metadata in scratch.full_kv_scale_metadata
    )
    oracle_buffers = len(getattr(session, "_int8_prefill_oracle_buffers", {}))
    mirror_requirement_passed = not require_no_bf16_mirror or mirror_bytes == 0
    passed = bool(
        getattr(scratch, "kv_storage_layout", "uniform") == expected_layout
        and complete_partition
        and fixed_layer_policy_passed
        and scale_granularity_passed
        and mirror_requirement_passed
        and oracle_buffers == 0
    )
    return {
        "passed": passed,
        "storage_layout": getattr(scratch, "kv_storage_layout", "uniform"),
        "bf16_full_attention_indices": bf16_indices,
        "int8_full_attention_indices": int8_indices,
        "expected_bf16_full_attention_indices": expected_bf16,
        "expected_int8_full_attention_indices": expected_int8,
        "fixed_layer_policy_passed": fixed_layer_policy_passed,
        "complete_layer_partition": complete_partition,
        "scale_granularity": expected_granularity,
        "scale_granularity_passed": scale_granularity_passed,
        "primary_kv_bytes": primary_bytes,
        "scale_bytes": scale_bytes,
        "total_kv_bytes": primary_bytes + scale_bytes,
        "bf16_mirror_bytes": mirror_bytes,
        "require_no_bf16_mirror": bool(require_no_bf16_mirror),
        "bf16_mirror_requirement_passed": mirror_requirement_passed,
        "int8_prefill_oracle_buffer_count": oracle_buffers,
    }


def _run_gguf_policy(
    *,
    runner: Qwen35GGUFFullStackRunner,
    cases: Sequence[PromptCase],
    policy: ResolvedKVPolicy,
    decode_steps: int,
    max_sequence_length: int,
    compiler_version: str | None,
    require_cached_build: bool,
    teacher_runs: dict[str, PromptRun] | None = None,
    expected_bf16_layers: Sequence[int] | None = None,
    expected_int8_layers: Sequence[int] | None = None,
    require_no_bf16_mirror: bool = False,
) -> PolicyRun:
    prompts: dict[str, PromptRun] = {}
    started = time.perf_counter()
    with Qwen35GGUFResidentSession(
        runner.model_path,
        runtime=runner.runtime,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        backend=runner.backend,
        shared_runner=runner,
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=False,
        use_gemv_decode=False,
        kv_policy=policy.create_policy(),
        kv_scale_dtype=policy.scale_dtype,
        kv_scale_granularity=policy.scale_granularity,
    ) as session:
        for case_index, case in enumerate(cases, 1):
            print(
                f"[native-mixed-kv] gguf {policy.requested_storage} "
                f"prompt {case_index}/{len(cases)}: {case.prompt_id}",
                file=sys.stderr,
                flush=True,
            )
            session.reset()
            case_started = time.perf_counter()
            rows: list[np.ndarray] = []
            generated: list[int] = []
            decode_inputs: list[int] = []
            first = session.prefill(
                list(case.tokens),
                use_bulk=True,
                bulk_attention_mode="bulk",
                return_logits=True,
            )
            rows.append(_checked_logits(first.logits, f"GGUF {case.prompt_id} prefill"))
            generated.append(int(first.token_id))
            teachers = None
            if teacher_runs is not None:
                teachers = _teacher_inputs(teacher_runs[case.prompt_id], decode_steps)
            current = first
            for step_index in range(decode_steps):
                input_id = int(current.token_id) if teachers is None else int(teachers[step_index])
                decode_inputs.append(input_id)
                current = session.step(input_id, return_logits=True)
                generated.append(int(current.token_id))
                rows.append(
                    _checked_logits(current.logits, f"GGUF {case.prompt_id} decode[{step_index}]")
                )
            prompts[case.prompt_id] = PromptRun(
                prompt_id=case.prompt_id,
                logits=np.vstack(rows),
                generated_token_ids=generated,
                decode_input_ids=decode_inputs,
                elapsed_seconds=time.perf_counter() - case_started,
            )
        layout_audit = (
            _gguf_layout_audit(
                session,
                policy=policy,
                expected_bf16_layers=expected_bf16_layers,
                expected_int8_layers=expected_int8_layers,
                require_no_bf16_mirror=require_no_bf16_mirror,
            )
            if policy.requested_storage != "bf16"
            else {"passed": True, "required": False}
        )
    gc.collect()
    return PolicyRun(
        policy=policy,
        prompts=prompts,
        elapsed_seconds=time.perf_counter() - started,
        layout_audit=layout_audit,
    )


def _run_paro_policy(
    *,
    runner: Qwen35ParoNextTokenRunner,
    cases: Sequence[PromptCase],
    policy: ResolvedKVPolicy,
    decode_steps: int,
    max_sequence_length: int,
    compiler_version: str | None,
    require_cached_build: bool,
    teacher_runs: dict[str, PromptRun] | None = None,
) -> PolicyRun:
    prompts: dict[str, PromptRun] = {}
    started = time.perf_counter()
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=40,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=512, auto_tune_chunk_sizes=True),
        kv_policy=policy.create_policy(),
        kv_scale_dtype=policy.scale_dtype,
        kv_scale_granularity=policy.scale_granularity,
    ) as session:
        for case_index, case in enumerate(cases, 1):
            print(
                f"[native-mixed-kv] paro {policy.requested_storage} "
                f"prompt {case_index}/{len(cases)}: {case.prompt_id}",
                file=sys.stderr,
                flush=True,
            )
            session.reset()
            session._resolve_prefill_config_for_length(len(case.tokens))
            case_started = time.perf_counter()
            rows: list[np.ndarray] = []
            generated: list[int] = []
            decode_inputs: list[int] = []
            first = session.prefill_native(list(case.tokens), sample=True)
            if first is None:
                raise RuntimeError(f"PARO {case.prompt_id} prefill produced no token")
            rows.append(_read_paro_logits(session))
            generated.append(int(first.token_id))
            teachers = None
            if teacher_runs is not None:
                teachers = _teacher_inputs(teacher_runs[case.prompt_id], decode_steps)
            current = first
            for step_index in range(decode_steps):
                input_id = int(current.token_id) if teachers is None else int(teachers[step_index])
                decode_inputs.append(input_id)
                current = session.step(
                    input_id,
                    position=len(case.tokens) + step_index,
                    sample=True,
                )
                if current is None:
                    raise RuntimeError(f"PARO {case.prompt_id} decode produced no token")
                generated.append(int(current.token_id))
                rows.append(_read_paro_logits(session))
            prompts[case.prompt_id] = PromptRun(
                prompt_id=case.prompt_id,
                logits=np.vstack(rows),
                generated_token_ids=generated,
                decode_input_ids=decode_inputs,
                elapsed_seconds=time.perf_counter() - case_started,
            )
        layout_audit = (
            session.kv_memory_audit()
            if policy.requested_storage == "tail4_hadamard_group32"
            else {"passed": True, "required": False}
        )
    gc.collect()
    return PolicyRun(
        policy=policy,
        prompts=prompts,
        elapsed_seconds=time.perf_counter() - started,
        layout_audit=layout_audit,
    )


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _prompt_comparison(
    case: PromptCase,
    reference: PromptRun,
    candidate: PromptRun,
    *,
    kl_threshold: float,
    top1_threshold: float,
) -> dict[str, Any]:
    metrics = evaluate_logits(
        reference.logits,
        candidate.logits,
        kl_threshold=kl_threshold,
        top1_threshold=top1_threshold,
    )
    reference_top1 = np.argmax(reference.logits, axis=-1).astype(np.int64)
    candidate_top1 = np.argmax(candidate.logits, axis=-1).astype(np.int64)
    return {
        "id": case.prompt_id,
        "category": case.category,
        "split": case.split,
        "profile": case.profile,
        "prompt_tokens": len(case.tokens),
        "prompt_token_sha256": case.token_sha256,
        "positions": int(reference.logits.shape[0]),
        "passed": bool(metrics.passed),
        "mean_kl": float(metrics.kl_mean),
        "max_kl": float(metrics.kl_max),
        "top1_agreement": float(metrics.top1_agreement),
        "reference_top1": reference_top1.tolist(),
        "candidate_top1": candidate_top1.tolist(),
        "top1_mismatch_count": int(np.count_nonzero(reference_top1 != candidate_top1)),
        "reference_generated_token_ids": reference.generated_token_ids,
        "candidate_generated_token_ids": candidate.generated_token_ids,
        "candidate_teacher_input_ids": candidate.decode_input_ids,
        "reference_logits_sha256": _array_sha256(reference.logits),
        "candidate_logits_sha256": _array_sha256(candidate.logits),
        "reference_elapsed_seconds": float(reference.elapsed_seconds),
        "candidate_elapsed_seconds": float(candidate.elapsed_seconds),
    }


def _engine_summary(
    *,
    engine: str,
    cases: Sequence[PromptCase],
    reference: PolicyRun,
    candidate: PolicyRun,
    kl_threshold: float,
    top1_threshold: float,
) -> dict[str, Any]:
    rows = [
        _prompt_comparison(
            case,
            reference.prompts[case.prompt_id],
            candidate.prompts[case.prompt_id],
            kl_threshold=kl_threshold,
            top1_threshold=top1_threshold,
        )
        for case in cases
    ]
    all_reference = np.vstack([reference.prompts[case.prompt_id].logits for case in cases])
    all_candidate = np.vstack([candidate.prompts[case.prompt_id].logits for case in cases])
    aggregate = evaluate_logits(
        all_reference,
        all_candidate,
        kl_threshold=kl_threshold,
        top1_threshold=top1_threshold,
    )
    passed = bool(all(row["passed"] for row in rows) and candidate.layout_audit.get("passed", False))
    return {
        "engine": engine,
        "passed": passed,
        "reference_policy": kv_policy_json(reference.policy),
        "candidate_policy": kv_policy_json(candidate.policy),
        "reference_elapsed_seconds": float(reference.elapsed_seconds),
        "candidate_elapsed_seconds": float(candidate.elapsed_seconds),
        "candidate_layout_audit": candidate.layout_audit,
        "rows": rows,
        "summary": {
            "prompt_count": len(rows),
            "positions": int(all_reference.shape[0]),
            "passed_prompt_count": sum(bool(row["passed"]) for row in rows),
            "mean_kl": float(aggregate.kl_mean),
            "max_kl": float(aggregate.kl_max),
            "top1_agreement": float(aggregate.top1_agreement),
            "worst_prompt_mean_kl": max(float(row["mean_kl"]) for row in rows),
            "worst_prompt_max_kl": max(float(row["max_kl"]) for row in rows),
            "minimum_prompt_top1_agreement": min(float(row["top1_agreement"]) for row in rows),
            "failing_prompt_ids": [str(row["id"]) for row in rows if not row["passed"]],
        },
    }


def _command(args: argparse.Namespace) -> str:
    command = (
        "python3 scripts/qwen35_native_mixed_kv_suite.py"
        f" --engine {args.engine}"
        f" --backend {args.backend}"
        f" --gguf-model {args.gguf_model}"
        f" --paro-model {args.paro_model}"
        f" --prompts {args.prompts}"
        f" --prompt-length {args.prompt_length}"
        f" --decode-steps {args.decode_steps}"
        f" --candidate-kv-storage {args.candidate_kv_storage}"
        f" --kl-threshold {args.kl_threshold}"
        f" --top1-threshold {args.top1_threshold}"
    )
    if args.max_sequence_length:
        command += f" --max-sequence-length {args.max_sequence_length}"
    if args.require_no_bf16_mirror:
        command += " --require-no-bf16-mirror"
    if args.expected_bf16_full_layers is not None:
        command += f" --expected-bf16-full-layers {args.expected_bf16_full_layers}"
    if args.expected_int8_full_layers is not None:
        command += f" --expected-int8-full-layers {args.expected_int8_full_layers}"
    if args.compiler_version_file is not None:
        command += f" --compiler-version-file {args.compiler_version_file}"
    if args.require_cached_build:
        command += " --require-cached-build"
    if args.json is not None:
        command += f" --json {args.json}"
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.prompt_length <= 0 or args.decode_steps < 0:
        raise ValueError("prompt length must be positive and decode steps non-negative")
    minimum_sequence_length = int(args.prompt_length) + int(args.decode_steps) + 2
    if args.max_sequence_length < 0:
        raise ValueError("max sequence length must be non-negative")
    if args.max_sequence_length and args.max_sequence_length < minimum_sequence_length:
        raise ValueError(
            f"max sequence length {args.max_sequence_length} is below required "
            f"{minimum_sequence_length}"
        )
    if args.require_no_bf16_mirror and args.engine not in {"gguf", "both"}:
        raise ValueError("--require-no-bf16-mirror requires a GGUF engine run")
    expected_bf16_layers = _parse_layer_indices(args.expected_bf16_full_layers)
    expected_int8_layers = _parse_layer_indices(args.expected_int8_full_layers)
    if (expected_bf16_layers is None) != (expected_int8_layers is None):
        raise ValueError("expected BF16 and INT8 layer expressions must be supplied together")
    compiler_version = _read_compiler_version(args.compiler_version_file)
    prompt_rows = load_prompt_rows(args.prompts)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(args.gguf_model))
    cases = _build_prompt_cases(
        tokenizer,
        prompt_rows,
        prompt_length=int(args.prompt_length),
        include_mixed_v1=True,
        heldout_ids=DEFAULT_HELDOUT_PROMPT_IDS,
    )
    max_sequence_length = int(args.max_sequence_length or minimum_sequence_length)
    reference_policy = resolve_kv_policy("bf16")
    candidate_policy = resolve_kv_policy(
        args.candidate_kv_storage,
        scale_dtype="fp16",
    )
    engines: dict[str, Any] = {}
    started = time.perf_counter()

    if args.engine in {"gguf", "both"}:
        gguf_runner = Qwen35GGUFFullStackRunner(
            args.gguf_model,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            backend=args.backend,
        )
        try:
            gguf_reference = _run_gguf_policy(
                runner=gguf_runner,
                cases=cases,
                policy=reference_policy,
                decode_steps=args.decode_steps,
                max_sequence_length=max_sequence_length,
                compiler_version=compiler_version,
                require_cached_build=args.require_cached_build,
            )
            gguf_candidate = _run_gguf_policy(
                runner=gguf_runner,
                cases=cases,
                policy=candidate_policy,
                decode_steps=args.decode_steps,
                max_sequence_length=max_sequence_length,
                compiler_version=compiler_version,
                require_cached_build=args.require_cached_build,
                teacher_runs=gguf_reference.prompts,
                expected_bf16_layers=expected_bf16_layers,
                expected_int8_layers=expected_int8_layers,
                require_no_bf16_mirror=args.require_no_bf16_mirror,
            )
            engines["gguf"] = _engine_summary(
                engine="gguf",
                cases=cases,
                reference=gguf_reference,
                candidate=gguf_candidate,
                kl_threshold=args.kl_threshold,
                top1_threshold=args.top1_threshold,
            )
        finally:
            gguf_runner.close()

    if args.engine in {"paro", "both"}:
        paro_runner = Qwen35ParoNextTokenRunner(
            args.paro_model,
            shared_expert_format="packed_paro_w4",
            backend=args.backend,
        )
        paro_reference = _run_paro_policy(
            runner=paro_runner,
            cases=cases,
            policy=reference_policy,
            decode_steps=args.decode_steps,
            max_sequence_length=max_sequence_length,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
        paro_candidate = _run_paro_policy(
            runner=paro_runner,
            cases=cases,
            policy=candidate_policy,
            decode_steps=args.decode_steps,
            max_sequence_length=max_sequence_length,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            teacher_runs=paro_reference.prompts,
        )
        engines["paro"] = _engine_summary(
            engine="paro",
            cases=cases,
            reference=paro_reference,
            candidate=paro_candidate,
            kl_threshold=args.kl_threshold,
            top1_threshold=args.top1_threshold,
        )

    passed = bool(engines and all(bool(payload["passed"]) for payload in engines.values()))
    return {
        "schema": 1,
        "mode": "qwen35_native_mixed_kv_suite",
        "status": "accepted" if passed else "rejected_correctness",
        "passed": passed,
        "performance_claim": False,
        "command": _command(args),
        "hardware": f"configured backend {args.backend}",
        "backend": args.backend,
        "environment": {
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "hipengine_hip_arch": os.environ.get("HIPENGINE_HIP_ARCH"),
            "compiler_version_first_line": (
                None if compiler_version is None else compiler_version.splitlines()[0]
            ),
            "hipengine_gguf_int8_kv_bf16_prefix_full_layers": os.environ.get(
                "HIPENGINE_GGUF_INT8_KV_BF16_PREFIX_FULL_LAYERS"
            ),
            "hipengine_gguf_int8_kv_bf16_full_layers": os.environ.get(
                "HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS"
            ),
            "hipengine_gguf_int8_kv_allow_unverified_long": os.environ.get(
                "HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG"
            ),
        },
        "gguf_model": str(args.gguf_model),
        "paro_model": str(args.paro_model),
        "prompts": str(args.prompts),
        "prompt_ids": [case.prompt_id for case in cases],
        "prompt_length": int(args.prompt_length),
        "decode_steps": int(args.decode_steps),
        "max_sequence_length": max_sequence_length,
        "quality_thresholds": {
            "mean_kl_max": float(args.kl_threshold),
            "top1_agreement_min": float(args.top1_threshold),
            "all_prompts_must_pass": True,
        },
        "engines": engines,
        "elapsed_seconds": time.perf_counter() - started,
        "notes": [
            "Candidate decode consumes BF16 reference tokens; every logit comparison has matched token history.",
            "All ten committed natural prompts plus mixed_v1 must pass independently.",
            "Timings are diagnostic only and are not retained performance evidence.",
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("gguf", "paro", "both"), default="both")
    parser.add_argument(
        "--backend",
        choices=("hip_gfx1100", "hip_gfx1151"),
        default="hip_gfx1100",
    )
    parser.add_argument("--gguf-model", type=Path, default=DEFAULT_GGUF_MODEL)
    parser.add_argument("--paro-model", type=Path, default=DEFAULT_PARO_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=0,
        help=(
            "Resident capacity; 0 uses prompt length plus decode steps. "
            "Use >8192 to force GGUF long/no-mirror policy."
        ),
    )
    parser.add_argument(
        "--require-no-bf16-mirror",
        action="store_true",
        help="Fail the candidate layout audit if persistent BF16 mirror bytes remain.",
    )
    parser.add_argument(
        "--expected-bf16-full-layers",
        help="Required zero-based BF16 full-attention indices, e.g. 0-7.",
    )
    parser.add_argument(
        "--expected-int8-full-layers",
        help="Required zero-based INT8 full-attention indices, e.g. 8,9.",
    )
    parser.add_argument(
        "--candidate-kv-storage",
        choices=("int8_per_token_head", "tail4_hadamard_group32"),
        default="tail4_hadamard_group32",
    )
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
