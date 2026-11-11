#!/usr/bin/env python3
"""Task-quality gate for the direct INT8 prefill route vs the strict oracle-bridge.

R0404 execution-profile task gate (docs/EXECUTION-PROFILES.md §6.5). Runs the
committed RF1 long-context task packet free-running under two arms on the SAME
mirror-free INT8 KV resident session:

- strict: ``HIPENGINE_GGUF_INT8_PREFILL_DIRECT=0`` — the pure-INT8 oracle-bridge
  prefill (BF16 oracle pair + AOTriton), the route's declared strict fallback;
- candidate: ``HIPENGINE_GGUF_INT8_PREFILL_DIRECT=1`` — the direct INT8 prefill
  attention that reads the retained INT8 K/V store without the oracle bridge.

Each arm generates greedily and the answer is scored ``choice_exact``. The gate
binds a predeclared paired non-inferiority criterion per category: a category
regresses iff the candidate answers incorrectly a task the strict arm answered
correctly. It is a task-quality input, not a performance claim, and does not by
itself promote the route to an arithmetic default.

The session construction mirrors
``scripts/execution_profile_gguf_int8_direct_prefill_gate.py::_capture`` so the
candidate arm engages the same mirror-free INT8 direct route the numerical
envelope gate qualified; it fails closed if the route would not run (otherwise
both arms would be a BF16 bit-identity and the gate would measure nothing).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.execution_profile_gguf_int8_direct_prefill_gate import (
    DIRECT_ENV,
    PURE_INT8_ENV,
    _DIRECT_ROUTE_MIN_MAX_POSITIONS,
    _candidate,
    _rearm_bulk_workspace,
    route_environment,
)
from scripts.gguf_mtp_long_context_task_gate import (
    CATEGORIES,
    _TokenizerAdapter,
    _decode,
    load_tasks,
    score_task_output,
)
from scripts.qwen35_paro_kv_quality_smoke import _build_prompt_tokens

KIND = "hipengine_execution_profile_gguf_int8_direct_prefill_task_gate"
SCHEMA_VERSION = 1
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_SUITE = REPO_ROOT / "benchmarks/prompts/mtp-realworld-long-context.jsonl"


class TaskGateError(RuntimeError):
    """Raised when the task gate cannot be evaluated honestly."""


def _generate_greedy(
    session: Any,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    stop_token_ids: frozenset[int],
) -> list[int]:
    """Free-running greedy generation (prefill + step loop), stopping at EOS/EOT.

    Mirrors RF1 ``_run_ar`` but halts on a stop token so a reasoning model that
    finishes its answer early does not burn the budget on special tokens.
    """
    session.reset()
    first = session.prefill(list(prompt_ids), use_bulk=True, return_logits=False)
    output: list[int] = []
    token = int(first.token_id)
    while len(output) < int(max_new_tokens):
        output.append(token)
        if token in stop_token_ids:
            break
        token = int(session.step(token, return_logits=False).token_id)
    return output


@contextmanager
def _open_int8_session(
    *,
    model: Path,
    backend: str,
    kv_layout: str,
    max_sequence_length: int,
    attn_aotriton_min_tokens: int,
    use_wmma_prefill: bool,
    use_gemv_decode: bool,
    require_cached_build: bool,
    compiler_version: str | None,
) -> Iterator[tuple[Any, str, str]]:
    """Build the mirror-free INT8 resident session exactly as the numerical gate does."""
    from hipengine.kernels.backends import hip_target_arch_for_backend, resolve_backend
    from hipengine.kvcache import resolve_kv_policy
    from hipengine.loading.gguf import gguf_execution_fingerprint, scan_gguf
    from hipengine.models import resolve_model
    from hipengine.models.kv_capabilities import (
        KVCapabilityKey,
        model_artifact_identity,
        resolve_kv_capability,
    )
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    gguf_info = scan_gguf(model)
    if kv_layout == "tail4_hadamard":
        policy = resolve_kv_policy(
            "tail4_hadamard_group32",
            scale_dtype="fp16",
            scale_granularity="hadamard_group32",
        )
        kv_scale_dtype_arg = "fp16"
        kv_scale_granularity_arg = "hadamard_group32"
    else:
        policy = resolve_kv_policy(
            "int8_per_token_head",
            scale_dtype="fp32",
            scale_granularity="per_token_head",
        )
        kv_scale_dtype_arg = "fp32"
        kv_scale_granularity_arg = "per_token_head"

    identity = model_artifact_identity(model)
    if not identity.content_verified:
        raise TaskGateError(f"model identity unavailable: {identity.error}")
    capability_backend = resolve_backend(str(backend), warn=False)
    model_plugin = resolve_model(gguf_info.architecture or "")
    quant_name = str(getattr(gguf_info, "file_type_name", "") or "").strip().lower()
    if quant_name.startswith("mostly_"):
        quant_name = quant_name[len("mostly_"):]
    if not quant_name:
        raise TaskGateError("GGUF metadata does not expose file_type_name")
    capability_key = KVCapabilityKey(
        artifact_sha256=identity.sha256,
        artifact_size_bytes=identity.size_bytes,
        artifact_execution_fingerprint=gguf_execution_fingerprint(gguf_info),
        backend=capability_backend,
        target_arch=hip_target_arch_for_backend(capability_backend),
        weight_quant=f"gguf_{quant_name}",
        kv_storage=policy.storage_dtype.value,
        storage_layout=str(policy.storage_layout),
        scale_dtype=policy.scale_dtype.value,
        scale_granularity=str(policy.scale_granularity),
    )
    plugin_resolver = getattr(model_plugin, "resolve_kv_capability", None)
    if callable(plugin_resolver):
        kv_resolution = plugin_resolver(key=capability_key, artifact=identity)
    else:
        kv_resolution = resolve_kv_capability(
            (),
            key=capability_key,
            artifact=identity,
            declarations=getattr(model_plugin, "kv_capability_declarations", ()),
        )
    if kv_resolution.runtime_action != "admit":
        raise TaskGateError(
            "INT8 KV capability not admitted for this artifact/backend "
            f"(status={kv_resolution.status}): {kv_resolution.reason}; the direct "
            "INT8 prefill route cannot be task-evaluated on a BF16 fallback"
        )
    kv_capability = copy.deepcopy(kv_resolution.as_dict())

    with route_environment(PURE_INT8_ENV), Qwen35GGUFResidentSession(
        model,
        backend=str(backend),
        compiler_version=compiler_version,
        require_cached_build=bool(require_cached_build),
        max_sequence_length=max_sequence_length,
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=int(attn_aotriton_min_tokens)),
        kv_policy=policy.create_policy(),
        kv_capability=kv_capability,
        kv_scale_dtype=kv_scale_dtype_arg,
        kv_scale_granularity=kv_scale_granularity_arg,
        use_wmma_prefill=bool(use_wmma_prefill),
        use_gemv_decode=bool(use_gemv_decode),
    ) as session:
        if session.runner is None:
            raise TaskGateError("GGUF resident session closed during setup")
        if kv_layout == "tail4_hadamard":
            if (
                str(getattr(session, "kv_storage_layout", "")) != "tail4_hadamard_group32"
                or str(getattr(session, "kv_scale_granularity", "")) != "hadamard_group32"
            ):
                raise TaskGateError(
                    "tail4-Hadamard layout not engaged "
                    f"(layout={getattr(session, 'kv_storage_layout', None)}, "
                    f"granularity={getattr(session, 'kv_scale_granularity', None)}); "
                    "the direct Hadamard prefill leaf would not run"
                )
        elif (
            getattr(session, "int8_bf16_full_attention_layer_indices", ())
            or getattr(session, "int8_bf16_prefix_full_attention_layers", 0)
            or not getattr(session, "int8_kv_no_mirror_qualified", False)
        ):
            raise TaskGateError(
                "INT8 KV session will not engage mirror-free INT8 "
                f"(no_mirror_qualified={getattr(session, 'int8_kv_no_mirror_qualified', None)}, "
                f"prefix={getattr(session, 'int8_bf16_prefix_full_attention_layers', None)}, "
                f"indices={getattr(session, 'int8_bf16_full_attention_layer_indices', None)}); "
                "the direct INT8 prefill route would not run"
            )
        resolved_backend = str(session.runner.backend)
        target_arch = str(session.runner.target_arch)
        if resolved_backend != capability_backend:
            raise TaskGateError(
                f"capability resolved for backend {capability_backend!r} but the session "
                f"runs {resolved_backend!r}; the INT8 admission would not describe the route"
            )
        yield session, resolved_backend, target_arch


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model.is_file():
        raise TaskGateError(f"model does not exist: {args.model}")
    tasks = load_tasks(args.suite)
    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]
    if not tasks:
        raise TaskGateError("selected task suite is empty")
    candidate = _candidate(str(args.kernel), kv_layout=str(args.kv_layout))

    from hipengine.loading.gguf import scan_gguf
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    stop_token_ids = frozenset(
        int(value)
        for value in (
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "eot_token_id", None),
        )
        if value is not None
    )
    adapter = _TokenizerAdapter(tokenizer)
    expanded = {
        str(task["id"]): _build_prompt_tokens(
            adapter,
            task,
            context_tokens=int(args.context_tokens or task["target_context_tokens"]),
        )
        for task in tasks
    }
    maximum_prompt = max(len(tokens) for tokens, _meta in expanded.values())
    required_sequence = maximum_prompt + int(args.max_new_tokens) + 2
    max_sequence_length = max(_DIRECT_ROUTE_MIN_MAX_POSITIONS, required_sequence)

    compiler_version = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.read_text(encoding="utf-8")
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with _open_int8_session(
        model=args.model,
        backend=str(args.backend),
        kv_layout=str(args.kv_layout),
        max_sequence_length=max_sequence_length,
        attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens),
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=bool(args.use_gemv_decode),
        require_cached_build=bool(args.require_cached_build),
        compiler_version=compiler_version,
    ) as (session, resolved_backend, target_arch):
        for index, task in enumerate(tasks):
            task_id = str(task["id"])
            prompt_ids, prompt_metadata = expanded[task_id]
            task_started = time.perf_counter()
            arm_outputs: dict[str, list[int]] = {}
            for arm_name, env in (
                ("strict", candidate.strict_environment),
                ("candidate", candidate.environment),
            ):
                with route_environment(env):
                    _rearm_bulk_workspace(session)
                    arm_outputs[arm_name] = _generate_greedy(
                        session, prompt_ids, int(args.max_new_tokens), stop_token_ids
                    )
            strict_text = _decode(tokenizer, arm_outputs["strict"])
            candidate_text = _decode(tokenizer, arm_outputs["candidate"])
            strict_score = score_task_output(strict_text, task)
            candidate_score = score_task_output(candidate_text, task)
            regressed = bool(strict_score["passed"] and not candidate_score["passed"])
            rows.append(
                {
                    "id": task_id,
                    "category": str(task["category"]),
                    **prompt_metadata,
                    "expected_choice": str(task["expected_choice"]),
                    "max_new_tokens": int(args.max_new_tokens),
                    "strict": {
                        "output_token_ids": arm_outputs["strict"],
                        "output_text": strict_text,
                        "score": strict_score,
                    },
                    "candidate": {
                        "output_token_ids": arm_outputs["candidate"],
                        "output_text": candidate_text,
                        "score": candidate_score,
                    },
                    "output_ids_exact": arm_outputs["strict"] == arm_outputs["candidate"],
                    "regressed": regressed,
                    "wall_seconds": time.perf_counter() - task_started,
                }
            )
            print(
                f"{index + 1}/{len(tasks)} {task_id} [{task['category']}]: "
                f"strict={'PASS' if strict_score['passed'] else 'FAIL'} "
                f"candidate={'PASS' if candidate_score['passed'] else 'FAIL'} "
                f"{'REGRESSED' if regressed else 'ok'}",
                flush=True,
            )

    # Predeclared paired non-inferiority: a category regresses iff the candidate
    # misses a task the strict arm answered correctly. Per-category, no averaging.
    per_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        cat_rows = [row for row in rows if row["category"] == category]
        strict_correct = sum(bool(row["strict"]["score"]["passed"]) for row in cat_rows)
        candidate_correct = sum(bool(row["candidate"]["score"]["passed"]) for row in cat_rows)
        regressions = sum(bool(row["regressed"]) for row in cat_rows)
        per_category[category] = {
            "total": len(cat_rows),
            "strict_correct": strict_correct,
            "candidate_correct": candidate_correct,
            "regressions": regressions,
            "non_inferior": regressions == 0,
        }
    total = len(rows)
    strict_correct = sum(bool(row["strict"]["score"]["passed"]) for row in rows)
    candidate_correct = sum(bool(row["candidate"]["score"]["passed"]) for row in rows)
    regressions = sum(bool(row["regressed"]) for row in rows)
    non_inferior = all(bool(value["non_inferior"]) for value in per_category.values())
    status = "passed" if non_inferior else "failed"

    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="int8",
        command=command,
        environment={DIRECT_ENV: "1", **PURE_INT8_ENV},
    )
    measurement_valid = not bool(provenance.get("dirty", False))
    return {
        "schema": SCHEMA_VERSION,
        "kind": KIND,
        "status": status,
        "verdict": "pass" if non_inferior else "fail",
        "passed": non_inferior,
        "measurement_valid": measurement_valid,
        "criterion": (
            "per-category paired non-inferiority: a category regresses iff the candidate "
            "answers incorrectly a task the strict oracle-bridge arm answered correctly "
            "(choice_exact, greedy free-running); gate passes iff no category regresses"
        ),
        "candidate": {
            "name": candidate.name,
            "kv_layout": str(args.kv_layout),
            "kernel": str(args.kernel),
            "mechanism": candidate.mechanism,
            "strict_fallback": candidate.strict_fallback,
        },
        "suite": str(args.suite),
        "context_tokens": int(args.context_tokens or 0) or None,
        "summary": {
            "total": total,
            "strict_correct": strict_correct,
            "candidate_correct": candidate_correct,
            "regressions": regressions,
            "output_ids_exact_count": sum(bool(row["output_ids_exact"]) for row in rows),
            "non_inferior": non_inferior,
        },
        "per_category": per_category,
        "rows": rows,
        "provenance": provenance,
        "wall_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--kv-layout", choices=("per_token_head", "tail4_hadamard"), default="per_token_head")
    parser.add_argument("--kernel", choices=("flash", "sequential"), default="flash")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--context-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=512)
    parser.add_argument("--use-wmma-prefill", action="store_true")
    parser.add_argument("--use-gemv-decode", action="store_true")
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    command = [sys.executable, *sys.argv]
    try:
        payload = run(args, command=command)
    except TaskGateError as exc:
        print(f"task gate error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(
        f"status={payload['status']} measurement_valid={payload['measurement_valid']} "
        f"strict={payload['summary']['strict_correct']}/{payload['summary']['total']} "
        f"candidate={payload['summary']['candidate_correct']}/{payload['summary']['total']} "
        f"regressions={payload['summary']['regressions']}"
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())