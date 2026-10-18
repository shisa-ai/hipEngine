#!/usr/bin/env python3
"""Focused task review for Qwen4Exp AR batch width-4 free outputs.

This is a diagnostic harness, not a production gate.  It generates longer
greedy free outputs for the four prompt IDs whose 32-token free outputs
diverged in the passed 450-row width-4 numerical run, while preserving the
original fixture-order width-4 groups and their companions.  It then executes
any complete generated code in a bounded, isolated subprocess.

It does not change the runtime, dispatch, execution profile, or calibrated
gate.  The numerical gate is not rerun here; the committed compact artifact is
the reference for the passed numerical evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.qwen4exp_ar_batch_candidate import (
    BatchCandidateRequest,
    Qwen4ExpARBatchCandidate,
    plan_batch_composition,
)
from scripts.qwen4exp_canonical_ar_bench import (
    _git_metadata,
    _host_metadata,
    _write_json,
    token_ids_sha256,
)
from scripts.qwen4exp_layer2_profile_gate import (
    _free_trajectory,
    _json_value,
    _make_generator,
)

DIVERGENT_PROMPTS = (
    "code_topological_sort",
    "general_en_explain",
    "heldout_general_ja_migration",
    "heldout_general_ja_speculative",
)
DEFAULT_GROUPS = (0, 1, 3)
NUMERICAL_SOURCE = "1a39836962fa2ef2e38da423c7a588671a4c298f"
DEFAULT_REFERENCE_ARTIFACT = (
    REPO_ROOT
    / "benchmarks/results/2026-10-18-zbook-qwen4exp-arbatch-w4-normative-task-review.json"
)
DEFAULT_RAW_GATE_JSON = Path("/tmp/qwen4exp_ar_batch_w4_20261018T044348Z.json")
DEFAULT_MODEL_ROOT = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def _parse_str_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one name")
    return parsed


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    quality = payload.get("quality", {})
    admission = payload.get("admission", {})
    if payload.get("status") != "requires_task_review":
        raise SystemExit(
            f"reference artifact status is {payload.get('status')!r}, "
            "expected 'requires_task_review'"
        )
    if int(quality.get("rows", 0)) != 450 or not bool(quality.get("passed")):
        raise SystemExit(
            "reference artifact does not record the passed 450-row numerical gate"
        )
    if not bool(admission.get("eligible")) or admission.get("admitted") is not False:
        raise SystemExit(
            "reference artifact admission is not eligible=true, admitted=false"
        )
    return payload


def _load_raw_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise SystemExit(f"raw gate reference does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_changes_since_numerical_source() -> list[str]:
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                f"{NUMERICAL_SOURCE}..HEAD",
                "--",
                "hipengine",
                "scripts",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise SystemExit(f"cannot inspect git history: {exc}") from exc
    if result.returncode != 0:
        raise SystemExit(f"git diff failed: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def _output_record(
    ids: Sequence[int],
    tokenizer: Any,
    *,
    prompt_id: str,
    route: str,
    run: int,
    budget: int,
    group_index: int | None = None,
    group_prompt_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    values = [int(token) for token in ids]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_index = None
    if eos_token_id is not None:
        for index, token in enumerate(values):
            if token == int(eos_token_id):
                eos_index = index
                break
    answer_ids = values[: eos_index + 1] if eos_index is not None else values
    text_raw = str(tokenizer.decode(values, skip_special=False))
    text_clean = str(tokenizer.decode(answer_ids, skip_special=True))
    return {
        "prompt_id": prompt_id,
        "route": route,
        "run": int(run),
        "group_index": group_index,
        "group_prompt_ids": list(group_prompt_ids) if group_prompt_ids else None,
        "budget": int(budget),
        "token_count": len(values),
        "answer_token_count": len(answer_ids),
        "eos_token_id": eos_token_id,
        "eos_index": eos_index,
        "ended_with_eos": eos_index is not None,
        "budget_exhausted": len(values) >= int(budget),
        "ids": values,
        "ids_sha256": token_ids_sha256(values),
        "answer_ids": answer_ids,
        "answer_ids_sha256": token_ids_sha256(answer_ids),
        "text_raw": text_raw,
        "text": text_clean,
    }


def _extract_code(text: str) -> dict[str, Any]:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", text, re.S)
    if blocks:
        code = "\n\n".join(block.strip() for block in blocks if block.strip())
        has_fence = True
        fence_closed = True
    else:
        stripped = text.strip()
        code_like = stripped.startswith(
            ("def ", "from ", "import ", "class ", "@", "#!")
        )
        code = stripped if code_like else ""
        has_fence = "```" in text
        fence_closed = False
    has_function = bool(re.search(r"\bdef\s+topo_sort\s*\(", code))
    has_cycle_error = bool(
        re.search(r"\bValueError\b", code) or re.search(r"\braise\b", code)
    )
    has_tests = bool(
        re.search(r"\bassert\b", code)
        or re.search(r"\bdef\s+test_\w+\s*\(", code)
        or re.search(r"\bpytest\b", code)
        or "__main__" in code
    )
    complete = bool(code) and has_function and has_cycle_error and has_tests
    return {
        "code": code,
        "has_fence": has_fence,
        "fence_closed": fence_closed,
        "has_function": has_function,
        "has_cycle_error": has_cycle_error,
        "has_tests": has_tests,
        "complete": complete,
        "code_sha256": _sha256_text(code) if code else None,
    }


def _resource_limits(timeout: int) -> Any:
    def limit() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        limit_bytes = 2 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    return limit


def _run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            start_new_session=True,
            preexec_fn=_resource_limits(timeout),
            check=False,
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "timed_out": False,
            "duration_seconds": round(time.monotonic() - started, 3),
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "command": list(command),
            "returncode": None,
            "timed_out": True,
            "duration_seconds": round(time.monotonic() - started, 3),
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
        }


def _execute_code(code: str, *, timeout: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="qwen4exp-task-review-") as tmp:
        directory = Path(tmp)
        path = directory / "generated_solution.py"
        path.write_text(code + "\n", encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": tmp,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        direct = _run_subprocess(
            [sys.executable, "-I", str(path)],
            cwd=directory,
            env=env,
            timeout=timeout,
        )
        pytest_run = _run_subprocess(
            [sys.executable, "-I", "-m", "pytest", "-q", str(path)],
            cwd=directory,
            env=env,
            timeout=timeout,
        )
        has_main = "__main__" in code
        has_pytest = bool(
            re.search(r"\bpytest\b", code)
            or re.search(r"\bdef\s+test_\w+\s*\(", code)
        )
        direct_relevant = bool(has_main or not has_pytest)
        pytest_relevant = bool(has_pytest)
        checks: list[bool] = []
        if direct_relevant:
            checks.append(direct["returncode"] == 0)
        if pytest_relevant:
            checks.append(pytest_run["returncode"] == 0)
        passed = all(checks) if checks else direct["returncode"] == 0
        return {
            "code_sha256": _sha256_text(code),
            "timeout_seconds": timeout,
            "has_main": has_main,
            "has_pytest": has_pytest,
            "direct_relevant": direct_relevant,
            "pytest_relevant": pytest_relevant,
            "direct": direct,
            "pytest": pytest_run,
            "passed": passed,
        }


def _select_groups(
    groups: Sequence[Any],
    requested: Sequence[int],
    raw_reference: dict[str, Any] | None,
) -> list[Any]:
    selected: list[Any] = []
    for index in requested:
        if index < 0 or index >= len(groups):
            raise SystemExit(f"group index {index} is outside {len(groups)} groups")
        group = groups[index]
        if group.width != 4:
            raise SystemExit(
                f"group {index} has width {group.width}; expected the original width 4"
            )
        if raw_reference is not None:
            raw_groups = raw_reference.get("batch_candidate", {}).get("groups", [])
            if index >= len(raw_groups):
                raise SystemExit(f"raw reference has no group {index}")
            expected = list(raw_groups[index].get("prompt_ids", []))
            actual = list(group.prompt_ids)
            if actual != expected:
                raise SystemExit(
                    f"group {index} composition mismatch: derived={actual} raw={expected}"
                )
        selected.append(group)
    return selected


def _reference_prompt_hashes(
    raw_reference: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if raw_reference is None:
        return {}
    return {
        str(row["id"]): dict(row)
        for row in raw_reference.get("prompts", [])
        if isinstance(row, dict) and "id" in row
    }


def _verify_prompt_tokens(
    prompt_rows: Sequence[dict[str, Any]],
    prompt_tokens: dict[str, list[int]],
    raw_reference: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    reference_hashes = _reference_prompt_hashes(raw_reference)
    records: list[dict[str, Any]] = []
    for row in prompt_rows:
        prompt_id = str(row["id"])
        tokens = prompt_tokens[prompt_id]
        record = {
            "id": prompt_id,
            "category": str(row["category"]),
            "suite": str(row["suite"]),
            "prompt": str(row["prompt"]),
            "prompt_sha256": _sha256_text(str(row["prompt"])),
            "prompt_tokens": len(tokens),
            "prompt_token_ids_sha256": token_ids_sha256(tokens),
        }
        reference = reference_hashes.get(prompt_id)
        if reference is not None:
            record["reference_prompt_tokens"] = reference.get("prompt_tokens")
            record["reference_prompt_token_ids_sha256"] = reference.get(
                "prompt_token_ids_sha256"
            )
            if int(reference.get("prompt_tokens", -1)) != len(tokens):
                raise SystemExit(
                    f"prompt token count mismatch for {prompt_id}: "
                    f"derived={len(tokens)} raw={reference.get('prompt_tokens')}"
                )
            if str(reference.get("prompt_token_ids_sha256")) != record[
                "prompt_token_ids_sha256"
            ]:
                raise SystemExit(
                    f"prompt token hash mismatch for {prompt_id}; "
                    "the focused review is not using the original prompt"
                )
        records.append(record)
    return records


def _prompt_record(records: Sequence[dict[str, Any]], prompt_id: str) -> dict[str, Any]:
    for record in records:
        if record["id"] == prompt_id:
            return record
    raise SystemExit(f"prompt {prompt_id!r} is not in the loaded fixture")


def _collect_outputs(
    *,
    route: str,
    prompt_id: str,
    tokenizer: Any,
    run: int,
    budget: int,
    ids: Sequence[int],
    group_index: int | None = None,
    group_prompt_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    record = _output_record(
        ids,
        tokenizer,
        prompt_id=prompt_id,
        route=route,
        run=run,
        budget=budget,
        group_index=group_index,
        group_prompt_ids=group_prompt_ids,
    )
    record["code_review"] = (
        _extract_code(record["text"]) if prompt_id == "code_topological_sort" else None
    )
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, default=DEFAULT_REFERENCE_ARTIFACT)
    parser.add_argument(
        "--reference-raw-gate-json",
        type=Path,
        default=DEFAULT_RAW_GATE_JSON,
    )
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--groups", type=_parse_int_csv, default=DEFAULT_GROUPS)
    parser.add_argument(
        "--strict-prompts",
        type=_parse_str_csv,
        default=DIVERGENT_PROMPTS,
    )
    parser.add_argument("--code-timeout", type=int, default=10)
    parser.add_argument("--max-sequence-length", type=int, default=2051)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--hip-arch", default="gfx1151")
    args = parser.parse_args(argv)

    if args.tokens <= 0:
        raise SystemExit("--tokens must be positive")
    if args.runs != 2:
        raise SystemExit("this focused review requires exactly two runs per route")
    if args.code_timeout <= 0:
        raise SystemExit("--code-timeout must be positive")
    if not args.model_root.is_dir():
        raise SystemExit(f"model root does not exist: {args.model_root}")

    reference = _load_reference_artifact(args.reference_artifact)
    raw_reference = _load_raw_reference(args.reference_raw_gate_json)
    runtime_changes = _runtime_changes_since_numerical_source()
    if runtime_changes:
        raise SystemExit(
            "runtime or scripts changed since the numerical source "
            f"{NUMERICAL_SOURCE}: {runtime_changes}"
        )

    os.environ.setdefault("HIPENGINE_HIP_ARCH", str(args.hip_arch))
    # Match the gate's live registration order.  The strict execution-profile
    # plan and the selected Q4_K prefill owners are not registered by merely
    # importing the runtime; the gate registers them before constructing either
    # generator.
    from hipengine.generation.qwen4_exp_profiles import (
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
        register_gguf_q4_k_selected_prefill_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    register_gguf_q4_k_selected_prefill_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()

    prompt_rows = _load_suites(list(DEFAULT_PROMPTS))
    if len(prompt_rows) != 18:
        raise SystemExit(f"expected 18 fixture prompts, got {len(prompt_rows)}")

    started = time.monotonic()
    generator_args = argparse.Namespace(
        model_root=args.model_root,
        max_sequence_length=int(args.max_sequence_length),
        prefill_chunk_size=int(args.prefill_chunk_size),
    )

    strict_outputs: dict[str, list[dict[str, Any]]] = {}
    prompt_tokens: dict[str, list[int]] = {}
    strict_generator = None
    try:
        strict_generator, strict_profile, _ = _make_generator(generator_args, "strict")
        for row in prompt_rows:
            prompt_id = str(row["id"])
            prompt_tokens[prompt_id] = [
                int(token)
                for token in build_chat_prompt(
                    strict_generator.tokenizer, str(row["prompt"])
                )
            ]
        prompt_records = _verify_prompt_tokens(prompt_rows, prompt_tokens, raw_reference)
        for prompt_id in args.strict_prompts:
            _prompt_record(prompt_records, prompt_id)
            strict_outputs[prompt_id] = []
            for run in range(1, args.runs + 1):
                result = _free_trajectory(
                    strict_generator.runner,
                    strict_generator.tokenizer,
                    prompt_tokens[prompt_id],
                    int(args.tokens),
                )
                record = _collect_outputs(
                    route="strict",
                    prompt_id=prompt_id,
                    tokenizer=strict_generator.tokenizer,
                    run=run,
                    budget=int(args.tokens),
                    ids=result["ids"],
                )
                strict_outputs[prompt_id].append(record)
                print(
                    f"strict {prompt_id} run {run}/{args.runs}: "
                    f"{record['token_count']} tokens, eos={record['eos_index']}",
                    flush=True,
                )
    finally:
        if strict_generator is not None:
            strict_generator.close()

    requests = [
        BatchCandidateRequest(
            prompt_id=str(row["id"]),
            request_id=index + 1,
            prompt_ids=tuple(prompt_tokens[str(row["id"])]),
            forced_ids=(0,),
        )
        for index, row in enumerate(prompt_rows)
    ]
    groups = plan_batch_composition(requests, 4)
    selected_groups = _select_groups(groups, args.groups, raw_reference)
    group_records = [
        {
            "index": int(group.index),
            "width": int(group.width),
            "prompt_ids": list(group.prompt_ids),
            "request_ids": [int(request.request_id) for request in group.requests],
            "prompt_token_ids_sha256": {
                request.prompt_id: token_ids_sha256(request.prompt_ids)
                for request in group.requests
            },
        }
        for group in selected_groups
    ]

    batch_outputs: dict[str, list[dict[str, Any]]] = {
        prompt_id: [] for prompt_id in DIVERGENT_PROMPTS
    }
    batch_generator = None
    adapter = None
    try:
        batch_generator, batch_profile, _ = _make_generator(
            generator_args, "production"
        )
        adapter = Qwen4ExpARBatchCandidate(
            batch_generator,
            width=4,
            label="task_review_w4",
        )
        for run in range(1, args.runs + 1):
            for group in selected_groups:
                free = adapter.run_free_group(group, steps=int(args.tokens))
                for request in group.requests:
                    prompt_id = request.prompt_id
                    record = _collect_outputs(
                        route="batch",
                        prompt_id=prompt_id,
                        tokenizer=batch_generator.tokenizer,
                        run=run,
                        budget=int(args.tokens),
                        ids=free[prompt_id]["ids"],
                        group_index=int(group.index),
                        group_prompt_ids=group.prompt_ids,
                    )
                    if prompt_id in batch_outputs:
                        batch_outputs[prompt_id].append(record)
                    print(
                        f"batch group {group.index} run {run}/{args.runs} "
                        f"{prompt_id}: {record['token_count']} tokens, "
                        f"eos={record['eos_index']}",
                        flush=True,
                    )
    finally:
        if adapter is not None:
            adapter.close()
        if batch_generator is not None:
            batch_generator.close()

    code_execution: dict[str, dict[str, Any]] = {}
    for route_outputs in (strict_outputs, batch_outputs):
        for prompt_id, runs in route_outputs.items():
            if prompt_id != "code_topological_sort":
                continue
            for record in runs:
                code_review = record.get("code_review") or {}
                if not code_review.get("complete"):
                    continue
                code = str(code_review["code"])
                code_sha = _sha256_text(code)
                if code_sha in code_execution:
                    record["code_execution_sha256"] = code_sha
                    record["code_execution_reused"] = True
                    continue
                print(
                    f"code {record['route']} run {record['run']}: "
                    f"executing {code_sha[:12]}",
                    flush=True,
                )
                execution = _execute_code(code, timeout=int(args.code_timeout))
                code_execution[code_sha] = execution
                record["code_execution_sha256"] = code_sha
                record["code_execution_reused"] = False

    determinism: dict[str, dict[str, Any]] = {}
    for prompt_id in DIVERGENT_PROMPTS:
        entry: dict[str, Any] = {}
        for route, outputs in (
            ("strict", strict_outputs.get(prompt_id, [])),
            ("batch", batch_outputs.get(prompt_id, [])),
        ):
            if len(outputs) != 2:
                entry[route] = {"runs": len(outputs), "repeat_exact": None}
                continue
            entry[route] = {
                "runs": 2,
                "repeat_exact": outputs[0]["ids_sha256"] == outputs[1]["ids_sha256"],
                "answer_repeat_exact": outputs[0]["answer_ids_sha256"]
                == outputs[1]["answer_ids_sha256"],
                "run_sha256": [output["ids_sha256"] for output in outputs],
                "answer_sha256": [
                    output["answer_ids_sha256"] for output in outputs
                ],
            }
        strict_runs = strict_outputs.get(prompt_id, [])
        batch_runs = batch_outputs.get(prompt_id, [])
        if strict_runs and batch_runs:
            entry["strict_vs_batch_answer_exact"] = (
                strict_runs[0]["answer_ids_sha256"]
                == batch_runs[0]["answer_ids_sha256"]
            )
            entry["strict_vs_batch_ids_exact"] = (
                strict_runs[0]["ids_sha256"] == batch_runs[0]["ids_sha256"]
            )
        determinism[prompt_id] = entry

    elapsed = round(time.monotonic() - started, 3)
    payload = {
        "schema": 1,
        "kind": "qwen4exp_arbatch_focused_task_review_raw",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "diagnostic_only",
        "scope": {
            "purpose": "focused longer free-output semantic review; not a gate rerun",
            "prompt_ids": list(args.strict_prompts),
            "group_indices": list(args.groups),
            "tokens": int(args.tokens),
            "runs_per_route": int(args.runs),
            "numerical_gate_rerun": False,
            "runtime_or_default_changed": False,
        },
        "source": {
            "head": _git_metadata(REPO_ROOT),
            "numerical_source": NUMERICAL_SOURCE,
            "runtime_changes_since_numerical_source": runtime_changes,
        },
        "host": _host_metadata(),
        "command": [sys.executable, *sys.argv],
        "model": {
            "root": str(args.model_root),
            "quant": "Q4_K_XL",
            "backend": "hip_gfx1151",
            "profile": "production",
        },
        "settings": {
            "tokens": int(args.tokens),
            "runs": int(args.runs),
            "groups": list(args.groups),
            "strict_prompts": list(args.strict_prompts),
            "code_timeout_seconds": int(args.code_timeout),
            "max_sequence_length": int(args.max_sequence_length),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "hip_arch": str(args.hip_arch),
            "batch_width": 4,
            "batch_composition": "fixture_order_consecutive_groups",
            "forced_ids_placeholder": [0],
            "forced_ids_note": (
                "free generation does not consume forced_ids; the placeholder "
                "only satisfies the request dataclass and preserves group planning"
            ),
        },
        "reference": {
            "compact_artifact": {
                "path": str(args.reference_artifact.resolve()),
                "sha256": _sha256_file(args.reference_artifact),
                "status": reference.get("status"),
                "quality_rows": reference.get("quality", {}).get("rows"),
                "quality_passed": reference.get("quality", {}).get("passed"),
                "admission_eligible": reference.get("admission", {}).get("eligible"),
                "admission_admitted": reference.get("admission", {}).get("admitted"),
                "admission_reasons": reference.get("admission", {}).get("reasons"),
                "task_review": reference.get("task_review"),
            },
            "raw_gate_json": (
                {
                    "path": str(args.reference_raw_gate_json.resolve()),
                    "sha256": _sha256_file(args.reference_raw_gate_json),
                    "status": raw_reference.get("status"),
                    "source": raw_reference.get("source"),
                }
                if raw_reference is not None
                else None
            ),
        },
        "prompts": prompt_records,
        "groups": group_records,
        "strict": strict_outputs,
        "batch": batch_outputs,
        "determinism": determinism,
        "code_execution": code_execution,
        "elapsed_seconds": elapsed,
        "profiles": {
            "strict_manifest": _json_value(strict_profile.manifest),
            "strict_manifest_sha256": strict_profile.manifest_sha256,
            "batch_manifest": _json_value(batch_profile.manifest),
            "batch_manifest_sha256": batch_profile.manifest_sha256,
        },
        "notes": [
            "This file contains diagnostic free outputs only; it does not override "
            "the calibrated numerical/task gate.",
            "The original width-4 groups 0, 1, and 3 are rerun with all companions "
            "in fixture order; divergent prompts are not regrouped.",
            "A missing EOS means the budget was exhausted; semantic completeness is "
            "reviewed separately and is not automatically a failure or pass.",
        ],
    }
    _write_json(args.output, payload)
    print(f"wrote {args.output} in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
