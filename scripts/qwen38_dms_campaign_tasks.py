#!/usr/bin/env python3
"""Frozen G3 long-context task generator/runner/comparer for the Qwen3.8 DMS campaign.

Phase A campaign tooling (docs/campaigns/DMS-SELECTOR-IMPROVEMENT.md Sections
4 and 11).  Three subcommands:

- ``build`` — consume one sealed data manifest/split plus the exact model
  tokenizer and emit an immutable 24-case G3 task manifest for one target
  context (32K or 128K) and suite identity.
- ``run`` — execute exactly one arm (``dense``, ``no_evict``, ``sidecar``)
  over the frozen task manifest under a dense-compatible greedy schedule for
  a fixed max answer length, recording generations, parser verdicts, the
  selected DMS route/capacity/count digest, memory teardown, and identity
  hashes with no silent fallback.  ``--diagnostic free-running`` instead runs
  the non-binding 256-token free-running diagnostic on ordinary sealed
  prompts.
- ``compare`` — apply frozen G3 to dense, frozen-baseline, and candidate
  result files from the same task manifest.

This tool is offline evaluation only; it is not a production routing surface
and never conditions runtime routing on prompt identity.  Answer keys are
evaluator-only and are never supplied as a separate selector feature or
control signal; each case's answer occurs only inside its dependency
statement, where the task semantics require it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from hipengine.benchmark.dms_campaign import CATEGORIES, dms_digest, load_manifest
from hipengine.benchmark.dms_tasks import (
    DEFAULT_FREE_RUNNING_TOKENS,
    DEFAULT_MAX_ANSWER_TOKENS,
    DENSE_ARM,
    DMS_ARMS,
    FREE_RUNNING_RUN_KIND,
    RUN_ARMS,
    SUPPORTED_TARGET_TOKENS,
    SUITES,
    TASK_MANIFEST_KIND,
    TASK_RUN_KIND,
    compare_g3,
    free_running_metrics,
)

SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    return {
        "commit": commit,
        "scoped_dirty_diff": sorted(
            line[3:] for line in dirty.splitlines() if line[3:].strip()
        ),
    }


def _evaluator_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    library = root / "hipengine" / "benchmark" / "dms_tasks.py"
    script = Path(__file__)
    return {
        "library_path": str(library),
        "library_sha256": _sha256(library),
        "script_path": str(script),
        "script_sha256": _sha256(script),
    }


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="emit an immutable 24-case G3 task manifest")
    build.add_argument("--model", type=Path, required=True,
                       help="exact model GGUF; its tokenizer defines all case token lengths")
    build.add_argument("--data-manifest", type=Path, required=True)
    build.add_argument("--split", required=True,
                       help="sealed manifest split that exclusively supplies filler sources")
    build.add_argument("--suite", choices=SUITES, required=True)
    build.add_argument("--target-tokens", type=int, required=True,
                       help=f"target context; one of {list(SUPPORTED_TARGET_TOKENS)}")
    build.add_argument("--seed", type=int, required=True)
    build.add_argument("--expected-sequences-per-category", type=int, default=None,
                       help="optional declared sequence count each category must supply exactly")
    build.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="execute exactly one arm over the frozen tasks")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--arm", choices=RUN_ARMS, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--metadata", type=Path, default=None,
                     help="DMS sidecar metadata; required for no_evict/sidecar, forbidden for dense")
    run.add_argument("--backend", default="hip_gfx1151")
    run.add_argument("--codec", choices=("bf16", "int8_evaluation"), default="bf16",
                     help="Offline candidate codec; INT8 evaluation does not qualify serving.")
    run.add_argument("--max-answer-tokens", type=int, default=DEFAULT_MAX_ANSWER_TOKENS)
    run.add_argument("--diagnostic", choices=("tasks", "free-running"), default="tasks")
    run.add_argument("--task-manifest", type=Path, default=None,
                     help="frozen G3 task manifest (required for --diagnostic tasks)")
    run.add_argument("--fail-on-fail", action="store_true")
    # Free-running diagnostic options.
    run.add_argument("--data-manifest", type=Path, default=None)
    run.add_argument("--split", default=None)
    run.add_argument("--categories", default=",".join(CATEGORIES))
    run.add_argument("--expected-sequences-per-category", type=int, default=None)
    run.add_argument("--prompt-tokens", type=int, default=None)
    run.add_argument("--free-running-tokens", type=int, default=DEFAULT_FREE_RUNNING_TOKENS)
    run.add_argument("--dense-reference", type=Path, default=None,
                     help="dense free-running result file for divergence metrics")

    compare = subparsers.add_parser("compare", help="apply frozen G3 to three task-run results")
    compare.add_argument("--dense", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--fail-on-fail", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# build


def _load_tokenizer(model_path: Path):
    # Runtime import: the tokenizer stack stays out of module import time so
    # tests and help text never load the model stack.
    from hipengine.loading.gguf import GGUFReader
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    return Qwen35GGUFTokenizer.from_gguf_info(GGUFReader(str(model_path)).info)


def cmd_build(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.benchmark.dms_tasks import build_task_manifest

    target_tokens = int(args.target_tokens)
    if target_tokens not in SUPPORTED_TARGET_TOKENS:
        raise ValueError(
            f"target-tokens {target_tokens} not in the frozen set {list(SUPPORTED_TARGET_TOKENS)}"
        )
    tokenizer = _load_tokenizer(args.model)
    sequences = load_manifest(
        args.data_manifest,
        split=str(args.split),
        categories=CATEGORIES,
        expected_sequences_per_category=(
            int(args.expected_sequences_per_category)
            if args.expected_sequences_per_category
            else None
        ),
    )
    pool = [
        {
            "sequence_id": record["sequence_id"],
            "source_id": record["source_id"],
            "category": record["category"],
            "normalized_text_sha256": record["normalized_text_sha256"],
            "token_ids": record["token_ids"],
        }
        for record in sequences
    ]
    manifest = build_task_manifest(
        pool=pool,
        suite=str(args.suite),
        target_tokens=target_tokens,
        seed=int(args.seed),
        tokenize=tokenizer.encode,
        data_manifest={
            "path": str(Path(args.data_manifest).resolve()),
            "sha256": _sha256(args.data_manifest),
            "split": str(args.split),
        },
        model={
            "path": str(Path(args.model).resolve()),
            "sha256": _sha256(args.model),
            "tokenizer": "Qwen35GGUFTokenizer from GGUF metadata (exact model tokenizer)",
        },
    )
    _write_output(args.output, manifest)
    return manifest


# ---------------------------------------------------------------------------
# run


def _greedy_generate(session: Any, prompt: list[int], *, max_new_tokens: int, eos_token_id: int | None):
    seed = session.prefill(prompt, use_bulk=True, bulk_attention_mode="bulk")
    generated: list[int] = []
    token = int(seed.token_id)
    while len(generated) < int(max_new_tokens):
        generated.append(token)
        if (
            len(generated) >= int(max_new_tokens)
            or (eos_token_id is not None and token == int(eos_token_id))
        ):
            break
        result = session.step(token)
        token = int(result.token_id)
    return generated


def _memory_stats() -> dict[str, Any]:
    from hipengine.core.memory import memory_stats

    return memory_stats()


def _teardown_ok(baseline: dict[str, Any], after: dict[str, Any]) -> bool:
    return (
        after["current_allocated_bytes"] == baseline["current_allocated_bytes"]
        and after["active_allocations"] == baseline["active_allocations"]
    )


def _dms_route(snapshot: dict[str, Any], arm: str) -> dict[str, Any]:
    backend = snapshot.get("backend", {})
    if not backend.get("device_payloads"):
        raise AssertionError(f"DMS arm ({arm}) ran without device payloads")
    decision_source = backend.get("decision_source")
    if not decision_source:
        raise AssertionError(f"DMS arm ({arm}) recorded no decision source; silent fallback")
    return {
        "decision_mode": arm,
        "decision_source": decision_source,
        "artifact_fingerprint": backend.get("artifact_fingerprint"),
        "dms_digest": dms_digest(snapshot),
        "no_silent_fallback": True,
    }


def _session_kwargs(args: argparse.Namespace, max_positions: int, *, max_new_tokens: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_sequence_length": max_positions,
        "use_wmma_prefill": True,
        "use_gemv_decode": True,
    }
    if args.arm != DENSE_ARM:
        from hipengine.kvcache.dms import (
            create_dms_bf16_backend,
            create_dms_int8_evaluation_backend,
        )

        kwargs.update(
            {
                "dms_metadata_path": args.metadata,
                "dms_decision_mode": args.arm,
                "dms_backend_factory": {
                    "bf16": create_dms_bf16_backend,
                    "int8_evaluation": create_dms_int8_evaluation_backend,
                }[str(args.codec)],
                "dms_max_new_tokens": int(max_new_tokens),
            }
        )
    return kwargs


def _validate_run_arm(args: argparse.Namespace) -> None:
    if args.arm == DENSE_ARM:
        if args.metadata is not None:
            raise ValueError("the dense arm must not carry sidecar metadata")
    elif args.arm in DMS_ARMS:
        if args.metadata is None:
            raise ValueError(f"the {args.arm} arm requires sidecar metadata")
    else:
        raise ValueError(f"unknown arm {args.arm!r}")
    if int(args.max_answer_tokens) <= 0:
        raise ValueError("max-answer-tokens must be positive")


def _validate_task_manifest(
    manifest: dict[str, Any],
    *,
    model_sha256: str,
) -> None:
    from hipengine.benchmark.dms_tasks import validate_case

    if manifest.get("kind") != TASK_MANIFEST_KIND:
        raise ValueError("input is not a frozen G3 task manifest")
    if not bool(manifest.get("immutable")):
        raise ValueError("G3 task manifest is not marked immutable")
    if int(manifest.get("case_count", -1)) != 24 or len(manifest.get("cases") or []) != 24:
        raise ValueError("G3 task manifest must contain exactly 24 cases")
    if int(manifest.get("target_tokens", -1)) not in SUPPORTED_TARGET_TOKENS:
        raise ValueError("G3 task manifest target length is not frozen")
    expected_model = str((manifest.get("model") or {}).get("sha256", ""))
    if not expected_model or expected_model != str(model_sha256):
        raise ValueError("G3 task manifest model hash does not match --model")
    case_ids: set[str] = set()
    for case in manifest["cases"]:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in case_ids:
            raise ValueError("G3 task manifest case IDs must be non-empty and unique")
        case_ids.add(case_id)
        problems = validate_case(case)
        if problems:
            raise ValueError(f"invalid G3 case {case_id}: {'; '.join(problems)}")


def _validate_dense_reference(
    reference: dict[str, Any],
    *,
    model_sha256: str,
    data_manifest_sha256: str,
    free_running_tokens: int,
) -> None:
    if reference.get("kind") != FREE_RUNNING_RUN_KIND:
        raise ValueError("dense reference is not a free-running run result")
    if str(reference.get("arm")) != DENSE_ARM:
        raise ValueError("dense reference arm must be dense")
    if str((reference.get("model") or {}).get("sha256", "")) != str(model_sha256):
        raise ValueError("dense reference model hash does not match --model")
    if str((reference.get("data_manifest") or {}).get("sha256", "")) != str(data_manifest_sha256):
        raise ValueError("dense reference data-manifest hash does not match")
    if int((reference.get("schedule") or {}).get("free_running_tokens", -1)) != int(free_running_tokens):
        raise ValueError("dense reference free-running schedule does not match")


def cmd_run_tasks(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.benchmark.dms_tasks import parse_answer
    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFFullStackRunner,
        Qwen35GGUFResidentSession,
    )

    if args.task_manifest is None:
        raise ValueError("--task-manifest is required for --diagnostic tasks")
    task_manifest_path = Path(args.task_manifest)
    manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    model_sha256 = _sha256(args.model)
    _validate_task_manifest(manifest, model_sha256=model_sha256)
    max_answer_tokens = int(args.max_answer_tokens)
    tokenizer = _load_tokenizer(args.model)
    eos_token_id = tokenizer.eos_token_id

    started = time.perf_counter()
    memory_baseline = _memory_stats()
    runner = Qwen35GGUFFullStackRunner(args.model, backend=str(args.backend))
    loaded_at = time.perf_counter()
    case_results: list[dict[str, Any]] = []
    try:
        for case in manifest["cases"]:
            prompt = [int(token) for token in case["token_ids"]]
            case_started = time.perf_counter()
            with Qwen35GGUFResidentSession(
                args.model,
                backend=str(args.backend),
                shared_runner=runner,
                **_session_kwargs(
                    args,
                    max_positions=len(prompt) + max_answer_tokens,
                    max_new_tokens=max_answer_tokens,
                ),
            ) as session:
                generated = _greedy_generate(
                    session,
                    prompt,
                    max_new_tokens=max_answer_tokens,
                    eos_token_id=eos_token_id,
                )
                route: dict[str, Any] | None = None
                if args.arm != DENSE_ARM:
                    if session._dms_dense_prefill_pool is not None:
                        raise AssertionError(f"DMS arm ({args.arm}) retained dense prefill pool")
                    route = _dms_route(session._dms_backend.observability_snapshot(), str(args.arm))
            parser = parse_answer(tokenizer.decode(generated, skip_special=True), case["answer_text"])
            case_results.append(
                {
                    "case_id": case["case_id"],
                    "family": case["family"],
                    "category": case["category"],
                    "placement": case["placement"],
                    "target_tokens": case["target_tokens"],
                    "dependency_position": case["dependency_position"],
                    "query_position": case["query_position"],
                    "answer_token_ids_sha256": case["answer_token_ids_sha256"],
                    "prompt_token_ids_sha256": case["token_ids_sha256"],
                    "generated_token_ids": generated,
                    "generated_text": tokenizer.decode(generated, skip_special=True),
                    "parser": parser,
                    "correct": bool(parser["correct"]),
                    "route": route,
                    "timing_seconds": time.perf_counter() - case_started,
                }
            )
    finally:
        runner.close()
    memory_after_close = _memory_stats()
    teardown_ok = _teardown_ok(memory_baseline, memory_after_close)
    if not teardown_ok:
        raise AssertionError(
            "tracked device allocations did not return to baseline after teardown"
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": TASK_RUN_KIND,
        "status": "completed",
        "performance_claim": False,
        "host": socket.gethostname(),
        "backend": str(args.backend),
        "codec": str(args.codec),
        "arm": str(args.arm),
        "metadata": (
            {"path": str(Path(args.metadata).resolve()), "sha256": _sha256(args.metadata)}
            if args.metadata is not None
            else None
        ),
        "model": {"path": str(Path(args.model).resolve()), "sha256": model_sha256},
        "task_manifest": {
            "path": str(task_manifest_path.resolve()),
            "sha256": _sha256(task_manifest_path),
            "suite": manifest.get("suite"),
            "target_tokens": manifest.get("target_tokens"),
            "case_count": manifest.get("case_count"),
        },
        "evaluator": _evaluator_hashes(),
        "schedule": {
            "sampling": "greedy (dense-compatible argmax schedule; identical across arms)",
            "max_answer_tokens": max_answer_tokens,
        },
        "cases": case_results,
        "resource_checks": {
            "dense_prefill_pool_released": (
                None if args.arm == DENSE_ARM else True
            ),
            "teardown_to_baseline": teardown_ok,
            "memory_baseline": memory_baseline,
            "memory_after_close": memory_after_close,
        },
        "timing": {
            "load_seconds": loaded_at - started,
            "total_seconds": time.perf_counter() - started,
        },
        "provenance": _git(),
    }
    _write_output(args.output, result)
    return result


def cmd_run_free_running(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFFullStackRunner,
        Qwen35GGUFResidentSession,
    )

    if args.data_manifest is None or not args.split:
        raise ValueError("free-running diagnostic requires --data-manifest and --split")
    free_running_tokens = int(args.free_running_tokens)
    if free_running_tokens <= 0:
        raise ValueError("free-running-tokens must be positive")
    categories = _parse_csv(args.categories)
    sequences = load_manifest(
        args.data_manifest,
        split=str(args.split),
        categories=categories,
        expected_sequences_per_category=(
            int(args.expected_sequences_per_category)
            if args.expected_sequences_per_category
            else None
        ),
        prompt_tokens=int(args.prompt_tokens) if args.prompt_tokens else None,
    )
    sequences.sort(key=lambda record: str(record["sequence_id"]))
    tokenizer = _load_tokenizer(args.model)
    eos_token_id = tokenizer.eos_token_id

    model_sha256 = _sha256(args.model)
    data_manifest_sha256 = _sha256(args.data_manifest)
    dense_reference = None
    if args.dense_reference is not None:
        dense_reference = json.loads(Path(args.dense_reference).read_text(encoding="utf-8"))
        _validate_dense_reference(
            dense_reference,
            model_sha256=model_sha256,
            data_manifest_sha256=data_manifest_sha256,
            free_running_tokens=free_running_tokens,
        )

    started = time.perf_counter()
    memory_baseline = _memory_stats()
    runner = Qwen35GGUFFullStackRunner(args.model, backend=str(args.backend))
    loaded_at = time.perf_counter()
    prompt_results: list[dict[str, Any]] = []
    try:
        for record in sequences:
            prompt = [int(token) for token in record["token_ids"]]
            prompt_sha = hashlib.sha256(
                json.dumps(prompt).encode("utf-8")
            ).hexdigest()
            with Qwen35GGUFResidentSession(
                args.model,
                backend=str(args.backend),
                shared_runner=runner,
                **_session_kwargs(
                    args,
                    max_positions=len(prompt) + free_running_tokens,
                    max_new_tokens=free_running_tokens,
                ),
            ) as session:
                generated = _greedy_generate(
                    session,
                    prompt,
                    max_new_tokens=free_running_tokens,
                    eos_token_id=eos_token_id,
                )
                route = None
                if args.arm != DENSE_ARM:
                    if session._dms_dense_prefill_pool is not None:
                        raise AssertionError(f"DMS arm ({args.arm}) retained dense prefill pool")
                    route = _dms_route(session._dms_backend.observability_snapshot(), str(args.arm))
            metrics: dict[str, Any] | None = None
            if dense_reference is not None:
                match = [
                    entry
                    for entry in dense_reference.get("prompts", [])
                    if entry.get("prompt_token_ids_sha256") == prompt_sha
                ]
                if len(match) != 1:
                    raise ValueError(
                        f"dense reference must have exactly one prompt matching sequence "
                        f"{record['sequence_id']}; got {len(match)}"
                    )
                metrics = free_running_metrics(
                    match[0]["generated_token_ids"],
                    generated,
                    fixed_length=free_running_tokens,
                )
            prompt_results.append(
                {
                    "sequence_id": record["sequence_id"],
                    "category": record["category"],
                    "split": record["split"],
                    "prompt_tokens": len(prompt),
                    "prompt_token_ids_sha256": prompt_sha,
                    "generated_token_ids": generated,
                    "generated_text": tokenizer.decode(generated, skip_special=True),
                    "route": route,
                    "free_running_metrics": metrics,
                }
            )
    finally:
        runner.close()
    memory_after_close = _memory_stats()
    teardown_ok = _teardown_ok(memory_baseline, memory_after_close)
    if not teardown_ok:
        raise AssertionError(
            "tracked device allocations did not return to baseline after teardown"
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": FREE_RUNNING_RUN_KIND,
        "status": "completed",
        "performance_claim": False,
        "host": socket.gethostname(),
        "backend": str(args.backend),
        "codec": str(args.codec),
        "arm": str(args.arm),
        "metadata": (
            {"path": str(Path(args.metadata).resolve()), "sha256": _sha256(args.metadata)}
            if args.metadata is not None
            else None
        ),
        "model": {"path": str(Path(args.model).resolve()), "sha256": model_sha256},
        "data_manifest": {
            "path": str(Path(args.data_manifest).resolve()),
            "sha256": data_manifest_sha256,
            "split": str(args.split),
        },
        "evaluator": _evaluator_hashes(),
        "schedule": {
            "sampling": "greedy (dense-compatible argmax schedule; identical across arms)",
            "free_running_tokens": free_running_tokens,
        },
        "prompts": prompt_results,
        "non_binding": True,
        "note": (
            "free-running first divergence / comparable prefix / fixed-length "
            "match are diagnostics; they never substitute for G3 task scoring"
        ),
        "resource_checks": {
            "dense_prefill_pool_released": (
                None if args.arm == DENSE_ARM else True
            ),
            "teardown_to_baseline": teardown_ok,
            "memory_baseline": memory_baseline,
            "memory_after_close": memory_after_close,
        },
        "timing": {
            "load_seconds": loaded_at - started,
            "total_seconds": time.perf_counter() - started,
        },
        "provenance": _git(),
    }
    _write_output(args.output, result)
    return result


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_run_arm(args)
    if str(args.diagnostic) == "free-running":
        return cmd_run_free_running(args)
    return cmd_run_tasks(args)


# ---------------------------------------------------------------------------
# compare


def cmd_compare(args: argparse.Namespace) -> dict[str, Any]:
    dense = json.loads(Path(args.dense).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    verdict = compare_g3(dense, baseline, candidate)
    verdict["sources"] = {
        "dense": {"path": str(Path(args.dense).resolve()), "sha256": _sha256(args.dense)},
        "baseline": {"path": str(Path(args.baseline).resolve()), "sha256": _sha256(args.baseline)},
        "candidate": {"path": str(Path(args.candidate).resolve()), "sha256": _sha256(args.candidate)},
    }
    verdict["evaluator"] = _evaluator_hashes()
    verdict["provenance"] = _git()
    _write_output(args.output, verdict)
    return verdict


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "build":
        manifest = cmd_build(args)
        summary = {
            "command": "build",
            "output": str(args.output),
            "suite": manifest["suite"],
            "target_tokens": manifest["target_tokens"],
            "case_count": manifest["case_count"],
            "correlated_groups": manifest["filler_correlations"]["correlated_groups"],
        }
    elif args.command == "run":
        result = cmd_run(args)
        if result["kind"] == TASK_RUN_KIND:
            correct = sum(1 for case in result["cases"] if case["correct"])
            summary = {
                "command": "run",
                "arm": result["arm"],
                "output": str(args.output),
                "case_count": len(result["cases"]),
                "correct": correct,
            }
        else:
            summary = {
                "command": "run",
                "arm": result["arm"],
                "diagnostic": "free-running",
                "output": str(args.output),
                "prompt_count": len(result["prompts"]),
                "non_binding": True,
            }
    else:
        verdict = cmd_compare(args)
        summary = {
            "command": "compare",
            "gate": verdict["gate"],
            "status": verdict["status"],
            "passed": verdict["passed"],
            "errors": verdict.get("errors", []),
            "failures": verdict.get("failures", []),
            "dense_failures": [
                failure["case_id"] for failure in verdict.get("dense_failures", [])
            ],
        }
        args = args  # fail-on-fail flag lives on this namespace
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.command == "compare" and args.fail_on_fail and not verdict["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
