#!/usr/bin/env python3
"""Compare packed GGUF AR to independent c1 on frozen category prompts.

This is a byte-exact B4 correctness harness, never a performance benchmark. It
runs the canonical ten-prompt mtp-bench category suite and the frozen category
heldouts in packed groups of at most c4, then compares generated tokens, every
post-layer BF16 row, all Conv/GDN state, and live BF16 K/V with independent c1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import (
    DEFAULT_FULL_PROMPT_IDS,
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    load_prompt_rows,
    prompt_sha256,
)
from scripts.gguf_packed_ar_state_oracle import (
    _CAPTURE_PREFILL_GDN_ENV,
    _GDN_PREFILL_MODE_ENV,
    _capture_state,
    _compare_layer_hidden_sessions,
    _compare_state_rows,
    _prefill_c1_with_layer_hidden,
    _temporary_env,
)

DEFAULT_HELDOUTS = REPO_ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"
_EXPECTED_CATEGORIES = frozenset({"code", "general_en", "general_ja", "mixed_ja_en"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _group_prompt_rows(
    rows: Sequence[dict[str, Any]],
    *,
    max_group_size: int = 4,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    size = int(max_group_size)
    if size <= 0:
        raise ValueError("max_group_size must be positive")
    values = tuple(rows)
    if not values:
        raise ValueError("category oracle requires at least one prompt")
    return tuple(
        tuple(values[start : start + size])
        for start in range(0, len(values), size)
    )


def _validate_prompt_contract(
    primary: Sequence[dict[str, Any]],
    heldouts: Sequence[dict[str, Any]],
) -> None:
    primary_ids = tuple(str(row["id"]) for row in primary)
    if primary_ids != tuple(DEFAULT_FULL_PROMPT_IDS):
        raise ValueError(
            "primary prompt IDs/order must match the canonical ten-prompt suite"
        )
    all_rows = (*primary, *heldouts)
    all_ids = [str(row["id"]) for row in all_rows]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("primary and heldout prompt IDs must be unique")
    primary_categories = {str(row["category"]) for row in primary}
    heldout_categories = {str(row["category"]) for row in heldouts}
    if primary_categories != _EXPECTED_CATEGORIES:
        raise ValueError("primary prompts must cover every required category")
    if heldout_categories != _EXPECTED_CATEGORIES:
        raise ValueError("heldouts must cover every required category")


def _prompt_manifest(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(row["id"]),
            "category": str(row["category"]),
            "prompt_chars": len(str(row["prompt"])),
            "prompt_sha256": prompt_sha256(str(row["prompt"])),
        }
        for row in rows
    ]


def _repeat_determinism(
    prompt_repeats: dict[str, list[dict[str, Any]]],
) -> tuple[bool, list[str]]:
    mismatches = [
        prompt_id
        for prompt_id, repeats in prompt_repeats.items()
        if len({str(row["trajectory_sha256"]) for row in repeats}) != 1
    ]
    return not mismatches, sorted(mismatches)


def _session_build_policy(args: argparse.Namespace) -> dict[str, Any]:
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.expanduser().read_text(
            encoding="utf-8"
        ).strip()
        if not compiler_version:
            raise ValueError(
                f"compiler version file is empty: {args.compiler_version_file}"
            )
    return {
        "compiler_version": compiler_version,
        "require_cached_build": bool(args.require_cached_build),
    }


def _run_group(
    *,
    owner: Any,
    packed_sessions: tuple[Any, ...],
    reference_sessions: tuple[Any, ...],
    rows: tuple[dict[str, Any], ...],
    prompt_tokens: tuple[tuple[int, ...], ...],
    layer_ids: tuple[int, ...],
    decode_steps: int,
    repeat_index: int,
    suite: str,
    group_index: int,
) -> dict[str, Any]:
    for session in (*packed_sessions, *reference_sessions):
        session.reset()

    with _temporary_env({_GDN_PREFILL_MODE_ENV: "exact"}):
        with _temporary_env({_CAPTURE_PREFILL_GDN_ENV: "1"}):
            packed_results = owner.prefill_batch_native(
                prompt_tokens,
                sessions=packed_sessions,
                return_logits=False,
                capture_layer_output_hidden=layer_ids,
            )
        reference_tokens = [
            _prefill_c1_with_layer_hidden(session, prompt, layer_ids)
            for session, prompt in zip(
                reference_sessions,
                prompt_tokens,
                strict=True,
            )
        ]
    packed_tokens = [int(result.token_id) for result in packed_results]
    trajectories = [[token] for token in packed_tokens]
    reference_trajectories = [[token] for token in reference_tokens]

    hidden_comparisons, hidden_mismatches = _compare_layer_hidden_sessions(
        packed_sessions,
        reference_sessions,
        row_indices=tuple(range(len(rows))),
        layer_ids=layer_ids,
        phase="prefill_hidden",
        step=0,
    )
    initial_packed = [_capture_state(session) for session in packed_sessions]
    initial_reference = [_capture_state(session) for session in reference_sessions]
    initial_mismatches = _compare_state_rows(initial_packed, initial_reference)

    for step_index in range(1, int(decode_steps) + 1):
        with _temporary_env({_CAPTURE_PREFILL_GDN_ENV: "1"}):
            packed_tokens = [
                int(result.token_id)
                for result in owner.step_batch_native(
                    packed_tokens,
                    sessions=packed_sessions,
                    positions=[int(session.position) for session in packed_sessions],
                    return_logits=False,
                    scatter_state=False,
                    capture_layer_output_hidden=layer_ids,
                )
            ]
        reference_tokens = [
            int(
                session.step(
                    token,
                    return_logits=False,
                    capture_layer_output_hidden=layer_ids,
                ).token_id
            )
            for session, token in zip(
                reference_sessions,
                reference_tokens,
                strict=True,
            )
        ]
        for trajectory, token in zip(trajectories, packed_tokens, strict=True):
            trajectory.append(int(token))
        for trajectory, token in zip(
            reference_trajectories,
            reference_tokens,
            strict=True,
        ):
            trajectory.append(int(token))
        compared, mismatches = _compare_layer_hidden_sessions(
            packed_sessions,
            reference_sessions,
            row_indices=tuple(range(len(rows))),
            layer_ids=layer_ids,
            phase="decode_hidden",
            step=step_index,
        )
        hidden_comparisons += compared
        hidden_mismatches.extend(mismatches)

    dirty_before_flush = bool(owner._packed_decode_state_dirty)
    flush_executed = bool(owner.flush_packed_decode_state())
    final_packed = [_capture_state(session) for session in packed_sessions]
    final_reference = [_capture_state(session) for session in reference_sessions]
    final_mismatches = _compare_state_rows(final_packed, final_reference)
    token_exact = trajectories == reference_trajectories
    passed = bool(
        token_exact
        and not initial_mismatches
        and not hidden_mismatches
        and dirty_before_flush
        and flush_executed
        and not final_mismatches
    )

    prompt_results = []
    for prompt_index, (row, tokens, trajectory) in enumerate(
        zip(rows, prompt_tokens, trajectories, strict=True)
    ):
        prompt_hidden_mismatches = [
            mismatch
            for mismatch in hidden_mismatches
            if int(mismatch["row"]) == prompt_index
        ]
        prompt_initial_mismatches = [
            mismatch
            for mismatch in initial_mismatches
            if int(mismatch["row"]) == prompt_index
        ]
        prompt_final_mismatches = [
            mismatch
            for mismatch in final_mismatches
            if int(mismatch["row"]) == prompt_index
        ]
        prompt_results.append(
            {
                "id": str(row["id"]),
                "category": str(row["category"]),
                "prompt_tokens": len(tokens),
                "trajectory_steps": len(trajectory),
                "trajectory_sha256": _sha256_json(trajectory),
                "initial_token": int(trajectory[0]),
                "final_token": int(trajectory[-1]),
                "token_exact": trajectory == reference_trajectories[prompt_index],
                "hidden_comparisons": len(layer_ids) * (int(decode_steps) + 1),
                "hidden_mismatches": len(prompt_hidden_mismatches),
                "initial_state_mismatches": len(prompt_initial_mismatches),
                "final_state_mismatches": len(prompt_final_mismatches),
                "passed": bool(
                    trajectory == reference_trajectories[prompt_index]
                    and not prompt_hidden_mismatches
                    and not prompt_initial_mismatches
                    and not prompt_final_mismatches
                ),
            }
        )

    first_divergence = None
    if hidden_mismatches:
        first_divergence = {"component": "layer_output_hidden", **hidden_mismatches[0]}
    elif initial_mismatches:
        first_divergence = {"phase": "prefill_state", **initial_mismatches[0]}
    elif not token_exact:
        first_divergence = {"phase": "tokens"}
    elif final_mismatches:
        first_divergence = {"phase": "final_state", **final_mismatches[0]}
    return {
        "suite": suite,
        "group_index": int(group_index),
        "repeat": int(repeat_index),
        "width": len(rows),
        "prompt_ids": [str(row["id"]) for row in rows],
        "prompt_lengths": [len(tokens) for tokens in prompt_tokens],
        "packed_prefill_plan": dict(owner.last_packed_prefill_plan),
        "token_exact": token_exact,
        "hidden_comparisons": int(hidden_comparisons),
        "hidden_mismatches": len(hidden_mismatches),
        "initial_state_mismatches": len(initial_mismatches),
        "dirty_before_flush": dirty_before_flush,
        "flush_executed": flush_executed,
        "final_state_mismatches": len(final_mismatches),
        "first_divergence": first_divergence,
        "passed": passed,
        "prompts": prompt_results,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repeats = int(args.repeats)
    decode_steps = int(args.decode_steps)
    group_size = int(args.group_size)
    if repeats < 3:
        raise ValueError("B4 category oracle requires at least three repeats")
    if decode_steps <= 0:
        raise ValueError("decode_steps must be positive")
    if group_size != 4:
        raise ValueError("B4 category oracle requires the production c4 group cap")
    model = args.model.expanduser().resolve()
    prompts_path = args.prompts.expanduser().resolve()
    heldouts_path = args.heldouts.expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    primary = load_prompt_rows(prompts_path)
    heldouts = load_prompt_rows(heldouts_path)
    _validate_prompt_contract(primary, heldouts)
    build_policy = _session_build_policy(args)

    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(model))
    all_rows = (*primary, *heldouts)
    tokens_by_id = {
        str(row["id"]): tuple(build_chat_prompt(tokenizer, str(row["prompt"])))
        for row in all_rows
    }
    max_prompt_tokens = max(len(tokens) for tokens in tokens_by_id.values())
    max_sequence_length = max_prompt_tokens + decode_steps + 2
    if max_sequence_length >= 1024:
        raise ValueError("B4 category oracle currently requires context < 1024")

    grouped_suites = (
        ("primary", _group_prompt_rows(primary, max_group_size=group_size)),
        ("heldout", _group_prompt_rows(heldouts, max_group_size=group_size)),
    )
    with ExitStack() as stack:
        owner = stack.enter_context(
            Qwen35GGUFResidentSession(
                model,
                backend=str(args.backend),
                max_sequence_length=max_sequence_length,
                **build_policy,
            )
        )
        if owner.runner is None or owner.runner.weights is None:
            raise RuntimeError("GGUF category owner runner is closed")
        sessions = [owner]
        for _ in range(2 * group_size - 1):
            sessions.append(
                stack.enter_context(
                    Qwen35GGUFResidentSession(
                        model,
                        backend=str(args.backend),
                        runtime=owner.runtime,
                        shared_runner=owner.runner,
                        max_sequence_length=max_sequence_length,
                        **build_policy,
                    )
                )
            )
        layer_ids = tuple(range(len(owner.runner.weights.config.layer_types)))
        group_results: list[dict[str, Any]] = []
        prompt_repeats: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for repeat_index in range(repeats):
            for suite, groups in grouped_suites:
                for group_index, group in enumerate(groups):
                    width = len(group)
                    result = _run_group(
                        owner=owner,
                        packed_sessions=tuple(sessions[:width]),
                        reference_sessions=tuple(sessions[group_size : group_size + width]),
                        rows=group,
                        prompt_tokens=tuple(tokens_by_id[str(row["id"])] for row in group),
                        layer_ids=layer_ids,
                        decode_steps=decode_steps,
                        repeat_index=repeat_index,
                        suite=suite,
                        group_index=group_index,
                    )
                    group_results.append(result)
                    for prompt in result["prompts"]:
                        prompt_repeats[str(prompt["id"])].append(prompt)
                    print(
                        f"repeat={repeat_index} suite={suite} group={group_index} "
                        f"width={width} passed={result['passed']}",
                        flush=True,
                    )
        target_arch = str(owner.runner.target_arch)
        resolved_backend = str(owner.runner.backend)

    deterministic, nondeterministic_prompts = _repeat_determinism(prompt_repeats)
    all_groups_passed = all(bool(result["passed"]) for result in group_results)
    all_prompts_passed = all(
        bool(prompt["passed"])
        for repeats_rows in prompt_repeats.values()
        for prompt in repeats_rows
    )
    hidden_comparisons = sum(int(result["hidden_comparisons"]) for result in group_results)
    hidden_mismatches = sum(int(result["hidden_mismatches"]) for result in group_results)
    initial_state_mismatches = sum(
        int(result["initial_state_mismatches"]) for result in group_results
    )
    final_state_mismatches = sum(
        int(result["final_state_mismatches"]) for result in group_results
    )
    passed = bool(
        all_groups_passed
        and all_prompts_passed
        and deterministic
        and hidden_mismatches == 0
        and initial_state_mismatches == 0
        and final_state_mismatches == 0
    )
    first_divergence = next(
        (
            result["first_divergence"]
            for result in group_results
            if result["first_divergence"] is not None
        ),
        None,
    )
    category_counts = Counter(str(row["category"]) for row in primary)
    heldout_category_counts = Counter(str(row["category"]) for row in heldouts)
    command = [sys.executable, "scripts/gguf_packed_ar_category_oracle.py", *sys.argv[1:]]
    return {
        "schema": 1,
        "kind": "gfx1100_gguf_packed_ar_category_oracle",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "performance_claim": False,
        "provenance": collect_artifact_provenance(
            repo_root=REPO_ROOT,
            configured_backend=str(args.backend),
            resolved_backend=resolved_backend,
            target_arch=target_arch,
            model_path=model,
            quant="gguf_q4_k_m",
            kv_dtype="bf16",
            command=command,
            build_profile=(
                "packed AR category equality; strict-exact GDN; groups <=c4; "
                "cached resident builds required" if args.require_cached_build else
                "packed AR category equality; strict-exact GDN; groups <=c4"
            ),
            timing_protocol=(
                "correctness only; canonical category plus heldout prompts; "
                "packed-vs-independent-c1 tokens/hidden/state/KV"
            ),
            warmups=0,
            repetitions=repeats,
            profiler={"used": False, "performance_measurement": False},
        ),
        "command": shlex.join(command),
        "build": {
            "compiler_version_file": (
                None if args.compiler_version_file is None else str(args.compiler_version_file)
            ),
            "compiler_version_supplied": build_policy["compiler_version"] is not None,
            "require_cached_build": build_policy["require_cached_build"],
        },
        "model": str(model),
        "backend": resolved_backend,
        "target_arch": target_arch,
        "contract": {
            "sampling": "greedy_top1",
            "gdn_prefill_mode": "exact",
            "kv_dtype": "bf16",
            "max_group_size": group_size,
            "decode_steps": decode_steps,
            "repeats": repeats,
            "layer_count": len(layer_ids),
            "state_and_hidden_comparison": "bit exact",
        },
        "prompt_suites": {
            "primary": {
                "path": str(prompts_path),
                "sha256": _sha256_file(prompts_path),
                "count": len(primary),
                "category_counts": dict(sorted(category_counts.items())),
                "prompts": _prompt_manifest(primary),
            },
            "heldout": {
                "path": str(heldouts_path),
                "sha256": _sha256_file(heldouts_path),
                "count": len(heldouts),
                "category_counts": dict(sorted(heldout_category_counts.items())),
                "prompts": _prompt_manifest(heldouts),
            },
        },
        "rollup": {
            "prompt_count": len(all_rows),
            "primary_prompt_count": len(primary),
            "heldout_prompt_count": len(heldouts),
            "group_executions": len(group_results),
            "group_widths_per_repeat": [
                len(group)
                for _, groups in grouped_suites
                for group in groups
            ],
            "prompt_repeat_executions": sum(len(rows) for rows in prompt_repeats.values()),
            "token_comparisons": len(all_rows) * repeats * (decode_steps + 1),
            "hidden_comparisons": hidden_comparisons,
            "hidden_mismatches": hidden_mismatches,
            "initial_state_mismatches": initial_state_mismatches,
            "final_state_mismatches": final_state_mismatches,
            "all_groups_passed": all_groups_passed,
            "all_prompts_passed": all_prompts_passed,
            "repeat_deterministic": deterministic,
            "nondeterministic_prompts": nondeterministic_prompts,
        },
        "groups": group_results,
        "prompt_repeats": dict(sorted(prompt_repeats.items())),
        "first_divergence": first_divergence,
        "notes": [
            "The primary suite is the complete committed mtpbench code/general_en/general_ja/mixed_ja_en fixture.",
            "The separate frozen category-heldout file is included because B3 changed long-prompt route selection.",
            "Groups are c4+c4+c2 for the ten primary prompts and c4+c4 for eight heldouts; no prompt is duplicated to fill c4.",
            "No timing or performance claim is made.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--heldouts", type=Path, default=DEFAULT_HELDOUTS)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.json)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
