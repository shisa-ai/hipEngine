#!/usr/bin/env python3
# ruff: noqa: E402
"""Screen asymmetric GGUF K/V formats across fixed mixed and natural prompts.

The harness keeps one Q4_K_M resident session and runs the committed ten-prompt
mtpbench category suite plus the existing ``mixed_v1`` control at one exact
context shape. Natural prompts are expanded with a deterministic rotation of
the other committed suite rows, while preserving the selected row as the final
query. Candidate caches are host-quantized and reconstructed into BF16 before
teacher-forced decode.

This is representation-fidelity evidence only. It does not measure native
storage, kernel performance, or task quality.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.gguf_mtp_bench import (
    IM_END_TOKEN,
    IM_START_TOKEN,
    THINK_END_TOKEN,
    THINK_START_TOKEN,
)
from scripts.gguf_mtp_category_bench import (
    DEFAULT_FULL_PROMPT_IDS,
    DEFAULT_HELDOUT_PROMPT_IDS,
    load_prompt_rows,
)
from scripts.qwen35_gguf_kv_format_ablation import (
    DEFAULT_MODEL,
    _cache_layout,
    _fixed_mixed_prompt_tokens,
    _prompt_sha256,
    _run_candidate_screen,
)
from scripts.qwen35_paro_kv_format_ablation import (
    _compact_run,
    _git_provenance,
    _parse_candidates,
)

DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl"
DEFAULT_CANDIDATES = (
    "baseline_max,group32,hadamard_group32,"
    "key_int8_value_bf16,key_group32_value_bf16,key_group16_value_bf16,"
    "key_hadamard_group32_value_bf16,key_bf16_value_int8,"
    "key_bf16_value_group32,key_group32_value_group16,key_group16_value_group32"
)
NATURAL_PROFILE = "natural_corpus_v1"
MIXED_PROFILE = "mixed_v1"


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    category: str
    split: str
    profile: str
    tokens: tuple[int, ...]
    token_sha256: str
    source_prompt_sha256: str | None
    current_query_tokens: int


def _text_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repeat_to_length(tokens: Sequence[int], length: int) -> list[int]:
    if int(length) < 0:
        raise ValueError("repeat length must be non-negative")
    if int(length) == 0:
        return []
    source = [int(token) for token in tokens]
    if not source:
        raise ValueError("cannot repeat an empty token sequence")
    repeats = math.ceil(int(length) / len(source))
    return (source * repeats)[: int(length)]


def _chat_envelope(tokenizer: Any) -> tuple[list[int], list[int]]:
    prefix = [IM_START_TOKEN, *[int(token) for token in tokenizer.encode("user\n")]]
    suffix = [
        IM_END_TOKEN,
        *[int(token) for token in tokenizer.encode("\n")],
        IM_START_TOKEN,
        *[int(token) for token in tokenizer.encode("assistant\n")],
        THINK_START_TOKEN,
        *[int(token) for token in tokenizer.encode("\n\n")],
        THINK_END_TOKEN,
        *[int(token) for token in tokenizer.encode("\n\n")],
    ]
    return prefix, suffix


def _natural_case_tokens(
    tokenizer: Any,
    prompt_rows: Sequence[dict[str, Any]],
    *,
    current_index: int,
    prompt_length: int,
) -> tuple[tuple[int, ...], int]:
    prefix, suffix = _chat_envelope(tokenizer)
    available = int(prompt_length) - len(prefix) - len(suffix)
    if available <= 0:
        raise ValueError("prompt_length is too short for the Qwen chat envelope")
    current = prompt_rows[int(current_index)]
    current_text = (
        f"\n\n[CURRENT {current['category']}:{current['id']}]\n"
        f"{current['prompt']}"
    )
    current_tokens = [int(token) for token in tokenizer.encode(current_text)]
    if len(current_tokens) > available:
        raise ValueError(
            f"prompt_length {prompt_length} cannot preserve final query {current['id']!r} "
            f"({len(current_tokens)} body tokens, {available} available)"
        )
    ordered_others = [
        *prompt_rows[int(current_index) + 1 :],
        *prompt_rows[: int(current_index)],
    ]
    filler_text = "".join(
        f"\n\n[{row['category']}:{row['id']}]\n{row['prompt']}"
        for row in ordered_others
    )
    filler_tokens = [int(token) for token in tokenizer.encode(filler_text)]
    filler = _repeat_to_length(filler_tokens, available - len(current_tokens))
    tokens = tuple([*prefix, *filler, *current_tokens, *suffix])
    if len(tokens) != int(prompt_length):  # pragma: no cover - construction invariant
        raise AssertionError("natural prompt construction did not reach exact length")
    return tokens, len(current_tokens)


def _build_prompt_cases(
    tokenizer: Any,
    prompt_rows: Sequence[dict[str, Any]],
    *,
    prompt_length: int,
    include_mixed_v1: bool,
    heldout_ids: Iterable[str],
) -> list[PromptCase]:
    heldout = {str(prompt_id) for prompt_id in heldout_ids}
    cases: list[PromptCase] = []
    for index, row in enumerate(prompt_rows):
        tokens, current_query_tokens = _natural_case_tokens(
            tokenizer,
            prompt_rows,
            current_index=index,
            prompt_length=prompt_length,
        )
        prompt_id = str(row["id"])
        cases.append(
            PromptCase(
                prompt_id=prompt_id,
                category=str(row["category"]),
                split="heldout" if prompt_id in heldout else "train",
                profile=NATURAL_PROFILE,
                tokens=tokens,
                token_sha256=_prompt_sha256(tokens),
                source_prompt_sha256=_text_sha256(str(row["prompt"])),
                current_query_tokens=int(current_query_tokens),
            )
        )
    if include_mixed_v1:
        mixed_tokens = tuple(_fixed_mixed_prompt_tokens(int(prompt_length)))
        cases.append(
            PromptCase(
                prompt_id=MIXED_PROFILE,
                category="mixed_synthetic",
                split="control",
                profile=MIXED_PROFILE,
                tokens=mixed_tokens,
                token_sha256=_prompt_sha256(mixed_tokens),
                source_prompt_sha256=None,
                current_query_tokens=0,
            )
        )
    return cases


def _prompt_metadata(case: PromptCase) -> dict[str, Any]:
    return {
        "id": case.prompt_id,
        "category": case.category,
        "split": case.split,
        "profile": case.profile,
        "token_count": len(case.tokens),
        "token_sha256": case.token_sha256,
        "source_prompt_sha256": case.source_prompt_sha256,
        "current_query_tokens": int(case.current_query_tokens),
        "distinct_tokens": len(set(case.tokens)),
    }


def _compact_prompt_result(case: PromptCase, screen: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": _prompt_metadata(case),
        "reference": _compact_run(screen["reference"]),
        "full_attention_layers": int(screen["full_layers"]),
        "baseline_memory": screen["baseline_memory"],
        "forced_input_ids": [int(token) for token in screen["forced_ids"]],
        "candidates": screen["rows"],
    }


def _scope_summary(entries: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    if not entries:
        return {
            "prompt_count": 0,
            "positions": 0,
            "mean_kl": None,
            "max_kl": None,
            "top1_agreement": None,
            "worst_prompt_mean_kl": None,
            "worst_prompt_id": None,
            "prompt_gate_pass_count": 0,
            "all_prompt_gates_passed": False,
        }
    kl_values: list[float] = []
    top1_matches: list[bool] = []
    for _prompt, candidate in entries:
        gate = candidate["logit_gate"]
        kl_values.extend(float(value) for value in gate["kl"])
        top1_matches.extend(bool(value) for value in gate["top1_matches"])
    worst_prompt, worst_candidate = max(
        entries,
        key=lambda item: (float(item[1]["logit_gate"]["mean_kl"]), str(item[0]["id"])),
    )
    passed = sum(bool(candidate["quality_gate_passed"]) for _prompt, candidate in entries)
    return {
        "prompt_count": len(entries),
        "positions": len(kl_values),
        "mean_kl": float(sum(kl_values) / len(kl_values)),
        "max_kl": float(max(kl_values)),
        "top1_agreement": float(sum(top1_matches) / len(top1_matches)),
        "worst_prompt_mean_kl": float(worst_candidate["logit_gate"]["mean_kl"]),
        "worst_prompt_id": str(worst_prompt["id"]),
        "prompt_gate_pass_count": int(passed),
        "all_prompt_gates_passed": bool(passed == len(entries)),
    }


def _aggregate_candidates(
    prompt_results: Sequence[dict[str, Any]],
    *,
    extra_budget_bytes: int,
) -> list[dict[str, Any]]:
    if not prompt_results:
        raise ValueError("expected at least one prompt result")
    candidate_names = [str(row["name"]) for row in prompt_results[0]["candidates"]]
    if not candidate_names:
        raise ValueError("expected at least one candidate result")
    expected_names = set(candidate_names)
    for result in prompt_results:
        actual_names = {str(row["name"]) for row in result["candidates"]}
        if actual_names != expected_names:
            raise ValueError("candidate names differ across prompt results")

    summaries: list[dict[str, Any]] = []
    for name in candidate_names:
        entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for result in prompt_results:
            candidate = next(row for row in result["candidates"] if row["name"] == name)
            entries.append((result["prompt"], candidate))
        natural = [entry for entry in entries if entry[0]["profile"] == NATURAL_PROFILE]
        train = [entry for entry in natural if entry[0]["split"] == "train"]
        heldout = [entry for entry in natural if entry[0]["split"] == "heldout"]
        mixed = [entry for entry in entries if entry[0]["profile"] == MIXED_PROFILE]
        categories = {
            category: _scope_summary([entry for entry in natural if entry[0]["category"] == category])
            for category in sorted({str(entry[0]["category"]) for entry in natural})
        }
        memory_values = {int(candidate["target_context_memory"]["total_bytes"]) for _prompt, candidate in entries}
        extra_values = {int(candidate["extra_bytes_over_baseline"]) for _prompt, candidate in entries}
        if len(memory_values) != 1 or len(extra_values) != 1:
            raise ValueError(f"candidate {name} memory accounting differs across prompts")
        all_passed = all(bool(candidate["quality_gate_passed"]) for _prompt, candidate in entries)
        first_failed = next(
            (str(prompt["id"]) for prompt, candidate in entries if not candidate["quality_gate_passed"]),
            None,
        )
        first = entries[0][1]
        summaries.append(
            {
                "name": name,
                "format": {
                    key: first.get(key)
                    for key in (
                        "strategy",
                        "k_mode",
                        "v_mode",
                        "k_group_size",
                        "v_group_size",
                        "k_clip_ratio",
                        "v_clip_ratio",
                        "hadamard_group_size",
                    )
                },
                "target_context_memory": first["target_context_memory"],
                "extra_bytes_over_baseline": next(iter(extra_values)),
                "within_extra_memory_budget": bool(next(iter(extra_values)) <= int(extra_budget_bytes)),
                "all_prompt_gates_passed": bool(all_passed),
                "first_failed_prompt": first_failed,
                "transfer_eligible": bool(
                    all_passed
                    and natural
                    and heldout
                    and mixed
                    and next(iter(extra_values)) <= int(extra_budget_bytes)
                ),
                "scopes": {
                    "all": _scope_summary(entries),
                    "natural_full": _scope_summary(natural),
                    "train": _scope_summary(train),
                    "heldout": _scope_summary(heldout),
                    "mixed_v1": _scope_summary(mixed),
                    "categories": categories,
                },
            }
        )
    return summaries


def _validate_canonical_rows(rows: Sequence[dict[str, Any]]) -> None:
    actual_ids = tuple(str(row["id"]) for row in rows)
    if actual_ids != DEFAULT_FULL_PROMPT_IDS:
        raise ValueError(
            "asymmetric suite requires the canonical mtpbench prompt IDs in committed order: "
            f"{actual_ids!r}"
        )
    categories = {str(row["category"]) for row in rows}
    heldout_categories = {
        str(row["category"])
        for row in rows
        if str(row["id"]) in DEFAULT_HELDOUT_PROMPT_IDS
    }
    if heldout_categories != categories:
        raise ValueError("canonical heldout split must cover every prompt category")


def run(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.core.dtype import DType
    from hipengine.kvcache.policy import FixedPagedKVPolicy
    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    started = time.perf_counter()
    prompt_rows = load_prompt_rows(args.prompts)
    _validate_canonical_rows(prompt_rows)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    prompt_cases = _build_prompt_cases(
        tokenizer,
        prompt_rows,
        prompt_length=int(args.prompt_length),
        include_mixed_v1=bool(args.include_mixed_v1),
        heldout_ids=DEFAULT_HELDOUT_PROMPT_IDS,
    )
    if not any(case.profile == MIXED_PROFILE for case in prompt_cases):
        raise ValueError("retained asymmetric suite requires --include-mixed-v1")

    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
        if not compiler_version.strip():
            raise ValueError("compiler version file is empty")
    candidates = _parse_candidates(args.candidates, head_dim=256)
    extra_budget_bytes = int(float(args.extra_budget_gib) * 1024**3)
    prefill_config = PrefillConfig(attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens))
    prompt_results: list[dict[str, Any]] = []
    with Qwen35GGUFResidentSession(
        args.model,
        backend=args.backend,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=int(args.prompt_length) + int(args.decode_steps) + 2,
        use_wmma_prefill=True,
        use_gemv_decode=True,
        prefill_config=prefill_config,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        layout = _cache_layout(session)
        if layout.head_dim != 256:
            candidates = _parse_candidates(args.candidates, head_dim=layout.head_dim)
        for index, case in enumerate(prompt_cases, start=1):
            print(
                f"[asymmetric-kv] {index}/{len(prompt_cases)} {case.prompt_id}",
                file=sys.stderr,
                flush=True,
            )
            screen = _run_candidate_screen(
                session,
                prompt_tokens=case.tokens,
                prompt_length=len(case.tokens),
                decode_steps=int(args.decode_steps),
                sample_tokens=int(args.sample_tokens),
                scale_dtype=args.scale_dtype,
                target_context_tokens=int(args.target_context_tokens),
                candidates=candidates,
                kl_threshold=float(args.kl_threshold),
                top1_threshold=float(args.top1_threshold),
                extra_budget_bytes=extra_budget_bytes,
            )
            prompt_results.append(_compact_prompt_result(case, screen))
            del screen
            gc.collect()
        target_arch = session.runner.target_arch
        resolved_backend = session.backend
        layout = _cache_layout(session)
    gc.collect()

    candidate_summary = _aggregate_candidates(
        prompt_results,
        extra_budget_bytes=extra_budget_bytes,
    )
    elapsed_seconds = float(time.perf_counter() - started)
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=args.backend,
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16_reference_host_asymmetric_emulation",
        command=["python3", "scripts/qwen35_gguf_kv_asymmetric_suite.py", *sys.argv[1:]],
        timing_protocol="setup-inclusive diagnostic wall; quality claim only",
        warmups=0,
        repetitions=1,
    )
    return {
        "schema": 1,
        "kind": "qwen35_gguf_kv_asymmetric_prompt_suite",
        "status": "diagnostic_complete",
        "performance_claim": False,
        "provenance": provenance,
        "git": _git_provenance(),
        "model": str(args.model.resolve()),
        "backend": resolved_backend,
        "target_arch": target_arch,
        "prompt_suite": {
            "path": str(args.prompts.resolve()),
            "sha256": _file_sha256(args.prompts),
            "natural_prompt_ids": [str(row["id"]) for row in prompt_rows],
            "natural_prompt_count": len(prompt_rows),
            "heldout_ids": sorted(DEFAULT_HELDOUT_PROMPT_IDS),
            "categories": sorted({str(row["category"]) for row in prompt_rows}),
            "includes_mixed_v1": True,
            "total_cases": len(prompt_cases),
            "expansion": (
                "natural_corpus_v1: exact Qwen chat envelope; deterministic rotation/repetition of the "
                "other nine committed prompts; selected prompt preserved in full as final user query"
            ),
        },
        "workload": {
            "prompt_length": int(args.prompt_length),
            "decode_steps": int(args.decode_steps),
            "sample_tokens": min(int(args.sample_tokens), int(args.prompt_length)),
            "teacher": "per-prompt BF16 reference tokens",
        },
        "shape": {
            "full_attention_layers": len(layout.full_layer_ids),
            "full_attention_layer_ids": list(layout.full_layer_ids),
            "num_kv_heads": int(layout.num_kv_heads),
            "head_dim": int(layout.head_dim),
            "scale_dtype": args.scale_dtype,
        },
        "quality_thresholds": {
            "kl_mean_max_per_prompt": float(args.kl_threshold),
            "top1_agreement_min_per_prompt": float(args.top1_threshold),
            "transfer_rule": "every natural full/train/heldout/category row and mixed_v1 must pass",
        },
        "target_memory": {
            "context_tokens": int(args.target_context_tokens),
            "extra_budget_bytes": extra_budget_bytes,
            "baseline": prompt_results[0]["baseline_memory"],
        },
        "prompt_results": prompt_results,
        "candidate_summary": candidate_summary,
        "transfer_eligible_candidates": [
            row["name"] for row in candidate_summary if row["transfer_eligible"]
        ],
        "elapsed_seconds": elapsed_seconds,
        "wall_time_budget_seconds": float(args.wall_time_budget_seconds),
        "wall_time_budget_met": bool(elapsed_seconds <= float(args.wall_time_budget_seconds)),
        "notes": [
            "Host emulation only; no native storage, throughput, or support claim.",
            "All prompt/candidate comparisons use identical weights and per-prompt teacher-forced history.",
            "BF16 components are copied exactly; Hadamard is applied only to quantized components.",
            "The current decode row attends in BF16 and is round-tripped immediately afterward.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--sample-tokens", type=int, default=128)
    parser.add_argument("--include-mixed-v1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--scale-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--target-context-tokens", type=int, default=262144)
    parser.add_argument("--extra-budget-gib", type=float, default=1.5)
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--wall-time-budget-seconds", type=float, default=1200.0)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=512)
    parser.add_argument("--backend", choices=("auto", "hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    for name in ("prompt_length", "sample_tokens", "target_context_tokens"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.decode_steps) < 0 or float(args.extra_budget_gib) < 0.0:
        raise ValueError("decode steps and extra budget must be non-negative")
    if float(args.wall_time_budget_seconds) <= 0.0:
        raise ValueError("wall-time budget must be positive")
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
