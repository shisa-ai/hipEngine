#!/usr/bin/env python3
# ruff: noqa: E402
"""Matched true-AR and transactional MTP suite for dense Qwen3.6 GGUF.

The suite loads one target and one trailing-NextN provider, warms each route,
then runs the committed natural-prompt category fixture through true scalar AR
and exact B1/B2/B3 MTP. Throughput is transition-normalized: the first visible
sample comes from prefill, so a 25-output request has 24 timed decode
transitions. Optional ROCTX markers make proposal/verify/commit windows usable
as a rocprofv3 leaf without profiling a parent process.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import sys
import time
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.loading import load_gguf_index
from hipengine.runtime.qwen35_gguf_mtp import (
    Qwen35GGUFMTPDecodeSession,
    Qwen35GGUFTransactionalVerifier,
)
from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNDraftProvider
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import load_prompt_rows
from scripts.qwen35_gguf_bench import _gguf_tensor_inventory_summary

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl"
HELDOUT_PROMPT_IDS = frozenset(
    {
        "code_markdown_table",
        "general_en_explain",
        "general_ja_explain",
        "mixed_ja_en_review",
    }
)
ROCTX_MARKER_PREFIX = "qwen36_dense_mtp_"
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


def parse_candidate_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not budgets or any(item not in {1, 2, 3} for item in budgets):
        raise ValueError("candidate budgets must be a comma-separated subset of 1,2,3")
    if len(set(budgets)) != len(budgets):
        raise ValueError("candidate budgets must not contain duplicates")
    return budgets


def timed_transition_count(max_new_tokens: int) -> int:
    outputs = int(max_new_tokens)
    if outputs <= 0:
        raise ValueError("max_new_tokens must be positive")
    return outputs - 1


def suite_speed_claim_eligible(
    *,
    prompts_path: Path,
    prompt_ids: Sequence[str],
    max_new_tokens: int,
    all_exact: bool,
) -> bool:
    return bool(
        all_exact
        and Path(prompts_path).resolve() == DEFAULT_PROMPTS.resolve()
        and tuple(str(prompt_id) for prompt_id in prompt_ids) == FULL_PROMPT_IDS
        and int(max_new_tokens) == 25
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_sha256(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<q", int(token)))
    return digest.hexdigest()


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _aggregate_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    prompt_ids = _ordered_unique([str(row["id"]) for row in rows])
    categories = _ordered_unique([str(row["category"]) for row in rows])
    visible_outputs = sum(int(row["visible_outputs"]) for row in rows)
    transitions = sum(int(row["timed_transitions"]) for row in rows)
    decode_seconds = sum(float(row["decode_seconds"]) for row in rows)
    request_wall_seconds = sum(float(row["request_wall_seconds"]) for row in rows)
    accepted = sum(int(row.get("accepted_draft_tokens", 0)) for row in rows)
    proposed = sum(int(row.get("proposed_draft_tokens", 0)) for row in rows)
    cycles = sum(int(row.get("cycles", 0)) for row in rows)
    target_rows = sum(int(row.get("target_forward_rows", 0)) for row in rows)
    stage_seconds: dict[str, float] = defaultdict(float)
    for row in rows:
        raw_stages = row.get("stage_seconds", {})
        if not isinstance(raw_stages, dict):
            raise TypeError("stage_seconds must be a mapping")
        for key, value in raw_stages.items():
            stage_seconds[str(key)] += float(value)
    stage_total = sum(stage_seconds.values())
    tolerance = max(1e-9, decode_seconds * 1e-6)
    return {
        "prompt_count": len(prompt_ids),
        "request_count": len(rows),
        "prompt_ids": prompt_ids,
        "categories": categories,
        "visible_outputs": visible_outputs,
        "timed_transitions": transitions,
        "decode_seconds": decode_seconds,
        "decode_tok_s_weighted": (
            transitions / decode_seconds if transitions > 0 and decode_seconds > 0.0 else 0.0
        ),
        "request_wall_seconds": request_wall_seconds,
        "client_transition_tok_s": (
            transitions / request_wall_seconds
            if transitions > 0 and request_wall_seconds > 0.0
            else 0.0
        ),
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "draft_acceptance": accepted / proposed if proposed > 0 else None,
        "accepted_per_output": accepted / visible_outputs if visible_outputs > 0 else None,
        "accepted_per_transition": accepted / transitions if transitions > 0 else None,
        "cycles": cycles,
        "target_passes": cycles,
        "visible_transitions_per_cycle": transitions / cycles if cycles > 0 else None,
        "target_forward_rows": target_rows,
        "stage_seconds": dict(sorted(stage_seconds.items())),
        "stage_reconciliation_error_seconds": decode_seconds - stage_total,
        "stage_reconciled": abs(decode_seconds - stage_total) <= tolerance,
    }


def aggregate_scopes(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    selected = list(rows)
    train = [row for row in selected if str(row["id"]) not in HELDOUT_PROMPT_IDS]
    heldout = [row for row in selected if str(row["id"]) in HELDOUT_PROMPT_IDS]
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        by_category[str(row["category"])].append(row)
    return {
        "full": _aggregate_rows(selected),
        "train": _aggregate_rows(train),
        "heldout": _aggregate_rows(heldout),
        "categories": {
            category: _aggregate_rows(category_rows)
            for category, category_rows in sorted(by_category.items())
        },
    }


class _RoctxMarkers:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._push = None
        self._pop = None
        if not self.enabled:
            return
        library = ctypes.CDLL("libroctx64.so")
        self._push = library.roctxRangePushA
        self._pop = library.roctxRangePop
        self._push.argtypes = [ctypes.c_char_p]
        self._push.restype = ctypes.c_int
        self._pop.argtypes = []
        self._pop.restype = ctypes.c_int

    def push(self, name: str) -> None:
        if self._push is not None:
            self._push(name.encode("utf-8"))

    def pop(self) -> None:
        if self._pop is not None:
            self._pop()


class _CallLedger:
    def __init__(self, markers: _RoctxMarkers) -> None:
        self.markers = markers
        self.recording = True
        self._counter = 0
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._marker_names: dict[str, list[str]] = defaultdict(list)

    def reset(self) -> None:
        self._counter = 0
        self._samples.clear()
        self._marker_names.clear()

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        if not self.recording:
            yield
            return
        self._counter += 1
        marker_name = f"{ROCTX_MARKER_PREFIX}{phase}_{self._counter:06d}"
        self._marker_names[str(phase)].append(marker_name)
        self.markers.push(marker_name)
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.markers.pop()
            self._samples[str(phase)].append(elapsed)

    def snapshot(self) -> dict[str, object]:
        return {
            "totals_seconds": {
                phase: sum(samples) for phase, samples in sorted(self._samples.items())
            },
            "samples_seconds": {
                phase: list(samples) for phase, samples in sorted(self._samples.items())
            },
            "marker_names": {
                phase: list(names) for phase, names in sorted(self._marker_names.items())
            },
        }


class _TimedDraftProvider:
    def __init__(self, provider: Any, ledger: _CallLedger) -> None:
        self._provider = provider
        self._ledger = ledger

    def propose(self, *args: Any, **kwargs: Any) -> Any:
        with self._ledger.measure("proposal"):
            return self._provider.propose(*args, **kwargs)

    def advance_full_accept_tail(self, *args: Any, **kwargs: Any) -> Any:
        with self._ledger.measure("proposal_update"):
            return self._provider.advance_full_accept_tail(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


class _TimedVerifier:
    def __init__(self, verifier: Any, ledger: _CallLedger) -> None:
        self._verifier = verifier
        self._ledger = ledger

    def prepare(self, *args: Any, **kwargs: Any) -> Any:
        with self._ledger.measure("target_verify"):
            return self._verifier.prepare(*args, **kwargs)

    def commit(self, *args: Any, **kwargs: Any) -> Any:
        with self._ledger.measure("target_commit"):
            return self._verifier.commit(*args, **kwargs)

    def finish(self, *args: Any, **kwargs: Any) -> Any:
        with self._ledger.measure("target_finish"):
            return self._verifier.finish(*args, **kwargs)

    def rollback(self, *args: Any, **kwargs: Any) -> Any:
        with self._ledger.measure("target_rollback"):
            return self._verifier.rollback(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._verifier, name)


def _run_ar(
    target: Qwen35GGUFResidentSession,
    prompt_tokens: Sequence[int],
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    target.reset()
    request_started = time.perf_counter()
    prefill_started = request_started
    first = target.prefill(prompt_tokens, use_bulk=False, return_logits=False)
    prefill_seconds = time.perf_counter() - prefill_started
    generated = [int(first.token_id)]
    decode_started = time.perf_counter()
    while len(generated) < int(max_new_tokens):
        generated.append(int(target.step(generated[-1], return_logits=False).token_id))
    decode_seconds = time.perf_counter() - decode_started
    request_wall_seconds = time.perf_counter() - request_started
    transitions = timed_transition_count(max_new_tokens)
    return {
        "token_ids": generated,
        "token_sha256_i64": _token_sha256(generated),
        "visible_outputs": len(generated),
        "timed_transitions": transitions,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "request_wall_seconds": request_wall_seconds,
        "client_transition_tok_s": (
            transitions / request_wall_seconds
            if transitions > 0 and request_wall_seconds > 0.0
            else 0.0
        ),
        "decode_tok_s_transition_normalized": (
            transitions / decode_seconds if transitions > 0 and decode_seconds > 0.0 else 0.0
        ),
        "accepted_draft_tokens": 0,
        "proposed_draft_tokens": 0,
        "cycles": 0,
        "target_passes": 0,
        "target_forward_rows": transitions,
        "stage_seconds": {"autoregressive_step": decode_seconds},
    }


def _run_mtp(
    decoder: Qwen35GGUFMTPDecodeSession,
    ledger: _CallLedger,
    prompt_tokens: Sequence[int],
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    ledger.reset()
    with ledger.measure("generation"):
        result = decoder.generate(
            prompt_tokens,
            max_new_tokens=int(max_new_tokens),
            return_cycle_logits=False,
            use_bulk_prefill=False,
            prefill_draft=True,
        )
    instrumentation = ledger.snapshot()
    totals = instrumentation["totals_seconds"]
    assert isinstance(totals, dict)
    commit_finish_seconds = float(totals.get("target_commit", 0.0)) + float(
        totals.get("target_finish", 0.0)
    )
    accounted = float(result.proposal_seconds) + float(result.verify_seconds) + commit_finish_seconds
    residual = float(result.decode_seconds) - accounted
    if residual < -max(1e-6, float(result.decode_seconds) * 1e-5):
        raise RuntimeError(
            "MTP stage timers exceed complete decode wall: "
            f"decode={result.decode_seconds:.9f}s accounted={accounted:.9f}s"
        )
    residual = max(0.0, residual)
    transitions = len(result.token_ids) - 1
    expected_transitions = timed_transition_count(max_new_tokens)
    if transitions != expected_transitions:
        raise RuntimeError(
            f"MTP returned {transitions} timed transitions, expected {expected_transitions}"
        )
    proposed = sum(len(record["draft_tokens"]) for record in result.cycle_records)
    payload = result.to_json_dict()
    payload.update(
        {
            "token_sha256_i64": _token_sha256(result.token_ids),
            "visible_outputs": len(result.token_ids),
            "timed_transitions": transitions,
            "request_wall_seconds": float(totals.get("generation", 0.0)),
            "client_transition_tok_s": (
                transitions / float(totals.get("generation", 0.0))
                if transitions > 0 and float(totals.get("generation", 0.0)) > 0.0
                else 0.0
            ),
            "decode_tok_s_transition_normalized": (
                transitions / result.decode_seconds
                if transitions > 0 and result.decode_seconds > 0.0
                else 0.0
            ),
            "decode_tok_s_legacy_visible_output_numerator": float(result.decode_tok_s),
            "proposed_draft_tokens": int(proposed),
            "draft_acceptance": (
                result.accepted_draft_tokens / proposed if proposed > 0 else None
            ),
            "accepted_per_output": (
                result.accepted_draft_tokens / len(result.token_ids)
                if result.token_ids
                else None
            ),
            "accepted_per_transition": (
                result.accepted_draft_tokens / transitions if transitions > 0 else None
            ),
            "target_passes": int(result.cycles),
            "stage_seconds": {
                "proposal": float(result.proposal_seconds),
                "target_verify": float(result.verify_seconds),
                "target_commit_finish": commit_finish_seconds,
                "scheduler_accept_replay_host_residual": residual,
            },
            "stage_instrumentation": instrumentation,
        }
    )
    return payload


def _attach_identity(
    row: dict[str, object],
    *,
    prompt: dict[str, Any],
    run_index: int,
) -> dict[str, object]:
    return {
        "run": int(run_index),
        "id": str(prompt["id"]),
        "category": str(prompt["category"]),
        "prompt_chars": len(str(prompt["prompt"])),
        "prompt_sha256": hashlib.sha256(str(prompt["prompt"]).encode("utf-8")).hexdigest(),
        **row,
    }


def _add_ratios(mtp: dict[str, object], ar: dict[str, object]) -> None:
    for scope in ("full", "train", "heldout"):
        mtp_row = mtp[scope]
        ar_row = ar[scope]
        assert isinstance(mtp_row, dict) and isinstance(ar_row, dict)
        denominator = float(ar_row["decode_tok_s_weighted"])
        mtp_row["mtp_vs_true_ar"] = (
            float(mtp_row["decode_tok_s_weighted"]) / denominator if denominator > 0.0 else None
        )
    mtp_categories = mtp["categories"]
    ar_categories = ar["categories"]
    assert isinstance(mtp_categories, dict) and isinstance(ar_categories, dict)
    for category, mtp_row in mtp_categories.items():
        if category not in ar_categories:
            continue
        ar_row = ar_categories[category]
        assert isinstance(mtp_row, dict) and isinstance(ar_row, dict)
        denominator = float(ar_row["decode_tok_s_weighted"])
        mtp_row["mtp_vs_true_ar"] = (
            float(mtp_row["decode_tok_s_weighted"]) / denominator if denominator > 0.0 else None
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--max-new-tokens", type=int, default=25)
    parser.add_argument("--candidate-budgets", type=parse_candidate_budgets, default=(1, 2, 3))
    parser.add_argument(
        "--target-verify-mode",
        choices=("native", "serial-exact"),
        default="native",
        help="dense target block route; serial-exact is the rollback control",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--roctx-markers", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if int(args.max_new_tokens) <= 1:
        raise ValueError("--max-new-tokens must exceed one for transition-normalized timing")
    if int(args.runs) <= 0:
        raise ValueError("--runs must be positive")
    model = Path(args.model).resolve()
    prompts_path = Path(args.prompts).resolve()
    prompts = load_prompt_rows(prompts_path)
    if args.limit is not None:
        prompts = prompts[: max(0, int(args.limit))]
    if not prompts:
        raise ValueError("selected prompt suite is empty")

    model_info = load_gguf_index(model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(model_info)
    encoded_prompts: dict[str, tuple[int, ...]] = {
        str(prompt["id"]): tuple(
            int(token) for token in build_chat_prompt(tokenizer, str(prompt["prompt"]), reasoning="off")
        )
        for prompt in prompts
    }
    max_prompt_tokens = max(len(tokens) for tokens in encoded_prompts.values())
    max_sequence_length = int(args.max_sequence_length) or (
        max_prompt_tokens + int(args.max_new_tokens) + max(args.candidate_budgets) + 8
    )
    if max_sequence_length < max_prompt_tokens + int(args.max_new_tokens):
        raise ValueError("--max-sequence-length is too small for the selected suite")
    compiler_version = (
        None
        if args.compiler_version_file is None
        else Path(args.compiler_version_file).read_text(encoding="utf-8")
    )

    reset_memory_stats()
    markers = _RoctxMarkers(bool(args.roctx_markers))
    ledger = _CallLedger(markers)
    ar_rows: list[dict[str, object]] = []
    mtp_rows: dict[str, list[dict[str, object]]] = {
        str(budget): [] for budget in args.candidate_budgets
    }
    load_started = time.perf_counter()
    with Qwen35GGUFResidentSession(
        model,
        max_sequence_length=max_sequence_length,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as target:
        target.select_prefill_quant(str(args.quant))
        provider = Qwen35GGUFNextNDraftProvider.from_model(
            model,
            max_positions=max_sequence_length,
            max_requests=1,
            runtime=target.runtime,
            require_cached_build=bool(args.require_cached_build),
        )
        verifier = Qwen35GGUFTransactionalVerifier(
            target,
            max_candidate_budget=max(args.candidate_budgets),
            quant=str(args.quant),
            target_verify_mode=str(args.target_verify_mode),
        )
        timed_provider = _TimedDraftProvider(provider, ledger)
        timed_verifier = _TimedVerifier(verifier, ledger)
        decoders = {
            int(budget): Qwen35GGUFMTPDecodeSession(
                target,
                timed_provider,
                candidate_budget=int(budget),
                quant=str(args.quant),
                verifier=timed_verifier,
                owns_verifier=False,
            )
            for budget in args.candidate_budgets
        }
        load_seconds = time.perf_counter() - load_started
        try:
            if bool(args.warmup):
                warmup_tokens = tuple(
                    int(token)
                    for token in build_chat_prompt(
                        tokenizer,
                        "Write a Python function add(a, b). Return only code.",
                        reasoning="off",
                    )
                )
                warmup_outputs = min(int(args.max_new_tokens), max(5, max(args.candidate_budgets) + 2))
                ledger.recording = False
                _run_ar(target, warmup_tokens, max_new_tokens=warmup_outputs)
                for budget in args.candidate_budgets:
                    _run_mtp(
                        decoders[int(budget)],
                        ledger,
                        warmup_tokens,
                        max_new_tokens=warmup_outputs,
                    )
                ledger.recording = True

            for run_index in range(int(args.runs)):
                for prompt in prompts:
                    prompt_id = str(prompt["id"])
                    prompt_tokens = encoded_prompts[prompt_id]
                    ar = _attach_identity(
                        _run_ar(
                            target,
                            prompt_tokens,
                            max_new_tokens=int(args.max_new_tokens),
                        ),
                        prompt=prompt,
                        run_index=run_index,
                    )
                    ar["prompt_tokens"] = len(prompt_tokens)
                    ar_rows.append(ar)
                    print(
                        f"AR run={run_index} prompt={prompt_id} "
                        f"tok_s={float(ar['decode_tok_s_transition_normalized']):.6f}",
                        flush=True,
                    )
                    for budget in args.candidate_budgets:
                        mtp = _attach_identity(
                            _run_mtp(
                                decoders[int(budget)],
                                ledger,
                                prompt_tokens,
                                max_new_tokens=int(args.max_new_tokens),
                            ),
                            prompt=prompt,
                            run_index=run_index,
                        )
                        mtp["prompt_tokens"] = len(prompt_tokens)
                        mtp["exact_greedy_match"] = mtp["token_ids"] == ar["token_ids"]
                        mtp["speedup_vs_true_ar"] = (
                            float(mtp["decode_tok_s_transition_normalized"])
                            / float(ar["decode_tok_s_transition_normalized"])
                        )
                        mtp_rows[str(budget)].append(mtp)
                        print(
                            f"MTP B{budget} run={run_index} prompt={prompt_id} "
                            f"tok_s={float(mtp['decode_tok_s_transition_normalized']):.6f} "
                            f"accepted={int(mtp['accepted_draft_tokens'])} "
                            f"exact={bool(mtp['exact_greedy_match'])}",
                            flush=True,
                        )
        finally:
            for decoder in decoders.values():
                decoder.close()
            verifier.close()
            provider.close()

    ar_summary = aggregate_scopes(ar_rows)
    mtp_summaries = {
        budget: aggregate_scopes(rows) for budget, rows in mtp_rows.items()
    }
    for summary in mtp_summaries.values():
        _add_ratios(summary, ar_summary)
    all_exact = all(
        bool(row["exact_greedy_match"])
        and bool(row["gpu_accept_match_cpu"])
        and abs(sum(float(value) for value in row["stage_seconds"].values()) - float(row["decode_seconds"]))
        <= max(1e-9, float(row["decode_seconds"]) * 1e-6)
        for rows in mtp_rows.values()
        for row in rows
    )
    prompt_file_sha256 = _sha256_file(prompts_path)
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend="auto",
        resolved_backend="hip_gfx1100",
        target_arch="gfx1100",
        model_path=model,
        quant=str(args.quant),
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        environment={
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get("HIPENGINE_GGUF_DECODE_REPACK"),
            "HIPENGINE_REQUIRE_CACHED_BUILD": os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD"),
        },
        build_profile="qwen36_dense_gguf_ar_mtp_suite",
        timing_protocol="natural25 first-output-from-prefill; 24 transition-normalized decode steps",
        warmups=1 if bool(args.warmup) else 0,
        repetitions=int(args.runs),
        profiler={"enabled": bool(args.roctx_markers), "kind": "roctx marker leaf"},
        hipcc_version=compiler_version,
    )
    return {
        "schema": 1,
        "kind": "qwen36_dense_gguf_ar_mtp_suite",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_exact" if all_exact else "correctness_failed",
        "performance_claim": False,
        "speed_claim_eligible": bool(
            suite_speed_claim_eligible(
                prompts_path=prompts_path,
                prompt_ids=[str(prompt["id"]) for prompt in prompts],
                max_new_tokens=int(args.max_new_tokens),
                all_exact=all_exact,
            )
            and not bool(provenance["dirty"])
            and provenance["device_name"] == "AMD Radeon Pro W7900"
        ),
        "provenance": provenance,
        "command": shlex.join([str(Path(sys.executable).resolve()), *sys.argv]),
        "model": {
            "path": str(model),
            "size_bytes": model.stat().st_size,
            "architecture": model_info.architecture,
            "file_type": model_info.file_type_name,
            "tensor_count": model_info.tensor_count,
            "gguf_inventory": _gguf_tensor_inventory_summary(model_info),
        },
        "workload": {
            "prompt_file": str(prompts_path),
            "prompt_file_sha256": prompt_file_sha256,
            "prompt_count": len(prompts),
            "prompt_ids": [str(prompt["id"]) for prompt in prompts],
            "categories": sorted({str(prompt["category"]) for prompt in prompts}),
            "heldout_ids": sorted(HELDOUT_PROMPT_IDS),
            "train_ids": [
                str(prompt["id"])
                for prompt in prompts
                if str(prompt["id"]) not in HELDOUT_PROMPT_IDS
            ],
            "prompt_render": "Qwen chat template; reasoning off",
            "max_new_tokens_visible": int(args.max_new_tokens),
            "timed_decode_transitions": timed_transition_count(args.max_new_tokens),
            "candidate_budgets": list(args.candidate_budgets),
            "target_verify_mode": str(args.target_verify_mode),
            "runs": int(args.runs),
            "warmup": bool(args.warmup),
            "max_sequence_length": max_sequence_length,
            "sampling": {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "min_p": 0.0,
                "seed": 12345,
            },
        },
        "timing_contract": {
            "first_visible_output": "target prefill sample; excluded from decode wall",
            "decode_numerator": "max_new_tokens - 1 timed transitions",
            "prefill": "excluded from AR and MTP decode wall",
            "mtp_complete_wall": "proposal + target_verify + target_commit_finish + scheduler_accept_replay_host_residual",
            "profile_markers": bool(args.roctx_markers),
            "roctx_marker_prefix": ROCTX_MARKER_PREFIX,
        },
        "load_seconds": load_seconds,
        "correctness": {
            "all_exact_greedy": all_exact,
            "all_gpu_accept_match_cpu": all(
                bool(row["gpu_accept_match_cpu"])
                for rows in mtp_rows.values()
                for row in rows
            ),
        },
        "summary": {"true_ar": ar_summary, "mtp": mtp_summaries},
        "rows": {"true_ar": ar_rows, "mtp": mtp_rows},
        "memory_after_close": memory_stats(),
    }


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "true_ar": payload["summary"]["true_ar"]["full"],
                "mtp": {
                    budget: summary["full"]
                    for budget, summary in payload["summary"]["mtp"].items()
                },
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if payload["status"] == "complete_exact" else 1


if __name__ == "__main__":
    raise SystemExit(main())
