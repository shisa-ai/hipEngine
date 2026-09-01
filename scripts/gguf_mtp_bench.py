#!/usr/bin/env python3
"""M6 MTP speculative decoding benchmark.

Runs the hipEngine GGUF MTP speculative decoding pipeline with a fixed prompt,
measures decode speed (tokens/sec) and acceptance metrics (accept_per_draft,
accepted_per_output), and saves a compact JSON artifact in benchmarks/results/.

Usage:
    python3 scripts/gguf_mtp_bench.py [--model GGUF_PATH] [--cycles N]
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

GGUF_PATH = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DEFAULT_PROMPT = "What is the capital of France?"
DEFAULT_ROOT_TOPK_ACCEPT = 40
DEFAULT_SIBLING_TOPK_ACCEPT = 1
DEFAULT_TOPK_BRANCH_REDRAFT = False
DEFAULT_MTP_DRAFT_WARMUP = True
DEFAULT_TARGET_GRAPH_VERIFY = True
DEFAULT_MTP_DEVICE_KV_CACHE = False
DEFAULT_RESIDENT_MTP_DRAFT = os.environ.get("HIPENGINE_GGUF_RESIDENT_MTP_DRAFT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
IM_START_TOKEN = 248045
IM_END_TOKEN = 248046
THINK_START_TOKEN = 248068
THINK_END_TOKEN = 248069


def build_chat_prompt(tokenizer, user_prompt: str = DEFAULT_PROMPT, *, reasoning: str = "off") -> list[int]:
    """Build the llama.cpp-compatible Qwen chat prompt for GGUF MTP.

    ``reasoning='off'`` is hipEngine's retained default: an empty thinking block
    ending with exactly two newlines after ``</think>``.  ``reasoning='open'`` is
    a parity diagnostic for llama.cpp CLI/server traces that stop the prompt at
    ``<think>\n\n`` and let the model generate the closing reasoning text.
    ``reasoning='none'`` omits the thinking block entirely.
    """
    if reasoning not in {"off", "none", "open"}:
        raise ValueError("build_chat_prompt supports only reasoning='off'/'none'/'open'")
    prompt = (
        [IM_START_TOKEN]
        + tokenizer.encode(f"user\n{user_prompt}")
        + [IM_END_TOKEN]
        + tokenizer.encode("\n")
        + [IM_START_TOKEN]
        + tokenizer.encode("assistant\n")
    )
    if reasoning == "off":
        prompt += [THINK_START_TOKEN] + tokenizer.encode("\n\n") + [THINK_END_TOKEN] + tokenizer.encode("\n\n")
    elif reasoning == "open":
        prompt += [THINK_START_TOKEN] + tokenizer.encode("\n\n")
    return prompt


def select_prompt_tokens(
    tokenizer,
    *,
    user_prompt: str,
    reasoning: str,
    raw_token_ids: tuple[int, ...] = (),
) -> list[int]:
    """Select an exact raw-token context or build the existing chat prompt."""
    if raw_token_ids:
        return [int(token) for token in raw_token_ids]
    return build_chat_prompt(tokenizer, user_prompt, reasoning=reasoning)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _parse_nonnegative_int_csv(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a comma-separated list of non-negative integers") from exc
        if value < 0:
            raise argparse.ArgumentTypeError("cache row indices must be non-negative")
        if value not in seen:
            values.append(value)
            seen.add(value)
    return tuple(values)


def _parse_nonnegative_token_csv(raw: str) -> tuple[int, ...]:
    try:
        return _parse_nonnegative_int_csv(raw)
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError("must be a comma-separated list of non-negative token IDs") from exc


def target_block_direct_commit_is_exact(verify_mode: str, *, start_position: int, rows: int) -> bool:
    if verify_mode in {"native", "serial-exact"}:
        return True
    if verify_mode == "bulk":
        return int(start_position) + int(rows) < 1024
    return False


def target_block_direct_partial_commit_is_exact(verify_mode: str) -> bool:
    return verify_mode in {"native", "serial-exact"}


def target_block_direct_partial_commit_is_llama_style(
    verify_mode: str,
    direct_partial_replay_mode: str,
) -> bool:
    """Return true for llama.cpp-style captured-row partial accepts.

    This is intentionally not an exactness predicate.  llama.cpp MTP accepts the
    verifier hidden row produced by the block pass; it does not restore and
    replay the accepted prefix through a token-serial target path on normal
    partial accepts.  hipEngine keeps the exact/default path separate and uses
    this only for the explicit llama replication lane.
    """

    return verify_mode == "bulk" and direct_partial_replay_mode == "direct-commit"


def target_block_needs_linear_state_snapshot(
    *,
    direct_state_commit: bool,
    direct_state_commit_exact_mode: bool,
    direct_partial_commit_exact_mode: bool = True,
    direct_partial_commit_llama_style: bool = False,
) -> bool:
    direct_partial_commit_supported = bool(direct_partial_commit_exact_mode) or bool(
        direct_partial_commit_llama_style
    )
    return not (
        bool(direct_state_commit)
        and bool(direct_state_commit_exact_mode)
        and direct_partial_commit_supported
    )


def invalidate_target_graph_after_block_verify(target_graph):
    """Close a graph captured before block verification advanced resident state."""

    target_graph.close()
    return (
        None,
        False,
        "target block verifier advanced resident state; fell back to eager target verification",
    )


def target_block_state_replay_uses_serial_exact(
    *,
    replay_state_commit: bool,
    direct_state_commit: bool,
    verify_mode: str,
    direct_partial_replay_mode: str = "serial-exact",
) -> bool:
    """Return whether accepted-prefix replay must use token-serial target state.

    ``--target-block-replay-state-commit`` deliberately scores with the selected
    block verifier first, then restores and advances only the accepted prefix
    through the serial-exact path so scoring semantics do not depend on the
    linear-state capture kernels.
    """

    if bool(replay_state_commit) or verify_mode == "serial-exact":
        return True
    if bool(direct_state_commit):
        return direct_partial_replay_mode not in {"bulk-state-only", "native-state-only", "direct-commit"}
    return False


def _get_hw_info() -> dict:
    """Get GPU hardware info."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["rocminfo"], stderr=subprocess.DEVNULL, text=True, timeout=10)
        for line in out.split("\n"):
            if "Marketing Name" in line:
                gpu = line.split(":", 1)[1].strip()
                return {"gpu": gpu, "arch": "gfx1151"}
    except Exception:
        pass
    return {"gpu": "unknown", "arch": "gfx1151"}


def select_topk_tokens(
    logits_row: "np.ndarray", *, k: int = 10, draft_depth: int = 0
) -> tuple[int, list[int]]:
    """Return the greedy (argmax) token and descending top-k token IDs.

    This is a pure greedy top-k selection: the selected token is always the
    argmax of ``logits_row``, regardless of ``draft_depth``. ``draft_depth`` is
    retained only for call-site symmetry.

    NOTE (anti-gaming): do not add prompt-specific token-id overrides here.
    Hardcoding selections so a fixed prompt's drafts "accept" inflates the
    benchmark acceptance rate without improving the drafter, does not generalize
    to other prompts, and is an INVALID benchmark. The guard test
    ``test_select_topk_tokens_is_pure_argmax_no_prompt_specific_rerank`` fails if
    any such override is reintroduced. See AGENTS.md / docs/BENCHMARK.md.
    """
    if logits_row.ndim != 1:
        raise ValueError("logits_row must be rank-1")
    requested_k = min(max(int(k), 1), int(logits_row.shape[0]))
    top_idx = (
        np.asarray([int(np.argmax(logits_row))], dtype=np.int64)
        if requested_k == 1
        else np.argpartition(logits_row, -requested_k)[-requested_k:]
    )
    top_sorted = (
        top_idx
        if requested_k == 1
        else top_idx[np.argsort(logits_row[top_idx])[::-1]]
    )
    candidate_pool = [int(t) for t in top_sorted]
    return candidate_pool[0], candidate_pool[:requested_k]


def draft_topk_scores_from_logits(logits_row: "np.ndarray", token_ids: list[int]) -> list[float]:
    """Return logits for already-selected top-k token IDs."""

    if logits_row.ndim != 1:
        raise ValueError("logits_row must be rank-1")
    scores: list[float] = []
    for token_id in token_ids:
        token = int(token_id)
        if token < 0 or token >= int(logits_row.shape[0]):
            raise ValueError("top-k token id outside logits row")
        scores.append(float(logits_row[token]))
    return scores


def draft_topk_margins_from_scores(scores: list[float]) -> list[float]:
    """Return top-1 logit minus each top-k logit."""

    if not scores:
        return []
    top = float(scores[0])
    return [float(top - float(score)) for score in scores]


def hidden_state_summary(
    row: "np.ndarray",
    *,
    label: str,
    depth: int | None = None,
    token_id: int | None = None,
    position: int | None = None,
) -> dict[str, object]:
    """Return compact numeric diagnostics for an FP32 hidden row."""

    values = np.ascontiguousarray(np.asarray(row, dtype=np.float32).reshape(-1))
    if values.size == 0:
        raise ValueError("cannot summarize an empty hidden row")
    values64 = values.astype(np.float64, copy=False)
    result: dict[str, object] = {
        "label": str(label),
        "size": int(values.size),
        "sha256_16": hashlib.sha256(values.view(np.uint8)).hexdigest()[:16],
        "mean": round(float(np.mean(values64)), 8),
        "rms": round(float(np.sqrt(np.mean(values64 * values64))), 8),
        "min": round(float(np.min(values64)), 8),
        "max": round(float(np.max(values64)), 8),
        "first8": [round(float(x), 8) for x in values[:8]],
        "last8": [round(float(x), 8) for x in values[-8:]],
    }
    if depth is not None:
        result["depth"] = int(depth)
    if token_id is not None:
        result["token_id"] = int(token_id)
    if position is not None:
        result["position"] = int(position)
    return result


def validate_draft_n_max(draft_n_max: int) -> int:
    """Validate the benchmark's currently implemented diagnostic draft depth.

    ``scripts/gguf_mtp_bench.py`` is a correctness/acceptance diagnostic, not the
    prompt-suite performance runner. Its local chained draft loop can exercise
    B1 through B5 now; artifacts still label the MTP context mode so these rows
    are not confused with the future persistent GGUF MTP context implementation.
    """
    draft_n_max = int(draft_n_max)
    if not 1 <= draft_n_max <= 8:
        raise ValueError("draft_n_max must be in 1..8 for GGUF MTP diagnostics")
    return draft_n_max


def compute_speculative_metrics(cycles: list[dict]) -> dict:
    """Compute MTP metrics with explicit llama.cpp-compatible denominators.

    A verify cycle always emits one target/corrective token. Accepted draft
    tokens are also visible output tokens, so ``accepted_per_output`` uses
    ``accepted_draft_tokens / visible_output_token_count`` rather than dividing
    by verify-cycle count. This follows docs/MTP-gguf.md's denominator contract.
    """
    if not cycles:
        raise ValueError("cycles must be non-empty")

    def require_int(cycle: dict, index: int, field: str, *, positive: bool = False) -> int:
        if field not in cycle:
            raise ValueError(f"cycles[{index}].{field} is required")
        value = cycle[field]
        if type(value) is not int:
            raise ValueError(f"cycles[{index}].{field} must be an integer")
        if positive and value <= 0:
            raise ValueError(f"cycles[{index}].{field} must be positive")
        if not positive and value < 0:
            raise ValueError(f"cycles[{index}].{field} must be non-negative")
        return value

    def require_timing(cycle: dict, index: int, field: str) -> float:
        if field not in cycle:
            raise ValueError(f"cycles[{index}].{field} is required")
        value = cycle[field]
        if type(value) not in (int, float):
            raise ValueError(f"cycles[{index}].{field} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"cycles[{index}].{field} must be finite")
        if result < 0.0:
            raise ValueError(f"cycles[{index}].{field} must be non-negative")
        return result

    def optional_int(cycle: dict, index: int, field: str) -> int:
        if field not in cycle:
            return 0
        value = cycle[field]
        if type(value) is not int:
            raise ValueError(f"cycles[{index}].{field} must be an integer")
        if value < 0:
            raise ValueError(f"cycles[{index}].{field} must be non-negative")
        return value

    def optional_timing(cycle: dict, index: int, field: str) -> float | None:
        if field not in cycle:
            return None
        return require_timing(cycle, index, field)

    def optional_stage_timings(cycle: dict, index: int) -> dict[str, float]:
        raw = cycle.get("stage_timings_ms")
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"cycles[{index}].stage_timings_ms must be an object")
        out: dict[str, float] = {}
        for name, value in raw.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"cycles[{index}].stage_timings_ms keys must be non-empty strings")
            if type(value) not in (int, float):
                raise ValueError(f"cycles[{index}].stage_timings_ms.{name} must be numeric")
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"cycles[{index}].stage_timings_ms.{name} must be finite")
            if result < 0.0:
                raise ValueError(f"cycles[{index}].stage_timings_ms.{name} must be non-negative")
            out[name] = result
        return out

    def bump_hist(hist: dict[str, int], value: int) -> None:
        key = str(int(value))
        hist[key] = hist.get(key, 0) + 1

    verify_cycle_count = len(cycles)
    total_drafts = 0
    total_accepted = 0
    visible_output_tokens = 0
    total_ar_ms = 0.0
    total_draft_ms = 0.0
    cycle_wall_ms_total = 0.0
    cycle_wall_count = 0
    stage_timing_totals_ms: dict[str, float] = {}
    target_verify_layer_passes = 0
    target_verify_rows_evaluated = 0
    target_verify_serial_rows = 0
    target_verify_graph_rows = 0
    target_verify_block_passes = 0
    target_verify_block_rows = 0
    target_verify_replay_rows = 0
    target_verify_direct_commit_rows = 0
    target_verify_discarded_rows = 0
    cycle_histograms: dict[str, dict[str, int]] = {
        "generated_draft_tokens": {},
        "accepted_draft_tokens": {},
        "visible_output_tokens": {},
        "target_verify_layer_passes": {},
        "target_verify_rows_evaluated": {},
        "target_verify_block_rows": {},
        "target_verify_replay_rows": {},
        "target_verify_direct_commit_rows": {},
        "target_verify_discarded_rows": {},
        "target_verify_rows_minus_visible_output": {},
    }
    for index, cycle in enumerate(cycles):
        if not isinstance(cycle, dict):
            raise ValueError(f"cycles[{index}] must be an object")
        generated_drafts = require_int(cycle, index, "generated_draft_tokens")
        accepted_drafts = require_int(cycle, index, "accepted_draft_tokens")
        visible_output = require_int(cycle, index, "visible_output_tokens", positive=True)
        if accepted_drafts > generated_drafts:
            raise ValueError(f"cycles[{index}].accepted_draft_tokens must be <= generated_draft_tokens")
        if accepted_drafts > visible_output:
            raise ValueError(f"cycles[{index}].accepted_draft_tokens must be <= visible_output_tokens")
        bump_hist(cycle_histograms["generated_draft_tokens"], generated_drafts)
        bump_hist(cycle_histograms["accepted_draft_tokens"], accepted_drafts)
        bump_hist(cycle_histograms["visible_output_tokens"], visible_output)
        total_drafts += generated_drafts
        total_accepted += accepted_drafts
        visible_output_tokens += visible_output
        total_ar_ms += require_timing(cycle, index, "ar_decode_ms")
        total_draft_ms += require_timing(cycle, index, "mtp_draft_ms")
        cycle_wall_ms = optional_timing(cycle, index, "cycle_wall_ms")
        if cycle_wall_ms is not None:
            cycle_wall_ms_total += cycle_wall_ms
            cycle_wall_count += 1
        for name, stage_ms in optional_stage_timings(cycle, index).items():
            stage_timing_totals_ms[name] = stage_timing_totals_ms.get(name, 0.0) + stage_ms
        cycle_target_verify_layer_passes = optional_int(cycle, index, "target_verify_layer_passes")
        cycle_target_verify_rows_evaluated = optional_int(cycle, index, "target_verify_rows_evaluated")
        cycle_target_verify_serial_rows = optional_int(cycle, index, "target_verify_serial_rows")
        cycle_target_verify_graph_rows = optional_int(cycle, index, "target_verify_graph_rows")
        cycle_target_verify_block_passes = optional_int(cycle, index, "target_verify_block_passes")
        cycle_target_verify_block_rows = optional_int(cycle, index, "target_verify_block_rows")
        cycle_target_verify_replay_rows = optional_int(cycle, index, "target_verify_replay_rows")
        cycle_target_verify_direct_commit_rows = optional_int(cycle, index, "target_verify_direct_commit_rows")
        cycle_target_verify_discarded_rows = optional_int(cycle, index, "target_verify_discarded_rows")
        target_verify_layer_passes += cycle_target_verify_layer_passes
        target_verify_rows_evaluated += cycle_target_verify_rows_evaluated
        target_verify_serial_rows += cycle_target_verify_serial_rows
        target_verify_graph_rows += cycle_target_verify_graph_rows
        target_verify_block_passes += cycle_target_verify_block_passes
        target_verify_block_rows += cycle_target_verify_block_rows
        target_verify_replay_rows += cycle_target_verify_replay_rows
        target_verify_direct_commit_rows += cycle_target_verify_direct_commit_rows
        target_verify_discarded_rows += cycle_target_verify_discarded_rows
        bump_hist(cycle_histograms["target_verify_layer_passes"], cycle_target_verify_layer_passes)
        bump_hist(cycle_histograms["target_verify_rows_evaluated"], cycle_target_verify_rows_evaluated)
        bump_hist(cycle_histograms["target_verify_block_rows"], cycle_target_verify_block_rows)
        bump_hist(cycle_histograms["target_verify_replay_rows"], cycle_target_verify_replay_rows)
        bump_hist(cycle_histograms["target_verify_direct_commit_rows"], cycle_target_verify_direct_commit_rows)
        bump_hist(cycle_histograms["target_verify_discarded_rows"], cycle_target_verify_discarded_rows)
        if "target_verify_rows_evaluated" in cycle:
            bump_hist(
                cycle_histograms["target_verify_rows_minus_visible_output"],
                cycle_target_verify_rows_evaluated - visible_output,
            )
    total_cycle_ms = total_ar_ms + total_draft_ms

    accept_per_draft = total_accepted / total_drafts if total_drafts > 0 else 0.0
    accepted_per_output = total_accepted / visible_output_tokens if visible_output_tokens > 0 else 0.0
    visible_tokens_per_cycle = visible_output_tokens / verify_cycle_count if verify_cycle_count > 0 else 0.0
    avg_cycle_ms = total_cycle_ms / verify_cycle_count if verify_cycle_count > 0 else 0.0
    avg_ar_decode_ms = total_ar_ms / verify_cycle_count if verify_cycle_count > 0 else 0.0
    avg_mtp_draft_ms = total_draft_ms / verify_cycle_count if verify_cycle_count > 0 else 0.0
    avg_ms_per_visible_token = total_cycle_ms / visible_output_tokens if visible_output_tokens > 0 else 0.0
    tokens_per_sec = 1000.0 / avg_ms_per_visible_token if avg_ms_per_visible_token > 0 else 0.0
    ar_baseline_tokens_per_sec = (
        1000.0 * visible_output_tokens / total_ar_ms if total_ar_ms > 0 and visible_output_tokens > 0 else 0.0
    )
    speedup_vs_ar_visible = (
        tokens_per_sec / ar_baseline_tokens_per_sec if ar_baseline_tokens_per_sec > 0 else 0.0
    )
    target_verify_layer_passes_per_output = (
        target_verify_layer_passes / visible_output_tokens if visible_output_tokens > 0 else 0.0
    )
    target_verify_rows_per_output = (
        target_verify_rows_evaluated / visible_output_tokens if visible_output_tokens > 0 else 0.0
    )
    target_verify_replay_rows_per_output = (
        target_verify_replay_rows / visible_output_tokens if visible_output_tokens > 0 else 0.0
    )

    result = {
        "accept_per_draft": accept_per_draft,
        "accepted_per_output": accepted_per_output,
        "avg_cycle_ms": avg_cycle_ms,
        "avg_decode_ms": avg_cycle_ms,  # backward-compatible alias for older artifacts
        "avg_ar_decode_ms": avg_ar_decode_ms,
        "avg_mtp_draft_ms": avg_mtp_draft_ms,
        "avg_ms_per_visible_token": avg_ms_per_visible_token,
        "tokens_per_sec": tokens_per_sec,
        "ar_baseline_tokens_per_sec": ar_baseline_tokens_per_sec,
        "speedup_vs_ar_visible": speedup_vs_ar_visible,
        "visible_tokens_per_cycle": visible_tokens_per_cycle,
        "total_accepted": total_accepted,
        "total_drafts": total_drafts,
        "total_output_tokens": visible_output_tokens,
        "verify_cycle_count": verify_cycle_count,
        "total_cycle_ms": total_cycle_ms,
        "target_verify_layer_passes": target_verify_layer_passes,
        "target_verify_rows_evaluated": target_verify_rows_evaluated,
        "target_verify_serial_rows": target_verify_serial_rows,
        "target_verify_graph_rows": target_verify_graph_rows,
        "target_verify_block_passes": target_verify_block_passes,
        "target_verify_block_rows": target_verify_block_rows,
        "target_verify_replay_rows": target_verify_replay_rows,
        "target_verify_direct_commit_rows": target_verify_direct_commit_rows,
        "target_verify_discarded_rows": target_verify_discarded_rows,
        "target_verify_layer_passes_per_output": target_verify_layer_passes_per_output,
        "target_verify_rows_per_output": target_verify_rows_per_output,
        "target_verify_replay_rows_per_output": target_verify_replay_rows_per_output,
        "cycle_histograms": cycle_histograms,
        "denominators": {
            "accept_per_draft": "accepted_draft_tokens / generated_draft_tokens",
            "accepted_per_output": "accepted_draft_tokens / visible_output_token_count",
            "visible_tokens_per_cycle": "visible_output_token_count / verify_cycle_count",
            "tokens_per_sec": "visible_output_token_count / total_cycle_wall_time",
            "target_verify_layer_passes_per_output": "target layer-streaming passes / visible_output_token_count",
            "target_verify_rows_per_output": "target verifier rows evaluated / visible_output_token_count",
            "target_verify_replay_rows_per_output": "accepted-prefix replay rows / visible_output_token_count",
        },
    }
    if cycle_wall_count > 0:
        cycle_wall_over_legacy_ms_total = (
            cycle_wall_ms_total - total_cycle_ms if cycle_wall_count == verify_cycle_count else None
        )
        result.update(
            {
                "cycle_wall_ms_total": cycle_wall_ms_total,
                "cycle_wall_ms_count": cycle_wall_count,
                "cycle_wall_ms_per_output": (
                    cycle_wall_ms_total / visible_output_tokens if visible_output_tokens > 0 else 0.0
                ),
                "cycle_wall_over_legacy_ms_total": cycle_wall_over_legacy_ms_total,
                "cycle_wall_over_legacy_ms_per_output": (
                    cycle_wall_over_legacy_ms_total / visible_output_tokens
                    if cycle_wall_over_legacy_ms_total is not None and visible_output_tokens > 0
                    else None
                ),
            }
        )
    if stage_timing_totals_ms:
        result["stage_timing_totals_ms"] = dict(sorted(stage_timing_totals_ms.items()))
        result["stage_timing_per_cycle_ms"] = {
            name: ms / verify_cycle_count for name, ms in sorted(stage_timing_totals_ms.items())
        }
        result["stage_timing_per_output_ms"] = {
            name: ms / visible_output_tokens
            for name, ms in sorted(stage_timing_totals_ms.items())
            if visible_output_tokens > 0
        }
    return result


def llama_cpp_mtp_catchup_rows(
    prompt_tokens: list[int] | tuple[int, ...],
    prompt_hidden_seeds: "np.ndarray",
) -> tuple[list[int], "np.ndarray"]:
    """Build llama.cpp-style initial MTP catch-up rows for a prompt.

    The MTP context mirrors target prompt tokens with the hidden row shifted one
    position to the right: row 0 uses an all-zero pending hidden row, and row i
    uses the target post-output_norm hidden from prompt token i-1.
    """
    tokens = [int(token) for token in prompt_tokens]
    hidden = np.ascontiguousarray(prompt_hidden_seeds, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError("prompt_hidden_seeds must have shape [prompt_tokens, hidden_size]")
    if len(tokens) != int(hidden.shape[0]):
        raise ValueError("prompt_tokens and prompt_hidden_seeds must have the same length")
    if not tokens:
        raise ValueError("prompt_tokens must be non-empty")
    zero = np.zeros((1, hidden.shape[1]), dtype=np.float32)
    shifted = zero if hidden.shape[0] == 1 else np.concatenate([zero, hidden[:-1]], axis=0)
    return tokens, np.ascontiguousarray(shifted, dtype=np.float32)


def llama_cpp_acceptance_from_target_samples(
    draft_tokens: list[int],
    target_samples: list[int],
) -> dict[str, object]:
    """Summarize llama.cpp draft-MTP accept/commit semantics.

    ``target_samples`` are the target-model greedy samples for rows
    ``[sampled_token] + accepted_draft_prefix``.  The list must include the
    corrective target row after the accepted prefix, so a fully accepted B2 draft
    has three target samples: draft0, draft1, corrective.
    """
    if not draft_tokens:
        raise ValueError("draft_tokens must be non-empty")
    if not target_samples:
        raise ValueError("target_samples must be non-empty")

    drafts = [int(token) for token in draft_tokens]
    targets = [int(token) for token in target_samples]
    accepted = 0
    for draft_token, target_token in zip(drafts, targets, strict=False):
        if draft_token != target_token:
            break
        accepted += 1
        if accepted == len(drafts):
            break

    if len(targets) <= accepted:
        raise ValueError(
            "target_samples must include the corrective target token after the accepted prefix"
        )

    output_tokens = targets[:accepted] + [targets[accepted]]
    comparison_target_tokens = targets[: min(len(drafts), len(targets))]
    return {
        "accepted_draft_tokens": accepted,
        "visible_output_tokens": len(output_tokens),
        "output_tokens": output_tokens,
        "comparison_target_tokens": comparison_target_tokens,
        "pending_hidden_row_index": accepted,
    }


def sibling_topk_acceptance_from_target_samples(
    draft_tokens: list[int],
    draft_topk_tokens: list[list[int]],
    target_samples: list[int],
    *,
    root_topk_accept: int,
    sibling_topk_accept: int,
) -> dict[str, object] | None:
    """Accept the first non-argmax sibling selected by exact target verification.

    For every generated linear draft row, the draft logits define a top-k sibling
    set.  The target may select a non-argmax sibling at the first mismatch; after
    that one branch accept, the target emits one corrective token because no
    branch-specific child rows were generated for that sibling.
    """
    if not draft_tokens or not draft_topk_tokens or not target_samples:
        return None
    drafts = [int(token) for token in draft_tokens]
    targets = [int(token) for token in target_samples]
    accepted_prefix = 0
    for depth, draft_token in enumerate(drafts):
        if depth >= len(targets):
            return None
        if int(targets[depth]) == draft_token:
            accepted_prefix += 1
            continue
        limit = root_topk_accept if depth == 0 else sibling_topk_accept
        if limit <= 1 or depth >= len(draft_topk_tokens):
            return None
        if int(targets[depth]) not in [int(token) for token in draft_topk_tokens[depth][:limit]]:
            return None
        accepted = depth + 1
        if len(targets) <= accepted:
            raise ValueError("target_samples must include the corrective target token after sibling top-k acceptance")
        output_tokens = targets[:accepted] + [targets[accepted]]
        return {
            "accepted_draft_tokens": accepted,
            "visible_output_tokens": len(output_tokens),
            "output_tokens": output_tokens,
            "comparison_target_tokens": output_tokens,
            "pending_hidden_row_index": accepted,
            "topk_branch_depth": depth,
        }
    return None


def root_topk_acceptance_from_target_samples(
    draft_tokens: list[int],
    draft_topk_tokens: list[list[int]],
    target_samples: list[int],
    *,
    root_topk_accept: int,
) -> dict[str, object] | None:
    """Accept a depth-0 branch when the target token is in the root top-k set."""
    acceptance = sibling_topk_acceptance_from_target_samples(
        draft_tokens,
        draft_topk_tokens,
        target_samples,
        root_topk_accept=root_topk_accept,
        sibling_topk_accept=1,
    )
    if acceptance is not None:
        acceptance.pop("topk_branch_depth", None)
    return acceptance


def count_topk_draft_candidates(
    linear_draft_count: int,
    *,
    root_topk_accept: int,
    sibling_topk_accept: int,
    sibling_topk_max_depth: int,
) -> int:
    """Count exposed draft candidates for linear+top-k sibling proposal accounting."""

    if linear_draft_count <= 0:
        return 0
    count = int(linear_draft_count)
    if root_topk_accept > 1:
        count += int(root_topk_accept) - 1
    if linear_draft_count > 1 and sibling_topk_accept > 1:
        sibling_rows = min(int(linear_draft_count) - 1, int(sibling_topk_max_depth))
        count += max(0, sibling_rows) * (int(sibling_topk_accept) - 1)
    return count


def target_membership_in_draft_topk(
    comparison_target_tokens: list[int],
    draft_topk_tokens: list[list[int]],
) -> tuple[list[bool], list[int | None]]:
    """Report whether each compared target token appears in available draft top-k rows."""

    target_in_draft_topk: list[bool] = []
    target_rank_in_draft_topk: list[int | None] = []
    for depth, target in enumerate(comparison_target_tokens):
        if depth >= len(draft_topk_tokens):
            target_in_draft_topk.append(False)
            target_rank_in_draft_topk.append(None)
            continue
        topk = draft_topk_tokens[depth]
        if target in topk:
            target_in_draft_topk.append(True)
            target_rank_in_draft_topk.append(topk.index(target) + 1)
        else:
            target_in_draft_topk.append(False)
            target_rank_in_draft_topk.append(None)
    return target_in_draft_topk, target_rank_in_draft_topk


def _draft_top1_prob(logits_row: np.ndarray) -> float:
    """Compute the softmax probability of the argmax token."""
    shifted = logits_row - logits_row.max()
    exp = np.exp(shifted)
    return float(exp.max() / exp.sum())


def _score_top_k_rows(logits_row: np.ndarray, *, top_k: int) -> list[dict[str, float | int]]:
    row = np.asarray(logits_row, dtype=np.float32).reshape(-1)
    if row.size == 0:
        return []
    k = min(max(1, int(top_k)), int(row.size))
    indices = np.argpartition(-row, k - 1)[:k]
    order = np.lexsort((indices, -row[indices]))
    sorted_indices = indices[order]
    top_score = float(row[int(sorted_indices[0])])
    return [
        {
            "rank": int(rank),
            "token": int(token),
            "score": float(row[int(token)]),
            "delta_vs_top1": float(row[int(token)] - top_score),
        }
        for rank, token in enumerate(sorted_indices.tolist(), start=1)
    ]


def _score_candidate_rows(
    logits_row: np.ndarray,
    candidate_tokens: list[int] | tuple[int, ...],
) -> list[dict[str, float | int | None]]:
    row = np.asarray(logits_row, dtype=np.float32).reshape(-1)
    if row.size == 0:
        return []
    top_token = int(np.argmax(row))
    top_score = float(row[top_token])
    records: list[dict[str, float | int | None]] = []
    seen: set[int] = set()
    for token in candidate_tokens:
        token = int(token)
        if token in seen:
            continue
        seen.add(token)
        if token < 0 or token >= int(row.size):
            records.append({"token": token, "score": None, "delta_vs_top1": None})
            continue
        records.append(
            {
                "token": token,
                "score": float(row[token]),
                "delta_vs_top1": float(row[token] - top_score),
            }
        )
    return records


def _target_lm_head_score_rows(
    logits: np.ndarray | None,
    *,
    input_tokens: list[int],
    target_tokens: list[int],
    draft_tokens: list[int],
    extra_candidate_tokens: tuple[int, ...] = (),
    top_k: int = 10,
) -> list[dict[str, object]]:
    if logits is None:
        return []
    logits_arr = np.asarray(logits, dtype=np.float32)
    if logits_arr.ndim != 2:
        raise ValueError("target lm-head logits must be rank-2")
    rows = min(int(logits_arr.shape[0]), len(input_tokens), len(target_tokens))
    records: list[dict[str, object]] = []
    for row_index in range(rows):
        candidate_tokens: list[int] = []
        candidate_tokens.extend(int(token) for token in draft_tokens)
        candidate_tokens.extend(int(token) for token in target_tokens)
        candidate_tokens.extend(int(token) for token in extra_candidate_tokens)
        row = logits_arr[row_index]
        records.append(
            {
                "row": int(row_index),
                "position_offset": int(row_index),
                "input_token": int(input_tokens[row_index]),
                "target_token": int(target_tokens[row_index]),
                "top_k": _score_top_k_rows(row, top_k=top_k),
                "candidate_scores": _score_candidate_rows(row, candidate_tokens),
            }
        )
    return records


def _target_hidden_seed_summary_rows(
    hidden_rows: list[np.ndarray],
    score_rows: list[dict[str, object]],
    *,
    cycle_start_seq_position: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for score_row in score_rows:
        row_index = int(score_row.get("row", -1))
        record: dict[str, object] = {
            "row": row_index,
            "input_token": int(score_row["input_token"]) if "input_token" in score_row else None,
            "target_token": int(score_row["target_token"]) if "target_token" in score_row else None,
            "hidden_available": bool(0 <= row_index < len(hidden_rows)),
        }
        if 0 <= row_index < len(hidden_rows):
            record["summary"] = hidden_state_summary(
                hidden_rows[row_index],
                label="target_hidden_seed",
                depth=row_index,
                token_id=int(score_row["target_token"]) if "target_token" in score_row else None,
                position=int(cycle_start_seq_position + row_index),
            )
        records.append(record)
    return records


def _rope_tables(*, max_positions: int, rotary_dim: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute split-half RoPE cos/sin tables (mirrors qwen35_gguf_runner._rope_tables)."""
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(rotary_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(base), -2.0 * dims / np.float32(rotary_dim))
    freqs = positions * inv_freq
    cos_half = np.cos(freqs).astype(np.float32, copy=False)
    sin_half = np.sin(freqs).astype(np.float32, copy=False)
    cos = np.concatenate([cos_half, cos_half], axis=1).astype(np.float32, copy=False)
    sin = np.concatenate([sin_half, sin_half], axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MTP speculative decoding benchmark")
    parser.add_argument("--model", default=GGUF_PATH, help="GGUF model path")
    parser.add_argument("--decode-repack", action=argparse.BooleanOptionalAction, default=True, help="Use the production resident T16 decode-repack GGUF path (default: true)")
    parser.add_argument("--use-wmma-prefill", action=argparse.BooleanOptionalAction, default=True, help="Request WMMA prefill for the resident GGUF session (default: true)")
    parser.add_argument("--use-gemv-decode", action=argparse.BooleanOptionalAction, default=True, help="Request GEMV decode for the resident GGUF session (default: true)")
    parser.add_argument(
        "--mtp-draft-warmup",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MTP_DRAFT_WARMUP,
        help=(
            "Run one stateless untimed MTP draft after prefill to warm kernel/library/weight caches "
            "before measured speculative cycles (default: true; use --no-mtp-draft-warmup for cold-start diagnostics)."
        ),
    )
    parser.add_argument(
        "--target-graph-verify",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TARGET_GRAPH_VERIFY,
        help=(
            "Use resident GGUF decode graph replay for target verification with a replay-window context cap "
            "and fp32 hidden-seed capture (default: true; use --no-target-graph-verify for eager-step diagnostics)."
        ),
    )
    parser.add_argument(
        "--target-graph-batched-verify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic only: replay one full strict verifier block per cycle and record generated IDs plus "
            "FP32 hidden seeds. Requires top-1 strict acceptance; aborts if the whole draft prefix is not accepted."
        ),
    )
    parser.add_argument(
        "--target-block-verify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic: verify strict top-1 draft chains with the GGUF target row-bulk continuation path. "
            "Snapshots linear recurrent state and rolls back/replays the consumed prefix on partial accepts."
        ),
    )
    parser.add_argument(
        "--native-spec-target-cycle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic N1 route: submit admitted B2/three-row target blocks through the "
            "versioned NativeSpecCycle C++ launcher. Unsupported rows/configurations retain "
            "the exact Python target-block fallback. Capture remains state-generation-bound."
        ),
    )
    parser.add_argument(
        "--native-spec-device-accept-commit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "N2 extension: keep strict-chain acceptance, selected recurrent/hidden commit, "
            "target KV cursor transaction, and bounded visible output summary inside the "
            "reusable native graph. Requires the device-seed llama-compat route."
        ),
    )
    parser.add_argument(
        "--native-spec-complete-cycle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "N3 extension: route strict device-chained proposal, N2 target accept/commit, "
            "MTP-KV rollback/repair, reseed, and cursor/result accounting through one GGUF "
            "provider-adapter call. Requires the explicit llama-compat N2 route."
        ),
    )
    parser.add_argument(
        "--native-spec-proposal-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "N3P extension: replay strict B1/B2 NextN proposal through one reusable native "
            "graph submission before the unchanged N2 target transaction."
        ),
    )
    parser.add_argument(
        "--target-block-verify-mode",
        choices=("bulk", "native", "serial-exact"),
        default="bulk",
        help="Attention scheduler for --target-block-verify (default: bulk; serial-exact is a slow correctness baseline).",
    )
    parser.add_argument(
        "--target-block-min-rows",
        type=int,
        default=0,
        help=(
            "Minimum block rows (prev + drafts) required to use --target-block-verify. "
            "0 (default) means use the GGUF ssm_conv_kernel (=4), the historical gate. "
            "Set 2 to allow B1/B2 block verify: verify_target_block is bit-exact vs "
            "serial-exact at rows 2-3 (probe p02a) and direct commit is exact for "
            "start_position+rows<1024, so smaller blocks attempt fewer rows per cycle."
        ),
    )
    parser.add_argument(
        "--target-block-direct-state-commit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic: with strict --target-block-verify, capture verifier row states and commit the "
            "accepted row directly instead of restoring and replaying the accepted prefix."
        ),
    )
    parser.add_argument(
        "--target-block-replay-state-commit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic: with strict --target-block-verify, keep the first verifier pass on the "
            "non-capturing block path, then restore and replay the accepted prefix through "
            "serial-exact state after every block. This preserves non-capturing block token "
            "semantics when linear-state capture would select a different verifier kernel path."
        ),
    )
    parser.add_argument(
        "--target-block-direct-partial-replay-mode",
        choices=("serial-exact", "serial-state-only", "bulk-state-only", "native-state-only", "direct-commit"),
        default="serial-exact",
        help=(
            "Diagnostic: when direct-state block verification rejects or partially accepts a block, "
            "restore the snapshot and replay the accepted prefix with either the serial-exact target "
            "path (default, semantic baseline), the serial-exact path with LM-head sampling skipped, "
            "the bulk verifier in advance_state_only mode, or the native row-serial-attention verifier "
            "in advance_state_only mode. direct-commit skips replay and commits the captured verifier "
            "row, matching llama.cpp's normal MTP accept lifecycle. State-only modes never use replayed "
            "token IDs for scoring."
        ),
    )
    parser.add_argument(
        "--target-b1-branch-safe-block-verify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic: for B1/root-top-k routes, verify [prev, draft0] in one target block. "
            "Use row 1 only when row 0 strictly accepts draft0; otherwise restore/replay row 0 "
            "and fall back to a serial corrective step for accepted root branches."
        ),
    )
    parser.add_argument(
        "--target-block-wmma-prefill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use WMMA/selected-prefill kernels inside --target-block-verify. Default false because B3/B5 "
            "small blocks benchmark faster on the GEMV prefill fallback."
        ),
    )
    parser.add_argument(
        "--target-block-sync-stage-timings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic only: when cycle stage timings are enabled, insert synchronization points "
            "inside the target block verifier to attribute linear/full attention and MoE sections. "
            "This changes timing and is not a performance route."
        ),
    )
    parser.add_argument(
        "--verify-dp4a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "OPT-IN, DEFAULT OFF, ACCURACY-DEGRADING: enable llama.cpp-style dp4a (q8_1) "
            "selected-expert verify GEMVs (sets the HIPENGINE_GGUF_*_SELECTED_DP4A flags). "
            "Trades correctness for speed: FAILS the ja correctness gate (greedy top-1 0.700 "
            "< 0.90 vs cpu_reference; code 1.000). For users who want max accuracy-traded MTP "
            "perf. Best measured: ~61.6 tok/s / 1.132x AR (B5) - still below llama HIP 67.3."
        ),
    )
    parser.add_argument(
        "--verify-dense-q8-dp4a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "OPT-IN diagnostic: retain raw Q8_0 sidecars for linear-attention attn_qkv/attn_gate "
            "T16 weights and route rows>1 verifier blocks through q8_1/dp4a dense Q8 GEMVs. "
            "Intended for llama-compat parity A/B only; default off."
        ),
    )
    parser.add_argument(
        "--verify-dense-q8-dp4a-all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "OPT-IN diagnostic: retain raw Q8_0 sidecars and route all supported rows>1 dense "
            "Q8 verifier projections through q8_1/dp4a rowtile wrappers. Includes the pair route "
            "from --verify-dense-q8-dp4a plus singleton and Q/K/V triple projections. Default off."
        ),
    )
    parser.add_argument(
        "--verify-dense-q8-dp4a-shared",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "OPT-IN diagnostic: when raw Q8 sidecars are materialized, also route rows>1 "
            "shared-expert Q8_0 gate/up/down verifier projections through q8_1/dp4a rowtile "
            "wrappers. Intended for llama-compat parity A/B only; default off."
        ),
    )
    parser.add_argument(
        "--verify-dense-q8-dp4a-f32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "OPT-IN diagnostic: when raw Q8 sidecars are materialized, route rows>1 "
            "F32-activation singleton verifier projections through f32->q8_1 plus raw-Q8 dp4a. "
            "Intended for direct-state ssm_out parity A/B only; default off."
        ),
    )
    parser.add_argument(
        "--verify-lm-head-q6-top1-dp4a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "OPT-IN accuracy-traded llama-compat diagnostic: materialize an X8 Q6_K "
            "lm-head sidecar and sample verifier block rows through q8_1/dp4a top-1 "
            "directly, skipping full verifier logits plus argmax. Default off."
        ),
    )
    parser.add_argument(
        "--selected-down-x8-repack",
        choices=("off", "q5", "q6", "both"),
        default="off",
        help=(
            "OPT-IN accuracy-traded llama-compat diagnostic: materialize selected-down "
            "Q5_K/Q6_K experts with the X8 q8_1/dp4a replacement layout. Current retained "
            "compat lane uses q6 only; q5/both remain diagnostic/rejected."
        ),
    )
    parser.add_argument(
        "--selected-gate-up-x8",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "OPT-IN accuracy-traded llama-compat diagnostic: materialize selected gate/up "
            "Q4_K experts with the X8 q8_1/dp4a replacement layout. Default off."
        ),
    )
    parser.add_argument(
        "--selected-gate-up-raw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "OPT-IN llama-compat diagnostic: keep selected gate/up Q4_K experts in raw "
            "GGUF layout so --verify-dp4a routes them through the raw selected-dual "
            "q8_1/dp4a body, closer to llama.cpp mul_mat_vec_q_moe. Default off."
        ),
    )
    parser.add_argument(
        "--llama-compat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "OPT-IN diagnostic: force the GGUF MTP route toward llama.cpp semantics "
            "(B2, p_min=0, full draft vocab, no B1 probe/fallback, shifted MTP context "
            "replay + device KV, one target block verify per cycle). Use with "
            "--verify-dp4a for the llama.cpp accuracy-traded speed regime. Default off; "
            "the shipped exact route is unchanged."
        ),
    )
    parser.add_argument("--cycles", type=int, default=10, help="Number of speculate-verify cycles")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=0,
        help=(
            "Optional llama.cpp server parity cap. When >0, stop once visible output "
            "tokens reach this request budget and clamp each speculative draft window "
            "to leave one corrective target token, matching llama.cpp's n_remaining - 1 "
            "draft limit. The default 0 keeps the historical fixed-cycle bench."
        ),
    )
    parser.add_argument("--draft-n-max", type=int, default=1, help="Max draft tokens per cycle")
    parser.add_argument(
        "--adaptive-draft-window",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reduce the measured draft window after a partial accept so later cycles avoid "
            "known-bad deeper draft depths."
        ),
    )
    parser.add_argument(
        "--adaptive-ar-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After a low-accept cycle, stop drafting for the rest of this prompt and run "
            "plain target AR cycles through the graph verifier."
        ),
    )
    parser.add_argument(
        "--adaptive-ar-fallback-max-accepted",
        type=int,
        default=0,
        help="Trigger --adaptive-ar-fallback when accepted draft tokens are <= this value.",
    )
    parser.add_argument(
        "--adaptive-ar-fallback-cooldown",
        type=int,
        default=0,
        help=(
            "Make --adaptive-ar-fallback recoverable. 0 (default) = permanent latch "
            "(historical behavior): one low-accept cycle disables drafting for the rest "
            "of the prompt. N>0 = after N AR-only cycles, re-probe one drafting cycle; "
            "re-arm for N more if it misses again. Keeps drafting on categories with good "
            "acceptance that a single hard-token miss would otherwise kill."
        ),
    )
    parser.add_argument(
        "--adaptive-block-after-full-accept",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Production verifier selector: start with serial graph verification, then enable "
            "--target-block-verify only after a cycle fully accepts its proposed draft chain."
        ),
    )
    parser.add_argument(
        "--fused-b1-block-probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic: when --adaptive-block-after-full-accept is probing with B1, verify "
            "[prev, draft0] with one strict target block instead of falling back to the "
            "serial target step loop. Requires strict top-1, --target-block-min-rows 2, "
            "and direct/replay correctness from --target-block-verify."
        ),
    )
    parser.add_argument(
        "--adaptive-probe-draft-n-max",
        type=int,
        default=3,
        help=(
            "Draft window used before --adaptive-block-after-full-accept promotes to the "
            "configured --draft-n-max block verifier window."
        ),
    )
    parser.add_argument(
        "--mtp-draft-vocab-cap",
        type=int,
        default=0,
        help=(
            "Diagnostic only: limit MTP draft lm-head/argmax to the first N token IDs "
            "(0 = full vocabulary). Must be full-suite validated before becoming a retained default."
        ),
    )
    parser.add_argument(
        "--adaptive-full-vocab-after-cap-miss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic only: when --mtp-draft-vocab-cap produces a low-accept miss, try the next "
            "cycle with a full-vocabulary resident draft runner instead of immediately staying in "
            "plain AR fallback."
        ),
    )
    parser.add_argument(
        "--adaptive-strict-block-probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic hybrid policy: begin with strict top-1 block-promotion probing; if the "
            "first probe cycles do not meet the accepted-token threshold, switch to the generic "
            "root-top-k B1 serial fallback for the rest of the prompt."
        ),
    )
    parser.add_argument(
        "--adaptive-strict-probe-cycles",
        type=int,
        default=2,
        help="Number of initial cycles used by --adaptive-strict-block-probe (default: 2).",
    )
    parser.add_argument(
        "--adaptive-strict-probe-min-accepted",
        type=int,
        default=2,
        help=(
            "Minimum accepted draft tokens required in every strict probe cycle to keep the "
            "strict block policy (default: 2)."
        ),
    )
    parser.add_argument(
        "--adaptive-strict-fallback-draft-n-max",
        type=int,
        default=1,
        help="Draft window after a failed strict probe (default: 1 = B1 fallback).",
    )
    parser.add_argument(
        "--adaptive-strict-fallback-root-topk",
        type=int,
        default=40,
        help="Root top-k acceptance after a failed strict probe (default: 40).",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt text before the assistant turn")
    parser.add_argument(
        "--prompt-token-ids",
        type=_parse_nonnegative_token_csv,
        default=(),
        metavar="ID,ID,...",
        help=(
            "Use these exact raw token IDs as the complete prompt context, bypassing "
            "chat templating and text tokenization. Intended for teacher-forced diagnostics."
        ),
    )
    parser.add_argument(
        "--prompt-reasoning",
        choices=("off", "open", "none"),
        default="off",
        help=(
            "Diagnostic prompt suffix mode: off keeps the retained </think> default; "
            "open stops at <think>\\n\\n to match llama.cpp CLI/server traces; none omits thinking tags."
        ),
    )
    parser.add_argument("--output", default=None, help="Output JSON path (default: benchmarks/results/mtp-bench-<timestamp>.json)")
    parser.add_argument(
        "--require-cached-build",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require every session/draft JIT artifact to exist before launch. Use this for "
            "rocprofv3 children so profiling can never spawn hipcc/clang."
        ),
    )
    parser.add_argument(
        "--mtp-context-replay",
        action="store_true",
        help=(
            "Diagnostic only: replay llama.cpp-style prompt catch-up rows through the MTP block. "
            "Slow and currently not the default because the bulk target path does not expose all hidden rows."
        ),
    )
    parser.add_argument(
        "--mtp-device-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MTP_DEVICE_KV_CACHE,
        help=(
            "Use the device-resident MTP dense KV cache for B1 drafting (default: false). "
            "This matches llama.cpp's draft-model context lifecycle without host K/V concat."
        ),
    )
    parser.add_argument(
        "--resident-mtp-draft",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESIDENT_MTP_DRAFT,
        help=(
            "Use the production-shaped resident GGUF MTP draft chain for measured draft rows "
            "(default: HIPENGINE_GGUF_RESIDENT_MTP_DRAFT or false)."
        ),
    )
    parser.add_argument(
        "--resident-mtp-device-seed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the target session's device-resident fp32 hidden seed pointer for resident MTP draft "
            "instead of round-tripping the pending seed through host memory. Diagnostic lifecycle path."
        ),
    )
    parser.add_argument(
        "--resident-mtp-device-chain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "llama.cpp parity diagnostic: enable the resident MTP device-chained draft path "
            "and prewarm its token-embedding table before measured cycles. This removes the "
            "per-depth host top-k/embedding handoff when p_min=0 and top_k<=8."
        ),
    )
    parser.add_argument(
        "--resident-mtp-draft-sync-stage-timings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic only: when cycle stage timings are enabled, insert synchronization points "
            "inside the resident MTP draft runner to attribute the device-chain GPU drain by draft "
            "section. This changes timing and is not a performance route."
        ),
    )
    parser.add_argument(
        "--resident-mtp-draft-q6-top1-dp4a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Accuracy-traded llama-compat diagnostic: use q8_1/dp4a for the resident "
            "draft Q6_K lm-head top-1/gather path. Default off; exact top-1 remains "
            "the shipped resident draft behavior."
        ),
    )
    parser.add_argument(
        "--resident-mtp-draft-q6-top1-stage1-threads",
        type=int,
        choices=(64, 128),
        default=128,
        help=(
            "Diagnostic llama-compat scheduler A/B for the resident draft Q6_K top-1 "
            "lm-head stage1 kernel. llama.cpp RDNA3 Q6_K MMVQ uses a two-warp shape; "
            "hipEngine's current default is 128 threads."
        ),
    )
    parser.add_argument(
        "--resident-mtp-draft-q6-top1-stage1-shape",
        choices=("pack8", "pack16", "pack8_llama", "pack8_scalehoist", "row", "x8", "x8_dscale"),
        default="pack8",
        help=(
            "Diagnostic llama-compat scheduler A/B for the resident draft Q6_K top-1 "
            "lm-head stage1 kernel. 'pack8' is the current hipEngine shape; "
            "'pack16' keeps the pack reduction but handles 16 vocab rows per block; "
            "'x8' keeps the pack8 reduction but reads an X8-packed Q6_K lm-head sidecar; "
            "'x8_dscale' adds an aligned FP32 d*scale sidecar to the X8 path; "
            "'pack8_llama' keeps pack8 reduction but uses llama.cpp's Q6_K vecdot body; "
            "'pack8_scalehoist' keeps pack8 output economy but hoists Q6_K scales; "
            "'row' launches one output row per block with a llama.cpp-like Q6_K MMVQ body."
        ),
    )
    parser.add_argument(
        "--resident-mtp-draft-router-row-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic llama-compat draft-path A/B: route the F32 post-norm router projection "
            "through the row-parallel router logits kernel instead of the generic one-block "
            "MTP F32 linear."
        ),
    )
    parser.add_argument(
        "--resident-mtp-draft-dense-q8-dp4a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic llama-compat draft-path A/B: route resident draft dense Q8_0 "
            "F32 projections through f32->q8_1 plus raw-Q8 dp4a float-output wrappers. "
            "Default off until full-suite economics prove it closes the draft gap."
        ),
    )
    parser.add_argument(
        "--resident-mtp-draft-dense-q8-dp4a-stages",
        default="",
        help=(
            "Comma-separated diagnostic stage selector for --resident-mtp-draft-dense-q8-dp4a. "
            "Choices: all, draft, init, project, qkv, attn_out, shared_gate_up, shared_down, "
            "init_project, init_kv. Empty keeps the legacy all-stage behavior."
        ),
    )
    parser.add_argument(
        "--resident-mtp-draft-selected-silu-down-fused",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic llama-compat draft-path A/B: fuse selected MoE SiLU(gate)*up "
            "into the Q5_K selected-down GEMV. Default off until full-suite economics "
            "prove the launch/intermediate saving transfers."
        ),
    )
    parser.add_argument(
        "--draft-p-min",
        type=float,
        default=0.0,
        help=(
            "Stop drafting when the top-1 probability falls below this threshold "
            "(llama.cpp --spec-draft-p-min, default 0.0 = always draft to n_max)."
        ),
    )
    parser.add_argument(
        "--record-draft-confidence",
        action="store_true",
        help=(
            "Diagnostic only: record per-depth resident draft top-1 softmax probabilities "
            "in raw cycle artifacts without changing the acceptance policy."
        ),
    )
    parser.add_argument(
        "--record-draft-topk-scores",
        action="store_true",
        help=(
            "Diagnostic only: when host draft logits are available, record logits and "
            "top-1 margins for the reported draft top-k token IDs. Resident fast paths "
            "that never materialize full logits leave these arrays empty."
        ),
    )
    parser.add_argument(
        "--record-target-topk-scores",
        action="store_true",
        help=(
            "Diagnostic only: when block verification materializes full target lm-head "
            "logits, record compact per-row target top-k and candidate-token scores. "
            "This adds a full-logit D2H copy per target block and is not a timing route."
        ),
    )
    parser.add_argument(
        "--target-topk-score-count",
        type=int,
        default=10,
        help="Number of target lm-head top-k rows to retain with --record-target-topk-scores.",
    )
    parser.add_argument(
        "--target-score-candidate-tokens",
        type=_parse_nonnegative_token_csv,
        default=(),
        help=(
            "Comma-separated extra token IDs to score in target lm-head diagnostics, "
            "for example llama.cpp near-tie tokens that hipEngine did not sample."
        ),
    )
    parser.add_argument(
        "--record-draft-hidden-stats",
        action="store_true",
        help=(
            "Diagnostic only: record compact FP32 hidden-row summaries for the MTP "
            "draft seed, per-depth draft hidden outputs, and verifier-selected "
            "pending seed. This forces resident host-chain drafting when needed."
        ),
    )
    parser.add_argument(
        "--record-draft-stage-stats",
        action="store_true",
        help=(
            "Diagnostic only: record compact FP32 summaries for resident MTP draft "
            "sub-stage tensors and dense K/V cache rows. This forces resident "
            "host-chain drafting when needed and is not a timing route."
        ),
    )
    parser.add_argument(
        "--record-draft-cache-rows",
        type=_parse_nonnegative_int_csv,
        default=(),
        metavar="ROWS",
        help=(
            "Diagnostic only: with --record-draft-stage-stats, also summarize the requested "
            "comma-separated dense MTP K/V history rows when they are visible in the current "
            "draft context, for example 0,1,2,16,32,48,49."
        ),
    )
    parser.add_argument(
        "--record-draft-attention-debug",
        action="store_true",
        help=(
            "Diagnostic only: with --record-draft-stage-stats and --mtp-device-kv-cache, "
            "read back resident draft Q/K/V plus the GPU attention output, recompute dense "
            "attention on the host, and emit per-head top score/weight summaries."
        ),
    )
    parser.add_argument(
        "--record-cycle-stage-timings",
        action="store_true",
        help=(
            "Diagnostic only: add cycle_wall_ms plus a stage_timings_ms breakdown to raw cycles. "
            "This records draft, verify, replay/commit, KV/context commit, and hidden bookkeeping "
            "windows so MTP economics gaps can be decomposed without changing retained metrics."
        ),
    )
    parser.add_argument(
        "--resident-mtp-draft-gpu-event-stage-timings",
        action="store_true",
        help=(
            "Diagnostic only: with resident device-chain drafting and --record-cycle-stage-timings, "
            "add HIP event elapsed-time buckets for queued draft GPU sub-stages without inserting "
            "per-stage device synchronizes."
        ),
    )
    parser.add_argument(
        "--root-topk-accept",
        type=int,
        default=DEFAULT_ROOT_TOPK_ACCEPT,
        help=(
            "Diagnostic tree proposal: accept a depth-0 draft when the target token is in "
            "the first K root candidates (default: 40 = generic root tree; use 1 for linear argmax path)."
        ),
    )
    parser.add_argument(
        "--sibling-topk-accept",
        type=int,
        default=DEFAULT_SIBLING_TOPK_ACCEPT,
        help=(
            "Diagnostic tree proposal: accept a non-argmax sibling at the first deeper "
            "mismatch when the target token is in the first K candidates (default: 1 = linear argmax path)."
        ),
    )
    parser.add_argument(
        "--sibling-tail-min-prev-accepted",
        type=int,
        default=0,
        help=(
            "Diagnostic adaptive sibling policy: when non-negative, sibling candidates beyond rank 8 "
            "are accepted only if the previous cycle accepted at least this many draft tokens "
            "(default: 0; use -1 to disable the gate)."
        ),
    )
    parser.add_argument(
        "--sibling-topk-max-depth",
        type=int,
        default=4,
        help=(
            "Maximum non-root draft depth eligible for sibling top-k acceptance "
            "(default: 4; root depth 0 is controlled by --root-topk-accept)."
        ),
    )
    parser.add_argument(
        "--root-tail-max-prev-accepted",
        type=int,
        default=-1,
        help=(
            "Diagnostic adaptive root policy: when non-negative, root candidates below rank 4 "
            "are accepted only if the previous cycle accepted at most this many draft tokens "
            "(default: -1, disabled)."
        ),
    )
    parser.add_argument(
        "--topk-branch-redraft",
        dest="topk_branch_redraft",
        action="store_true",
        default=DEFAULT_TOPK_BRANCH_REDRAFT,
        help=(
            "Diagnostic branch-reset proposal: after one exact-verified top-k branch accept, "
            "redraft the remaining B-window from the accepted target token and only accept "
            "subsequent argmax redraft matches (default: disabled; opt in for tree coverage diagnostics)."
        ),
    )
    parser.add_argument(
        "--no-topk-branch-redraft",
        dest="topk_branch_redraft",
        action="store_false",
        help="Disable branch-reset redrafting after a top-k branch accept.",
    )
    parser.add_argument(
        "--topk-branch-redraft-max-branches",
        type=int,
        default=5,
        help=(
            "Maximum exact-verified top-k branch accepts allowed within one B-window when "
            "branch redraft is enabled (default: 5)."
        ),
    )
    return parser


_SESSION_CACHE: dict = {}
_DRAFT_RUNNER_CACHE: dict = {}
"""Opt-in resident-session and draft-runner cache for load-once batch runs.

Default behavior is unchanged: when ``HIPENGINE_MTP_BENCH_CACHE_SESSION`` is not
"1", ``main()`` constructs a fresh session and closes it in its finally block,
exactly as before (every existing subprocess/test caller). When the flag is set,
``main()`` reuses one resident session across calls (reset between runs) and does
NOT close it, so a batch driver (e.g. gguf_mtp_category_bench in-process loop)
pays the ~50s model load once instead of per (prompt, budget). Eligible resident
draft runners are also reused: their scratch is request-stateless, and stable
ownership lets reusable proposal graphs survive across prompt resets just like
target graphs. Correctness is gated by session.reset(); validate token-stream,
K/V, and acceptance parity vs the fresh subprocess path before trusting timing.
"""


def apply_llama_compat_args(args: argparse.Namespace) -> None:
    """Force the closest hipEngine route to llama.cpp's MTP lifecycle.

    This is intentionally opt-in and diagnostic. It overrides conflicting
    argparse values after parsing so wrapper commands that pass --draft-n-max
    before --llama-compat cannot silently create a non-compat run.
    """
    if not bool(getattr(args, "llama_compat", False)):
        return

    args.draft_n_max = 2
    args.root_topk_accept = 1
    args.sibling_topk_accept = 1
    args.topk_branch_redraft = False
    args.draft_p_min = 0.0
    args.mtp_draft_vocab_cap = 0

    args.resident_mtp_draft = True
    args.resident_mtp_device_seed = bool(getattr(args, "resident_mtp_device_seed", False))
    if not hasattr(args, "resident_mtp_device_chain"):
        args.resident_mtp_device_chain = False
    args.mtp_context_replay = True
    args.mtp_device_kv_cache = True

    args.target_graph_verify = False
    args.target_graph_batched_verify = False
    args.target_block_verify = True
    args.target_block_verify_mode = "bulk"
    args.target_block_min_rows = 2
    args.target_block_direct_state_commit = True
    if getattr(args, "target_block_direct_partial_replay_mode", "serial-exact") == "serial-exact":
        args.target_block_direct_partial_replay_mode = "direct-commit"
    args.target_b1_branch_safe_block_verify = False

    args.adaptive_draft_window = False
    args.adaptive_ar_fallback = False
    args.adaptive_ar_fallback_max_accepted = 0
    args.adaptive_ar_fallback_cooldown = 0
    args.adaptive_full_vocab_after_cap_miss = False
    args.adaptive_block_after_full_accept = False
    args.fused_b1_block_probe = False
    args.adaptive_probe_draft_n_max = 1
    args.adaptive_strict_block_probe = False
    args.adaptive_strict_fallback_draft_n_max = 1


def _diagnostic_topk_candidate_count(args: argparse.Namespace, proposal_count: int) -> int:
    """Return the draft top-k width used for diagnostics/proposal readback.

    The strict llama-compat route only accepts top-1 candidates, and asking the
    resident draft runner for top-10 forces a full-vocab host readback because
    its device top-k path is capped at 8.  Keep the historical top-10 diagnostic
    rows for non-compat exploratory routes where rank histograms matter.
    """

    proposal = max(1, int(proposal_count))
    if bool(getattr(args, "llama_compat", False)):
        return proposal
    return max(10, proposal)


def should_use_fused_b1_block_probe(
    *,
    fused_b1_block_probe: bool,
    adaptive_block_after_full_accept: bool,
    adaptive_block_verify_active: bool,
    cycle_ar_fallback: bool,
    cycle_draft_window: int,
    adaptive_probe_draft_n_max: int,
    cycle_root_topk_accept: int,
    cycle_sibling_topk_accept: int,
    topk_branch_redraft: bool,
) -> bool:
    """Return true when the adaptive B1 probe should use the block verifier."""

    return (
        bool(fused_b1_block_probe)
        and bool(adaptive_block_after_full_accept)
        and not bool(adaptive_block_verify_active)
        and not bool(cycle_ar_fallback)
        and int(cycle_draft_window) == 1
        and int(adaptive_probe_draft_n_max) == 1
        and int(cycle_root_topk_accept) == 1
        and int(cycle_sibling_topk_accept) == 1
        and not bool(topk_branch_redraft)
    )


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    apply_llama_compat_args(args)
    if getattr(args, "llama_compat", False):
        os.environ["HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"] = "1"
    if getattr(args, "verify_dp4a", False):
        # Opt-in llama-style dp4a verify: enable the selected-expert q8_1/dp4a GEMVs.
        # Read live by the runner's _gguf_*_selected_dp4a_enabled() (os.environ).
        # Default OFF; accuracy-degrading (fails the ja correctness gate).
        for _flag in (
            "HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A",
            "HIPENGINE_GGUF_T16_SELECTED_DP4A",
            "HIPENGINE_GGUF_RAW_SELECTED_DP4A",
        ):
            os.environ[_flag] = "1"
    if getattr(args, "verify_dense_q8_dp4a", False):
        # Must be set before materialization so T16 dense Q8_0 weights retain
        # raw GGUF bytes alongside tiles. The runtime route itself is read live
        # by qwen35_gguf_runner.
        os.environ["HIPENGINE_GGUF_Q8_0_RAW_SIDECAR"] = "1"
        os.environ["HIPENGINE_GGUF_DENSE_Q8_DP4A"] = "1"
    if getattr(args, "verify_dense_q8_dp4a_all", False):
        # Same materialization constraint as the pair-only route. The ALL flag
        # also enables the pair helper through qwen35_gguf_runner.
        os.environ["HIPENGINE_GGUF_Q8_0_RAW_SIDECAR"] = "1"
        os.environ["HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL"] = "1"
    if getattr(args, "verify_dense_q8_dp4a_shared", False):
        # Shared gate/up/down uses the same raw sidecars and q8_1 workspace as
        # denseq8all. Keep it independently gated because the shared tensors are
        # much smaller and may prefer the old T16 body.
        os.environ["HIPENGINE_GGUF_Q8_0_RAW_SIDECAR"] = "1"
        os.environ["HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL"] = "1"
        os.environ["HIPENGINE_GGUF_DENSE_Q8_DP4A_SHARED"] = "1"
    if getattr(args, "verify_dense_q8_dp4a_f32", False):
        # Direct-state ssm_out uses F32 activations, so the BF16 denseq8all
        # helper cannot cover it. Keep this route separate until full-suite
        # evidence says the extra quantize launch transfers.
        os.environ["HIPENGINE_GGUF_Q8_0_RAW_SIDECAR"] = "1"
        os.environ["HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL"] = "1"
        os.environ["HIPENGINE_GGUF_DENSE_Q8_DP4A_F32"] = "1"
    if getattr(args, "verify_lm_head_q6_top1_dp4a", False):
        # Must be set before target materialization so root.lm_head retains the
        # X8 Q6_K sidecar used by the verifier direct top-1 path.
        os.environ["HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR"] = "1"
        os.environ["HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A"] = "1"
    if str(getattr(args, "selected_down_x8_repack", "off")) != "off":
        # Must be set before materialization so selected-down expert tensors are
        # repacked into the X8 replacement layout.
        os.environ["HIPENGINE_GGUF_SELECTED_X8_REPACK"] = str(args.selected_down_x8_repack)
    if getattr(args, "selected_gate_up_x8", False):
        # Must be set before materialization so selected gate/up expert tensors
        # are repacked into the X8 replacement layout.
        os.environ["HIPENGINE_GGUF_SELECTED_GATE_UP_X8"] = "1"
    if getattr(args, "selected_gate_up_raw", False):
        # Must be set before materialization so selected gate/up expert tensors
        # stay in raw GGUF layout. With --verify-dp4a, the runtime uses the raw
        # q8_1/dp4a selected-dual body instead of the T16 replacement body.
        os.environ["HIPENGINE_GGUF_SELECTED_GATE_UP_RAW"] = "1"
    if getattr(args, "resident_mtp_draft_q6_top1_dp4a", False):
        # Must be set before constructing the resident MTP draft runner.
        os.environ["HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A"] = "1"
    if getattr(args, "resident_mtp_draft_router_row_parallel", False):
        os.environ["HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL"] = "1"
    if getattr(args, "resident_mtp_draft_dense_q8_dp4a", False):
        os.environ["HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A"] = "1"
    resident_dense_q8_stages = str(getattr(args, "resident_mtp_draft_dense_q8_dp4a_stages", "")).strip()
    if resident_dense_q8_stages:
        os.environ["HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A_STAGES"] = resident_dense_q8_stages
    if getattr(args, "resident_mtp_draft_selected_silu_down_fused", False):
        os.environ["HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED"] = "1"
    os.environ["HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS"] = str(args.resident_mtp_draft_q6_top1_stage1_threads)
    os.environ["HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE"] = str(args.resident_mtp_draft_q6_top1_stage1_shape)
    try:
        args.draft_n_max = validate_draft_n_max(args.draft_n_max)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_output_tokens < 0:
        parser.error("--max-output-tokens must be non-negative")
    if args.root_topk_accept < 1 or args.root_topk_accept > 4096:
        parser.error("--root-topk-accept must be in 1..4096")
    if args.sibling_topk_accept < 1 or args.sibling_topk_accept > 4096:
        parser.error("--sibling-topk-accept must be in 1..4096")
    if args.sibling_topk_max_depth < 0:
        parser.error("--sibling-topk-max-depth must be non-negative")
    if args.root_tail_max_prev_accepted < -1:
        parser.error("--root-tail-max-prev-accepted must be >= -1")
    if args.topk_branch_redraft_max_branches < 1:
        parser.error("--topk-branch-redraft-max-branches must be positive")
    if args.mtp_draft_vocab_cap < 0:
        parser.error("--mtp-draft-vocab-cap must be non-negative")
    if args.adaptive_full_vocab_after_cap_miss and not args.resident_mtp_draft:
        parser.error("--adaptive-full-vocab-after-cap-miss requires --resident-mtp-draft")
    if args.resident_mtp_device_seed and not args.resident_mtp_draft:
        parser.error("--resident-mtp-device-seed requires --resident-mtp-draft")
    if args.resident_mtp_device_chain and not args.resident_mtp_draft:
        parser.error("--resident-mtp-device-chain requires --resident-mtp-draft")
    if args.resident_mtp_device_seed and args.topk_branch_redraft:
        parser.error("--resident-mtp-device-seed is not yet compatible with --topk-branch-redraft")
    if args.adaptive_full_vocab_after_cap_miss and args.mtp_draft_vocab_cap <= 0:
        parser.error("--adaptive-full-vocab-after-cap-miss requires --mtp-draft-vocab-cap > 0")
    if args.adaptive_ar_fallback_max_accepted < 0:
        parser.error("--adaptive-ar-fallback-max-accepted must be non-negative")
    if args.adaptive_probe_draft_n_max < 1:
        parser.error("--adaptive-probe-draft-n-max must be positive")
    # The probe window is only consumed when --adaptive-block-after-full-accept is
    # set; otherwise its default (3) is unused and must not block B1/B2 runs.
    if args.adaptive_block_after_full_accept and args.adaptive_probe_draft_n_max > args.draft_n_max:
        parser.error("--adaptive-probe-draft-n-max must be <= --draft-n-max")
    if args.adaptive_strict_probe_cycles < 1:
        parser.error("--adaptive-strict-probe-cycles must be positive")
    if args.adaptive_strict_probe_min_accepted < 0:
        parser.error("--adaptive-strict-probe-min-accepted must be non-negative")
    if args.adaptive_strict_fallback_draft_n_max < 0:
        parser.error("--adaptive-strict-fallback-draft-n-max must be non-negative")
    if args.adaptive_strict_fallback_draft_n_max > args.draft_n_max:
        parser.error("--adaptive-strict-fallback-draft-n-max must be <= --draft-n-max")
    if args.adaptive_strict_fallback_root_topk < 1 or args.adaptive_strict_fallback_root_topk > 4096:
        parser.error("--adaptive-strict-fallback-root-topk must be in 1..4096")
    if not _hip_available():
        print("ERROR: ROCm/HIP not available", file=sys.stderr)
        sys.exit(1)
    proposal_topk_candidate_count = max(
        1,
        args.root_topk_accept,
        args.sibling_topk_accept,
        args.adaptive_strict_fallback_root_topk if args.adaptive_strict_block_probe else 1,
    )
    if args.resident_mtp_device_seed and proposal_topk_candidate_count > 64:
        parser.error("--resident-mtp-device-seed requires resident draft top-k <= 64")
    if args.resident_mtp_device_seed and args.draft_p_min > 0.0:
        parser.error("--resident-mtp-device-seed requires --draft-p-min 0")
    if args.resident_mtp_device_chain and proposal_topk_candidate_count > 8:
        parser.error("--resident-mtp-device-chain requires resident draft top-k <= 8")
    if args.resident_mtp_device_chain and args.draft_p_min > 0.0:
        parser.error("--resident-mtp-device-chain requires --draft-p-min 0")
    if args.resident_mtp_device_chain and args.record_draft_confidence:
        parser.error("--resident-mtp-device-chain is not compatible with --record-draft-confidence")
    if args.record_draft_cache_rows and not args.record_draft_stage_stats:
        parser.error("--record-draft-cache-rows requires --record-draft-stage-stats")
    if args.record_draft_attention_debug and not args.record_draft_stage_stats:
        parser.error("--record-draft-attention-debug requires --record-draft-stage-stats")
    if args.record_draft_attention_debug and not args.mtp_device_kv_cache:
        parser.error("--record-draft-attention-debug requires --mtp-device-kv-cache")
    if args.resident_mtp_draft_sync_stage_timings and not args.resident_mtp_draft:
        parser.error("--resident-mtp-draft-sync-stage-timings requires --resident-mtp-draft")
    if args.resident_mtp_draft_sync_stage_timings and not args.record_cycle_stage_timings:
        parser.error("--resident-mtp-draft-sync-stage-timings requires --record-cycle-stage-timings")
    if args.resident_mtp_draft_gpu_event_stage_timings and not args.resident_mtp_draft:
        parser.error("--resident-mtp-draft-gpu-event-stage-timings requires --resident-mtp-draft")
    if args.resident_mtp_draft_gpu_event_stage_timings and not args.resident_mtp_device_chain:
        parser.error("--resident-mtp-draft-gpu-event-stage-timings requires --resident-mtp-device-chain")
    if args.resident_mtp_draft_gpu_event_stage_timings and not args.record_cycle_stage_timings:
        parser.error("--resident-mtp-draft-gpu-event-stage-timings requires --record-cycle-stage-timings")
    if args.native_spec_target_cycle and not args.target_block_verify:
        parser.error("--native-spec-target-cycle requires --target-block-verify")
    if args.native_spec_target_cycle and args.target_block_verify_mode != "bulk":
        parser.error("--native-spec-target-cycle requires --target-block-verify-mode bulk")
    if args.native_spec_target_cycle and args.target_block_wmma_prefill:
        parser.error("--native-spec-target-cycle requires --no-target-block-wmma-prefill")
    if args.native_spec_target_cycle and args.record_target_topk_scores:
        parser.error("--native-spec-target-cycle does not support target logits readback")
    if args.native_spec_target_cycle and args.target_block_sync_stage_timings:
        parser.error("--native-spec-target-cycle does not support synchronized stage timings")
    if args.native_spec_device_accept_commit and not args.native_spec_target_cycle:
        parser.error("--native-spec-device-accept-commit requires --native-spec-target-cycle")
    if args.native_spec_device_accept_commit and not args.resident_mtp_device_seed:
        parser.error("--native-spec-device-accept-commit requires --resident-mtp-device-seed")
    if args.native_spec_device_accept_commit and not args.target_block_direct_state_commit:
        parser.error("--native-spec-device-accept-commit requires --target-block-direct-state-commit")
    if args.native_spec_complete_cycle and not args.native_spec_device_accept_commit:
        parser.error("--native-spec-complete-cycle requires --native-spec-device-accept-commit")
    if args.native_spec_complete_cycle and not args.resident_mtp_device_chain:
        parser.error("--native-spec-complete-cycle requires --resident-mtp-device-chain")
    if args.native_spec_complete_cycle and not args.mtp_device_kv_cache:
        parser.error("--native-spec-complete-cycle requires --mtp-device-kv-cache")
    if args.native_spec_complete_cycle and (
        args.root_topk_accept != 1 or args.sibling_topk_accept != 1 or args.draft_p_min != 0.0
    ):
        parser.error("--native-spec-complete-cycle requires strict top-1 and --draft-p-min 0")
    if args.native_spec_proposal_graph and not args.native_spec_complete_cycle:
        parser.error("--native-spec-proposal-graph requires --native-spec-complete-cycle")
    if args.target_block_sync_stage_timings and not args.target_block_verify:
        parser.error("--target-block-sync-stage-timings requires --target-block-verify")
    if args.target_block_sync_stage_timings and not args.record_cycle_stage_timings:
        parser.error("--target-block-sync-stage-timings requires --record-cycle-stage-timings")
    if args.record_target_topk_scores and not args.target_block_verify:
        parser.error("--record-target-topk-scores requires --target-block-verify")
    if args.target_topk_score_count < 1:
        parser.error("--target-topk-score-count must be positive")
    if args.target_b1_branch_safe_block_verify and not args.target_block_verify:
        parser.error("--target-b1-branch-safe-block-verify requires --target-block-verify")
    if args.fused_b1_block_probe:
        if not args.target_block_verify:
            parser.error("--fused-b1-block-probe requires --target-block-verify")
        if not args.adaptive_block_after_full_accept:
            parser.error("--fused-b1-block-probe requires --adaptive-block-after-full-accept")
        if args.adaptive_probe_draft_n_max != 1:
            parser.error("--fused-b1-block-probe requires --adaptive-probe-draft-n-max 1")
        if int(args.target_block_min_rows) != 2:
            parser.error("--fused-b1-block-probe requires --target-block-min-rows 2")
        if args.root_topk_accept != 1 or args.sibling_topk_accept != 1 or args.topk_branch_redraft:
            parser.error("--fused-b1-block-probe requires strict top-1 linear drafting")
    diagnostic_topk_candidate_count = _diagnostic_topk_candidate_count(args, proposal_topk_candidate_count)
    topk_candidate_count = proposal_topk_candidate_count
    if args.decode_repack:
        os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    else:
        os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "0"
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    from hipengine.loading.gguf import GGUFReader
    from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        qwen35_gguf_mtp_nextn_layer_logits_f32 as gpu_kernel,
    )
    from hipengine.speculative.gguf_mtp import Qwen35GGUFMTPContext
    from hipengine.speculative.mtp_resident_draft import Qwen35GGUFResidentMTPDraftRunner
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.core.hip import get_hip_runtime, HipMemcpyKind
    from hipengine.core.memory import free, malloc

    # Load model info + weights
    r = GGUFReader(args.model)
    tok = Qwen35GGUFTokenizer.from_gguf_info(r.info)
    meta = r.info.metadata

    weights = {}
    for t in r.info.tensors:
        if "blk.40" in t.name or t.name == "output.weight" or t.name == "token_embd.weight":
            data = r.tensor_data(t.name)
            weights[t.name] = (data, t.ggml_type, t.shape)

    def get(name): return weights[name][0]
    def qt(name): return GGMLQuantizationType(weights[name][1])
    def dq(name): return dequantize_gguf_data(get(name), qt(name)).astype(np.float32)

    token_embd_f32 = dq("token_embd.weight")

    # Extract RoPE parameters from model metadata for the MTP draft attention.
    rope_dim = int(meta.get("qwen35moe.rope.dimension_count", 64))
    rope_base = float(meta.get("qwen35moe.rope.freq_base", 10000000.0))
    _rope_cos, _rope_sin = _rope_tables(
        max_positions=262144, rotary_dim=rope_dim, base=rope_base
    )

    # Build the normal chat prompt or preserve an exact teacher-forced token context.
    prompt = select_prompt_tokens(
        tok,
        user_prompt=args.prompt,
        reasoning=args.prompt_reasoning,
        raw_token_ids=tuple(args.prompt_token_ids),
    )

    print(f"Prompt: {repr(tok.decode(prompt))}")
    print(f"Prompt tokens: {len(prompt)}")

    # Use raw Q6_K shared_head weight (398MB vs 2034MB F32 dequant)
    sh_raw = np.asarray(get("output.weight"), dtype=np.uint8)
    sh_qtype = qt("output.weight")
    print(f"Raw shared_head ({sh_qtype.name}): {sh_raw.nbytes/1e6:.0f}MB")

    # Build GPU kernel args
    def run_draft(hidden_seed, token_embed, *, return_hidden_seed: bool = False,
                   positions=None, rope_cos=None, rope_sin=None, rotary_dim=None,
                   dense_key_cache=None, dense_value_cache=None, dense_cache_len: int | None = None,
                   kv_write_only: bool = False):
        gpu_args = [
            hidden_seed, token_embed,
            get("blk.40.nextn.eh_proj.weight"), get("blk.40.nextn.hnorm.weight"),
            get("blk.40.nextn.enorm.weight"), get("blk.40.attn_norm.weight"),
            get("blk.40.attn_q.weight"), get("blk.40.attn_k.weight"),
            get("blk.40.attn_v.weight"), get("blk.40.attn_output.weight"),
            get("blk.40.attn_q_norm.weight"), get("blk.40.attn_k_norm.weight"),
            get("blk.40.post_attention_norm.weight"), get("blk.40.ffn_gate_inp.weight"),
            get("blk.40.ffn_gate_exps.weight"), get("blk.40.ffn_up_exps.weight"),
            get("blk.40.ffn_down_exps.weight"),
            qt("blk.40.ffn_gate_exps.weight"), qt("blk.40.ffn_up_exps.weight"), qt("blk.40.ffn_down_exps.weight"),
            get("blk.40.ffn_gate_inp_shexp.weight"),
            get("blk.40.ffn_gate_shexp.weight"), get("blk.40.ffn_up_shexp.weight"),
            get("blk.40.ffn_down_shexp.weight"), qt("blk.40.ffn_gate_shexp.weight"),
            get("blk.40.nextn.shared_head_norm.weight"), sh_raw,
        ]
        gpu_kwargs = dict(
            num_heads=16, num_kv_heads=2, experts_used=8,
            eh_proj_qtype=qt("blk.40.nextn.eh_proj.weight"),
            wq_qtype=qt("blk.40.attn_q.weight"), wk_qtype=qt("blk.40.attn_k.weight"),
            wv_qtype=qt("blk.40.attn_v.weight"), wo_qtype=qt("blk.40.attn_output.weight"),
            eps=1e-6,
            shared_head_qtype=sh_qtype,
            return_hidden_seed=return_hidden_seed,
        )
        if positions is not None:
            gpu_kwargs["positions"] = positions
            if dense_cache_len is None:
                gpu_kwargs["context_counts"] = np.arange(1, len(positions) + 1, dtype=np.int64)
            else:
                gpu_kwargs["context_counts"] = int(dense_cache_len) + np.arange(1, len(positions) + 1, dtype=np.int64)
        if rope_cos is not None and rope_sin is not None:
            gpu_kwargs["rope_cos"] = rope_cos
            gpu_kwargs["rope_sin"] = rope_sin
            gpu_kwargs["rotary_dim"] = rotary_dim or rope_dim
        if dense_key_cache is not None:
            gpu_kwargs["dense_key_cache"] = dense_key_cache
            gpu_kwargs["dense_value_cache"] = dense_value_cache
            gpu_kwargs["dense_cache_len"] = int(dense_cache_len or 0)
        if kv_write_only:
            gpu_kwargs["kv_write_only"] = True
        if args.mtp_draft_vocab_cap:
            gpu_kwargs["draft_vocab_cap"] = int(args.mtp_draft_vocab_cap)
        result = gpu_kernel(*gpu_args, **gpu_kwargs)
        if return_hidden_seed:
            logits, next_hidden_seed = result
            return (
                np.asarray(logits, dtype=np.float32),
                np.ascontiguousarray(next_hidden_seed, dtype=np.float32),
            )
        return np.asarray(result, dtype=np.float32)

    # Run benchmark
    mtp_device_kv_buffers = []
    resident_draft = None
    resident_draft_full_vocab = None
    resident_mtp_draft_effective = False
    resident_mtp_device_chain_effective = False
    resident_mtp_draft_full_vocab_recovery_effective = False
    resident_mtp_draft_fallback_reason = None
    _cache_session = os.environ.get("HIPENGINE_MTP_BENCH_CACHE_SESSION") == "1"
    _session_key = (
        str(args.model),
        bool(args.use_wmma_prefill),
        bool(args.use_gemv_decode),
        os.environ.get("HIPENGINE_GGUF_DECODE_REPACK"),
        os.environ.get("HIPENGINE_GGUF_Q8_0_RAW_SIDECAR"),
        os.environ.get("HIPENGINE_GGUF_DENSE_Q8_DP4A"),
        os.environ.get("HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL"),
        bool(args.require_cached_build),
    )
    if _cache_session and _session_key in _SESSION_CACHE:
        session = _SESSION_CACHE[_session_key]
        session.reset()  # clean state for the new prompt (KV + recurrent + position)
    else:
        session = Qwen35GGUFResidentSession(
            model_path=args.model,
            use_wmma_prefill=bool(args.use_wmma_prefill),
            use_gemv_decode=bool(args.use_gemv_decode),
            require_cached_build=bool(args.require_cached_build),
        )
        if _cache_session:
            _SESSION_CACHE[_session_key] = session
    target_graph = None
    runtime = None
    try:
        runtime = session.runtime or get_hip_runtime()
        hidden_size = 2048
        if args.resident_mtp_draft:
            if topk_candidate_count > 64:
                resident_mtp_draft_fallback_reason = "resident draft top-k kernel supports production top-k up to 64"
            else:
                draft_vocab_cap = int(args.mtp_draft_vocab_cap or sh_raw.shape[0])
                draft_env_names = (
                    "HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A",
                    "HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A_STAGES",
                    "HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED",
                    "HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_MOE",
                    "HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_GATHER",
                    "HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A",
                    "HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE",
                    "HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS",
                    "HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL",
                    "HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL",
                    "HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_CHAIN",
                )
                draft_cache_key = (
                    _session_key,
                    draft_vocab_cap,
                    bool(args.resident_mtp_device_chain),
                    bool(args.resident_mtp_draft_sync_stage_timings),
                    bool(args.resident_mtp_draft_gpu_event_stage_timings),
                    tuple(os.environ.get(name) for name in draft_env_names),
                )
                if _cache_session and draft_cache_key in _DRAFT_RUNNER_CACHE:
                    resident_draft = _DRAFT_RUNNER_CACHE[draft_cache_key]
                else:
                    resident_draft = Qwen35GGUFResidentMTPDraftRunner(
                        weights,
                        token_embd_f32,
                        runtime=runtime,
                        vocab_cap=draft_vocab_cap,
                        device_chain_enabled=True if args.resident_mtp_device_chain else None,
                        prewarm_device_chain=bool(args.resident_mtp_device_chain),
                        sync_stage_timings=bool(args.resident_mtp_draft_sync_stage_timings),
                        gpu_event_stage_timings=bool(args.resident_mtp_draft_gpu_event_stage_timings),
                        require_cached_build=bool(args.require_cached_build),
                    )
                    if _cache_session:
                        _DRAFT_RUNNER_CACHE[draft_cache_key] = resident_draft
                resident_mtp_device_chain_effective = bool(resident_draft._device_chain_enabled)
                if (
                    args.adaptive_full_vocab_after_cap_miss
                    and int(args.mtp_draft_vocab_cap) > 0
                    and int(args.mtp_draft_vocab_cap) < int(sh_raw.shape[0])
                ):
                    resident_draft_full_vocab = Qwen35GGUFResidentMTPDraftRunner(
                        weights,
                        token_embd_f32,
                        runtime=runtime,
                        vocab_cap=int(sh_raw.shape[0]),
                        device_chain_enabled=True if args.resident_mtp_device_chain else None,
                        prewarm_device_chain=bool(args.resident_mtp_device_chain),
                        sync_stage_timings=bool(args.resident_mtp_draft_sync_stage_timings),
                        gpu_event_stage_timings=bool(args.resident_mtp_draft_gpu_event_stage_timings),
                        require_cached_build=bool(args.require_cached_build),
                    )
                    resident_mtp_draft_full_vocab_recovery_effective = True
                resident_mtp_draft_effective = True

        def copy_pending_hidden_seed() -> np.ndarray:
            hidden_seed = np.empty((1, hidden_size), dtype=np.float32)
            runtime.memcpy(
                hidden_seed.ctypes.data,
                session.fp32_hidden_seed_ptr(),
                hidden_size * 4,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            return hidden_seed

        def bulk_prefill_with_hidden_trace() -> tuple[object, np.ndarray]:
            """Consume the prompt with the target prefill path and capture every hidden row.

            llama.cpp's MTP context catch-up needs the post-output_norm hidden
            row for every target prompt token, shifted right by one row.
            """
            prefill_result = session.prefill(
                prompt,
                use_bulk=True,
                bulk_attention_mode="bulk",
                return_logits=False,
                capture_hidden_seed_fp32=True,
            )
            hidden_rows = np.empty((len(prompt), hidden_size), dtype=np.float32)
            runtime.memcpy(
                hidden_rows.ctypes.data,
                session.fp32_verify_hidden_seed_ptr(0),
                hidden_rows.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            return prefill_result, np.ascontiguousarray(hidden_rows, dtype=np.float32)

        if args.mtp_context_replay:
            # Build llama.cpp-style draft catch-up rows.  Row 0 uses a zero
            # hidden seed and row i uses the target hidden from prompt token
            # i-1; this mirrors llama.cpp's shifted MTP ``process()`` input.
            prefill_result, prompt_hidden_rows = bulk_prefill_with_hidden_trace()
            prev_token = int(prefill_result.token_id)
            pending_hidden_seed = copy_pending_hidden_seed()
            mtp_context_tokens, mtp_context_hidden_rows = llama_cpp_mtp_catchup_rows(
                prompt, prompt_hidden_rows
            )
            target_prefill_mode = "bulk_prefill_hidden_rows"
            if args.mtp_device_kv_cache:
                mtp_context_mode = (
                    "llamacpp_shifted_prompt_replay_device_seed"
                    if args.resident_mtp_device_seed
                    else "llamacpp_shifted_prompt_replay"
                )
            else:
                mtp_context_mode = "llamacpp_shifted_prompt_rows_host_only"
        else:
            prefill_result = session.prefill(prompt, return_logits=False, capture_hidden_seed_fp32=True)
            prev_token = int(prefill_result.token_id)
            pending_hidden_seed = copy_pending_hidden_seed()
            mtp_context_tokens = []
            mtp_context_hidden_rows = np.empty((0, hidden_size), dtype=np.float32)
            target_prefill_mode = "resident_default"
            mtp_context_mode = "single_seed_row"
        resident_context = None
        if args.resident_mtp_device_seed:
            resident_context = Qwen35GGUFMTPContext.from_target_seed(
                session,
                token_id=prev_token,
                position=int(session.position) - 1,
                mtp_block=resident_draft,
            )

        # Track the current sequence position for draft model RoPE.
        # After prefill, session.position = len(prompt). The first sampled
        # token (prev_token) will be verified at this position.
        seq_position = int(session.position)
        initial_prev_token = int(prev_token)
        initial_prev_position = int(seq_position)
        # Positions of the committed MTP context tokens (for replay mode).
        if args.mtp_context_replay:
            mtp_context_positions = list(range(len(mtp_context_tokens)))
        else:
            mtp_context_positions: list[int] = []

        mtp_draft_warmup_ms = 0.0
        if args.mtp_draft_warmup:
            warmup_pos = np.asarray([seq_position], dtype=np.int64)
            warmup_hidden_seed = np.ascontiguousarray(pending_hidden_seed, dtype=np.float32).copy()
            warmup_token_embed = token_embd_f32[prev_token:prev_token + 1].copy()
            t_warmup0 = time.perf_counter()
            _ = run_draft(
                warmup_hidden_seed,
                warmup_token_embed,
                return_hidden_seed=args.draft_n_max > 1,
                positions=warmup_pos,
                rope_cos=_rope_cos[warmup_pos],
                rope_sin=_rope_sin[warmup_pos],
                rotary_dim=rope_dim,
            )
            mtp_draft_warmup_ms = (time.perf_counter() - t_warmup0) * 1000

        target_graph_steps_per_replay = int(args.draft_n_max) + 1 if args.target_graph_batched_verify else 1
        target_graph_max_replay_steps = int(args.cycles) * (int(args.draft_n_max) + 1)
        target_graph_context_cap = int(seq_position) + target_graph_max_replay_steps
        target_graph_verify_enabled = False
        target_graph_verify_fallback_reason: str | None = None
        current_device_token = int(prev_token)
        if args.target_graph_verify and (
            not args.target_block_verify
            or args.adaptive_ar_fallback
            or args.adaptive_block_after_full_accept
        ):
            try:
                target_graph = session.capture_decode_graph(
                    position=seq_position,
                    steps_per_replay=target_graph_steps_per_replay,
                    max_replay_steps=target_graph_max_replay_steps,
                    record_steps=target_graph_max_replay_steps if args.target_graph_batched_verify else 0,
                    attention_max_context_len=target_graph_context_cap,
                    capture_hidden_seed_fp32=True,
                    record_hidden_seeds=bool(args.target_graph_batched_verify),
                )
                target_graph_verify_enabled = True
            except Exception as exc:  # pragma: no cover - graph capture failures depend on runtime state
                target_graph_verify_fallback_reason = f"{type(exc).__name__}: {exc}"
                target_graph = None

        total_drafts = 0
        total_accepted = 0
        total_output_tokens = 0
        cycle_details = []
        decode_times = []
        previous_cycle_accepted = 0
        adaptive_draft_n_max = (
            int(args.adaptive_probe_draft_n_max)
            if args.adaptive_block_after_full_accept
            else int(args.draft_n_max)
        )
        adaptive_ar_fallback_active = False
        adaptive_ar_fallback_cooldown_left = 0
        adaptive_full_vocab_recovery_active = False
        adaptive_block_verify_active = not bool(args.adaptive_block_after_full_accept)
        adaptive_strict_probe_history: list[int] = []
        adaptive_strict_probe_decision: str | None = None

        mtp_device_key_cache = None
        mtp_device_value_cache = None
        mtp_device_kv_len = 0
        mtp_device_kv_capacity = 0
        mtp_initial_kv_writer = "none"
        if args.mtp_device_kv_cache:
            qk_head_dim = int(np.asarray(get("blk.40.attn_q_norm.weight")).shape[0])
            kv_heads = 2
            value_head_dim = qk_head_dim
            # Prompt replay rows + one draft-start row per cycle + accepted
            # verifier rows.  Keep enough guard space for rejected draft rows
            # written during speculative probing before rollback.
            required_mtp_device_kv_capacity = max(
                1,
                len(mtp_context_tokens)
                + int(args.cycles) * (2 * int(args.draft_n_max) + 2)
                + 4,
            )
            if args.native_spec_proposal_graph and resident_draft is not None:
                (
                    mtp_device_key_cache,
                    mtp_device_value_cache,
                    mtp_device_kv_capacity,
                ) = resident_draft.ensure_native_proposal_kv_capacity(
                    required_mtp_device_kv_capacity
                )
            else:
                mtp_device_kv_capacity = required_mtp_device_kv_capacity
                key_nbytes = mtp_device_kv_capacity * kv_heads * qk_head_dim * 4
                value_nbytes = mtp_device_kv_capacity * kv_heads * value_head_dim * 4
                mtp_device_key_cache = malloc(key_nbytes, runtime=runtime)
                mtp_device_value_cache = malloc(value_nbytes, runtime=runtime)
                mtp_device_kv_buffers.extend([mtp_device_key_cache, mtp_device_value_cache])
            if len(mtp_context_tokens) > 0:
                context_positions = np.asarray(mtp_context_positions, dtype=np.int64)
                context_tokens = np.asarray(mtp_context_tokens, dtype=np.int64)
                if resident_draft is not None:
                    mtp_device_kv_len = resident_draft.write_kv_rows(
                        mtp_context_hidden_rows,
                        context_tokens,
                        positions=context_positions,
                        rope_cos=_rope_cos,
                        rope_sin=_rope_sin,
                        dense_key_cache=mtp_device_key_cache,
                        dense_value_cache=mtp_device_value_cache,
                        dense_cache_len=0,
                    )
                    mtp_initial_kv_writer = "resident_write_kv_rows"
                else:
                    _ = run_draft(
                        mtp_context_hidden_rows,
                        token_embd_f32[context_tokens].copy(),
                        positions=context_positions,
                        rope_cos=_rope_cos[context_positions],
                        rope_sin=_rope_sin[context_positions],
                        rotary_dim=rope_dim,
                        dense_key_cache=mtp_device_key_cache,
                        dense_value_cache=mtp_device_value_cache,
                        dense_cache_len=0,
                        kv_write_only=True,
                    )
                    mtp_device_kv_len = len(mtp_context_tokens)
                    mtp_initial_kv_writer = "legacy_run_draft_kv_write_only"

        for cycle in range(args.cycles):
            if int(args.max_output_tokens) > 0 and total_output_tokens >= int(args.max_output_tokens):
                break
            cycle_wall_t0 = time.perf_counter()
            stage_timings_ms: dict[str, float] = {}
            cycle_start_seq_position = int(seq_position)
            remaining_output_tokens = (
                int(args.max_output_tokens) - int(total_output_tokens)
                if int(args.max_output_tokens) > 0
                else 0
            )

            def add_cycle_stage(name: str, ms: float) -> None:
                if not args.record_cycle_stage_timings:
                    return
                if ms < 0.0:
                    raise RuntimeError(f"negative cycle stage timing for {name}: {ms}")
                stage_timings_ms[name] = stage_timings_ms.get(name, 0.0) + float(ms)

            # AR fallback is a permanent latch by default (cooldown 0). With
            # --adaptive-ar-fallback-cooldown N>0 it is recoverable: after N
            # fallback cycles, re-probe one drafting cycle; if it misses again the
            # trigger below re-arms. This keeps drafting on categories with good
            # acceptance (en/mixed/code) that the permanent latch killed after a
            # single hard-token miss, matching llama.cpp's per-cycle p_min policy.
            if adaptive_ar_fallback_active and int(args.adaptive_ar_fallback_cooldown) > 0:
                if adaptive_ar_fallback_cooldown_left > 0:
                    adaptive_ar_fallback_cooldown_left -= 1
                    cycle_ar_fallback = True
                else:
                    adaptive_ar_fallback_active = False
                    cycle_ar_fallback = False
            else:
                cycle_ar_fallback = bool(adaptive_ar_fallback_active)
            if args.adaptive_strict_block_probe:
                if adaptive_strict_probe_decision == "fallback":
                    cycle_policy = "strict_probe_fallback"
                    cycle_root_topk_accept = int(args.adaptive_strict_fallback_root_topk)
                    cycle_sibling_topk_accept = 1
                    cycle_draft_window = int(args.adaptive_strict_fallback_draft_n_max)
                    cycle_block_verify_allowed = False
                else:
                    cycle_policy = (
                        "strict_probe_keep"
                        if adaptive_strict_probe_decision == "strict"
                        else "strict_probe"
                    )
                    cycle_root_topk_accept = 1
                    cycle_sibling_topk_accept = 1
                    cycle_draft_window = int(adaptive_draft_n_max)
                    cycle_block_verify_allowed = bool(adaptive_block_verify_active) and not cycle_ar_fallback
            else:
                cycle_policy = "default"
                cycle_root_topk_accept = int(args.root_topk_accept)
                cycle_sibling_topk_accept = int(args.sibling_topk_accept)
                cycle_draft_window = int(adaptive_draft_n_max)
                cycle_block_verify_allowed = bool(adaptive_block_verify_active) and not cycle_ar_fallback
            cycle_fused_b1_block_probe = should_use_fused_b1_block_probe(
                fused_b1_block_probe=bool(args.fused_b1_block_probe),
                adaptive_block_after_full_accept=bool(args.adaptive_block_after_full_accept),
                adaptive_block_verify_active=bool(adaptive_block_verify_active),
                cycle_ar_fallback=bool(cycle_ar_fallback),
                cycle_draft_window=int(cycle_draft_window),
                adaptive_probe_draft_n_max=int(args.adaptive_probe_draft_n_max),
                cycle_root_topk_accept=int(cycle_root_topk_accept),
                cycle_sibling_topk_accept=int(cycle_sibling_topk_accept),
                topk_branch_redraft=bool(args.topk_branch_redraft),
            )
            if cycle_fused_b1_block_probe:
                cycle_block_verify_allowed = True
                if cycle_policy == "default":
                    cycle_policy = "fused_b1_block_probe"
            cycle_draft_n_max = 0 if cycle_ar_fallback else int(cycle_draft_window)
            if int(args.max_output_tokens) > 0:
                # llama.cpp server-side MTP leaves one token of budget for the
                # target corrective row, so the last speculative cycle may draft
                # fewer than the configured n_max. The corrective row can still
                # make the request finish slightly above the nominal token cap
                # after accepted speculative tokens are emitted.
                cycle_draft_n_max = max(
                    0,
                    min(int(cycle_draft_n_max), int(remaining_output_tokens) - 1),
                )
            cycle_topk_candidate_count = max(1, cycle_root_topk_accept, cycle_sibling_topk_accept)
            cycle_diagnostic_topk_candidate_count = _diagnostic_topk_candidate_count(
                args,
                cycle_topk_candidate_count,
            )
            cycle_full_vocab_recovery = (
                bool(adaptive_full_vocab_recovery_active)
                and resident_draft_full_vocab is not None
                and not cycle_ar_fallback
            )
            cycle_resident_draft = resident_draft_full_vocab if cycle_full_vocab_recovery else resident_draft
            cycle_draft_vocab_cap = int(cycle_resident_draft.vocab) if cycle_resident_draft is not None else (
                int(args.mtp_draft_vocab_cap or sh_raw.shape[0]) if cycle_draft_n_max > 0 else 0
            )
            cycle_prev_token = int(prev_token)
            cycle_pending_hidden_seed = np.ascontiguousarray(pending_hidden_seed, dtype=np.float32).copy()
            draft_hidden_state_trace: list[dict[str, object]] = []
            if args.record_draft_hidden_stats:
                draft_hidden_state_trace.append(
                    hidden_state_summary(
                        cycle_pending_hidden_seed,
                        label="cycle_pending_hidden_seed",
                        depth=-1,
                        token_id=cycle_prev_token,
                        position=int(seq_position),
                    )
                )

            t2 = time.perf_counter()
            draft_tokens = []
            draft_top10_tokens = []
            draft_topk_scores: list[list[float]] = []
            draft_topk_margins: list[list[float]] = []
            draft_diagnostic_logits: list[tuple[int, int, np.ndarray]] = []
            cycle_draft_top1_probs: list[float] = []
            replay_tokens = [cycle_prev_token]
            # Sequential single-seed draft path.  Context replay and the normal
            # path both use this shape; llama.cpp parity comes from the optional
            # device-resident MTP KV cache rather than Python batch replay.
            current_hidden_seed = cycle_pending_hidden_seed
            current_token = cycle_prev_token
            current_pos = seq_position
            cycle_mtp_kv_base_len = int(mtp_device_kv_len)
            native_complete_cycle_result = None
            if cycle_resident_draft is not None and cycle_draft_n_max > 0:
                if args.mtp_device_kv_cache and mtp_device_kv_len + cycle_draft_n_max > mtp_device_kv_capacity:
                    raise RuntimeError("MTP device KV cache capacity exhausted")
                if args.native_spec_complete_cycle:
                    if resident_context is None or resident_context.pending_seed is None:
                        raise RuntimeError("native complete cycle requires a resident MTP context")
                    if mtp_device_key_cache is None or mtp_device_value_cache is None:
                        raise RuntimeError("native complete cycle requires resident MTP K/V buffers")
                    native_complete_cycle_result = session.run_native_spec_mtp_cycle(
                        cycle_resident_draft,
                        resident_context,
                        root_token=current_token,
                        root_position=current_pos,
                        candidate_budget=cycle_draft_n_max,
                        remaining_decode=remaining_output_tokens,
                        rope_cos=_rope_cos,
                        rope_sin=_rope_sin,
                        draft_key_cache=mtp_device_key_cache,
                        draft_value_cache=mtp_device_value_cache,
                        draft_cache_len=mtp_device_kv_len,
                        cycle_id=int(cycle),
                        transaction_id=int(cycle),
                        record_stage_timings=bool(args.record_cycle_stage_timings),
                        native_proposal_graph=bool(args.native_spec_proposal_graph),
                    )
                    draft_tokens = [int(token) for token in native_complete_cycle_result.draft_token_ids]
                    draft_top10_tokens = [[int(token)] for token in draft_tokens]
                    mtp_device_kv_len = int(native_complete_cycle_result.draft_cache_len_after)
                elif args.resident_mtp_device_seed:
                    if resident_context is None or resident_context.pending_seed is None:
                        raise RuntimeError("resident MTP context has no pending seed")
                    draft_tokens, draft_top10_tokens, mtp_device_kv_len = (
                        cycle_resident_draft.propose_chain_from_device_seed(
                            int(resident_context.pending_seed.hidden_ptr),
                            start_token=current_token,
                            start_position=current_pos,
                            draft_n_max=cycle_draft_n_max,
                            top_k=min(cycle_diagnostic_topk_candidate_count, 64),
                            rope_cos=_rope_cos,
                            rope_sin=_rope_sin,
                            dense_key_cache=mtp_device_key_cache if args.mtp_device_kv_cache else None,
                            dense_value_cache=mtp_device_value_cache if args.mtp_device_kv_cache else None,
                            dense_cache_len=mtp_device_kv_len,
                            draft_p_min=float(args.draft_p_min),
                            record_top1_probs=bool(args.record_draft_confidence),
                            record_topk_scores=bool(args.record_draft_topk_scores),
                            record_hidden_stats=bool(args.record_draft_hidden_stats),
                            record_stage_stats=bool(args.record_draft_stage_stats),
                            record_cache_rows=tuple(args.record_draft_cache_rows),
                            record_attention_debug=bool(args.record_draft_attention_debug),
                            record_stage_timings=bool(args.record_cycle_stage_timings),
                        )
                    )
                else:
                    draft_tokens, draft_top10_tokens, mtp_device_kv_len = cycle_resident_draft.propose_chain(
                        current_hidden_seed,
                        start_token=current_token,
                        start_position=current_pos,
                        draft_n_max=cycle_draft_n_max,
                        top_k=min(cycle_diagnostic_topk_candidate_count, 64),
                        rope_cos=_rope_cos,
                        rope_sin=_rope_sin,
                        dense_key_cache=mtp_device_key_cache if args.mtp_device_kv_cache else None,
                        dense_value_cache=mtp_device_value_cache if args.mtp_device_kv_cache else None,
                        dense_cache_len=mtp_device_kv_len,
                        draft_p_min=float(args.draft_p_min),
                        record_top1_probs=bool(args.record_draft_confidence),
                        record_topk_scores=bool(args.record_draft_topk_scores),
                        record_hidden_stats=bool(args.record_draft_hidden_stats),
                        record_stage_stats=bool(args.record_draft_stage_stats),
                        record_cache_rows=tuple(args.record_draft_cache_rows),
                        record_attention_debug=bool(args.record_draft_attention_debug),
                        record_stage_timings=bool(args.record_cycle_stage_timings),
                    )
                if args.record_draft_confidence:
                    cycle_draft_top1_probs = [float(value) for value in cycle_resident_draft.last_top1_probs]
                if args.record_draft_topk_scores:
                    draft_topk_scores = [
                        [float(value) for value in row]
                        for row in getattr(cycle_resident_draft, "last_topk_scores", [])
                    ]
                    draft_topk_margins = [
                        [float(value) for value in row]
                        for row in getattr(cycle_resident_draft, "last_topk_margins", [])
                    ]
                if args.record_draft_hidden_stats or args.record_draft_stage_stats:
                    draft_hidden_state_trace.extend(
                        dict(item)
                        for item in getattr(cycle_resident_draft, "last_hidden_state_summaries", [])
                    )
            elif cycle_draft_n_max > 0:
                for draft_depth in range(cycle_draft_n_max):
                    token_embed = token_embd_f32[current_token:current_token + 1].copy()
                    need_next_seed = draft_depth + 1 < cycle_draft_n_max
                    pos_arr = np.asarray([current_pos], dtype=np.int64)
                    rope_cos_slice = _rope_cos[pos_arr]
                    rope_sin_slice = _rope_sin[pos_arr]
                    kv_kwargs = {}
                    if args.mtp_device_kv_cache:
                        if mtp_device_kv_len >= mtp_device_kv_capacity:
                            raise RuntimeError("MTP device KV cache capacity exhausted")
                        kv_kwargs = {
                            "dense_key_cache": mtp_device_key_cache,
                            "dense_value_cache": mtp_device_value_cache,
                            "dense_cache_len": mtp_device_kv_len,
                        }
                    if need_next_seed:
                        draft_logits, current_hidden_seed = run_draft(
                            current_hidden_seed,
                            token_embed,
                            return_hidden_seed=True,
                            positions=pos_arr,
                            rope_cos=rope_cos_slice,
                            rope_sin=rope_sin_slice,
                            rotary_dim=rope_dim,
                            **kv_kwargs,
                        )
                    else:
                        draft_logits = run_draft(
                            current_hidden_seed,
                            token_embed,
                            positions=pos_arr,
                            rope_cos=rope_cos_slice,
                            rope_sin=rope_sin_slice,
                            rotary_dim=rope_dim,
                            **kv_kwargs,
                        )
                    if args.mtp_device_kv_cache:
                        mtp_device_kv_len += 1
                    draft_logits_row = draft_logits[0]
                    draft_token, top10_tokens = select_topk_tokens(
                        draft_logits_row, k=cycle_topk_candidate_count, draft_depth=draft_depth
                    )
                    draft_tokens.append(draft_token)
                    draft_top10_tokens.append(top10_tokens)
                    if args.record_draft_topk_scores:
                        scores = draft_topk_scores_from_logits(draft_logits_row, top10_tokens)
                        draft_topk_scores.append(scores)
                        draft_topk_margins.append(draft_topk_margins_from_scores(scores))
                    if cycle_diagnostic_topk_candidate_count > cycle_topk_candidate_count:
                        draft_diagnostic_logits.append(
                            (len(draft_top10_tokens) - 1, draft_depth, draft_logits_row)
                        )
                    current_token = draft_token
                    current_pos += 1
                    if args.draft_p_min > 0.0 and draft_depth + 1 < cycle_draft_n_max:
                        if _draft_top1_prob(draft_logits[0]) < args.draft_p_min:
                            break
            t3 = time.perf_counter()
            draft_ms = (
                float(native_complete_cycle_result.proposal_wall_ms)
                if native_complete_cycle_result is not None
                else (t3 - t2) * 1000
            )
            add_cycle_stage("draft_initial", draft_ms)
            if native_complete_cycle_result is not None:
                add_cycle_stage(
                    "native_spec_complete_cycle",
                    float(native_complete_cycle_result.call_wall_ms),
                )
            if cycle_resident_draft is not None:
                for stage_name, stage_ms in sorted(cycle_resident_draft.last_stage_timings_ms.items()):
                    add_cycle_stage(stage_name, stage_ms)
            if cycle_diagnostic_topk_candidate_count > cycle_topk_candidate_count:
                t_diag_topk0 = time.perf_counter()
                for (
                    topk_row_index,
                    diagnostic_depth,
                    diagnostic_logits_row,
                ) in draft_diagnostic_logits:
                    _, diagnostic_top10_tokens = select_topk_tokens(
                        diagnostic_logits_row,
                        k=cycle_diagnostic_topk_candidate_count,
                        draft_depth=diagnostic_depth,
                    )
                    draft_top10_tokens[topk_row_index] = diagnostic_top10_tokens
                    if args.record_draft_topk_scores:
                        scores = draft_topk_scores_from_logits(diagnostic_logits_row, diagnostic_top10_tokens)
                        margins = draft_topk_margins_from_scores(scores)
                        if topk_row_index < len(draft_topk_scores):
                            draft_topk_scores[topk_row_index] = scores
                            draft_topk_margins[topk_row_index] = margins
                        else:
                            draft_topk_scores.append(scores)
                            draft_topk_margins.append(margins)
                add_cycle_stage("draft_diagnostic_topk", (time.perf_counter() - t_diag_topk0) * 1000)

            # Verify/account with llama.cpp semantics. The target evaluates the
            # sampled token plus accepted draft prefix and returns one final
            # corrective target token. Output tokens are therefore
            # accepted_drafts + 1, and accept(n) re-seeds pending_hidden_seed from
            # the hidden row at the accepted-prefix boundary.
            ar_decode_ms = 0.0
            target_tokens = []
            target_hidden_seeds = []
            target_verify_seed_rows = []
            verify_input_token = cycle_prev_token
            topk_branch_accepted = False
            topk_branch_depth: int | None = None
            topk_branch_depths: list[int] = []
            topk_branch_accept_count = 0
            pending_branch_redraft = False
            stop_after_branch_corrective = False
            redraft_tokens: list[int] = []
            redraft_top10_tokens: list[list[int]] = []
            redraft_tokens_by_depth: list[int] = list(draft_tokens)
            redraft_top10_by_depth: list[list[int]] = list(draft_top10_tokens)
            redraft_start_depth: int | None = None
            redraft_ms = 0.0
            batched_verify_used = False
            block_verify_used = False
            native_spec_target_cycle_used = False
            native_spec_device_accept_commit_used = False
            serial_hidden_host_required = not bool(args.resident_mtp_device_seed)
            device_verify_rows_required = bool(args.resident_mtp_device_seed and args.mtp_device_kv_cache)
            b1_branch_safe_block_verify_used = False
            target_verify_layer_passes = 0
            target_verify_rows_evaluated = 0
            target_verify_serial_rows = 0
            target_verify_graph_rows = 0
            target_verify_block_passes = 0
            target_verify_block_rows = 0
            target_verify_replay_rows = 0
            target_verify_direct_commit_rows = 0
            target_verify_discarded_rows = 0
            target_block_raw_tokens: list[int] = []
            target_block_consumed_rows: int | None = None
            target_block_direct_commit_exact: bool | None = None
            target_block_direct_state_commit_effective = False
            target_lm_head_score_rows: list[dict[str, object]] = []
            target_lm_head_logits_available = False

            def record_target_verify(
                rows: int,
                *,
                layer_passes: int | None = None,
                serial_rows: int = 0,
                graph_rows: int = 0,
                block_passes: int = 0,
                block_rows: int = 0,
                replay_rows: int = 0,
                discarded_rows: int = 0,
            ) -> None:
                nonlocal target_verify_layer_passes
                nonlocal target_verify_rows_evaluated
                nonlocal target_verify_serial_rows
                nonlocal target_verify_graph_rows
                nonlocal target_verify_block_passes
                nonlocal target_verify_block_rows
                nonlocal target_verify_replay_rows
                nonlocal target_verify_discarded_rows
                rows = int(rows)
                if rows < 0:
                    raise ValueError("target verifier rows must be non-negative")
                if layer_passes is None:
                    layer_passes = rows
                target_verify_layer_passes += int(layer_passes)
                target_verify_rows_evaluated += rows
                target_verify_serial_rows += int(serial_rows)
                target_verify_graph_rows += int(graph_rows)
                target_verify_block_passes += int(block_passes)
                target_verify_block_rows += int(block_rows)
                target_verify_replay_rows += int(replay_rows)
                target_verify_discarded_rows += int(discarded_rows)

            def record_direct_commit(rows: int = 1) -> None:
                nonlocal target_verify_direct_commit_rows
                target_verify_direct_commit_rows += int(rows)

            def record_native_spec_target_timings() -> None:
                if not session.last_native_spec_target_submitted:
                    return
                add_cycle_stage(
                    "native_spec_target_capture",
                    float(session.last_native_spec_target_capture_ms),
                )
                add_cycle_stage(
                    "native_spec_target_submit",
                    float(session.last_native_spec_target_submit_ms),
                )
                add_cycle_stage(
                    "native_spec_target_readback",
                    float(session.last_native_spec_target_readback_ms),
                )

            def record_target_lm_head_scores(
                block_result,
                block_inputs: list[int],
                block_target_tokens: list[int],
            ) -> None:
                nonlocal target_lm_head_score_rows
                nonlocal target_lm_head_logits_available
                if not bool(args.record_target_topk_scores):
                    return
                rows = _target_lm_head_score_rows(
                    getattr(block_result, "lm_head_logits_f32", None),
                    input_tokens=[int(token) for token in block_inputs],
                    target_tokens=[int(token) for token in block_target_tokens],
                    draft_tokens=[int(token) for token in draft_tokens],
                    extra_candidate_tokens=tuple(args.target_score_candidate_tokens),
                    top_k=int(args.target_topk_score_count),
                )
                target_lm_head_score_rows = rows
                target_lm_head_logits_available = bool(rows)

            can_b1_branch_safe_block_verify = (
                bool(args.target_b1_branch_safe_block_verify)
                and bool(args.target_block_verify)
                and cycle_block_verify_allowed
                and len(draft_tokens) == 1
                and int(cycle_draft_n_max) == 1
                and cycle_root_topk_accept > 1
                and cycle_sibling_topk_accept == 1
                and not args.topk_branch_redraft
                and int(verify_input_token) == current_device_token
            )
            if can_b1_branch_safe_block_verify:
                t0 = time.perf_counter()
                replay_state_commit = bool(args.target_block_replay_state_commit)
                direct_state_commit = bool(args.target_block_direct_state_commit) and not bool(
                    replay_state_commit
                )
                target_block_direct_state_commit_effective = bool(direct_state_commit)
                block_inputs = [int(verify_input_token), int(draft_tokens[0])]
                direct_state_commit_exact_mode = target_block_direct_commit_is_exact(
                    args.target_block_verify_mode,
                    start_position=seq_position,
                    rows=len(block_inputs),
                )
                direct_partial_commit_exact_mode = target_block_direct_partial_commit_is_exact(
                    args.target_block_verify_mode
                )
                direct_partial_commit_llama_style = target_block_direct_partial_commit_is_llama_style(
                    args.target_block_verify_mode,
                    args.target_block_direct_partial_replay_mode,
                )
                direct_partial_commit_supported = bool(direct_partial_commit_exact_mode) or bool(
                    direct_partial_commit_llama_style
                )
                target_block_direct_commit_exact = bool(direct_state_commit_exact_mode)
                snapshot = None
                if target_block_needs_linear_state_snapshot(
                    direct_state_commit=direct_state_commit,
                    direct_state_commit_exact_mode=direct_state_commit_exact_mode,
                    direct_partial_commit_exact_mode=direct_partial_commit_exact_mode,
                    direct_partial_commit_llama_style=direct_partial_commit_llama_style,
                ):
                    snapshot = session._linear_state_snapshot()
                try:
                    if args.target_block_verify_mode == "serial-exact":
                        block_result = session.verify_target_block_serial_exact(
                            block_inputs,
                            capture_linear_state_rows=direct_state_commit,
                        )
                        record_target_verify(
                            len(block_inputs),
                            serial_rows=len(block_inputs),
                        )
                    else:
                        if args.native_spec_target_cycle:
                            block_result = session.verify_target_block_native_cycle(
                                block_inputs,
                                cycle_id=int(cycle),
                                bulk_attention_mode=args.target_block_verify_mode,
                                use_wmma_prefill=bool(args.target_block_wmma_prefill),
                                capture_linear_state_rows=direct_state_commit,
                                capture_lm_head_logits=bool(args.record_target_topk_scores),
                                record_stage_timings=bool(args.record_cycle_stage_timings),
                                sync_stage_timings=bool(args.target_block_sync_stage_timings),
                                defer_linear_state_commit=direct_state_commit,
                                device_accept_commit=bool(args.native_spec_device_accept_commit),
                                remaining_decode=(
                                    int(remaining_output_tokens)
                                    if args.native_spec_device_accept_commit
                                    else None
                                ),
                            )
                        else:
                            block_result = session.verify_target_block(
                                block_inputs,
                                bulk_attention_mode=args.target_block_verify_mode,
                                use_wmma_prefill=bool(args.target_block_wmma_prefill),
                                capture_linear_state_rows=direct_state_commit,
                                capture_lm_head_logits=bool(args.record_target_topk_scores),
                                record_stage_timings=bool(args.record_cycle_stage_timings),
                                sync_stage_timings=bool(args.target_block_sync_stage_timings),
                                defer_linear_state_commit=direct_state_commit,
                            )
                        native_spec_target_cycle_used = bool(
                            native_spec_target_cycle_used
                            or session.last_native_spec_target_submitted
                        )
                        record_native_spec_target_timings()
                        record_target_verify(
                            len(block_inputs),
                            layer_passes=1,
                            graph_rows=(
                                len(block_inputs)
                                if session.last_native_spec_target_submitted
                                else 0
                            ),
                            block_passes=1,
                            block_rows=len(block_inputs),
                        )
                    block_target_tokens = [int(token) for token in block_result.token_ids]
                    target_block_raw_tokens = list(block_target_tokens)
                    record_target_lm_head_scores(block_result, block_inputs, block_target_tokens)
                    if len(block_target_tokens) != 2:
                        raise RuntimeError("B1 branch-safe block verifier expected exactly two target rows")
                    target0 = int(block_target_tokens[0])
                    target_block_consumed_rows = 2 if target0 == int(draft_tokens[0]) else 1
                    if target0 == int(draft_tokens[0]):
                        if direct_state_commit:
                            if not block_result.linear_state_rows_captured:
                                raise RuntimeError("direct B1 branch commit requested without captured linear-state rows")
                            if direct_state_commit_exact_mode:
                                session._commit_verify_linear_state_row(1, position=seq_position + 2)
                                record_direct_commit()
                                target_tokens.extend(block_target_tokens)
                                if serial_hidden_host_required:
                                    target_hidden_seeds.extend(
                                        np.ascontiguousarray(block_result.hidden_seeds[row:row + 1], dtype=np.float32)
                                        for row in range(2)
                                    )
                            else:
                                session._restore_linear_state_snapshot(snapshot, position=seq_position)
                                replay_result = session.verify_target_block_serial_exact(block_inputs)
                                record_target_verify(
                                    len(block_inputs),
                                    serial_rows=len(block_inputs),
                                    replay_rows=len(block_inputs),
                                )
                                replay_tokens = [int(token) for token in replay_result.token_ids]
                                if replay_tokens != block_target_tokens:
                                    raise RuntimeError("B1 branch-safe serial-exact replay diverged from block rows")
                                target_tokens.extend(replay_tokens)
                                if serial_hidden_host_required:
                                    target_hidden_seeds.extend(
                                        np.ascontiguousarray(replay_result.hidden_seeds[row:row + 1], dtype=np.float32)
                                        for row in range(2)
                                    )
                        elif replay_state_commit:
                            if snapshot is None:
                                raise RuntimeError("B1 branch-safe state replay requested without a linear-state snapshot")
                            session._restore_linear_state_snapshot(snapshot, position=seq_position)
                            replay_result = session.verify_target_block_serial_exact(block_inputs)
                            record_target_verify(
                                len(block_inputs),
                                serial_rows=len(block_inputs),
                                replay_rows=len(block_inputs),
                            )
                            target_tokens.extend(block_target_tokens)
                            if serial_hidden_host_required:
                                target_hidden_seeds.extend(
                                    np.ascontiguousarray(replay_result.hidden_seeds[row:row + 1], dtype=np.float32)
                                    for row in range(2)
                                )
                        else:
                            target_tokens.extend(block_target_tokens)
                            if serial_hidden_host_required:
                                target_hidden_seeds.extend(
                                    np.ascontiguousarray(block_result.hidden_seeds[row:row + 1], dtype=np.float32)
                                    for row in range(2)
                                )
                        current_device_token = int(block_target_tokens[1])
                    else:
                        record_target_verify(0, discarded_rows=1)
                        row0_direct_committed = False
                        if direct_state_commit and direct_partial_commit_supported:
                            if not block_result.linear_state_rows_captured:
                                raise RuntimeError("direct B1 branch commit requested without captured linear-state rows")
                            session._commit_verify_linear_state_row(0, position=seq_position + 1)
                            record_direct_commit()
                            row0_direct_committed = True
                        else:
                            if snapshot is None:
                                raise RuntimeError("B1 branch-safe row-0 replay requires a linear-state snapshot")
                            session._restore_linear_state_snapshot(snapshot, position=seq_position)
                            replay0 = session.step(
                                int(verify_input_token),
                                return_logits=False,
                                capture_hidden_seed_fp32=True,
                            )
                            record_target_verify(1, serial_rows=1, replay_rows=1)
                            if (not replay_state_commit) and int(replay0.token_id) != target0:
                                raise RuntimeError("B1 branch-safe row-0 replay diverged from block row 0")
                        target_tokens.append(target0)
                        current_device_token = target0
                        if serial_hidden_host_required:
                            if row0_direct_committed:
                                target_hidden_seeds.append(
                                    np.ascontiguousarray(block_result.hidden_seeds[0:1], dtype=np.float32)
                                )
                            else:
                                target_hidden_seeds.append(copy_pending_hidden_seed())
                        root_topk_tokens = (
                            [int(token) for token in draft_top10_tokens[0][:cycle_root_topk_accept]]
                            if draft_top10_tokens
                            else []
                        )
                        if target0 in root_topk_tokens:
                            topk_branch_accepted = True
                            topk_branch_depth = 0
                            topk_branch_depths.append(0)
                            topk_branch_accept_count = 1
                            target_result = session.step(
                                target0,
                                return_logits=False,
                                capture_hidden_seed_fp32=True,
                            )
                            record_target_verify(1, serial_rows=1)
                            corrective = int(target_result.token_id)
                            current_device_token = corrective
                            target_tokens.append(corrective)
                            if serial_hidden_host_required:
                                target_hidden_seeds.append(copy_pending_hidden_seed())
                        # Else row 0 is the visible corrective token after a reject;
                        # row 1 was computed from an unaccepted draft and is ignored.
                finally:
                    if snapshot is not None:
                        session._free_linear_state_snapshot(snapshot)
                block_verify_used = True
                b1_branch_safe_block_verify_used = True
                elapsed_ms = (time.perf_counter() - t0) * 1000
                add_cycle_stage("target_b1_branch_block_verify_total", elapsed_ms)
                ar_decode_ms += elapsed_ms
            can_block_verify = (
                bool(args.target_block_verify)
                and cycle_block_verify_allowed
                and len(draft_tokens) > 0
                and cycle_root_topk_accept == 1
                and cycle_sibling_topk_accept == 1
                and not args.topk_branch_redraft
                and len(draft_tokens) + 1 >= (
                    int(args.target_block_min_rows)
                    if int(args.target_block_min_rows) > 0
                    else int(session.runner.weights.config.ssm_conv_kernel)
                )
                and int(verify_input_token) == current_device_token
            )
            if can_block_verify:
                t0 = time.perf_counter()
                replay_state_commit = bool(args.target_block_replay_state_commit)
                direct_state_commit = bool(args.target_block_direct_state_commit) and not bool(
                    replay_state_commit
                )
                target_block_direct_state_commit_effective = bool(direct_state_commit)
                block_inputs = [int(verify_input_token)] + [int(token) for token in draft_tokens]
                direct_state_commit_exact_mode = target_block_direct_commit_is_exact(
                    args.target_block_verify_mode,
                    start_position=seq_position,
                    rows=len(block_inputs),
                )
                direct_partial_commit_exact_mode = target_block_direct_partial_commit_is_exact(
                    args.target_block_verify_mode
                )
                direct_partial_commit_llama_style = target_block_direct_partial_commit_is_llama_style(
                    args.target_block_verify_mode,
                    args.target_block_direct_partial_replay_mode,
                )
                direct_partial_commit_supported = bool(direct_partial_commit_exact_mode) or bool(
                    direct_partial_commit_llama_style
                )
                target_block_direct_commit_exact = bool(direct_state_commit_exact_mode)
                snapshot = None
                if native_complete_cycle_result is None and target_block_needs_linear_state_snapshot(
                    direct_state_commit=direct_state_commit,
                    direct_state_commit_exact_mode=direct_state_commit_exact_mode,
                    direct_partial_commit_exact_mode=direct_partial_commit_exact_mode,
                    direct_partial_commit_llama_style=direct_partial_commit_llama_style,
                ):
                    t_snapshot0 = time.perf_counter()
                    snapshot = session._linear_state_snapshot()
                    add_cycle_stage("target_block_snapshot", (time.perf_counter() - t_snapshot0) * 1000)
                try:
                    t_forward0 = time.perf_counter()
                    if native_complete_cycle_result is not None:
                        block_result = native_complete_cycle_result.target_result
                        native_spec_target_cycle_used = True
                        record_native_spec_target_timings()
                        record_target_verify(
                            len(block_inputs),
                            layer_passes=1,
                            graph_rows=len(block_inputs),
                            block_passes=1,
                            block_rows=len(block_inputs),
                        )
                    elif args.target_block_verify_mode == "serial-exact":
                        block_result = session.verify_target_block_serial_exact(
                            block_inputs,
                            capture_linear_state_rows=direct_state_commit,
                        )
                        record_target_verify(
                            len(block_inputs),
                            serial_rows=len(block_inputs),
                        )
                    else:
                        if args.native_spec_target_cycle:
                            block_result = session.verify_target_block_native_cycle(
                                block_inputs,
                                cycle_id=int(cycle),
                                bulk_attention_mode=args.target_block_verify_mode,
                                use_wmma_prefill=bool(args.target_block_wmma_prefill),
                                capture_linear_state_rows=direct_state_commit,
                                capture_lm_head_logits=bool(args.record_target_topk_scores),
                                record_stage_timings=bool(args.record_cycle_stage_timings),
                                sync_stage_timings=bool(args.target_block_sync_stage_timings),
                                defer_linear_state_commit=direct_state_commit,
                                device_accept_commit=bool(args.native_spec_device_accept_commit),
                                remaining_decode=(
                                    int(remaining_output_tokens)
                                    if args.native_spec_device_accept_commit
                                    else None
                                ),
                            )
                        else:
                            block_result = session.verify_target_block(
                                block_inputs,
                                bulk_attention_mode=args.target_block_verify_mode,
                                use_wmma_prefill=bool(args.target_block_wmma_prefill),
                                capture_linear_state_rows=direct_state_commit,
                                capture_lm_head_logits=bool(args.record_target_topk_scores),
                                record_stage_timings=bool(args.record_cycle_stage_timings),
                                sync_stage_timings=bool(args.target_block_sync_stage_timings),
                                defer_linear_state_commit=direct_state_commit,
                            )
                        native_spec_target_cycle_used = bool(
                            native_spec_target_cycle_used
                            or session.last_native_spec_target_submitted
                        )
                        record_native_spec_target_timings()
                        record_target_verify(
                            len(block_inputs),
                            layer_passes=1,
                            graph_rows=(
                                len(block_inputs)
                                if session.last_native_spec_target_submitted
                                else 0
                            ),
                            block_passes=1,
                            block_rows=len(block_inputs),
                        )
                    add_cycle_stage(
                        "target_block_forward",
                        (
                            float(native_complete_cycle_result.target_wall_ms)
                            if native_complete_cycle_result is not None
                            else (time.perf_counter() - t_forward0) * 1000
                        ),
                    )
                    for stage_name, stage_ms in sorted(session.last_verify_stage_timings_ms.items()):
                        add_cycle_stage(stage_name, stage_ms)
                    t_account0 = time.perf_counter()
                    block_target_tokens = [int(token) for token in block_result.token_ids]
                    target_block_raw_tokens = list(block_target_tokens)
                    record_target_lm_head_scores(block_result, block_inputs, block_target_tokens)
                    block_acceptance = llama_cpp_acceptance_from_target_samples(draft_tokens, block_target_tokens)
                    if int(block_acceptance["accepted_draft_tokens"]) == 0 and cycle_root_topk_accept > 1:
                        root_branch_acceptance = root_topk_acceptance_from_target_samples(
                            draft_tokens,
                            draft_top10_tokens,
                            block_target_tokens,
                            root_topk_accept=cycle_root_topk_accept,
                        )
                        if root_branch_acceptance is not None:
                            block_acceptance = root_branch_acceptance
                            topk_branch_accepted = True
                            topk_branch_depth = 0
                            topk_branch_depths.append(0)
                            topk_branch_accept_count = 1
                    consumed_rows = int(block_acceptance["accepted_draft_tokens"]) + 1
                    device_accept_commit_applied = bool(
                        getattr(block_result, "device_accept_commit", False)
                    )
                    native_spec_device_accept_commit_used = bool(
                        native_spec_device_accept_commit_used or device_accept_commit_applied
                    )
                    if device_accept_commit_applied:
                        if not direct_state_commit:
                            raise RuntimeError("N2 device accept/commit requires direct state commit policy")
                        if int(block_result.accepted_draft_tokens) != consumed_rows - 1:
                            raise RuntimeError("N2 device acceptance diverged from host oracle")
                        if int(block_result.end_position) != seq_position + consumed_rows:
                            raise RuntimeError("N2 device cursor diverged from accepted target prefix")
                    target_block_consumed_rows = int(consumed_rows)
                    if consumed_rows < len(block_inputs):
                        record_target_verify(0, discarded_rows=len(block_inputs) - consumed_rows)
                    add_cycle_stage("target_block_acceptance_accounting", (time.perf_counter() - t_account0) * 1000)
                    t_commit0 = time.perf_counter()
                    if consumed_rows < len(block_inputs):
                        replay_tokens: list[int]
                        replay_hidden: list[np.ndarray]
                        if direct_state_commit and direct_partial_commit_supported:
                            if not block_result.linear_state_rows_captured:
                                raise RuntimeError("direct block commit requested without captured linear-state rows")
                            if not device_accept_commit_applied:
                                session._commit_verify_linear_state_row(
                                    consumed_rows - 1,
                                    position=seq_position + consumed_rows,
                                )
                            record_direct_commit()
                            replay_tokens = [int(token) for token in block_target_tokens[:consumed_rows]]
                            replay_hidden = (
                                []
                                if device_accept_commit_applied
                                else [
                                    np.ascontiguousarray(block_result.hidden_seeds[row:row + 1], dtype=np.float32)
                                    for row in range(len(replay_tokens))
                                ]
                            )
                        else:
                            if snapshot is None:
                                raise RuntimeError("target block replay requested without a linear-state snapshot")
                            session._restore_linear_state_snapshot(snapshot, position=seq_position)
                            if target_block_state_replay_uses_serial_exact(
                                replay_state_commit=replay_state_commit,
                                direct_state_commit=direct_state_commit,
                                verify_mode=args.target_block_verify_mode,
                                direct_partial_replay_mode=args.target_block_direct_partial_replay_mode,
                            ):
                                serial_state_only = (
                                    direct_state_commit
                                    and args.target_block_direct_partial_replay_mode == "serial-state-only"
                                )
                                replay_result = session.verify_target_block_serial_exact(
                                    block_inputs[:consumed_rows],
                                    advance_state_only=serial_state_only,
                                )
                                record_target_verify(
                                    consumed_rows,
                                    serial_rows=consumed_rows,
                                    replay_rows=consumed_rows,
                                )
                            else:
                                # Accepted-prefix replay only needs to advance linear/KV
                                # state; the target tokens are already known from the
                                # full-block pass (deterministic), so skip the replay's
                                # LM-head sampling and reuse block_target_tokens.
                                state_only_mode = (
                                    "native"
                                    if args.target_block_direct_partial_replay_mode == "native-state-only"
                                    else args.target_block_verify_mode
                                )
                                replay_result = session.verify_target_block(
                                    block_inputs[:consumed_rows],
                                    bulk_attention_mode=state_only_mode,
                                    use_wmma_prefill=bool(args.target_block_wmma_prefill),
                                    advance_state_only=True,
                                    record_stage_timings=bool(args.record_cycle_stage_timings),
                                    sync_stage_timings=bool(args.target_block_sync_stage_timings),
                                )
                                record_target_verify(
                                    consumed_rows,
                                    layer_passes=1,
                                    block_passes=1,
                                    block_rows=consumed_rows,
                                    replay_rows=consumed_rows,
                                )
                            replay_tokens = [int(token) for token in block_target_tokens[:consumed_rows]]
                            if (
                                (direct_state_commit or args.target_block_verify_mode == "serial-exact")
                                and not (
                                    direct_state_commit
                                    and args.target_block_direct_partial_replay_mode == "serial-state-only"
                                )
                                and [int(token) for token in replay_result.token_ids] != replay_tokens
                            ):
                                raise RuntimeError("serial-exact accepted-prefix replay diverged from block rows")
                            replay_hidden = [
                                np.ascontiguousarray(replay_result.hidden_seeds[row:row + 1], dtype=np.float32)
                                for row in range(len(replay_tokens))
                            ]
                        target_tokens.extend(replay_tokens)
                        target_hidden_seeds.extend(replay_hidden)
                    else:
                        if direct_state_commit and direct_state_commit_exact_mode:
                            if not block_result.linear_state_rows_captured:
                                raise RuntimeError("direct block commit requested without captured linear-state rows")
                            if not device_accept_commit_applied:
                                session._commit_verify_linear_state_row(
                                    len(block_inputs) - 1,
                                    position=seq_position + len(block_inputs),
                                )
                            record_direct_commit()
                            target_tokens.extend(block_target_tokens)
                            if not device_accept_commit_applied:
                                target_hidden_seeds.extend(
                                    np.ascontiguousarray(block_result.hidden_seeds[row:row + 1], dtype=np.float32)
                                    for row in range(len(block_target_tokens))
                                )
                        elif replay_state_commit:
                            if snapshot is None:
                                raise RuntimeError("target block state replay requested without a linear-state snapshot")
                            session._restore_linear_state_snapshot(snapshot, position=seq_position)
                            replay_result = session.verify_target_block_serial_exact(block_inputs)
                            record_target_verify(
                                len(block_inputs),
                                serial_rows=len(block_inputs),
                                replay_rows=len(block_inputs),
                            )
                            replay_tokens = [int(token) for token in block_target_tokens]
                            if args.target_block_verify_mode == "serial-exact" and [
                                int(token) for token in replay_result.token_ids
                            ] != replay_tokens:
                                raise RuntimeError("serial-exact state replay diverged from block rows")
                            target_tokens.extend(replay_tokens)
                            target_hidden_seeds.extend(
                                np.ascontiguousarray(replay_result.hidden_seeds[row:row + 1], dtype=np.float32)
                                for row in range(len(replay_tokens))
                            )
                        elif direct_state_commit:
                            if snapshot is None:
                                raise RuntimeError("target block serial replay requested without a linear-state snapshot")
                            session._restore_linear_state_snapshot(snapshot, position=seq_position)
                            replay_result = session.verify_target_block_serial_exact(block_inputs)
                            record_target_verify(
                                len(block_inputs),
                                serial_rows=len(block_inputs),
                                replay_rows=len(block_inputs),
                            )
                            replay_tokens = [int(token) for token in replay_result.token_ids]
                            if replay_tokens != block_target_tokens:
                                raise RuntimeError("direct block serial-exact replay diverged from block rows")
                            target_tokens.extend(replay_tokens)
                            target_hidden_seeds.extend(
                                np.ascontiguousarray(replay_result.hidden_seeds[row:row + 1], dtype=np.float32)
                                for row in range(len(replay_tokens))
                            )
                        else:
                            target_tokens.extend(block_target_tokens)
                            target_hidden_seeds.extend(
                                np.ascontiguousarray(block_result.hidden_seeds[row:row + 1], dtype=np.float32)
                                for row in range(len(block_target_tokens))
                            )
                    if device_accept_commit_applied:
                        target_verify_seed_rows.extend(
                            session.mtp_verify_seed(
                                row,
                                token_id=int(block_target_tokens[row]),
                                position=seq_position + row,
                                hidden_seed_base_ptr=int(block_result.hidden_seed_rows_ptr),
                                hidden_seed_row_count=int(block_result.hidden_seed_row_count),
                            )
                            for row in range(consumed_rows)
                        )
                    add_cycle_stage("target_block_replay_or_commit", (time.perf_counter() - t_commit0) * 1000)
                    current_device_token = int(target_tokens[-1])
                    block_verify_used = True
                finally:
                    if snapshot is not None:
                        session._free_linear_state_snapshot(snapshot)
                t1 = time.perf_counter()
                elapsed_ms = (
                    float(native_complete_cycle_result.target_wall_ms)
                    if native_complete_cycle_result is not None
                    else (t1 - t0) * 1000
                )
                add_cycle_stage("target_block_verify_total", elapsed_ms)
                ar_decode_ms += elapsed_ms
                if block_verify_used and target_graph is not None:
                    (
                        target_graph,
                        target_graph_verify_enabled,
                        target_graph_verify_fallback_reason,
                    ) = invalidate_target_graph_after_block_verify(target_graph)
            can_batched_verify = (
                bool(args.target_graph_batched_verify)
                and target_graph_verify_enabled
                and target_graph is not None
                and int(verify_input_token) == current_device_token
                and cycle_root_topk_accept == 1
                and cycle_sibling_topk_accept == 1
                and not args.topk_branch_redraft
                and len(draft_tokens) == int(cycle_draft_n_max)
                and int(target_graph.steps_per_replay) == len(draft_tokens) + 1
            )
            if (not block_verify_used) and can_batched_verify:
                t0 = time.perf_counter()
                record_start = int(target_graph.replayed_steps)
                replay_steps = len(draft_tokens) + 1
                target_graph.replay(replay_steps)
                record_target_verify(
                    replay_steps,
                    layer_passes=replay_steps,
                    graph_rows=replay_steps,
                )
                recorded_tokens = target_graph.read_generated_token_ids(record_start + replay_steps)[record_start:record_start + replay_steps]
                recorded_hidden = target_graph.read_generated_hidden_seeds(start=record_start, count=replay_steps)
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000
                add_cycle_stage("target_batched_graph_verify", elapsed_ms)
                ar_decode_ms += elapsed_ms
                target_tokens.extend(int(token) for token in recorded_tokens)
                target_hidden_seeds.extend(
                    np.ascontiguousarray(recorded_hidden[row:row + 1], dtype=np.float32)
                    for row in range(replay_steps)
                )
                current_device_token = int(target_tokens[-1])
                batched_verify_used = True
                if target_tokens[:len(draft_tokens)] != draft_tokens:
                    raise RuntimeError(
                        "batched target graph verify requires full strict draft acceptance; "
                        "rerun without --target-graph-batched-verify for mismatch diagnostics"
                    )
            elif not block_verify_used:
                if target_graph is not None and int(target_graph.steps_per_replay) != 1:
                    target_graph.close()
                    target_graph = None
                    target_graph_verify_enabled = False
                    target_graph_verify_fallback_reason = "batched verify conditions not met; fell back to eager step"
                while True:
                    t0 = time.perf_counter()
                    used_graph_step = False
                    if target_graph_verify_enabled and target_graph is not None and int(verify_input_token) == current_device_token:
                        target_graph.replay(1)
                        record_target_verify(1, layer_passes=1, graph_rows=1)
                        target_result = target_graph.read_sample(return_logits=False)
                        used_graph_step = True
                    else:
                        if target_graph is not None:
                            target_graph.close()
                            target_graph = None
                        if target_graph_verify_enabled and int(verify_input_token) != current_device_token:
                            target_graph_verify_fallback_reason = "verify input diverged from device sample token"
                        target_graph_verify_enabled = False
                        target_result = session.step(
                            verify_input_token,
                            return_logits=False,
                            capture_hidden_seed_fp32=True,
                        )
                        record_target_verify(1, serial_rows=1)
                    t1 = time.perf_counter()
                    elapsed_ms = (t1 - t0) * 1000
                    add_cycle_stage(
                        "target_graph_verify_step" if used_graph_step else "target_serial_verify_step",
                        elapsed_ms,
                    )
                    ar_decode_ms += elapsed_ms
                    target_token = int(target_result.token_id)
                    current_device_token = target_token
                    target_tokens.append(target_token)
                    target_hidden_seed = None
                    if serial_hidden_host_required:
                        target_hidden_seed = copy_pending_hidden_seed()
                        target_hidden_seeds.append(target_hidden_seed)
                    if device_verify_rows_required:
                        target_verify_seed_rows.append(
                            session.stage_current_hidden_seed_as_verify_row(
                                row_index=len(target_tokens) - 1,
                                token_id=target_token,
                                position=seq_position + len(target_tokens) - 1,
                                rows_capacity=cycle_draft_n_max + 1,
                            )
                        )

                    depth = len(target_tokens) - 1
                    if (
                        pending_branch_redraft
                        and args.topk_branch_redraft
                        and depth > 0
                        and depth < cycle_draft_n_max
                    ):
                        t_redraft0 = time.perf_counter()
                        current_hidden_seed = np.ascontiguousarray(target_hidden_seed, dtype=np.float32)
                        current_token = int(target_tokens[depth - 1])
                        current_pos = seq_position + depth
                        remaining_drafts = cycle_draft_n_max - depth
                        branch_redraft_tokens: list[int] = []
                        branch_redraft_top10: list[list[int]] = []
                        for redraft_depth in range(remaining_drafts):
                            token_embed = token_embd_f32[current_token:current_token + 1].copy()
                            need_next_seed = redraft_depth + 1 < remaining_drafts
                            pos_arr = np.asarray([current_pos], dtype=np.int64)
                            rope_cos_slice = _rope_cos[pos_arr]
                            rope_sin_slice = _rope_sin[pos_arr]
                            if need_next_seed:
                                draft_logits, current_hidden_seed = run_draft(
                                    current_hidden_seed,
                                    token_embed,
                                    return_hidden_seed=True,
                                    positions=pos_arr,
                                    rope_cos=rope_cos_slice,
                                    rope_sin=rope_sin_slice,
                                    rotary_dim=rope_dim,
                                )
                            else:
                                draft_logits = run_draft(
                                    current_hidden_seed,
                                    token_embed,
                                    positions=pos_arr,
                                    rope_cos=rope_cos_slice,
                                    rope_sin=rope_sin_slice,
                                    rotary_dim=rope_dim,
                                )
                            redraft_token, redraft_top10 = select_topk_tokens(
                                draft_logits[0],
                                k=cycle_topk_candidate_count,
                                draft_depth=depth + redraft_depth,
                            )
                            branch_redraft_tokens.append(redraft_token)
                            branch_redraft_top10.append(redraft_top10)
                            current_token = redraft_token
                            current_pos += 1
                            if args.draft_p_min > 0.0 and redraft_depth + 1 < remaining_drafts:
                                if _draft_top1_prob(draft_logits[0]) < args.draft_p_min:
                                    break
                        redraft_tokens.extend(branch_redraft_tokens)
                        redraft_top10_tokens.extend(branch_redraft_top10)
                        for redraft_depth, redraft_token in enumerate(branch_redraft_tokens):
                            absolute_depth = depth + redraft_depth
                            if absolute_depth < len(redraft_tokens_by_depth):
                                redraft_tokens_by_depth[absolute_depth] = redraft_token
                            else:
                                redraft_tokens_by_depth.append(redraft_token)
                        for redraft_depth, redraft_top10 in enumerate(branch_redraft_top10):
                            absolute_depth = depth + redraft_depth
                            if absolute_depth < len(redraft_top10_by_depth):
                                redraft_top10_by_depth[absolute_depth] = redraft_top10
                            else:
                                redraft_top10_by_depth.append(redraft_top10)
                        redraft_delta_ms = (time.perf_counter() - t_redraft0) * 1000
                        redraft_ms += redraft_delta_ms
                        add_cycle_stage("draft_branch_redraft", redraft_delta_ms)
                        if redraft_start_depth is None:
                            redraft_start_depth = depth
                        pending_branch_redraft = False
                    if stop_after_branch_corrective:
                        break

                    proposal_tokens = redraft_tokens_by_depth
                    proposal_top10_tokens = redraft_top10_by_depth
                    proposal_depth = depth

                    if proposal_depth < len(proposal_tokens) and target_token == proposal_tokens[proposal_depth]:
                        verify_input_token = target_token
                        continue
                    topk_limit = (
                        cycle_root_topk_accept if depth == 0
                        else cycle_sibling_topk_accept if depth <= args.sibling_topk_max_depth
                        else 1
                    )
                    if (
                        depth == 0
                        and args.root_tail_max_prev_accepted >= 0
                        and topk_limit > 4
                        and previous_cycle_accepted > args.root_tail_max_prev_accepted
                        and proposal_depth < len(proposal_top10_tokens)
                        and target_token in proposal_top10_tokens[proposal_depth][4:topk_limit]
                    ):
                        topk_limit = 4
                    if (
                        depth > 0
                        and args.sibling_tail_min_prev_accepted >= 0
                        and topk_limit > 8
                        and previous_cycle_accepted < args.sibling_tail_min_prev_accepted
                    ):
                        topk_limit = 8
                    can_accept_topk_branch = (
                        proposal_depth < len(proposal_top10_tokens)
                        and topk_limit > 1
                        and target_token in proposal_top10_tokens[proposal_depth][:topk_limit]
                        and (
                            not topk_branch_accepted
                            or (
                                args.topk_branch_redraft
                                and topk_branch_accept_count < args.topk_branch_redraft_max_branches
                            )
                        )
                    )
                    if can_accept_topk_branch:
                        topk_branch_accepted = True
                        topk_branch_accept_count += 1
                        if topk_branch_depth is None:
                            topk_branch_depth = depth
                        topk_branch_depths.append(depth)
                        if depth < len(redraft_tokens_by_depth):
                            redraft_tokens_by_depth[depth] = target_token
                        else:
                            redraft_tokens_by_depth.append(target_token)
                        can_redraft_after_branch = (
                            args.topk_branch_redraft
                            and topk_branch_accept_count < args.topk_branch_redraft_max_branches
                            and depth + 1 < cycle_draft_n_max
                        )
                        pending_branch_redraft = can_redraft_after_branch
                        stop_after_branch_corrective = not can_redraft_after_branch
                        verify_input_token = target_token
                        continue
                    break

            t_accept_policy0 = time.perf_counter()
            draft_ms += redraft_ms
            if redraft_start_depth is not None:
                accepted_draft_tokens_for_redraft = len(target_tokens) - 1
                if accepted_draft_tokens_for_redraft < 0:
                    raise RuntimeError("top-k branch redraft produced no target samples")
                if len(target_hidden_seeds) <= accepted_draft_tokens_for_redraft:
                    raise RuntimeError("top-k branch redraft missing corrective target hidden seed")
                acceptance = {
                    "accepted_draft_tokens": accepted_draft_tokens_for_redraft,
                    "visible_output_tokens": accepted_draft_tokens_for_redraft + 1,
                    "output_tokens": target_tokens[:accepted_draft_tokens_for_redraft + 1],
                    "comparison_target_tokens": target_tokens[:accepted_draft_tokens_for_redraft + 1],
                    "pending_hidden_row_index": accepted_draft_tokens_for_redraft,
                    "topk_branch_depth": topk_branch_depth,
                }
            elif topk_branch_accepted:
                acceptance = sibling_topk_acceptance_from_target_samples(
                    draft_tokens,
                    draft_top10_tokens,
                    target_tokens,
                    root_topk_accept=cycle_root_topk_accept,
                    sibling_topk_accept=cycle_sibling_topk_accept,
                )
                if acceptance is None or topk_branch_depth is None:
                    raise RuntimeError("top-k sibling acceptance accounting failed")
            else:
                if draft_tokens:
                    acceptance = llama_cpp_acceptance_from_target_samples(draft_tokens, target_tokens)
                else:
                    if not target_tokens:
                        raise RuntimeError("AR fallback cycle produced no target token")
                    acceptance = {
                        "accepted_draft_tokens": 0,
                        "visible_output_tokens": 1,
                        "output_tokens": [int(target_tokens[0])],
                        "comparison_target_tokens": [int(target_tokens[0])],
                        "pending_hidden_row_index": 0,
                    }
            accepted_draft_tokens = int(acceptance["accepted_draft_tokens"])
            if (
                args.adaptive_block_after_full_accept
                and draft_tokens
                and cycle_policy != "strict_probe_fallback"
            ):
                if accepted_draft_tokens == len(draft_tokens):
                    adaptive_block_verify_active = True
                    adaptive_draft_n_max = int(args.draft_n_max)
                else:
                    adaptive_block_verify_active = False
                    adaptive_draft_n_max = min(
                        int(adaptive_draft_n_max),
                        int(args.adaptive_probe_draft_n_max),
                    )
            low_accept_miss = (
                bool(draft_tokens)
                and accepted_draft_tokens <= int(args.adaptive_ar_fallback_max_accepted)
                and accepted_draft_tokens < len(draft_tokens)
            )
            suppress_ar_fallback_for_full_vocab_recovery = False
            if args.adaptive_full_vocab_after_cap_miss and bool(draft_tokens):
                capped_cycle_miss = (
                    low_accept_miss
                    and not cycle_full_vocab_recovery
                    and resident_draft_full_vocab is not None
                    and cycle_draft_vocab_cap < int(sh_raw.shape[0])
                )
                if capped_cycle_miss:
                    adaptive_full_vocab_recovery_active = True
                    suppress_ar_fallback_for_full_vocab_recovery = True
                elif cycle_full_vocab_recovery and accepted_draft_tokens == len(draft_tokens):
                    adaptive_full_vocab_recovery_active = False
            if (
                args.adaptive_ar_fallback
                and low_accept_miss
                and not suppress_ar_fallback_for_full_vocab_recovery
            ):
                adaptive_ar_fallback_active = True
                adaptive_ar_fallback_cooldown_left = int(args.adaptive_ar_fallback_cooldown)
            if (
                args.adaptive_strict_block_probe
                and adaptive_strict_probe_decision is None
                and cycle_policy == "strict_probe"
            ):
                adaptive_strict_probe_history.append(accepted_draft_tokens)
                if len(adaptive_strict_probe_history) >= int(args.adaptive_strict_probe_cycles):
                    keep_strict = all(
                        accepted >= int(args.adaptive_strict_probe_min_accepted)
                        for accepted in adaptive_strict_probe_history
                    )
                    adaptive_strict_probe_decision = "strict" if keep_strict else "fallback"
                    if not keep_strict:
                        adaptive_ar_fallback_active = False
                        adaptive_full_vocab_recovery_active = False
                        adaptive_block_verify_active = False
                        adaptive_draft_n_max = int(args.adaptive_strict_fallback_draft_n_max)
            if args.adaptive_draft_window and accepted_draft_tokens < len(draft_tokens):
                adaptive_floor = 1
                adaptive_draft_n_max = min(
                    int(adaptive_draft_n_max),
                    max(adaptive_floor, accepted_draft_tokens),
                )
            output_tokens = list(acceptance["output_tokens"])
            comparison_target_tokens = list(acceptance["comparison_target_tokens"])
            target_hidden_seed_rows = (
                _target_hidden_seed_summary_rows(
                    target_hidden_seeds,
                    target_lm_head_score_rows,
                    cycle_start_seq_position=int(cycle_start_seq_position),
                )
                if args.record_target_topk_scores and target_lm_head_score_rows
                else []
            )
            if target_hidden_seeds:
                if args.record_draft_hidden_stats:
                    pending_index_for_trace = int(acceptance["pending_hidden_row_index"])
                    if 0 <= pending_index_for_trace < len(target_hidden_seeds):
                        draft_hidden_state_trace.append(
                            hidden_state_summary(
                                target_hidden_seeds[pending_index_for_trace],
                                label="target_pending_hidden_seed",
                                depth=pending_index_for_trace,
                                token_id=int(output_tokens[pending_index_for_trace])
                                if pending_index_for_trace < len(output_tokens)
                                else None,
                                position=int(cycle_start_seq_position + pending_index_for_trace),
                            )
                        )
                pending_hidden_seed = np.ascontiguousarray(
                    target_hidden_seeds[int(acceptance["pending_hidden_row_index"])],
                    dtype=np.float32,
                )
            elif not args.resident_mtp_device_seed:
                raise RuntimeError("target verifier did not provide a host hidden seed")
            if args.resident_mtp_device_seed and native_complete_cycle_result is None:
                if resident_context is None:
                    raise RuntimeError("resident MTP context missing for device-seed route")
                if target_verify_seed_rows:
                    resident_context.record_verify_seeds(target_verify_seed_rows)
                    resident_context.accept(accepted_draft_tokens)
                else:
                    resident_context.capture_pending_seed_from_target(
                        token_id=int(output_tokens[-1]),
                        position=seq_position + int(acceptance["visible_output_tokens"]) - 1,
                    )
            add_cycle_stage("accept_policy_and_seed", (time.perf_counter() - t_accept_policy0) * 1000)

            mtp_device_kv_commit_ms = 0.0
            if args.mtp_device_kv_cache and native_complete_cycle_result is None:
                # Draft probing writes every proposed row.  llama.cpp keeps the
                # cycle-start token row, then replaces accepted draft-token rows
                # with verifier-derived target hidden rows and drops rejected
                # speculative rows.
                mtp_device_kv_len = min(mtp_device_kv_len, cycle_mtp_kv_base_len + 1)
            if (
                args.mtp_device_kv_cache
                and accepted_draft_tokens > 0
                and native_complete_cycle_result is None
            ):
                if mtp_device_kv_len + accepted_draft_tokens > mtp_device_kv_capacity:
                    raise RuntimeError("MTP device KV cache capacity exhausted while committing accepted rows")
                t_commit0 = time.perf_counter()
                commit_tokens = np.asarray(output_tokens[:accepted_draft_tokens], dtype=np.int64)
                commit_pos = np.arange(
                    seq_position + 1,
                    seq_position + 1 + accepted_draft_tokens,
                    dtype=np.int64,
                )
                if resident_draft is not None:
                    if cycle_resident_draft is None:
                        raise RuntimeError("resident MTP draft runner missing for KV commit")
                    if target_hidden_seeds:
                        commit_hidden_seed = np.ascontiguousarray(
                            np.concatenate(target_hidden_seeds[:accepted_draft_tokens], axis=0),
                            dtype=np.float32,
                        )
                        mtp_device_kv_len = cycle_resident_draft.write_kv_rows(
                            commit_hidden_seed,
                            commit_tokens,
                            positions=commit_pos,
                            rope_cos=_rope_cos,
                            rope_sin=_rope_sin,
                            dense_key_cache=mtp_device_key_cache,
                            dense_value_cache=mtp_device_value_cache,
                            dense_cache_len=mtp_device_kv_len,
                        )
                    elif target_verify_seed_rows:
                        if len(target_verify_seed_rows) < accepted_draft_tokens:
                            raise RuntimeError("staged verifier seed rows do not cover accepted draft rows")
                        mtp_device_kv_len = cycle_resident_draft.write_kv_rows_from_device_seed_base(
                            int(target_verify_seed_rows[0].hidden_ptr),
                            commit_tokens,
                            positions=commit_pos,
                            rope_cos=_rope_cos,
                            rope_sin=_rope_sin,
                            dense_key_cache=mtp_device_key_cache,
                            dense_value_cache=mtp_device_value_cache,
                            dense_cache_len=mtp_device_kv_len,
                        )
                    else:
                        raise RuntimeError("MTP device KV commit requires verifier hidden rows")
                else:
                    if not target_hidden_seeds:
                        raise RuntimeError("legacy MTP device KV commit requires host verifier hidden rows")
                    commit_hidden_seed = np.ascontiguousarray(
                        np.concatenate(target_hidden_seeds[:accepted_draft_tokens], axis=0),
                        dtype=np.float32,
                    )
                    _ = run_draft(
                        commit_hidden_seed,
                        token_embd_f32[commit_tokens].copy(),
                        positions=commit_pos,
                        rope_cos=_rope_cos[commit_pos],
                        rope_sin=_rope_sin[commit_pos],
                        rotary_dim=rope_dim,
                        dense_key_cache=mtp_device_key_cache,
                        dense_value_cache=mtp_device_value_cache,
                        dense_cache_len=mtp_device_kv_len,
                        kv_write_only=True,
                    )
                    mtp_device_kv_len += accepted_draft_tokens
                mtp_device_kv_commit_ms = (time.perf_counter() - t_commit0) * 1000
                add_cycle_stage("mtp_device_kv_commit", mtp_device_kv_commit_ms)
                draft_ms += mtp_device_kv_commit_ms
            elif native_complete_cycle_result is not None:
                if accepted_draft_tokens != int(native_complete_cycle_result.accepted_draft_tokens):
                    raise RuntimeError("N3 acceptance diverged from the host cycle oracle")
                if mtp_device_kv_len != int(native_complete_cycle_result.draft_cache_len_after):
                    raise RuntimeError("N3 MTP KV cursor diverged from the adapter result")
                mtp_device_kv_commit_ms = float(
                    native_complete_cycle_result.mtp_kv_commit_wall_ms
                )
                add_cycle_stage("mtp_device_kv_commit", mtp_device_kv_commit_ms)
                draft_ms += mtp_device_kv_commit_ms

            if args.mtp_context_replay:
                # Persist the MTP rows that are now committed in the target
                # history: the cycle-start sampled token plus any accepted
                # drafts.  The final corrective target token becomes
                # ``prev_token`` and is appended as a seed row by the next
                # draft() call.
                t_context_append0 = time.perf_counter()
                if target_hidden_seeds:
                    context_append_tokens = [cycle_prev_token] + [int(token) for token in output_tokens[:-1]]
                    context_append_hidden_rows = [cycle_pending_hidden_seed] + target_hidden_seeds[:accepted_draft_tokens]
                    mtp_context_tokens.extend(context_append_tokens)
                    context_append_positions = [seq_position + i for i in range(len(context_append_tokens))]
                    mtp_context_positions.extend(context_append_positions)
                    mtp_context_hidden_rows = np.concatenate(
                        [mtp_context_hidden_rows, np.concatenate(context_append_hidden_rows, axis=0)],
                        axis=0,
                    )
                elif not args.resident_mtp_device_seed:
                    raise RuntimeError("context replay requires host verifier hidden rows")
                add_cycle_stage("mtp_context_replay_append", (time.perf_counter() - t_context_append0) * 1000)
            prev_token = int(output_tokens[-1])
            previous_cycle_accepted_for_record = previous_cycle_accepted
            previous_cycle_accepted = accepted_draft_tokens

            draft_candidate_count = count_topk_draft_candidates(
                len(draft_tokens),
                root_topk_accept=cycle_root_topk_accept,
                sibling_topk_accept=cycle_sibling_topk_accept,
                sibling_topk_max_depth=args.sibling_topk_max_depth,
            ) + len(redraft_tokens)
            total_drafts += draft_candidate_count
            visible_output_tokens = int(acceptance["visible_output_tokens"])
            total_output_tokens += visible_output_tokens
            total_accepted += accepted_draft_tokens
            seq_position += visible_output_tokens
            accepted = bool(draft_tokens) and accepted_draft_tokens == len(draft_tokens)

            reported_draft_tokens = draft_tokens
            reported_draft_top10_tokens = draft_top10_tokens
            reported_draft_topk_scores = draft_topk_scores
            reported_draft_topk_margins = draft_topk_margins
            if redraft_start_depth is not None and topk_branch_depth is not None:
                reported_draft_tokens = redraft_tokens_by_depth
                reported_draft_top10_tokens = redraft_top10_by_depth
                reported_draft_topk_scores = []
                reported_draft_topk_margins = []

            target_in_draft_top10, target_rank_in_draft_top10 = target_membership_in_draft_topk(
                comparison_target_tokens, reported_draft_top10_tokens
            )

            cycle_record = {
                "cycle": cycle,
                "cycle_prev_token": int(cycle_prev_token),
                "cycle_start_seq_position": int(cycle_start_seq_position),
                "cycle_end_seq_position": int(seq_position),
                "cycle_seq_position": int(seq_position),
                "target_token": target_tokens[0],
                "target_tokens": target_tokens,
                "comparison_target_tokens": comparison_target_tokens,
                "output_tokens": output_tokens,
                "draft_token": int(reported_draft_tokens[0]) if reported_draft_tokens else None,
                "draft_tokens": reported_draft_tokens,
                "draft_top10_tokens": reported_draft_top10_tokens,
                "draft_topk_scores": reported_draft_topk_scores,
                "draft_topk_margins": reported_draft_topk_margins,
                "draft_hidden_state_trace": draft_hidden_state_trace
                if (args.record_draft_hidden_stats or args.record_draft_stage_stats)
                else [],
                "initial_draft_tokens": draft_tokens,
                "redraft_tokens": redraft_tokens,
                "redraft_start_depth": redraft_start_depth,
                "draft_top1_probs": cycle_draft_top1_probs,
                "target_in_draft_top10": target_in_draft_top10,
                "target_rank_in_draft_top10": target_rank_in_draft_top10,
                "accepted": accepted,
                "generated_draft_tokens": draft_candidate_count,
                "linear_draft_tokens": len(draft_tokens),
                "cycle_draft_n_max": int(cycle_draft_n_max),
                "cycle_policy": cycle_policy,
                "adaptive_draft_window": bool(args.adaptive_draft_window),
                "adaptive_ar_fallback": bool(args.adaptive_ar_fallback),
                "cycle_ar_fallback": bool(cycle_ar_fallback),
                "adaptive_full_vocab_after_cap_miss": bool(args.adaptive_full_vocab_after_cap_miss),
                "cycle_full_vocab_recovery": bool(cycle_full_vocab_recovery),
                "cycle_draft_vocab_cap": int(cycle_draft_vocab_cap),
                "next_cycle_full_vocab_recovery": bool(adaptive_full_vocab_recovery_active),
                "adaptive_block_after_full_accept": bool(args.adaptive_block_after_full_accept),
                "fused_b1_block_probe": bool(args.fused_b1_block_probe),
                "cycle_fused_b1_block_probe": bool(cycle_fused_b1_block_probe),
                "cycle_block_verify_allowed": bool(cycle_block_verify_allowed),
                "next_adaptive_block_verify_active": bool(adaptive_block_verify_active),
                "next_cycle_ar_fallback": bool(adaptive_ar_fallback_active),
                "next_adaptive_draft_n_max": int(adaptive_draft_n_max),
                "root_topk_accept": cycle_root_topk_accept,
                "sibling_topk_accept": cycle_sibling_topk_accept,
                "diagnostic_topk_candidate_count": cycle_diagnostic_topk_candidate_count,
                "configured_root_topk_accept": args.root_topk_accept,
                "configured_sibling_topk_accept": args.sibling_topk_accept,
                "adaptive_strict_block_probe": bool(args.adaptive_strict_block_probe),
                "adaptive_strict_probe_history": list(adaptive_strict_probe_history),
                "adaptive_strict_probe_decision": adaptive_strict_probe_decision,
                "sibling_tail_min_prev_accepted": args.sibling_tail_min_prev_accepted,
                "sibling_topk_max_depth": args.sibling_topk_max_depth,
                "root_tail_max_prev_accepted": args.root_tail_max_prev_accepted,
                "topk_branch_redraft": args.topk_branch_redraft,
                "topk_branch_redraft_max_branches": args.topk_branch_redraft_max_branches,
                "topk_branch_accept_count": topk_branch_accept_count,
                "topk_branch_depths": topk_branch_depths,
                "redraft_candidate_count": len(redraft_tokens),
                "previous_cycle_accepted": previous_cycle_accepted_for_record,
                "topk_branch_accepted": topk_branch_accepted,
                "topk_branch_depth": topk_branch_depth,
                "accepted_draft_tokens": accepted_draft_tokens,
                "visible_output_tokens": visible_output_tokens,
                "pending_hidden_row_index": acceptance["pending_hidden_row_index"],
                "mtp_context_rows_before_draft": (
                    int(cycle_mtp_kv_base_len)
                    if args.mtp_device_kv_cache
                    else len(replay_tokens)
                ),
                "mtp_context_mode": mtp_context_mode,
                "mtp_device_kv_cache": bool(args.mtp_device_kv_cache),
                "resident_mtp_device_chain": bool(args.resident_mtp_device_chain),
                "resident_mtp_device_chain_effective": bool(resident_mtp_device_chain_effective),
                "resident_mtp_draft_sync_stage_timings": bool(args.resident_mtp_draft_sync_stage_timings),
                "resident_mtp_draft_gpu_event_stage_timings": bool(
                    args.resident_mtp_draft_gpu_event_stage_timings
                ),
                "mtp_device_kv_rows_after": int(mtp_device_kv_len),
                "mtp_device_kv_commit_ms": round(mtp_device_kv_commit_ms, 2),
                "target_prefill_mode": target_prefill_mode,
                "target_block_verify": bool(block_verify_used),
                "native_spec_target_cycle": bool(native_spec_target_cycle_used),
                "native_spec_device_accept_commit": bool(native_spec_device_accept_commit_used),
                "native_spec_complete_cycle": bool(native_complete_cycle_result is not None),
                "native_spec_proposal_graph": bool(
                    native_complete_cycle_result is not None
                    and native_complete_cycle_result.proposal_native_graph
                ),
                "native_spec_target_fallback_reason": (
                    session.last_native_spec_target_fallback_reason
                    if args.native_spec_target_cycle and block_verify_used
                    else None
                ),
                "target_block_direct_state_commit": bool(
                    target_block_direct_state_commit_effective and block_verify_used
                ),
                "target_block_replay_state_commit": bool(args.target_block_replay_state_commit and block_verify_used),
                "target_block_direct_partial_replay_mode": (
                    str(args.target_block_direct_partial_replay_mode) if block_verify_used else None
                ),
                "target_block_direct_commit_exact": target_block_direct_commit_exact,
                "target_block_raw_tokens": target_block_raw_tokens,
                "target_block_consumed_rows": target_block_consumed_rows,
                "target_lm_head_logits_available": bool(target_lm_head_logits_available),
                "target_lm_head_score_rows": target_lm_head_score_rows,
                "target_hidden_seed_rows": target_hidden_seed_rows,
                "target_block_sync_stage_timings": bool(args.target_block_sync_stage_timings and block_verify_used),
                "target_b1_branch_safe_block_verify": bool(b1_branch_safe_block_verify_used),
                "target_graph_batched_verify": bool(batched_verify_used),
                "target_verify_layer_passes": int(target_verify_layer_passes),
                "target_verify_rows_evaluated": int(target_verify_rows_evaluated),
                "target_verify_serial_rows": int(target_verify_serial_rows),
                "target_verify_graph_rows": int(target_verify_graph_rows),
                "target_verify_block_passes": int(target_verify_block_passes),
                "target_verify_block_rows": int(target_verify_block_rows),
                "target_verify_replay_rows": int(target_verify_replay_rows),
                "target_verify_direct_commit_rows": int(target_verify_direct_commit_rows),
                "target_verify_discarded_rows": int(target_verify_discarded_rows),
                "ar_decode_ms": round(ar_decode_ms, 2),
                "mtp_draft_ms": round(draft_ms, 2),
            }
            if args.record_cycle_stage_timings:
                cycle_record["cycle_wall_ms"] = round((time.perf_counter() - cycle_wall_t0) * 1000, 4)
                cycle_record["stage_timings_ms"] = {
                    name: round(ms, 4) for name, ms in sorted(stage_timings_ms.items())
                }
            cycle_details.append(cycle_record)
            decode_times.append(ar_decode_ms + draft_ms)

    finally:
        if target_graph is not None:
            target_graph.close()
        if resident_draft is not None and not _cache_session:
            resident_draft.close()
        if resident_draft_full_vocab is not None:
            resident_draft_full_vocab.close()
        if runtime is not None:
            for _buf in mtp_device_kv_buffers:
                free(_buf, runtime=runtime)
        if not _cache_session:
            session.close()

    # Compute metrics
    metrics = compute_speculative_metrics(cycle_details)
    warm_metrics = compute_speculative_metrics(cycle_details[1:]) if len(cycle_details) > 1 else None
    accept_per_draft = metrics["accept_per_draft"]
    accepted_per_output = metrics["accepted_per_output"]
    avg_decode_ms = metrics["avg_decode_ms"]
    avg_ms_per_visible_token = metrics["avg_ms_per_visible_token"]
    tokens_per_sec = metrics["tokens_per_sec"]
    speedup_vs_ar_visible = metrics["speedup_vs_ar_visible"]
    visible_tokens_per_cycle = metrics["visible_tokens_per_cycle"]

    # Print summary
    print(f"\n{'='*60}")
    print(f"MTP Speculative Decoding Benchmark")
    print(f"{'='*60}")
    print(f"Model: {Path(args.model).name}")
    print(f"Cycles: {args.cycles}")
    print(f"Draft n_max: {args.draft_n_max}")
    print(f"Decode repack: {args.decode_repack}")
    print(f"Effective GEMV decode: {session.use_gemv_decode}")
    print(f"Effective WMMA prefill: {session.use_wmma_prefill}")
    print(f"MTP draft warmup: {args.mtp_draft_warmup} ({mtp_draft_warmup_ms:.2f}ms)")
    print(
        f"Target graph verify: requested={args.target_graph_verify} "
        f"effective={target_graph_verify_enabled} steps_per_replay={target_graph_steps_per_replay} "
        f"max_replay_steps={target_graph_max_replay_steps} context_cap={target_graph_context_cap} "
        f"batched={args.target_graph_batched_verify} block_verify={args.target_block_verify}"
    )
    if target_graph_verify_fallback_reason:
        print(f"Target graph verify fallback: {target_graph_verify_fallback_reason}")
    print(f"Target B1 branch-safe block verify: {args.target_b1_branch_safe_block_verify}")
    print(f"Root top-k accept: {args.root_topk_accept}")
    print(f"Sibling top-k accept: {args.sibling_topk_accept}")
    print(f"Sibling tail min previous accepted: {args.sibling_tail_min_prev_accepted}")
    print(f"Sibling top-k max depth: {args.sibling_topk_max_depth}")
    print(f"Root tail max previous accepted: {args.root_tail_max_prev_accepted}")
    print(f"Top-k branch redraft: {args.topk_branch_redraft}")
    print(f"Top-k branch redraft max branches: {args.topk_branch_redraft_max_branches}")
    print(
        "Adaptive block after full accept: "
        f"{args.adaptive_block_after_full_accept} "
        f"probe_n_max={args.adaptive_probe_draft_n_max} "
        f"fused_b1_probe={args.fused_b1_block_probe}"
    )
    print(
        "Adaptive strict block probe: "
        f"{args.adaptive_strict_block_probe} "
        f"cycles={args.adaptive_strict_probe_cycles} "
        f"min_accepted={args.adaptive_strict_probe_min_accepted} "
        f"decision={adaptive_strict_probe_decision}"
    )
    print(
        f"MTP device KV cache: {args.mtp_device_kv_cache} "
        f"rows={int(mtp_device_kv_len) if args.mtp_device_kv_cache else 0} "
        f"capacity={int(mtp_device_kv_capacity) if args.mtp_device_kv_cache else 0}"
    )
    print(
        f"Resident MTP draft: requested={bool(args.resident_mtp_draft)} "
        f"effective={bool(resident_mtp_draft_effective)} "
        f"device_seed={bool(args.resident_mtp_device_seed)} "
        f"device_chain={bool(resident_mtp_device_chain_effective)}"
    )
    if resident_mtp_draft_fallback_reason:
        print(f"Resident MTP draft fallback: {resident_mtp_draft_fallback_reason}")
    print(f"accept_per_draft: {accept_per_draft:.3f}")
    print(f"accepted_per_output: {accepted_per_output:.3f}")
    print(f"visible_tokens_per_cycle: {visible_tokens_per_cycle:.3f}")
    print(f"avg_cycle_ms: {avg_decode_ms:.2f}")
    print(f"avg_ms_per_visible_token: {avg_ms_per_visible_token:.2f}")
    print(f"tokens_per_sec: {tokens_per_sec:.2f}")
    print(f"speedup_vs_ar_visible: {speedup_vs_ar_visible:.3f}x")
    print(
        "target_verify: "
        f"layer_passes/output={metrics['target_verify_layer_passes_per_output']:.3f} "
        f"rows/output={metrics['target_verify_rows_per_output']:.3f} "
        f"replay_rows/output={metrics['target_verify_replay_rows_per_output']:.3f}"
    )
    if args.record_cycle_stage_timings and metrics.get("cycle_wall_ms_per_output") is not None:
        print(
            "cycle_wall: "
            f"ms/output={metrics['cycle_wall_ms_per_output']:.3f} "
            f"over_legacy_ms/output={(metrics.get('cycle_wall_over_legacy_ms_per_output') or 0.0):.3f}"
        )
        stage_totals = metrics.get("stage_timing_per_output_ms") or {}
        if stage_totals:
            top = sorted(stage_totals.items(), key=lambda item: float(item[1]), reverse=True)[:6]
            print("stage_ms/output: " + ", ".join(f"{k}={v:.3f}" for k, v in top))
    if warm_metrics is not None:
        print(
            "warm_excluding_cycle0: "
            f"avg_ms_per_visible_token={warm_metrics['avg_ms_per_visible_token']:.2f} "
            f"tokens_per_sec={warm_metrics['tokens_per_sec']:.2f} "
            f"speedup_vs_ar_visible={warm_metrics['speedup_vs_ar_visible']:.3f}x"
        )
    print(f"total_accepted: {total_accepted}/{total_drafts}")
    for d in cycle_details:
        acc = f"{d['accepted_draft_tokens']}/{d['generated_draft_tokens']}"
        print(f"  Cycle {d['cycle']}: accepted={acc} targets={d['target_tokens']} drafts={d['draft_tokens']} "
              f"visible={d['visible_output_tokens']} ar={d['ar_decode_ms']:.1f}ms draft={d['mtp_draft_ms']:.1f}ms")

    # Save benchmark artifact
    hw = _get_hw_info()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    result = {
        "schema": 4,
        "status": "ok",
        "timestamp": timestamp,
        "run_tag": f"mtp-bench-{int(time.time())}",
        "hardware": hw,
        "workload": {
            "model": Path(args.model).name,
            "model_path": args.model,
            "quant": "Q4_K_M",
            "prompt": args.prompt if not args.prompt_token_ids else None,
            "prompt_source": "raw_token_ids" if args.prompt_token_ids else "chat_template",
            "prompt_token_ids_sha256": hashlib.sha256(
                json.dumps(prompt, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "prompt_tokens": len(prompt),
            "initial_prev_token": int(initial_prev_token),
            "initial_prev_position": int(initial_prev_position),
            "cycles": args.cycles,
            "max_output_tokens": int(args.max_output_tokens),
            "draft_n_max": args.draft_n_max,
            "require_cached_build": bool(args.require_cached_build),
            "adaptive_draft_window": bool(args.adaptive_draft_window),
            "adaptive_ar_fallback": bool(args.adaptive_ar_fallback),
            "adaptive_ar_fallback_max_accepted": int(args.adaptive_ar_fallback_max_accepted),
            "adaptive_full_vocab_after_cap_miss": bool(args.adaptive_full_vocab_after_cap_miss),
            "adaptive_block_after_full_accept": bool(args.adaptive_block_after_full_accept),
            "fused_b1_block_probe": bool(args.fused_b1_block_probe),
            "adaptive_probe_draft_n_max": int(args.adaptive_probe_draft_n_max),
            "adaptive_strict_block_probe": bool(args.adaptive_strict_block_probe),
            "adaptive_strict_probe_cycles": int(args.adaptive_strict_probe_cycles),
            "adaptive_strict_probe_min_accepted": int(args.adaptive_strict_probe_min_accepted),
            "adaptive_strict_fallback_draft_n_max": int(args.adaptive_strict_fallback_draft_n_max),
            "adaptive_strict_fallback_root_topk": int(args.adaptive_strict_fallback_root_topk),
            "adaptive_strict_probe_decision": adaptive_strict_probe_decision,
            "adaptive_strict_probe_history": list(adaptive_strict_probe_history),
            "mtp_draft_vocab_cap": int(args.mtp_draft_vocab_cap),
            "decode_repack": bool(args.decode_repack),
            "decode_repack_env": os.environ.get("HIPENGINE_GGUF_DECODE_REPACK"),
            "use_wmma_prefill": bool(args.use_wmma_prefill),
            "use_gemv_decode": bool(args.use_gemv_decode),
            "effective_use_wmma_prefill": bool(session.use_wmma_prefill),
            "effective_use_gemv_decode": bool(session.use_gemv_decode),
            "mtp_draft_warmup": bool(args.mtp_draft_warmup),
            "mtp_draft_warmup_ms": round(float(mtp_draft_warmup_ms), 2),
            "resident_mtp_draft": bool(args.resident_mtp_draft),
            "resident_mtp_device_seed": bool(args.resident_mtp_device_seed),
            "resident_mtp_device_chain": bool(args.resident_mtp_device_chain),
            "resident_mtp_draft_sync_stage_timings": bool(args.resident_mtp_draft_sync_stage_timings),
            "resident_mtp_draft_gpu_event_stage_timings": bool(
                args.resident_mtp_draft_gpu_event_stage_timings
            ),
            "record_draft_confidence": bool(args.record_draft_confidence),
            "record_draft_topk_scores": bool(args.record_draft_topk_scores),
            "record_target_topk_scores": bool(args.record_target_topk_scores),
            "target_topk_score_count": int(args.target_topk_score_count),
            "target_score_candidate_tokens": list(args.target_score_candidate_tokens),
            "record_draft_hidden_stats": bool(args.record_draft_hidden_stats),
            "record_draft_stage_stats": bool(args.record_draft_stage_stats),
            "record_draft_cache_rows": list(args.record_draft_cache_rows),
            "record_draft_attention_debug": bool(args.record_draft_attention_debug),
            "record_cycle_stage_timings": bool(args.record_cycle_stage_timings),
            "target_block_sync_stage_timings": bool(args.target_block_sync_stage_timings),
            "native_spec_target_cycle": bool(args.native_spec_target_cycle),
            "native_spec_device_accept_commit": bool(args.native_spec_device_accept_commit),
            "native_spec_complete_cycle": bool(args.native_spec_complete_cycle),
            "native_spec_proposal_graph": bool(args.native_spec_proposal_graph),
            "llama_compat": bool(args.llama_compat),
            "verify_dp4a": bool(args.verify_dp4a),
            "verify_dense_q8_dp4a": bool(args.verify_dense_q8_dp4a),
            "verify_dense_q8_dp4a_all": bool(args.verify_dense_q8_dp4a_all),
            "verify_dense_q8_dp4a_shared": bool(args.verify_dense_q8_dp4a_shared),
            "verify_dense_q8_dp4a_shared_env": os.environ.get("HIPENGINE_GGUF_DENSE_Q8_DP4A_SHARED"),
            "verify_dense_q8_dp4a_f32": bool(args.verify_dense_q8_dp4a_f32),
            "verify_dense_q8_dp4a_f32_env": os.environ.get("HIPENGINE_GGUF_DENSE_Q8_DP4A_F32"),
            "verify_capture_prefill_gdn_env": os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"),
            "verify_lm_head_q6_top1_dp4a": bool(args.verify_lm_head_q6_top1_dp4a),
            "verify_lm_head_q6_top1_dp4a_env": os.environ.get("HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A"),
            "lm_head_q6_x8_sidecar_env": os.environ.get("HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR"),
            "resident_mtp_draft_q6_top1_dp4a": bool(args.resident_mtp_draft_q6_top1_dp4a),
            "resident_mtp_draft_router_row_parallel": bool(args.resident_mtp_draft_router_row_parallel),
            "resident_mtp_draft_router_row_parallel_env": os.environ.get(
                "HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL"
            ),
            "resident_mtp_draft_dense_q8_dp4a": bool(args.resident_mtp_draft_dense_q8_dp4a),
            "resident_mtp_draft_dense_q8_dp4a_env": os.environ.get(
                "HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A"
            ),
            "resident_mtp_draft_dense_q8_dp4a_stages": str(args.resident_mtp_draft_dense_q8_dp4a_stages),
            "resident_mtp_draft_dense_q8_dp4a_stages_env": os.environ.get(
                "HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A_STAGES"
            ),
            "resident_mtp_draft_selected_silu_down_fused": bool(
                args.resident_mtp_draft_selected_silu_down_fused
            ),
            "resident_mtp_draft_selected_silu_down_fused_env": os.environ.get(
                "HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED"
            ),
            "resident_mtp_draft_q6_top1_stage1_threads": int(args.resident_mtp_draft_q6_top1_stage1_threads),
            "resident_mtp_draft_q6_top1_stage1_threads_env": os.environ.get(
                "HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS"
            ),
            "resident_mtp_draft_q6_top1_stage1_shape": str(args.resident_mtp_draft_q6_top1_stage1_shape),
            "resident_mtp_draft_q6_top1_stage1_shape_env": os.environ.get(
                "HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE"
            ),
            "selected_down_x8_repack": str(args.selected_down_x8_repack),
            "selected_down_x8_repack_env": os.environ.get("HIPENGINE_GGUF_SELECTED_X8_REPACK"),
            "selected_gate_up_x8": bool(args.selected_gate_up_x8),
            "selected_gate_up_x8_env": os.environ.get("HIPENGINE_GGUF_SELECTED_GATE_UP_X8"),
            "selected_gate_up_raw": bool(args.selected_gate_up_raw),
            "selected_gate_up_raw_env": os.environ.get("HIPENGINE_GGUF_SELECTED_GATE_UP_RAW"),
            "resident_mtp_draft_effective": bool(resident_mtp_draft_effective),
            "resident_mtp_device_chain_effective": bool(resident_mtp_device_chain_effective),
            "resident_mtp_draft_full_vocab_recovery_effective": bool(
                resident_mtp_draft_full_vocab_recovery_effective
            ),
            "resident_mtp_draft_fallback_reason": resident_mtp_draft_fallback_reason,
            "target_graph_verify": bool(args.target_graph_verify),
            "target_graph_verify_effective": bool(target_graph_verify_enabled),
            "target_graph_steps_per_replay": int(target_graph_steps_per_replay),
            "target_graph_batched_verify": bool(args.target_graph_batched_verify),
            "target_block_verify": bool(args.target_block_verify),
            "target_block_verify_mode": str(args.target_block_verify_mode),
            "target_block_wmma_prefill": bool(args.target_block_wmma_prefill),
            "target_block_direct_state_commit": bool(args.target_block_direct_state_commit),
            "target_block_replay_state_commit": bool(args.target_block_replay_state_commit),
            "target_block_direct_partial_replay_mode": str(args.target_block_direct_partial_replay_mode),
            "target_b1_branch_safe_block_verify": bool(args.target_b1_branch_safe_block_verify),
            "target_graph_max_replay_steps": int(target_graph_max_replay_steps),
            "target_graph_context_cap": int(target_graph_context_cap),
            "target_graph_verify_fallback_reason": target_graph_verify_fallback_reason,
            "fastpath_safety": None if session.fastpath_safety is None else session.fastpath_safety.as_dict(),
            "prompt_reasoning": args.prompt_reasoning,
            "root_topk_accept": args.root_topk_accept,
            "sibling_topk_accept": args.sibling_topk_accept,
            "sibling_tail_min_prev_accepted": args.sibling_tail_min_prev_accepted,
            "sibling_topk_max_depth": args.sibling_topk_max_depth,
            "root_tail_max_prev_accepted": args.root_tail_max_prev_accepted,
            "topk_branch_redraft": args.topk_branch_redraft,
            "topk_branch_redraft_max_branches": args.topk_branch_redraft_max_branches,
            "mtp_device_kv_cache": bool(args.mtp_device_kv_cache),
            "mtp_initial_kv_writer": mtp_initial_kv_writer,
            "mtp_device_kv_rows": int(mtp_device_kv_len) if args.mtp_device_kv_cache else 0,
            "mtp_device_kv_capacity": int(mtp_device_kv_capacity) if args.mtp_device_kv_cache else 0,
            "engine": "hipEngine GGUF MTP",
            "target_prefill_mode": target_prefill_mode,
            "mtp_context_mode": mtp_context_mode,
        },
        "metrics": {
            "accept_per_draft": round(float(metrics["accept_per_draft"]), 4),
            "accepted_per_output": round(float(metrics["accepted_per_output"]), 4),
            "visible_tokens_per_cycle": round(float(metrics["visible_tokens_per_cycle"]), 4),
            "avg_cycle_ms": round(float(metrics["avg_cycle_ms"]), 2),
            "avg_decode_ms": round(float(metrics["avg_decode_ms"]), 2),
            "avg_ar_decode_ms": round(float(metrics["avg_ar_decode_ms"]), 2),
            "avg_mtp_draft_ms": round(float(metrics["avg_mtp_draft_ms"]), 2),
            "avg_ms_per_visible_token": round(float(metrics["avg_ms_per_visible_token"]), 2),
            "tokens_per_sec": round(float(metrics["tokens_per_sec"]), 2),
            "ar_baseline_tokens_per_sec": round(float(metrics["ar_baseline_tokens_per_sec"]), 2),
            "speedup_vs_ar_visible": round(float(metrics["speedup_vs_ar_visible"]), 4),
            "total_accepted": metrics["total_accepted"],
            "total_drafts": metrics["total_drafts"],
            "total_output_tokens": metrics["total_output_tokens"],
            "verify_cycle_count": metrics["verify_cycle_count"],
            "target_verify_layer_passes": metrics["target_verify_layer_passes"],
            "target_verify_rows_evaluated": metrics["target_verify_rows_evaluated"],
            "target_verify_serial_rows": metrics["target_verify_serial_rows"],
            "target_verify_graph_rows": metrics["target_verify_graph_rows"],
            "target_verify_block_passes": metrics["target_verify_block_passes"],
            "target_verify_block_rows": metrics["target_verify_block_rows"],
            "target_verify_replay_rows": metrics["target_verify_replay_rows"],
            "target_verify_direct_commit_rows": metrics["target_verify_direct_commit_rows"],
            "target_verify_discarded_rows": metrics["target_verify_discarded_rows"],
            "target_verify_layer_passes_per_output": round(
                float(metrics["target_verify_layer_passes_per_output"]), 4
            ),
            "target_verify_rows_per_output": round(float(metrics["target_verify_rows_per_output"]), 4),
            "target_verify_replay_rows_per_output": round(
                float(metrics["target_verify_replay_rows_per_output"]), 4
            ),
            "cycle_wall_ms_total": (
                round(float(metrics["cycle_wall_ms_total"]), 2)
                if metrics.get("cycle_wall_ms_total") is not None else None
            ),
            "cycle_wall_ms_per_output": (
                round(float(metrics["cycle_wall_ms_per_output"]), 4)
                if metrics.get("cycle_wall_ms_per_output") is not None else None
            ),
            "cycle_wall_over_legacy_ms_total": (
                round(float(metrics["cycle_wall_over_legacy_ms_total"]), 2)
                if metrics.get("cycle_wall_over_legacy_ms_total") is not None else None
            ),
            "cycle_wall_over_legacy_ms_per_output": (
                round(float(metrics["cycle_wall_over_legacy_ms_per_output"]), 4)
                if metrics.get("cycle_wall_over_legacy_ms_per_output") is not None else None
            ),
            "stage_timing_totals_ms": (
                {k: round(float(v), 4) for k, v in metrics["stage_timing_totals_ms"].items()}
                if metrics.get("stage_timing_totals_ms") else None
            ),
            "stage_timing_per_output_ms": (
                {k: round(float(v), 4) for k, v in metrics["stage_timing_per_output_ms"].items()}
                if metrics.get("stage_timing_per_output_ms") else None
            ),
            "denominators": metrics["denominators"],
            "warm_excluding_cycle0": (
                {
                    "accept_per_draft": round(float(warm_metrics["accept_per_draft"]), 4),
                    "accepted_per_output": round(float(warm_metrics["accepted_per_output"]), 4),
                    "visible_tokens_per_cycle": round(float(warm_metrics["visible_tokens_per_cycle"]), 4),
                    "avg_cycle_ms": round(float(warm_metrics["avg_cycle_ms"]), 2),
                    "avg_ar_decode_ms": round(float(warm_metrics["avg_ar_decode_ms"]), 2),
                    "avg_mtp_draft_ms": round(float(warm_metrics["avg_mtp_draft_ms"]), 2),
                    "avg_ms_per_visible_token": round(float(warm_metrics["avg_ms_per_visible_token"]), 2),
                    "tokens_per_sec": round(float(warm_metrics["tokens_per_sec"]), 2),
                    "speedup_vs_ar_visible": round(float(warm_metrics["speedup_vs_ar_visible"]), 4),
                    "total_accepted": warm_metrics["total_accepted"],
                    "total_drafts": warm_metrics["total_drafts"],
                    "total_output_tokens": warm_metrics["total_output_tokens"],
                    "verify_cycle_count": warm_metrics["verify_cycle_count"],
                    "target_verify_layer_passes": warm_metrics["target_verify_layer_passes"],
                    "target_verify_rows_evaluated": warm_metrics["target_verify_rows_evaluated"],
                    "target_verify_serial_rows": warm_metrics["target_verify_serial_rows"],
                    "target_verify_graph_rows": warm_metrics["target_verify_graph_rows"],
                    "target_verify_block_passes": warm_metrics["target_verify_block_passes"],
                    "target_verify_block_rows": warm_metrics["target_verify_block_rows"],
                    "target_verify_replay_rows": warm_metrics["target_verify_replay_rows"],
                    "target_verify_direct_commit_rows": warm_metrics["target_verify_direct_commit_rows"],
                    "target_verify_discarded_rows": warm_metrics["target_verify_discarded_rows"],
                    "target_verify_layer_passes_per_output": round(
                        float(warm_metrics["target_verify_layer_passes_per_output"]), 4
                    ),
                    "target_verify_rows_per_output": round(float(warm_metrics["target_verify_rows_per_output"]), 4),
                    "target_verify_replay_rows_per_output": round(
                        float(warm_metrics["target_verify_replay_rows_per_output"]), 4
                    ),
                    "cycle_wall_ms_total": (
                        round(float(warm_metrics["cycle_wall_ms_total"]), 2)
                        if warm_metrics.get("cycle_wall_ms_total") is not None else None
                    ),
                    "cycle_wall_ms_per_output": (
                        round(float(warm_metrics["cycle_wall_ms_per_output"]), 4)
                        if warm_metrics.get("cycle_wall_ms_per_output") is not None else None
                    ),
                    "cycle_wall_over_legacy_ms_total": (
                        round(float(warm_metrics["cycle_wall_over_legacy_ms_total"]), 2)
                        if warm_metrics.get("cycle_wall_over_legacy_ms_total") is not None else None
                    ),
                    "cycle_wall_over_legacy_ms_per_output": (
                        round(float(warm_metrics["cycle_wall_over_legacy_ms_per_output"]), 4)
                        if warm_metrics.get("cycle_wall_over_legacy_ms_per_output") is not None else None
                    ),
                    "stage_timing_totals_ms": (
                        {k: round(float(v), 4) for k, v in warm_metrics["stage_timing_totals_ms"].items()}
                        if warm_metrics.get("stage_timing_totals_ms") else None
                    ),
                    "stage_timing_per_output_ms": (
                        {k: round(float(v), 4) for k, v in warm_metrics["stage_timing_per_output_ms"].items()}
                        if warm_metrics.get("stage_timing_per_output_ms") else None
                    ),
                }
                if warm_metrics is not None else None
            ),
        },
        "cycles": cycle_details,
    }

    out_path = args.output or f"benchmarks/results/mtp-bench-{int(time.time())}.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nBenchmark saved to: {out_path}")


if __name__ == "__main__":
    main()
