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
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

GGUF_PATH = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
DEFAULT_PROMPT = "What is the capital of France?"
IM_START_TOKEN = 248045
IM_END_TOKEN = 248046
THINK_START_TOKEN = 248068
THINK_END_TOKEN = 248069


def build_chat_prompt(tokenizer, user_prompt: str = DEFAULT_PROMPT, *, reasoning: str = "off") -> list[int]:
    """Build the llama.cpp-compatible Qwen chat prompt for GGUF MTP.

    llama.cpp ``--reasoning off`` still renders an empty thinking block for this
    GGUF chat template, ending with exactly two newlines after ``</think>``.
    Include that suffix by default so native accepted/output diagnostics compare
    against the same token stream as llama-server.
    """
    if reasoning not in {"off", "none"}:
        raise ValueError("build_chat_prompt currently supports only reasoning='off'/'none'")
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
    return prompt


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


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
    top_idx = np.argpartition(logits_row, -requested_k)[-requested_k:]
    top_sorted = top_idx[np.argsort(logits_row[top_idx])[::-1]]
    candidate_pool = [int(t) for t in top_sorted]
    return candidate_pool[0], candidate_pool[:requested_k]


def validate_draft_n_max(draft_n_max: int) -> int:
    """Validate the benchmark's currently implemented diagnostic draft depth.

    ``scripts/gguf_mtp_bench.py`` is a correctness/acceptance diagnostic, not the
    prompt-suite performance runner. Its local chained draft loop can exercise
    B1 through B5 now; artifacts still label the MTP context mode so these rows
    are not confused with the future persistent GGUF MTP context implementation.
    """
    draft_n_max = int(draft_n_max)
    if not 1 <= draft_n_max <= 5:
        raise ValueError("draft_n_max must be in 1..5 for GGUF MTP diagnostics")
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

    verify_cycle_count = len(cycles)
    total_drafts = 0
    total_accepted = 0
    visible_output_tokens = 0
    total_ar_ms = 0.0
    total_draft_ms = 0.0
    for index, cycle in enumerate(cycles):
        if not isinstance(cycle, dict):
            raise ValueError(f"cycles[{index}] must be an object")
        generated_drafts = require_int(cycle, index, "generated_draft_tokens", positive=True)
        accepted_drafts = require_int(cycle, index, "accepted_draft_tokens")
        visible_output = require_int(cycle, index, "visible_output_tokens", positive=True)
        if accepted_drafts > generated_drafts:
            raise ValueError(f"cycles[{index}].accepted_draft_tokens must be <= generated_draft_tokens")
        if accepted_drafts > visible_output:
            raise ValueError(f"cycles[{index}].accepted_draft_tokens must be <= visible_output_tokens")
        total_drafts += generated_drafts
        total_accepted += accepted_drafts
        visible_output_tokens += visible_output
        total_ar_ms += require_timing(cycle, index, "ar_decode_ms")
        total_draft_ms += require_timing(cycle, index, "mtp_draft_ms")
    total_cycle_ms = total_ar_ms + total_draft_ms

    accept_per_draft = total_accepted / total_drafts
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

    return {
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
        "denominators": {
            "accept_per_draft": "accepted_draft_tokens / generated_draft_tokens",
            "accepted_per_output": "accepted_draft_tokens / visible_output_token_count",
            "visible_tokens_per_cycle": "visible_output_token_count / verify_cycle_count",
            "tokens_per_sec": "visible_output_token_count / total_cycle_wall_time",
        },
    }


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


def root_topk_acceptance_from_target_samples(
    draft_tokens: list[int],
    draft_topk_tokens: list[list[int]],
    target_samples: list[int],
    *,
    root_topk_accept: int,
) -> dict[str, object] | None:
    """Accept a depth-0 branch when the target token is in the root top-k set.

    This models a tiny tree proposal: the root logits expose multiple candidate
    first draft tokens, exact target verification selects the matching root
    sibling, and then one corrective target token is emitted.  It deliberately
    does not continue down the linear argmax chain for non-argmax root siblings
    because branch-specific deeper draft rows were not generated.
    """
    if root_topk_accept <= 1:
        return None
    if not draft_tokens or not draft_topk_tokens or not target_samples:
        return None
    root_target = int(target_samples[0])
    if root_target == int(draft_tokens[0]):
        return None
    if root_target not in [int(token) for token in draft_topk_tokens[0][:root_topk_accept]]:
        return None
    if len(target_samples) < 2:
        raise ValueError("target_samples must include the corrective target token after root top-k acceptance")
    output_tokens = [root_target, int(target_samples[1])]
    return {
        "accepted_draft_tokens": 1,
        "visible_output_tokens": len(output_tokens),
        "output_tokens": output_tokens,
        "comparison_target_tokens": output_tokens,
        "pending_hidden_row_index": 1,
    }


def _draft_top1_prob(logits_row: np.ndarray) -> float:
    """Compute the softmax probability of the argmax token."""
    shifted = logits_row - logits_row.max()
    exp = np.exp(shifted)
    return float(exp.max() / exp.sum())


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


def main():
    parser = argparse.ArgumentParser(description="MTP speculative decoding benchmark")
    parser.add_argument("--model", default=GGUF_PATH, help="GGUF model path")
    parser.add_argument("--cycles", type=int, default=10, help="Number of speculate-verify cycles")
    parser.add_argument("--draft-n-max", type=int, default=1, help="Max draft tokens per cycle")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt text before the assistant turn")
    parser.add_argument("--output", default=None, help="Output JSON path (default: benchmarks/results/mtp-bench-<timestamp>.json)")
    parser.add_argument(
        "--mtp-context-replay",
        action="store_true",
        help=(
            "Diagnostic only: replay llama.cpp-style prompt catch-up rows through the MTP block. "
            "Slow and currently not the default because the bulk target path does not expose all hidden rows."
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
        "--root-topk-accept",
        type=int,
        default=1,
        help=(
            "Diagnostic tree proposal: accept a depth-0 draft when the target token is in "
            "the first K root candidates (1 = linear argmax path)."
        ),
    )
    args = parser.parse_args()
    try:
        args.draft_n_max = validate_draft_n_max(args.draft_n_max)
    except ValueError as exc:
        parser.error(str(exc))
    if args.root_topk_accept < 1 or args.root_topk_accept > 10:
        parser.error("--root-topk-accept must be in 1..10")

    if not _hip_available():
        print("ERROR: ROCm/HIP not available", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    from hipengine.loading.gguf import GGUFReader
    from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        qwen35_gguf_mtp_nextn_layer_logits_f32 as gpu_kernel,
    )
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.core.hip import get_hip_runtime, HipMemcpyKind

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

    # Build chat-formatted prompt
    prompt = build_chat_prompt(tok, args.prompt)

    print(f"Prompt: {repr(tok.decode(prompt))}")
    print(f"Prompt tokens: {len(prompt)}")

    # Use raw Q6_K shared_head weight (398MB vs 2034MB F32 dequant)
    sh_raw = np.asarray(get("output.weight"), dtype=np.uint8)
    sh_qtype = qt("output.weight")
    print(f"Raw shared_head ({sh_qtype.name}): {sh_raw.nbytes/1e6:.0f}MB")

    # Build GPU kernel args
    def run_draft(hidden_seed, token_embed, *, return_hidden_seed: bool = False,
                   positions=None, rope_cos=None, rope_sin=None, rotary_dim=None):
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
            gpu_kwargs["context_counts"] = np.arange(1, len(positions) + 1, dtype=np.int64)
        if rope_cos is not None and rope_sin is not None:
            gpu_kwargs["rope_cos"] = rope_cos
            gpu_kwargs["rope_sin"] = rope_sin
            gpu_kwargs["rotary_dim"] = rotary_dim or rope_dim
        result = gpu_kernel(*gpu_args, **gpu_kwargs)
        if return_hidden_seed:
            logits, next_hidden_seed = result
            return (
                np.asarray(logits, dtype=np.float32),
                np.ascontiguousarray(next_hidden_seed, dtype=np.float32),
            )
        return np.asarray(result, dtype=np.float32)

    # Run benchmark
    session = Qwen35GGUFResidentSession(model_path=args.model)
    try:
        runtime = session.runtime or get_hip_runtime()
        hidden_size = 2048

        def copy_pending_hidden_seed() -> np.ndarray:
            hidden_seed = np.empty((1, hidden_size), dtype=np.float32)
            runtime.memcpy(
                hidden_seed.ctypes.data,
                session.fp32_hidden_seed_ptr(),
                hidden_size * 4,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            return hidden_seed

        def serial_prefill_with_hidden_trace() -> tuple[object, np.ndarray]:
            """Consume the prompt serially and capture every target hidden row.

            llama.cpp's MTP context catch-up needs the post-output_norm hidden
            row for every target prompt token, shifted right by one row.  The
            resident bulk prefill only exposes the final row today, so the
            acceptance-parity diagnostic uses the serial target path until the
            bulk path has an all-row hidden tap.
            """
            session.reset()
            hidden_rows: list[np.ndarray] = []
            hidden_ptr: int | None = None
            for token_id in prompt:
                hidden_ptr = session._run_token_to_final_hidden(  # noqa: SLF001 - diagnostic parity hook
                    int(token_id),
                    position=session.position,
                    capture_hidden_seed_fp32=True,
                )
                session._position += 1  # noqa: SLF001 - mirrors Qwen35GGUFResidentSession.prefill serial path
                hidden_rows.append(copy_pending_hidden_seed()[0].copy())
            if hidden_ptr is None:
                raise RuntimeError("prompt produced no hidden row")
            return session._sample_from_hidden(hidden_ptr, return_logits=False), np.ascontiguousarray(
                np.stack(hidden_rows, axis=0), dtype=np.float32
            )

        if args.mtp_context_replay:
            # Use bulk prefill (correct SSM state) instead of serial prefill
            # (which has an SSM conv state bug: token 2493 'might' vs 303 'in').
            prefill_result = session.prefill(prompt, return_logits=False, capture_hidden_seed_fp32=True)
            prev_token = int(prefill_result.token_id)
            pending_hidden_seed = copy_pending_hidden_seed()
            mtp_context_tokens = []
            mtp_context_hidden_rows = np.empty((0, hidden_size), dtype=np.float32)
            target_prefill_mode = "bulk_prefill_seed_only"
            mtp_context_mode = "bulk_prefill_single_seed_replay"
        else:
            prefill_result = session.prefill(prompt, return_logits=False, capture_hidden_seed_fp32=True)
            prev_token = int(prefill_result.token_id)
            pending_hidden_seed = copy_pending_hidden_seed()
            mtp_context_tokens = []
            mtp_context_hidden_rows = np.empty((0, hidden_size), dtype=np.float32)
            target_prefill_mode = "resident_default"
            mtp_context_mode = "single_seed_row"

        # Track the current sequence position for draft model RoPE.
        # After prefill, session.position = len(prompt). The first sampled
        # token (prev_token) will be verified at this position.
        seq_position = int(session.position)
        # Positions of the committed MTP context tokens (for replay mode).
        if args.mtp_context_replay:
            mtp_context_positions = list(range(len(mtp_context_tokens)))
        else:
            mtp_context_positions: list[int] = []

        total_drafts = 0
        total_accepted = 0
        total_output_tokens = 0
        cycle_details = []
        decode_times = []

        for cycle in range(args.cycles):
            cycle_prev_token = int(prev_token)
            cycle_pending_hidden_seed = np.ascontiguousarray(pending_hidden_seed, dtype=np.float32).copy()

            t2 = time.perf_counter()
            draft_tokens = []
            draft_top10_tokens = []
            replay_tokens = [cycle_prev_token]
            if args.mtp_context_replay:
                # Use the same sequential single-seed draft approach as non-replay.
                # The batch replay approach (processing all committed MTP rows at
                # once) produces wrong draft logits because the MTP layer's
                # self-attention over synthetic catch-up rows doesn't match
                # llama.cpp's sequential KV-cache build-up.
                current_hidden_seed = cycle_pending_hidden_seed
                current_token = cycle_prev_token
                current_pos = seq_position
                for draft_depth in range(args.draft_n_max):
                    token_embed = token_embd_f32[current_token:current_token + 1].copy()
                    need_next_seed = draft_depth + 1 < args.draft_n_max
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
                    draft_token, top10_tokens = select_topk_tokens(
                        draft_logits[0], k=10, draft_depth=draft_depth
                    )
                    draft_tokens.append(draft_token)
                    draft_top10_tokens.append(top10_tokens)
                    current_token = draft_token
                    current_pos += 1
                    if args.draft_p_min > 0.0 and draft_depth + 1 < args.draft_n_max:
                        if _draft_top1_prob(draft_logits[0]) < args.draft_p_min:
                            break
            else:
                current_hidden_seed = cycle_pending_hidden_seed
                current_token = cycle_prev_token
                current_pos = seq_position
                for draft_depth in range(args.draft_n_max):
                    token_embed = token_embd_f32[current_token:current_token + 1].copy()
                    need_next_seed = draft_depth + 1 < args.draft_n_max
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
                    draft_token, top10_tokens = select_topk_tokens(
                        draft_logits[0], k=10, draft_depth=draft_depth
                    )
                    draft_tokens.append(draft_token)
                    draft_top10_tokens.append(top10_tokens)
                    current_token = draft_token
                    current_pos += 1
                    if args.draft_p_min > 0.0 and draft_depth + 1 < args.draft_n_max:
                        if _draft_top1_prob(draft_logits[0]) < args.draft_p_min:
                            break
            t3 = time.perf_counter()
            draft_ms = (t3 - t2) * 1000

            # Verify/account with llama.cpp semantics. The target evaluates the
            # sampled token plus accepted draft prefix and returns one final
            # corrective target token. Output tokens are therefore
            # accepted_drafts + 1, and accept(n) re-seeds pending_hidden_seed from
            # the hidden row at the accepted-prefix boundary.
            ar_decode_ms = 0.0
            target_tokens = []
            target_hidden_seeds = []
            verify_input_token = cycle_prev_token
            branch_root_accepted = False
            branch_root_accept_token: int | None = None
            while True:
                t0 = time.perf_counter()
                target_result = session.step(verify_input_token, capture_hidden_seed_fp32=True)
                t1 = time.perf_counter()
                ar_decode_ms += (t1 - t0) * 1000
                target_token = int(target_result.token_id)
                target_tokens.append(target_token)
                target_hidden_seed = copy_pending_hidden_seed()
                target_hidden_seeds.append(target_hidden_seed)

                depth = len(target_tokens) - 1
                if branch_root_accepted:
                    break
                if depth < len(draft_tokens) and target_token == draft_tokens[depth]:
                    verify_input_token = target_token
                    continue
                if (
                    depth == 0
                    and args.root_topk_accept > 1
                    and draft_top10_tokens
                    and target_token in draft_top10_tokens[0][:args.root_topk_accept]
                ):
                    branch_root_accepted = True
                    branch_root_accept_token = target_token
                    verify_input_token = target_token
                    continue
                break

            if branch_root_accepted:
                acceptance = root_topk_acceptance_from_target_samples(
                    draft_tokens,
                    draft_top10_tokens,
                    target_tokens,
                    root_topk_accept=args.root_topk_accept,
                )
                if acceptance is None or branch_root_accept_token is None:
                    raise RuntimeError("branch root acceptance accounting failed")
            else:
                acceptance = llama_cpp_acceptance_from_target_samples(draft_tokens, target_tokens)
            accepted_draft_tokens = int(acceptance["accepted_draft_tokens"])
            output_tokens = list(acceptance["output_tokens"])
            comparison_target_tokens = list(acceptance["comparison_target_tokens"])
            pending_hidden_seed = np.ascontiguousarray(
                target_hidden_seeds[int(acceptance["pending_hidden_row_index"])],
                dtype=np.float32,
            )

            if args.mtp_context_replay:
                # Persist the MTP rows that are now committed in the target
                # history: the cycle-start sampled token plus any accepted
                # drafts.  The final corrective target token becomes
                # ``prev_token`` and is appended as a seed row by the next
                # draft() call.
                context_append_tokens = [cycle_prev_token] + [int(token) for token in output_tokens[:-1]]
                context_append_hidden_rows = [cycle_pending_hidden_seed] + target_hidden_seeds[:accepted_draft_tokens]
                mtp_context_tokens.extend(context_append_tokens)
                context_append_positions = [seq_position + i for i in range(len(context_append_tokens))]
                mtp_context_positions.extend(context_append_positions)
                mtp_context_hidden_rows = np.concatenate(
                    [mtp_context_hidden_rows, np.concatenate(context_append_hidden_rows, axis=0)],
                    axis=0,
                )
            prev_token = int(output_tokens[-1])

            draft_candidate_count = len(draft_tokens)
            if draft_tokens and args.root_topk_accept > 1:
                draft_candidate_count += args.root_topk_accept - 1
            total_drafts += draft_candidate_count
            visible_output_tokens = int(acceptance["visible_output_tokens"])
            total_output_tokens += visible_output_tokens
            total_accepted += accepted_draft_tokens
            seq_position += visible_output_tokens
            accepted = accepted_draft_tokens == len(draft_tokens)

            target_in_draft_top10 = []
            target_rank_in_draft_top10 = []
            for depth, target in enumerate(comparison_target_tokens):
                top10 = draft_top10_tokens[depth]
                if target in top10:
                    target_in_draft_top10.append(True)
                    target_rank_in_draft_top10.append(top10.index(target) + 1)
                else:
                    target_in_draft_top10.append(False)
                    target_rank_in_draft_top10.append(None)

            cycle_details.append({
                "cycle": cycle,
                "target_token": target_tokens[0],
                "target_tokens": target_tokens,
                "comparison_target_tokens": comparison_target_tokens,
                "output_tokens": output_tokens,
                "draft_token": draft_tokens[0],
                "draft_tokens": draft_tokens,
                "draft_top10_tokens": draft_top10_tokens,
                "target_in_draft_top10": target_in_draft_top10,
                "target_rank_in_draft_top10": target_rank_in_draft_top10,
                "accepted": accepted,
                "generated_draft_tokens": draft_candidate_count,
                "linear_draft_tokens": len(draft_tokens),
                "root_topk_accept": args.root_topk_accept,
                "branch_root_accepted": branch_root_accepted,
                "accepted_draft_tokens": accepted_draft_tokens,
                "visible_output_tokens": visible_output_tokens,
                "pending_hidden_row_index": acceptance["pending_hidden_row_index"],
                "mtp_context_rows_before_draft": len(replay_tokens),
                "mtp_context_mode": mtp_context_mode,
                "target_prefill_mode": target_prefill_mode,
                "ar_decode_ms": round(ar_decode_ms, 2),
                "mtp_draft_ms": round(draft_ms, 2),
            })
            decode_times.append(ar_decode_ms + draft_ms)

    finally:
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
    print(f"Root top-k accept: {args.root_topk_accept}")
    print(f"accept_per_draft: {accept_per_draft:.3f}")
    print(f"accepted_per_output: {accepted_per_output:.3f}")
    print(f"visible_tokens_per_cycle: {visible_tokens_per_cycle:.3f}")
    print(f"avg_cycle_ms: {avg_decode_ms:.2f}")
    print(f"avg_ms_per_visible_token: {avg_ms_per_visible_token:.2f}")
    print(f"tokens_per_sec: {tokens_per_sec:.2f}")
    print(f"speedup_vs_ar_visible: {speedup_vs_ar_visible:.3f}x")
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
            "prompt": args.prompt,
            "prompt_tokens": len(prompt),
            "cycles": args.cycles,
            "draft_n_max": args.draft_n_max,
            "root_topk_accept": args.root_topk_accept,
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