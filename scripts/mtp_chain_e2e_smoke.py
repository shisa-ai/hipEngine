#!/usr/bin/env python3
"""Correctness-first native MTP chain E2E smoke through the shared verifier.

This is an intermediate Task #41/#40 bridge.  It uses the native MTP proposal
chain to build candidate-only DraftBatch rows, then calls
Qwen35ParoResidentSession.verify_chain_bulk_and_commit.  To keep the first E2E
hook simple, proposal input hidden rows are copied back to host before invoking
``run_smoke(..., target_hidden_bits_override=...)``; therefore this script is not
a throughput benchmark and must not be promoted as a speed row.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _RoctxProfilerWindow:
    """Minimal roctxProfilerResume/Pause helper for selected-region rocprofv3.

    The harness opens libroctx64.so lazily; if it fails (no rocprofv3 ROCTX SDK
    overlay on LD_LIBRARY_PATH) the window is a no-op and the bench still runs.
    Also exposes range push/pop and per-pass marker times so rocprofv3 1.1.0
    hosts (which silently drop --selected-regions) can post-process the trace
    using the per-pass wall-clock ns boundaries instead.
    """

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._lib: ctypes.CDLL | None = None
        self._resume = None
        self._pause = None
        self._push = None
        self._pop = None
        self._active = False
        if not self.enabled:
            return
        try:
            self._lib = ctypes.CDLL("libroctx64.so")
        except OSError as exc:
            print(f"warning: roctxProfiler requested but libroctx64.so unavailable: {exc}", file=sys.stderr)
            self._lib = None
            return
        self._resume = getattr(self._lib, "roctxProfilerResume", None)
        self._pause = getattr(self._lib, "roctxProfilerPause", None)
        self._push = getattr(self._lib, "roctxRangePushA", None)
        self._pop = getattr(self._lib, "roctxRangePop", None)
        if self._push is not None:
            self._push.argtypes = [ctypes.c_char_p]
            self._push.restype = ctypes.c_int
        if self._pop is not None:
            self._pop.argtypes = []
            self._pop.restype = ctypes.c_int

    def resume(self) -> None:
        if self._resume is not None and not self._active:
            self._resume(0)
            self._active = True

    def pause(self) -> None:
        if self._pause is not None and self._active:
            self._pause(0)
            self._active = False

    def range_push(self, name: str) -> None:
        if self._push is not None:
            self._push(name.encode("utf-8"))

    def range_pop(self) -> None:
        if self._pop is not None:
            self._pop()

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative import MTP_CHAIN_CANDIDATE_BUDGETS, MtpDraftRequest, TargetVerifyBatch, compile_mtp_chain
from hipengine.speculative.mtp_native import NativeMtpChainProposer
from scripts.mtp_native_decode_step_smoke import run_smoke as run_native_mtp_proposal
from scripts.dflash_chain_e2e_bench import _build_branching_topk_tree_target_batch

DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")


def _topk_softmax_top1(topk_logits: Sequence[float]) -> float:
    """Confidence proxy: top-1 probability over the depth's top-K logits.

    Restricted-vocab softmax (only the K emitted logits) -- a lower bound on the
    true top-1 probability, sufficient for an online whole-cycle gate curve.
    """
    if not topk_logits:
        return 1.0
    vals = [float(x) for x in topk_logits]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    z = sum(exps)
    return (exps[0] / z) if z > 0 else 1.0


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _zero_accept_streaks(accepted_lengths: Sequence[int]) -> list[int]:
    streaks: list[int] = []
    current = 0
    for accepted in accepted_lengths:
        if int(accepted) == 0:
            current += 1
        elif current:
            streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    return streaks


def _zero_accept_streaks_from_diagnostic_cycles(cycles: Sequence[dict[str, Any]]) -> list[int]:
    streaks: list[int] = []
    current = 0
    for cycle in cycles:
        if str(cycle.get("policy") or "mtp_verify") != "mtp_verify":
            if current:
                streaks.append(current)
                current = 0
            continue
        if int(cycle.get("accepted") or 0) == 0:
            current += 1
        elif current:
            streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    return streaks


def _diagnostic_reason(
    *,
    accepted: int,
    active_budget: int,
    proposed: int | None,
    target: int | None,
    draft_vocab_cap: int | None,
) -> str:
    if int(accepted) >= int(active_budget):
        return "full_accept"
    if proposed is None:
        return "missing_candidate"
    if target is None:
        return "missing_target_top1"
    if draft_vocab_cap is not None and int(target) >= int(draft_vocab_cap):
        return "target_outside_draft_vocab_cap"
    if int(proposed) == int(target):
        return "accepted_prefix_inconsistent"
    return "draft_top1_miss"


def _acceptance_diagnostic_cycle(
    *,
    cycle: int,
    decode_offset: int,
    root_position: int,
    root_token: int,
    candidates: Sequence[int],
    target_top1: Sequence[int],
    bonus_token: int | None,
    active_budget: int,
    verify_budget: int,
    accepted: int,
    draft_vocab_cap: int | None,
    use_tree: bool = False,
) -> dict[str, Any]:
    target_by_depth: dict[int, int] = {}
    for depth, token in enumerate(target_top1[: int(active_budget)]):
        target_by_depth[int(depth)] = int(token)
    for depth in range(min(int(accepted), len(candidates))):
        target_by_depth.setdefault(int(depth), int(candidates[depth]))
    if int(accepted) < int(active_budget) and bonus_token is not None:
        target_by_depth.setdefault(int(accepted), int(bonus_token))
    first_rejected_depth = int(accepted) if int(accepted) < int(active_budget) else None
    proposed = (
        int(candidates[first_rejected_depth])
        if first_rejected_depth is not None and first_rejected_depth < len(candidates)
        else None
    )
    target = target_by_depth.get(int(first_rejected_depth)) if first_rejected_depth is not None else None
    reason = (
        "tree_mode_not_classified"
        if use_tree
        else _diagnostic_reason(
            accepted=int(accepted),
            active_budget=int(active_budget),
            proposed=proposed,
            target=target,
            draft_vocab_cap=draft_vocab_cap,
        )
    )
    per_depth: list[dict[str, Any]] = []
    for depth in range(int(active_budget)):
        depth_proposed = int(candidates[depth]) if depth < len(candidates) else None
        depth_target = target_by_depth.get(int(depth))
        per_depth.append(
            {
                "depth": int(depth),
                "proposed_token": depth_proposed,
                "target_top1_token": depth_target,
                "accepted": bool(depth < int(accepted)),
                "target_in_draft_vocab_cap": (
                    None
                    if depth_target is None or draft_vocab_cap is None
                    else bool(int(depth_target) < int(draft_vocab_cap))
                ),
                "proposed_matches_target": (
                    None
                    if depth_proposed is None or depth_target is None
                    else bool(int(depth_proposed) == int(depth_target))
                ),
            }
        )
    return {
        "cycle": int(cycle),
        "decode_offset": int(decode_offset),
        "root_position": int(root_position),
        "root_token": int(root_token),
        "active_budget": int(active_budget),
        "verify_budget": int(verify_budget),
        "accepted": int(accepted),
        "full_accept": bool(int(accepted) >= int(active_budget)),
        "first_rejected_depth": first_rejected_depth,
        "first_rejected_proposed_token": proposed,
        "first_rejected_target_top1_token": target,
        "first_rejected_target_in_draft_vocab_cap": (
            None if target is None or draft_vocab_cap is None else bool(int(target) < int(draft_vocab_cap))
        ),
        "first_rejected_reason": reason,
        "per_depth": per_depth,
    }


def _summarize_acceptance_diagnostics(
    cycles: Sequence[dict[str, Any]],
    *,
    accepted_lengths: Sequence[int],
    draft_vocab_cap: int | None,
) -> dict[str, Any]:
    accept_depth_hist: dict[str, int] = {}
    accept_depth_hist_mtp_verify_only: dict[str, int] = {}
    first_reject_depth_hist: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    policy_counts: dict[str, int] = {}
    per_depth: dict[str, dict[str, int]] = {}
    for accepted in accepted_lengths:
        key = str(int(accepted))
        accept_depth_hist[key] = accept_depth_hist.get(key, 0) + 1
    for cycle in cycles:
        policy = str(cycle.get("policy") or "mtp_verify")
        policy_counts[policy] = policy_counts.get(policy, 0) + 1
        if policy == "mtp_verify":
            key = str(int(cycle.get("accepted") or 0))
            accept_depth_hist_mtp_verify_only[key] = accept_depth_hist_mtp_verify_only.get(key, 0) + 1
        reason = str(cycle.get("first_rejected_reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        depth = cycle.get("first_rejected_depth")
        if depth is not None:
            key = str(int(depth))
            first_reject_depth_hist[key] = first_reject_depth_hist.get(key, 0) + 1
        for row in cycle.get("per_depth") or []:
            key = str(int(row.get("depth", 0)))
            bucket = per_depth.setdefault(
                key,
                {
                    "observed": 0,
                    "accepted": 0,
                    "rejected": 0,
                    "target_outside_draft_vocab_cap": 0,
                    "draft_top1_miss": 0,
                },
            )
            bucket["observed"] += 1
            if row.get("accepted"):
                bucket["accepted"] += 1
            else:
                bucket["rejected"] += 1
                if row.get("target_in_draft_vocab_cap") is False:
                    bucket["target_outside_draft_vocab_cap"] += 1
                elif row.get("proposed_matches_target") is False:
                    bucket["draft_top1_miss"] += 1
    zero_streaks = _zero_accept_streaks_from_diagnostic_cycles(cycles)
    return {
        "enabled": True,
        "draft_vocab_cap": draft_vocab_cap,
        "cycles": list(cycles),
        "summary": {
            "cycle_count": len(cycles),
            "policy_counts": policy_counts,
            "accept_depth_hist": accept_depth_hist,
            "accept_depth_hist_mtp_verify_only": accept_depth_hist_mtp_verify_only,
            "first_reject_depth_hist": first_reject_depth_hist,
            "first_reject_reason_counts": reason_counts,
            "zero_accept_streaks": zero_streaks,
            "max_zero_accept_streak": max(zero_streaks) if zero_streaks else 0,
            "per_depth": per_depth,
        },
    }


def _mtp_proposer_skip_unused_reads_enabled() -> bool:
    """Skip MTP proposer host reads/results that the persistent chain discards."""

    return _env_flag("HIPENGINE_MTP_PROPOSER_SKIP_UNUSED_READS", True)


def _mtp_skip_canonicalize_after_verify_enabled() -> bool:
    """Keep verifier-shaped scratch live after MTP verify cycles when requested."""

    return _env_flag("HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY", True)


def _mtp_overlap_verify_commit_proposer_enabled() -> bool:
    """Run proposer update on a side stream while verifier commit drains."""

    return _env_flag("HIPENGINE_MTP_OVERLAP_VERIFY_COMMIT_PROPOSER", False)


class _OptionalHipStream:
    def __init__(self, runtime: Any, *, enabled: bool) -> None:
        self.runtime = runtime
        self.enabled = bool(enabled)
        self.stream = 0

    def __enter__(self) -> int:
        if self.enabled:
            self.stream = int(self.runtime.stream_create())
        return self.stream

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.stream:
            self.runtime.stream_synchronize(self.stream)
            self.runtime.stream_destroy(self.stream)
            self.stream = 0


def _capture_tensor(buffer: DeviceBuffer, rows: int, hidden: int) -> Tensor:
    return Tensor.from_handle(buffer.ptr, (int(rows), int(hidden)), DType.BF16, Device("hip", 0))


def _read_capture_row(buffer: DeviceBuffer, row: int, hidden: int) -> np.ndarray:
    host = np.zeros((1, int(hidden)), dtype=np.uint16)
    offset = int(row) * int(hidden) * DType.BF16.itemsize
    view = DeviceBuffer(ptr=buffer.ptr + offset, nbytes=host.nbytes)
    copy_device_to_host(host_array_ptr(host), view, host.nbytes)
    return host


def _run_ar_baseline(
    model: Path,
    prompt_tokens: Sequence[int],
    *,
    decode_tokens: int,
    backend: str,
    runner: Qwen35ParoNextTokenRunner | None = None,
) -> tuple[list[int], dict[str, Any]]:
    if runner is None:
        runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_tokens) + int(decode_tokens) + 2
    started = time.perf_counter()
    generated: list[int] = []
    prefill_seconds = 0.0
    decode_seconds = 0.0
    with Qwen35ParoResidentSession(runner, max_sequence_length=max_sequence) as session:
        next_result = None
        prefill_started = time.perf_counter()
        for pos, token in enumerate(prompt_tokens):
            next_result = session.step(int(token), position=pos, sample=(pos == len(prompt_tokens) - 1))
        if next_result is None:
            raise RuntimeError("prompt did not produce a root token")
        prefill_seconds = time.perf_counter() - prefill_started
        root = int(next_result.token_id)
        context = len(prompt_tokens)
        decode_started = time.perf_counter()
        for _offset in range(int(decode_tokens)):
            generated.append(root)
            next_result = session.step(root, position=context, sample=True)
            if next_result is None:
                raise RuntimeError("AR decode step produced no token")
            root = int(next_result.token_id)
            context += 1
        decode_seconds = time.perf_counter() - decode_started
    seconds = time.perf_counter() - started
    return generated, {
        "seconds": seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "tok_s": len(generated) / seconds if seconds > 0 else None,
        "decode_tok_s": len(generated) / decode_seconds if decode_seconds > 0 else None,
    }


def _target_batch(root: int, context: int, candidates: Sequence[int], active_count: int, candidate_budget: int | None = None) -> TargetVerifyBatch:
    budget = int(candidate_budget if candidate_budget is not None else active_count)
    return TargetVerifyBatch.from_draft(
        compile_mtp_chain(
            [MtpDraftRequest(request_id=0, root_position=int(context), candidate_tokens=tuple(int(x) for x in candidates), active_count=int(active_count))],
            candidate_budget=budget,
        ),
        root_tokens=(int(root),),
        root_positions=(int(context),),
    )


def _run_spec_smoke(
    model: Path,
    prompt_tokens: Sequence[int],
    *,
    decode_tokens: int,
    candidate_budget: int,
    backend: str,
    chain_attn_mode: str,
    tree_mode: str = "chain",
    tree_top_k: int = 2,
    confidence_threshold: float = 0.0,
    draft_p_min: float = 0.0,
    runner: Qwen35ParoNextTokenRunner | None = None,
) -> tuple[list[int], dict[str, Any]]:
    if tree_mode not in {"chain", "branching_topk"}:
        raise ValueError("tree_mode must be chain or branching_topk")
    if runner is None:
        runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_tokens) + int(decode_tokens) + int(candidate_budget) + 4
    max_batch_size = int(candidate_budget) + 1
    generated: list[int] = []
    accepted_lengths: list[int] = []
    proposal_trace: list[dict[str, Any]] = []
    # Top-k oracle: per cycle, the rank of the target's chosen depth-1 token in
    # the MTP head's top-k (1 = top-1 match/accepted, 2..K = root-branch
    # rescuable, 0 = absent from top-k). Bounds the tree-draft acceptance gain.
    oracle_depth1_ranks: list[int] = []
    # Tree-path curve instrumentation.
    accept_depth_hist: dict[int, int] = {}
    gated_to_chain_cycles = 0
    gpu_accept_match_all = True
    tree_nodes_total = 0
    # Per-position p-min truncation instrumentation.
    pmin_ar_cycles = 0            # truncated all the way to AR (no verify)
    pmin_truncated_cycles = 0     # truncated below active_budget but still verified
    wasted_verify_zero_cycles = 0  # ran a multi-draft verify that accepted 0
    pmin_eff_budgets: list[int] = []
    proposal_seconds = 0.0
    verify_seconds = 0.0
    target_forward_calls = 0
    capture_rows = max_sequence + int(candidate_budget) + 2
    capture_buf: DeviceBuffer | None = None
    started = time.perf_counter()
    canonicalize_after_verify = not _mtp_skip_canonicalize_after_verify_enabled()
    with Qwen35ParoResidentSession(runner, max_sequence_length=max_sequence, max_batch_size=max_batch_size) as session:
        hidden = int(session.config.hidden_size)
        capture_layer_id = int(session.layer_limit) - 1
        capture_buf = malloc(capture_rows * hidden * DType.BF16.itemsize, runtime=session.runtime)
        capture = _capture_tensor(capture_buf, capture_rows, hidden)
        try:
            next_result = None
            for pos, token in enumerate(prompt_tokens):
                next_result = session.step_with_hidden_taps(
                    int(token),
                    position=pos,
                    capture_layer_ids=(capture_layer_id,),
                    capture_hidden_concat=capture,
                    capture_row=pos,
                    sample=(pos == len(prompt_tokens) - 1),
                )
            if next_result is None:
                raise RuntimeError("prompt did not produce a root token")
            root = int(next_result.token_id)
            context = len(prompt_tokens)
            previous_hidden_row = context - 1
            cycles = 0
            while len(generated) < int(decode_tokens):
                remaining = int(decode_tokens) - len(generated)
                active_budget = min(int(candidate_budget), max(0, remaining - 1))
                # The native MTP chain proposer only compiles allowed budgets
                # {1,2,3,5}; near the decode tail active_budget can land on 4.
                # Snap down to the largest allowed budget so the proposer never
                # compiles an invalid chain (drafting fewer tokens is safe --
                # only accepted tokens are committed and the tail is truncated).
                if active_budget > 0:
                    allowed = [b for b in MTP_CHAIN_CANDIDATE_BUDGETS if b <= active_budget]
                    active_budget = max(allowed) if allowed else 0
                if active_budget <= 0:
                    step_result = session.step_with_hidden_taps(
                        root,
                        position=context,
                        capture_layer_ids=(capture_layer_id,),
                        capture_hidden_concat=capture,
                        capture_row=context,
                        sample=True,
                    )
                    if step_result is None:
                        raise RuntimeError("terminal AR step produced no root")
                    generated.append(root)
                    root = int(step_result.token_id)
                    previous_hidden_row = context
                    context += 1
                    continue
                cycles += 1
                target_hidden = _read_capture_row(capture_buf, previous_hidden_row, hidden)
                t_prop = time.perf_counter()
                proposal = run_native_mtp_proposal(
                    model,
                    root_token=root,
                    root_position=context,
                    draft_budget=active_budget,
                    torch_compare=False,
                    target_hidden_bits_override=target_hidden,
                )
                proposal_seconds += time.perf_counter() - t_prop
                candidates = [int(token) for token in proposal["candidate_tokens"][:active_budget]]
                draft = proposal["draft_batch"]
                candidate_topk = proposal.get("candidate_topk") or []
                candidate_topk_values = proposal.get("candidate_topk_values") or []
                # Per-position p-min truncation (chain path; DFlash --draft-p-min
                # analog): keep leading drafts while the head's top-1 prob proxy
                # >= p_min, stop at the first low-confidence depth. eff_budget is
                # snapped to the allowed MTP budgets. Exactness is preserved --
                # the chain still commits only target-verified tokens + the
                # target correction, so the sequence stays identical to AR.
                pmin_eff_budget = active_budget
                if draft_p_min > 0.0 and tree_mode == "chain" and candidate_topk_values:
                    keep = 0
                    for d in range(active_budget):
                        p1 = _topk_softmax_top1(candidate_topk_values[d]) if d < len(candidate_topk_values) else 1.0
                        if p1 >= draft_p_min:
                            keep += 1
                        else:
                            break
                    allowed_pmin = [b for b in MTP_CHAIN_CANDIDATE_BUDGETS if b <= keep]
                    pmin_eff_budget = max(allowed_pmin) if allowed_pmin else 0
                    if pmin_eff_budget < active_budget:
                        pmin_truncated_cycles += 1
                    candidates = candidates[:pmin_eff_budget]
                pmin_eff_budgets.append(int(pmin_eff_budget))
                if draft_p_min > 0.0 and tree_mode == "chain" and pmin_eff_budget == 0:
                    # Head is unsure at depth 1 -> plain AR step, no speculation
                    # this cycle (the 'draft-N-accept-0' -> AR conversion).
                    step_result = session.step_with_hidden_taps(
                        root,
                        position=context,
                        capture_layer_ids=(capture_layer_id,),
                        capture_hidden_concat=capture,
                        capture_row=context,
                        sample=True,
                    )
                    if step_result is None:
                        raise RuntimeError("p-min AR step produced no token")
                    pmin_ar_cycles += 1
                    accepted_lengths.append(0)
                    accept_depth_hist[0] = accept_depth_hist.get(0, 0) + 1
                    generated.append(root)
                    previous_hidden_row = context
                    context += 1
                    root = int(step_result.token_id)
                    continue
                # Online whole-cycle confidence gate: if the depth-1 top-1
                # probability proxy is below threshold, fall back from the wide
                # branching tree to the cheaper chain (top-1) verify for this
                # cycle -- i.e. only spend the wider tree on confident cycles.
                gate_low_confidence = False
                if confidence_threshold > 0.0 and candidate_topk_values:
                    p0 = _topk_softmax_top1(candidate_topk_values[0])
                    if p0 < confidence_threshold:
                        gate_low_confidence = True
                        gated_to_chain_cycles += 1
                use_tree = tree_mode == "branching_topk" and not gate_low_confidence and len(candidate_topk) >= 1
                t_verify = time.perf_counter()
                if use_tree:
                    max_depth = min(active_budget, len(candidate_topk), len(candidate_topk_values))
                    compiled = _build_branching_topk_tree_target_batch(
                        root_token=root,
                        root_position=context,
                        topk_tokens=candidate_topk,
                        topk_values=candidate_topk_values,
                        candidate_budget=active_budget,
                        tree_top_k=int(tree_top_k),
                        max_depth=max_depth,
                    )
                    target_batch = compiled.target_batch
                    tree_nodes_total += int(compiled.active_count)
                    verify = session.verify_tree_bulk_and_commit(
                        target_batch,
                        base_slot=0,
                        capture_layer_ids=(capture_layer_id,),
                        capture_hidden_concat=capture,
                        capture_row_start=context,
                        canonicalize_after=canonicalize_after_verify,
                    )
                    accepted_tokens = list(verify.accepted_tokens)
                else:
                    # chain (top-1) path. pmin_eff_budget == active_budget unless
                    # per-position p-min truncated the chain this cycle.
                    target_batch = _target_batch(root, context, candidates, pmin_eff_budget, candidate_budget=pmin_eff_budget)
                    verify = session.verify_chain_bulk_and_commit(
                        target_batch,
                        base_slot=0,
                        capture_layer_ids=(capture_layer_id,),
                        capture_hidden_concat=capture,
                        capture_row_start=context,
                        chain_attn_mode=chain_attn_mode,
                        canonicalize_after=canonicalize_after_verify,
                    )
                    accepted_tokens = candidates[: int(verify.accepted_count)]
                verify_seconds += time.perf_counter() - t_verify
                target_forward_calls += int(verify.target_forward_calls)
                accepted = int(verify.accepted_count)
                accepted_lengths.append(accepted)
                accept_depth_hist[accepted] = accept_depth_hist.get(accepted, 0) + 1
                if accepted == 0 and not use_tree and pmin_eff_budget > 0:
                    wasted_verify_zero_cycles += 1
                if verify.gpu_accept_match_cpu is not None:
                    gpu_accept_match_all = gpu_accept_match_all and bool(verify.gpu_accept_match_cpu)
                # Depth-1 top-k oracle: where does the target's chosen next token
                # rank in the MTP head's depth-1 top-k?  Recovered uniformly for
                # chain and tree: if any draft token was accepted the target's
                # next token is accepted_tokens[0]; otherwise it is the
                # correction (next_token / commit_token). verify.target_top1 is
                # empty in tree mode, so do not depend on it here.
                if candidate_topk:
                    if accepted_tokens:
                        target_next = int(accepted_tokens[0])
                    elif verify.next_token is not None:
                        target_next = int(verify.next_token)
                    else:
                        target_next = int(verify.commit_token)
                    d1_topk = [int(x) for x in candidate_topk[0]]
                    rank = (d1_topk.index(target_next) + 1) if target_next in d1_topk else 0
                    oracle_depth1_ranks.append(rank)
                committed = [root, *accepted_tokens]
                generated.extend(committed)
                bonus = int(verify.next_token) if verify.next_token is not None else int(verify.target_top1[min(accepted, len(verify.target_top1) - 1)])
                if len(proposal_trace) < 16:
                    proposal_trace.append(
                        {
                            "cycle": cycles,
                            "root_position": context,
                            "root_token": root,
                            "draft_candidates": candidates,
                            "target_top1_path": list(map(int, verify.target_top1[: 1 + active_budget])),
                            "target_top1_values": list(map(float, verify.target_top1_values[: 1 + active_budget])),
                            "accepted": accepted,
                            "committed_tokens": committed,
                            "bonus_token": bonus,
                            "target_parent_rows": list(map(int, target_batch.parent_rows)),
                            "verify_graph": verify.graph,
                            "gpu_accept_match_cpu": bool(verify.gpu_accept_match_cpu) if verify.gpu_accept_match_cpu is not None else None,
                            "proposal_native_seconds": float(proposal["native_seconds"]),
                            "proposal_draft_batch": draft,
                        }
                    )
                previous_hidden_row = context + len(committed) - 1
                context += len(committed)
                root = bonus
        finally:
            if capture_buf is not None:
                free(capture_buf, runtime=session.runtime)
    seconds = time.perf_counter() - started
    n_oracle = len(oracle_depth1_ranks)
    top1 = sum(1 for r in oracle_depth1_ranks if r == 1)
    in_topk = sum(1 for r in oracle_depth1_ranks if r >= 1)
    rescuable = sum(1 for r in oracle_depth1_ranks if r >= 2)
    rank_hist: dict[int, int] = {}
    for r in oracle_depth1_ranks:
        rank_hist[r] = rank_hist.get(r, 0) + 1
    topk_oracle = {
        "cycles": n_oracle,
        "k": 8,
        "depth1_top1_match": top1,
        "depth1_top1_rate": (top1 / n_oracle) if n_oracle else None,
        "depth1_in_topk": in_topk,
        "depth1_in_topk_rate": (in_topk / n_oracle) if n_oracle else None,
        "depth1_rescuable_2_to_k": rescuable,
        "depth1_rescuable_rate": (rescuable / n_oracle) if n_oracle else None,
        "rank_histogram": {str(k): rank_hist[k] for k in sorted(rank_hist)},
        "note": "rank 1 = chain already accepts; 2..k = root-branch tree could rescue; 0 = target token absent from MTP top-k (unrescuable at depth 1).",
    }
    n_cycles = len(accepted_lengths)
    avg_accepted = (sum(accepted_lengths) / n_cycles) if n_cycles else 0.0
    # alpha (per-token accept rate) = accepted draft tokens / drafted tokens.
    # Drafted tokens per cycle = active_budget (chain) or tree node budget; the
    # smoke runs a fixed candidate_budget so we normalize by candidate_budget.
    alpha = (sum(accepted_lengths) / (n_cycles * int(candidate_budget))) if n_cycles and candidate_budget > 0 else 0.0
    # Visible tokens/cycle = committed (root + accepted) = 1 + avg_accepted; the
    # reviewer-preferred comparable metric vs llama.cpp's p-min-inflated alpha.
    visible_tokens_per_cycle = 1.0 + avg_accepted
    return generated[: int(decode_tokens)], {
        "seconds": seconds,
        "tok_s": int(decode_tokens) / seconds if seconds > 0 else None,
        "topk_oracle": topk_oracle,
        "proposal_seconds": proposal_seconds,
        "verify_seconds": verify_seconds,
        "verify_seconds_per_cycle": (verify_seconds / n_cycles) if n_cycles else None,
        "accepted_lengths": accepted_lengths,
        "acceptance_rate": alpha,
        "alpha": alpha,
        "avg_accepted": avg_accepted,
        "visible_tokens_per_cycle": visible_tokens_per_cycle,
        "cycles": n_cycles,
        "tree_mode": tree_mode,
        "tree_top_k": int(tree_top_k),
        "confidence_threshold": float(confidence_threshold),
        "gated_to_chain_cycles": gated_to_chain_cycles,
        "tree_nodes_total": tree_nodes_total,
        "draft_p_min": float(draft_p_min),
        "pmin_ar_cycles": pmin_ar_cycles,
        "pmin_truncated_cycles": pmin_truncated_cycles,
        "wasted_verify_zero_cycles": wasted_verify_zero_cycles,
        "zero_accept_cycles": accept_depth_hist.get(0, 0),
        "zero_accept_rate": (accept_depth_hist.get(0, 0) / n_cycles) if n_cycles else None,
        "avg_eff_budget": (sum(pmin_eff_budgets) / len(pmin_eff_budgets)) if pmin_eff_budgets else None,
        "avg_verify_rows": ((sum(b for b in pmin_eff_budgets if b > 0) + sum(1 for b in pmin_eff_budgets if b > 0)) / max(1, sum(1 for b in pmin_eff_budgets if b > 0))) if any(b > 0 for b in pmin_eff_budgets) else None,
        "accept_depth_histogram": {str(k): accept_depth_hist[k] for k in sorted(accept_depth_hist)},
        "gpu_accept_match_cpu": gpu_accept_match_all,
        "proposal_trace_sample": proposal_trace,
        "target_forward_calls": target_forward_calls,
        "chain_attn_mode": chain_attn_mode,
        "canonicalize_after_verify": bool(canonicalize_after_verify),
        "note": "Correctness smoke only: proposal hidden rows are copied D2H and MTP weights are reloaded per proposal call.",
    }


def _run_acceptance_curve(
    model: Path,
    prompt_tokens: Sequence[int],
    *,
    decode_tokens: int,
    candidate_budget: int,
    backend: str,
    chain_attn_mode: str,
    tree_top_ks: Sequence[int],
    confidence_thresholds: Sequence[float],
) -> dict[str, Any]:
    """Realized-acceptance curve over (branch width, confidence threshold).

    Shares ONE resident target runner across the AR baseline and every spec
    config so only one copy of the 35B target stays in VRAM.  Each config runs
    an independent committed decode (the proposer reloads MTP weights per
    proposal call -- correctness-first, not a tok/s path).  Draft depth is the
    accept-depth histogram already returned per config.
    """
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    ar_tokens, ar = _run_ar_baseline(
        model, prompt_tokens, decode_tokens=int(decode_tokens), backend=backend, runner=runner
    )
    configs: list[tuple[str, int, float]] = [("chain", 1, 0.0)]
    for k in tree_top_ks:
        for thr in confidence_thresholds:
            configs.append(("branching_topk", int(k), float(thr)))
    curve: list[dict[str, Any]] = []
    for mode, k, thr in configs:
        spec_tokens, spec = _run_spec_smoke(
            model,
            prompt_tokens,
            decode_tokens=int(decode_tokens),
            candidate_budget=int(candidate_budget),
            backend=backend,
            chain_attn_mode=chain_attn_mode,
            tree_mode=mode,
            tree_top_k=int(k),
            confidence_threshold=float(thr),
            runner=runner,
        )
        curve.append(
            {
                "tree_mode": mode,
                "tree_top_k": int(k),
                "confidence_threshold": float(thr),
                "alpha": spec["alpha"],
                "avg_accepted": spec["avg_accepted"],
                "visible_tokens_per_cycle": spec["visible_tokens_per_cycle"],
                "cycles": spec["cycles"],
                "accept_depth_histogram": spec["accept_depth_histogram"],
                "gated_to_chain_cycles": spec["gated_to_chain_cycles"],
                "tree_nodes_total": spec["tree_nodes_total"],
                "exact_ar_match": spec_tokens == ar_tokens,
                "gpu_accept_match_cpu": spec["gpu_accept_match_cpu"],
                "verify_seconds_per_cycle": spec["verify_seconds_per_cycle"],
                "tok_s_diagnostic": spec["tok_s"],
                "topk_oracle": spec["topk_oracle"],
            }
        )
    return {
        "status": "passed",
        "performance_claim": False,
        "model": str(model),
        "backend": backend,
        "prompt_tokens": list(prompt_tokens),
        "decode_tokens": int(decode_tokens),
        "candidate_budget": int(candidate_budget),
        "chain_attn_mode": chain_attn_mode,
        "ar_tokens": ar_tokens,
        "ar": ar,
        "ar_tok_s": ar["tok_s"],
        "acceptance_curve": curve,
        "note": (
            "Realized MTP acceptance curve (correctness-first). alpha = accepted / "
            "(cycles*candidate_budget); visible_tokens_per_cycle = 1+avg_accepted is "
            "the comparable metric vs llama.cpp p-min-inflated alpha. Per (B+1)/C_B, "
            "this does NOT beat AR until the #98->#105/#101 dispatch floor lands."
        ),
    }


# #98 verify launch model (per cycle): fixed + marginal*rows, rows = eff_budget+1.
_VERIFY_FIXED_LAUNCHES = 711.0
_VERIFY_MARGINAL_LAUNCHES_PER_ROW = 240.0
_AR_LAUNCHES = 920.0  # plain AR decode step (#98)


def _run_pmin_sweep(
    model: Path,
    prompt_tokens: Sequence[int],
    *,
    decode_tokens: int,
    backend: str,
    chain_attn_mode: str,
    budgets: Sequence[int],
    p_mins: Sequence[float],
) -> dict[str, Any]:
    """Per-position p-min truncation x B-down sweep on the MTP chain path.

    Shares ONE resident target runner across the AR baseline and every config.
    Reports zero-accept-cycle reduction, avg-accept, an estimated C_B from the
    #98 launch model, and the diagnostic reload tok/s.  p-min is gated on a
    restricted top-8 softmax proxy (overestimates the true full-vocab top-1
    probability), so a proxy threshold is milder than the same numeric value on
    true vocab probability -- the measured truncation rates are the grounded
    comparison, not the bare threshold.
    """
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    ar_tokens, ar = _run_ar_baseline(
        model, prompt_tokens, decode_tokens=int(decode_tokens), backend=backend, runner=runner
    )
    sweep: list[dict[str, Any]] = []
    baseline_zero_rate: dict[int, float] = {}
    for B in budgets:
        for pmin in p_mins:
            spec_tokens, spec = _run_spec_smoke(
                model,
                prompt_tokens,
                decode_tokens=int(decode_tokens),
                candidate_budget=int(B),
                backend=backend,
                chain_attn_mode=chain_attn_mode,
                tree_mode="chain",
                draft_p_min=float(pmin),
                runner=runner,
            )
            n = int(spec["cycles"]) or 1
            avg_rows = spec["avg_verify_rows"]
            verify_cycles = n - int(spec["pmin_ar_cycles"])
            # Estimated verify launches/cycle from the #98 model (verify cycles
            # carry fixed+marginal*rows; pmin-AR cycles carry one AR step).
            est_verify_launches = (
                (_VERIFY_FIXED_LAUNCHES + _VERIFY_MARGINAL_LAUNCHES_PER_ROW * avg_rows) if avg_rows else 0.0
            )
            est_cycle_launches = (
                (verify_cycles * est_verify_launches + int(spec["pmin_ar_cycles"]) * _AR_LAUNCHES) / n
            )
            # C_B proxy: estimated verify launches per VISIBLE token committed.
            visible_per_cycle = float(spec["visible_tokens_per_cycle"]) or 1.0
            est_launches_per_visible_token = est_cycle_launches / visible_per_cycle
            row = {
                "candidate_budget": int(B),
                "draft_p_min": float(pmin),
                "alpha": round(float(spec["alpha"]), 4),
                "avg_accepted": round(float(spec["avg_accepted"]), 3),
                "visible_tokens_per_cycle": round(visible_per_cycle, 3),
                "cycles": n,
                "zero_accept_cycles": int(spec["zero_accept_cycles"]),
                "zero_accept_rate": round(float(spec["zero_accept_rate"]), 4) if spec["zero_accept_rate"] is not None else None,
                "wasted_verify_zero_cycles": int(spec["wasted_verify_zero_cycles"]),
                "pmin_ar_cycles": int(spec["pmin_ar_cycles"]),
                "pmin_truncated_cycles": int(spec["pmin_truncated_cycles"]),
                "avg_eff_budget": round(float(spec["avg_eff_budget"]), 3) if spec["avg_eff_budget"] is not None else None,
                "avg_verify_rows": round(float(avg_rows), 3) if avg_rows else None,
                "est_verify_launches_per_cycle": round(est_cycle_launches, 1),
                "est_launches_per_visible_token": round(est_launches_per_visible_token, 1),
                "exact_ar_match": spec_tokens == ar_tokens,
                "gpu_accept_match_cpu": spec["gpu_accept_match_cpu"],
                "accept_depth_histogram": spec["accept_depth_histogram"],
                "tok_s_diagnostic": spec["tok_s"],
            }
            if float(pmin) == 0.0:
                baseline_zero_rate[int(B)] = row["zero_accept_rate"] or 0.0
            sweep.append(row)
    # Zero-accept-cycle reduction vs the p_min=0 baseline at the same B, and
    # wasted-multi-row-verify reduction (the cost p-min actually removes).
    for row in sweep:
        base = baseline_zero_rate.get(row["candidate_budget"])
        row["zero_accept_rate_reduction_vs_pmin0"] = (
            round(base - row["zero_accept_rate"], 4) if base is not None and row["zero_accept_rate"] is not None else None
        )
    return {
        "status": "passed",
        "performance_claim": False,
        "model": str(model),
        "backend": backend,
        "prompt_tokens": list(prompt_tokens),
        "decode_tokens": int(decode_tokens),
        "chain_attn_mode": chain_attn_mode,
        "ar_tokens": ar_tokens,
        "ar": ar,
        "ar_tok_s": ar["tok_s"],
        "pmin_sweep": sweep,
        "verify_launch_model": {
            "fixed_launches": _VERIFY_FIXED_LAUNCHES,
            "marginal_launches_per_row": _VERIFY_MARGINAL_LAUNCHES_PER_ROW,
            "ar_launches": _AR_LAUNCHES,
            "source": "#98 census (benchmarks/results/2026-06-09-hipengine-m16-ar-verify-launch-census.json)",
        },
        "note": (
            "Per-position p-min truncation x B-down sweep (chain, correctness-first). "
            "p-min uses a restricted top-8 softmax proxy (overestimates true vocab "
            "top-1 prob), so the proxy threshold is milder than the same value on "
            "true prob -- read the measured truncation/zero-accept rates, not the bare "
            "threshold. C_B is estimated from the #98 launch model; reload tok/s is "
            "diagnostic only."
        ),
    }


def _run_spec_persistent_device(
    model: Path,
    prompt_tokens: Sequence[int],
    *,
    decode_tokens: int,
    candidate_budget: int,
    active_budget_cap: int = 0,
    backend: str,
    chain_attn_mode: str,
    graph_mode: str = "off",
    draft_p_min: float = 0.0,
    tree_mode: str = "chain",
    tree_top_k: int = 2,
    confidence_threshold: float = 0.0,
    rocprof_warmup_cycles: int = 0,
    rocprof_verify_cycles: int = 0,
    acceptance_diagnostics: bool = False,
    ar_fallback_zero_streak: int = 0,
    ar_fallback_after_mtp_cycles: int = 0,
    ar_fallback_tokens: int = 1,
    ar_fallback_until_end: bool = False,
) -> tuple[list[int], dict[str, Any]]:
    confidence_threshold = float(confidence_threshold)
    active_budget_cap = int(active_budget_cap)
    ar_fallback_zero_streak = int(ar_fallback_zero_streak)
    ar_fallback_after_mtp_cycles = int(ar_fallback_after_mtp_cycles)
    ar_fallback_tokens = int(ar_fallback_tokens)
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if active_budget_cap < 0:
        raise ValueError("active_budget_cap must be >= 0")
    if active_budget_cap > int(candidate_budget):
        raise ValueError("active_budget_cap cannot exceed candidate_budget")
    if ar_fallback_zero_streak < 0:
        raise ValueError("ar_fallback_zero_streak must be >= 0")
    if ar_fallback_after_mtp_cycles < 0:
        raise ValueError("ar_fallback_after_mtp_cycles must be >= 0")
    if ar_fallback_tokens < 1:
        raise ValueError("ar_fallback_tokens must be >= 1")
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_tokens) + int(decode_tokens) + int(candidate_budget) + 4
    generated: list[int] = []
    accepted_lengths: list[int] = []
    proposal_trace: list[dict[str, Any]] = []
    verify_seconds = 0.0
    proposal_prefill_seconds = 0.0
    proposal_decode_update_seconds = 0.0
    proposal_snapshot_saves = 0
    proposal_snapshot_skips = 0
    ar_fallback_cycles = 0
    ar_fallback_tokens_committed = 0
    ar_fallback_seconds = 0.0
    ar_fallback_proposer_update_seconds = 0.0
    confidence_ar_fallback_cycles = 0
    confidence_ar_fallback_tokens = 0
    acceptance_diagnostic_cycles: list[dict[str, Any]] = []
    draft_vocab_cap: int | None = None
    capture_rows = max_sequence + 2
    capture_buf: DeviceBuffer | None = None
    started = time.perf_counter()
    active_budgets: list[int] = []
    skip_unused_proposer_reads = _mtp_proposer_skip_unused_reads_enabled()
    canonicalize_after_verify = not _mtp_skip_canonicalize_after_verify_enabled()
    overlap_verify_commit_proposer = _mtp_overlap_verify_commit_proposer_enabled()
    # Always load libroctx64 so range_push/pop markers fire even when the
    # selected-region window is off (rocprofv3 1.1.0 path). The resume/pause
    # path is still gated on rocprof_verify_cycles>0 below.
    rocprof_window = _RoctxProfilerWindow(enabled=True)
    rocprof_resume_window_enabled = int(rocprof_verify_cycles) > 0
    rocprof_window_meta: dict[str, Any] = {
        "enabled": bool(rocprof_resume_window_enabled),
        "warmup_cycles": int(rocprof_warmup_cycles),
        "verify_cycles": int(rocprof_verify_cycles),
        "profiled_cycle_range": None,
        "profiled_cycle_seconds": None,
    }
    rocprof_window_started = False
    rocprof_window_done = False  # one-shot — keep the profiled region a single contiguous span
    rocprof_window_first_cycle: int | None = None
    rocprof_window_last_cycle: int | None = None
    rocprof_window_t_start: float | None = None
    rocprof_window_t_end: float | None = None
    # Per-cycle wall-clock ns boundaries. Used by the rocprof post-processor on
    # rocprofv3 hosts where --selected-regions is broken so it can filter the
    # kernel trace by verifier-cycle window via timestamp arithmetic.
    cycle_marker_ns: list[tuple[int, int, int]] = []
    with Qwen35ParoResidentSession(runner, max_sequence_length=max_sequence, max_batch_size=int(candidate_budget) + 1) as session:
        hidden = int(session.config.hidden_size)
        capture_layer_id = int(session.layer_limit) - 1
        capture_buf = malloc(capture_rows * hidden * DType.BF16.itemsize, runtime=session.runtime)
        capture = _capture_tensor(capture_buf, capture_rows, hidden)
        verifier_no_capture = Tensor.from_handle(0, (int(candidate_budget) + 1, 0), DType.BF16, Device("hip", 0))
        try:
            next_result = None
            for pos, token in enumerate(prompt_tokens):
                next_result = session.step_with_hidden_taps(
                    int(token),
                    position=pos,
                    capture_layer_ids=(capture_layer_id,),
                    capture_hidden_concat=capture,
                    capture_row=pos,
                    sample=(pos == len(prompt_tokens) - 1),
                )
            if next_result is None:
                raise RuntimeError("prompt did not produce a root token")
            root = int(next_result.token_id)
            context = len(prompt_tokens)
            with NativeMtpChainProposer(
                model,
                max_positions=max_sequence + int(decode_tokens) + 4,
                max_mtp_tokens=len(prompt_tokens) + 2 * int(decode_tokens) + 8,
                runtime=session.runtime,
            ) as proposer, _OptionalHipStream(session.runtime, enabled=overlap_verify_commit_proposer) as proposer_update_stream:
                draft_vocab_cap = int(proposer.draft_vocab)
                prefill_started = time.perf_counter()
                proposer.prefill_from_target_hidden_rows(
                    prompt_tokens,
                    capture_base_ptr=capture_buf.ptr,
                    seed_token=root,
                    read_expert_topk=not skip_unused_proposer_reads,
                    read_lm_head_value=not skip_unused_proposer_reads,
                )
                proposal_prefill_seconds += time.perf_counter() - prefill_started
                cycles = 0
                mtp_verify_cycles_completed = 0
                zero_accept_streak = 0
                decode_started = time.perf_counter()
                while len(generated) < int(decode_tokens):
                    decode_offset = len(generated)
                    remaining = int(decode_tokens) - len(generated)
                    active_budget = min(int(candidate_budget), max(0, remaining - 1))
                    if active_budget_cap > 0:
                        active_budget = min(active_budget, active_budget_cap)
                    if active_budget <= 0:
                        step_result = session.step_with_hidden_taps(
                            root,
                            position=context,
                            capture_layer_ids=(capture_layer_id,),
                            capture_hidden_concat=capture,
                            capture_row=context,
                            sample=True,
                        )
                        if step_result is None:
                            raise RuntimeError("terminal AR step produced no root")
                        generated.append(root)
                        root = int(step_result.token_id)
                        context += 1
                        break
                    force_after_cycle = (
                        ar_fallback_after_mtp_cycles > 0
                        and mtp_verify_cycles_completed >= ar_fallback_after_mtp_cycles
                    )
                    force_after_zero_streak = (
                        ar_fallback_zero_streak > 0
                        and zero_accept_streak >= ar_fallback_zero_streak
                    )
                    if force_after_cycle or force_after_zero_streak:
                        fallback_reason = (
                            "ar_fallback_after_mtp_cycles"
                            if force_after_cycle
                            else "ar_fallback_zero_accept_streak"
                        )
                        fallback_window = (
                            int(decode_tokens) - len(generated)
                            if ar_fallback_until_end
                            else min(int(ar_fallback_tokens), int(decode_tokens) - len(generated))
                        )
                        for _fallback_idx in range(fallback_window):
                            if len(generated) >= int(decode_tokens):
                                break
                            cycles += 1
                            decode_offset = len(generated)
                            cycle_t_ns_start = time.perf_counter_ns()
                            rocprof_window.range_push(f"mtp_ar_fallback_cycle_{cycles}")
                            fallback_started = time.perf_counter()
                            streak_before_fallback = int(zero_accept_streak)
                            step_result = session.step(
                                root,
                                position=context,
                                sample=True,
                            )
                            if step_result is None:
                                raise RuntimeError("AR fallback step produced no root")
                            bonus = int(step_result.token_id)
                            committed = [root]
                            generated.extend(committed)
                            accepted_lengths.append(0)
                            active_budgets.append(0)
                            ar_fallback_cycles += 1
                            ar_fallback_tokens_committed += 1
                            if acceptance_diagnostics:
                                acceptance_diagnostic_cycles.append(
                                    {
                                        "cycle": int(cycles),
                                        "decode_offset": int(decode_offset),
                                        "root_position": int(context),
                                        "root_token": int(root),
                                        "policy": "ar_fallback",
                                        "active_budget": 0,
                                        "verify_budget": 0,
                                        "accepted": 0,
                                        "full_accept": False,
                                        "first_rejected_depth": None,
                                        "first_rejected_proposed_token": None,
                                        "first_rejected_target_top1_token": int(bonus),
                                        "first_rejected_target_in_draft_vocab_cap": (
                                            None if draft_vocab_cap is None else bool(int(bonus) < int(draft_vocab_cap))
                                        ),
                                        "first_rejected_reason": fallback_reason,
                                        "zero_accept_streak_before_fallback": streak_before_fallback,
                                        "mtp_verify_cycles_before_fallback": int(mtp_verify_cycles_completed),
                                        "fallback_window_tokens": int(fallback_window),
                                        "fallback_until_end": bool(ar_fallback_until_end),
                                        "per_depth": [],
                                    }
                                )
                            if len(proposal_trace) < 16:
                                proposal_trace.append(
                                    {
                                        "cycle": int(cycles),
                                        "policy": "ar_fallback",
                                        "root_position": int(context),
                                        "root_token": int(root),
                                        "accepted": 0,
                                        "committed_tokens": committed,
                                        "bonus_token": int(bonus),
                                        "zero_accept_streak_before_fallback": streak_before_fallback,
                                        "mtp_verify_cycles_before_fallback": int(mtp_verify_cycles_completed),
                                        "fallback_reason": fallback_reason,
                                        "fallback_until_end": bool(ar_fallback_until_end),
                                        "proposer_cache_len_before_update": int(proposer.cache_len),
                                    }
                                )
                            if (not ar_fallback_until_end) and len(generated) < int(decode_tokens):
                                update_started = time.perf_counter()
                                rocprof_window.range_push(f"mtp_ar_fallback_proposer_update_{cycles}")
                                proposer.advance_with_previous_hidden(
                                    input_token=bonus,
                                    position=proposer.position + 1,
                                    read_expert_topk=not skip_unused_proposer_reads,
                                    read_lm_head_value=not skip_unused_proposer_reads,
                                )
                                rocprof_window.range_pop()
                                update_delta = time.perf_counter() - update_started
                                proposal_decode_update_seconds += update_delta
                                ar_fallback_proposer_update_seconds += update_delta
                            context += 1
                            root = bonus
                            zero_accept_streak = 0
                            ar_fallback_seconds += time.perf_counter() - fallback_started
                            cycle_t_ns_end = time.perf_counter_ns()
                            rocprof_window.range_pop()
                            cycle_marker_ns.append((cycles, cycle_t_ns_start, cycle_t_ns_end))
                        continue
                    cycles += 1
                    if (
                        rocprof_resume_window_enabled
                        and not rocprof_window_done
                        and not rocprof_window_started
                        and cycles > int(rocprof_warmup_cycles)
                    ):
                        rocprof_window.resume()
                        rocprof_window_started = True
                        rocprof_window_first_cycle = cycles
                        rocprof_window_t_start = time.perf_counter()
                    rocprof_window.range_push(f"mtp_verify_cycle_{cycles}")
                    cycle_t_ns_start = time.perf_counter_ns()
                    rocprof_window.range_push(f"mtp_proposer_draft_{cycles}")
                    candidates = [int(proposer.current.token)]
                    confidence_p1: float | None = None
                    if tree_mode == "chain" and confidence_threshold > 0.0:
                        confidence_p1 = float(proposer.top1_prob_proxy())
                    if (
                        tree_mode == "chain"
                        and confidence_threshold > 0.0
                        and confidence_p1 is not None
                        and confidence_p1 < confidence_threshold
                    ):
                        rocprof_window.range_pop()
                        rocprof_window.range_push(f"mtp_confidence_ar_fallback_{cycles}")
                        fallback_started = time.perf_counter()
                        step_result = session.step(
                            root,
                            position=context,
                            sample=True,
                        )
                        if step_result is None:
                            raise RuntimeError("confidence-gated AR fallback step produced no root")
                        bonus = int(step_result.token_id)
                        committed = [root]
                        generated.extend(committed)
                        accepted_lengths.append(0)
                        active_budgets.append(0)
                        ar_fallback_cycles += 1
                        ar_fallback_tokens_committed += 1
                        confidence_ar_fallback_cycles += 1
                        confidence_ar_fallback_tokens += 1
                        if acceptance_diagnostics:
                            acceptance_diagnostic_cycles.append(
                                {
                                    "cycle": int(cycles),
                                    "decode_offset": int(decode_offset),
                                    "root_position": int(context),
                                    "root_token": int(root),
                                    "policy": "confidence_ar_fallback",
                                    "active_budget": 0,
                                    "verify_budget": 0,
                                    "accepted": 0,
                                    "full_accept": False,
                                    "first_rejected_depth": 0,
                                    "first_rejected_proposed_token": int(candidates[0]),
                                    "first_rejected_target_top1_token": int(bonus),
                                    "first_rejected_target_in_draft_vocab_cap": (
                                        None if draft_vocab_cap is None else bool(int(bonus) < int(draft_vocab_cap))
                                    ),
                                    "first_rejected_reason": "low_confidence_whole_cycle_gate",
                                    "confidence_threshold": float(confidence_threshold),
                                    "depth1_top1_prob_proxy": float(confidence_p1),
                                    "per_depth": [
                                        {
                                            "depth": 0,
                                            "accepted": False,
                                            "proposed_token": int(candidates[0]),
                                            "target_top1_token": int(bonus),
                                            "proposed_matches_target": bool(int(candidates[0]) == int(bonus)),
                                            "target_in_draft_vocab_cap": (
                                                None if draft_vocab_cap is None else bool(int(bonus) < int(draft_vocab_cap))
                                            ),
                                            "top1_prob_proxy": float(confidence_p1),
                                        }
                                    ],
                                }
                            )
                        if len(proposal_trace) < 16:
                            proposal_trace.append(
                                {
                                    "cycle": int(cycles),
                                    "policy": "confidence_ar_fallback",
                                    "root_position": int(context),
                                    "root_token": int(root),
                                    "draft_candidates": candidates,
                                    "accepted": 0,
                                    "committed_tokens": committed,
                                    "bonus_token": int(bonus),
                                    "confidence_threshold": float(confidence_threshold),
                                    "depth1_top1_prob_proxy": float(confidence_p1),
                                    "proposer_cache_len_before_update": int(proposer.cache_len),
                                }
                            )
                        if len(generated) < int(decode_tokens):
                            update_started = time.perf_counter()
                            rocprof_window.range_push(f"mtp_confidence_ar_fallback_proposer_update_{cycles}")
                            proposer.advance_with_previous_hidden(
                                input_token=bonus,
                                position=proposer.position + 1,
                                read_expert_topk=not skip_unused_proposer_reads,
                                read_lm_head_value=not skip_unused_proposer_reads,
                            )
                            rocprof_window.range_pop()
                            update_delta = time.perf_counter() - update_started
                            proposal_decode_update_seconds += update_delta
                            ar_fallback_proposer_update_seconds += update_delta
                        context += 1
                        root = bonus
                        zero_accept_streak = 0
                        ar_fallback_seconds += time.perf_counter() - fallback_started
                        cycle_t_ns_end = time.perf_counter_ns()
                        rocprof_window.range_pop()
                        cycle_marker_ns.append((cycles, cycle_t_ns_start, cycle_t_ns_end))
                        continue
                    snapshots = [proposer.save_state(0)]
                    proposal_snapshot_saves += 1
                    # Gated branching tree (#99 -> persistent): on
                    # low-confidence depth-1 cycles, spend the budget on a
                    # root-sibling branch instead of a deeper chain. Tree and
                    # chain use the SAME padded rows=B+1 verify shape (fixed
                    # rows keeps the cached graphs valid; see #107).
                    use_tree = False
                    topk_per_depth: list[tuple[list[int], list[float]]] = []
                    if tree_mode == "branching_topk" and confidence_threshold > 0.0 and active_budget >= 2:
                        d1_ids, d1_vals = proposer.vocab_topk(k=8)
                        topk_per_depth.append((d1_ids, d1_vals))
                        # Per #99: spend the wider tree on confident cycles,
                        # fall back to the deeper chain on low-confidence ones.
                        use_tree = _topk_softmax_top1(d1_vals) >= confidence_threshold
                    if use_tree:
                        # Tree shape at budget B: depth-1 top-(tree_top_k) +
                        # top-1 chain through depth B-1 (total B candidates).
                        for _depth in range(2, active_budget):
                            proposer.advance_with_previous_hidden(
                                input_token=int(topk_per_depth[-1][0][0]),
                                position=proposer.position + 1,
                                read_expert_topk=not skip_unused_proposer_reads,
                                read_lm_head_value=not skip_unused_proposer_reads,
                            )
                            topk_per_depth.append(proposer.vocab_topk(k=8))
                        candidates = [int(ids[0]) for ids, _vals in topk_per_depth]
                    else:
                        # DFlash-style per-position confidence floor (#100):
                        # stop drafting at the first low-confidence depth.
                        # Keep >=1 candidate so the verify shape stays fixed.
                        for draft_idx in range(1, active_budget):
                            if draft_p_min > 0.0 and proposer.top1_prob_proxy() < draft_p_min:
                                break
                            proposer.advance_with_previous_hidden(
                                input_token=candidates[-1],
                                position=proposer.position + 1,
                                read_expert_topk=not skip_unused_proposer_reads,
                                read_lm_head_value=not skip_unused_proposer_reads,
                            )
                            if (not skip_unused_proposer_reads) or draft_idx < active_budget - 1:
                                snapshots.append(proposer.save_state(draft_idx))
                                proposal_snapshot_saves += 1
                            else:
                                proposal_snapshot_skips += 1
                            candidates.append(int(proposer.current.token))
                        active_budget = len(candidates)
                    active_budgets.append(active_budget)
                    rocprof_window.range_pop()
                    # Keep the verify at a FIXED rows=B+1 shape: each rows
                    # bucket re-reserves verifier scratch at its own shape,
                    # freeing buffers other rows' cached graphs hold (#107
                    # hang under p-min truncation). Padded rows are inert
                    # (active_mask) and the batched wall is row-invariant.
                    verify_budget = int(candidate_budget)
                    t_verify = time.perf_counter()
                    rocprof_window.range_push(f"mtp_verify_pass_{cycles}")
                    if use_tree:
                        compiled = _build_branching_topk_tree_target_batch(
                            root_token=root,
                            root_position=context,
                            topk_tokens=[ids for ids, _ in topk_per_depth],
                            topk_values=[vals for _, vals in topk_per_depth],
                            candidate_budget=verify_budget,
                            tree_top_k=int(tree_top_k),
                            max_depth=len(topk_per_depth),
                        )
                        target_batch = compiled.target_batch
                        verify = session.verify_tree_bulk_and_commit(
                            target_batch,
                            base_slot=0,
                            capture_layer_ids=(),
                            capture_hidden_concat=verifier_no_capture,
                            capture_row_start=0,
                            graph_mode=graph_mode,
                            canonicalize_after=canonicalize_after_verify,
                        )
                        accepted_tokens = [int(t) for t in verify.accepted_tokens]
                    else:
                        target_batch = _target_batch(root, context, candidates, active_budget, candidate_budget=verify_budget)
                        verify = session.verify_chain_bulk_and_commit(
                            target_batch,
                            base_slot=0,
                            capture_layer_ids=(),
                            capture_hidden_concat=verifier_no_capture,
                            capture_row_start=0,
                            chain_attn_mode=chain_attn_mode,
                            graph_mode=graph_mode,
                            canonicalize_after=canonicalize_after_verify,
                            synchronize_after_commit=not overlap_verify_commit_proposer,
                        )
                        accepted_tokens = candidates[: int(verify.accepted_count)]
                    rocprof_window.range_pop()
                    verify_seconds += time.perf_counter() - t_verify
                    mtp_verify_cycles_completed += 1
                    accepted = int(verify.accepted_count)
                    accepted_lengths.append(accepted)
                    if accepted == 0:
                        zero_accept_streak += 1
                    else:
                        zero_accept_streak = 0
                    bonus = int(verify.next_token) if verify.next_token is not None else int(verify.target_top1[min(accepted, len(verify.target_top1) - 1)])
                    if acceptance_diagnostics:
                        acceptance_diagnostic_cycles.append(
                            _acceptance_diagnostic_cycle(
                                cycle=cycles,
                                decode_offset=decode_offset,
                                root_position=context,
                                root_token=root,
                                candidates=candidates,
                                target_top1=verify.target_top1[: 1 + active_budget],
                                bonus_token=bonus,
                                active_budget=active_budget,
                                verify_budget=verify_budget,
                                accepted=accepted,
                                draft_vocab_cap=draft_vocab_cap,
                                use_tree=use_tree,
                            )
                        )
                    committed = [root, *accepted_tokens]
                    generated.extend(committed)
                    if len(proposal_trace) < 16:
                        proposal_trace.append(
                            {
                                "cycle": cycles,
                                "root_position": context,
                                "root_token": root,
                                "draft_candidates": candidates,
                                "target_top1_path": list(map(int, verify.target_top1[: 1 + active_budget])),
                                "target_top1_values": list(map(float, verify.target_top1_values[: 1 + active_budget])),
                                "accepted": accepted,
                                "committed_tokens": committed,
                                "bonus_token": bonus,
                                "target_parent_rows": list(map(int, target_batch.parent_rows)),
                                "verify_graph": verify.graph,
                                "gpu_accept_match_cpu": bool(verify.gpu_accept_match_cpu) if verify.gpu_accept_match_cpu is not None else None,
                                "proposer_cache_len_before_update": int(proposer.cache_len),
                            }
                        )
                    update_started = time.perf_counter()
                    rocprof_window.range_push(f"mtp_proposer_update_{cycles}")
                    update_stream = proposer_update_stream if (overlap_verify_commit_proposer and not use_tree) else 0
                    if len(generated) < int(decode_tokens):
                        if use_tree:
                            # Tree accepts may follow the sibling branch, so
                            # replay the accepted path from the cycle root.
                            proposer.restore_state(snapshots[0], stream=update_stream)
                            for token in accepted_tokens:
                                proposer.advance_with_previous_hidden(
                                    input_token=int(token),
                                    position=proposer.position + 1,
                                    need_result=not skip_unused_proposer_reads,
                                    read_expert_topk=not skip_unused_proposer_reads,
                                    read_lm_head_value=not skip_unused_proposer_reads,
                                    stream=update_stream,
                                )
                        elif accepted < active_budget - 1:
                            proposer.restore_state(snapshots[accepted], stream=update_stream)
                        elif accepted >= active_budget:
                            # After candidate generation, the live proposer state is
                            # already equivalent to snapshots[active_budget - 1].
                            # Reuse it and consume the final accepted candidate before
                            # the target bonus token instead of doing a redundant
                            # synchronous D2D restore.
                            proposer.advance_with_previous_hidden(
                                input_token=candidates[-1],
                                position=proposer.position + 1,
                                need_result=not skip_unused_proposer_reads,
                                read_expert_topk=not skip_unused_proposer_reads,
                                read_lm_head_value=not skip_unused_proposer_reads,
                                stream=update_stream,
                            )
                        proposer.advance_with_previous_hidden(
                            input_token=bonus,
                            position=proposer.position + 1,
                            read_expert_topk=not skip_unused_proposer_reads,
                            read_lm_head_value=not skip_unused_proposer_reads,
                            stream=update_stream,
                        )
                    rocprof_window.range_pop()
                    proposal_decode_update_seconds += time.perf_counter() - update_started
                    context += len(committed)
                    root = bonus
                    cycle_t_ns_end = time.perf_counter_ns()
                    rocprof_window.range_pop()
                    cycle_marker_ns.append((cycles, cycle_t_ns_start, cycle_t_ns_end))
                    if (
                        rocprof_resume_window_enabled
                        and rocprof_window_started
                        and rocprof_window_first_cycle is not None
                        and cycles >= rocprof_window_first_cycle + int(rocprof_verify_cycles) - 1
                    ):
                        rocprof_window.pause()
                        rocprof_window_last_cycle = cycles
                        rocprof_window_t_end = time.perf_counter()
                        rocprof_window_started = False
                        rocprof_window_done = True
                if rocprof_resume_window_enabled and rocprof_window_started:
                    rocprof_window.pause()
                    rocprof_window_last_cycle = cycles
                    rocprof_window_t_end = time.perf_counter()
                    rocprof_window_started = False
                    rocprof_window_done = True
                decode_seconds = time.perf_counter() - decode_started
        finally:
            if capture_buf is not None:
                free(capture_buf, runtime=session.runtime)
    seconds = time.perf_counter() - started
    if rocprof_window_first_cycle is not None and rocprof_window_last_cycle is not None:
        rocprof_window_meta["profiled_cycle_range"] = [int(rocprof_window_first_cycle), int(rocprof_window_last_cycle)]
        if rocprof_window_t_start is not None and rocprof_window_t_end is not None:
            rocprof_window_meta["profiled_cycle_seconds"] = float(rocprof_window_t_end - rocprof_window_t_start)
    spec: dict[str, Any] = {
        "seconds": seconds,
        "decode_seconds": decode_seconds,
        "tok_s": int(decode_tokens) / seconds if seconds > 0 else None,
        "decode_tok_s": int(decode_tokens) / decode_seconds if decode_seconds > 0 else None,
        "proposal_prefill_seconds": proposal_prefill_seconds,
        "proposal_decode_update_seconds": proposal_decode_update_seconds,
        "proposal_snapshot_saves": int(proposal_snapshot_saves),
        "proposal_snapshot_skips": int(proposal_snapshot_skips),
        "verify_seconds": verify_seconds,
        "active_budget_cap": int(active_budget_cap),
        "verify_budget": int(candidate_budget),
        "ar_fallback_zero_streak": int(ar_fallback_zero_streak),
        "ar_fallback_after_mtp_cycles": int(ar_fallback_after_mtp_cycles),
        "ar_fallback_tokens_per_window": int(ar_fallback_tokens),
        "ar_fallback_until_end": bool(ar_fallback_until_end),
        "ar_fallback_cycles": int(ar_fallback_cycles),
        "ar_fallback_tokens": int(ar_fallback_tokens_committed),
        "ar_fallback_seconds": float(ar_fallback_seconds),
        "ar_fallback_proposer_update_seconds": float(ar_fallback_proposer_update_seconds),
        "confidence_threshold": float(confidence_threshold),
        "confidence_ar_fallback_cycles": int(confidence_ar_fallback_cycles),
        "confidence_ar_fallback_tokens": int(confidence_ar_fallback_tokens),
        "accepted_lengths": accepted_lengths,
        "active_budgets": active_budgets,
        "acceptance_rate": (sum(accepted_lengths) / sum(active_budgets)) if active_budgets and sum(active_budgets) else 0.0,
        "proposal_trace_sample": proposal_trace,
        "chain_attn_mode": chain_attn_mode,
        "proposal_impl": "persistent_device",
        "proposer_skip_unused_reads": bool(skip_unused_proposer_reads),
        "canonicalize_after_verify": bool(canonicalize_after_verify),
        "overlap_verify_commit_proposer": bool(overlap_verify_commit_proposer),
        "note": "Persistent native MTP provider: weights/cache resident, target hidden stays on device, and unused proposer metadata/results/snapshots are skipped by default.",
        "rocprof_window": rocprof_window_meta,
        "cycle_marker_ns": [
            {"cycle": cycle_idx, "start_perf_ns": start_ns, "end_perf_ns": end_ns}
            for cycle_idx, start_ns, end_ns in cycle_marker_ns
        ],
    }
    if acceptance_diagnostics:
        spec["acceptance_diagnostics"] = _summarize_acceptance_diagnostics(
            acceptance_diagnostic_cycles,
            accepted_lengths=accepted_lengths,
            draft_vocab_cap=draft_vocab_cap,
        )
    return generated[: int(decode_tokens)], spec


def _parse_int_list(text: str) -> list[int]:
    return [int(p.strip()) for p in str(text).split(",") if p.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(p.strip()) for p in str(text).split(",") if p.strip()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model)
    prompt_tokens = tuple(int(part.strip()) for part in str(args.prompt_tokens).split(",") if part.strip())
    if not prompt_tokens:
        raise ValueError("at least one prompt token is required")
    if bool(getattr(args, "acceptance_curve", False)):
        return _run_acceptance_curve(
            model,
            prompt_tokens,
            decode_tokens=int(args.decode_tokens),
            candidate_budget=int(args.candidate_budget),
            backend=str(args.backend),
            chain_attn_mode=str(args.chain_attn_mode),
            tree_top_ks=_parse_int_list(args.curve_tree_top_ks),
            confidence_thresholds=_parse_float_list(args.curve_thresholds),
        )
    if bool(getattr(args, "pmin_sweep", False)):
        return _run_pmin_sweep(
            model,
            prompt_tokens,
            decode_tokens=int(args.decode_tokens),
            backend=str(args.backend),
            chain_attn_mode=str(args.chain_attn_mode),
            budgets=_parse_int_list(args.sweep_budgets),
            p_mins=_parse_float_list(args.sweep_pmins),
        )
    ar_tokens, ar = _run_ar_baseline(model, prompt_tokens, decode_tokens=int(args.decode_tokens), backend=str(args.backend))
    if args.proposal_impl in {"persistent_device", "persistent_device_b1"}:
        spec_tokens, spec = _run_spec_persistent_device(
            model,
            prompt_tokens,
            decode_tokens=int(args.decode_tokens),
            candidate_budget=int(args.candidate_budget),
            active_budget_cap=int(getattr(args, "active_budget_cap", 0)),
            backend=str(args.backend),
            chain_attn_mode=str(args.chain_attn_mode),
            graph_mode=str(args.graph_mode),
            draft_p_min=float(getattr(args, "draft_p_min", 0.0)),
            tree_mode=str(getattr(args, "tree_mode", "chain")),
            tree_top_k=int(getattr(args, "tree_top_k", 2)),
            confidence_threshold=float(getattr(args, "confidence_threshold", 0.0)),
            rocprof_warmup_cycles=int(getattr(args, "rocprof_warmup_cycles", 0)),
            rocprof_verify_cycles=int(getattr(args, "rocprof_verify_cycles", 0)),
            acceptance_diagnostics=bool(getattr(args, "acceptance_diagnostics", False)),
            ar_fallback_zero_streak=int(getattr(args, "ar_fallback_zero_streak", 0)),
            ar_fallback_after_mtp_cycles=int(getattr(args, "ar_fallback_after_mtp_cycles", 0)),
            ar_fallback_tokens=int(getattr(args, "ar_fallback_tokens", 1)),
            ar_fallback_until_end=bool(getattr(args, "ar_fallback_until_end", False)),
        )
    else:
        spec_tokens, spec = _run_spec_smoke(
            model,
            prompt_tokens,
            decode_tokens=int(args.decode_tokens),
            candidate_budget=int(args.candidate_budget),
            backend=str(args.backend),
            chain_attn_mode=str(args.chain_attn_mode),
            tree_mode=str(getattr(args, "tree_mode", "chain")),
            tree_top_k=int(getattr(args, "tree_top_k", 2)),
            confidence_threshold=float(getattr(args, "confidence_threshold", 0.0)),
            draft_p_min=float(getattr(args, "draft_p_min", 0.0)),
        )
    return {
        "status": "passed" if spec_tokens == ar_tokens else "exact_ar_mismatch",
        "performance_claim": False,
        "model": str(model),
        "prompt_tokens": list(prompt_tokens),
        "decode_tokens": int(args.decode_tokens),
        "candidate_budget": int(args.candidate_budget),
        "active_budget_cap": int(getattr(args, "active_budget_cap", 0)),
        "ar_tokens": ar_tokens,
        "mtp_tokens": spec_tokens,
        "exact_ar_match": spec_tokens == ar_tokens,
        "ar": ar,
        "mtp": spec,
        "proposal_impl": str(args.proposal_impl),
        "decision_reason": "Native MTP proposal rows reached verify_chain_bulk_and_commit and exact AR was checked. persistent_device keeps MTP weights/cache resident, but artifacts remain diagnostic until acceptance and speed gates pass.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", default="151646")
    parser.add_argument("--decode-tokens", type=int, default=3)
    parser.add_argument("--candidate-budget", type=int, default=2)
    parser.add_argument(
        "--active-budget-cap",
        type=int,
        default=0,
        help=(
            "persistent_device diagnostic: cap the active drafted candidates "
            "while keeping verifier allocation/rows at --candidate-budget. "
            "0 disables. Used to test max-shape adaptive-budget safety without "
            "changing verifier scratch shape mid-run."
        ),
    )
    parser.add_argument("--proposal-impl", choices=("reload_d2h", "persistent_device", "persistent_device_b1"), default="reload_d2h")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--chain-attn-mode", choices=("c1_loop", "batched", "decode_batched"), default="c1_loop")
    parser.add_argument("--graph-mode", choices=("off", "auto", "validate"), default="off")
    parser.add_argument("--tree-mode", choices=("chain", "branching_topk"), default="chain", help="reload_d2h only: chain (top-1 verify_chain) or branching_topk (balanced DDTree via verify_tree_bulk_and_commit, reusing the MTP head per-depth top-k + values)")
    parser.add_argument("--tree-top-k", type=int, default=2, help="branch width per depth for --tree-mode branching_topk (1..8)")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="online whole-cycle gate: drop to AR when depth-1 top-K-softmax top-1 prob < threshold (0 disables)")
    parser.add_argument("--acceptance-curve", action="store_true", help="reload_d2h only: sweep the realized acceptance curve over branch width x confidence threshold, sharing one resident target runner. Reports alpha, visible tokens/cycle, accept-depth histogram, exact_ar_match, gpu_accept_match_cpu per config.")
    parser.add_argument("--curve-tree-top-ks", default="2,3,4", help="comma-separated branch widths for --acceptance-curve")
    parser.add_argument("--curve-thresholds", default="0.0", help="comma-separated confidence thresholds for --acceptance-curve")
    parser.add_argument("--draft-p-min", type=float, default=0.0, help="chain paths (reload_d2h + persistent_device): per-position draft confidence floor (DFlash analog). Stop drafting at the first depth whose head top-1 prob proxy < p_min; truncate-to-0 cycles become plain AR. 0 disables.")
    parser.add_argument("--pmin-sweep", action="store_true", help="reload_d2h only: sweep per-position p-min x B (chain path), sharing one resident target runner. Reports zero-accept reduction, avg-accept, estimated C_B (#98 launch model), tok/s.")
    parser.add_argument("--sweep-budgets", default="1,2,3", help="comma-separated candidate budgets for --pmin-sweep (must be in {1,2,3,5})")
    parser.add_argument("--sweep-pmins", default="0.0,0.5,0.65", help="comma-separated p-min thresholds for --pmin-sweep")
    parser.add_argument(
        "--acceptance-diagnostics",
        action="store_true",
        help=(
            "persistent_device only: include per-cycle acceptance-density diagnostics "
            "(accept depth histograms, first rejected depth/reason, cap representability, "
            "and zero-accept streaks) in the JSON output. Diagnostics only; no policy change."
        ),
    )
    parser.add_argument(
        "--ar-fallback-zero-streak",
        type=int,
        default=0,
        help=(
            "persistent_device only: after this many consecutive zero-accept MTP "
            "verify cycles, skip the next --ar-fallback-tokens tokens through the "
            "target AR path while keeping proposer state aligned. 0 disables."
        ),
    )
    parser.add_argument(
        "--ar-fallback-tokens",
        type=int,
        default=1,
        help=(
            "persistent_device only: number of target AR tokens to emit per "
            "--ar-fallback-zero-streak trigger. Exactness policy diagnostic; "
            "default 1."
        ),
    )
    parser.add_argument(
        "--ar-fallback-after-mtp-cycles",
        type=int,
        default=0,
        help=(
            "persistent_device only: diagnostic override to route through target "
            "AR after exactly this many MTP verifier cycles. 0 disables. Use with "
            "--ar-fallback-until-end to bracket verifier-commit state drift."
        ),
    )
    parser.add_argument(
        "--ar-fallback-until-end",
        action="store_true",
        help=(
            "persistent_device only: when --ar-fallback-zero-streak triggers, "
            "finish the remaining decode with plain target AR and do not realign "
            "the proposer for MTP resume."
        ),
    )
    parser.add_argument(
        "--rocprof-warmup-cycles",
        type=int,
        default=0,
        help=(
            "persistent_device only: skip this many verify cycles before opening the "
            "roctxProfilerResume window. Use to discard cold-cache iterations from the "
            "rocprofv3 --selected-regions trace."
        ),
    )
    parser.add_argument(
        "--rocprof-verify-cycles",
        type=int,
        default=0,
        help=(
            "persistent_device only: number of verify cycles to keep inside the "
            "roctxProfilerResume window. 0 disables the window (no profiling region)."
        ),
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--out", type=Path, help="alias for --json (artifact path)")
    args = parser.parse_args()
    result = run(args)
    out_path = args.out or args.json
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if "pmin_sweep" in result:
        summary = {
            "status": result["status"],
            "ar_tok_s": result["ar_tok_s"],
            "sweep": [
                {
                    "B": row["candidate_budget"],
                    "pmin": row["draft_p_min"],
                    "alpha": row["alpha"],
                    "avg_acc": row["avg_accepted"],
                    "zero_rate": row["zero_accept_rate"],
                    "zero_red": row["zero_accept_rate_reduction_vs_pmin0"],
                    "ar_cyc": row["pmin_ar_cycles"],
                    "est_launch_per_tok": row["est_launches_per_visible_token"],
                    "exact_ar": row["exact_ar_match"],
                }
                for row in result["pmin_sweep"]
            ],
        }
        print(json.dumps(summary, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    if "acceptance_curve" in result:
        summary = {
            "status": result["status"],
            "ar_tok_s": result["ar_tok_s"],
            "curve": [
                {
                    "mode": row["tree_mode"],
                    "k": row["tree_top_k"],
                    "thr": row["confidence_threshold"],
                    "alpha": round(float(row["alpha"]), 4),
                    "vis_tok_per_cycle": round(float(row["visible_tokens_per_cycle"]), 3),
                    "exact_ar": row["exact_ar_match"],
                    "gpu_match": row["gpu_accept_match_cpu"],
                }
                for row in result["acceptance_curve"]
            ],
        }
        print(json.dumps(summary, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    print(json.dumps({"status": result["status"], "exact_ar_match": result["exact_ar_match"], "ar": result["ar_tokens"], "mtp": result["mtp_tokens"], "accepted": result["mtp"]["accepted_lengths"], "mtp_tok_s_diagnostic": result["mtp"]["tok_s"], "ar_tok_s": result["ar"]["tok_s"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
