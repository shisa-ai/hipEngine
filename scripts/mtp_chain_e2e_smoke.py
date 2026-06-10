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
    proposal_seconds = 0.0
    verify_seconds = 0.0
    target_forward_calls = 0
    capture_rows = max_sequence + int(candidate_budget) + 2
    capture_buf: DeviceBuffer | None = None
    started = time.perf_counter()
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
                    )
                    accepted_tokens = list(verify.accepted_tokens)
                else:
                    # chain (top-1) path, or gated-to-AR (active_budget honored
                    # but a single-row chain effectively verifies the root).
                    target_batch = _target_batch(root, context, candidates, active_budget)
                    verify = session.verify_chain_bulk_and_commit(
                        target_batch,
                        base_slot=0,
                        capture_layer_ids=(capture_layer_id,),
                        capture_hidden_concat=capture,
                        capture_row_start=context,
                        chain_attn_mode=chain_attn_mode,
                    )
                    accepted_tokens = candidates[: int(verify.accepted_count)]
                verify_seconds += time.perf_counter() - t_verify
                target_forward_calls += int(verify.target_forward_calls)
                accepted = int(verify.accepted_count)
                accepted_lengths.append(accepted)
                accept_depth_hist[accepted] = accept_depth_hist.get(accepted, 0) + 1
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
        "accept_depth_histogram": {str(k): accept_depth_hist[k] for k in sorted(accept_depth_hist)},
        "gpu_accept_match_cpu": gpu_accept_match_all,
        "proposal_trace_sample": proposal_trace,
        "target_forward_calls": target_forward_calls,
        "chain_attn_mode": chain_attn_mode,
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


def _run_spec_persistent_device(
    model: Path,
    prompt_tokens: Sequence[int],
    *,
    decode_tokens: int,
    candidate_budget: int,
    backend: str,
    chain_attn_mode: str,
    graph_mode: str = "off",
    rocprof_warmup_cycles: int = 0,
    rocprof_verify_cycles: int = 0,
) -> tuple[list[int], dict[str, Any]]:
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_tokens) + int(decode_tokens) + int(candidate_budget) + 4
    generated: list[int] = []
    accepted_lengths: list[int] = []
    proposal_trace: list[dict[str, Any]] = []
    verify_seconds = 0.0
    proposal_prefill_seconds = 0.0
    proposal_decode_update_seconds = 0.0
    capture_rows = max_sequence + 2
    capture_buf: DeviceBuffer | None = None
    started = time.perf_counter()
    active_budgets: list[int] = []
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
            ) as proposer:
                prefill_started = time.perf_counter()
                proposer.prefill_from_target_hidden_rows(prompt_tokens, capture_base_ptr=capture_buf.ptr, seed_token=root)
                proposal_prefill_seconds += time.perf_counter() - prefill_started
                cycles = 0
                decode_started = time.perf_counter()
                while len(generated) < int(decode_tokens):
                    remaining = int(decode_tokens) - len(generated)
                    active_budget = min(int(candidate_budget), max(0, remaining - 1))
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
                    snapshots = [proposer.save_state(0)]
                    candidates = [int(proposer.current.token)]
                    for draft_idx in range(1, active_budget):
                        proposer.advance_with_previous_hidden(input_token=candidates[-1], position=proposer.position + 1)
                        snapshots.append(proposer.save_state(draft_idx))
                        candidates.append(int(proposer.current.token))
                    active_budgets.append(active_budget)
                    verify_budget = active_budget if active_budget in MTP_CHAIN_CANDIDATE_BUDGETS else int(candidate_budget)
                    target_batch = _target_batch(root, context, candidates, active_budget, candidate_budget=verify_budget)
                    t_verify = time.perf_counter()
                    rocprof_window.range_push(f"mtp_verify_pass_{cycles}")
                    verify = session.verify_chain_bulk_and_commit(
                        target_batch,
                        base_slot=0,
                        capture_layer_ids=(),
                        capture_hidden_concat=verifier_no_capture,
                        capture_row_start=0,
                        chain_attn_mode=chain_attn_mode,
                        graph_mode=graph_mode,
                    )
                    rocprof_window.range_pop()
                    verify_seconds += time.perf_counter() - t_verify
                    accepted = int(verify.accepted_count)
                    accepted_lengths.append(accepted)
                    committed = [root, *candidates[:accepted]]
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
                                "proposer_cache_len_before_update": int(proposer.cache_len),
                            }
                        )
                    update_started = time.perf_counter()
                    if len(generated) < int(decode_tokens):
                        if accepted < active_budget - 1:
                            proposer.restore_state(snapshots[accepted])
                        elif accepted >= active_budget:
                            # After candidate generation, the live proposer state is
                            # already equivalent to snapshots[active_budget - 1].
                            # Reuse it and consume the final accepted candidate before
                            # the target bonus token instead of doing a redundant
                            # synchronous D2D restore.
                            proposer.advance_with_previous_hidden(input_token=candidates[-1], position=proposer.position + 1)
                        proposer.advance_with_previous_hidden(input_token=bonus, position=proposer.position + 1)
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
    return generated[: int(decode_tokens)], {
        "seconds": seconds,
        "decode_seconds": decode_seconds,
        "tok_s": int(decode_tokens) / seconds if seconds > 0 else None,
        "decode_tok_s": int(decode_tokens) / decode_seconds if decode_seconds > 0 else None,
        "proposal_prefill_seconds": proposal_prefill_seconds,
        "proposal_decode_update_seconds": proposal_decode_update_seconds,
        "verify_seconds": verify_seconds,
        "accepted_lengths": accepted_lengths,
        "active_budgets": active_budgets,
        "acceptance_rate": (sum(accepted_lengths) / sum(active_budgets)) if active_budgets and sum(active_budgets) else 0.0,
        "proposal_trace_sample": proposal_trace,
        "chain_attn_mode": chain_attn_mode,
        "proposal_impl": "persistent_device",
        "note": "Persistent native MTP provider: weights/cache resident and target hidden stays on device; selected expert ids are still host-orchestrated.",
        "rocprof_window": rocprof_window_meta,
        "cycle_marker_ns": [
            {"cycle": cycle_idx, "start_perf_ns": start_ns, "end_perf_ns": end_ns}
            for cycle_idx, start_ns, end_ns in cycle_marker_ns
        ],
    }


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
    ar_tokens, ar = _run_ar_baseline(model, prompt_tokens, decode_tokens=int(args.decode_tokens), backend=str(args.backend))
    if args.proposal_impl in {"persistent_device", "persistent_device_b1"}:
        spec_tokens, spec = _run_spec_persistent_device(
            model,
            prompt_tokens,
            decode_tokens=int(args.decode_tokens),
            candidate_budget=int(args.candidate_budget),
            backend=str(args.backend),
            chain_attn_mode=str(args.chain_attn_mode),
            graph_mode=str(args.graph_mode),
            rocprof_warmup_cycles=int(getattr(args, "rocprof_warmup_cycles", 0)),
            rocprof_verify_cycles=int(getattr(args, "rocprof_verify_cycles", 0)),
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
        )
    return {
        "status": "passed" if spec_tokens == ar_tokens else "exact_ar_mismatch",
        "performance_claim": False,
        "model": str(model),
        "prompt_tokens": list(prompt_tokens),
        "decode_tokens": int(args.decode_tokens),
        "candidate_budget": int(args.candidate_budget),
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
