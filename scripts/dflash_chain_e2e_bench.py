#!/usr/bin/env python3
"""Full-model DFlash chain E2E benchmark driver.

This is the hipEngine runner that executes the real packed target model and
native DFlash drafter with a same-session AR control.  The default verifier is
``native_bulk_bplus1``: it runs the root plus fixed-budget candidate chain in
one B+1-row target forward against the resident KV/state, uses GPU accept
metadata, and commits the selected row state.  ``serial_in_place_single_slot``
remains available as a diagnostic fallback.  Rows are not promotable unless the
artifact says the native bulk verifier ran and the normal speed/correctness
speed gates pass.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.prompts import DEFAULT_STABLE_PROMPT_FIXTURE, file_sha256, load_prompt_records
from hipengine.benchmark.speculative import DEFAULT_DFLASH_DRAFTER, DEFAULT_TARGET_MODEL, SpeculativeBenchmarkModels, build_speculative_artifact
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc, memory_stats, reset_memory_stats
from hipengine.core.tensor import Tensor
from hipengine.kernels.backends import hip_target_arch_environment
from hipengine.kernels.hip_gfx1100.convert import build_cast
from hipengine.kernels.hip_gfx1100.linear import build_lm_head, topk_f32_rows_i32
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import w8a16_linear_bf16_f32_out
from hipengine.kernels.hip_gfx1100.speculative import build_dflash_drafter
from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
    _drafter_dense_use_add_rmsnorm,
    dflash_add_bf16,
    dflash_add_rmsnorm_bf16,
    dflash_concat_rows_bf16,
    dflash_concat_rows_f32,
    dflash_dense_bf16_to_bf16,
    dflash_dense_bf16_to_f32,
    dflash_gqa_attention_f32_bf16,
    dflash_gqa_attention_f32_bf16_bucketed,
    dflash_head_rmsnorm_rotary_f32,
    dflash_key_rmsnorm_rotary_f32,
    dflash_prepare_noise_inputs_bf16_i32,
    dflash_qkv_proj_bf16_mixed,
    dflash_prepare_noise_inputs_f16_to_bf16_i32,
    dflash_rmsnorm_bf16,
    dflash_silu_mul_bf16,
)
from hipengine.loading import load_weight_index
from hipengine.loading.dflash import load_dflash_drafter_bf16_weights, validate_dflash_artifact_pair
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative import AdaptiveBudgetConfig, AdaptiveBudgetController, DraftBatch, TargetVerifyBatch

DEFAULT_TARGET_PATH = "/models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e"
DEFAULT_DRAFTER_PATH = "/models/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719"
DEFAULT_TARGET_REVISION = "501ef8635e5cfb5a7497d232358ca8d1afc0c66e"
DEFAULT_DRAFTER_REVISION = "42d3b34d588423cdae7ba8f53a8cf7789346a719"
_ADAPTIVE_AR_GUARD_REASONS = {"remaining_tokens_guard", "probe_amortization_guard"}
_PROFILE_ROUTE_VALUES = {"ar", "chain", "tree", "spec"}


class _Roctx:
    """Minimal ROCTX range helper for rocprof marker slicing."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._lib = None
        self._resume = None
        self._pause = None
        if not self.enabled:
            return
        try:
            self._lib = ctypes.CDLL("libroctx64.so")
            self._lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
            self._lib.roctxRangePushA.restype = ctypes.c_int
            self._lib.roctxRangePop.argtypes = []
            self._lib.roctxRangePop.restype = ctypes.c_int
            self._resume = getattr(self._lib, "roctxProfilerResume", None)
            self._pause = getattr(self._lib, "roctxProfilerPause", None)
            if self._resume is not None:
                self._resume.argtypes = [ctypes.c_int]
                self._resume.restype = None
            if self._pause is not None:
                self._pause.argtypes = [ctypes.c_int]
                self._pause.restype = None
        except OSError as exc:
            print(f"warning: --roctx requested but libroctx64.so could not be loaded: {exc}", file=sys.stderr)
            self._lib = None

    def push(self, name: str) -> None:
        if self._lib is not None:
            self._lib.roctxRangePushA(name.encode("utf-8"))

    def pop(self) -> None:
        if self._lib is not None:
            self._lib.roctxRangePop()

    def profiler_resume(self) -> None:
        if self._resume is not None:
            self._resume(0)

    def profiler_pause(self) -> None:
        if self._pause is not None:
            self._pause(0)

    @property
    def profiler_controls_available(self) -> bool:
        return self._resume is not None and self._pause is not None


def _hf_snapshot_identity(path: Path, *, default_name: str, default_revision: str) -> tuple[str, str]:
    """Infer a Hugging Face model id/revision from a local snapshot path."""

    parts = path.resolve().parts
    try:
        snapshot_index = parts.index("snapshots")
    except ValueError:
        return default_name, default_revision
    if snapshot_index <= 0 or snapshot_index + 1 >= len(parts):
        return default_name, default_revision
    raw_name = parts[snapshot_index - 1]
    if not raw_name.startswith("models--"):
        return default_name, default_revision
    model_name = raw_name.removeprefix("models--").replace("--", "/")
    return model_name, parts[snapshot_index + 1]


def _canonical_profile_route(value: Any) -> str:
    route = str(value).strip().lower()
    if route in {"dflash", "dflash_chain"}:
        route = "chain"
    if route in {"branching_topk", "ddtree"}:
        route = "tree"
    if route not in _PROFILE_ROUTE_VALUES:
        raise ValueError(f"profile route must be one of {sorted(_PROFILE_ROUTE_VALUES)}, got {value!r}")
    return route


def _load_profile_route_manifest(path: Path | None) -> tuple[str, dict[str, str], dict[str, Any] | None]:
    if path is None:
        return "spec", {}, None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--profile-route-manifest must be a JSON object")
    default = _canonical_profile_route(raw.get("default", "spec"))
    routes_raw = raw.get("routes", raw)
    if not isinstance(routes_raw, dict):
        raise ValueError("profile route manifest 'routes' must be a JSON object")
    routes: dict[str, str] = {}
    for key, value in routes_raw.items():
        if key in {"default", "routes", "notes", "description", "terminal_ar_tokens", "draft_budgets"}:
            continue
        routes[str(key)] = _canonical_profile_route(value)
    return default, routes, raw


def _profile_prompt_keys(prompt: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(prompt.get("id") or ""),
        str(prompt.get("prompt_ids_sha256") or ""),
        str(prompt.get("prompt_text_sha256") or ""),
        str(prompt.get("benchmark_group") or ""),
    )


def _profile_route_for_prompt(prompt: dict[str, Any], *, default: str, routes: dict[str, str]) -> str:
    for key in _profile_prompt_keys(prompt):
        if key and key in routes:
            return routes[key]
    return default


def _terminal_ar_tokens_for_prompt(prompt: dict[str, Any], *, default: int, manifest: dict[str, Any] | None) -> int:
    if manifest is None or "terminal_ar_tokens" not in manifest:
        return int(default)
    raw = manifest.get("terminal_ar_tokens")
    if raw is None:
        return int(default)
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, dict):
        value = raw.get("default", default)
        for key in _profile_prompt_keys(prompt):
            if key and key in raw:
                value = raw[key]
                break
    else:
        raise ValueError("profile route manifest terminal_ar_tokens must be an integer or object")
    parsed = int(value)
    if parsed < 0:
        raise ValueError("profile route manifest terminal_ar_tokens values must be non-negative")
    return parsed


def _parse_draft_budget_list(value: Any) -> list[int]:
    if isinstance(value, int):
        budgets = [int(value)]
    elif isinstance(value, str):
        budgets = [int(part) for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        budgets = [int(part) for part in value]
    else:
        raise ValueError("draft budget values must be an integer, comma string, or list")
    if not budgets:
        raise ValueError("draft budget list must not be empty")
    if any(budget <= 0 for budget in budgets):
        raise ValueError("draft budgets must be positive integers")
    return budgets


def _draft_budgets_for_prompt(prompt: dict[str, Any], *, default: Sequence[int], manifest: dict[str, Any] | None) -> list[int]:
    if manifest is None or "draft_budgets" not in manifest:
        return [int(budget) for budget in default]
    raw = manifest.get("draft_budgets")
    if raw is None:
        return [int(budget) for budget in default]
    if isinstance(raw, dict):
        value: Any = raw.get("default", list(default))
        for key in _profile_prompt_keys(prompt):
            if key and key in raw:
                value = raw[key]
                break
    else:
        value = raw
    return _parse_draft_budget_list(value)


@dataclass(frozen=True)
class DraftResult:
    candidate_tokens: tuple[int, ...]
    draft_seconds: float
    finite_logits: bool
    d2h_vector_reads: int
    d2h_vector_values: int
    phase_seconds: dict[str, float]
    graph: dict[str, Any]
    topk_tokens: tuple[tuple[int, ...], ...] = ()
    topk_values: tuple[tuple[float, ...], ...] = ()
    top1_probabilities: tuple[float, ...] = ()


def _dflash_cross_bucket(context_tokens: int) -> int:
    """BeeLlama-style cross_bucket() shape function for graph cache keys.

    - ``<= 16`` rounds up to 16.
    - ``<= 128`` rounds up to the next power of two.
    - ``> 128`` rounds up to the next multiple of 128.

    Matches the bucketing in BeeLlama's ``cross_bucket()`` (llama-context.cpp
    ~line 3649).  Cycle-to-cycle live context lengths land in the same bucket
    for long stretches, which is what lets a single captured HIP graph replay
    across many cycles.
    """

    n = int(context_tokens)
    if n <= 16:
        return 16
    if n <= 128:
        bucket = 16
        while bucket < n:
            bucket *= 2
        return bucket
    return ((n + 127) // 128) * 128


def _top1_probability_from_topk(row_values: Sequence[float]) -> float:
    """Return the top-1 probability after softmax over a compact top-k row."""

    if not row_values:
        raise ValueError("top-k row must not be empty")
    values = [float(value) for value in row_values]
    if len(values) == 1:
        return 1.0
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    denom = sum(exps)
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return float(exps[0] / denom)


def _top1_probabilities_from_topk(topk_values: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(_top1_probability_from_topk(row) for row in topk_values)


def _confidence_limited_active_count(
    top1_probabilities: Sequence[float],
    *,
    max_active: int,
    p_min: float,
) -> int:
    """Stop a chain draft at the first candidate below the confidence floor."""

    if max_active < 0:
        raise ValueError("max_active must be non-negative")
    if p_min <= 0.0:
        return int(max_active)
    count = 0
    for probability in top1_probabilities[: int(max_active)]:
        if float(probability) < float(p_min):
            break
        count += 1
    return count


@dataclass(frozen=True)
class DFlashDrafterGraphBucket:
    candidate_budget: int
    block_size: int
    query_rows: int
    context_tokens: int
    max_context_tokens: int
    num_layers: int
    hidden_size: int
    bucket_context_tokens: int
    bucket_mode: str = "exact"
    draft_top_k: int = 1
    mode: str = "append_only_projected_context_and_kv"

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_budget": self.candidate_budget,
            "block_size": self.block_size,
            "query_rows": self.query_rows,
            "context_tokens": self.context_tokens,
            "bucket_context_tokens": self.bucket_context_tokens,
            "bucket_mode": self.bucket_mode,
            "max_context_tokens": self.max_context_tokens,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "draft_top_k": self.draft_top_k,
            "mode": self.mode,
        }

    @property
    def key(self) -> tuple[int, int, int, int, int, int, int, int, str, str]:
        # Cache key drops live ``context_tokens`` when bucketed so cycles with
        # the same ``bucket_context_tokens`` reuse the captured graph.  In
        # ``exact`` mode the bucket equals context_tokens, restoring the old
        # per-cycle key behavior.
        return (
            self.candidate_budget,
            self.block_size,
            self.query_rows,
            self.bucket_context_tokens,
            self.max_context_tokens,
            self.num_layers,
            self.hidden_size,
            self.draft_top_k,
            self.mode,
            self.bucket_mode,
        )


@dataclass
class DFlashDrafterGraphEntry:
    bucket: DFlashDrafterGraphBucket
    graph: int
    graph_exec: int
    stream: int
    validation_passed: bool
    direct_tokens: tuple[int, ...]
    graph_tokens: tuple[int, ...]
    capture_seconds: float
    instantiate_seconds: float
    validation_seconds: float
    replay_count: int = 0


def _build_chain_target_batch(
    *,
    root_token: int,
    root_position: int,
    candidates: Sequence[int],
    candidate_budget: int,
    active_count: int,
) -> TargetVerifyBatch:
    """Build fixed-budget root+B target metadata for native chain verification."""

    if candidate_budget <= 0:
        raise ValueError("candidate_budget must be positive")
    if active_count < 0 or active_count > candidate_budget:
        raise ValueError("active_count must be in [0, candidate_budget]")
    padded = [0] * candidate_budget
    for index, token in enumerate(candidates[:active_count]):
        padded[index] = int(token)
    draft = DraftBatch(
        request_ids=(0,),
        candidate_tokens=tuple(padded),
        parent_positions=tuple(int(root_position) + index for index in range(candidate_budget)),
        draft_depths=tuple(index + 1 for index in range(candidate_budget)),
        row_to_request=tuple(0 for _ in range(candidate_budget)),
        active_mask=tuple(index < active_count for index in range(candidate_budget)),
        mode="verify_chain",
    )
    return TargetVerifyBatch.from_draft(draft, root_tokens=(int(root_token),), root_positions=(int(root_position),))


def _build_flat_fan_tree_target_batch(
    *,
    root_token: int,
    root_position: int,
    candidates: Sequence[int],
    candidate_budget: int,
    active_count: int,
) -> TargetVerifyBatch:
    """Build a depth-1 flat-fan tree from the chain drafter's candidates.

    Each of the ``B`` chain candidates becomes a depth-1 sibling of the root
    (``tree_parents = (-1,) * B``).  All siblings share RoPE phase
    ``root_position + 1`` but get unique cache slots via the tree verifier's
    cache-slot disambiguation.  This is the MINIMUM tree shape that exercises
    the tree-aware GQA gate kernel + ancestor mask on real prompts.  The
    expected acceptance is at most 1 token per cycle (chain DFlash with B
    candidates can chain accepts up to B), so this shape is NOT intended to
    beat chain on tok/s -- it measures the tree verifier kernel cost on a
    realistic decode loop.
    """

    if candidate_budget <= 0:
        raise ValueError("candidate_budget must be positive")
    if active_count < 0 or active_count > candidate_budget:
        raise ValueError("active_count must be in [0, candidate_budget]")
    padded = [0] * candidate_budget
    for index, token in enumerate(candidates[:active_count]):
        padded[index] = int(token)
    draft = DraftBatch(
        request_ids=(0,),
        candidate_tokens=tuple(padded),
        # All candidates branch from the root at the same depth, so they
        # share the same parent_position (root_position) and depth 1.
        parent_positions=tuple(int(root_position) for _ in range(candidate_budget)),
        draft_depths=tuple(1 for _ in range(candidate_budget)),
        row_to_request=tuple(0 for _ in range(candidate_budget)),
        # tree_parents = -1 for every candidate means "parent is the root".
        tree_parents=tuple(-1 for _ in range(candidate_budget)),
        active_mask=tuple(index < active_count for index in range(candidate_budget)),
        mode="verify_tree",
    )
    return TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(int(root_token),),
        root_positions=(int(root_position),),
    )


def _build_chain_as_tree_target_batch(
    *,
    root_token: int,
    root_position: int,
    candidates: Sequence[int],
    candidate_budget: int,
    active_count: int,
) -> TargetVerifyBatch:
    """Wrap the chain drafter output as a degenerate (linear) tree.

    parent_rows form a single chain (``tree_parents = (-1, 0, 1, ..., B-2)``),
    which is the chain DFlash topology re-expressed in tree mode.  Used to
    measure the tree verifier kernel overhead vs chain at the SAME logical
    accept rate (both can accept up to B tokens in sequence).  Per-row K/V
    layout is dense -- ancestor mask is lower-triangular -- so this is
    bit-equal to the chain batched path on the full-attention layers.
    """

    if candidate_budget <= 0:
        raise ValueError("candidate_budget must be positive")
    if active_count < 0 or active_count > candidate_budget:
        raise ValueError("active_count must be in [0, candidate_budget]")
    padded = [0] * candidate_budget
    for index, token in enumerate(candidates[:active_count]):
        padded[index] = int(token)
    draft = DraftBatch(
        request_ids=(0,),
        candidate_tokens=tuple(padded),
        parent_positions=tuple(int(root_position) + index for index in range(candidate_budget)),
        draft_depths=tuple(index + 1 for index in range(candidate_budget)),
        row_to_request=tuple(0 for _ in range(candidate_budget)),
        tree_parents=tuple(-1 if index == 0 else index - 1 for index in range(candidate_budget)),
        active_mask=tuple(index < active_count for index in range(candidate_budget)),
        mode="verify_tree",
    )
    return TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(int(root_token),),
        root_positions=(int(root_position),),
    )


@dataclass(frozen=True)
class _CompiledTopKTree:
    target_batch: TargetVerifyBatch
    active_candidate_tokens: tuple[int, ...]
    tree_parents: tuple[int, ...]
    draft_depths: tuple[int, ...]
    child_ranks: tuple[int, ...]
    cumulative_scores: tuple[float, ...]
    active_count: int


@dataclass(frozen=True)
class _TopKTreeNode:
    token: int
    parent: int
    depth: int
    child_rank: int
    score: float


def _compile_balanced_topk_tree_nodes(
    *,
    topk_tokens: Sequence[Sequence[int]],
    topk_values: Sequence[Sequence[float]],
    candidate_budget: int,
    tree_top_k: int,
    max_depth: int,
) -> tuple[_TopKTreeNode, ...]:
    """Compile per-depth top-K logits into a fixed-budget topological DDTree.

    MVP policy: balanced breadth-first expansion.  Depth 1 contains up to K
    siblings from the root.  Deeper levels expand every frontier parent
    round-robin by child rank, so B=4,K=2 yields parents ``[-1, -1, 0, 1]``
    (two root choices, then one continuation under each).  The DFlash query
    block supplies one top-K distribution per depth; deeper child logits are
    therefore depth-conditioned rather than branch-specific.  This is the
    smallest true branching tree that exercises DDTree accept/commit semantics
    without adding a branch re-forward drafter yet.
    """

    if candidate_budget <= 0:
        raise ValueError("candidate_budget must be positive")
    if tree_top_k <= 0 or tree_top_k > 8:
        raise ValueError("tree_top_k must be in [1, 8]")
    if max_depth <= 0:
        return ()
    if len(topk_tokens) < max_depth or len(topk_values) < max_depth:
        raise ValueError("topk rows must cover max_depth")

    nodes: list[_TopKTreeNode] = []
    frontier: list[int] = [-1]
    for depth in range(1, int(max_depth) + 1):
        row_tokens = tuple(int(token) for token in topk_tokens[depth - 1])
        row_values = tuple(float(value) for value in topk_values[depth - 1])
        if len(row_tokens) < tree_top_k or len(row_values) < tree_top_k:
            raise ValueError("each topk row must contain at least tree_top_k entries")
        next_frontier: list[int] = []
        for rank in range(int(tree_top_k)):
            for parent in frontier:
                if len(nodes) >= candidate_budget:
                    return tuple(nodes)
                parent_score = 0.0 if parent < 0 else nodes[parent].score
                node = _TopKTreeNode(
                    token=int(row_tokens[rank]),
                    parent=int(parent),
                    depth=int(depth),
                    child_rank=int(rank),
                    score=float(parent_score + row_values[rank]),
                )
                nodes.append(node)
                next_frontier.append(len(nodes) - 1)
        frontier = next_frontier
        if not frontier:
            break
    return tuple(nodes)


def _build_branching_topk_tree_target_batch(
    *,
    root_token: int,
    root_position: int,
    topk_tokens: Sequence[Sequence[int]],
    topk_values: Sequence[Sequence[float]],
    candidate_budget: int,
    tree_top_k: int,
    max_depth: int,
) -> _CompiledTopKTree:
    """Build a padded fixed-row DDTree verifier batch from row-wise top-K."""

    if candidate_budget <= 0:
        raise ValueError("candidate_budget must be positive")
    nodes = _compile_balanced_topk_tree_nodes(
        topk_tokens=topk_tokens,
        topk_values=topk_values,
        candidate_budget=candidate_budget,
        tree_top_k=tree_top_k,
        max_depth=max_depth,
    )
    padded_tokens = [0] * candidate_budget
    parent_positions = [int(root_position)] * candidate_budget
    draft_depths = [1] * candidate_budget
    tree_parents = [-1] * candidate_budget
    child_ranks = [0] * candidate_budget
    cumulative_scores = [float("-inf")] * candidate_budget
    active_mask = [False] * candidate_budget
    for index, node in enumerate(nodes):
        padded_tokens[index] = int(node.token)
        parent_positions[index] = int(root_position) + int(node.depth) - 1
        draft_depths[index] = int(node.depth)
        tree_parents[index] = int(node.parent)
        child_ranks[index] = int(node.child_rank)
        cumulative_scores[index] = float(node.score)
        active_mask[index] = True
    draft = DraftBatch(
        request_ids=(0,),
        candidate_tokens=tuple(padded_tokens),
        parent_positions=tuple(parent_positions),
        draft_depths=tuple(draft_depths),
        row_to_request=tuple(0 for _ in range(candidate_budget)),
        tree_parents=tuple(tree_parents),
        active_mask=tuple(active_mask),
        mode="verify_tree",
    )
    target_batch = TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(int(root_token),),
        root_positions=(int(root_position),),
    )
    return _CompiledTopKTree(
        target_batch=target_batch,
        active_candidate_tokens=tuple(int(node.token) for node in nodes),
        tree_parents=tuple(int(node.parent) for node in nodes),
        draft_depths=tuple(int(node.depth) for node in nodes),
        child_ranks=tuple(int(node.child_rank) for node in nodes),
        cumulative_scores=tuple(float(node.score) for node in nodes),
        active_count=len(nodes),
    )


class NativeDFlashChainDrafter:
    """Correctness-first native DFlash top-1/top-K chain proposer.

    The implementation uses append-only projected-context and per-layer K/V caches
    so per-cycle proposals only process query rows.
    """

    def __init__(
        self,
        *,
        session: Qwen35ParoResidentSession,
        drafter_model: str | Path,
        max_context_tokens: int,
        candidate_budget: int,
        compiler_version: str | None,
        require_cached_build: bool,
        sync_draft_phases: bool = False,
        graph_mode: str = "off",
        fusion_mode: str = "off",
        draft_top_k: int = 1,
        bucket_mode: str = "exact",
        query_mode: str = "block",
    ) -> None:
        self.session = session
        self.runtime = session.runtime
        self.device = Device("hip", 0)
        self.candidate_budget = int(candidate_budget)
        self.draft_top_k = int(draft_top_k)
        if self.draft_top_k <= 0 or self.draft_top_k > 8:
            raise ValueError("draft_top_k must be in [1, 8]")
        self.sync_draft_phases = bool(sync_draft_phases)
        if graph_mode not in {"off", "auto", "validate"}:
            raise ValueError("graph_mode must be off, auto, or validate")
        self.graph_mode = graph_mode
        if fusion_mode not in {"off", "qkv"}:
            raise ValueError("fusion_mode must be off or qkv")
        self.fusion_mode = fusion_mode
        if bucket_mode not in {"exact", "cross_bucket"}:
            raise ValueError("bucket_mode must be exact or cross_bucket")
        self.bucket_mode = bucket_mode
        if query_mode not in {"block", "budget_prefix"}:
            raise ValueError("query_mode must be block or budget_prefix")
        self.query_mode = query_mode
        # When bucketed propose runs, this device-resident scalar carries the
        # live context length so the same captured graph can replay across
        # cycles with different live counts.
        self._bucket_live_count_pending: int | None = None
        self._fusion_counts: Counter[str] = Counter()
        self._graph_cache: dict[tuple[int, int, int, int, int, int, int, int, str, str], DFlashDrafterGraphEntry] = {}
        self._graph_status_counts: Counter[str] = Counter()
        self._graph_validation_failures = 0
        self._graph_fallback_reasons: Counter[str] = Counter()
        self._graph_last: dict[str, Any] | None = None
        # Cache of how many context rows have already been projected through
        # ``fc + hidden_norm`` and live in self.projected_context_norm.  The
        # drafter only re-projects the newly committed tail rows per cycle.
        self._cached_projected_rows = 0
        # Track whether the projected_context_norm cache covers a contiguous
        # prefix; used by ``commit_context_rows`` to detect stale state and to
        # transparently rebuild the prefix on demand.
        self._cache_invalidated = False
        # Per-layer KV cache state: rotated K (FP32) and V (BF16) for every
        # committed context row.  Mirrors ``_cached_projected_rows`` and is
        # extended each cycle through ``commit_context_rows``.
        self._cached_kv_rows = 0
        self._kv_cache_invalidated = False
        self.drafter_index = load_weight_index(drafter_model)
        self.weights = load_dflash_drafter_bf16_weights(
            self.drafter_index,
            runtime=self.runtime,
            device=self.device,
            layer_limit=None,
        )
        self.config = self.weights.config
        if self.candidate_budget <= 0 or self.candidate_budget >= self.config.block_size:
            raise ValueError("candidate_budget must be in [1, block_size - 1]")
        self.max_context_tokens = int(max_context_tokens)
        self.hidden = int(self.config.hidden_size)
        self.intermediate = int(self.config.intermediate_size)
        self.q_heads = int(self.config.num_attention_heads)
        self.kv_heads = int(self.config.num_key_value_heads)
        self.head_dim = int(self.config.head_dim)
        self.attn_features = self.q_heads * self.head_dim
        self.kv_features = self.kv_heads * self.head_dim
        self.vocab_size = int(self.config.vocab_size)
        self.block_size = int(self.config.block_size)
        self.query_rows = self.block_size if self.query_mode == "block" else self.candidate_budget + 1
        if self.query_rows <= self.candidate_budget or self.query_rows > self.block_size:
            raise ValueError("query_rows must fit candidate_budget + root within block_size")
        self.buffers: list[DeviceBuffer] = []
        with hip_target_arch_environment(session.target_arch):
            self.library = build_dflash_drafter(load=True, compiler_version=compiler_version, require_cached=require_cached_build)
            self.lm_library = build_lm_head(load=True, compiler_version=compiler_version, require_cached=require_cached_build)
            self.cast_library = build_cast(load=True, compiler_version=compiler_version, require_cached=require_cached_build)
        self._allocate()

    @property
    def target_layer_ids(self) -> tuple[int, ...]:
        return self.config.target_layer_ids

    @property
    def target_hidden_concat(self) -> Tensor:
        return self._target_hidden_concat

    @property
    def graph_summary(self) -> dict[str, Any]:
        validation_passed = None
        if self._graph_cache:
            validation_passed = self._graph_validation_failures == 0
        return {
            "mode": self.graph_mode,
            "bucket_mode": self.bucket_mode,
            "query_mode": self.query_mode,
            "query_rows": int(self.query_rows),
            "status_counts": dict(sorted(self._graph_status_counts.items())),
            "cache_entries": len(self._graph_cache),
            "validation_failures": self._graph_validation_failures,
            "validation_passed": validation_passed,
            "fallback_reasons": dict(sorted(self._graph_fallback_reasons.items())),
            "last": self._graph_last,
        }

    @property
    def fusion_summary(self) -> dict[str, Any]:
        return {
            "mode": self.fusion_mode,
            "counts": dict(sorted(self._fusion_counts.items())),
            "fallback": self.fusion_mode == "off",
            "active": self.fusion_mode == "qkv",
        }

    def close(self) -> None:
        for entry in list(self._graph_cache.values()):
            try:
                self.runtime.graph_exec_destroy(entry.graph_exec)
            except Exception:
                pass
            try:
                self.runtime.graph_destroy(entry.graph)
            except Exception:
                pass
            try:
                self.runtime.stream_destroy(entry.stream)
            except Exception:
                pass
        self._graph_cache.clear()
        self.weights.free(runtime=self.runtime)
        for buffer in reversed(self.buffers):
            free(buffer, runtime=self.runtime)
        self.buffers.clear()

    def __enter__(self) -> "NativeDFlashChainDrafter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def propose(self, *, root_token: int, root_position: int, context_tokens: int) -> DraftResult:
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive before DFlash draft")
        if context_tokens > self.max_context_tokens:
            raise ValueError("context_tokens exceeds DFlash context capacity")
        t0 = time.perf_counter()
        phases: dict[str, float] = {}
        self._write_root_inputs(root_token=root_token, root_position=root_position)
        phases["key_positions_h2d"] = 0.0
        self._ensure_context_cache(context_tokens=context_tokens, phases=phases)

        graph_info: dict[str, Any]
        if self.graph_mode == "off":
            self._run_propose_kernels(context_tokens=context_tokens, stream=0, phases=phases)
            self.runtime.device_synchronize()
            graph_info = self._graph_info(
                status="disabled",
                bucket=self._bucket_for(context_tokens),
                replayed=False,
                validation_passed=None,
                fallback_reason="drafter graph mode is off",
            )
            self._graph_fallback_reasons["drafter graph mode is off"] += 1
        else:
            graph_info = self._run_or_validate_graph_bucket(context_tokens=context_tokens, phases=phases)

        topk, topk_values = self._read_topk()
        draft_seconds = time.perf_counter() - t0
        phases["total"] = draft_seconds
        phases["graph_overhead"] = float(graph_info.get("overhead_seconds") or 0.0)
        if graph_info.get("status") == "replayed":
            phases["graph_replay"] = phases["graph_overhead"]
            phases.setdefault("noise_prepare", 0.0)
            phases.setdefault("decoder_layers", 0.0)
            phases.setdefault("final_norm", 0.0)
            phases.setdefault("lm_head", 0.0)
            phases.setdefault("topk_and_readback", 0.0)
            phases.setdefault("slowest_decoder_layer", 0.0)
        self._graph_last = graph_info
        self._graph_status_counts[str(graph_info.get("status", "unknown"))] += 1
        topk_token_rows = tuple(tuple(int(x) for x in row) for row in topk.tolist())
        topk_value_rows = tuple(tuple(float(x) for x in row) for row in topk_values.tolist())
        top1_probabilities = _top1_probabilities_from_topk(topk_value_rows)
        return DraftResult(
            candidate_tokens=tuple(int(row[0]) for row in topk_token_rows),
            draft_seconds=draft_seconds,
            finite_logits=bool(np.isfinite(topk_values).all()),
            d2h_vector_reads=2,
            d2h_vector_values=2 * self.candidate_budget * self.draft_top_k,
            phase_seconds=phases,
            graph=graph_info,
            topk_tokens=topk_token_rows,
            topk_values=topk_value_rows,
            top1_probabilities=top1_probabilities,
        )

    def _write_root_inputs(self, *, root_token: int, root_position: int) -> None:
        root = np.asarray([int(root_token)], dtype=np.int32)
        pos = np.asarray([int(root_position)], dtype=np.int32)
        copy_host_to_device(self._buffer_for(self.root_tokens), host_array_ptr(root), runtime=self.runtime)
        copy_host_to_device(self._buffer_for(self.root_positions), host_array_ptr(pos), runtime=self.runtime)

    def _ensure_context_cache(self, *, context_tokens: int, phases: dict[str, float]) -> None:
        phase_t = time.perf_counter()
        context_projection_rebuild_rows = 0
        if self._cache_invalidated or self._cached_projected_rows < context_tokens:
            rebuild_start = 0 if self._cache_invalidated else self._cached_projected_rows
            rebuild_count = context_tokens - rebuild_start
            self._project_context_rows(start=rebuild_start, count=rebuild_count)
            self._cached_projected_rows = context_tokens
            self._cache_invalidated = False
            context_projection_rebuild_rows = rebuild_count
        if self._kv_cache_invalidated or self._cached_kv_rows < context_tokens:
            kv_start = 0 if self._kv_cache_invalidated else self._cached_kv_rows
            kv_count = context_tokens - kv_start
            self._project_kv_cache_rows(start=kv_start, count=kv_count)
            self._cached_kv_rows = context_tokens
            self._kv_cache_invalidated = False
        self._record_phase(phases, "context_projection", phase_t)
        phases["context_projection_rebuild_rows"] = float(context_projection_rebuild_rows)
        phases["context_projection_cached_rows"] = float(self._cached_projected_rows)
        phases["kv_cache_cached_rows"] = float(self._cached_kv_rows)

    def _run_propose_kernels(
        self,
        *,
        context_tokens: int,
        stream: int = 0,
        phases: dict[str, float] | None = None,
        bucket_ctx: int | None = None,
    ) -> None:
        """Run the DFlash propose forward.

        When ``bucket_ctx`` is ``None`` (default) the kernels run with shape
        ``context_tokens + query_rows`` for K/V (exact path, same as before).
        When ``bucket_ctx`` is an int ``>= context_tokens`` the kernels run with
        shape ``bucket_ctx + query_rows`` and use the bucketed attention kernel;
        the live context length is read from ``self.live_context_len`` on the
        device so a captured HIP graph can replay across cycles whose live
        counts share the same bucket.  The caller must populate
        ``self.live_context_len`` (via ``_write_live_context_len``) before any
        replay path reads it.
        """
        phase_t = time.perf_counter()
        query_rows = int(self.query_rows)
        prepare = (
            dflash_prepare_noise_inputs_bf16_i32
            if self.session.embedding.tensor.dtype == DType.BF16
            else dflash_prepare_noise_inputs_f16_to_bf16_i32
        )
        prepare(
            self.root_tokens.ptr,
            self.root_positions.ptr,
            self.session.embedding.tensor.ptr,
            self.noise_ids.ptr,
            self.query_positions.ptr,
            self.query_hidden_a.ptr,
            1,
            query_rows,
            self.hidden,
            self.session.vocab_size,
            self.config.mask_token_id,
            threads=256,
            stream=stream,
            library=self.library,
            runtime=self.runtime,
        )
        if phases is not None:
            self._record_phase(phases, "noise_prepare", phase_t)
        query_in = self.query_hidden_a
        query_out = self.query_hidden_b
        layer_seconds: list[float] = []
        layers_t = time.perf_counter()
        for layer in range(self.config.num_hidden_layers):
            layer_t = time.perf_counter()
            query_out = self._run_layer(
                layer,
                context_tokens=context_tokens,
                query_in=query_in,
                query_out=query_out,
                stream=stream,
                bucket_ctx=bucket_ctx,
            )
            query_in, query_out = query_out, query_in
            if phases is not None and self.sync_draft_phases:
                self.runtime.device_synchronize()
            if phases is not None:
                layer_seconds.append(time.perf_counter() - layer_t)
        if phases is not None:
            if self.sync_draft_phases:
                self.runtime.device_synchronize()
            phases["decoder_layers"] = time.perf_counter() - layers_t
            phases["slowest_decoder_layer"] = max(layer_seconds) if layer_seconds else 0.0
        phase_t = time.perf_counter()
        dflash_rmsnorm_bf16(
            query_in.ptr,
            self.weights.tensor("norm.weight").ptr,
            self.final_norm.ptr,
            query_rows,
            self.hidden,
            threads=128,
            stream=stream,
            library=self.library,
            runtime=self.runtime,
        )
        if phases is not None:
            self._record_phase(phases, "final_norm", phase_t)
        phase_t = time.perf_counter()
        logits_ptr = self.logits.ptr
        w8a16_linear_bf16_f32_out(
            self.final_norm.ptr + self.hidden * DType.BF16.itemsize,
            self.session.lm_head_weight.tensor.ptr,
            self.session.lm_head_scale.tensor.ptr,
            logits_ptr,
            self.candidate_budget,
            self.hidden,
            self.vocab_size,
            threads=128,
            stream=stream,
            library=self.session.libraries["w8a16"],
            runtime=self.runtime,
        )
        if phases is not None:
            self._record_phase(phases, "lm_head", phase_t)
        phase_t = time.perf_counter()
        topk_f32_rows_i32(
            logits_ptr,
            self.top1_values.ptr,
            self.top1_ids.ptr,
            self.candidate_budget,
            self.vocab_size,
            self.draft_top_k,
            threads=256,
            stream=stream,
            library=self.lm_library,
            runtime=self.runtime,
        )
        if phases is not None:
            self.runtime.device_synchronize()
            phases["topk_and_readback"] = time.perf_counter() - phase_t

    def _read_topk(self) -> tuple[np.ndarray, np.ndarray]:
        topk = np.empty((self.candidate_budget, self.draft_top_k), dtype=np.int32)
        topk_values = np.empty((self.candidate_budget, self.draft_top_k), dtype=np.float32)
        copy_device_to_host(host_array_ptr(topk), self._buffer_for(self.top1_ids), runtime=self.runtime)
        copy_device_to_host(host_array_ptr(topk_values), self._buffer_for(self.top1_values), runtime=self.runtime)
        return topk, topk_values

    def _bucket_for(self, context_tokens: int) -> DFlashDrafterGraphBucket:
        ctx = int(context_tokens)
        if self.bucket_mode == "cross_bucket":
            bucket_ctx = _dflash_cross_bucket(ctx)
            # Cap the bucket so we never reserve more KV rows than the
            # drafter has allocated for context.  ``max_context_tokens`` is the
            # absolute upper bound of context rows the drafter can hold.
            bucket_ctx = min(bucket_ctx, self.max_context_tokens)
        else:
            bucket_ctx = ctx
        return DFlashDrafterGraphBucket(
            candidate_budget=self.candidate_budget,
            block_size=self.block_size,
            query_rows=self.query_rows,
            context_tokens=ctx,
            bucket_context_tokens=int(bucket_ctx),
            bucket_mode=self.bucket_mode,
            max_context_tokens=self.max_context_tokens,
            num_layers=int(self.config.num_hidden_layers),
            hidden_size=self.hidden,
            draft_top_k=self.draft_top_k,
        )

    def _graph_info(
        self,
        *,
        status: str,
        bucket: DFlashDrafterGraphBucket,
        replayed: bool,
        validation_passed: bool | None,
        fallback_reason: str | None = None,
        overhead_seconds: float = 0.0,
        capture_seconds: float | None = None,
        instantiate_seconds: float | None = None,
        validation_seconds: float | None = None,
        cache_hit: bool = False,
        direct_tokens: tuple[int, ...] | None = None,
        graph_tokens: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "mode": self.graph_mode,
            "bucket_key": bucket.as_dict(),
            "replayed": bool(replayed),
            "cache_hit": bool(cache_hit),
            "validation_passed": validation_passed,
            "fallback_reason": fallback_reason,
            "overhead_seconds": float(overhead_seconds),
            "capture_seconds": None if capture_seconds is None else float(capture_seconds),
            "instantiate_seconds": None if instantiate_seconds is None else float(instantiate_seconds),
            "validation_seconds": None if validation_seconds is None else float(validation_seconds),
            "direct_tokens": None if direct_tokens is None else list(direct_tokens),
            "graph_tokens": None if graph_tokens is None else list(graph_tokens),
            "cache_entries": len(self._graph_cache),
        }

    def _run_or_validate_graph_bucket(self, *, context_tokens: int, phases: dict[str, float]) -> dict[str, Any]:
        bucket = self._bucket_for(context_tokens)
        # ``bucket_arg`` is the bucketed KV context length used by the graphed
        # kernels; ``None`` falls back to exact (legacy) shape.
        bucket_arg: int | None = (
            int(bucket.bucket_context_tokens) if self.bucket_mode == "cross_bucket" else None
        )
        if bucket_arg is not None:
            # The bucketed path always reads the live count from the device
            # scalar; update it before any kernel that consumes it, including
            # the direct fallback used for validation.
            self._write_live_context_len(int(context_tokens))
        entry = self._graph_cache.get(bucket.key)
        if entry is not None and self.graph_mode == "auto":
            launch_t = time.perf_counter()
            # Context cache/materialization work above uses the default stream;
            # synchronize before replaying on the graph-owned stream.
            self.runtime.device_synchronize()
            self.runtime.graph_launch(entry.graph_exec, entry.stream)
            self.runtime.stream_synchronize(entry.stream)
            entry.replay_count += 1
            return self._graph_info(
                status="replayed",
                bucket=bucket,
                replayed=True,
                validation_passed=entry.validation_passed,
                overhead_seconds=time.perf_counter() - launch_t,
                capture_seconds=entry.capture_seconds,
                instantiate_seconds=entry.instantiate_seconds,
                validation_seconds=entry.validation_seconds,
                cache_hit=True,
                direct_tokens=entry.direct_tokens,
                graph_tokens=entry.graph_tokens,
            )

        # Cache miss: run the normal direct path once for the value returned by
        # this propose() call, then capture and replay the exact same fixed-shape
        # body as a validation sample.  In decode workloads context_tokens changes
        # every cycle, so this commonly records "captured_validated" without
        # useful cache hits; the artifact makes that visible.
        direct_t = time.perf_counter()
        # Direct (validation reference) runs the bucketed propose with the SAME
        # bucket shape as the graph capture so the cached graph and the direct
        # path are bit-equivalent.  In exact mode bucket_arg is None and this
        # reduces to the legacy unbucketed path.
        self._run_propose_kernels(
            context_tokens=context_tokens, stream=0, phases=phases, bucket_ctx=bucket_arg
        )
        self.runtime.device_synchronize()
        direct_tokens_arr, _ = self._read_topk()
        direct_tokens = tuple(int(x) for x in direct_tokens_arr.reshape(-1).tolist())
        direct_seconds = time.perf_counter() - direct_t
        graph = 0
        stream = 0
        capture_seconds = 0.0
        instantiate_seconds = 0.0
        validation_seconds = 0.0
        try:
            # Ensure root/position copies and context/KV cache updates are visible
            # to the non-default capture stream.
            self.runtime.device_synchronize()
            stream = self.runtime.stream_create()
            capture_t = time.perf_counter()
            self.runtime.stream_begin_capture(stream)
            try:
                self._run_propose_kernels(
                    context_tokens=context_tokens,
                    stream=stream,
                    phases=None,
                    bucket_ctx=bucket_arg,
                )
                graph = self.runtime.stream_end_capture(stream)
            except Exception:
                try:
                    self.runtime.stream_end_capture(stream)
                except Exception:
                    pass
                raise
            capture_seconds = time.perf_counter() - capture_t
            instantiate_t = time.perf_counter()
            graph_exec = self.runtime.graph_instantiate(graph)
            instantiate_seconds = time.perf_counter() - instantiate_t
            validate_t = time.perf_counter()
            self.runtime.graph_launch(graph_exec, stream)
            self.runtime.stream_synchronize(stream)
            graph_tokens_arr, _ = self._read_topk()
            graph_tokens = tuple(int(x) for x in graph_tokens_arr.reshape(-1).tolist())
            validation_seconds = time.perf_counter() - validate_t
            validation_passed = graph_tokens == direct_tokens
            if not validation_passed:
                self._graph_validation_failures += 1
                reason = "graph replay candidates differed from direct fallback"
                self._graph_fallback_reasons[reason] += 1
                # Restore direct fallback outputs before propose() performs its
                # final readback; validation failure must not perturb the chain.
                self._run_propose_kernels(
                    context_tokens=context_tokens,
                    stream=0,
                    phases=None,
                    bucket_ctx=bucket_arg,
                )
                self.runtime.device_synchronize()
                self.runtime.graph_exec_destroy(graph_exec)
                self.runtime.graph_destroy(graph)
                self.runtime.stream_destroy(stream)
                return self._graph_info(
                    status="validation_failed",
                    bucket=bucket,
                    replayed=False,
                    validation_passed=False,
                    fallback_reason=reason,
                    overhead_seconds=time.perf_counter() - direct_t - direct_seconds,
                    capture_seconds=capture_seconds,
                    instantiate_seconds=instantiate_seconds,
                    validation_seconds=validation_seconds,
                    direct_tokens=direct_tokens,
                    graph_tokens=graph_tokens,
                )
            entry = DFlashDrafterGraphEntry(
                bucket=bucket,
                graph=graph,
                graph_exec=graph_exec,
                stream=stream,
                validation_passed=True,
                direct_tokens=direct_tokens,
                graph_tokens=graph_tokens,
                capture_seconds=capture_seconds,
                instantiate_seconds=instantiate_seconds,
                validation_seconds=validation_seconds,
                replay_count=1,
            )
            self._graph_cache[bucket.key] = entry
            status = "captured_validated" if self.graph_mode == "validate" else "captured_validated_miss"
            return self._graph_info(
                status=status,
                bucket=bucket,
                replayed=self.graph_mode == "auto",
                validation_passed=True,
                overhead_seconds=time.perf_counter() - direct_t - direct_seconds,
                capture_seconds=capture_seconds,
                instantiate_seconds=instantiate_seconds,
                validation_seconds=validation_seconds,
                cache_hit=False,
                direct_tokens=direct_tokens,
                graph_tokens=graph_tokens,
            )
        except Exception as exc:
            if graph:
                try:
                    self.runtime.graph_destroy(graph)
                except Exception:
                    pass
            if stream:
                try:
                    self.runtime.stream_destroy(stream)
                except Exception:
                    pass
            reason = f"capture_failed: {exc}"
            self._graph_fallback_reasons[reason] += 1
            return self._graph_info(
                status="capture_failed_fallback",
                bucket=bucket,
                replayed=False,
                validation_passed=None,
                fallback_reason=reason,
                overhead_seconds=time.perf_counter() - direct_t - direct_seconds,
                capture_seconds=capture_seconds,
                instantiate_seconds=instantiate_seconds,
                validation_seconds=validation_seconds,
                direct_tokens=direct_tokens,
            )

    def _record_phase(self, phases: dict[str, float], name: str, started_at: float) -> None:
        if self.sync_draft_phases:
            self.runtime.device_synchronize()
        phases[name] = time.perf_counter() - started_at

    def warmup_context(self, context_tokens: int) -> None:
        """Project the full prefill target-hidden context into the persistent caches.

        Called once after prefill and once after each cycle commit so that the
        per-call ``propose()`` path can skip both the per-cycle ``fc + hidden_norm``
        AND the per-layer context-side K/V projection on rows that have not
        changed.  The K cache stores rotated FP32 keys and the V cache stores
        BF16 values.
        """
        if context_tokens < 0:
            raise ValueError("context_tokens must be non-negative")
        if context_tokens == 0:
            self._cached_projected_rows = 0
            self._cached_kv_rows = 0
            self._cache_invalidated = False
            self._kv_cache_invalidated = False
            return
        self._project_context_rows(start=0, count=int(context_tokens))
        self._cached_projected_rows = int(context_tokens)
        self._cache_invalidated = False
        self._project_kv_cache_rows(start=0, count=int(context_tokens))
        self._cached_kv_rows = int(context_tokens)
        self._kv_cache_invalidated = False

    def commit_context_rows(self, *, start: int, count: int) -> None:
        """Append newly captured target-hidden rows into the projected + KV caches.

        ``start`` is the absolute context position of the first new row (matches
        the ``capture_row`` used by the verify forwards) and ``count`` is the
        number of committed rows for this cycle.  The drafter assumes
        ``self.target_hidden_concat[start:start+count]`` has already been written
        by the verify forwards before this call.  Both the projected-context and
        the per-layer K/V caches are extended in the same call so they stay in
        lockstep.
        """
        if start < 0 or count < 0:
            raise ValueError("start and count must be non-negative")
        if count == 0:
            return
        if start > self._cached_projected_rows or start > self._cached_kv_rows:
            # Hole in coverage; fall back to a full rebuild next propose().
            self._cache_invalidated = True
            self._kv_cache_invalidated = True
            return
        self._project_context_rows(start=start, count=count)
        self._cached_projected_rows = max(self._cached_projected_rows, start + count)
        self._project_kv_cache_rows(start=start, count=count)
        self._cached_kv_rows = max(self._cached_kv_rows, start + count)

    def _project_kv_cache_rows(self, *, start: int, count: int) -> None:
        if count <= 0:
            return
        if start < 0 or start + count > self.max_context_tokens:
            raise ValueError("KV context row range outside drafter capacity")
        bf16_bytes = DType.BF16.itemsize
        fp32_bytes = DType.FP32.itemsize
        proj_src_ptr = self.projected_context_norm.ptr + start * self.hidden * bf16_bytes
        pos_ptr = self.context_positions.ptr + start * DType.INT32.itemsize
        max_positions = int(self.cos.shape[0])
        for layer in range(int(self.config.num_hidden_layers)):
            prefix = f"layers.{layer}"
            k_dst_ptr = (
                self.kv_cache_keys.ptr
                + (layer * self.max_context_tokens + start) * self.kv_features * fp32_bytes
            )
            v_dst_ptr = (
                self.kv_cache_values.ptr
                + (layer * self.max_context_tokens + start) * self.kv_features * bf16_bytes
            )
            dflash_dense_bf16_to_f32(
                proj_src_ptr,
                self.weights.tensor(f"{prefix}.self_attn.k_proj.weight").ptr,
                self.kv_commit_k_raw.ptr,
                count,
                self.hidden,
                self.kv_features,
                threads=128,
                library=self.library,
                runtime=self.runtime,
            )
            dflash_key_rmsnorm_rotary_f32(
                self.kv_commit_k_raw.ptr,
                self.weights.tensor(f"{prefix}.self_attn.k_norm.weight").ptr,
                self.cos.ptr,
                self.sin.ptr,
                pos_ptr,
                k_dst_ptr,
                count,
                self.kv_heads,
                self.head_dim,
                self.head_dim,
                max_positions,
                threads=128,
                library=self.library,
                runtime=self.runtime,
            )
            dflash_dense_bf16_to_bf16(
                proj_src_ptr,
                self.weights.tensor(f"{prefix}.self_attn.v_proj.weight").ptr,
                v_dst_ptr,
                count,
                self.hidden,
                self.kv_features,
                threads=128,
                library=self.library,
                runtime=self.runtime,
            )

    def _project_context_rows(self, *, start: int, count: int) -> None:
        if count <= 0:
            return
        if start < 0 or start + count > self.max_context_tokens:
            raise ValueError("context row range outside drafter capacity")
        bf16_bytes = DType.BF16.itemsize
        concat_stride = self.config.target_hidden_concat_size * bf16_bytes
        hidden_stride = self.hidden * bf16_bytes
        src_ptr = self.target_hidden_concat.ptr + start * concat_stride
        proj_ptr = self.projected_context.ptr + start * hidden_stride
        norm_ptr = self.projected_context_norm.ptr + start * hidden_stride
        dflash_dense_bf16_to_bf16(
            src_ptr,
            self.weights.tensor("fc.weight").ptr,
            proj_ptr,
            count,
            self.config.target_hidden_concat_size,
            self.hidden,
            threads=128,
            library=self.library,
            runtime=self.runtime,
        )
        dflash_rmsnorm_bf16(
            proj_ptr,
            self.weights.tensor("hidden_norm.weight").ptr,
            norm_ptr,
            count,
            self.hidden,
            threads=128,
            library=self.library,
            runtime=self.runtime,
        )

    def _allocate(self) -> None:
        self.root_tokens = self._empty((1,), DType.INT32)
        self.root_positions = self._empty((1,), DType.INT32)
        self.noise_ids = self._empty((1, self.block_size), DType.INT32)
        self.query_positions = self._empty((1, self.block_size), DType.INT32)
        self.key_positions = self._empty((1, self.max_context_tokens + self.block_size), DType.INT32)
        self._target_hidden_concat = self._empty((self.max_context_tokens, self.config.target_hidden_concat_size), DType.BF16)
        self.projected_context = self._empty((self.max_context_tokens, self.hidden), DType.BF16)
        self.projected_context_norm = self._empty((self.max_context_tokens, self.hidden), DType.BF16)
        self.query_hidden_a = self._empty((self.block_size, self.hidden), DType.BF16)
        self.query_hidden_b = self._empty((self.block_size, self.hidden), DType.BF16)
        self.norm = self._empty((self.block_size, self.hidden), DType.BF16)
        self.q_raw = self._empty((self.block_size, self.attn_features), DType.FP32)
        self.k_ctx = self._empty((self.max_context_tokens, self.kv_features), DType.FP32)
        self.k_q = self._empty((self.block_size, self.kv_features), DType.FP32)
        self.k_all = self._empty((1, self.max_context_tokens + self.block_size, self.kv_features), DType.FP32)
        self.v_ctx = self._empty((self.max_context_tokens, self.kv_features), DType.BF16)
        self.v_q = self._empty((self.block_size, self.kv_features), DType.BF16)
        self.v_all = self._empty((1, self.max_context_tokens + self.block_size, self.kv_features), DType.BF16)
        self.q_rot = self._empty((1, self.block_size, self.q_heads, self.head_dim), DType.FP32)
        self.k_rot = self._empty((1, self.max_context_tokens + self.block_size, self.kv_heads, self.head_dim), DType.FP32)
        self.attn = self._empty((1, self.block_size, self.q_heads, self.head_dim), DType.BF16)
        self.attn_proj = self._empty((self.block_size, self.hidden), DType.BF16)
        self.hidden_attn = self._empty((self.block_size, self.hidden), DType.BF16)
        self.post = self._empty((self.block_size, self.hidden), DType.BF16)
        self.gate = self._empty((self.block_size, self.intermediate), DType.BF16)
        self.up = self._empty((self.block_size, self.intermediate), DType.BF16)
        self.act = self._empty((self.block_size, self.intermediate), DType.BF16)
        self.mlp = self._empty((self.block_size, self.hidden), DType.BF16)
        self.final_norm = self._empty((self.block_size, self.hidden), DType.BF16)
        self.logits = self._empty((self.candidate_budget, self.vocab_size), DType.FP32)
        self.top1_values = self._empty((self.candidate_budget, self.draft_top_k), DType.FP32)
        self.top1_ids = self._empty((self.candidate_budget, self.draft_top_k), DType.INT32)
        cos, sin = _rotary_tables(self.max_context_tokens + self.block_size + 8, self.head_dim, theta=float(self.config.rope_theta))
        self.cos = self._load_array(cos, DType.FP32)
        self.sin = self._load_array(sin, DType.FP32)
        # Phase C caches: per-layer rotated K (FP32) and V (BF16) for context
        # rows.  Per-cycle propose() only computes the block_size-sized query
        # K/V/Q + rotary; context K_ctx and V_ctx come from these caches.
        n_layers = int(self.config.num_hidden_layers)
        self.kv_cache_keys = self._empty(
            (n_layers, self.max_context_tokens, self.kv_features), DType.FP32
        )
        self.kv_cache_values = self._empty(
            (n_layers, self.max_context_tokens, self.kv_features), DType.BF16
        )
        # 1D context positions tensor [0, 1, ..., max_context-1].
        self.context_positions = self._empty((self.max_context_tokens,), DType.INT32)
        positions_host = np.arange(self.max_context_tokens, dtype=np.int32)
        copy_host_to_device(
            self._buffer_for(self.context_positions),
            host_array_ptr(positions_host),
            runtime=self.runtime,
        )
        # Scratch tensor for raw K rows before RMSNorm+rotary (one cycle worth).
        self.kv_commit_k_raw = self._empty(
            (self.max_context_tokens, self.kv_features), DType.FP32
        )
        # Rotated query-side K output (block_size rows), separate from k_rot so
        # we can concat cached K_ctx_rotated + k_q_rot directly into k_rot.
        self.k_q_rot = self._empty(
            (1, self.block_size, self.kv_features), DType.FP32
        )
        # R2.3: device-resident live context length scalar.  The bucketed
        # attention kernel reads this on every launch so a single captured HIP
        # graph can replay across cycles whose live context lengths share the
        # same bucket.  Initialized to 0; cycle code writes the value via
        # ``_write_live_context_len`` before launching kernels that read it.
        self.live_context_len = self._empty((1,), DType.INT32)
        zero_i32 = np.zeros((1,), dtype=np.int32)
        copy_host_to_device(
            self._buffer_for(self.live_context_len),
            host_array_ptr(zero_i32),
            runtime=self.runtime,
        )

    def _write_live_context_len(self, value: int) -> None:
        """Update the device-resident live context length scalar.

        Called per cycle from the bucketed propose path before any kernel that
        reads it; the small H2D copy is intentionally OUTSIDE any captured HIP
        graph so the value reaching the kernel is the latest live count even
        when the graph itself is replayed.
        """

        arr = np.asarray([int(value)], dtype=np.int32)
        copy_host_to_device(
            self._buffer_for(self.live_context_len),
            host_array_ptr(arr),
            runtime=self.runtime,
        )

    def _run_layer(
        self,
        layer: int,
        *,
        context_tokens: int,
        query_in: Tensor,
        query_out: Tensor,
        stream: int = 0,
        bucket_ctx: int | None = None,
    ) -> Tensor:
        prefix = f"layers.{layer}"
        # When ``bucket_ctx`` is set, concat reserves ``bucket_ctx`` context
        # rows (padded with zero/stale data beyond ``context_tokens``) and
        # attention masks those padded rows via the device-resident
        # ``live_context_len`` scalar.  This is what lets a captured HIP graph
        # replay across cycles whose live ``context_tokens`` share the same
        # bucket.
        if bucket_ctx is not None:
            if bucket_ctx < context_tokens:
                raise ValueError("bucket_ctx must be >= context_tokens")
            kv_context_len = int(bucket_ctx)
        else:
            kv_context_len = int(context_tokens)
        query_rows = int(self.query_rows)
        total_kv = kv_context_len + query_rows
        fp32_bytes = DType.FP32.itemsize
        bf16_bytes = DType.BF16.itemsize
        k_layer_base = self.kv_cache_keys.ptr + layer * self.max_context_tokens * self.kv_features * fp32_bytes
        v_layer_base = self.kv_cache_values.ptr + layer * self.max_context_tokens * self.kv_features * bf16_bytes
        dflash_rmsnorm_bf16(query_in.ptr, self.weights.tensor(f"{prefix}.input_layernorm.weight").ptr, self.norm.ptr, query_rows, self.hidden, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        if self.fusion_mode == "qkv":
            self._fusion_counts["qkv"] += 1
            dflash_qkv_proj_bf16_mixed(
                self.norm.ptr,
                self.weights.tensor(f"{prefix}.self_attn.q_proj.weight").ptr,
                self.weights.tensor(f"{prefix}.self_attn.k_proj.weight").ptr,
                self.weights.tensor(f"{prefix}.self_attn.v_proj.weight").ptr,
                self.q_raw.ptr,
                self.k_q.ptr,
                self.v_q.ptr,
                query_rows,
                self.hidden,
                self.attn_features,
                self.kv_features,
                threads=128,
                stream=stream,
                library=self.library,
                runtime=self.runtime,
            )
        else:
            self._fusion_counts["qkv_unfused"] += 1
            dflash_dense_bf16_to_f32(self.norm.ptr, self.weights.tensor(f"{prefix}.self_attn.q_proj.weight").ptr, self.q_raw.ptr, query_rows, self.hidden, self.attn_features, threads=128, stream=stream, library=self.library, runtime=self.runtime)
            dflash_dense_bf16_to_f32(self.norm.ptr, self.weights.tensor(f"{prefix}.self_attn.k_proj.weight").ptr, self.k_q.ptr, query_rows, self.hidden, self.kv_features, threads=128, stream=stream, library=self.library, runtime=self.runtime)
            dflash_dense_bf16_to_bf16(self.norm.ptr, self.weights.tensor(f"{prefix}.self_attn.v_proj.weight").ptr, self.v_q.ptr, query_rows, self.hidden, self.kv_features, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        # Q-rotary + K_q-rotary on the query rows only.  Cached K_ctx_rotated is
        # concatenated below; no context-side rotary is recomputed.
        dflash_head_rmsnorm_rotary_f32(
            self.q_raw.ptr,
            self.k_q.ptr,
            self.weights.tensor(f"{prefix}.self_attn.q_norm.weight").ptr,
            self.weights.tensor(f"{prefix}.self_attn.k_norm.weight").ptr,
            self.cos.ptr,
            self.sin.ptr,
            self.query_positions.ptr,
            self.query_positions.ptr,
            self.q_rot.ptr,
            self.k_q_rot.ptr,
            1,
            query_rows,
            query_rows,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.head_dim,
            self.cos.shape[0],
            threads=128,
            stream=stream,
            library=self.library,
            runtime=self.runtime,
        )
        dflash_concat_rows_f32(
            k_layer_base,
            self.k_q_rot.ptr,
            self.k_rot.ptr,
            1,
            kv_context_len,
            query_rows,
            self.kv_features,
            threads=128,
            stream=stream,
            library=self.library,
            runtime=self.runtime,
        )
        dflash_concat_rows_bf16(
            v_layer_base,
            self.v_q.ptr,
            self.v_all.ptr,
            1,
            kv_context_len,
            query_rows,
            self.kv_features,
            threads=128,
            stream=stream,
            library=self.library,
            runtime=self.runtime,
        )
        if bucket_ctx is not None:
            dflash_gqa_attention_f32_bf16_bucketed(
                self.q_rot.ptr,
                self.k_rot.ptr,
                self.v_all.ptr,
                self.attn.ptr,
                self.live_context_len.ptr,
                1,
                query_rows,
                total_kv,
                kv_context_len,
                self.q_heads,
                self.kv_heads,
                self.head_dim,
                threads=128,
                stream=stream,
                library=self.library,
                runtime=self.runtime,
            )
        else:
            dflash_gqa_attention_f32_bf16(
                self.q_rot.ptr,
                self.k_rot.ptr,
                self.v_all.ptr,
                self.attn.ptr,
                1,
                query_rows,
                total_kv,
                self.q_heads,
                self.kv_heads,
                self.head_dim,
                threads=128,
                stream=stream,
                library=self.library,
                runtime=self.runtime,
            )
        dflash_dense_bf16_to_bf16(self.attn.ptr, self.weights.tensor(f"{prefix}.self_attn.o_proj.weight").ptr, self.attn_proj.ptr, query_rows, self.attn_features, self.hidden, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        if _drafter_dense_use_add_rmsnorm():
            # R3.6 C1: fused add+rmsnorm; numerically equivalent to the unfused
            # path because the residual sum is rounded to BF16 before the RMS
            # reduction reads it.  Writes both ``hidden_attn`` (residual sum,
            # used by the MLP residual path below) and ``post`` (normalized
            # input to gate/up).
            dflash_add_rmsnorm_bf16(
                query_in.ptr,
                self.attn_proj.ptr,
                self.weights.tensor(f"{prefix}.post_attention_layernorm.weight").ptr,
                self.hidden_attn.ptr,
                self.post.ptr,
                query_rows,
                self.hidden,
                threads=256,
                stream=stream,
                library=self.library,
                runtime=self.runtime,
            )
        else:
            dflash_add_bf16(query_in.ptr, self.attn_proj.ptr, self.hidden_attn.ptr, query_rows * self.hidden, threads=256, stream=stream, library=self.library, runtime=self.runtime)
            dflash_rmsnorm_bf16(self.hidden_attn.ptr, self.weights.tensor(f"{prefix}.post_attention_layernorm.weight").ptr, self.post.ptr, query_rows, self.hidden, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.post.ptr, self.weights.tensor(f"{prefix}.mlp.gate_proj.weight").ptr, self.gate.ptr, query_rows, self.hidden, self.intermediate, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.post.ptr, self.weights.tensor(f"{prefix}.mlp.up_proj.weight").ptr, self.up.ptr, query_rows, self.hidden, self.intermediate, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        dflash_silu_mul_bf16(self.gate.ptr, self.up.ptr, self.act.ptr, query_rows * self.intermediate, threads=256, stream=stream, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.act.ptr, self.weights.tensor(f"{prefix}.mlp.down_proj.weight").ptr, self.mlp.ptr, query_rows, self.intermediate, self.hidden, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        dflash_add_bf16(self.hidden_attn.ptr, self.mlp.ptr, query_out.ptr, query_rows * self.hidden, threads=256, stream=stream, library=self.library, runtime=self.runtime)
        return query_out


    def _empty(self, shape: tuple[int, ...], dtype: DType) -> Tensor:
        nbytes = int(math.prod(shape)) * dtype.itemsize
        buf = malloc(nbytes, runtime=self.runtime)
        self.buffers.append(buf)
        return Tensor.from_handle(buf.ptr, shape, dtype, self.device)

    def _load_array(self, array: np.ndarray, dtype: DType) -> Tensor:
        tensor = self._empty(tuple(int(x) for x in array.shape), dtype)
        copy_host_to_device(self._buffer_for(tensor), host_array_ptr(np.ascontiguousarray(array)), runtime=self.runtime)
        return tensor

    def _buffer_for(self, tensor: Tensor) -> DeviceBuffer:
        for buffer in self.buffers:
            if buffer.ptr == tensor.ptr:
                return buffer
        raise KeyError(f"no owning buffer for tensor pointer 0x{tensor.ptr:x}")


def run_ar_tokens(
    *,
    model: Path,
    prompt_ids: Sequence[int],
    decode_tokens: int,
    backend: str,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    prefill_config: PrefillConfig,
) -> tuple[list[int], dict[str, Any]]:
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_ids) + decode_tokens + 1
    reset_memory_stats()
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_layers=max_layers,
        max_batch_size=1,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=prefill_config,
    ) as session:
        t0 = time.perf_counter()
        next_result = None
        for pos, token in enumerate(prompt_ids):
            next_result = session.step(int(token), position=pos, sample=(pos == len(prompt_ids) - 1))
        if next_result is None:
            raise RuntimeError("AR prefill produced no token")
        prefill_seconds = time.perf_counter() - t0
        generated: list[int] = []
        next_token = int(next_result.token_id)
        t1 = time.perf_counter()
        finite = True
        for offset in range(decode_tokens):
            generated.append(next_token)
            result = session.step(next_token, position=len(prompt_ids) + offset, sample=True)
            if result is None:
                raise RuntimeError("AR decode step produced no token")
            finite = finite and math.isfinite(float(result.logit))
            next_token = int(result.token_id)
        decode_seconds = time.perf_counter() - t1
        memory = memory_stats()
        metadata = {
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "finite_logits": finite,
            "decode_tok_s": decode_tokens / decode_seconds if decode_seconds > 0 else None,
            "memory": memory,
            "backend": session.backend,
            "target_arch": session.target_arch,
        }
    return generated, metadata


def run_dflash_tokens(
    *,
    model: Path,
    drafter_model: Path,
    prompt_ids: Sequence[int],
    decode_tokens: int,
    candidate_budget: int,
    backend: str,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    prefill_config: PrefillConfig,
    drafter_graph_mode: str = "off",
    drafter_fusion_mode: str = "off",
    drafter_bucket_mode: str = "exact",
    drafter_query_mode: str = "block",
    adaptive_budget_mode: str = "off",
    adaptive_min_remaining_tokens: int = 0,
    adaptive_probe_amortization_tokens: int = 128,
    ar_decode_tok_s_estimate: float | None = None,
) -> tuple[list[int], dict[str, Any]]:
    if adaptive_budget_mode not in {"off", "on"}:
        raise ValueError("adaptive_budget_mode must be off or on")
    if adaptive_min_remaining_tokens < 0:
        raise ValueError("adaptive_min_remaining_tokens must be non-negative")
    if adaptive_probe_amortization_tokens < 0:
        raise ValueError("adaptive_probe_amortization_tokens must be non-negative")
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_ids) + decode_tokens + candidate_budget + 2
    max_batch_size = candidate_budget + 2
    reset_memory_stats()
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_layers=max_layers,
        max_batch_size=max_batch_size,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=prefill_config,
    ) as session:
        with NativeDFlashChainDrafter(
            session=session,
            drafter_model=drafter_model,
            max_context_tokens=max_sequence,
            candidate_budget=candidate_budget,
            compiler_version=compiler_version,
            require_cached_build=require_cached_build,
            graph_mode=drafter_graph_mode,
            fusion_mode=drafter_fusion_mode,
            bucket_mode=drafter_bucket_mode,
            query_mode=drafter_query_mode,
        ) as drafter:
            t0 = time.perf_counter()
            next_result = None
            for pos, token in enumerate(prompt_ids):
                next_result = session.step_with_hidden_taps(
                    int(token),
                    position=pos,
                    capture_layer_ids=drafter.target_layer_ids,
                    capture_hidden_concat=drafter.target_hidden_concat,
                    capture_row=pos,
                    sample=(pos == len(prompt_ids) - 1),
                )
            if next_result is None:
                raise RuntimeError("DFlash prefill produced no root token")
            prefill_seconds = time.perf_counter() - t0
            root_token = int(next_result.token_id)
            context_tokens = len(prompt_ids)
            # Pre-project the entire prefill context once; the per-cycle
            # propose() path will then only project the newly committed rows.
            drafter.warmup_context(context_tokens)
            adaptive_budget = AdaptiveBudgetController(
                enabled=adaptive_budget_mode == "on",
                ar_decode_tok_s_estimate=ar_decode_tok_s_estimate,
                config=AdaptiveBudgetConfig(
                    min_remaining_tokens_for_dflash=adaptive_min_remaining_tokens,
                    probe_min_amortization_tokens=adaptive_probe_amortization_tokens,
                ),
            )
            generated: list[int] = []
            accepted_lengths: list[int] = []
            draft_seconds_total = 0.0
            verify_seconds_total = 0.0
            commit_seconds_total = 0.0
            d2h_vector_reads = 0
            d2h_vector_values = 0
            cycles = 0
            verify_rows_total = 0
            finite = True
            t1 = time.perf_counter()
            while len(generated) < decode_tokens:
                cycles += 1
                remaining = decode_tokens - len(generated)
                active_budget = min(candidate_budget, max(0, remaining - 1))
                decision = adaptive_budget.begin_cycle(
                    cycle=cycles,
                    context_tokens=context_tokens,
                    remaining_tokens=remaining,
                    active_budget=active_budget,
                )
                if active_budget <= 0 or not decision.use_dflash:
                    # No spec budget left (or adaptive budget locked DFlash out):
                    # one bare AR step on slot 0.
                    terminal_ar_bypass = active_budget <= 0 or decision.reason in _ADAPTIVE_AR_GUARD_REASONS
                    verify_rows_total += 1
                    cycle_context_tokens = context_tokens
                    t_cycle = time.perf_counter()
                    t_verify = t_cycle
                    result = _slot_step(
                        session,
                        root_token,
                        position=context_tokens,
                        slot=0,
                        drafter=None if terminal_ar_bypass else drafter,
                        capture_row=None if terminal_ar_bypass else context_tokens,
                    )
                    verify_seconds_total += time.perf_counter() - t_verify
                    if not terminal_ar_bypass:
                        t_commit = time.perf_counter()
                        drafter.commit_context_rows(start=context_tokens, count=1)
                        commit_seconds_total += time.perf_counter() - t_commit
                    generated.append(root_token)
                    root_token = int(result.token_id)
                    context_tokens += 1
                    adaptive_budget.record_ar_cycle(
                        decision if active_budget > 0 else None,
                        cycle=cycles,
                        cycle_wall_ms=(time.perf_counter() - t_cycle) * 1000.0,
                        context_tokens=cycle_context_tokens,
                        forced_reason="no_spec_budget" if active_budget <= 0 else decision.reason,
                        update_state=active_budget > 0 and decision.reason not in _ADAPTIVE_AR_GUARD_REASONS,
                    )
                    continue
                verify_rows_total += 1 + active_budget
                cycle_context_tokens = context_tokens
                t_cycle = time.perf_counter()
                draft = drafter.propose(root_token=root_token, root_position=context_tokens, context_tokens=context_tokens)
                candidates = list(draft.candidate_tokens[:active_budget])
                draft_seconds_total += draft.draft_seconds
                d2h_vector_reads += draft.d2h_vector_reads
                d2h_vector_values += draft.d2h_vector_values
                t_verify = time.perf_counter()
                # In-place verify on slot 0; the loop never steps into a rejected
                # candidate (the compare is BEFORE the step), so no roll-back path
                # is needed and per-candidate state copies are not necessary.
                parent_result = _slot_step(
                    session,
                    root_token,
                    position=context_tokens,
                    slot=0,
                    drafter=drafter,
                    capture_row=context_tokens,
                )
                target_top1 = [int(parent_result.token_id)]
                accepted = 0
                bonus = int(parent_result.token_id)
                finite = finite and math.isfinite(float(parent_result.logit))
                for idx, cand in enumerate(candidates):
                    if target_top1[-1] != int(cand):
                        bonus = target_top1[-1]
                        break
                    accepted += 1
                    result = _slot_step(
                        session,
                        int(cand),
                        position=context_tokens + idx + 1,
                        slot=0,
                        drafter=drafter,
                        capture_row=context_tokens + idx + 1,
                    )
                    finite = finite and math.isfinite(float(result.logit))
                    target_top1.append(int(result.token_id))
                    bonus = int(result.token_id)
                verify_elapsed = time.perf_counter() - t_verify
                verify_seconds_total += verify_elapsed
                accepted_lengths.append(accepted)
                committed = [root_token, *candidates[:accepted]]
                t_commit = time.perf_counter()
                drafter.commit_context_rows(start=context_tokens, count=len(committed))
                commit_elapsed = time.perf_counter() - t_commit
                commit_seconds_total += commit_elapsed
                adaptive_budget.record_dflash_cycle(
                    decision,
                    visible_tokens=len(committed),
                    cycle_wall_ms=(time.perf_counter() - t_cycle) * 1000.0,
                    accepted_tokens=accepted,
                    active_budget=active_budget,
                    draft_ms=draft.draft_seconds * 1000.0,
                    verify_ms=verify_elapsed * 1000.0,
                    commit_ms=commit_elapsed * 1000.0,
                    context_tokens=cycle_context_tokens,
                )
                generated.extend(committed)
                root_token = int(bonus)
                context_tokens += len(committed)
            decode_seconds = time.perf_counter() - t1
            memory = memory_stats()
            metadata = {
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "draft_seconds": draft_seconds_total,
                "target_verify_seconds": verify_seconds_total,
                "commit_seconds": commit_seconds_total,
                "accepted_lengths": accepted_lengths,
                "target_verify_rows": verify_rows_total,
                "draft_calls": cycles,
                "finite_draft_logits": finite,
                "finite_verify_logits": finite,
                "decode_tok_s": decode_tokens / decode_seconds if decode_seconds > 0 else None,
                "d2h": {"scalar_reads": cycles, "vector_reads": d2h_vector_reads, "scalar_values": cycles, "vector_values": d2h_vector_values, "full_logits_readbacks": 0},
                "memory": memory,
                "backend": session.backend,
                "target_arch": session.target_arch,
                "verifier_mode": "serial_in_place_single_slot",
                "native_bulk_verifier": False,
                "drafter_context_mode": "append_only_projected_context_and_kv",
                "adaptive_budget": adaptive_budget.summary(),
            }
    return generated[:decode_tokens], metadata


def _run_profile_ar_on_session(
    *,
    session: Qwen35ParoResidentSession,
    prompt_ids: Sequence[int],
    decode_tokens: int,
    base_slot: int,
) -> tuple[list[int], dict[str, Any]]:
    """Run the routed row as plain AR on a non-control slot.

    This is benchmark plumbing for profile/oracle route diagnostics.  It keeps
    the same-session AR control intact on slot 0 while measuring the cost of a
    policy choosing "do not speculate" for this prompt.
    """

    t0 = time.perf_counter()
    next_result = None
    for pos, token in enumerate(prompt_ids):
        next_result = _slot_step(
            session,
            int(token),
            position=pos,
            slot=base_slot,
            drafter=None,
            capture_row=None,
            sample=(pos == len(prompt_ids) - 1),
        )
    if next_result is None:
        raise RuntimeError("profile-routed AR prefill produced no token")
    prefill_seconds = time.perf_counter() - t0

    generated: list[int] = []
    next_token = int(next_result.token_id)
    finite = True
    t1 = time.perf_counter()
    for offset in range(decode_tokens):
        generated.append(next_token)
        result = _slot_step(
            session,
            next_token,
            position=len(prompt_ids) + offset,
            slot=base_slot,
            drafter=None,
            capture_row=None,
            sample=True,
        )
        finite = finite and math.isfinite(float(result.logit))
        next_token = int(result.token_id)
    decode_seconds = time.perf_counter() - t1
    metadata = {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "draft_seconds": 0.0,
        "target_verify_seconds": decode_seconds,
        "commit_seconds": 0.0,
        "accepted_lengths": [],
        "target_verify_rows": decode_tokens,
        "target_forward_calls": decode_tokens,
        "target_bulk_forward_calls": 0,
        "target_serial_forward_calls": decode_tokens,
        "target_bulk_rows": 0,
        "canonical_commit_replay_rows": 0,
        "target_forwards_per_draft_call": None,
        "gpu_accept_match_cpu": True,
        "verifier_graph": {"mode": "off", "status_counts": {}, "validation_passed": None, "last": None},
        "draft_calls": 0,
        "decode_cycles": decode_tokens,
        "draft_tokens_proposed": 0,
        "tree_active_nodes_total": 0,
        "tree_top_k": 0,
        "draft_top_k": 0,
        "draft_p_min": 0.0,
        "drafter_query_mode": None,
        "drafter_query_rows": 0,
        "drafter_block_size": 0,
        "confidence_limited_cycles": 0,
        "tree_compiler": None,
        "draft_native_phase_seconds": {},
        "draft_graph": {"mode": "off", "status_counts": {}, "validation_passed": None, "last": None},
        "draft_fusion": None,
        "proposal_trace_sample": [],
        "proposal_trace_count": 0,
        "finite_draft_logits": True,
        "finite_verify_logits": finite,
        "decode_tok_s": decode_tokens / decode_seconds if decode_seconds > 0 else None,
        "d2h": {
            "scalar_reads": decode_tokens,
            "vector_reads": 0,
            "scalar_values": decode_tokens,
            "vector_values": 0,
            "full_logits_readbacks": 0,
            "notes": ["profile route selected plain AR; no draft or native-bulk verifier ran"],
        },
        "memory": memory_stats(),
        "backend": session.backend,
        "target_arch": session.target_arch,
        "verifier_mode": "profile_route_ar",
        "verifier_graph_mode": "off",
        "verifier_chain_attn_mode": None,
        "verifier_tree_mode": "ar",
        "verifier_state_strategy": "profile_route_ar",
        "canonical_commit_mode": None,
        "native_bulk_verifier": False,
        "drafter_context_mode": "none",
        "adaptive_budget": {
            "mode": "profile_route",
            "enabled": False,
            "state": "AR_LOCKED",
            "config": {},
            "mode_counts": {"ar": decode_tokens},
            "decision_counts": {"ar": decode_tokens},
            "transitions": [],
            "cycle_log": [],
            "cycle_log_truncated": False,
            "profit_ms_mean": None,
            "profit_ms_min": None,
            "profit_ms_max": None,
        },
        "draft_phase_timing_mode": "none",
        "base_slot": base_slot,
        "branch_slot_start": None,
        "verifier_state_copies_per_cycle": 0.0,
        "verifier_state_copies_total": 0,
        "profile_route": "ar",
    }
    return generated[:decode_tokens], metadata


def run_same_session_pair(
    *,
    model: Path,
    drafter_model: Path,
    prompt_ids: Sequence[int],
    decode_tokens: int,
    candidate_budget: int,
    backend: str,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    prefill_config: PrefillConfig,
    sync_draft_phases: bool = False,
    verifier_mode: str = "native_bulk_bplus1",
    verifier_graph_mode: str = "off",
    drafter_graph_mode: str = "off",
    drafter_fusion_mode: str = "off",
    drafter_bucket_mode: str = "exact",
    drafter_query_mode: str = "block",
    draft_top_k: int = 1,
    draft_p_min: float = 0.0,
    whole_cycle_gate: float = 0.0,
    adaptive_budget_mode: str = "off",
    adaptive_min_remaining_tokens: int = 0,
    adaptive_probe_amortization_tokens: int = 128,
    terminal_ar_tokens: int = 0,
    chain_attn_mode: str = "c1_loop",
    tree_mode: str = "chain",
    tree_top_k: int = 1,
    canonical_commit_mode: str = "replay",
    profile_route: str = "spec",
    roctx: _Roctx | None = None,
    rocprof_selected_region: str = "none",
) -> tuple[tuple[list[int], dict[str, Any]], tuple[list[int], dict[str, Any]]]:
    """Run AR control and DFlash chain in one resident target session.

    Slot 0 is reserved for the AR control.  Slot 1 is the DFlash committed
    state.  ``native_bulk_bplus1`` advances it through one root+B target forward
    per cycle; ``serial_in_place_single_slot`` remains a fallback.  The target
    weights/libraries/session are identical while the per-slot recurrent/KV
    state remains independent for exact token comparison.
    """

    if adaptive_budget_mode not in {"off", "on"}:
        raise ValueError("adaptive_budget_mode must be off or on")
    if adaptive_min_remaining_tokens < 0:
        raise ValueError("adaptive_min_remaining_tokens must be non-negative")
    if adaptive_probe_amortization_tokens < 0:
        raise ValueError("adaptive_probe_amortization_tokens must be non-negative")
    if terminal_ar_tokens < 0:
        raise ValueError("terminal_ar_tokens must be non-negative")
    if canonical_commit_mode not in {"replay", "bulk_direct", "branch_copy"}:
        raise ValueError("canonical_commit_mode must be replay, bulk_direct, or branch_copy")
    if canonical_commit_mode != "replay" and (verifier_mode != "native_bulk_bplus1" or tree_mode != "chain"):
        raise ValueError("non-replay canonical_commit_mode requires native_bulk_bplus1 chain mode")
    profile_route = _canonical_profile_route(profile_route)
    if draft_top_k <= 0 or draft_top_k > 8:
        raise ValueError("draft_top_k must be in [1, 8]")
    if draft_p_min < 0.0 or draft_p_min > 1.0:
        raise ValueError("draft_p_min must be in [0, 1]")
    if draft_p_min > 0.0:
        if tree_mode != "chain":
            raise ValueError("draft_p_min is currently supported only for chain mode")
        if draft_top_k < 2:
            raise ValueError("draft_p_min requires draft_top_k >= 2")
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_ids) + decode_tokens + candidate_budget + 2
    max_batch_size = max(4, candidate_budget + 3)
    reset_memory_stats()
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence,
        max_layers=max_layers,
        max_batch_size=max_batch_size,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=prefill_config,
    ) as session:
        t0 = time.perf_counter()
        next_result = None
        for pos, token in enumerate(prompt_ids):
            next_result = session.step(int(token), position=pos, sample=(pos == len(prompt_ids) - 1))
        if next_result is None:
            raise RuntimeError("same-session AR prefill produced no token")
        ar_prefill_seconds = time.perf_counter() - t0
        ar_generated: list[int] = []
        next_token = int(next_result.token_id)
        ar_finite = True
        t1 = time.perf_counter()
        for offset in range(decode_tokens):
            ar_generated.append(next_token)
            result = session.step(next_token, position=len(prompt_ids) + offset, sample=True)
            if result is None:
                raise RuntimeError("same-session AR decode step produced no token")
            ar_finite = ar_finite and math.isfinite(float(result.logit))
            next_token = int(result.token_id)
        ar_decode_seconds = time.perf_counter() - t1
        ar_meta = {
            "prefill_seconds": ar_prefill_seconds,
            "decode_seconds": ar_decode_seconds,
            "finite_logits": ar_finite,
            "decode_tok_s": decode_tokens / ar_decode_seconds if ar_decode_seconds > 0 else None,
            "memory": memory_stats(),
            "backend": session.backend,
            "target_arch": session.target_arch,
            "same_session_control": True,
            "same_process_control": True,
            "control_slot": 0,
        }
        if profile_route == "ar":
            spec_tokens, spec_meta = _run_profile_ar_on_session(
                session=session,
                prompt_ids=prompt_ids,
                decode_tokens=decode_tokens,
                base_slot=1,
            )
            spec_meta["same_session_control"] = True
            spec_meta["same_process_control"] = True
            return (ar_generated, ar_meta), (spec_tokens, spec_meta)
        routed_tree_mode = "branching_topk" if profile_route == "tree" else ("chain" if profile_route == "chain" else tree_mode)
        routed_canonical_commit_mode = "replay" if routed_tree_mode != "chain" else canonical_commit_mode
        with NativeDFlashChainDrafter(
            session=session,
            drafter_model=drafter_model,
            max_context_tokens=max_sequence,
            candidate_budget=candidate_budget,
            compiler_version=compiler_version,
            require_cached_build=require_cached_build,
            sync_draft_phases=sync_draft_phases,
            graph_mode=drafter_graph_mode,
            fusion_mode=drafter_fusion_mode,
            bucket_mode=drafter_bucket_mode,
            query_mode=drafter_query_mode,
            draft_top_k=tree_top_k if routed_tree_mode == "branching_topk" else draft_top_k,
        ) as drafter:
            spec_tokens, spec_meta = _run_dflash_chain_on_session(
                session=session,
                drafter=drafter,
                prompt_ids=prompt_ids,
                decode_tokens=decode_tokens,
                candidate_budget=candidate_budget,
                base_slot=1,
                branch_slot_start=2,
                verifier_mode=verifier_mode,
                verifier_graph_mode=verifier_graph_mode,
                adaptive_budget_mode=adaptive_budget_mode,
                adaptive_min_remaining_tokens=adaptive_min_remaining_tokens,
                adaptive_probe_amortization_tokens=adaptive_probe_amortization_tokens,
                terminal_ar_tokens=terminal_ar_tokens,
                ar_decode_tok_s_estimate=ar_meta["decode_tok_s"],
                chain_attn_mode=chain_attn_mode,
                tree_mode=routed_tree_mode,
                tree_top_k=tree_top_k,
                draft_p_min=draft_p_min,
                whole_cycle_gate=whole_cycle_gate,
                canonical_commit_mode=routed_canonical_commit_mode,
                roctx=roctx,
                rocprof_selected_region=rocprof_selected_region,
            )
        spec_meta["same_session_control"] = True
        spec_meta["same_process_control"] = True
        if profile_route != "spec":
            spec_meta["profile_route"] = routed_tree_mode
        return (ar_generated, ar_meta), (spec_tokens, spec_meta)


def _run_dflash_chain_on_session(
    *,
    session: Qwen35ParoResidentSession,
    drafter: NativeDFlashChainDrafter,
    prompt_ids: Sequence[int],
    decode_tokens: int,
    candidate_budget: int,
    base_slot: int,
    branch_slot_start: int,
    verifier_mode: str = "native_bulk_bplus1",
    verifier_graph_mode: str = "off",
    adaptive_budget_mode: str = "off",
    adaptive_min_remaining_tokens: int = 0,
    adaptive_probe_amortization_tokens: int = 128,
    terminal_ar_tokens: int = 0,
    ar_decode_tok_s_estimate: float | None = None,
    chain_attn_mode: str = "c1_loop",
    tree_mode: str = "chain",
    tree_top_k: int = 1,
    draft_p_min: float = 0.0,
    whole_cycle_gate: float = 0.0,
    canonical_commit_mode: str = "replay",
    roctx: _Roctx | None = None,
    rocprof_selected_region: str = "none",
) -> tuple[list[int], dict[str, Any]]:
    if adaptive_budget_mode not in {"off", "on"}:
        raise ValueError("adaptive_budget_mode must be off or on")
    if adaptive_min_remaining_tokens < 0:
        raise ValueError("adaptive_min_remaining_tokens must be non-negative")
    if adaptive_probe_amortization_tokens < 0:
        raise ValueError("adaptive_probe_amortization_tokens must be non-negative")
    if terminal_ar_tokens < 0:
        raise ValueError("terminal_ar_tokens must be non-negative")
    if canonical_commit_mode not in {"replay", "bulk_direct", "branch_copy"}:
        raise ValueError("canonical_commit_mode must be replay, bulk_direct, or branch_copy")
    if canonical_commit_mode != "replay" and (verifier_mode != "native_bulk_bplus1" or tree_mode != "chain"):
        raise ValueError("non-replay canonical_commit_mode requires native_bulk_bplus1 chain mode")
    if draft_p_min < 0.0 or draft_p_min > 1.0:
        raise ValueError("draft_p_min must be in [0, 1]")
    if draft_p_min > 0.0 and tree_mode != "chain":
        raise ValueError("draft_p_min is currently supported only for chain mode")
    if whole_cycle_gate < 0.0 or whole_cycle_gate > 1.0:
        raise ValueError("whole_cycle_gate must be in [0, 1]")
    whole_cycle_gate_threshold = float(whole_cycle_gate)
    _wc_env = os.environ.get("HIPENGINE_DFLASH_WHOLE_CYCLE_GATE")
    if whole_cycle_gate_threshold <= 0.0 and _wc_env is not None and _wc_env.strip():
        # Backward-compat: env var still activates the gate for artifact reproduction.
        whole_cycle_gate_threshold = float(_wc_env)
    if whole_cycle_gate_threshold > 0.0 and tree_mode != "chain":
        raise ValueError("whole_cycle_gate is currently supported only for chain mode")
    t0 = time.perf_counter()
    next_result = None
    for pos, token in enumerate(prompt_ids):
        next_result = _slot_step(
            session,
            int(token),
            position=pos,
            slot=base_slot,
            drafter=drafter,
            capture_row=pos,
            sample=(pos == len(prompt_ids) - 1),
        )
    if next_result is None:
        raise RuntimeError("DFlash prefill produced no root token")
    prefill_seconds = time.perf_counter() - t0
    root_token = int(next_result.token_id)
    context_tokens = len(prompt_ids)
    # Pre-project the entire prefill context once; per-cycle propose() then
    # only projects the newly committed tail.
    drafter.warmup_context(context_tokens)
    adaptive_budget = AdaptiveBudgetController(
        enabled=adaptive_budget_mode == "on",
        ar_decode_tok_s_estimate=ar_decode_tok_s_estimate,
        config=AdaptiveBudgetConfig(
            min_remaining_tokens_for_dflash=adaptive_min_remaining_tokens,
            probe_min_amortization_tokens=adaptive_probe_amortization_tokens,
        ),
    )
    generated: list[int] = []
    accepted_lengths: list[int] = []
    confidence_trace: list[dict[str, Any]] = []
    draft_seconds_total = 0.0
    verify_seconds_total = 0.0
    commit_seconds_total = 0.0
    d2h_vector_reads = 0
    d2h_vector_values = 0
    cycles = 0
    draft_calls = 0
    draft_tokens_proposed = 0
    verify_rows_total = 0
    draft_phase_seconds: dict[str, float] = {}
    draft_graph_status_counts: Counter[str] = Counter()
    draft_graph_validation_seen = False
    draft_graph_validation_passed = True
    proposal_trace: list[dict[str, Any]] = []
    finite_draft = True
    finite_verify = True
    gpu_accept_match_cpu = True
    target_bulk_forward_calls = 0
    target_serial_forward_calls = 0
    target_bulk_rows_total = 0
    verifier_graph_status_counts: Counter[str] = Counter()
    verifier_graph_last: dict[str, Any] | None = None
    verifier_graph_validation_seen = False
    verifier_graph_validation_passed = True
    target_accept_scalar_reads = 0
    target_accept_scalar_values = 0
    confidence_limited_cycles = 0
    terminal_ar_cycles = 0
    canonical_commit_replay_rows = 0
    t1 = time.perf_counter()
    state_copies = 0
    tree_active_nodes_total = 0
    while len(generated) < decode_tokens:
        cycles += 1
        remaining = decode_tokens - len(generated)
        max_accept_depth = max(0, remaining - 1)
        active_budget = (
            candidate_budget
            if tree_mode == "branching_topk" and max_accept_depth > 0
            else min(candidate_budget, max_accept_depth)
        )
        decision = adaptive_budget.begin_cycle(
            cycle=cycles,
            context_tokens=context_tokens,
            remaining_tokens=remaining,
            active_budget=active_budget,
        )
        terminal_ar_guard = (
            active_budget > 0
            and int(terminal_ar_tokens) > 0
            and remaining < int(terminal_ar_tokens)
        )
        if active_budget <= 0 or terminal_ar_guard or not decision.use_dflash:
            if terminal_ar_guard:
                terminal_ar_cycles += 1
            terminal_ar_bypass = (
                active_budget <= 0
                or terminal_ar_guard
                or decision.reason in _ADAPTIVE_AR_GUARD_REASONS
            )
            cycle_context_tokens = context_tokens
            t_cycle = time.perf_counter()
            verify_rows_total += 1
            t_verify = time.perf_counter()
            result = _slot_step(
                session,
                root_token,
                position=context_tokens,
                slot=base_slot,
                drafter=None if terminal_ar_bypass else drafter,
                capture_row=None if terminal_ar_bypass else context_tokens,
            )
            verify_seconds_total += time.perf_counter() - t_verify
            target_serial_forward_calls += 1
            finite_verify = finite_verify and math.isfinite(float(result.logit))
            bonus = int(result.token_id)
            if not terminal_ar_bypass:
                t_commit = time.perf_counter()
                drafter.commit_context_rows(start=context_tokens, count=1)
                commit_seconds_total += time.perf_counter() - t_commit
            generated.append(root_token)
            root_token = int(bonus)
            context_tokens += 1
            adaptive_budget.record_ar_cycle(
                decision if active_budget > 0 else None,
                cycle=cycles,
                cycle_wall_ms=(time.perf_counter() - t_cycle) * 1000.0,
                context_tokens=cycle_context_tokens,
                forced_reason=(
                    "no_spec_budget"
                    if active_budget <= 0
                    else ("terminal_ar_guard" if terminal_ar_guard else decision.reason)
                ),
                update_state=(
                    active_budget > 0
                    and not terminal_ar_guard
                    and decision.reason not in _ADAPTIVE_AR_GUARD_REASONS
                ),
            )
            continue
        cycle_context_tokens = context_tokens
        t_cycle = time.perf_counter()
        draft = drafter.propose(root_token=root_token, root_position=context_tokens, context_tokens=context_tokens)
        compiled_tree: _CompiledTopKTree | None = None
        requested_active_budget = int(active_budget)
        confidence_limited = False
        if tree_mode == "branching_topk":
            compiled_tree = _build_branching_topk_tree_target_batch(
                root_token=root_token,
                root_position=context_tokens,
                topk_tokens=draft.topk_tokens,
                topk_values=draft.topk_values,
                candidate_budget=candidate_budget,
                tree_top_k=tree_top_k,
                max_depth=min(max_accept_depth, candidate_budget),
            )
            candidates = list(compiled_tree.active_candidate_tokens)
            draft_nodes_this_cycle = int(compiled_tree.active_count)
            tree_active_nodes_total += draft_nodes_this_cycle
        else:
            if whole_cycle_gate_threshold > 0.0:
                # Whole-cycle confidence gate (deployable): keep the FULL chain
                # when the drafter's depth-1 confidence is high, else drop to AR
                # (active_budget=0 -> verify root only). No mid-chain truncation.
                _p0 = float(draft.top1_probabilities[0]) if draft.top1_probabilities else 1.0
                if _p0 < whole_cycle_gate_threshold:
                    active_budget = 0
                    confidence_limited = True
                    confidence_limited_cycles += 1
            elif draft_p_min > 0.0:
                active_budget = _confidence_limited_active_count(
                    draft.top1_probabilities,
                    max_active=active_budget,
                    p_min=draft_p_min,
                )
                confidence_limited = active_budget != requested_active_budget
                if confidence_limited:
                    confidence_limited_cycles += 1
            candidates = list(draft.candidate_tokens[:active_budget])
            draft_nodes_this_cycle = active_budget
        draft_calls += 1
        draft_tokens_proposed += draft_nodes_this_cycle
        draft_seconds_total += draft.draft_seconds
        for phase_name, phase_seconds in draft.phase_seconds.items():
            value = float(phase_seconds)
            if phase_name == "slowest_decoder_layer":
                draft_phase_seconds[phase_name] = max(draft_phase_seconds.get(phase_name, 0.0), value)
            else:
                draft_phase_seconds[phase_name] = draft_phase_seconds.get(phase_name, 0.0) + value
        graph_status = str(draft.graph.get("status", "unknown"))
        draft_graph_status_counts[graph_status] += 1
        validation = draft.graph.get("validation_passed")
        if validation is not None:
            draft_graph_validation_seen = True
        if validation is False:
            draft_graph_validation_passed = False
        finite_draft = finite_draft and draft.finite_logits
        d2h_vector_reads += draft.d2h_vector_reads
        d2h_vector_values += draft.d2h_vector_values
        if roctx is not None:
            if rocprof_selected_region == "dflash_verify":
                roctx.profiler_resume()
            roctx.push(f"dflash_verify_pass_{draft_calls}")
        t_verify = time.perf_counter()
        verify_result = None
        target_batch = None
        if active_budget <= 0:
            result = _slot_step(
                session,
                root_token,
                position=context_tokens,
                slot=base_slot,
                drafter=drafter,
                capture_row=context_tokens,
            )
            target_serial_forward_calls += 1
            verify_rows_total += 1
            target_top1 = [int(result.token_id)]
            accepted = 0
            accepted_tokens = []
            bonus = int(result.token_id)
            finite_verify = finite_verify and math.isfinite(float(result.logit))
            verify_elapsed = time.perf_counter() - t_verify
            verify_seconds_total += verify_elapsed
        elif verifier_mode == "native_bulk_bplus1":
            verifier_slot = base_slot
            use_branch_slot = tree_mode == "chain" and canonical_commit_mode in {"replay", "branch_copy"}
            if use_branch_slot:
                verifier_slot = branch_slot_start
                session.copy_slot_state(base_slot, verifier_slot, kv_rows=context_tokens)
                state_copies += 1
            if tree_mode == "branching_topk":
                if compiled_tree is None:
                    raise RuntimeError("branching_topk tree was not compiled")
                target_batch = compiled_tree.target_batch
                verify_result = session.verify_tree_bulk_and_commit(
                    target_batch,
                    base_slot=verifier_slot,
                    capture_layer_ids=drafter.config.target_layer_ids,
                    capture_hidden_concat=drafter.target_hidden_concat,
                    capture_row_start=context_tokens,
                )
            elif tree_mode == "chain_as_tree":
                target_batch = _build_chain_as_tree_target_batch(
                    root_token=root_token,
                    root_position=context_tokens,
                    candidates=candidates,
                    candidate_budget=candidate_budget,
                    active_count=active_budget,
                )
                verify_result = session.verify_tree_bulk_and_commit(
                    target_batch,
                    base_slot=verifier_slot,
                    capture_layer_ids=drafter.config.target_layer_ids,
                    capture_hidden_concat=drafter.target_hidden_concat,
                    capture_row_start=context_tokens,
                )
            else:
                chain_candidate_budget = active_budget if draft_p_min > 0.0 else candidate_budget
                target_batch = _build_chain_target_batch(
                    root_token=root_token,
                    root_position=context_tokens,
                    candidates=candidates,
                    candidate_budget=chain_candidate_budget,
                    active_count=active_budget,
                )
                verify_result = session.verify_chain_bulk_and_commit(
                    target_batch,
                    base_slot=verifier_slot,
                    capture_layer_ids=drafter.config.target_layer_ids,
                    capture_hidden_concat=drafter.target_hidden_concat,
                    capture_row_start=context_tokens,
                    graph_mode=verifier_graph_mode,
                    chain_attn_mode=chain_attn_mode,
                )
            verify_rows_total += int(verify_result.rows)
            target_top1_count = int(verify_result.rows) if tree_mode == "branching_topk" else 1 + active_budget
            target_top1 = list(verify_result.target_top1[:target_top1_count])
            accepted = int(verify_result.accepted_count)
            bonus = int(verify_result.next_token) if verify_result.next_token is not None else int(target_top1[-1])
            finite_verify = finite_verify and bool(verify_result.finite_logits)
            gpu_accept_match_cpu = gpu_accept_match_cpu and bool(verify_result.gpu_accept_match_cpu)
            if verify_result.graph:
                verifier_graph_last = verify_result.graph
                graph_status = str(verify_result.graph.get("status", "unknown"))
                verifier_graph_status_counts[graph_status] += 1
                validation = verify_result.graph.get("validation_passed")
                if validation is not None:
                    verifier_graph_validation_seen = True
                if validation is False:
                    verifier_graph_validation_passed = False
            target_bulk_forward_calls += int(verify_result.target_forward_calls)
            target_bulk_rows_total += int(verify_result.rows)
            target_accept_scalar_reads += 7
            target_accept_scalar_values += 7
        elif verifier_mode == "serial_in_place_single_slot":
            # In-place verify on base_slot: every forward advances state to the
            # committed prefix.  No per-candidate state copies because the loop never
            # steps into a rejected candidate (compare is BEFORE the step).
            serial_forwards_this_cycle = 1
            parent_result = _slot_step(
                session,
                root_token,
                position=context_tokens,
                slot=base_slot,
                drafter=drafter,
                capture_row=context_tokens,
            )
            target_serial_forward_calls += 1
            target_top1 = [int(parent_result.token_id)]
            accepted = 0
            bonus = int(parent_result.token_id)
            finite_verify = finite_verify and math.isfinite(float(parent_result.logit))
            for idx, cand in enumerate(candidates):
                if target_top1[-1] != int(cand):
                    bonus = target_top1[-1]
                    break
                accepted += 1
                result = _slot_step(
                    session,
                    int(cand),
                    position=context_tokens + idx + 1,
                    slot=base_slot,
                    drafter=drafter,
                    capture_row=context_tokens + idx + 1,
                )
                target_serial_forward_calls += 1
                serial_forwards_this_cycle += 1
                finite_verify = finite_verify and math.isfinite(float(result.logit))
                target_top1.append(int(result.token_id))
                bonus = int(result.token_id)
            verify_rows_total += serial_forwards_this_cycle
        else:
            raise ValueError(f"unknown verifier_mode {verifier_mode!r}")
        if active_budget > 0:
            if (
                verify_result is not None
                and verifier_mode == "native_bulk_bplus1"
                and tree_mode == "chain"
                and canonical_commit_mode == "replay"
            ):
                target_top1 = []
                accepted = 0
                accepted_tokens = []
                committed_probe = [int(root_token), *[int(token) for token in candidates[: int(verify_result.accepted_count)]]]
                bonus = int(verify_result.next_token) if verify_result.next_token is not None else int(verify_result.commit_token)
                for row_idx, token in enumerate(committed_probe):
                    result = _slot_step(
                        session,
                        int(token),
                        position=context_tokens + row_idx,
                        slot=base_slot,
                        drafter=drafter,
                        capture_row=context_tokens + row_idx,
                    )
                    target_serial_forward_calls += 1
                    verify_rows_total += 1
                    canonical_commit_replay_rows += 1
                    finite_verify = finite_verify and math.isfinite(float(result.logit))
                    target_token = int(result.token_id)
                    target_top1.append(target_token)
                    bonus = target_token
                    if row_idx < len(committed_probe) - 1:
                        next_candidate = int(committed_probe[row_idx + 1])
                        if target_token != next_candidate:
                            break
                        accepted += 1
                        accepted_tokens.append(next_candidate)
            elif (
                verify_result is not None
                and verifier_mode == "native_bulk_bplus1"
                and tree_mode == "chain"
                and canonical_commit_mode == "branch_copy"
                and verifier_slot != base_slot
            ):
                session.copy_slot_state(verifier_slot, base_slot, kv_rows=context_tokens + 1 + accepted)
                state_copies += 1
            verify_elapsed = time.perf_counter() - t_verify
            verify_seconds_total += verify_elapsed
        if roctx is not None:
            roctx.pop()
            if rocprof_selected_region == "dflash_verify":
                roctx.profiler_pause()
        accepted_lengths.append(accepted)
        # Deployability oracle: per-cycle drafter confidence vs accepted count.
        # Does the cheap pre-verifier signal (draft top-1 softmax probs) predict
        # acceptance so an online gate can skip the verifier on likely rejects?
        confidence_trace.append(
            {
                "top1_probs": [float(p) for p in draft.top1_probabilities[:requested_active_budget]],
                "accepted": int(accepted),
                "active_budget": int(requested_active_budget),
                "cycle_wall_ms": (time.perf_counter() - t_cycle) * 1000.0,
            }
        )
        accepted_tokens = (
            list(verify_result.accepted_tokens)
            if verify_result is not None and verifier_mode == "native_bulk_bplus1" and tree_mode in {"chain_as_tree", "branching_topk"}
            else candidates[:accepted]
        )
        committed = [root_token, *accepted_tokens]
        t_commit = time.perf_counter()
        drafter.commit_context_rows(start=context_tokens, count=len(committed))
        commit_elapsed = time.perf_counter() - t_commit
        commit_seconds_total += commit_elapsed
        adaptive_budget.record_dflash_cycle(
            decision,
            visible_tokens=len(committed),
            cycle_wall_ms=(time.perf_counter() - t_cycle) * 1000.0,
            accepted_tokens=accepted,
            active_budget=active_budget,
            draft_ms=draft.draft_seconds * 1000.0,
            verify_ms=verify_elapsed * 1000.0,
            commit_ms=commit_elapsed * 1000.0,
            context_tokens=cycle_context_tokens,
        )
        if len(proposal_trace) < 16:
            trace_row = {
                "cycle": cycles,
                "root_position": context_tokens,
                "root_token": int(root_token),
                "draft_candidates": [int(token) for token in candidates],
                "draft_topk_tokens": [list(map(int, row)) for row in draft.topk_tokens],
                "draft_top1_probabilities": [float(x) for x in draft.top1_probabilities],
                "draft_p_min": float(draft_p_min),
                "active_budget_requested": int(requested_active_budget),
                "active_budget_after_confidence": int(active_budget),
                "confidence_limited": bool(confidence_limited),
                "target_top1_path": [int(token) for token in target_top1],
                "accepted": int(accepted),
                "accepted_tokens": [int(token) for token in accepted_tokens],
                "committed_tokens": [int(token) for token in committed],
                "bonus_token": int(bonus),
                "verifier_mode": verifier_mode,
                "tree_mode": tree_mode,
                "drafter_graph_status": graph_status,
                "drafter_graph_bucket": draft.graph.get("bucket_key"),
            }
            if verify_result is not None and target_batch is not None and verifier_mode == "native_bulk_bplus1":
                trace_row["commit_row"] = int(verify_result.commit_row)
                trace_row["commit_token"] = int(verify_result.commit_token)
                trace_row["tree_shape"] = list(target_batch.tree_shape)
                trace_row["target_parent_rows"] = list(map(int, target_batch.parent_rows))
                trace_row["target_draft_depths"] = list(map(int, target_batch.draft_depths))
                trace_row["active_mask"] = [bool(x) for x in target_batch.active_mask]
                if compiled_tree is not None:
                    trace_row["compiled_tree"] = {
                        "active_count": int(compiled_tree.active_count),
                        "active_candidate_tokens": list(map(int, compiled_tree.active_candidate_tokens)),
                        "tree_parents": list(map(int, compiled_tree.tree_parents)),
                        "draft_depths": list(map(int, compiled_tree.draft_depths)),
                        "child_ranks": list(map(int, compiled_tree.child_ranks)),
                        "cumulative_scores": [float(x) for x in compiled_tree.cumulative_scores],
                    }
            proposal_trace.append(trace_row)
        generated.extend(committed)
        root_token = int(bonus)
        context_tokens += len(committed)
    decode_seconds = time.perf_counter() - t1
    # Deployability oracle: does the drafter's depth-1 confidence predict
    # acceptance, so an online gate can route spec-vs-AR without a probe?
    _conf_rows = [c for c in confidence_trace if c.get("top1_probs")]
    if _conf_rows:
        _p1 = [float(c["top1_probs"][0]) for c in _conf_rows]
        _acc = [int(c["accepted"]) for c in _conf_rows]
        _n = len(_conf_rows)
        _mean_p1 = sum(_p1) / _n
        _mean_acc = sum(_acc) / _n
        _cov = sum((a - _mean_p1) * (b - _mean_acc) for a, b in zip(_p1, _acc)) / _n
        _var_p1 = sum((a - _mean_p1) ** 2 for a in _p1) / _n
        _var_acc = sum((b - _mean_acc) ** 2 for b in _acc) / _n
        _corr = (_cov / math.sqrt(_var_p1 * _var_acc)) if _var_p1 > 0 and _var_acc > 0 else None
        _p1_acc0 = [p for p, a in zip(_p1, _acc) if a == 0]
        _p1_acc_ge1 = [p for p, a in zip(_p1, _acc) if a >= 1]
        _bins: list[dict[str, Any]] = []
        for lo, hi in [(0.0, 0.8), (0.8, 0.9), (0.9, 0.97), (0.97, 0.999), (0.999, 1.01)]:
            _sel = [(p, a) for p, a in zip(_p1, _acc) if lo <= p < hi]
            if _sel:
                _bins.append({
                    "p1_lo": lo, "p1_hi": hi, "n": len(_sel),
                    "p_accept_ge1": sum(1 for _, a in _sel if a >= 1) / len(_sel),
                    "mean_accepted": sum(a for _, a in _sel) / len(_sel),
                })
        confidence_oracle = {
            "cycles": _n,
            "mean_p1": _mean_p1,
            "mean_accepted": _mean_acc,
            "corr_p1_accepted": _corr,
            "p1_mean_when_accept0": (sum(_p1_acc0) / len(_p1_acc0)) if _p1_acc0 else None,
            "p1_mean_when_accept_ge1": (sum(_p1_acc_ge1) / len(_p1_acc_ge1)) if _p1_acc_ge1 else None,
            "n_accept0": len(_p1_acc0),
            "n_accept_ge1": len(_p1_acc_ge1),
            "bins_by_p1": _bins,
            "note": "p1 = drafter depth-1 top-1 softmax prob (needs --draft-top-k>=2). If high p1 -> high P(accept>=1), an online confidence gate can route spec-vs-AR without a probe.",
        }
    else:
        confidence_oracle = {"cycles": 0, "note": "no confidence trace (run with --draft-top-k >= 2)"}
    _conf_out = os.environ.get("HIPENGINE_DFLASH_CONF_ORACLE_OUT")
    if _conf_out:
        with open(_conf_out, "a", encoding="utf-8") as _cf:
            _cf.write(json.dumps({"oracle": confidence_oracle, "trace": confidence_trace}) + "\n")
    metadata = {
        "confidence_oracle": confidence_oracle,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "draft_seconds": draft_seconds_total,
        "target_verify_seconds": verify_seconds_total,
        "commit_seconds": commit_seconds_total,
        "accepted_lengths": accepted_lengths,
        "target_verify_rows": verify_rows_total,
        "target_forward_calls": target_bulk_forward_calls + target_serial_forward_calls,
        "target_bulk_forward_calls": target_bulk_forward_calls,
        "target_serial_forward_calls": target_serial_forward_calls,
        "target_bulk_rows": target_bulk_rows_total,
        "canonical_commit_replay_rows": int(canonical_commit_replay_rows),
        "target_forwards_per_draft_call": (
            target_bulk_forward_calls / draft_calls
            if verifier_mode == "native_bulk_bplus1" and draft_calls
            else (target_serial_forward_calls / draft_calls if draft_calls else None)
        ),
        "gpu_accept_match_cpu": gpu_accept_match_cpu,
        "verifier_graph": {
            "mode": verifier_graph_mode,
            "status_counts": dict(sorted(verifier_graph_status_counts.items())),
            "validation_passed": verifier_graph_validation_passed if verifier_graph_validation_seen else None,
            "last": verifier_graph_last,
        },
        "draft_calls": draft_calls,
        "decode_cycles": cycles,
        "draft_tokens_proposed": draft_tokens_proposed,
        "tree_active_nodes_total": int(tree_active_nodes_total),
        "tree_top_k": int(tree_top_k),
        "draft_top_k": int(drafter.draft_top_k),
        "draft_p_min": float(draft_p_min),
        "whole_cycle_gate": float(whole_cycle_gate_threshold),
        "terminal_ar_tokens": int(terminal_ar_tokens),
        "terminal_ar_cycles": int(terminal_ar_cycles),
        "drafter_query_mode": drafter.query_mode,
        "drafter_query_rows": int(drafter.query_rows),
        "drafter_block_size": int(drafter.block_size),
        "confidence_limited_cycles": int(confidence_limited_cycles),
        "tree_compiler": "balanced_breadth_first_depth_topk" if tree_mode == "branching_topk" else None,
        "draft_native_phase_seconds": draft_phase_seconds,
        "draft_graph": {
            **drafter.graph_summary,
            "status_counts": dict(sorted(draft_graph_status_counts.items())),
            "validation_passed": draft_graph_validation_passed if draft_graph_validation_seen else None,
        },
        "draft_fusion": drafter.fusion_summary,
        "proposal_trace_sample": proposal_trace,
        "proposal_trace_count": draft_calls,
        "finite_draft_logits": finite_draft,
        "finite_verify_logits": finite_verify,
        "decode_tok_s": decode_tokens / decode_seconds if decode_seconds > 0 else None,
        "d2h": {
            "scalar_reads": (verify_rows_total if verifier_mode == "serial_in_place_single_slot" else target_serial_forward_calls + target_accept_scalar_reads),
            "vector_reads": d2h_vector_reads + (2 * target_bulk_forward_calls if verifier_mode == "native_bulk_bplus1" else 0),
            "scalar_values": (verify_rows_total if verifier_mode == "serial_in_place_single_slot" else target_serial_forward_calls + target_accept_scalar_values),
            "vector_values": d2h_vector_values + (2 * target_bulk_rows_total if verifier_mode == "native_bulk_bplus1" else 0),
            "full_logits_readbacks": 0,
            "notes": ["draft and verifier finite checks read compact top-k ids/values only; full logits are not copied"],
        },
        "memory": memory_stats(),
        "backend": session.backend,
        "target_arch": session.target_arch,
        "verifier_mode": verifier_mode,
        "verifier_graph_mode": verifier_graph_mode,
        "verifier_chain_attn_mode": chain_attn_mode,
        "verifier_tree_mode": tree_mode,
        "verifier_state_strategy": (
            (
                "branch_slot_bulk_verify_plus_c1_canonical_commit"
                if canonical_commit_mode == "replay"
                else (
                    "direct_bulk_canonical_commit"
                    if canonical_commit_mode == "bulk_direct"
                    else "branch_slot_bulk_verify_plus_branch_copy_commit"
                )
            )
            if verifier_mode == "native_bulk_bplus1" and tree_mode == "chain"
            else "in_place_verifier_commit"
        ),
        "canonical_commit_mode": canonical_commit_mode if verifier_mode == "native_bulk_bplus1" and tree_mode == "chain" else None,
        "native_bulk_verifier": verifier_mode == "native_bulk_bplus1",
        "drafter_context_mode": "append_only_projected_context_and_kv",
        "adaptive_budget": adaptive_budget.summary(),
        "draft_phase_timing_mode": "synchronized" if drafter.sync_draft_phases else "enqueue_until_final_sync",
        "base_slot": base_slot,
        "branch_slot_start": branch_slot_start,
        "verifier_state_copies_per_cycle": (state_copies / draft_calls) if draft_calls else 0,
        "verifier_state_copies_total": int(state_copies),
    }
    return generated[:decode_tokens], metadata


def _slot_step(
    session: Qwen35ParoResidentSession,
    token_id: int,
    *,
    position: int,
    slot: int,
    drafter: NativeDFlashChainDrafter | None = None,
    capture_row: int | None = None,
    sample: bool = True,
):
    session._set_slot_token_embedding(int(token_id), slot=slot)
    session._set_slot_position(int(position), slot=slot)
    kwargs: dict[str, Any] = {}
    if drafter is not None:
        if capture_row is None:
            raise ValueError("capture_row is required with drafter")
        kwargs = {
            "capture_layer_ids": drafter.target_layer_ids,
            "capture_hidden_concat": drafter.target_hidden_concat,
            "capture_row": int(capture_row),
        }
    hidden = session._run_layers(position=int(position), slot=slot, persist_aliases=False, stream=0, **kwargs)
    if not sample:
        return None
    return session._sample_from_hidden(hidden)


def _rotary_tables(max_positions: int, head_dim: int, theta: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(head_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(theta), -2.0 * dims / np.float32(head_dim))
    angles = positions * inv_freq
    cos_half = np.cos(angles).astype(np.float32)
    sin_half = np.sin(angles).astype(np.float32)
    return np.concatenate([cos_half, cos_half], axis=1), np.concatenate([sin_half, sin_half], axis=1)


def _row_for_artifact(prompt: dict[str, Any], budget: int, ar: tuple[list[int], dict[str, Any]], spec: tuple[list[int], dict[str, Any]]) -> dict[str, Any]:
    ar_tokens, ar_meta = ar
    spec_tokens, spec_meta = spec
    phase_seconds = spec_meta.get("draft_native_phase_seconds", {}) or {}
    drafter_context_mode = str(spec_meta.get("drafter_context_mode") or "")
    if drafter_context_mode == "append_only_projected_context_and_kv":
        draft_context_full_rebuild_seconds = 0.0
        draft_context_append_seconds = float(spec_meta.get("commit_seconds") or 0.0)
        draft_query_seconds = float(spec_meta.get("draft_seconds") or 0.0)
    elif drafter_context_mode.startswith("append_only"):
        draft_context_full_rebuild_seconds = 0.0
        draft_context_append_seconds = float(spec_meta.get("commit_seconds") or 0.0)
        draft_query_seconds = float(spec_meta.get("draft_seconds") or 0.0)
    else:
        draft_context_full_rebuild_seconds = float(phase_seconds.get("context_projection", spec_meta.get("draft_seconds") or 0.0))
        draft_context_append_seconds = 0.0
        draft_query_seconds = max(0.0, float(spec_meta.get("draft_seconds") or 0.0) - draft_context_full_rebuild_seconds)
    draft_graph = spec_meta.get("draft_graph") or {}
    graph_last = draft_graph.get("last") or {}
    graph_counts = draft_graph.get("status_counts") or {}
    graph_replay_steps = int(graph_counts.get("replayed", 0)) + int(graph_counts.get("captured_validated", 0)) + int(graph_counts.get("captured_validated_miss", 0))
    if graph_counts.get("replayed"):
        graph_status = "captured"
        graph_fallback_reason = None
    elif graph_counts.get("captured_validated") or graph_counts.get("captured_validated_miss"):
        graph_status = "captured"
        graph_fallback_reason = (
            "validated graph capture, but no cache-hit replay in decode because context_tokens changes every cycle"
            if not graph_counts.get("replayed")
            else None
        )
    elif graph_counts.get("capture_failed_fallback"):
        graph_status = "capture_failed"
        graph_fallback_reason = graph_last.get("fallback_reason")
    elif graph_counts.get("disabled"):
        graph_status = "not_captured"
        graph_fallback_reason = graph_last.get("fallback_reason") or "drafter graph mode is off"
    else:
        graph_status = "not_captured"
        graph_fallback_reason = None
    graph_bucket = graph_last.get("bucket_key") or {"mode": "dflash_drafter_propose", "draft_budget": budget, "verifier": spec_meta["verifier_mode"]}
    profile_route = str(spec_meta.get("profile_route") or "")
    tree_mode = str(spec_meta.get("verifier_tree_mode") or "chain")
    if profile_route == "ar":
        proposal_mode = "ar"
        verify_mode = "ar_decode"
    else:
        proposal_mode = "branching_topk" if tree_mode == "branching_topk" else "chain"
        verify_mode = "verify_tree" if tree_mode in {"chain_as_tree", "branching_topk"} else "verify_chain"
    draft_top_k = int(spec_meta.get("draft_top_k") or 1)

    return {
        "prompt": {
            "id": prompt.get("id"),
            "dataset": prompt.get("dataset"),
            "category": prompt.get("benchmark_group"),
            "prompt_tokens": prompt.get("prompt_tokens"),
            "prompt_ids_sha256": prompt.get("prompt_ids_sha256"),
            "prompt_text_sha256": prompt.get("prompt_text_sha256"),
            "prompt_preview": prompt.get("prompt_preview"),
            "representative": bool(prompt.get("representative")),
        },
        "config": {
            "name": f"full_model_{proposal_mode}_b{0 if profile_route == 'ar' else budget}",
            "provider": "profile_route" if profile_route else "dflash",
            "proposal_mode": proposal_mode,
            "verify_mode": verify_mode,
            "draft_budget": 0 if profile_route == "ar" else budget,
            "topk": 0 if profile_route == "ar" else draft_top_k,
            "draft_p_min": spec_meta.get("draft_p_min"),
            "whole_cycle_gate": spec_meta.get("whole_cycle_gate"),
            "tree_mode": tree_mode,
            "tree_budget": budget if verify_mode == "verify_tree" else None,
            "profile_route": profile_route or None,
        },
        "ar": {
            "same_session_control": bool(ar_meta.get("same_session_control", False)),
            "same_process_control": bool(ar_meta.get("same_process_control", True)),
            "decode_seconds": ar_meta["decode_seconds"],
            "finite_logits": ar_meta["finite_logits"],
            "generated_ids": ar_tokens,
        },
        "spec": {
            "decode_seconds": spec_meta["decode_seconds"],
            "draft_seconds": spec_meta["draft_seconds"],
            "draft_context_full_rebuild_seconds": draft_context_full_rebuild_seconds,
            "draft_context_append_seconds": draft_context_append_seconds,
            "draft_query_seconds": draft_query_seconds,
            "draft_native_phase_seconds": phase_seconds,
            "draft_graph": draft_graph,
            "draft_fusion": spec_meta.get("draft_fusion"),
            "adaptive_budget": spec_meta.get("adaptive_budget"),
            "drafter_context_mode": spec_meta.get("drafter_context_mode"),
            "draft_phase_timing_mode": spec_meta.get("draft_phase_timing_mode"),
            "proposal_trace_sample": spec_meta.get("proposal_trace_sample", []),
            "proposal_trace_count": spec_meta.get("proposal_trace_count", spec_meta["draft_calls"]),
            "target_verify_seconds": spec_meta["target_verify_seconds"],
            "commit_seconds": spec_meta["commit_seconds"],
            "target_verify_rows": spec_meta["target_verify_rows"],
            "target_forward_calls": spec_meta.get("target_forward_calls"),
            "target_bulk_forward_calls": spec_meta.get("target_bulk_forward_calls"),
            "target_serial_forward_calls": spec_meta.get("target_serial_forward_calls"),
            "target_bulk_rows": spec_meta.get("target_bulk_rows"),
            "canonical_commit_replay_rows": spec_meta.get("canonical_commit_replay_rows"),
            "target_forwards_per_draft_call": spec_meta.get("target_forwards_per_draft_call"),
            "gpu_accept_match_cpu": spec_meta.get("gpu_accept_match_cpu"),
            "verifier_graph": spec_meta.get("verifier_graph"),
            "draft_tokens_proposed": spec_meta.get("draft_tokens_proposed", spec_meta["draft_calls"] * budget),
            "draft_tokens": spec_meta.get("draft_tokens_proposed", spec_meta["draft_calls"] * budget),
            "confidence_limited_cycles": spec_meta.get("confidence_limited_cycles"),
            "tree_active_nodes_total": spec_meta.get("tree_active_nodes_total"),
            "tree_top_k": spec_meta.get("tree_top_k"),
            "draft_top_k": spec_meta.get("draft_top_k"),
            "draft_p_min": spec_meta.get("draft_p_min"),
            "whole_cycle_gate": spec_meta.get("whole_cycle_gate"),
            "terminal_ar_tokens": spec_meta.get("terminal_ar_tokens"),
            "terminal_ar_cycles": spec_meta.get("terminal_ar_cycles"),
            "drafter_query_mode": spec_meta.get("drafter_query_mode"),
            "drafter_query_rows": spec_meta.get("drafter_query_rows"),
            "drafter_block_size": spec_meta.get("drafter_block_size"),
            "tree_compiler": spec_meta.get("tree_compiler"),
            "accepted_draft_tokens": sum(int(x) for x in spec_meta["accepted_lengths"]),
            "accepted_lengths": spec_meta["accepted_lengths"],
            "confidence_oracle": spec_meta.get("confidence_oracle"),
            "draft_calls": spec_meta["draft_calls"],
            "finite_draft_logits": spec_meta["finite_draft_logits"],
            "finite_verify_logits": spec_meta["finite_verify_logits"],
            "generated_ids": spec_tokens,
            "d2h": spec_meta["d2h"],
            "graph": {
                "status": graph_status,
                "replay_steps": graph_replay_steps,
                "bucket_key": graph_bucket,
                "validation_passed": draft_graph.get("validation_passed"),
                "fallback_reason": graph_fallback_reason,
            },
            "verifier_mode": spec_meta["verifier_mode"],
            "verifier_tree_mode": spec_meta.get("verifier_tree_mode"),
            "verifier_state_strategy": spec_meta.get("verifier_state_strategy"),
            "canonical_commit_mode": spec_meta.get("canonical_commit_mode"),
            "verifier_state_copies_per_cycle": spec_meta.get("verifier_state_copies_per_cycle"),
            "verifier_state_copies_total": spec_meta.get("verifier_state_copies_total"),
            "native_bulk_verifier": spec_meta["native_bulk_verifier"],
            "same_session_control": bool(spec_meta.get("same_session_control", False)),
            "same_process_control": bool(spec_meta.get("same_process_control", True)),
            "backend": spec_meta.get("backend"),
            "target_arch": spec_meta.get("target_arch"),
        },
        "quality_gate": {"exact_match_ar": ar_tokens == spec_tokens, "finite_ar_logits": ar_meta["finite_logits"], "finite_dflash_draft_logits": spec_meta["finite_draft_logits"], "finite_dflash_verify_logits": spec_meta["finite_verify_logits"]},
        "memory": {"peak_allocated_bytes": spec_meta["memory"].get("peak_allocated_bytes", 0), "peak_reserved_bytes": 0, "hip_used_peak_sampled_bytes": 0},
        "decode_tokens": len(ar_tokens),
    }


def _select_prompts(path: Path, *, groups: set[str], limit: int) -> list[dict[str, Any]]:
    rows = [row for row in load_prompt_records(path) if str(row.get("benchmark_group")) in groups]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no prompt rows selected")
    for row in rows:
        if not row.get("prompt_ids"):
            raise ValueError(f"prompt row {row.get('id')} lacks prompt_ids")
    return rows


def _git_context() -> dict[str, Any]:
    def run(cmd: list[str]) -> str | None:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True, timeout=10)
        return proc.stdout.strip() if proc.returncode == 0 else None
    status = run(["git", "status", "--porcelain"])
    return {"hipengine_commit": run(["git", "rev-parse", "HEAD"]), "hipengine_branch": run(["git", "branch", "--show-current"]), "hipengine_dirty": bool(status), "hipengine_status_porcelain": status}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_PATH)
    parser.add_argument("--drafter-model", default=DEFAULT_DRAFTER_PATH)
    parser.add_argument("--prompt-fixture", type=Path, default=DEFAULT_STABLE_PROMPT_FIXTURE)
    parser.add_argument("--prompt-groups", default="code_promotion,robustness")
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--draft-budgets", default="4")
    parser.add_argument("--backend", default="auto", choices=("auto", "hip_gfx1100", "hip_gfx1151"))
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--sync-draft-phases", action="store_true", help="Diagnostic only: synchronize after major drafter phases before timing them")
    parser.add_argument("--verifier-mode", choices=("native_bulk_bplus1", "serial_in_place_single_slot"), default="native_bulk_bplus1")
    parser.add_argument("--verifier-graph", choices=("off", "auto", "validate"), default="off", help="Prototype HIP graph capture for native B+1 verifier forward+accept; auto replays fixed rows/capture-width buckets")
    parser.add_argument("--full-attn-chain-mode", choices=("c1_loop", "batched", "decode_batched"), default="c1_loop", help="Native B+1 verifier full-attention scheduling: c1_loop (per-row resident decode kernels), batched (one prefill-style batched pass per layer), or decode_batched (small-B row-batched decode attention + batched projections/MoE)")
    parser.add_argument(
        "--tree-mode",
        choices=("chain", "chain_as_tree", "branching_topk"),
        default="chain",
        help=(
            "Verifier topology: chain (default) uses verify_chain_bulk_and_commit;"
            " chain_as_tree wraps the chain candidates as a degenerate (linear)"
            " tree and routes through verify_tree_bulk_and_commit -- same accept"
            " profile, isolates the tree kernel's overhead vs the chain batched"
            " path; branching_topk compiles row-wise drafter top-K into a"
            " balanced breadth-first DDTree and routes through verify_tree_bulk_and_commit."
        ),
    )
    parser.add_argument("--tree-top-k", type=int, default=2, help="Top-K per drafter depth row for --tree-mode branching_topk (1..8; ignored by chain modes)")
    parser.add_argument("--draft-top-k", type=int, default=1, help="Top-K readback per DFlash chain draft row (1..8). Use >=2 with --draft-p-min")
    parser.add_argument("--draft-p-min", type=float, default=0.0, help="Optional chain draft confidence floor. Stops verification at the first top-1 probability below this value")
    parser.add_argument("--whole-cycle-gate", type=float, default=0.0, help="Deployable whole-cycle confidence gate threshold in (0,1]. When the drafter depth-1 top-1 probability is below this value the cycle drops to plain AR (verify root only) instead of running the chain; no mid-chain truncation. Requires --tree-mode chain and --draft-top-k >= 2, and is mutually exclusive with --draft-p-min. Overrides HIPENGINE_DFLASH_WHOLE_CYCLE_GATE when > 0")
    parser.add_argument("--drafter-graph", choices=("off", "auto", "validate"), default="off", help="Prototype HIP graph capture for native DFlash propose(); auto replays cache hits, validate records capture parity without requiring reuse")
    parser.add_argument("--drafter-fusion", choices=("off", "qkv"), default="off", help="Enable prototype DFlash drafter kernel fusions; qkv fuses query-side Q/K/V projections with unfused fallback available")
    parser.add_argument(
        "--drafter-bucket",
        choices=("exact", "cross_bucket"),
        default="exact",
        help=(
            "Bucket shape function used by the drafter HIP graph cache key. 'exact'"
            " keys on the live context length (each cycle is unique -> no replay)."
            " 'cross_bucket' rounds the live context length to a BeeLlama-style"
            " bucket so consecutive cycles can share a captured graph; uses the"
            " bucketed attention kernel with a device-resident live count."
        ),
    )
    parser.add_argument(
        "--drafter-query-mode",
        choices=("block", "budget_prefix"),
        default="block",
        help=(
            "DFlash drafter query sequence length. 'block' preserves the z-lab"
            " full block_size contract. 'budget_prefix' is a speed diagnostic"
            " that runs only root+B query rows; proposals can differ because"
            " the drafter query-block attention is non-causal."
        ),
    )
    parser.add_argument(
        "--adaptive-budget",
        choices=("off", "on"),
        default="off",
        help=(
            "Enable the host-side DFlash-vs-AR adaptive budget controller. The"
            " controller uses the same-session AR decode rate as its baseline and"
            " routes negative-profit cycles to AR after a hysteretic cooldown."
        ),
    )
    parser.add_argument(
        "--adaptive-min-remaining-tokens",
        type=int,
        default=0,
        help=(
            "When adaptive budget is on, route to AR while remaining decode"
            " tokens are below this horizon. This avoids short prompts paying a"
            " failed DFlash probe that cannot be amortized."
        ),
    )
    parser.add_argument(
        "--adaptive-probe-amortization-tokens",
        type=int,
        default=128,
        help=(
            "When adaptive budget is on, require this many extra remaining"
            " decode tokens beyond --adaptive-min-remaining-tokens before"
            " starting or retrying a DFlash probe. Already-promoted DFlash uses"
            " the normal remaining-token horizon."
        ),
    )
    parser.add_argument(
        "--terminal-ar-tokens",
        type=int,
        default=0,
        help=(
            "Default-off diagnostic: route cycles with remaining decode tokens"
            " below this threshold through plain AR instead of drafting. This"
            " skips low-amortization terminal DFlash cycles without enabling the"
            " adaptive probe controller."
        ),
    )
    parser.add_argument(
        "--canonical-commit-mode",
        choices=("replay", "bulk_direct", "branch_copy"),
        default="replay",
        help=(
            "Native chain verifier commit strategy. 'replay' is the exact"
            " default: verify on a branch slot, then replay the accepted prefix"
            " through c=1 on the canonical slot. 'bulk_direct' trusts the bulk"
            " verifier as canonical. 'branch_copy' verifies on a branch slot and"
            " copies the bulk-committed branch state back to the canonical slot."
        ),
    )
    parser.add_argument(
        "--profile-route-manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON manifest for profile/oracle routing diagnostics."
            " Values may be ar, chain, tree, or spec. Keys match prompt id,"
            " prompt hash, or benchmark group; default falls back to spec."
            " An optional terminal_ar_tokens integer/object can override"
            " --terminal-ar-tokens globally or per prompt key."
        ),
    )
    parser.add_argument("--hardware-gpu", default=None, help="Human-readable GPU name to record in the benchmark artifact")
    parser.add_argument(
        "--roctx",
        action="store_true",
        help="Emit ROCTX ranges named dflash_verify_pass_N around each DFlash target-verify window for rocprof marker slicing.",
    )
    parser.add_argument(
        "--rocprof-selected-region",
        choices=("none", "dflash_verify"),
        default="none",
        help="Call roctxProfilerResume/Pause around the selected DFlash phase for rocprofv3 --selected-regions.",
    )
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.decode_tokens <= 0:
        raise ValueError("--decode-tokens must be positive")
    if args.tree_top_k <= 0 or args.tree_top_k > 8:
        raise ValueError("--tree-top-k must be in [1, 8]")
    if args.draft_top_k <= 0 or args.draft_top_k > 8:
        raise ValueError("--draft-top-k must be in [1, 8]")
    if args.draft_p_min < 0.0 or args.draft_p_min > 1.0:
        raise ValueError("--draft-p-min must be in [0, 1]")
    if args.adaptive_min_remaining_tokens < 0:
        raise ValueError("--adaptive-min-remaining-tokens must be non-negative")
    if args.adaptive_probe_amortization_tokens < 0:
        raise ValueError("--adaptive-probe-amortization-tokens must be non-negative")
    if args.terminal_ar_tokens < 0:
        raise ValueError("--terminal-ar-tokens must be non-negative")
    if args.canonical_commit_mode != "replay" and (args.verifier_mode != "native_bulk_bplus1" or args.tree_mode != "chain"):
        raise ValueError("--canonical-commit-mode bulk_direct/branch_copy requires native_bulk_bplus1 chain mode")
    if args.draft_p_min > 0.0:
        if args.tree_mode != "chain":
            raise ValueError("--draft-p-min currently supports --tree-mode chain only")
        if args.draft_top_k < 2:
            raise ValueError("--draft-p-min requires --draft-top-k >= 2")
    if args.whole_cycle_gate < 0.0 or args.whole_cycle_gate > 1.0:
        raise ValueError("--whole-cycle-gate must be in [0, 1]")
    if args.whole_cycle_gate > 0.0:
        if args.tree_mode != "chain":
            raise ValueError("--whole-cycle-gate currently supports --tree-mode chain only")
        if args.draft_top_k < 2:
            raise ValueError("--whole-cycle-gate requires --draft-top-k >= 2")
        if args.draft_p_min > 0.0:
            raise ValueError("--whole-cycle-gate and --draft-p-min are mutually exclusive")
    if args.tree_mode in {"chain_as_tree", "branching_topk"} and args.verifier_mode != "native_bulk_bplus1":
        raise ValueError("tree modes require --verifier-mode native_bulk_bplus1")
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8") if args.compiler_version_file else None
    target = Path(args.target_model)
    drafter = Path(args.drafter_model)
    validation = validate_dflash_artifact_pair(target_model=target, drafter_model=drafter, raise_on_error=True)
    prompts = _select_prompts(args.prompt_fixture, groups={x.strip() for x in args.prompt_groups.split(",") if x.strip()}, limit=args.max_prompts)
    profile_route_default, profile_routes, profile_route_manifest = _load_profile_route_manifest(args.profile_route_manifest)
    budgets = [int(x) for x in args.draft_budgets.split(",") if x.strip()]
    prefill_config = PrefillConfig(auto_tune_chunk_sizes=True)
    rows: list[dict[str, Any]] = []
    commands = {"benchmark": " ".join(shlex.quote(part) for part in ["python3", "scripts/dflash_chain_e2e_bench.py", *(argv if argv is not None else sys.argv[1:])])}
    roctx = _Roctx(enabled=args.roctx or args.rocprof_selected_region != "none")
    if args.rocprof_selected_region != "none" and not roctx.profiler_controls_available:
        print(
            "warning: --rocprof-selected-region requested but libroctx64.so lacks "
            "roctxProfilerResume/Pause; rocprofv3 --selected-regions will emit no samples "
            "unless an SDK ROCTX shim is first on LD_LIBRARY_PATH",
            file=sys.stderr,
        )
    for prompt in prompts:
        prompt_ids = [int(x) for x in prompt["prompt_ids"]]
        profile_route = _profile_route_for_prompt(prompt, default=profile_route_default, routes=profile_routes)
        terminal_ar_tokens_for_row = _terminal_ar_tokens_for_prompt(
            prompt,
            default=args.terminal_ar_tokens,
            manifest=profile_route_manifest,
        )
        draft_budgets_for_row = _draft_budgets_for_prompt(
            prompt,
            default=budgets,
            manifest=profile_route_manifest,
        )
        if profile_route == "ar":
            # Plain-AR route is independent of speculative budget; avoid
            # duplicating the same AR row when a manifest uses multi-budget
            # overrides for other prompts.
            draft_budgets_for_row = [draft_budgets_for_row[0]]
        for budget in draft_budgets_for_row:
            tree_top_k_for_row = args.tree_top_k if args.tree_mode == "branching_topk" or profile_route == "tree" else 1
            ar, spec = run_same_session_pair(
                model=target,
                drafter_model=drafter,
                prompt_ids=prompt_ids,
                decode_tokens=args.decode_tokens,
                candidate_budget=budget,
                backend=args.backend,
                max_layers=args.max_layers,
                compiler_version=compiler_version,
                require_cached_build=args.require_cached_build,
                prefill_config=prefill_config,
                sync_draft_phases=args.sync_draft_phases,
                verifier_mode=args.verifier_mode,
                verifier_graph_mode=args.verifier_graph,
                drafter_graph_mode=args.drafter_graph,
                drafter_fusion_mode=args.drafter_fusion,
                drafter_bucket_mode=args.drafter_bucket,
                drafter_query_mode=args.drafter_query_mode,
                draft_top_k=args.draft_top_k,
                draft_p_min=args.draft_p_min,
                whole_cycle_gate=args.whole_cycle_gate,
                adaptive_budget_mode=args.adaptive_budget,
                adaptive_min_remaining_tokens=args.adaptive_min_remaining_tokens,
                adaptive_probe_amortization_tokens=args.adaptive_probe_amortization_tokens,
                terminal_ar_tokens=terminal_ar_tokens_for_row,
                chain_attn_mode=args.full_attn_chain_mode,
                tree_mode=args.tree_mode,
                tree_top_k=tree_top_k_for_row,
                canonical_commit_mode=args.canonical_commit_mode,
                profile_route=profile_route,
                roctx=roctx,
                rocprof_selected_region=args.rocprof_selected_region,
            )
            rows.append(_row_for_artifact(prompt, budget, ar, spec))
    if args.tree_mode == "branching_topk":
        run_tag = "dflash-ddtree-branching-topk-full-model-e2e"
        artifact_summary = "Full-model hipEngine DFlash DDTree branching top-K E2E run with same-session AR control and native tree verifier"
        workload_shape = "full_model_dflash_ddtree_branching_topk_e2e"
    elif args.tree_mode == "chain_as_tree":
        run_tag = "dflash-ddtree-chain-as-tree-full-model-e2e"
        artifact_summary = "Full-model hipEngine DFlash DDTree chain_as_tree E2E run with same-session AR control and native tree verifier"
        workload_shape = "full_model_dflash_ddtree_chain_as_tree_e2e"
    else:
        run_tag = "dflash-chain-full-model-e2e"
        artifact_summary = "Full-model hipEngine DFlash chain E2E run with same-session AR control, native drafter, and serial/native target verifier"
        workload_shape = "full_model_dflash_chain_e2e"
    target_name, target_revision = _hf_snapshot_identity(
        target,
        default_name=DEFAULT_TARGET_MODEL,
        default_revision=DEFAULT_TARGET_REVISION,
    )
    drafter_name, drafter_revision = _hf_snapshot_identity(
        drafter,
        default_name=DEFAULT_DFLASH_DRAFTER,
        default_revision=DEFAULT_DRAFTER_REVISION,
    )
    artifact = build_speculative_artifact(
        run_tag=run_tag,
        summary=artifact_summary,
        rows=rows,
        models=SpeculativeBenchmarkModels(
            target_name=target_name,
            target_path=str(target),
            target_revision=target_revision,
            drafter_name=drafter_name,
            drafter_path=str(drafter),
            drafter_revision=drafter_revision,
        ),
        status="diagnostic",
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware={"backend": rows[0]["spec"].get("backend") if rows else args.backend, "arch": rows[0]["spec"].get("target_arch") if rows else None, "gpu": args.hardware_gpu},
        software={**_git_context(), "python": platform.python_version(), "platform": platform.platform(), "hipcc_version": compiler_version},
        workload={
            "shape": workload_shape,
            "provider": "dflash",
            "verify_modes": ["verify_tree" if args.tree_mode in {"chain_as_tree", "branching_topk"} else "verify_chain"],
            "draft_budgets": budgets,
            "decode_tokens": args.decode_tokens,
            "prompt_suite": str(args.prompt_fixture),
            "prompt_suite_sha256": file_sha256(args.prompt_fixture),
            "artifact_validation": validation,
            "verifier_mode": args.verifier_mode,
            "verifier_graph_mode": args.verifier_graph,
            "verifier_chain_attn_mode": args.full_attn_chain_mode,
            "verifier_tree_mode": args.tree_mode,
            "tree_top_k": args.tree_top_k if args.tree_mode == "branching_topk" else 1,
            "draft_top_k": args.tree_top_k if args.tree_mode == "branching_topk" else args.draft_top_k,
            "draft_p_min": args.draft_p_min,
            "whole_cycle_gate": args.whole_cycle_gate,
            "tree_compiler": "balanced_breadth_first_depth_topk" if args.tree_mode == "branching_topk" else None,
            "native_bulk_verifier": args.verifier_mode == "native_bulk_bplus1",
            "drafter_graph_mode": args.drafter_graph,
            "drafter_fusion_mode": args.drafter_fusion,
            "drafter_bucket_mode": args.drafter_bucket,
            "drafter_query_mode": args.drafter_query_mode,
            "adaptive_budget_mode": args.adaptive_budget,
            "adaptive_min_remaining_tokens": args.adaptive_min_remaining_tokens,
            "adaptive_probe_amortization_tokens": args.adaptive_probe_amortization_tokens,
            "terminal_ar_tokens": args.terminal_ar_tokens,
            "canonical_commit_mode": args.canonical_commit_mode,
            "profile_route_manifest": str(args.profile_route_manifest) if args.profile_route_manifest else None,
            "profile_route_default": profile_route_default,
            "profile_routes": profile_routes,
            "profile_route_draft_budgets": (profile_route_manifest or {}).get("draft_budgets") if profile_route_manifest else None,
            "profile_route_manifest_body": profile_route_manifest,
            "roctx_markers": bool(args.roctx),
            "rocprof_selected_region": args.rocprof_selected_region,
            "promotion_blocker": (
                "native B+1 verifier ran, but full chain must still beat same-session AR before promotion"
                if args.verifier_mode == "native_bulk_bplus1"
                else "serial in-place single-slot verifier still issues B+1 sequential single-token forwards per cycle; native bulk target verifier is required before promotion"
            ),
        },
        commands=commands,
        notes=[
            "Actual full-model target and native DFlash drafter execution with same-session AR control; diagnostic unless native bulk verification and speed gates pass.",
            "Prompt fixture includes code/general/multilingual categories via fixtures/dflash/stable_prompts.jsonl.",
            "Phase A optimization: single-slot in-place verify (no per-candidate state copies, no commit copy).",
            "Phase B optimization: append-only projected_context_norm cache; only newly committed rows are re-projected.",
            "Phase C optimization: append-only per-layer rotated K and V context cache; per-cycle propose() processes query rows only.",
        ],
        decision_reason=(
            "full-model diagnostic only: native bulk verifier did not produce a same-session AR speed win"
            if args.verifier_mode == "native_bulk_bplus1"
            else "full-model diagnostic only: serial_in_place_single_slot verifier is not the promotable native bulk verifier"
        ),
    )
    aggregate = artifact["measurements"]["aggregate"]
    native_bulk_promotable = args.verifier_mode == "native_bulk_bplus1"
    gates_passed = bool(aggregate["all_correctness_passed"] and aggregate["speed_gate_gt_1p10"])
    if native_bulk_promotable and gates_passed:
        decision_reason = "correctness and speed gate passed on native bulk verifier"
        artifact["status"] = "accepted"
        artifact["performance_claim"] = True
        artifact["decision_reason"] = decision_reason
        artifact["decision"]["accepted"] = True
        artifact["decision"]["reason"] = decision_reason
        artifact["workload"]["promotion_blocker"] = None
    elif native_bulk_promotable and not aggregate["all_correctness_passed"]:
        decision_reason = "one or more rows failed exact/finite correctness gates"
        artifact["decision_reason"] = decision_reason
        artifact["decision"]["reason"] = decision_reason
        artifact["workload"]["promotion_blocker"] = decision_reason
    elif native_bulk_promotable:
        decision_reason = "speed gate >1.10x AR not met"
        artifact["decision_reason"] = decision_reason
        artifact["decision"]["reason"] = decision_reason
        artifact["workload"]["promotion_blocker"] = decision_reason
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "all_correctness_passed": artifact["measurements"]["aggregate"]["all_correctness_passed"], "speedup_vs_ar": artifact["measurements"]["aggregate"].get("speedup_vs_ar"), "performance_claim": artifact["performance_claim"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
