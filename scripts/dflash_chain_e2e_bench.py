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
import json
import math
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
    dflash_add_bf16,
    dflash_concat_rows_bf16,
    dflash_concat_rows_f32,
    dflash_dense_bf16_to_bf16,
    dflash_dense_bf16_to_f32,
    dflash_gqa_attention_f32_bf16,
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
from hipengine.speculative import DraftBatch, TargetVerifyBatch

DEFAULT_TARGET_PATH = "/models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e"
DEFAULT_DRAFTER_PATH = "/models/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719"
DEFAULT_TARGET_REVISION = "501ef8635e5cfb5a7497d232358ca8d1afc0c66e"
DEFAULT_DRAFTER_REVISION = "42d3b34d588423cdae7ba8f53a8cf7789346a719"


@dataclass(frozen=True)
class DraftResult:
    candidate_tokens: tuple[int, ...]
    draft_seconds: float
    finite_logits: bool
    d2h_vector_reads: int
    d2h_vector_values: int
    phase_seconds: dict[str, float]
    graph: dict[str, Any]


@dataclass(frozen=True)
class DFlashDrafterGraphBucket:
    candidate_budget: int
    block_size: int
    context_tokens: int
    max_context_tokens: int
    num_layers: int
    hidden_size: int
    mode: str = "append_only_projected_context_and_kv"

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_budget": self.candidate_budget,
            "block_size": self.block_size,
            "context_tokens": self.context_tokens,
            "max_context_tokens": self.max_context_tokens,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "mode": self.mode,
        }

    @property
    def key(self) -> tuple[int, int, int, int, int, int, str]:
        return (
            self.candidate_budget,
            self.block_size,
            self.context_tokens,
            self.max_context_tokens,
            self.num_layers,
            self.hidden_size,
            self.mode,
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


class NativeDFlashChainDrafter:
    """Correctness-first native DFlash top-1 chain proposer.

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
    ) -> None:
        self.session = session
        self.runtime = session.runtime
        self.device = Device("hip", 0)
        self.candidate_budget = int(candidate_budget)
        self.sync_draft_phases = bool(sync_draft_phases)
        if graph_mode not in {"off", "auto", "validate"}:
            raise ValueError("graph_mode must be off, auto, or validate")
        self.graph_mode = graph_mode
        if fusion_mode not in {"off", "qkv"}:
            raise ValueError("fusion_mode must be off or qkv")
        self.fusion_mode = fusion_mode
        self._fusion_counts: Counter[str] = Counter()
        self._graph_cache: dict[tuple[int, int, int, int, int, int, str], DFlashDrafterGraphEntry] = {}
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

        top1, top1_values = self._read_top1()
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
        return DraftResult(
            candidate_tokens=tuple(int(x) for x in top1.reshape(-1).tolist()),
            draft_seconds=draft_seconds,
            finite_logits=bool(np.isfinite(top1_values).all()),
            d2h_vector_reads=2,
            d2h_vector_values=2 * self.candidate_budget,
            phase_seconds=phases,
            graph=graph_info,
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

    def _run_propose_kernels(self, *, context_tokens: int, stream: int = 0, phases: dict[str, float] | None = None) -> None:
        phase_t = time.perf_counter()
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
            self.block_size,
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
            query_out = self._run_layer(layer, context_tokens=context_tokens, query_in=query_in, query_out=query_out, stream=stream)
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
            self.block_size,
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
            1,
            threads=256,
            stream=stream,
            library=self.lm_library,
            runtime=self.runtime,
        )
        if phases is not None:
            self.runtime.device_synchronize()
            phases["topk_and_readback"] = time.perf_counter() - phase_t

    def _read_top1(self) -> tuple[np.ndarray, np.ndarray]:
        top1 = np.empty((self.candidate_budget, 1), dtype=np.int32)
        top1_values = np.empty((self.candidate_budget, 1), dtype=np.float32)
        copy_device_to_host(host_array_ptr(top1), self._buffer_for(self.top1_ids), runtime=self.runtime)
        copy_device_to_host(host_array_ptr(top1_values), self._buffer_for(self.top1_values), runtime=self.runtime)
        return top1, top1_values

    def _bucket_for(self, context_tokens: int) -> DFlashDrafterGraphBucket:
        return DFlashDrafterGraphBucket(
            candidate_budget=self.candidate_budget,
            block_size=self.block_size,
            context_tokens=int(context_tokens),
            max_context_tokens=self.max_context_tokens,
            num_layers=int(self.config.num_hidden_layers),
            hidden_size=self.hidden,
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
        self._run_propose_kernels(context_tokens=context_tokens, stream=0, phases=phases)
        self.runtime.device_synchronize()
        direct_tokens_arr, _ = self._read_top1()
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
                self._run_propose_kernels(context_tokens=context_tokens, stream=stream, phases=None)
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
            graph_tokens_arr, _ = self._read_top1()
            graph_tokens = tuple(int(x) for x in graph_tokens_arr.reshape(-1).tolist())
            validation_seconds = time.perf_counter() - validate_t
            validation_passed = graph_tokens == direct_tokens
            if not validation_passed:
                self._graph_validation_failures += 1
                reason = "graph replay candidates differed from direct fallback"
                self._graph_fallback_reasons[reason] += 1
                # Restore direct fallback outputs before propose() performs its
                # final readback; validation failure must not perturb the chain.
                self._run_propose_kernels(context_tokens=context_tokens, stream=0, phases=None)
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
        self.top1_values = self._empty((self.candidate_budget, 1), DType.FP32)
        self.top1_ids = self._empty((self.candidate_budget, 1), DType.INT32)
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

    def _run_layer(self, layer: int, *, context_tokens: int, query_in: Tensor, query_out: Tensor, stream: int = 0) -> Tensor:
        prefix = f"layers.{layer}"
        total_kv = context_tokens + self.block_size
        fp32_bytes = DType.FP32.itemsize
        bf16_bytes = DType.BF16.itemsize
        k_layer_base = self.kv_cache_keys.ptr + layer * self.max_context_tokens * self.kv_features * fp32_bytes
        v_layer_base = self.kv_cache_values.ptr + layer * self.max_context_tokens * self.kv_features * bf16_bytes
        dflash_rmsnorm_bf16(query_in.ptr, self.weights.tensor(f"{prefix}.input_layernorm.weight").ptr, self.norm.ptr, self.block_size, self.hidden, threads=128, stream=stream, library=self.library, runtime=self.runtime)
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
                self.block_size,
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
            dflash_dense_bf16_to_f32(self.norm.ptr, self.weights.tensor(f"{prefix}.self_attn.q_proj.weight").ptr, self.q_raw.ptr, self.block_size, self.hidden, self.attn_features, threads=128, stream=stream, library=self.library, runtime=self.runtime)
            dflash_dense_bf16_to_f32(self.norm.ptr, self.weights.tensor(f"{prefix}.self_attn.k_proj.weight").ptr, self.k_q.ptr, self.block_size, self.hidden, self.kv_features, threads=128, stream=stream, library=self.library, runtime=self.runtime)
            dflash_dense_bf16_to_bf16(self.norm.ptr, self.weights.tensor(f"{prefix}.self_attn.v_proj.weight").ptr, self.v_q.ptr, self.block_size, self.hidden, self.kv_features, threads=128, stream=stream, library=self.library, runtime=self.runtime)
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
            self.block_size,
            self.block_size,
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
            context_tokens,
            self.block_size,
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
            context_tokens,
            self.block_size,
            self.kv_features,
            threads=128,
            stream=stream,
            library=self.library,
            runtime=self.runtime,
        )
        dflash_gqa_attention_f32_bf16(self.q_rot.ptr, self.k_rot.ptr, self.v_all.ptr, self.attn.ptr, 1, self.block_size, total_kv, self.q_heads, self.kv_heads, self.head_dim, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.attn.ptr, self.weights.tensor(f"{prefix}.self_attn.o_proj.weight").ptr, self.attn_proj.ptr, self.block_size, self.attn_features, self.hidden, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        dflash_add_bf16(query_in.ptr, self.attn_proj.ptr, self.hidden_attn.ptr, self.block_size * self.hidden, threads=256, stream=stream, library=self.library, runtime=self.runtime)
        dflash_rmsnorm_bf16(self.hidden_attn.ptr, self.weights.tensor(f"{prefix}.post_attention_layernorm.weight").ptr, self.post.ptr, self.block_size, self.hidden, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.post.ptr, self.weights.tensor(f"{prefix}.mlp.gate_proj.weight").ptr, self.gate.ptr, self.block_size, self.hidden, self.intermediate, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.post.ptr, self.weights.tensor(f"{prefix}.mlp.up_proj.weight").ptr, self.up.ptr, self.block_size, self.hidden, self.intermediate, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        dflash_silu_mul_bf16(self.gate.ptr, self.up.ptr, self.act.ptr, self.block_size * self.intermediate, threads=256, stream=stream, library=self.library, runtime=self.runtime)
        dflash_dense_bf16_to_bf16(self.act.ptr, self.weights.tensor(f"{prefix}.mlp.down_proj.weight").ptr, self.mlp.ptr, self.block_size, self.intermediate, self.hidden, threads=128, stream=stream, library=self.library, runtime=self.runtime)
        dflash_add_bf16(self.hidden_attn.ptr, self.mlp.ptr, query_out.ptr, self.block_size * self.hidden, threads=256, stream=stream, library=self.library, runtime=self.runtime)
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
) -> tuple[list[int], dict[str, Any]]:
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
                if active_budget <= 0:
                    # No spec budget left for this cycle - one bare AR step on slot 0.
                    verify_rows_total += 1
                    t_verify = time.perf_counter()
                    result = _slot_step(
                        session,
                        root_token,
                        position=context_tokens,
                        slot=0,
                        drafter=drafter,
                        capture_row=context_tokens,
                    )
                    verify_seconds_total += time.perf_counter() - t_verify
                    generated.append(root_token)
                    root_token = int(result.token_id)
                    context_tokens += 1
                    continue
                verify_rows_total += 1 + active_budget
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
                verify_seconds_total += time.perf_counter() - t_verify
                accepted_lengths.append(accepted)
                committed = [root_token, *candidates[:accepted]]
                t_commit = time.perf_counter()
                drafter.commit_context_rows(start=context_tokens, count=len(committed))
                commit_seconds_total += time.perf_counter() - t_commit
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
    chain_attn_mode: str = "c1_loop",
    tree_mode: str = "chain",
) -> tuple[tuple[list[int], dict[str, Any]], tuple[list[int], dict[str, Any]]]:
    """Run AR control and DFlash chain in one resident target session.

    Slot 0 is reserved for the AR control.  Slot 1 is the DFlash committed
    state.  ``native_bulk_bplus1`` advances it through one root+B target forward
    per cycle; ``serial_in_place_single_slot`` remains a fallback.  The target
    weights/libraries/session are identical while the per-slot recurrent/KV
    state remains independent for exact token comparison.
    """

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
                chain_attn_mode=chain_attn_mode,
                tree_mode=tree_mode,
            )
        spec_meta["same_session_control"] = True
        spec_meta["same_process_control"] = True
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
    chain_attn_mode: str = "c1_loop",
    tree_mode: str = "chain",
) -> tuple[list[int], dict[str, Any]]:
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
    generated: list[int] = []
    accepted_lengths: list[int] = []
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
    t1 = time.perf_counter()
    state_copies = 0
    while len(generated) < decode_tokens:
        cycles += 1
        remaining = decode_tokens - len(generated)
        active_budget = min(candidate_budget, max(0, remaining - 1))
        if active_budget <= 0:
            verify_rows_total += 1
            t_verify = time.perf_counter()
            result = _slot_step(
                session,
                root_token,
                position=context_tokens,
                slot=base_slot,
                drafter=drafter,
                capture_row=context_tokens,
            )
            verify_seconds_total += time.perf_counter() - t_verify
            target_serial_forward_calls += 1
            finite_verify = finite_verify and math.isfinite(float(result.logit))
            generated.append(root_token)
            root_token = int(result.token_id)
            context_tokens += 1
            continue
        verify_rows_total += (1 + candidate_budget) if verifier_mode == "native_bulk_bplus1" else (1 + active_budget)
        draft = drafter.propose(root_token=root_token, root_position=context_tokens, context_tokens=context_tokens)
        candidates = list(draft.candidate_tokens[:active_budget])
        draft_calls += 1
        draft_tokens_proposed += active_budget
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
        t_verify = time.perf_counter()
        if verifier_mode == "native_bulk_bplus1":
            if tree_mode == "chain_as_tree":
                target_batch = _build_chain_as_tree_target_batch(
                    root_token=root_token,
                    root_position=context_tokens,
                    candidates=candidates,
                    candidate_budget=candidate_budget,
                    active_count=active_budget,
                )
                verify_result = session.verify_tree_bulk_and_commit(
                    target_batch,
                    base_slot=base_slot,
                    capture_layer_ids=drafter.config.target_layer_ids,
                    capture_hidden_concat=drafter.target_hidden_concat,
                    capture_row_start=context_tokens,
                )
            else:
                target_batch = _build_chain_target_batch(
                    root_token=root_token,
                    root_position=context_tokens,
                    candidates=candidates,
                    candidate_budget=candidate_budget,
                    active_count=active_budget,
                )
                verify_result = session.verify_chain_bulk_and_commit(
                    target_batch,
                    base_slot=base_slot,
                    capture_layer_ids=drafter.config.target_layer_ids,
                    capture_hidden_concat=drafter.target_hidden_concat,
                    capture_row_start=context_tokens,
                    graph_mode=verifier_graph_mode,
                    chain_attn_mode=chain_attn_mode,
                )
            target_top1 = list(verify_result.target_top1[: 1 + active_budget])
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
                finite_verify = finite_verify and math.isfinite(float(result.logit))
                target_top1.append(int(result.token_id))
                bonus = int(result.token_id)
        else:
            raise ValueError(f"unknown verifier_mode {verifier_mode!r}")
        verify_seconds_total += time.perf_counter() - t_verify
        accepted_lengths.append(accepted)
        committed = [root_token, *candidates[:accepted]]
        t_commit = time.perf_counter()
        drafter.commit_context_rows(start=context_tokens, count=len(committed))
        commit_seconds_total += time.perf_counter() - t_commit
        if len(proposal_trace) < 16:
            proposal_trace.append(
                {
                    "cycle": cycles,
                    "root_position": context_tokens,
                    "root_token": int(root_token),
                    "draft_candidates": [int(token) for token in candidates],
                    "target_top1_path": [int(token) for token in target_top1],
                    "accepted": int(accepted),
                    "committed_tokens": [int(token) for token in committed],
                    "bonus_token": int(bonus),
                    "verifier_mode": verifier_mode,
                    "drafter_graph_status": graph_status,
                    "drafter_graph_bucket": draft.graph.get("bucket_key"),
                }
            )
        generated.extend(committed)
        root_token = int(bonus)
        context_tokens += len(committed)
    decode_seconds = time.perf_counter() - t1
    metadata = {
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
            "notes": ["draft and verifier finite checks read top-1 ids/values only; full logits are not copied"],
        },
        "memory": memory_stats(),
        "backend": session.backend,
        "target_arch": session.target_arch,
        "verifier_mode": verifier_mode,
        "verifier_graph_mode": verifier_graph_mode,
        "verifier_chain_attn_mode": chain_attn_mode,
        "verifier_tree_mode": tree_mode,
        "native_bulk_verifier": verifier_mode == "native_bulk_bplus1",
        "drafter_context_mode": "append_only_projected_context_and_kv",
        "draft_phase_timing_mode": "synchronized" if drafter.sync_draft_phases else "enqueue_until_final_sync",
        "base_slot": base_slot,
        "branch_slot_start": branch_slot_start,
        "verifier_state_copies_per_cycle": 0,
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
        "config": {"name": f"full_model_chain_b{budget}", "provider": "dflash", "proposal_mode": "chain", "verify_mode": "verify_chain", "draft_budget": budget, "topk": 1},
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
            "target_forwards_per_draft_call": spec_meta.get("target_forwards_per_draft_call"),
            "gpu_accept_match_cpu": spec_meta.get("gpu_accept_match_cpu"),
            "verifier_graph": spec_meta.get("verifier_graph"),
            "draft_tokens_proposed": spec_meta.get("draft_tokens_proposed", spec_meta["draft_calls"] * budget),
            "draft_tokens": spec_meta.get("draft_tokens_proposed", spec_meta["draft_calls"] * budget),
            "accepted_draft_tokens": sum(int(x) for x in spec_meta["accepted_lengths"]),
            "accepted_lengths": spec_meta["accepted_lengths"],
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
    parser.add_argument("--full-attn-chain-mode", choices=("c1_loop", "batched"), default="c1_loop", help="Native B+1 verifier full-attention scheduling: c1_loop (per-row resident decode kernels, current default) or batched (one batched pass per layer using prefill primitives + c=1 MoE)")
    parser.add_argument(
        "--tree-mode",
        choices=("chain", "chain_as_tree"),
        default="chain",
        help=(
            "Verifier topology: chain (default) uses verify_chain_bulk_and_commit;"
            " chain_as_tree wraps the chain candidates as a degenerate (linear)"
            " tree and routes through verify_tree_bulk_and_commit -- same accept"
            " profile, isolates the tree kernel's overhead vs the chain batched"
            " path."
        ),
    )
    parser.add_argument("--drafter-graph", choices=("off", "auto", "validate"), default="off", help="Prototype HIP graph capture for native DFlash propose(); auto replays cache hits, validate records capture parity without requiring reuse")
    parser.add_argument("--drafter-fusion", choices=("off", "qkv"), default="off", help="Enable prototype DFlash drafter kernel fusions; qkv fuses query-side Q/K/V projections with unfused fallback available")
    parser.add_argument("--hardware-gpu", default=None, help="Human-readable GPU name to record in the benchmark artifact")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.decode_tokens <= 0:
        raise ValueError("--decode-tokens must be positive")
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8") if args.compiler_version_file else None
    target = Path(args.target_model)
    drafter = Path(args.drafter_model)
    validation = validate_dflash_artifact_pair(target_model=target, drafter_model=drafter, raise_on_error=True)
    prompts = _select_prompts(args.prompt_fixture, groups={x.strip() for x in args.prompt_groups.split(",") if x.strip()}, limit=args.max_prompts)
    budgets = [int(x) for x in args.draft_budgets.split(",") if x.strip()]
    prefill_config = PrefillConfig(auto_tune_chunk_sizes=True)
    rows: list[dict[str, Any]] = []
    commands = {"benchmark": " ".join(shlex.quote(part) for part in ["python3", "scripts/dflash_chain_e2e_bench.py", *(argv if argv is not None else sys.argv[1:])])}
    for prompt in prompts:
        prompt_ids = [int(x) for x in prompt["prompt_ids"]]
        for budget in budgets:
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
                chain_attn_mode=args.full_attn_chain_mode,
                tree_mode=args.tree_mode,
            )
            rows.append(_row_for_artifact(prompt, budget, ar, spec))
    artifact = build_speculative_artifact(
        run_tag="dflash-chain-full-model-e2e",
        summary="Full-model hipEngine DFlash chain E2E run with same-session AR control, native drafter, and serial in-place target verifier",
        rows=rows,
        models=SpeculativeBenchmarkModels(
            target_name=DEFAULT_TARGET_MODEL,
            target_path=str(target),
            target_revision=DEFAULT_TARGET_REVISION,
            drafter_name=DEFAULT_DFLASH_DRAFTER,
            drafter_path=str(drafter),
            drafter_revision=DEFAULT_DRAFTER_REVISION,
        ),
        status="diagnostic",
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware={"backend": rows[0]["spec"].get("backend") if rows else args.backend, "arch": rows[0]["spec"].get("target_arch") if rows else None, "gpu": args.hardware_gpu},
        software={**_git_context(), "python": platform.python_version(), "platform": platform.platform(), "hipcc_version": compiler_version},
        workload={
            "shape": "full_model_dflash_chain_e2e",
            "provider": "dflash",
            "verify_modes": ["verify_chain"],
            "draft_budgets": budgets,
            "decode_tokens": args.decode_tokens,
            "prompt_suite": str(args.prompt_fixture),
            "prompt_suite_sha256": file_sha256(args.prompt_fixture),
            "artifact_validation": validation,
            "verifier_mode": args.verifier_mode,
            "verifier_graph_mode": args.verifier_graph,
            "verifier_chain_attn_mode": args.full_attn_chain_mode,
            "verifier_tree_mode": args.tree_mode,
            "native_bulk_verifier": args.verifier_mode == "native_bulk_bplus1",
            "drafter_graph_mode": args.drafter_graph,
            "drafter_fusion_mode": args.drafter_fusion,
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
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "all_correctness_passed": artifact["measurements"]["aggregate"]["all_correctness_passed"], "speedup_vs_ar": artifact["measurements"]["aggregate"].get("speedup_vs_ar"), "performance_claim": artifact["performance_claim"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
