"""Qwen3.5 GGUF generation path."""

from __future__ import annotations

import concurrent.futures
import copy
import logging
import os
import socket
import sys
import threading
import time
import uuid
import weakref
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar, Iterator, Mapping, Sequence

import numpy as np

from hipengine.benchmark.provenance import collect_model_identity, detect_device_name
import hipengine.generation.qwen35_gguf_mtp2 as _qwen35_gguf_mtp2_module
from hipengine.core.dtype import DType
from hipengine.core.hip import HipError
from hipengine.core.memory import free, malloc, memory_stats
from hipengine.dispatch import (
    RequestState,
    SlotMove,
    WorkItem,
    WorkKind,
    plan_physical_batch_groups,
)
from hipengine.dispatch.d2_resolver import d2_partition
from hipengine.generation.batch_scheduler import (
    CompletedRequest,
    GeneratedToken,
    GeneratedTokenEvent,
    ResidentBatchScheduler,
)
from hipengine.generation.constraints import token_sequence_state_for_tokens
from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.engine_loop import (
    ExecutionFailure,
    GenerationAdmissionRejected,
)
from hipengine.generation.finish import finish_details_with_sampling_state
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    GenerationTelemetry,
    PromptInput,
    TokenLogprob,
    register_text_generator,
)
from hipengine.generation.sampling import (
    RowSamplingState,
    SamplingMode,
    plan_sampler,
    row_seed_for_index,
    select_token,
    supports_native_gpu_sampling,
    thinking_budget_state_from_params,
)
from hipengine.loading.gguf import GGUFModelInfo, GGUFReader
from hipengine.loading.qwen35_gguf import (
    FULL_ATTENTION,
    Qwen35GGUFConfig,
    qwen35_gguf_config_from_metadata,
)
from hipengine.loading.qwen35_gguf_nextn import required_qwen35_gguf_nextn_tensor_names
from hipengine.models.kv_capabilities import (
    KVCapabilityKey,
    KVCapabilityResolution,
    ModelArtifactIdentity,
    model_artifact_identity,
    resolve_kv_capability,
)
from hipengine.kvcache.pool import DeviceKVContiguityError
from hipengine.kvcache import (
    FixedPagedKVPolicy,
    RadixCache,
    resolve_kv_policy,
    resolve_prefix_cache_mode,
)
from hipengine.kernels.backends import (
    backend_package_capability,
    hip_target_arch_environment,
    hip_target_arch_for_backend,
    resolve_backend,
)
from hipengine.quant.gguf import dequantize_gguf_data
from hipengine.speculative.accounting import (
    accounting_timing_fields,
    record_provider_readiness,
    record_speculative_plan,
    speculative_output_accounting,
)
from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_gguf_runner import (
    PACKED_AR_PREFILL_CONTEXT_LIMIT,
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
    _GGUF_PACKED_WORKSPACE_LEASE_KEY,
    _GGUFResumablePrefillState,
    _PACKED_VERIFY_DEFAULT_SLOT_CAPACITY,
    _gguf_device_kv_contiguous_base_row,
    _gguf_gapped_slot_local_fast_route_available,
    _gguf_int8_bf16_full_attention_layer_indices,
    _gguf_packed_layer_outer_enabled,
    _admitted_no_mirror_int8_capability,
    _rope_tables as _gguf_rope_tables,
    estimate_qwen35_gguf_kv_capacity,
    packed_verify_workspace_lease_pages,
)
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

_LOGGER = logging.getLogger(__name__)


def _new_gguf_timing_batch_id(kind: str) -> str:
    return f"gguf-{str(kind)}-{uuid.uuid4().hex}"


_RESUMABLE_PREFILL_DONE = object()
"""Row marker: the resumable prefill produced its first token already."""


def _gguf_resumable_layer_count(session: Any) -> int:
    """Model layer count for scheduling a resumable prefill's layer budget."""

    config = getattr(
        getattr(getattr(session, "runner", None), "weights", None),
        "config",
        None,
    )
    layer_types = getattr(config, "layer_types", None)
    return 0 if layer_types is None else len(layer_types)


def packed_workspace_owner_inventory(
    sessions: Sequence[Any],
) -> tuple[int, int]:
    """Return (unique owner bytes, contributing session count).

    Resident slot views share one physical packed workspace through the batch
    owner: every view's ``packed_workspace_nbytes()`` reports the same buffers,
    so summing per session inflates the total by the view count (F6 in the
    server/direct parity roadmap: two views of one 1,024-byte allocation
    summed to 2,048). This helper deduplicates by buffer pointer across every
    session first, then sums. Growth owners (``full_attn_split_growth_buffers``)
    are included by the per-session helper.
    """

    unique: dict[int, int] = {}
    contributing = 0
    for session in sessions:
        size = getattr(session, "packed_workspace_nbytes", None)
        if not callable(size):
            continue
        contributing += 1
        for workspace in (
            getattr(session, "_packed_ar_attention_workspace", None),
            getattr(session, "_packed_verify_scratch", None),
            getattr(session, "_packed_verify_state", None),
        ):
            if workspace is None:
                continue
            for buffer in (
                *workspace.buffers,
                *(
                    getattr(workspace, "full_attn_split_growth_buffers", ())
                    if workspace is getattr(session, "_packed_verify_scratch", None)
                    else ()
                ),
            ):
                if buffer is None or int(buffer.ptr) == 0:
                    continue
                unique[int(buffer.ptr)] = int(buffer.nbytes)
    return sum(unique.values()), contributing


def prefill_transient_owner_inventory(sessions: Sequence[Any]) -> dict[str, Any]:
    """Owner-deduplicated prefill transient state for observability.

    Distinct accounting domains (F3/F4 in the server/direct parity roadmap):
    the per-layer/shared BF16 oracle owners, the bulk-prefill hidden/scratch
    owner, and the *actual* executor's last packed route - reported separately
    from the legacy INT8 lifetime plan mode, which the packed executor can
    override at the point of use (per-layer oracle keying) without the plan
    knowing.
    """

    oracle_buffers: dict[int, int] = {}
    hidden_buffers: dict[int, int] = {}
    oracle_owner_counts: list[int] = []
    oracle_capacity_positions: list[int] = []
    lifetime_modes: list[str] = []
    executor_routes: list[str] = []
    executor_modes: list[str] = []
    per_layer_flags: list[bool] = []
    kv_attention_sources: list[str | None] = []
    for session in sessions:
        # The session's own KV layout, read here because it gates whether the
        # resumable executor is even attempted (qwen35_gguf.py:8084). The last
        # execution manifest cannot answer this: by scrape time it is a decode
        # manifest, whose builder does not populate kv_attention_source.
        # ``kv_attention_source`` is a live-session property that raises for a
        # closed or partially constructed session; a telemetry scrape must
        # report that honestly as unknown rather than propagate the error.
        try:
            kv_attention_sources.append(session.kv_attention_source)
        except (RuntimeError, AttributeError):
            kv_attention_sources.append(None)
        oracle_buffers_by_session = getattr(
            session, "_int8_prefill_oracle_buffers", None
        )
        if isinstance(oracle_buffers_by_session, dict):
            oracle_owner_counts.append(len(oracle_buffers_by_session))
            for pair in oracle_buffers_by_session.values():
                try:
                    key_cache, value_cache = pair
                except (TypeError, ValueError):
                    continue
                for buffer in (key_cache, value_cache):
                    if buffer is not None and int(getattr(buffer, "ptr", 0)):
                        oracle_buffers[int(buffer.ptr)] = int(buffer.nbytes)
            capacity = getattr(session, "_int8_prefill_oracle_capacity_positions", None)
            if callable(capacity):
                try:
                    oracle_capacity_positions.append(int(capacity()))
                except (RuntimeError, AttributeError, ValueError):
                    pass
        plan = getattr(session, "_int8_prefill_lifetime_plan", None)
        lifetime_modes.append(str(getattr(plan, "mode", None)))
        last_plan = getattr(session, "last_packed_prefill_plan", None)
        executor_routes.append(
            str(last_plan.get("route")) if isinstance(last_plan, dict) else None
        )
        # The layer-outer executor records its distinction in executor_mode;
        # route alone cannot distinguish it from the chunk-outer fallback
        # (reviewer finding 4, 2026-09-10).
        executor_modes.append(
            str(last_plan.get("executor_mode")) if isinstance(last_plan, dict) else None
        )
        per_layer_flags.append(
            bool(getattr(session, "_int8_prefill_oracle_per_layer", False))
        )
        for buffer in (
            getattr(session, "_prefill_hidden_a", None),
            getattr(session, "_prefill_hidden_b", None),
            getattr(session, "_prefill_token_buf", None),
        ):
            if buffer is not None and int(getattr(buffer, "ptr", 0)):
                hidden_buffers[int(buffer.ptr)] = int(buffer.nbytes)
        bulk_scratch = getattr(session, "_bulk_prefill_scratch", None)
        if bulk_scratch is not None:
            for buffer in getattr(bulk_scratch, "buffers", ()):
                if buffer is not None and int(getattr(buffer, "ptr", 0)):
                    hidden_buffers[int(buffer.ptr)] = int(buffer.nbytes)
        # P6b: a suspended resumable prefill holds a dedicated copy of its live
        # hidden planes and linear state so interleaved packed decode cannot
        # overwrite it. That owner is real resident memory and must be counted
        # alongside the other prefill transients.
        suspended = getattr(session, "_resumable_prefill_scratch", None)
        if suspended is not None:
            for buffer in getattr(suspended, "buffers", ()):
                if buffer is not None and int(getattr(buffer, "ptr", 0)):
                    hidden_buffers[int(buffer.ptr)] = int(buffer.nbytes)
    return {
        "oracle_owner_bytes": sum(oracle_buffers.values()),
        "oracle_owner_counts": oracle_owner_counts,
        "oracle_owner_count_total": sum(oracle_owner_counts),
        "oracle_observed_peak_bytes": max(
            (
                int(getattr(session, "_int8_prefill_oracle_observed_peak_bytes", 0))
                for session in sessions
            ),
            default=0,
        ),
        "oracle_observed_peak_owners": max(
            (
                int(getattr(session, "_int8_prefill_oracle_observed_peak_owners", 0))
                for session in sessions
            ),
            default=0,
        ),
        "oracle_capacity_positions": oracle_capacity_positions,
        "hidden_and_bulk_owner_bytes": sum(hidden_buffers.values()),
        "int8_prefill_lifetime_plan_modes": lifetime_modes,
        "last_packed_executor_routes": executor_routes,
        "last_packed_executor_modes": executor_modes,
        "kv_attention_sources": kv_attention_sources,
        "oracle_per_layer_flags": per_layer_flags,
        "note": (
            "oracle_owner_bytes is the live per-layer/shared BF16 oracle pair"
            " total (unique buffers); oracle_capacity_positions is per-session"
            " pool-backed sizing, not prompt length; lifetime_plan_modes is the"
            " legacy plan while last_packed_executor_routes + oracle_per_layer"
            " report what actually ran"
        ),
    }


def _encode_prompt_timed(
    tokenizer: Any,
    prompt: PromptInput,
) -> tuple[list[int], float]:
    if not isinstance(prompt, str):
        return [int(token) for token in prompt], max(
            0.0,
            float(getattr(prompt, "tokenize_ms", 0.0)),
        )
    tokenize_started = time.perf_counter()
    token_ids = [int(token) for token in tokenizer.encode(prompt)]
    return token_ids, _timing_ms_since(tokenize_started)


def _encode_prompt(tokenizer: Any, prompt: PromptInput) -> list[int]:
    return _encode_prompt_timed(tokenizer, prompt)[0]


_LLAMA_COMPAT_MTP_ENV = {
    "HIPENGINE_GGUF_DECODE_REPACK": "1",
    "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1",
    "HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A": "1",
    "HIPENGINE_GGUF_T16_SELECTED_DP4A": "1",
    "HIPENGINE_GGUF_RAW_SELECTED_DP4A": "1",
    "HIPENGINE_GGUF_SELECTED_X8_REPACK": "q6",
    "HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A": "1",
    "HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE": "x8",
    "HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL": "1",
}
_GGUF_MTP_CONTEXT_REPLAY_MIN_PROMPT_TOKENS = 4
_MTP_SERVING_TARGET_BATCH_MAX_SLOTS = 4
_GGUF_AR_NATIVE_MAX_SLOTS = 8
# Resident request concurrency for the dense GGUF route when the caller does
# not set --max-active-requests. This is scheduler admission, not a KV-memory
# multiplier: the shared pool grows against its independent device budget.
# Pass --max-active-requests to change how many requests may be in flight.
_GGUF_RESIDENT_MODEL_LOOP_DEFAULT_CAPACITY = 4

# A retained prefix snapshot holds one full hybrid Conv/GDN state clone (~64 MiB
# for a 35B-A3B checkpoint) plus its KV pages. Retained boundaries are the
# cross-request working set: a conversation's next turn can only match a boundary
# that survived every intervening request's captures. The default count budget is
# the pre-fix effective value, max(1, capacity); a wider working set is opt-in
# because enabling it today is a measured regression (docs/REFACTOR.md). The
# 14-lane serving protocol needs 16 retained entries — one conversation's
# boundary must survive 13 intervening captures — and the byte budget bounds the
# state the wide set pins.
_PREFIX_RETAINED_STATE_BYTES_LIMIT = 1 << 30
_PREFIX_RETAINED_SNAPSHOTS_WIDE = 16
_GGUF_PREFIX_RETAINED_SNAPSHOTS_ENV = "HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS"
_GGUF_PREFIX_RETAINED_STATE_BYTES_ENV = "HIPENGINE_GGUF_PREFIX_RETAINED_STATE_BYTES"


def _gguf_prefix_retained_snapshot_limit(capacity: int) -> int:
    """Retained-snapshot count budget: pre-fix effective default, opt-in wider."""

    default = max(1, int(capacity))
    return _gguf_auto_context_int_env(
        _GGUF_PREFIX_RETAINED_SNAPSHOTS_ENV, default, minimum=default
    )


def _gguf_prefix_retained_state_bytes_limit() -> int:
    """State-byte ceiling for the retained snapshot working set."""

    return _gguf_auto_context_int_env(
        _GGUF_PREFIX_RETAINED_STATE_BYTES_ENV,
        _PREFIX_RETAINED_STATE_BYTES_LIMIT,
        minimum=1 << 20,
    )


_GGUF_PREFIX_GAPPED_SUFFIX_MAX_ENV = "HIPENGINE_GGUF_PREFIX_GAPPED_SUFFIX_MAX"
_GGUF_PREFIX_GAPPED_SUFFIX_MAX_DEFAULT = 512


def _gguf_prefix_gapped_suffix_max_tokens() -> int:
    """Largest suffix worth prefilling through a gapped hit on the slow route.

    Placement decides the route. A contiguous shared allocation keeps the
    slot-local prefill; a gapped one used to drop to the packed paged route,
    which was far slower per token and got worse as context grows (measured
    on Qwen3.5-0.8B against the full prefill the hit replaces: 2048+768 cost
    1.07x the miss, 4096+768 1.22x, while 2048+512 cost 0.63x and 4096+512
    0.87x). Past that the cheaper answer was to decline the hit and let the
    request take the fast private prefill.

    The gapped gather route changed the cost side: a gapped BF16 slot whose
    head-major KV buffers are admitted gathers into the same dense buffers
    the contiguous route uses and pays the same AOTriton attention cost, so
    the guard no longer applies to it (any suffix length wins). The budget
    below only binds when that fast gapped route is unavailable for the
    lease's backend/config - the gather kill-switch, a backend without
    head-major KV, or a context beyond the validated head-major allocation
    class - where the old paged-route costs still hold.

    A contiguous hit is never subject to this: it uses the same route the miss
    would have used, so it wins at any suffix length.
    """

    return _gguf_auto_context_int_env(
        _GGUF_PREFIX_GAPPED_SUFFIX_MAX_ENV,
        _GGUF_PREFIX_GAPPED_SUFFIX_MAX_DEFAULT,
        minimum=0,
    )


def _gguf_prefix_gapped_fast_route_available(lease: Any, context_tokens: int) -> bool:
    """Whether a gapped hit on this lease gathers onto the fast slot-local route."""

    session = getattr(lease, "session", None)
    runner = getattr(session, "runner", None)
    if runner is None:
        return False
    return _gguf_gapped_slot_local_fast_route_available(
        backend=str(getattr(runner, "backend", "")),
        context_tokens=int(context_tokens),
        kv_width=int(getattr(runner, "kv_width", 0)),
    )


_GGUF_PREFIX_BATCHED_SUFFIX_ENV = "HIPENGINE_GGUF_PREFIX_BATCHED_SUFFIX"


def _gguf_prefix_batched_suffix_enabled() -> bool:
    """Batched reused-suffix (\"extend\") prefill; the serial loop is the fallback.

    Default on: the packed paged prefill route accepts restored mid-sequence
    state at any context length. ``HIPENGINE_GGUF_PREFIX_BATCHED_SUFFIX=0``
    restores the serial ``session.step()`` suffix loop for rollback/bisection.
    """

    return os.environ.get(_GGUF_PREFIX_BATCHED_SUFFIX_ENV, "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _gguf_prefix_batched_suffix_chunk_eligible(chunk: tuple[int, ...]) -> bool:
    """Batch a reused suffix only when it is at least two rows wide.

    A one-row prefill does not take the bulk schedule: MoE has no rows==1 bulk
    path at all (``_run_post_attention_moe_rows`` refuses it) and the dense
    projections fall to their registered GEMV decode variants. Running a single
    suffix token through the batched route would therefore give a hit whose
    arithmetic differs from the wider chunk the same token sits in on a miss,
    for no speedup worth having - one token costs one step either way. The
    serial route handles it, which is also what this shape did before the
    batched route existed.
    """

    return len(chunk) >= 2


def _gguf_prefix_suffix_segments(
    session_position: int,
    prompt_length: int,
    chunk: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Split one reused-suffix chunk at the deepest prompt-aligned boundary.

    A boundary snapshot is only capturable while the session sits exactly on
    it, so a batched suffix prefill that would cross the deepest 256-aligned
    prompt boundary in one call runs as two batched segments with the capture
    in between.
    """

    boundary = (int(prompt_length) // 256) * 256
    start = int(session_position)
    end = start + len(chunk)
    if start < boundary < end:
        cut = boundary - start
        # A one-row segment would leave the bulk prefill schedule (see the
        # single-token-suffix decline in the admission path), so a split that
        # would strand a single token is not worth the mid-prefill snapshot.
        if cut >= 1 and len(chunk) - cut >= 2:
            return (chunk[:cut], chunk[cut:])
    return (chunk,)
# Superset of every shared-slot AR physical width a backend may register and use.
# Direct widths c3/c5/c6/c7 are admitted here so they can be certified via an
# explicit env override before the default advertised capability is expanded
# (see docs/reference/CONCURRENCY2.md). The default non-resident set stays (1, 2, 4, 8).
_GGUF_AR_PHYSICAL_BUCKET_WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8)
# Promoted 2026-08-20 after direct c3/c5/c6/c7 lifecycle certification (#36):
# every width in the superset is now an advertised default. The env override
# HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS remains for diagnostics.
_GGUF_AR_DEFAULT_PHYSICAL_WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8)


def _gguf_ar_physical_widths(
    backend: str | None = None,
    *,
    use_capability: bool = False,
) -> tuple[int, ...]:
    """Resolve the active shared-slot AR physical width set.

    An explicit ``HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS`` override
    (comma/space separated) widens or narrows the set for diagnostics and
    certification without changing the packaged production default. Otherwise
    the registered backend capability is used when ``use_capability`` is set
    (resident-batch owner), else the default advertised set. The result must be
    a sorted, strictly-increasing subset of
    ``_GGUF_AR_PHYSICAL_BUCKET_WIDTHS`` starting at c1.
    """
    override = os.environ.get(
        "HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", ""
    ).strip()
    if override:
        widths = tuple(int(item) for item in override.replace(",", " ").split())
    elif use_capability and backend is not None:
        widths = tuple(
            int(width)
            for width in backend_package_capability(
                backend, "GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", (1,)
            )
        )
    else:
        widths = _GGUF_AR_DEFAULT_PHYSICAL_WIDTHS
    if (
        not widths
        or widths[0] != 1
        or tuple(sorted(set(widths))) != widths
        or any(width not in _GGUF_AR_PHYSICAL_BUCKET_WIDTHS for width in widths)
    ):
        raise RuntimeError(
            "GGUF shared-slot physical widths must be sorted registered AR widths starting at c1"
        )
    return widths
_GGUFSessionPoolKey = tuple[
    str,
    bool | None,
    bool | None,
    int | None,
    int,
    tuple[str, str, str, str],
]
_GGUF_AR_PACKED_DECODE_ENV = "HIPENGINE_GGUF_AR_PACKED_DECODE"
_GGUF_AR_PACKED_PREFILL_ENV = "HIPENGINE_GGUF_AR_PACKED_PREFILL"
_GGUF_AR_STREAM_DECODE_ENV = "HIPENGINE_GGUF_AR_STREAM_DECODE"
_GGUF_AR_D2_COST_ARTIFACT_ENV = "HIPENGINE_GGUF_AR_D2_COST_ARTIFACT"
_GGUF_INT8_KV_DIAGNOSTIC_OVERRIDE_ENVS = (
    "HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED",
    "HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG",
)
_GGUF_DECODE_GRAPH_ENV = "HIPENGINE_GGUF_DECODE_GRAPH"
_GGUF_MTP_SERVER_PACKED_PREFILL_ENV = "HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL"
_GGUF_MTP_HOT_VOCAB_ENV = "HIPENGINE_GGUF_MTP_HOT_VOCAB"
_GGUF_SPECDEC2_STREAMING_PROMPT_ENV = "HIPENGINE_GGUF_SPECDEC2_STREAMING_PROMPT"
_GGUF_MTP_SERVER_STARTUP_WARMUP_ENV = "HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP"
_GGUF_MTP_SERVER_STREAM_DRAFT_ENV = "HIPENGINE_GGUF_MTP_SERVER_STREAM_DRAFT"
_GGUF_MTP_SERVER_STREAM_VERIFY_ENV = "HIPENGINE_GGUF_MTP_SERVER_STREAM_VERIFY"
_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER_ENV = "HIPENGINE_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER"
_GGUF_MTP_SERVER_VERIFY_MODE_ENV = "HIPENGINE_GGUF_MTP_VERIFY_MODE"
_GGUF_MTP_SERVER_CANDIDATE_BUDGET_ENV = "HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET"
# Provider prefix checkpoints: one entry per captured prefix, each holding a
# copy of that prefix's provider KV and recurrent state. Bounded capture keeps
# MTP usable after a target radix hit; zero remains the memory/bisection rollback.
_GGUF_MTP2_PREFIX_CHECKPOINT_ENTRIES_ENV = "HIPENGINE_MTP2_PREFIX_CHECKPOINT_ENTRIES"
_GGUF_MTP2_PREFIX_CHECKPOINT_DEFAULT_ENTRIES = 4
# The prefix cache's block granularity. The trie stores whole blocks and the
# snapshot path already keys on this same multiple, so the capture boundary is
# the last full block rather than the prompt's end.
_GGUF_PREFIX_CACHE_BLOCK_TOKENS = 256
_GGUF_MTP_SERVER_DEFAULT_VERIFY_MODE = "native"
_GGUF_MTP_SERVER_DEFAULT_CANDIDATE_BUDGET = 3

# Automatic resident context sizing.
#
# When the caller does not pin a context, the resident GGUF session prices the
# footprint against free HIP memory after weights load and takes the largest
# block-aligned context that fits with a safety reserve. Set
# HIPENGINE_GGUF_AUTO_CONTEXT=0 to keep the historical fixed default instead.
_GGUF_AUTO_CONTEXT_ENV = "HIPENGINE_GGUF_AUTO_CONTEXT"
_GGUF_AUTO_CONTEXT_RESERVE_MIB_ENV = "HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB"
_GGUF_AUTO_CONTEXT_TRANSIENT_KIB_ENV = "HIPENGINE_GGUF_KV_TRANSIENT_KIB_PER_TOKEN"
_GGUF_AUTO_CONTEXT_TRANSIENT_FIXED_MIB_ENV = "HIPENGINE_GGUF_KV_TRANSIENT_FIXED_MIB"
_GGUF_AUTO_CONTEXT_ATTEMPTS_ENV = "HIPENGINE_GGUF_AUTO_CONTEXT_ATTEMPTS"
_GGUF_AUTO_CONTEXT_BLOCK_SIZE = 256
_GGUF_AUTO_CONTEXT_MAX_ATTEMPTS = 4
_GGUF_AUTO_CONTEXT_FALLBACK_NUMERATOR = 3
_GGUF_AUTO_CONTEXT_FALLBACK_DENOMINATOR = 4

# The reserve covers device memory the capacity model does not price at all:
# HIP context, JIT-compiled kernel modules, AOTriton, and the KV pool's pointer
# tables. Those are allocated lazily *after* the free-memory reading the model
# prices against, so they cannot be inferred from it.
#
# Measured on the W7900 (48 GiB), 27B Q4_K_M, auto-selected 45,568 tokens at 4
# BF16 slots, after a 43,011-token prefill: whole-card use 42.91 GiB against
# 40.25 GiB of hipEngine-tracked allocations, i.e. **2.66 GiB untracked**, of
# which about 2.35 GiB materializes after the selection is made. The previous
# 512 MiB default did not cover that; the auto-selection only survived a
# full-depth prompt because the transient term is priced at its worst case
# (4.22 GiB) against a measured 0.30 GiB, and that 3.9 GiB of slack absorbed
# the shortfall. On a 24 GiB card the same arithmetic selected a context whose
# true peak was 26.2 GiB against 23.98 GiB of VRAM.
_GGUF_AUTO_CONTEXT_RESERVE_MIB_DEFAULT = 3072


def _gguf_auto_context_enabled() -> bool:
    return os.environ.get(_GGUF_AUTO_CONTEXT_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _gguf_auto_context_int_env(name: str, default: int, *, minimum: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return max(int(minimum), int(raw))
    except ValueError:
        return int(default)


def _gguf_auto_context_attempts() -> int:
    return _gguf_auto_context_int_env(
        _GGUF_AUTO_CONTEXT_ATTEMPTS_ENV,
        _GGUF_AUTO_CONTEXT_MAX_ATTEMPTS,
        minimum=1,
    )


def _gguf_auto_context_reserve_bytes() -> int:
    return _gguf_auto_context_int_env(
        _GGUF_AUTO_CONTEXT_RESERVE_MIB_ENV,
        _GGUF_AUTO_CONTEXT_RESERVE_MIB_DEFAULT,
        minimum=0,
    ) * 1024**2


def _gguf_auto_context_transient_bytes_per_token() -> int:
    return _gguf_auto_context_int_env(
        _GGUF_AUTO_CONTEXT_TRANSIENT_KIB_ENV,
        74,
        minimum=0,
    ) * 1024


def _gguf_auto_context_transient_fixed_bytes() -> int:
    return _gguf_auto_context_int_env(
        _GGUF_AUTO_CONTEXT_TRANSIENT_FIXED_MIB_ENV,
        1024,
        minimum=0,
    ) * 1024**2


def _gguf_allocation_failure(exc: BaseException) -> bool:
    """Return whether an exception is a device-allocation failure.

    HIP reports out-of-memory as ``HipError``; the host-side and pool paths
    raise ``MemoryError``. Both mean the same thing to the auto-context
    fallback, and neither should be swallowed for any other failure.
    """

    if isinstance(exc, MemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "oom" in message

_GGUF_PUBLIC_USE_WMMA_PREFILL = True
_GGUF_PUBLIC_USE_GEMV_DECODE = True
from hipengine.runtime.gguf_linear import (
    mtp_serving_target_use_wmma_prefill as _mtp_serving_target_use_wmma_prefill,
    prefill_f16_staging_for as _prefill_f16_staging_for_profile,
    q6_integer_mmq_for as _q6_integer_mmq_for_profile,
)


def _prefill_f16_staging_for(generator: object) -> bool:
    """Profile-scoped B2 prefill activation staging for one generator."""

    return _prefill_f16_staging_for_profile(
        getattr(generator, "execution_profile", None),
        profile_fell_back_to_strict=bool(
            getattr(generator, "execution_profile_fell_back_to_strict", True)
        ),
    )


def _q6_integer_mmq_for(generator: object) -> bool:
    """Profile-scoped B5 integer-MMQ resolution for one generator."""

    return _q6_integer_mmq_for_profile(
        getattr(generator, "execution_profile", None),
        profile_fell_back_to_strict=bool(
            getattr(generator, "execution_profile_fell_back_to_strict", True)
        ),
    )


def _mtp_serving_target_wmma_for(generator: object) -> bool:
    """Profile-scoped B1 transfer resolution for one generator."""

    return _mtp_serving_target_use_wmma_prefill(
        getattr(generator, "execution_profile", None),
        profile_fell_back_to_strict=bool(
            getattr(generator, "execution_profile_fell_back_to_strict", True)
        ),
    )

# Diagnostic escape hatch for the resident sessions' prefill route (docs/REFACTOR.md, "All-GEMV
# small-row prefill A/B"). Both shipping call sites below passed `use_wmma_prefill=True` literally,
# and `HIPENGINE_GGUF_WMMA_PREFILL` is opt-in-only, so no bench or diagnostic could take the WMMA
# prefill route away - `setattr` misses it too because the constant is bound as a default argument.
# This resolver is read at session-acquire time (once per session, not per token) and is a
# *diagnostic* only: unset means the production route, unchanged.
_GGUF_DIAGNOSTIC_WMMA_PREFILL_ENV = "HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL"
_WMMA_PREFILL_FALSY = frozenset({"0", "false", "no", "off"})
_WMMA_PREFILL_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _resident_session_wmma_prefill_default() -> bool:
    """Return the prefill route for the resident AR / MTP-target sessions, default shipped True.

    Raises on an unrecognised value rather than falling back: a typo silently reading as "route
    unchanged" is how a route A/B becomes a null result with no clue why (measured the hard way on
    2026-08-30 with HIPENGINE_GGUF_Q4K_ROWTILE, whose value the server session overrode).
    """
    raw = (os.environ.get(_GGUF_DIAGNOSTIC_WMMA_PREFILL_ENV) or "").strip().lower()
    if not raw:
        return True
    if raw in _WMMA_PREFILL_TRUTHY:
        return True
    if raw in _WMMA_PREFILL_FALSY:
        return False
    raise ValueError(
        f"invalid {_GGUF_DIAGNOSTIC_WMMA_PREFILL_ENV}={raw!r}; expected a boolean in "
        f"{sorted(_WMMA_PREFILL_TRUTHY)} or {sorted(_WMMA_PREFILL_FALSY)}"
    )


def _target_arch_scoped(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with hip_target_arch_environment(self.target_arch):
            return method(self, *args, **kwargs)

    return wrapper


def _target_arch_scoped_stream(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with hip_target_arch_environment(self.target_arch):
            yield from method(self, *args, **kwargs)

    return wrapper


def _gguf_ar_packed_decode_enabled() -> bool:
    return os.environ.get(_GGUF_AR_PACKED_DECODE_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


_GGUF_AR_D2_COST_CACHE: dict[tuple[object, ...], object] = {}


def _gguf_ar_resolve_cost_table(
    backend: str,
    *,
    target_arch: str,
    model_path: str | Path,
    quant: str,
    kv_dtype: str,
    physical_widths: Sequence[int],
) -> object | None:
    """Resolve an explicitly configured clean, exact-identity D2 cost map.

    D2 remains opt-in until the actual server passes the c1-c32 route,
    goodput/TTFT/ITL, dynamic lifecycle, memory, and final-drain gate. An absent
    setting returns ``None`` so the production owner uses the ceiling planner;
    an explicit invalid artifact raises.
    """

    raw_path = os.environ.get(_GGUF_AR_D2_COST_ARTIFACT_ENV, "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"D2 cost artifact does not exist: {path}")
    stat = path.stat()
    fingerprint = collect_model_identity(model_path)["fingerprint"]
    if not isinstance(fingerprint, Mapping) or fingerprint.get("exists") is not True:
        raise ValueError("D2 cost resolution requires a readable model fingerprint")
    device_name = detect_device_name()
    if not device_name:
        raise ValueError("D2 cost resolution requires the current HIP device identity")
    expected = {
        "backend": str(backend),
        "target_arch": str(target_arch),
        "host_name": socket.gethostname(),
        "device_name": device_name,
        "model_fingerprint": str(fingerprint["value"]),
        "quant": str(quant),
        "kv_dtype": str(kv_dtype),
        "execution_profile": "strict",
        "graph_mode": "captured_replay",
        "physical_widths": [int(width) for width in physical_widths],
    }
    cache_key = (
        str(path),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        tuple(
            (key, tuple(value) if isinstance(value, list) else value)
            for key, value in expected.items()
        ),
    )
    if cache_key in _GGUF_AR_D2_COST_CACHE:
        return _GGUF_AR_D2_COST_CACHE[cache_key]
    from hipengine.dispatch.d2_resolver import cost_table_from_artifact

    cost_table = cost_table_from_artifact(path, expected=expected)
    _GGUF_AR_D2_COST_CACHE.clear()
    _GGUF_AR_D2_COST_CACHE[cache_key] = cost_table
    return cost_table


def _gguf_ar_packed_prefill_enabled() -> bool:
    return os.environ.get(_GGUF_AR_PACKED_PREFILL_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_single_row_block_table_prefill_required(session: object) -> bool:
    """Use one prefill route for direct INT8 and all shifted KV allocations."""

    return bool(
        getattr(session, "kv_attention_source", None) == "int8_direct"
        or _gguf_device_kv_contiguous_base_row(session) != 0
    )


def _gguf_ar_stream_decode_enabled() -> bool:
    return os.environ.get(_GGUF_AR_STREAM_DECODE_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_decode_graph_enabled() -> bool:
    return os.environ.get(_GGUF_DECODE_GRAPH_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_mtp_hot_vocab_setting(execution_profile: str) -> str | None:
    """Resolve packaged production default, explicit map, or full-head rollback."""

    raw = os.environ.get(_GGUF_MTP_HOT_VOCAB_ENV)
    if raw is None:
        return "auto" if str(execution_profile) == "production" else None
    value = raw.strip()
    if value.lower() in {"", "0", "false", "off", "none"}:
        return None
    return value


def _gguf_mtp_server_packed_prefill_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_PACKED_PREFILL_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_specdec2_streaming_prompt_enabled() -> bool:
    return os.environ.get(
        _GGUF_SPECDEC2_STREAMING_PROMPT_ENV,
        "1",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _gguf_mtp_server_startup_warmup_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_STARTUP_WARMUP_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_mtp_server_stream_draft_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_STREAM_DRAFT_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_mtp_server_stream_verify_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_STREAM_VERIFY_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_mtp_server_defer_verify_scatter_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER_ENV, "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _gguf_mtp_server_target_verify_mode() -> str:
    """Return the dense MTP target verify mode for server serving.

    Defaults to ``native``: the fast llama.cpp-style native row-attention / GPU
    accept path validated on the dense MTP suites (``all_gpu_accept_match_cpu``
    with rare sub-token-level argmax differences vs AR).  ``serial_exact`` is
    the conservative rollback control that re-runs exact c=1 AR per candidate
    row; it is token-exact against AR but cannot beat AR decode speed.
    """

    raw = os.environ.get(_GGUF_MTP_SERVER_VERIFY_MODE_ENV, _GGUF_MTP_SERVER_DEFAULT_VERIFY_MODE)
    mode = str(raw).strip().lower().replace("-", "_")
    if mode not in {"serial_exact", "native"}:
        return _GGUF_MTP_SERVER_DEFAULT_VERIFY_MODE
    return mode


def _trace_mtp2_resolution(reason: str) -> None:
    """Echo why the MTP2 adapter did not resolve, under an opt-in env flag."""

    if os.environ.get("HIPENGINE_MTP2_TRACE_DECLINE", "").strip() not in {"", "0"}:
        print(f"[mtp2-resolve] {reason}", file=sys.stderr, flush=True)


def _gguf_mtp_server_candidate_budget() -> int:
    """Return the dense MTP candidate budget for server serving (default 3)."""

    raw = os.environ.get(_GGUF_MTP_SERVER_CANDIDATE_BUDGET_ENV, "")
    if raw is None or str(raw).strip() == "":
        return _GGUF_MTP_SERVER_DEFAULT_CANDIDATE_BUDGET
    try:
        budget = int(str(raw).strip())
    except ValueError:
        return _GGUF_MTP_SERVER_DEFAULT_CANDIDATE_BUDGET
    if not 1 <= budget <= _qwen35_gguf_mtp2_module.MTP2_MAX_CANDIDATE_DEPTH:
        return _GGUF_MTP_SERVER_DEFAULT_CANDIDATE_BUDGET
    return budget


def _gguf_mtp2_prefix_checkpoint_entries() -> int:
    """Bound provider prefix checkpoints; zero disables capture explicitly."""

    raw = os.environ.get(_GGUF_MTP2_PREFIX_CHECKPOINT_ENTRIES_ENV, "")
    if raw is None or str(raw).strip() == "":
        return _GGUF_MTP2_PREFIX_CHECKPOINT_DEFAULT_ENTRIES
    try:
        entries = int(str(raw).strip())
    except ValueError:
        return _GGUF_MTP2_PREFIX_CHECKPOINT_DEFAULT_ENTRIES
    if entries < 0:
        return _GGUF_MTP2_PREFIX_CHECKPOINT_DEFAULT_ENTRIES
    return entries


@dataclass(frozen=True)
class _GGUFMTPServingAssets:
    weights: dict[str, tuple[np.ndarray, int, tuple[int, ...]]]
    token_embd_f32: np.ndarray
    rope_cos: np.ndarray
    rope_sin: np.ndarray
    config: Qwen35GGUFConfig | None = None
    nextn_block_id: int = 40


@dataclass(frozen=True)
class _GGUFMTPServingRun:
    generated_ids: list[int]
    cycles: list[dict[str, Any]]
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class _GGUFMTPServingSlot:
    request_id: int
    prompt_ids: list[int]
    session: Qwen35GGUFResidentSession
    resident_draft: Any
    resident_context: Any
    mtp_key_cache: Any
    mtp_value_cache: Any
    mtp_buffers: list[Any]
    hidden_size: int
    prev_token: int
    seq_position: int
    generated_ids: list[int]
    cycles: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    session_pool_key: _GGUFSessionPoolKey | None = None
    draft_pool_key: Any | None = None
    mtp_device_kv_len: int = 0
    draft_stream: int = 0
    verify_stream: int = 0
    done: bool = False


@dataclass
class _GGUFARServingSlot:
    request_id: int
    prompt_ids: list[int]
    session: Qwen35GGUFResidentSession
    prev_token: int
    seq_position: int
    generated_ids: list[int]
    timing: dict[str, float] = field(default_factory=dict)
    session_pool_key: _GGUFSessionPoolKey | None = None
    done: bool = False
    native_compact_prefill: bool = False
    native_decode_steps: int = 0
    native_c1_decode_steps: int = 0
    serial_decode_steps: int = 0
    decode_stream: int = 0
    c1_decode_graph: Any | None = None
    packed_decode_graph: Any | None = None
    packed_decode_graph_unavailable: bool = False
    packed_decode_owner: Any | None = None


@dataclass
class _GGUFMTPDraftedCycle:
    slot: _GGUFMTPServingSlot
    advance_start: float
    cycle_mtp_kv_base_len: int
    draft_tokens: list[int]
    block_inputs: list[int]
    block_start: int
    direct_commit_exact: bool
    snapshot: Any | None = None


@dataclass
class _GGUFMTPVerifiedCycle:
    drafted: _GGUFMTPDraftedCycle
    block_result: Any
    block_target_tokens: list[int]
    acceptance: dict[str, Any]


@contextmanager
def _temporary_env(updates: dict[str, str]):
    previous = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            os.environ[name] = value
        yield previous
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _exact_env(values: dict[str, str | None]):
    previous = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _gguf_mtp_required_tensor_names(
    info: GGUFModelInfo,
) -> tuple[Qwen35GGUFConfig, int, tuple[str, ...]]:
    """Resolve the one trailing NextN block from the GGUF architecture."""

    config = qwen35_gguf_config_from_metadata(info)
    if len(config.ignored_block_ids) != 1:
        raise ValueError("GGUF speculative MTP requires exactly one trailing NextN block")
    block_id = int(config.ignored_block_ids[0])
    required = required_qwen35_gguf_nextn_tensor_names(block_id, config=config)
    root_names = ("token_embd.weight", config.lm_head_tensor_name)
    return config, block_id, tuple(dict.fromkeys((*root_names, *required)))


def _gguf_info_has_mtp_tensors(info: Any) -> bool:
    try:
        _config, _block_id, required = _gguf_mtp_required_tensor_names(info)
        by_name = {tensor.name for tensor in info.tensors}
    except Exception:
        return False
    return all(name in by_name for name in required)


def _timing_ms_since(start: float) -> float:
    return round(max(0.0, time.perf_counter() - start) * 1000.0, 3)


def _timing_add(timing: dict[str, float], key: str, start: float) -> None:
    timing[key] = round(float(timing.get(key, 0.0)) + _timing_ms_since(start), 3)


def _timing_add_ms(timing: dict[str, float], key: str, ms: float) -> None:
    timing[key] = round(float(timing.get(key, 0.0)) + max(0.0, float(ms)), 3)


def _timing_set(timing: dict[str, float], key: str, start: float) -> None:
    timing[key] = _timing_ms_since(start)


_LLAMA_COMPAT_DIRECT_CYCLE_MODES = frozenset(
    {
        "llama_compat_direct_commit",
        "llama_compat_native_complete_cycle",
    }
)


def _mtp_cycle_summary(cycles: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, Any]:
    direct_cycles = [
        cycle
        for cycle in cycles
        if str(cycle.get("mode", "")) in _LLAMA_COMPAT_DIRECT_CYCLE_MODES
    ]
    accept_hist: dict[str, int] = {}
    shape_hist: dict[str, int] = {}
    full_accept_cycles = 0
    partial_accept_cycles = 0
    reject_cycles = 0
    target_rows = 0
    linear_state_commit_rows = 0
    hidden_seed_rows_needed = 0
    for cycle in direct_cycles:
        generated = max(0, int(cycle.get("generated_draft_tokens", 0)))
        accepted = max(0, int(cycle.get("accepted_draft_tokens", 0)))
        rows = generated + 1
        target_rows += rows
        linear_state_commit_rows += 1
        hidden_seed_rows_needed += min(accepted + 1, rows)
        accept_key = str(accepted)
        accept_hist[accept_key] = accept_hist.get(accept_key, 0) + 1
        shape_key = f"draft{generated}_accept{accepted}"
        shape_hist[shape_key] = shape_hist.get(shape_key, 0) + 1
        if accepted <= 0:
            reject_cycles += 1
        elif accepted >= generated:
            full_accept_cycles += 1
        else:
            partial_accept_cycles += 1
    direct_count = len(direct_cycles)
    return {
        "direct_cycles": direct_count,
        "full_accept_cycles": full_accept_cycles,
        "partial_accept_cycles": partial_accept_cycles,
        "reject_cycles": reject_cycles,
        "full_accept_rate": (
            float(full_accept_cycles) / float(direct_count)
            if direct_count > 0
            else 0.0
        ),
        "accepted_draft_tokens_histogram": dict(sorted(accept_hist.items())),
        "cycle_shape_histogram": dict(sorted(shape_hist.items())),
        "linear_state_captured_rows": target_rows,
        "linear_state_commit_rows": linear_state_commit_rows,
        "linear_state_extra_rows": max(0, target_rows - linear_state_commit_rows),
        "hidden_seed_captured_rows": target_rows,
        "hidden_seed_needed_rows": hidden_seed_rows_needed,
        "hidden_seed_extra_rows": max(0, target_rows - hidden_seed_rows_needed),
    }


def _add_mtp_cycle_timing_metrics(
    timing: dict[str, float],
    cycles: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> None:
    cycle_rows = list(cycles)
    draft_tokens = sum(int(cycle.get("generated_draft_tokens", 0)) for cycle in cycle_rows)
    accepted_tokens = sum(int(cycle.get("accepted_draft_tokens", 0)) for cycle in cycle_rows)
    visible_tokens = sum(int(cycle.get("visible_output_tokens", 0)) for cycle in cycle_rows)
    target_rows = sum(
        int(cycle.get("generated_draft_tokens", 0)) + 1
        for cycle in cycle_rows
        if str(cycle.get("mode", "")) in _LLAMA_COMPAT_DIRECT_CYCLE_MODES
    )
    timing["mtp_cycles_count"] = float(len(cycle_rows))
    timing["mtp_generated_draft_tokens"] = float(draft_tokens)
    timing["mtp_accepted_draft_tokens"] = float(accepted_tokens)
    timing["mtp_visible_output_tokens"] = float(visible_tokens)
    timing["mtp_target_verify_rows"] = float(target_rows)
    timing["mtp_accept_per_draft"] = (
        float(accepted_tokens) / float(draft_tokens)
        if draft_tokens > 0
        else 0.0
    )
    summary = _mtp_cycle_summary(cycle_rows)
    timing["mtp_direct_cycles_count"] = float(summary["direct_cycles"])
    timing["mtp_full_accept_cycles"] = float(summary["full_accept_cycles"])
    timing["mtp_partial_accept_cycles"] = float(summary["partial_accept_cycles"])
    timing["mtp_reject_cycles"] = float(summary["reject_cycles"])
    timing["mtp_full_accept_rate"] = float(summary["full_accept_rate"])
    timing["mtp_linear_state_captured_rows"] = float(summary["linear_state_captured_rows"])
    timing["mtp_linear_state_commit_rows"] = float(summary["linear_state_commit_rows"])
    timing["mtp_linear_state_extra_rows"] = float(summary["linear_state_extra_rows"])
    timing["mtp_hidden_seed_captured_rows"] = float(summary["hidden_seed_captured_rows"])
    timing["mtp_hidden_seed_needed_rows"] = float(summary["hidden_seed_needed_rows"])
    timing["mtp_hidden_seed_extra_rows"] = float(summary["hidden_seed_extra_rows"])


def _llama_cpp_mtp_catchup_rows(
    prompt_tokens: list[int] | tuple[int, ...],
    prompt_hidden_seeds: np.ndarray,
) -> tuple[list[int], np.ndarray]:
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


def _llama_cpp_acceptance_from_target_samples(
    draft_tokens: list[int],
    target_samples: list[int],
) -> dict[str, object]:
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
        raise ValueError("target_samples must include the corrective target token")
    output_tokens = targets[:accepted] + [targets[accepted]]
    return {
        "accepted_draft_tokens": accepted,
        "visible_output_tokens": len(output_tokens),
        "output_tokens": output_tokens,
        "pending_hidden_row_index": accepted,
    }


def _new_mtp_context(target_session: Any, *, token_id: int, position: int, mtp_block: Any):
    from hipengine.speculative.gguf_mtp import Qwen35GGUFMTPContext

    return Qwen35GGUFMTPContext.from_target_seed(
        target_session,
        token_id=int(token_id),
        position=int(position),
        mtp_block=mtp_block,
    )


def _new_mtp_seed_row(
    *,
    token_id: int,
    position: int,
    hidden_ptr: int,
    hidden_size: int,
    source: str,
):
    from hipengine.speculative.gguf_mtp import Qwen35GGUFMTPSeedRow

    return Qwen35GGUFMTPSeedRow(
        token_id=int(token_id),
        position=int(position),
        hidden_ptr=int(hidden_ptr),
        hidden_size=int(hidden_size),
        source=str(source),
    )


def _new_mtp_draft_runner(
    assets: _GGUFMTPServingAssets,
    *,
    runtime: Any,
    require_cached_build: bool = False,
):
    from hipengine.speculative.mtp_resident_draft import Qwen35GGUFResidentMTPDraftRunner

    return Qwen35GGUFResidentMTPDraftRunner(
        assets.weights,
        assets.token_embd_f32,
        runtime=runtime,
        vocab_cap=int(assets.weights["output.weight"][0].shape[0]),
        device_chain_enabled=True,
        prewarm_device_chain=True,
        require_cached_build=bool(require_cached_build),
    )


def _allocate_mtp_dense_kv(
    *,
    runtime: Any,
    capacity: int,
    qk_head_dim: int,
    kv_heads: int = 2,
) -> tuple[Any, Any, list[Any]]:
    from hipengine.core.memory import malloc

    rows = int(capacity)
    key_nbytes = rows * int(kv_heads) * int(qk_head_dim) * 4
    value_nbytes = key_nbytes
    key_cache = malloc(key_nbytes, runtime=runtime)
    value_cache = malloc(value_nbytes, runtime=runtime)
    return key_cache, value_cache, [key_cache, value_cache]


def _free_mtp_buffers(buffers: list[Any], *, runtime: Any) -> None:
    from hipengine.core.memory import free

    for buffer in reversed(buffers):
        free(buffer, runtime=runtime)


@dataclass(frozen=True)
class _GGUFNativeBatchRun:
    generated_ids: dict[int, list[int]]
    native_decode_steps: int
    execution_paths: dict[str, str]
    scheduling: dict[str, Any]


@dataclass
class Qwen35GGUFBringupGenerator:
    """Public API GGUF greedy generator over a persistent resident session."""

    model_path: str | Path
    weight_index: GGUFModelInfo
    model_plugin: Any
    backend: str = "auto"
    bulk_prefill_attention_mode: str = "bulk"
    prefill_quant: str | None = None
    prefill_attn_aotriton_min_tokens: int | None = None
    native_batch_decode: bool = False
    native_batch_capacity: int = 8
    # Independent of model execution_profile: explicit strict debugging opt-out.
    native_sampler_algorithm: str = "sorted"
    engine_loop_config_defaults: Mapping[str, Any] = field(default_factory=dict, repr=False)
    server_plain_ar_max_active_requests: int | None = None
    server_plain_ar_max_active_requests_by_max_sequence_length: Mapping[int, int] = field(
        default_factory=dict,
        repr=False,
    )
    tokenizer: Qwen35GGUFTokenizer = field(init=False)
    last_batch_generation: dict[str, Any] | None = field(default=None, init=False, repr=False)
    last_generation_outputs: tuple[GenerationOutput, ...] = field(default=(), init=False, repr=False)
    _mtp_serving_assets: _GGUFMTPServingAssets | None = field(default=None, init=False, repr=False)
    _mtp_serving_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _shared_runner: Qwen35GGUFFullStackRunner | None = field(default=None, init=False, repr=False)
    _shared_runner_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _prepared_max_sequence_length: int | None = field(default=None, init=False, repr=False)
    _prepared_kv_policy: FixedPagedKVPolicy | None = field(default=None, init=False, repr=False)
    _prepared_kv_scale_dtype: str = field(default="fp16", init=False, repr=False)
    _prepared_kv_signature: tuple[str, str, str, str] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _kv_capability_resolution: KVCapabilityResolution | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _auto_resolved_max_sequence_length: int | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _auto_resolved_max_sequence_lengths: dict[tuple[int, bool], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _auto_context_estimate: Any | None = field(default=None, init=False, repr=False)
    _resident_model_runner: Any | None = field(default=None, init=False, repr=False)
    _kv_artifact_identity: ModelArtifactIdentity | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _shared_session_pool: dict[
        _GGUFSessionPoolKey,
        list[Qwen35GGUFResidentSession],
    ] = field(default_factory=dict, init=False, repr=False)
    _shared_session_pool_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _shared_mtp_draft_pool: dict[Any, list[Any]] = field(default_factory=dict, init=False, repr=False)
    _shared_mtp_draft_pool_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    supports_stream_logprobs: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.bulk_prefill_attention_mode not in {"bulk", "native"}:
            raise ValueError("bulk_prefill_attention_mode must be 'bulk' or 'native'")
        if (
            self.prefill_attn_aotriton_min_tokens is not None
            and int(self.prefill_attn_aotriton_min_tokens) < 0
        ):
            raise ValueError("prefill_attn_aotriton_min_tokens must be non-negative")
        if int(self.native_batch_capacity) < 2 or int(self.native_batch_capacity) > 8:
            raise ValueError("native_batch_capacity must be within [2, 8]")
        self.backend = resolve_backend(self.backend)
        self.tokenizer = Qwen35GGUFTokenizer.from_gguf_info(self.weight_index)
        self._defer_resident_session_policy_resolution = True

    @property
    def target_arch(self) -> str:
        backend = self.__dict__.get("backend")
        if backend is None:
            # Lightweight scheduling tests intentionally bypass the dataclass
            # initializer with ``__new__``. Keep that non-runtime fixture on
            # the historical source target; every real generator resolves and
            # stores a concrete backend in ``__post_init__``.
            backend = "hip_gfx1100"
            self.backend = backend
        return hip_target_arch_for_backend(backend)

    @staticmethod
    def _int8_kv_diagnostic_override_enabled() -> bool:
        return any(
            str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}
            for name in _GGUF_INT8_KV_DIAGNOSTIC_OVERRIDE_ENVS
        )

    def _kv_weight_quant_key(self) -> str:
        file_type_name = str(
            getattr(self.weight_index, "file_type_name", "") or ""
        ).strip()
        if file_type_name:
            normalized = file_type_name.lower()
            if normalized.startswith("mostly_"):
                normalized = normalized[len("mostly_") :]
            return f"gguf_{normalized}"
        return str(getattr(self.model_plugin, "default_quant", "unknown") or "unknown")

    def _kv_model_artifact_identity(self) -> ModelArtifactIdentity:
        cached = getattr(self, "_kv_artifact_identity", None)
        if cached is not None:
            return cached
        path = getattr(self.weight_index, "path", None) or self.model_path
        identity = model_artifact_identity(path)
        self._kv_artifact_identity = identity
        return identity

    def _resolve_int8_kv_capability(self, resolved: Any) -> KVCapabilityResolution:
        artifact = self._kv_model_artifact_identity()
        key = KVCapabilityKey(
            artifact_sha256=artifact.sha256,
            artifact_size_bytes=artifact.size_bytes,
            artifact_execution_fingerprint=self._artifact_execution_fingerprint(),
            backend=str(self.backend),
            target_arch=str(self.target_arch),
            weight_quant=self._kv_weight_quant_key(),
            kv_storage=resolved.storage_dtype.value,
            storage_layout=str(resolved.storage_layout),
            scale_dtype=resolved.scale_dtype.value,
            scale_granularity=str(resolved.scale_granularity),
        )
        plugin_resolver = getattr(self.model_plugin, "resolve_kv_capability", None)
        if callable(plugin_resolver):
            return plugin_resolver(key=key, artifact=artifact)
        return resolve_kv_capability(
            (),
            key=key,
            artifact=artifact,
            declarations=getattr(
                self.model_plugin, "kv_capability_declarations", ()
            ),
        )

    @property
    def kv_capability_provenance(self) -> dict[str, object]:
        resolution = getattr(self, "_kv_capability_resolution", None)
        if resolution is not None:
            payload = resolution.as_dict()
            payload["max_packed_rows"] = self._effective_packed_decode_rows()
            return payload
        return {
            "schema_version": 1,
            "status": "not_applicable",
            "runtime_action": "not_applicable",
            "promotion_eligible": False,
            "diagnostic_override": False,
            "requested": None,
            "effective_kv_storage": "bf16",
            "max_packed_rows": self._effective_packed_decode_rows(),
            "artifact": {
                "path": str(getattr(self.weight_index, "path", self.model_path)),
                "size_bytes": None,
                "sha256": None,
                "content_verified": False,
                "error": None,
            },
            "evidence": None,
            "reason": "BF16/default KV does not require approximate-KV capability evidence",
        }

    def _effective_packed_decode_rows(self) -> int | None:
        """Return the resident session's effective packed decode width.

        The declared ``max_direct_rows`` states what the direct INT8 batch leaf
        implements. The effective width can be lower, and on a long-context INT8
        session it usually is: the layout keeps a BF16 prefix of full-attention
        layers, a BF16 layer has no retained INT8 planes, and one packed batch
        cannot run the retained leaf and the standard leaf together. The session
        therefore caps the width at one row and serializes.

        Reporting this keeps the effective width distinguishable from the
        declared one; they differ silently otherwise. Returns ``None`` when no
        resident session is available to report.
        """

        widths: list[int] = []
        for holder in (self, getattr(self, "_resident_model_runner", None)):
            accessor = getattr(holder, "_resident_sessions", None)
            if not callable(accessor):
                continue
            widths.extend(
                int(getattr(session, "packed_decode_max_rows", 0) or 0)
                for session in accessor()
                if getattr(session, "packed_decode_max_rows", None) is not None
            )
        if not widths:
            return None
        return min(widths)
    def _resolve_request_kv_policy(
        self,
        params: Any | None,
    ) -> tuple[FixedPagedKVPolicy, str, tuple[str, str, str, str]]:
        requested = resolve_kv_policy(
            getattr(params, "kv_storage", "auto") or "auto",
            scale_dtype=getattr(params, "kv_scale_dtype", "fp16") or "fp16",
            scale_granularity=(
                getattr(params, "kv_scale_granularity", "per_token_head")
                or "per_token_head"
            ),
        )
        resolved = requested
        if requested.storage_dtype.value == "int8_per_token_head":
            capability = self._resolve_int8_kv_capability(requested)
            if capability.runtime_action != "admit":
                if self._int8_kv_diagnostic_override_enabled():
                    capability = capability.with_runtime_outcome(
                        effective_kv_storage=requested.storage_dtype.value,
                        runtime_action="diagnostic_override",
                        reason=(
                            f"{capability.reason}; explicit unverified INT8 KV "
                            "diagnostic override is enabled"
                        ),
                    )
                else:
                    resolved = resolve_kv_policy("bf16")
                    capability = capability.with_runtime_outcome(
                        effective_kv_storage=resolved.storage_dtype.value,
                        runtime_action="fallback_bf16",
                        reason=f"{capability.reason}; failed closed to BF16",
                    )
            self._kv_capability_resolution = capability
        else:
            self._kv_capability_resolution = None
        signature = (
            resolved.storage_dtype.value,
            resolved.storage_layout,
            resolved.scale_dtype.value,
            resolved.scale_granularity,
        )
        return resolved.create_policy(), resolved.scale_dtype.value, signature

    def _prepare_kv_policy(self, params: Any | None) -> None:
        current = getattr(self, "_prepared_kv_signature", None)
        requested_storage = getattr(params, "kv_storage", "auto") or "auto"
        if current is not None and str(requested_storage) == "auto":
            return
        if params is None or str(requested_storage) == "auto":
            hint = getattr(self, "request_kv_policy_hint", None)
            if hint is not None:
                # The eager model-load prepare passes no request. A
                # server-configured policy hint locks the real policy instead
                # of auto-resolving BF16, which would later reject the
                # explicit policy as "cannot change after preparation".
                hint_storage, hint_scale_dtype, hint_granularity = (
                    str(hint[0]),
                    str(hint[1]),
                    str(hint[2]),
                )
                params = SimpleNamespace(
                    kv_storage=hint_storage,
                    kv_scale_dtype=hint_scale_dtype,
                    kv_scale_granularity=hint_granularity,
                )
        policy, scale_dtype, signature = self._resolve_request_kv_policy(params)
        if current is not None and current != signature:
            raise ValueError(
                "GGUF resident session KV policy cannot change after preparation: "
                f"prepared={current!r} requested={signature!r}"
            )
        self._prepared_kv_policy = policy
        self._prepared_kv_scale_dtype = scale_dtype
        self._prepared_kv_signature = signature

    def _prepared_session_kv_kwargs(self) -> dict[str, Any]:
        signature = getattr(self, "_prepared_kv_signature", None)
        if signature in {None, ("bf16", "uniform", "fp16", "per_token_head")}:
            return {}
        return {
            "kv_policy": self._prepared_kv_policy,
            "kv_scale_dtype": self._prepared_kv_scale_dtype,
            "kv_scale_granularity": signature[3],
            "kv_capability": copy.deepcopy(self.kv_capability_provenance),
        }

    @_target_arch_scoped
    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: Any | None = None,
    ) -> int | None:
        """Materialize shared GGUF weights for server resident-session reuse."""

        if max_sequence_length is not None and int(max_sequence_length) <= 0:
            raise ValueError("max_sequence_length must be positive")
        if max_sequence_length is not None:
            requested = int(max_sequence_length)
            current = getattr(self, "_prepared_max_sequence_length", None)
            self._prepared_max_sequence_length = max(
                requested,
                0 if current is None else int(current),
            )
        self._prepare_kv_policy(sampling_params)
        self._get_shared_runner()
        return None if max_sequence_length is None else int(max_sequence_length)

    @_target_arch_scoped
    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        sampling_params: Any | None = None,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any]:
        """Warm server request shapes that are lazy in the GGUF resident path."""

        del max_new_tokens
        self._prepare_kv_policy(sampling_params)
        max_batch = max(1, int(max_batch_size))
        prompt_len = max(1, min(128, int(max_prompt_tokens)))
        result: dict[str, Any] = {
            "max_prompt_tokens": int(max_prompt_tokens),
            "max_batch_size": max_batch,
            "release_after_probe": bool(release_after_probe),
            "packed_ar_prefill_widths": [],
            "packed_ar_prefill_prompt_lengths": [],
            "packed_ar_prefill_skipped": False,
            "packed_mtp_prefill_widths": [],
            "packed_mtp_prefill_prompt_lengths": [],
            "packed_mtp_prefill_skipped": False,
            "packed_mtp_verify_widths": [],
            "packed_mtp_verify_prompt_lengths": [],
            "packed_mtp_verify_skipped": False,
        }

        shared_runner = self._get_shared_runner()
        vocab_size = int(getattr(shared_runner, "vocab_size", 32000) or 32000)
        max_token = max(0, vocab_size - 1)

        def warm_prompt_for(slot_index: int, target_len: int) -> tuple[int, ...]:
            length = int(target_len)
            if length >= 32:
                spread = min(8, max(1, length // 5))
                length = max(1, min(prompt_len, length + ((int(slot_index) * 5) % (2 * spread + 1)) - spread))
            return tuple(min(((pos + int(slot_index)) % max(1, max_token)) + 1, max_token) for pos in range(length)) or (0,)

        def warm_verify_tokens_for(slot_index: int) -> tuple[int, int]:
            first = min((int(slot_index) % max(1, max_token)) + 1, max_token)
            second = min(((int(slot_index) + 1) % max(1, max_token)) + 1, max_token)
            return (first, second)

        warm_prompt_lengths = sorted({min(prompt_len, 40), prompt_len})
        prefill_batch_available = callable(getattr(Qwen35GGUFResidentSession, "prefill_batch_native", None))

        if max_batch <= 1 or not _gguf_ar_packed_prefill_enabled():
            result["packed_ar_prefill_skipped"] = True
            result["packed_ar_prefill_reason"] = "batch_width_le_1_or_disabled"
            result["reason"] = "batch_width_le_1_or_disabled"
        elif not prefill_batch_available:
            result["packed_ar_prefill_skipped"] = True
            result["packed_ar_prefill_reason"] = "backend_hook_unavailable"
            result["reason"] = "backend_hook_unavailable"
        else:
            plain_ar_limit = self.server_plain_ar_max_active_requests
            ar_max_batch = (
                max_batch
                if plain_ar_limit is None
                else min(max_batch, max(1, int(plain_ar_limit)))
            )
            ar_widths = [width for width in (2, 4, 8) if width <= ar_max_batch]
            for width in sorted(set(ar_widths)):
                for target_len in warm_prompt_lengths:
                    sessions: list[Qwen35GGUFResidentSession] = []
                    keys: list[_GGUFSessionPoolKey | None] = []
                    unsupported = False
                    try:
                        for _slot in range(width):
                            session, key, _reused = self._acquire_shared_session(
                                shared_runner,
                                pool_name="ar_batch",
                                use_wmma_prefill=_resident_session_wmma_prefill_default(),
                                use_gemv_decode=True,
                            )
                            sessions.append(session)
                            keys.append(key)
                        with _temporary_env({"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}):
                            prefill_batch = getattr(sessions[0], "prefill_batch_native")
                            prefill_batch(
                                [warm_prompt_for(slot_index, target_len) for slot_index in range(width)],
                                sessions=sessions,
                                return_logits=False,
                            )
                    except NotImplementedError:
                        result["packed_ar_prefill_skipped"] = True
                        result["packed_ar_prefill_reason"] = f"packed_prefill_unsupported_width_{width}"
                        result["reason"] = f"packed_prefill_unsupported_width_{width}"
                        unsupported = True
                    except Exception:
                        for session in sessions:
                            session.close()
                        sessions = []
                        raise
                    finally:
                        while sessions:
                            session = sessions.pop()
                            key = keys.pop()
                            self._release_shared_session(key, session)
                    if unsupported:
                        break
                    result["packed_ar_prefill_prompt_lengths"].append(int(target_len))
                if result["packed_ar_prefill_skipped"]:
                    break
                result["packed_ar_prefill_widths"].append(width)
        result["packed_ar_prefill_prompt_lengths"] = sorted(set(result["packed_ar_prefill_prompt_lengths"]))

        if not _gguf_mtp_server_startup_warmup_enabled():
            result["packed_mtp_prefill_skipped"] = True
            result["packed_mtp_prefill_reason"] = "startup_warmup_disabled"
            result["packed_mtp_verify_skipped"] = True
            result["packed_mtp_verify_reason"] = "startup_warmup_disabled"
        elif max_batch <= 1 or not _gguf_mtp_server_packed_prefill_enabled():
            result["packed_mtp_prefill_skipped"] = True
            result["packed_mtp_prefill_reason"] = "batch_width_le_1_or_disabled"
            result["packed_mtp_verify_skipped"] = True
            result["packed_mtp_verify_reason"] = "batch_width_le_1_or_disabled"
        elif not self.supports_speculative_mtp:
            result["packed_mtp_prefill_skipped"] = True
            result["packed_mtp_prefill_reason"] = "mtp_tensors_unavailable"
            result["packed_mtp_verify_skipped"] = True
            result["packed_mtp_verify_reason"] = "mtp_tensors_unavailable"
        elif not prefill_batch_available:
            result["packed_mtp_prefill_skipped"] = True
            result["packed_mtp_prefill_reason"] = "backend_hook_unavailable"
            result["packed_mtp_verify_skipped"] = True
            result["packed_mtp_verify_reason"] = "backend_hook_unavailable"
        else:
            mtp_width_cap = max_batch
            mtp_widths = [width for width in (2, 4) if width <= mtp_width_cap]
            if mtp_width_cap > 1 and mtp_width_cap not in mtp_widths:
                mtp_widths.append(mtp_width_cap)
            for width in sorted(set(mtp_widths)):
                for target_len in warm_prompt_lengths:
                    sessions: list[Qwen35GGUFResidentSession] = []
                    session_keys: list[_GGUFSessionPoolKey | None] = []
                    unsupported = False
                    try:
                        for _slot in range(width):
                            session, session_key, _session_reused = self._acquire_shared_session(
                                shared_runner,
                                pool_name="mtp_target",
                                use_wmma_prefill=_resident_session_wmma_prefill_default(),
                                use_gemv_decode=True,
                            )
                            sessions.append(session)
                            session_keys.append(session_key)
                        with _temporary_env(_LLAMA_COMPAT_MTP_ENV):
                            chunk_start_index = 0
                            while chunk_start_index < width:
                                remaining = width - chunk_start_index
                                take = min(_MTP_SERVING_TARGET_BATCH_MAX_SLOTS, remaining)
                                if remaining > _MTP_SERVING_TARGET_BATCH_MAX_SLOTS and remaining - take == 1:
                                    take -= 1
                                chunk_sessions = sessions[chunk_start_index:chunk_start_index + take]
                                chunk_owner = chunk_sessions[0]
                                prefill_batch = getattr(chunk_owner, "prefill_batch_native")
                                warm_results = prefill_batch(
                                    [
                                        warm_prompt_for(slot_index, target_len)
                                        for slot_index in range(chunk_start_index, chunk_start_index + take)
                                    ],
                                    sessions=chunk_sessions,
                                    return_logits=False,
                                    return_hidden_seeds=True,
                                )
                                if warm_results is None:
                                    raise NotImplementedError("packed MTP prefill warmup returned no results")
                                if len(list(warm_results)) != take:
                                    raise RuntimeError("packed MTP prefill warmup returned the wrong result count")
                                verify_batch = getattr(chunk_owner, "verify_target_blocks_batch", None)
                                if callable(verify_batch) and not result["packed_mtp_verify_skipped"]:
                                    try:
                                        verify_results = verify_batch(
                                            [
                                                {
                                                    "session": session,
                                                    "input_token_ids": warm_verify_tokens_for(slot_index),
                                                    "bulk_attention_mode": "bulk",
                                                    "use_wmma_prefill": _mtp_serving_target_wmma_for(self),
                                                    "capture_linear_state_rows": True,
                                                    "defer_linear_state_commit": True,
                                                    "defer_state_scatter": _gguf_mtp_server_defer_verify_scatter_enabled(),
                                                }
                                                for slot_index, session in zip(
                                                    range(chunk_start_index, chunk_start_index + take),
                                                    chunk_sessions,
                                                    strict=True,
                                                )
                                            ]
                                        )
                                        if verify_results is None:
                                            raise NotImplementedError("packed MTP verifier warmup returned no results")
                                        if len(list(verify_results)) != take:
                                            raise RuntimeError("packed MTP verifier warmup returned the wrong result count")
                                    except NotImplementedError:
                                        result["packed_mtp_verify_skipped"] = True
                                        result["packed_mtp_verify_reason"] = f"packed_verify_unsupported_width_{take}"
                                elif not callable(verify_batch):
                                    result["packed_mtp_verify_skipped"] = True
                                    result["packed_mtp_verify_reason"] = "backend_hook_unavailable"
                                chunk_start_index += take
                    except NotImplementedError:
                        result["packed_mtp_prefill_skipped"] = True
                        result["packed_mtp_prefill_reason"] = f"packed_prefill_unsupported_width_{width}"
                        unsupported = True
                    except Exception:
                        for session in sessions:
                            session.close()
                        sessions = []
                        raise
                    finally:
                        while sessions:
                            session = sessions.pop()
                            session_key = session_keys.pop()
                            self._release_shared_session(session_key, session)
                    if unsupported:
                        break
                    result["packed_mtp_prefill_prompt_lengths"].append(int(target_len))
                    if not result["packed_mtp_verify_skipped"]:
                        result["packed_mtp_verify_prompt_lengths"].append(int(target_len))
                if result["packed_mtp_prefill_skipped"]:
                    break
                result["packed_mtp_prefill_widths"].append(width)
                if not result["packed_mtp_verify_skipped"]:
                    result["packed_mtp_verify_widths"].append(width)
        result["packed_mtp_prefill_prompt_lengths"] = sorted(set(result["packed_mtp_prefill_prompt_lengths"]))
        result["packed_mtp_verify_prompt_lengths"] = sorted(set(result["packed_mtp_verify_prompt_lengths"]))
        return result

    @_target_arch_scoped
    def create_resident_model_runner(
        self,
        *,
        capacity: int | None = None,
    ) -> "Qwen35GGUFResidentModelRunner":
        """Create the single scheduler-facing GGUF model owner for this generator."""

        runner = Qwen35GGUFResidentModelRunner(
            self,
            capacity=(
                _GGUF_RESIDENT_MODEL_LOOP_DEFAULT_CAPACITY
                if capacity is None
                else int(capacity)
            ),
        )
        # The server's resident-session lookups (context reporting, KVCache
        # summary, /ready payloads) resolve the owner from the generator. Keep
        # the newest owner reachable rather than requiring the caller to thread
        # it back through.
        self._resident_model_runner = runner
        return runner

    def _get_shared_runner(self) -> Qwen35GGUFFullStackRunner:
        runner = getattr(self, "_shared_runner", None)
        if runner is not None:
            return runner
        lock = getattr(self, "_shared_runner_lock", None)
        if lock is None:
            self._shared_runner_lock = threading.Lock()
            lock = self._shared_runner_lock
        with lock:
            runner = getattr(self, "_shared_runner", None)
            if runner is None:
                runner = Qwen35GGUFFullStackRunner(
                    self.model_path, backend=self.backend,
                    execution_routes=(("eager", "native_rows", "native_graph")
                                      if getattr(self, "native_batch_decode", False) else ("eager",)),
                )
                self._shared_runner = runner
            return runner

    def _prepared_shared_runner(self) -> Qwen35GGUFFullStackRunner | None:
        return getattr(self, "_shared_runner", None)

    def _ensure_shared_pools(self) -> None:
        if not hasattr(self, "_shared_session_pool"):
            self._shared_session_pool = {}
        if not hasattr(self, "_shared_session_pool_lock"):
            self._shared_session_pool_lock = threading.Lock()
        if not hasattr(self, "_shared_mtp_draft_pool"):
            self._shared_mtp_draft_pool = {}
        if not hasattr(self, "_shared_mtp_draft_pool_lock"):
            self._shared_mtp_draft_pool_lock = threading.Lock()

    def _packed_workspace_lease_needed(self, *, max_batch_size: int) -> bool:
        """Mirror ``configure_engine_loop``'s packed-KV lease decision.

        The lease is an execution workspace floor, separate from request
        concurrency. The capacity estimate has to know about it before the pool exists, so the
        decision is reproduced here from the same inputs. The policy lives on the
        resident model runner, not on the generator, so it is read from there.
        While that runner has not resolved its engine-loop policy yet the lease
        is assumed: promising context the pool will later need is the failure
        that costs an OOM.
        """

        if int(max_batch_size) > 1:
            return True
        if os.environ.get("HIPENGINE_GGUF_PACKED_KV_LEASE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True
        owner = getattr(self, "_resident_model_runner", None)
        config = getattr(owner, "_engine_loop_config", None)
        if config is None:
            return True
        if getattr(owner, "_prefix_cache_mode", "off") != "off":
            return True
        policy = (
            str(getattr(config, "speculative_mtp_serving", "auto") or "auto")
            .strip()
            .lower()
            .replace("-", "_")
        )
        return policy != "off"

    def resident_capacity_estimate(
        self,
        *,
        max_batch_size: int = 1,
        requested_context_tokens: int | None = None,
    ) -> Any | None:
        """Price one resident session against current free HIP memory.

        This is the per-session price a caller multiplies by the number of sessions it
        holds at once. The startup scratch probe is that caller: it acquires one session
        per probe slot, and each of those sessions preallocates its own KV pages and
        packed workspace lease.

        Returns ``None`` when the route cannot be priced (no HIP runtime, fake runner).
        """

        try:
            shared_runner = self._get_shared_runner()
        except Exception as exc:  # noqa: BLE001 - best-effort sizing read
            _LOGGER.debug("GGUF resident capacity estimate unavailable: %s", exc)
            return None
        return self._resident_capacity_estimate(
            shared_runner,
            max_batch_size=max(1, int(max_batch_size)),
            defer_kv_allocation=False,
            requested_context_tokens=requested_context_tokens,
        )

    def _resident_capacity_estimate(
        self,
        shared_runner: Qwen35GGUFFullStackRunner,
        *,
        max_batch_size: int,
        defer_kv_allocation: bool,
        requested_context_tokens: int | None,
    ) -> Any | None:
        """Price the resident GGUF footprint against current free HIP memory.

        Returns ``None`` when the inputs a real session needs are unavailable
        (fake runners in unit tests, no HIP runtime). Auto-sizing is best-effort
        and must never turn a working fixed-context path into a failure.
        """

        weights = getattr(shared_runner, "weights", None)
        cfg = getattr(weights, "config", None)
        runtime = getattr(shared_runner, "runtime", None)
        mem_get_info = getattr(runtime, "mem_get_info", None)
        if cfg is None or not callable(mem_get_info):
            return None
        try:
            free_bytes, _total_bytes = mem_get_info()
            hidden_size = int(shared_runner.hidden_size)
            ffn_size = int(shared_runner.ffn_size)
            q_width = int(shared_runner.q_width)
            kv_width = int(shared_runner.kv_width)
            linear_qkv_width = int(shared_runner.linear_qkv_width)
        except Exception as exc:  # noqa: BLE001 - best-effort sizing probe
            _LOGGER.debug("GGUF automatic context sizing unavailable: %s", exc)
            return None
        signature = getattr(self, "_prepared_kv_signature", None) or (
            "bf16",
            "uniform",
            "fp16",
            "per_token_head",
        )
        storage, storage_layout, scale_dtype, granularity = (str(part) for part in signature)
        configured_budget_mib = getattr(self, "_kv_pool_memory_budget_mib", None)
        if configured_budget_mib is not None:
            free_bytes = min(
                int(free_bytes),
                int(configured_budget_mib) * 1024**2
                + _gguf_auto_context_reserve_bytes(),
            )
        model_max = int(getattr(cfg, "context_length", 0) or 0)
        requested = int(requested_context_tokens or 0) or model_max or _GGUF_AUTO_CONTEXT_BLOCK_SIZE
        qualified = _admitted_no_mirror_int8_capability(
            getattr(self, "kv_capability_provenance", None)
        )
        full_attention_layers = sum(
            1 for layer_type in getattr(cfg, "layer_types", ()) if layer_type == FULL_ATTENTION
        )

        def _layer_indices(max_positions: int) -> tuple[int, ...]:
            if qualified:
                return ()
            return _gguf_int8_bf16_full_attention_layer_indices(
                kv_storage_dtype=DType.parse(storage),
                max_positions=int(max_positions),
                full_attention_layers=int(full_attention_layers),
            )

        try:
            return estimate_qwen35_gguf_kv_capacity(
                cfg,
                available_bytes=int(free_bytes),
                requested_context_tokens=int(requested),
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                q_width=q_width,
                kv_width=kv_width,
                linear_qkv_width=linear_qkv_width,
                fp16_recurrent_state=bool(getattr(shared_runner, "fp16_recurrent_state", False)),
                # Context admission is per request.  The shared device pool
                # grows against its memory budget, so scheduler concurrency
                # must not multiply the resident KV footprint estimate.
                max_batch_size=1,
                kv_storage_dtype=storage,
                kv_storage_layout=storage_layout,
                kv_scale_dtype=scale_dtype,
                kv_scale_granularity=granularity,
                int8_kv_no_mirror_qualified=qualified,
                int8_bf16_layer_indices_resolver=_layer_indices,
                allocate_kv_cache=not bool(defer_kv_allocation),
                workspace_lease_needed=self._packed_workspace_lease_needed(
                    max_batch_size=int(max_batch_size)
                ),
                reserve_bytes=_gguf_auto_context_reserve_bytes(),
                transient_bytes_per_token=_gguf_auto_context_transient_bytes_per_token(),
                transient_fixed_bytes=_gguf_auto_context_transient_fixed_bytes(),
            )
        except Exception as exc:  # noqa: BLE001 - best-effort sizing probe
            _LOGGER.debug("GGUF automatic context sizing unavailable: %s", exc)
            return None

    def _auto_context_cache_key(
        self, *, max_batch_size: int, defer_kv_allocation: bool
    ) -> tuple[int, bool]:
        del max_batch_size
        return (1, bool(defer_kv_allocation))

    def _record_auto_context_selection(
        self,
        *,
        max_batch_size: int,
        defer_kv_allocation: bool,
        context_tokens: int,
        estimate: Any | None = None,
    ) -> None:
        """Remember a resolved context and keep the reported value conservative.

        The cache is keyed only by allocation mode. Scheduler concurrency does
        not change the per-request context ceiling when KV pages come from the
        shared elastic pool.
        """

        key = self._auto_context_cache_key(
            max_batch_size=max_batch_size,
            defer_kv_allocation=defer_kv_allocation,
        )
        self._auto_resolved_max_sequence_lengths[key] = int(context_tokens)
        current = self._auto_resolved_max_sequence_length
        self._auto_resolved_max_sequence_length = (
            int(context_tokens)
            if current is None
            else min(int(current), int(context_tokens))
        )
        if estimate is not None:
            self._auto_context_estimate = estimate

    def _resolve_auto_context(
        self,
        shared_runner: Qwen35GGUFFullStackRunner,
        *,
        max_batch_size: int,
        defer_kv_allocation: bool,
    ) -> int | None:
        """Choose the resident context when the caller did not pin one.

        The selection is cached per allocation mode. Every request shares the
        same context ceiling; concurrency is enforced by the scheduler and does
        not cause a second context calculation.
        """

        cached = self._auto_resolved_max_sequence_lengths.get(
            self._auto_context_cache_key(
                max_batch_size=max_batch_size,
                defer_kv_allocation=defer_kv_allocation,
            )
        )
        if cached is not None:
            return int(cached)
        if not _gguf_auto_context_enabled():
            return None
        estimate = self._resident_capacity_estimate(
            shared_runner,
            max_batch_size=max_batch_size,
            defer_kv_allocation=defer_kv_allocation,
            requested_context_tokens=None,
        )
        if estimate is None:
            return None
        selected = int(estimate.allocatable_context_tokens)
        if selected <= 0:
            raise MemoryError(
                "automatic GGUF resident context sizing found no allocatable context tokens; "
                "free GPU memory, use --kv-storage int8_per_token_head, or pin a smaller "
                "--max-context-tokens"
            )
        # Re-price at the selected context so the attached estimate describes the
        # decision that was taken (and its ``requested_context_tokens`` matches
        # the session) instead of the model-max probe used to find it. Nothing
        # has been allocated in between, so free memory is unchanged.
        settled = self._resident_capacity_estimate(
            shared_runner,
            max_batch_size=max_batch_size,
            defer_kv_allocation=defer_kv_allocation,
            requested_context_tokens=selected,
        )
        self._record_auto_context_selection(
            max_batch_size=max_batch_size,
            defer_kv_allocation=defer_kv_allocation,
            context_tokens=selected,
            estimate=estimate if settled is None else settled,
        )
        _LOGGER.info(
            "GGUF auto context: selected %d tokens (model max %s, allocatable %d, "
            "usable %.2f GiB after %.2f GiB reserve, ~%d B/token)",
            selected,
            "unknown" if estimate.model_max_context_tokens <= 0 else str(estimate.model_max_context_tokens),
            int(estimate.allocatable_context_tokens),
            int(estimate.usable_bytes) / 1024**3,
            int(estimate.reserve_bytes) / 1024**3,
            int(estimate.marginal_bytes_per_token),
        )
        return selected

    def _recalibrated_auto_context(
        self,
        shared_runner: Qwen35GGUFFullStackRunner,
        *,
        failed_context: int,
        max_batch_size: int,
        defer_kv_allocation: bool,
    ) -> int:
        """Re-price after an allocation failure and give back extra headroom.

        The failed attempt is a hard upper bound, and the session constructor
        rolls its partial allocations back, so free memory here is real. Two
        cases are worth separating:

        * The re-priced model now claims less than what just failed. The model
          is tracking this machine, so trim its answer by a quarter and keep the
          context that is left.
        * The re-priced model still claims more than the allocator refused. Its
          transient term is not describing this machine, so halve instead of
          trimming, which converges inside the attempt budget.
        """

        block = _GGUF_AUTO_CONTEXT_BLOCK_SIZE
        candidate = int(failed_context)
        estimate = self._resident_capacity_estimate(
            shared_runner,
            max_batch_size=max_batch_size,
            defer_kv_allocation=defer_kv_allocation,
            requested_context_tokens=int(failed_context),
        )
        model_binds = False
        if estimate is not None:
            allocatable = int(estimate.allocatable_context_tokens)
            if 0 < allocatable < candidate:
                candidate = allocatable
                model_binds = True
        if model_binds:
            reduced = candidate * _GGUF_AUTO_CONTEXT_FALLBACK_NUMERATOR // _GGUF_AUTO_CONTEXT_FALLBACK_DENOMINATOR
        else:
            reduced = candidate // 2
        reduced = max(block, (reduced // block) * block)
        return min(reduced, int(failed_context) - block)

    def _construct_shared_session(
        self,
        shared_runner: Qwen35GGUFFullStackRunner,
        *,
        max_sequence_length: int | None,
        max_batch_size: int,
        defer_kv_allocation: bool,
        use_wmma_prefill: bool | None,
        use_gemv_decode: bool | None,
    ) -> Qwen35GGUFResidentSession:
        """Construct a resident session, backing the context off if it OOMs.

        The capacity model is conservative but not exact. A failed allocation is
        rolled back by the session constructor, so the honest response is to
        recompute from the memory that is actually free and retry smaller rather
        than surface a bare HIP error during startup.
        """

        context = None if max_sequence_length is None else int(max_sequence_length)
        # ``HIPENGINE_GGUF_AUTO_CONTEXT=0`` is the full rollback: no automatic
        # sizing and no backoff, so an allocation failure stays fatal exactly as
        # it was before automatic sizing existed.
        attempts = _gguf_auto_context_attempts() if _gguf_auto_context_enabled() else 1
        for attempt in range(attempts):
            session_kwargs = {} if context is None else {"max_sequence_length": int(context)}
            try:
                session = Qwen35GGUFResidentSession(
                    self.model_path,
                    backend=self.backend,
                    runtime=shared_runner.runtime,
                    shared_runner=shared_runner,
                    use_wmma_prefill=use_wmma_prefill,
                    use_gemv_decode=use_gemv_decode,
                    defer_kv_allocation=bool(defer_kv_allocation),
                    max_batch_size=int(max_batch_size),
                    **self._prepared_session_kv_kwargs(),
                    **session_kwargs,
                )
            except (HipError, MemoryError) as exc:
                if context is None or not _gguf_allocation_failure(exc):
                    raise
                if attempt + 1 >= attempts:
                    raise MemoryError(
                        "GGUF resident context sizing could not allocate a "
                        f"resident session at {context} tokens after {attempts} attempts "
                        f"(last error: {exc}); free GPU memory, use "
                        "--kv-storage int8_per_token_head, or pin a smaller --max-context-tokens"
                    ) from exc
                next_context = self._recalibrated_auto_context(
                    shared_runner,
                    failed_context=int(context),
                    max_batch_size=int(max_batch_size),
                    defer_kv_allocation=bool(defer_kv_allocation),
                )
                if next_context >= int(context):
                    raise
                _LOGGER.warning(
                    "GGUF context request: requested %d tokens failed to allocate (%s); "
                    "retrying at %d tokens",
                    int(context),
                    exc,
                    next_context,
                )
                context = next_context
            else:
                self._attach_capacity_estimate(
                    session,
                    shared_runner,
                    max_batch_size=int(max_batch_size),
                    defer_kv_allocation=bool(defer_kv_allocation),
                )
                return session
        raise MemoryError(
            "GGUF resident context sizing exhausted its attempts"
        )  # pragma: no cover - the loop always returns or raises

    def _attach_capacity_estimate(
        self,
        session: Qwen35GGUFResidentSession,
        shared_runner: Qwen35GGUFFullStackRunner,
        *,
        max_batch_size: int,
        defer_kv_allocation: bool,
    ) -> None:
        """Publish a capacity estimate on the session for server reporting.

        The server's KVCache summary and /ready payload read
        ``session.kv_capacity_estimate``; attaching it here is what makes the
        GGUF route report the same shape the PARO route already does.
        """

        context = getattr(session, "max_sequence_length", None)
        if context is None:
            return
        estimate = getattr(self, "_auto_context_estimate", None)
        if estimate is not None and int(estimate.requested_context_tokens) == int(context):
            session.kv_capacity_estimate = estimate
            return
        repriced = self._resident_capacity_estimate(
            shared_runner,
            max_batch_size=int(max_batch_size),
            defer_kv_allocation=bool(defer_kv_allocation),
            requested_context_tokens=int(context),
        )
        if repriced is not None:
            session.kv_capacity_estimate = repriced

    def _acquire_shared_session(
        self,
        shared_runner: Qwen35GGUFFullStackRunner,
        *,
        pool_name: str,
        use_wmma_prefill: bool | None = None,
        use_gemv_decode: bool | None = None,
        defer_kv_allocation: bool = False,
        max_batch_size: int = 1,
    ) -> tuple[Qwen35GGUFResidentSession, _GGUFSessionPoolKey, bool]:
        self._ensure_shared_pools()
        max_sequence_length = getattr(self, "_prepared_max_sequence_length", None)
        if getattr(self, "_prepared_kv_signature", None) is None:
            self._prepare_kv_policy(None)
        assert self._prepared_kv_signature is not None
        if max_sequence_length is None:
            max_sequence_length = self._resolve_auto_context(
                shared_runner,
                max_batch_size=int(max_batch_size),
                defer_kv_allocation=bool(defer_kv_allocation),
            )
        key = (
            str(pool_name),
            use_wmma_prefill,
            use_gemv_decode,
            max_sequence_length,
            int(max_batch_size),
            self._prepared_kv_signature,
        )
        with self._shared_session_pool_lock:
            pool = self._shared_session_pool.get(key)
            session = pool.pop() if pool else None
        if session is not None:
            reset = getattr(session, "reset", None)
            if callable(reset):
                reset()
            self._configure_session(session)
            return session, key, True
        session = self._construct_shared_session(
            shared_runner,
            max_sequence_length=max_sequence_length,
            max_batch_size=int(max_batch_size),
            defer_kv_allocation=bool(defer_kv_allocation),
            use_wmma_prefill=use_wmma_prefill,
            use_gemv_decode=use_gemv_decode,
        )
        effective = getattr(session, "max_sequence_length", None)
        if effective is not None and int(effective) != (
            None if max_sequence_length is None else int(max_sequence_length)
        ):
            # The session backed off to a smaller context than the pool key was
            # built from. Keep the key describing what the session actually owns,
            # or a later acquire would hand out a mismatched session under the
            # original key.
            max_sequence_length = int(effective)
            self._record_auto_context_selection(
                max_batch_size=int(max_batch_size),
                defer_kv_allocation=bool(defer_kv_allocation),
                context_tokens=int(effective),
            )
            key = (
                str(pool_name),
                use_wmma_prefill,
                use_gemv_decode,
                max_sequence_length,
                int(max_batch_size),
                self._prepared_kv_signature,
            )
        self._configure_session(session)
        return session, key, False

    def _release_shared_session(
        self,
        key: _GGUFSessionPoolKey | None,
        session: Qwen35GGUFResidentSession,
    ) -> None:
        self._ensure_shared_pools()
        if key is None:
            session.close()
            return
        try:
            reset = getattr(session, "reset", None)
            if callable(reset):
                reset()
        except Exception:
            session.close()
            raise
        with self._shared_session_pool_lock:
            self._shared_session_pool.setdefault(key, []).append(session)

    @contextmanager
    def _resident_session_scope(
        self,
        *,
        shared_runner: Qwen35GGUFFullStackRunner | None,
        pool_name: str,
        use_wmma_prefill: bool | None = _GGUF_PUBLIC_USE_WMMA_PREFILL,
        use_gemv_decode: bool | None = _GGUF_PUBLIC_USE_GEMV_DECODE,
    ):
        if shared_runner is None:
            session_kwargs: dict[str, Any] = {
                "backend": self.backend,
                **self._prepared_session_kv_kwargs(),
            }
            if use_wmma_prefill is not None:
                session_kwargs["use_wmma_prefill"] = bool(use_wmma_prefill)
            if use_gemv_decode is not None:
                session_kwargs["use_gemv_decode"] = bool(use_gemv_decode)
            with Qwen35GGUFResidentSession(self.model_path, **session_kwargs) as session:
                self._configure_session(session)
                yield session, False
            return
        session, key, reused = self._acquire_shared_session(
            shared_runner,
            pool_name=pool_name,
            use_wmma_prefill=use_wmma_prefill,
            use_gemv_decode=use_gemv_decode,
        )
        try:
            self._configure_session(session)
            yield session, reused
        except Exception:
            session.close()
            raise
        else:
            self._release_shared_session(key, session)

    def _acquire_mtp_draft_runner(
        self,
        assets: _GGUFMTPServingAssets,
        *,
        runtime: Any,
        pool_enabled: bool,
    ) -> tuple[Any, int | None, bool]:
        self._ensure_shared_pools()
        if not pool_enabled:
            return _new_mtp_draft_runner(assets, runtime=runtime), None, False
        key = int(id(runtime))
        with self._shared_mtp_draft_pool_lock:
            pool = self._shared_mtp_draft_pool.get(key)
            draft = pool.pop() if pool else None
        if draft is not None:
            return draft, key, True
        return _new_mtp_draft_runner(assets, runtime=runtime), key, False

    def _release_mtp_draft_runner(self, key: Any | None, draft: Any) -> None:
        self._ensure_shared_pools()
        if key is None:
            close = getattr(draft, "close", None)
            if callable(close):
                close()
            return
        with self._shared_mtp_draft_pool_lock:
            self._shared_mtp_draft_pool.setdefault(key, []).append(draft)

    def _acquire_dense_mtp_draft_provider(
        self,
        target: Qwen35GGUFResidentSession,
        *,
        max_positions: int,
        pool_enabled: bool,
        max_requests: int = 1,
    ) -> tuple[Any, Any | None, bool]:
        """Open or reuse the architecture-shaped dense NextN provider."""

        from hipengine.runtime.qwen35_gguf_nextn import (
            Qwen35GGUFNextNDraftProvider,
            borrow_qwen35_gguf_nextn_fallback_weights,
        )

        self._ensure_shared_pools()
        hot_vocab_path = _gguf_mtp_hot_vocab_setting(
            str(getattr(self, "execution_profile", "strict"))
        )
        key = (
            int(id(target.runtime)),
            "dense_nextn",
            int(max_positions),
            int(max_requests),
            hot_vocab_path,
        )
        if pool_enabled:
            with self._shared_mtp_draft_pool_lock:
                pool = self._shared_mtp_draft_pool.get(key)
                provider = pool.pop() if pool else None
            if provider is not None:
                return provider, key, True
        provider = Qwen35GGUFNextNDraftProvider.from_model(
            self.model_path,
            max_positions=int(max_positions),
            max_requests=int(max_requests),
            runtime=target.runtime,
            require_cached_build=bool(target.require_cached_build),
            borrowed_fallback_weights=borrow_qwen35_gguf_nextn_fallback_weights(target),
            hot_vocab_path=hot_vocab_path,
        )
        return provider, key if pool_enabled else None, False

    @_target_arch_scoped
    def close(self) -> None:
        """Close pooled sessions/drafts before releasing shared model weights."""

        if bool(getattr(self, "_closed", False)):
            return
        self._closed = True
        self._ensure_shared_pools()
        with self._shared_session_pool_lock:
            sessions = [
                session
                for pool in self._shared_session_pool.values()
                for session in pool
            ]
            self._shared_session_pool.clear()
        with self._shared_mtp_draft_pool_lock:
            drafts = [
                draft
                for pool in self._shared_mtp_draft_pool.values()
                for draft in pool
            ]
            self._shared_mtp_draft_pool.clear()
        lock = getattr(self, "_shared_runner_lock", None)
        if lock is None:
            self._shared_runner_lock = threading.Lock()
            lock = self._shared_runner_lock
        with lock:
            shared_runner = getattr(self, "_shared_runner", None)
            self._shared_runner = None
        self._mtp_serving_assets = None

        error: BaseException | None = None
        for resource in (*reversed(sessions), *reversed(drafts), shared_runner):
            if resource is None:
                continue
            closer = getattr(resource, "close", None)
            if not callable(closer):
                continue
            try:
                closer()
            except BaseException as exc:  # pragma: no cover - defensive cleanup
                if error is None:
                    error = exc
        if error is not None:
            raise error

    @property
    def native_sampler_provenance(self) -> dict[str, Any]:
        from hipengine.runtime.native_sampler import native_sampler_provenance

        return {
            **native_sampler_provenance(self.native_sampler_algorithm),
            "model_execution_profile": getattr(self, "execution_profile", None),
            "backend": self.backend,
        }

    def _configure_native_sampler(self, session: Qwen35GGUFResidentSession) -> None:
        # Sampling is an independently declared arithmetic axis (§3). Do not
        # infer strict sampler parity from a model-only execution-profile label.
        provenance = self.native_sampler_provenance
        workspace = getattr(session, "_native_sampler_workspace", None)
        if workspace is not None and not workspace.closed:
            if workspace.full_vocab_algorithm != self.native_sampler_algorithm:
                raise RuntimeError("cannot change live native sampler selection")
        session.native_sampler_algorithm = provenance["full_vocab_algorithm"]

    def _configure_session(self, session: Qwen35GGUFResidentSession) -> None:
        self._configure_native_sampler(session)
        # Prefill correctness policy is selected by the generator registry
        # factory, not by a quant branch in runtime dispatch.
        session.default_bulk_attention_mode = getattr(
            self,
            "bulk_prefill_attention_mode",
            "bulk",
        )
        session.use_prefill_f16_staging = _prefill_f16_staging_for(self)
        session.use_q6_integer_mmq = _q6_integer_mmq_for(self)
        prefill_quant = getattr(self, "prefill_quant", None)
        if prefill_quant is not None:
            session.select_prefill_quant(prefill_quant)
        aotriton_min_tokens = getattr(self, "prefill_attn_aotriton_min_tokens", None)
        if aotriton_min_tokens is not None:
            session.prefill_config = replace(
                session.prefill_config or PrefillConfig(),
                attn_aotriton_min_tokens=int(aotriton_min_tokens),
            )

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(int(token) for token in self.tokenizer.encode(str(text)))

    def detokenize(
        self,
        token_ids: Sequence[int],
        *,
        skip_special: bool = False,
    ) -> str:
        return self.tokenizer.decode(
            tuple(int(token) for token in token_ids),
            skip_special=bool(skip_special),
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def speculative_provider_capabilities(self):
        """Declare the model-attached NextN provider without engine branching."""

        from hipengine.benchmark.provenance import collect_model_identity
        from hipengine.speculative.registry import SpeculativeProviderCapabilities

        identity = collect_model_identity(Path(self.model_path).expanduser().resolve())
        return SpeculativeProviderCapabilities(
            provider_name="nextn",
            artifact_fingerprint=str(identity["fingerprint"]["value"]),
            attachment_mode="model_attached",
            supported_modes=("verify_chain",),
            max_verifier_rows=8,
            transaction_mode="journal",
            provider_state_key="shared_target_hidden",
            provider_kv_key="shared_target_kv",
            strict_fallback="target_ar",
        )

    @property
    def supports_speculative_mtp(self) -> bool:
        """Whether this GGUF inventory has the NextN tensors required for MTP."""

        return _gguf_info_has_mtp_tensors(self.weight_index)

    def _speculative_mtp_serving_key(
        self,
        *,
        realized_group_rows: int,
        resident_capacity: int,
        candidate_budget: int,
        sampling_mode: str,
        kv_storage: str,
        memory_fit: bool,
    ):
        """Build the resident artifact's model-plugin serving scope.

        Returns ``(evidence, key)``, or ``(evidence, None)`` when this model
        plugin owns unrelated artifacts (for example Q4_K_S) whose explicit
        compatibility route is independent of the typed serving evidence.
        """

        from hipengine.speculative.serving import SpeculativeMTPServingKey

        requested_kv = str(kv_storage or "auto")
        prepared_kv = getattr(self, "_prepared_kv_signature", None)
        effective_kv = (
            str(prepared_kv[0]) if prepared_kv is not None
            else "bf16" if requested_kv == "auto" else requested_kv
        )
        evidence = tuple(
            getattr(
                self.model_plugin,
                "speculative_mtp_serving_evidence",
                (),
            )
            or ()
        )
        artifact_path = Path(
            getattr(self.weight_index, "path", None) or self.model_path
        ).expanduser()
        try:
            artifact_size = int(artifact_path.stat().st_size)
        except OSError:
            artifact_size = None
        weight_quant = self._kv_weight_quant_key()
        implemented_storage = any(
            implementation.kv_storage == effective_kv
            for implementation in getattr(
                self.model_plugin, "speculative_mtp_serving_implementations", (),
            )
        )
        execution_fingerprint = self._artifact_execution_fingerprint()
        # The artifact axis is execution identity, so a revision whose bytes
        # differ but whose layouts match still resolves here.  Only the quant
        # family is a cheap pre-filter: a row for another quant cannot cover
        # this artifact on any axis.
        if not implemented_storage and not any(
            row.weight_quant == weight_quant for row in evidence
        ):
            return evidence, None
        key = SpeculativeMTPServingKey(
            # Provenance only: the resident file's SHA-256 is not an admission
            # axis, so it is not computed here.  Computing it would read the
            # whole artifact to compare a value no row reads.
            artifact_sha256=None,
            artifact_size_bytes=artifact_size,
            content_verified=execution_fingerprint is not None,
            backend=str(self.backend),
            target_arch=str(self.target_arch),
            weight_quant=weight_quant,
            kv_storage=effective_kv,
            kv_layout=str(prepared_kv[1]) if prepared_kv is not None else "uniform",
            realized_group_rows=int(realized_group_rows),
            resident_capacity=int(resident_capacity),
            candidate_budget=int(candidate_budget),
            sampling_mode=str(sampling_mode),
            memory_fit=bool(memory_fit),
            kv_scale_dtype=str(prepared_kv[2]) if prepared_kv is not None else None,
            kv_scale_granularity=str(prepared_kv[3]) if prepared_kv is not None else None,
            artifact_execution_fingerprint=execution_fingerprint,
        )
        return evidence, key

    def _artifact_execution_fingerprint(self) -> str | None:
        """Execution identity of the resident artifact, when it is computable.

        ``None`` leaves the key without an artifact identity, and admission then
        fails closed as ``artifact_identity_unverified``.
        """

        from hipengine.loading.gguf import gguf_execution_fingerprint

        try:
            return gguf_execution_fingerprint(self.weight_index)
        except Exception:
            return None

    def resolve_speculative_mtp_serving_plan(
        self,
        *,
        realized_group_rows: int,
        resident_capacity: int,
        candidate_budget: int,
        sampling_mode: str,
        kv_storage: str,
        memory_fit: bool,
        request_mode: str = "automatic",
    ):
        """Resolve the model-plugin-owned exact serving plan before mutation.

        ``request_mode`` is ``explicit`` for a request that asked for
        speculation and ``automatic`` otherwise.  Only the explicit mode may
        admit on implementation capability when no evidence row covers the
        cell; automatic policy stays with the retained evidence.
        """

        from hipengine.speculative.serving import resolve_speculative_mtp_serving_plan

        evidence, key = self._speculative_mtp_serving_key(
            realized_group_rows=realized_group_rows,
            resident_capacity=resident_capacity,
            candidate_budget=candidate_budget,
            sampling_mode=sampling_mode,
            kv_storage=kv_storage,
            memory_fit=memory_fit,
        )
        if key is None:
            # No typed plan applies; preserve the independent explicit
            # compatibility route without implying default evidence.
            return None
        resolver = getattr(
            self.model_plugin,
            "resolve_speculative_mtp_serving_plan",
            None,
        )
        if callable(resolver):
            try:
                return resolver(key=key, request_mode=str(request_mode))
            except TypeError:
                # Model plugins that predate the request-mode parameter keep
                # their evidence-only behaviour.
                return resolver(key=key)
        return resolve_speculative_mtp_serving_plan((), key=key)

    def max_qualified_speculative_candidate_budget(
        self,
        *,
        realized_group_rows: int,
        resident_capacity: int,
        sampling_mode: str,
        kv_storage: str = "auto",
        memory_fit: bool = True,
    ) -> int | None:
        """Deepest depth the resident artifact's evidence qualifies for this cell.

        The requested-depth axis is deliberately ignored: the answer describes
        what the retained evidence authorizes, which is how an omitted server
        budget resolves without a global constant.
        """

        evidence, key = self._speculative_mtp_serving_key(
            realized_group_rows=realized_group_rows,
            resident_capacity=resident_capacity,
            # Placeholder: the capability resolver drops the depth axis.
            candidate_budget=1,
            sampling_mode=sampling_mode,
            kv_storage=kv_storage,
            memory_fit=memory_fit,
        )
        if key is None or not evidence:
            return None
        resolver = getattr(self.model_plugin, "max_qualified_candidate_budget", None)
        if not callable(resolver):
            return None
        return resolver(key=key)

    def speculative_candidate_budget_default(self) -> int:
        """Depth this generator's own dense MTP path uses when unpinned."""

        return int(_gguf_mtp_server_candidate_budget())

    def generate(self, request: GenerationRequest) -> list[str]:
        outputs = self.generate_detailed(request)
        return [output.text for output in outputs]

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        for chunk in self.stream_detailed(request):
            yield chunk.text

    @_target_arch_scoped_stream
    def stream_detailed(self, request: GenerationRequest) -> Iterator[GenerationStreamChunk]:
        self.last_batch_generation = None
        self._prepare_kv_policy(request)
        if len(request.prompts) != 1:
            raise ValueError("streaming currently supports exactly one prompt")
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        if request.max_tokens == 0:
            return
        prompt_ids, tokenize_ms = _encode_prompt_timed(
            self.tokenizer,
            request.prompts[0],
        )
        raise_if_generation_deadline_expired(request)
        if not prompt_ids:
            raise ValueError("GGUF prompt tokenization produced no token IDs")
        plan = _gguf_sampler_plan(request)
        shared_runner = self._prepared_shared_runner()
        session_kwargs = (
            {
                "backend": self.backend,
                "runtime": shared_runner.runtime,
                "shared_runner": shared_runner,
                "use_wmma_prefill": _GGUF_PUBLIC_USE_WMMA_PREFILL,
                "use_gemv_decode": _GGUF_PUBLIC_USE_GEMV_DECODE,
                **self._prepared_session_kv_kwargs(),
            }
            if shared_runner is not None
            else {
                "backend": self.backend,
                "use_wmma_prefill": _GGUF_PUBLIC_USE_WMMA_PREFILL,
                "use_gemv_decode": _GGUF_PUBLIC_USE_GEMV_DECODE,
                **self._prepared_session_kv_kwargs(),
            }
        )
        with Qwen35GGUFResidentSession(self.model_path, **session_kwargs) as session:
            self._configure_native_sampler(session)
            if plan.mode is SamplingMode.GREEDY_FAST:
                yield from self._stream_greedy(
                    session,
                    prompt_ids,
                    request,
                    tokenize_ms=tokenize_ms,
                )
                return
            yield from self._stream_sampled(
                session,
                prompt_ids,
                request,
                row_index=0,
                tokenize_ms=tokenize_ms,
            )

    @_target_arch_scoped
    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        self._prepare_kv_policy(request)
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        plan = _gguf_sampler_plan(request)
        if request.max_tokens == 0:
            encoded_prompts = {
                index: _encode_prompt_timed(self.tokenizer, prompt)
                for index, prompt in enumerate(request.prompts)
            }
            prompt_rows_by_request = {
                index: encoded[0]
                for index, encoded in encoded_prompts.items()
            }
            self.last_generation_outputs = tuple(
                GenerationOutput(
                    text="",
                    generated_token_ids=(),
                    finish_details=_gguf_finish_details((), self.tokenizer, request),
                    telemetry=_gguf_telemetry(
                        prompt_rows_by_request[index],
                        (),
                        request,
                        row_index=index,
                        timing={"tokenize_ms": encoded_prompts[index][1]},
                    ),
                )
                for index, prompt in enumerate(request.prompts)
            )
            self.last_batch_generation = _gguf_last_batch_generation(
                self.tokenizer,
                request,
                plan,
                prompt_rows_by_request,
                {index: [] for index in prompt_rows_by_request},
                {index: [] for index in prompt_rows_by_request},
                outputs=self.last_generation_outputs,
            )
            return list(self.last_generation_outputs)
        outputs: list[GenerationOutput] = []
        prompt_rows_by_request: dict[int, list[int]] = {}
        generated_ids_by_request: dict[int, list[int]] = {}
        token_logprobs_by_request: dict[int, list[TokenLogprob]] = {}
        shared_runner = self._prepared_shared_runner()
        if (
            plan.mode is SamplingMode.GREEDY_FAST
            and len(request.prompts) > 1
            and shared_runner is not None
            and (_gguf_ar_packed_decode_enabled() or _gguf_ar_stream_decode_enabled())
        ):
            return self._generate_ar_serving_slots(shared_runner, request, plan=plan)
        if (
            plan.mode is SamplingMode.GREEDY_FAST
            and len(request.prompts) > 1
            and bool(getattr(self, "native_batch_decode", False))
        ):
            encoded_prompts = {
                row_index: _encode_prompt_timed(self.tokenizer, prompt)
                for row_index, prompt in enumerate(request.prompts)
            }
            prompt_rows_by_request = {
                row_index: encoded[0]
                for row_index, encoded in encoded_prompts.items()
            }
            if any(not prompt_ids for prompt_ids in prompt_rows_by_request.values()):
                raise ValueError("GGUF prompt tokenization produced no token IDs")
            max_sequence_length = max(
                256,
                max(len(prompt_ids) for prompt_ids in prompt_rows_by_request.values())
                + int(request.max_tokens),
            )
            native_capacity = min(
                int(getattr(self, "native_batch_capacity", 8)),
                len(prompt_rows_by_request),
            )
            with Qwen35GGUFResidentSession(
                self.model_path,
                max_sequence_length=max_sequence_length,
                max_batch_size=native_capacity,
                backend=self.backend,
                execution_routes=("eager", "native_rows", "native_graph"),
            ) as session:
                self._configure_session(session)
                native_run = self._generate_greedy_batch(
                    session,
                    prompt_rows_by_request,
                    request,
                    capacity=native_capacity,
                )
                generated_ids_by_request = native_run.generated_ids
            outputs = [
                GenerationOutput(
                    text=self.tokenizer.decode(generated_ids_by_request[row_index]),
                    generated_token_ids=tuple(generated_ids_by_request[row_index]),
                    finish_details=_gguf_finish_details(
                        generated_ids_by_request[row_index],
                        self.tokenizer,
                        request,
                    ),
                    telemetry=_gguf_telemetry(
                        prompt_rows_by_request[row_index],
                        generated_ids_by_request[row_index],
                        request,
                        row_index=row_index,
                        request_id=str(row_index),
                        phase="answer",
                        execution_path="gguf_native_continuous_decode",
                        native_compact_prefill=False,
                        native_caware_decode=True,
                        serial_decode_fallback=False,
                        native_sampler_rows=True,
                        timing={"tokenize_ms": encoded_prompts[row_index][1]},
                    ),
                )
                for row_index in range(len(prompt_rows_by_request))
            ]
            self.last_generation_outputs = tuple(outputs)
            self.last_batch_generation = _gguf_last_batch_generation(
                self.tokenizer,
                request,
                plan,
                prompt_rows_by_request,
                generated_ids_by_request,
                token_logprobs_by_request,
                outputs=self.last_generation_outputs,
                native_batch=True,
                native_decode_steps=native_run.native_decode_steps,
                execution_paths=native_run.execution_paths,
                scheduling=native_run.scheduling,
            )
            return outputs
        session_open_start = time.perf_counter()
        with self._resident_session_scope(
            shared_runner=shared_runner,
            pool_name="ar",
        ) as (session, _session_reused):
            session_open_ms = _timing_ms_since(session_open_start)
            for row_index, prompt in enumerate(request.prompts):
                row_start = time.perf_counter()
                row_timing: dict[str, float] = {"session_open_ms": session_open_ms}
                raise_if_generation_deadline_expired(request)
                prompt_ids, row_timing["tokenize_ms"] = _encode_prompt_timed(
                    self.tokenizer,
                    prompt,
                )
                prompt_rows_by_request[row_index] = prompt_ids
                raise_if_generation_deadline_expired(request)
                if not prompt_ids:
                    raise ValueError("GGUF prompt tokenization produced no token IDs")
                if plan.mode is SamplingMode.GREEDY_FAST:
                    generated_ids = self._generate_greedy(
                        session,
                        prompt_ids,
                        request,
                        timing=row_timing,
                    )
                    generated_ids_by_request[row_index] = list(generated_ids)
                    finish_details = _gguf_finish_details(generated_ids, self.tokenizer, request)
                    decode_text_start = time.perf_counter()
                    text = self.tokenizer.decode(generated_ids)
                    _timing_set(row_timing, "decode_text_ms", decode_text_start)
                    _timing_set(row_timing, "request_total_ms", row_start)
                    outputs.append(
                        GenerationOutput(
                            text=text,
                            generated_token_ids=generated_ids,
                            finish_details=finish_details,
                            telemetry=_gguf_telemetry(
                                prompt_ids,
                                generated_ids,
                                request,
                                row_index=row_index,
                                timing=row_timing,
                            ),
                        )
                    )
                else:
                    output = self._generate_sampled(
                        session,
                        prompt_ids,
                        request,
                        row_index=row_index,
                        timing=row_timing,
                    )
                    outputs.append(output)
                    token_logprobs_by_request[row_index] = list(output.token_logprobs)
                    if output.generated_token_ids is None:
                        raise RuntimeError("sampled GGUF generation did not expose generated token ids")
                    generated_ids_by_request[row_index] = list(output.generated_token_ids)
        self.last_generation_outputs = tuple(outputs)
        self.last_batch_generation = _gguf_last_batch_generation(
            self.tokenizer,
            request,
            plan,
            prompt_rows_by_request,
            generated_ids_by_request,
            token_logprobs_by_request,
            outputs=self.last_generation_outputs,
        )
        return outputs

    def _generate_ar_serving_slots(
        self,
        shared_runner: Qwen35GGUFFullStackRunner,
        request: GenerationRequest,
        *,
        plan: Any,
    ) -> list[GenerationOutput]:
        encoded_prompts: dict[int, list[int]] = {}
        tokenize_ms_by_request: dict[int, float] = {}
        for row_index, prompt in enumerate(request.prompts):
            raise_if_generation_deadline_expired(request)
            prompt_ids, tokenize_ms_by_request[row_index] = _encode_prompt_timed(
                self.tokenizer,
                prompt,
            )
            if not prompt_ids:
                raise ValueError("GGUF prompt tokenization produced no token IDs")
            encoded_prompts[row_index] = prompt_ids

        slots: list[_GGUFARServingSlot] = []
        try:
            slots = self._open_ar_serving_slots(
                shared_runner,
                encoded_prompts,
                tokenize_ms_by_request,
                request,
            )
            self._run_ar_serving_slots(slots, request)
            outputs: list[GenerationOutput] = []
            prompt_rows_by_request: dict[int, list[int]] = {}
            generated_ids_by_request: dict[int, list[int]] = {}
            token_logprobs_by_request: dict[int, list[TokenLogprob]] = {}
            for slot in sorted(slots, key=lambda item: item.request_id):
                row_timing = dict(slot.timing)
                generated_ids = list(slot.generated_ids)
                prompt_rows_by_request[slot.request_id] = list(slot.prompt_ids)
                generated_ids_by_request[slot.request_id] = generated_ids
                token_logprobs_by_request[slot.request_id] = []
                decode_text_start = time.perf_counter()
                text = self.tokenizer.decode(generated_ids)
                _timing_set(row_timing, "decode_text_ms", decode_text_start)
                outputs.append(
                    GenerationOutput(
                        text=text,
                        generated_token_ids=generated_ids,
                        finish_details=_gguf_finish_details(generated_ids, self.tokenizer, request),
                        telemetry=_gguf_telemetry(
                            slot.prompt_ids,
                            generated_ids,
                            request,
                            row_index=slot.request_id,
                            timing=row_timing,
                            execution_path="gguf_packed_ar_server_decode",
                            native_compact_prefill=slot.native_compact_prefill,
                            native_caware_decode=slot.native_decode_steps > 0,
                            serial_decode_fallback=slot.serial_decode_steps > 0,
                            native_sampler_rows=False,
                        ),
                    )
                )
            batch_id = _new_gguf_timing_batch_id("ar")
            outputs = _with_batch_timing_ownership(outputs, batch_id=batch_id)
            self.last_generation_outputs = tuple(outputs)
            native_compact_prefill = bool(slots) and all(slot.native_compact_prefill for slot in slots)
            native_decode_steps = max((slot.native_decode_steps for slot in slots), default=0)
            serial_decode_fallback = any(slot.serial_decode_steps > 0 for slot in slots)
            self.last_batch_generation = _gguf_last_batch_generation(
                self.tokenizer,
                request,
                plan,
                prompt_rows_by_request,
                generated_ids_by_request,
                token_logprobs_by_request,
                outputs=self.last_generation_outputs,
                execution_path="gguf_packed_ar_server_decode",
                native_compact_prefill=native_compact_prefill,
                native_decode_steps=native_decode_steps,
                native_caware_decode=native_decode_steps > 0,
                serial_decode_fallback=serial_decode_fallback,
            )
            self.last_batch_generation.update(
                {
                    "batch_id": batch_id,
                    "group_rows": len(outputs),
                    "timing_scope": "batch",
                    "timing_owner": True,
                }
            )
            self._close_ar_serving_slots(slots, reuse=True)
            slots = []
            return outputs
        except Exception:
            if slots:
                self._close_ar_serving_slots(slots, reuse=False)
            raise

    def _open_ar_serving_slots(
        self,
        shared_runner: Qwen35GGUFFullStackRunner,
        encoded_prompts: dict[int, list[int]],
        tokenize_ms_by_request: dict[int, float],
        request: GenerationRequest,
    ) -> list[_GGUFARServingSlot]:
        slots: list[_GGUFARServingSlot] = []
        try:
            for row_index in range(len(request.prompts)):
                raise_if_generation_deadline_expired(request)
                prompt_ids = encoded_prompts[row_index]
                timing: dict[str, float] = {
                    "tokenize_ms": float(tokenize_ms_by_request.get(row_index, 0.0))
                }
                session_open_start = time.perf_counter()
                session, session_pool_key, _session_reused = self._acquire_shared_session(
                    shared_runner,
                    pool_name="ar_batch",
                    use_wmma_prefill=_resident_session_wmma_prefill_default(),
                    use_gemv_decode=True,
                )
                _timing_set(timing, "session_open_ms", session_open_start)
                slot = _GGUFARServingSlot(
                    request_id=row_index,
                    prompt_ids=list(prompt_ids),
                    session=session,
                    prev_token=0,
                    seq_position=0,
                    generated_ids=[],
                    timing=timing,
                    session_pool_key=session_pool_key,
                )
                slots.append(slot)
            if self._try_prefill_ar_serving_slots_batch(slots, request):
                return slots
            for slot in slots:
                prefill_start = time.perf_counter()
                prefill_result = slot.session.prefill(slot.prompt_ids, return_logits=False)
                _timing_add(slot.timing, "prefill_ms", prefill_start)
                self._finish_ar_serving_slot_prefill(slot, int(prefill_result.token_id), request)
        except Exception:
            self._close_ar_serving_slots(slots, reuse=False)
            raise
        return slots

    def _try_prefill_ar_serving_slots_batch(
        self,
        slots: list[_GGUFARServingSlot],
        request: GenerationRequest,
    ) -> bool:
        if len(slots) <= 1 or not _gguf_ar_packed_prefill_enabled():
            return False
        chunks = self._ar_serving_slot_chunks(slots)
        if any(len(chunk) <= 1 for chunk in chunks):
            return False
        for chunk in chunks:
            prefill_batch = getattr(chunk[0].session, "prefill_batch_native", None)
            if not callable(prefill_batch):
                return False
            if any(not slot.prompt_ids for slot in chunk):
                return False
        batch_start = time.perf_counter()
        batch_results_by_request: dict[int, Any] = {}
        chunk_ms_by_request: dict[int, float] = {}
        completed_chunks = 0
        serial_prefill_request_ids: set[int] = set()
        try:
            with _temporary_env({"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}):
                for chunk in chunks:
                    prefill_batch = getattr(chunk[0].session, "prefill_batch_native")
                    prompt_ids = [tuple(int(token) for token in slot.prompt_ids) for slot in chunk]
                    sessions = [slot.session for slot in chunk]
                    chunk_start = time.perf_counter()
                    results = prefill_batch(
                        prompt_ids,
                        sessions=sessions,
                        return_logits=False,
                    )
                    if results is None:
                        raise NotImplementedError("GGUF AR packed prefill returned no results")
                    batch_results = list(results)
                    if len(batch_results) != len(chunk):
                        raise RuntimeError(
                            f"GGUF AR packed prefill returned {len(batch_results)} result(s) "
                            f"for {len(chunk)} live slot(s)"
                        )
                    chunk_ms = _timing_ms_since(chunk_start)
                    for slot, result in zip(chunk, batch_results, strict=True):
                        batch_results_by_request[int(slot.request_id)] = result
                        chunk_ms_by_request[int(slot.request_id)] = chunk_ms
                    completed_chunks += 1
        except NotImplementedError:
            if completed_chunks == 0:
                # Nothing was prefilled yet, so the caller's per-slot route can
                # serve the whole request; declining here keeps the packed
                # bound from rejecting a prompt the context admits.
                return False
            # A later width group hit the packed route's context bound. The
            # groups before it are already prefilled, so finish the remaining
            # slots on the strict per-session route instead of failing a
            # request whose prompt is inside the configured context.
            self._fallback_reasons["packed_prefill_unsupported_k0"] += 1
            for chunk in chunks[completed_chunks:]:
                for slot in chunk:
                    result = slot.session.prefill(
                        slot.prompt_ids,
                        return_logits=False,
                    )
                    batch_results_by_request[int(slot.request_id)] = result
                    chunk_ms_by_request[int(slot.request_id)] = _timing_ms_since(
                        batch_start
                    )
                    serial_prefill_request_ids.add(int(slot.request_id))
        prefill_ms = _timing_ms_since(batch_start)
        for slot in slots:
            result = batch_results_by_request[int(slot.request_id)]
            _timing_add_ms(slot.timing, "prefill_ms", prefill_ms)
            _timing_add_ms(slot.timing, "prefill_batch_ms", prefill_ms)
            if len(chunks) > 1:
                _timing_add_ms(
                    slot.timing,
                    "prefill_batch_chunk_ms",
                    float(chunk_ms_by_request.get(int(slot.request_id), 0.0)),
                )
            slot.native_compact_prefill = (
                int(slot.request_id) not in serial_prefill_request_ids
            )
            self._finish_ar_serving_slot_prefill(slot, int(getattr(result, "token_id")), request)
        return True

    def _finish_ar_serving_slot_prefill(
        self,
        slot: _GGUFARServingSlot,
        token_id: int,
        request: GenerationRequest,
    ) -> None:
        prev_token = int(token_id)
        generated_ids = [prev_token]
        slot.prev_token = prev_token
        slot.seq_position = int(slot.session.position)
        slot.generated_ids = generated_ids
        slot.done = (
            len(generated_ids) >= int(request.max_tokens)
            or _gguf_finished(generated_ids, self.tokenizer, request)
        )

    def _run_ar_serving_slots(
        self,
        slots: list[_GGUFARServingSlot],
        request: GenerationRequest,
    ) -> None:
        while any(not slot.done for slot in slots):
            live_slots = [slot for slot in slots if not slot.done]
            cycle_start = time.perf_counter()
            handled = False
            if _gguf_ar_packed_decode_enabled():
                handled = self._try_step_ar_serving_slots_batch(live_slots, request)
            if not handled:
                handled = self._try_step_ar_serving_slots_streams(live_slots, request)
            if not handled:
                for slot in live_slots:
                    self._step_ar_serving_slot_serial(slot, request)
            cycle_ms = _timing_ms_since(cycle_start)
            for slot in live_slots:
                _timing_add_ms(slot.timing, "slots_decode_phase_ms", cycle_ms)

    def _try_step_ar_serving_slots_streams(
        self,
        live_slots: list[_GGUFARServingSlot],
        request: GenerationRequest,
    ) -> bool:
        if len(live_slots) <= 1 or not _gguf_ar_stream_decode_enabled():
            return False
        self._flush_ar_packed_decode_owners(live_slots)
        for slot in live_slots:
            if not callable(getattr(slot.session, "step_async_top1", None)):
                return False
            if not callable(getattr(slot.session, "read_top1_sample", None)):
                return False
            runtime = getattr(slot.session, "runtime", None)
            if not callable(getattr(runtime, "stream_create", None)):
                return False
            if not callable(getattr(runtime, "stream_synchronize", None)):
                return False
            if not callable(getattr(runtime, "stream_destroy", None)):
                return False
        launch_start = time.perf_counter()
        launched: list[_GGUFARServingSlot] = []
        for slot in live_slots:
            if slot.decode_stream == 0:
                slot.decode_stream = int(slot.session.runtime.stream_create(nonblocking=True))
            slot.session.step_async_top1(int(slot.prev_token), position=int(slot.seq_position), stream=int(slot.decode_stream))
            launched.append(slot)
        for slot in launched:
            slot.session.runtime.stream_synchronize(int(slot.decode_stream))
        launch_ms = _timing_ms_since(launch_start)
        for slot in launched:
            result = slot.session.read_top1_sample()
            _timing_add_ms(slot.timing, "decode_stream_batch_ms", launch_ms)
            self._record_ar_serving_token(slot, int(result.token_id), request)
            slot.native_decode_steps += 1
        return True

    def _try_step_ar_serving_slots_batch(
        self,
        live_slots: list[_GGUFARServingSlot],
        request: GenerationRequest,
    ) -> bool:
        if len(live_slots) <= 1:
            return False
        chunks = self._ar_serving_slot_chunks(live_slots)
        if len(chunks) > 1 and _gguf_ar_stream_decode_enabled():
            streamed = self._try_step_ar_serving_slot_chunks_streams(chunks, request)
            if streamed:
                return True
        for chunk in chunks:
            if len(chunk) <= 1:
                self._flush_ar_packed_decode_owners(chunk)
                for slot in chunk:
                    self._step_ar_serving_slot_serial(slot, request)
                continue
            if not self._step_ar_serving_slot_chunk_packed(chunk, request):
                self._flush_ar_packed_decode_owners(chunk)
                for slot in chunk:
                    self._step_ar_serving_slot_serial(slot, request)
                continue
        return True

    def _ar_serving_slot_chunks(
        self,
        live_slots: list[_GGUFARServingSlot],
    ) -> list[list[_GGUFARServingSlot]]:
        chunks: list[list[_GGUFARServingSlot]] = []
        index = 0
        while index < len(live_slots):
            remaining = len(live_slots) - index
            take = min(_GGUF_AR_NATIVE_MAX_SLOTS, remaining)
            if remaining > _GGUF_AR_NATIVE_MAX_SLOTS and remaining - take == 1:
                take -= 1
            chunks.append(live_slots[index:index + take])
            index += take
        return chunks

    def _try_step_ar_serving_slot_chunks_streams(
        self,
        chunks: list[list[_GGUFARServingSlot]],
        request: GenerationRequest,
    ) -> bool:
        if len(chunks) <= 1:
            return False
        for chunk in chunks:
            if len(chunk) <= 1:
                return False
            owner_slot = chunk[0]
            step_batch = getattr(owner_slot.session, "step_batch_native", None)
            if not callable(step_batch):
                return False
            runtime = getattr(owner_slot.session, "runtime", None)
            if not callable(getattr(runtime, "stream_create", None)):
                return False
            if not callable(getattr(runtime, "stream_synchronize", None)):
                return False
            if not callable(getattr(runtime, "stream_destroy", None)):
                return False
        for chunk in chunks:
            self._flush_ar_packed_decode_owners_if_chunk_changed(chunk)
            owner_slot = chunk[0]
            if owner_slot.decode_stream == 0:
                owner_slot.decode_stream = int(owner_slot.session.runtime.stream_create(nonblocking=True))

        def step_chunk(chunk: list[_GGUFARServingSlot]) -> list[Any] | None:
            return self._step_ar_serving_slot_chunk_packed(
                chunk,
                request,
                stream=int(chunk[0].decode_stream),
                record_tokens=False,
            )

        stream_start = time.perf_counter()
        with _temporary_env({"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                futures = [(chunk, pool.submit(step_chunk, chunk)) for chunk in chunks]
                chunk_results: list[tuple[list[_GGUFARServingSlot], list[Any]]] = []
                for chunk, future in futures:
                    result = future.result()
                    if result is None:
                        raise RuntimeError("streamed packed AR decode chunk failed after launch")
                    chunk_results.append((chunk, result))
        stream_ms = _timing_ms_since(stream_start)
        for chunk, step_results in chunk_results:
            owner_session = chunk[0].session
            for slot, step_result in zip(chunk, step_results, strict=True):
                _timing_add_ms(slot.timing, "decode_batch_ms", stream_ms)
                _timing_add_ms(slot.timing, "decode_stream_chunks_ms", stream_ms)
                token = int(getattr(step_result, "token_id"))
                self._record_ar_serving_token(slot, token, request)
                slot.packed_decode_owner = owner_session
                slot.native_decode_steps += 1
        return True

    def _step_ar_serving_slot_chunk_packed(
        self,
        chunk: list[_GGUFARServingSlot],
        request: GenerationRequest,
        *,
        stream: int = 0,
        record_tokens: bool = True,
    ) -> list[Any] | None:
        if len(chunk) <= 1:
            return None
        first_session = chunk[0].session
        step_batch = getattr(first_session, "step_batch_native", None)
        if not callable(step_batch):
            return None
        token_ids = [int(slot.prev_token) for slot in chunk]
        sessions = [slot.session for slot in chunk]
        positions = [int(slot.seq_position) for slot in chunk]
        decode_start = time.perf_counter()
        try:
            if stream:
                batch_result = step_batch(
                    token_ids,
                    sessions=sessions,
                    positions=positions,
                    return_logits=False,
                    scatter_state=False,
                    stream=int(stream),
                )
            else:
                self._flush_ar_packed_decode_owners_if_chunk_changed(chunk)
                with _temporary_env({"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}):
                    batch_result = step_batch(
                        token_ids,
                        sessions=sessions,
                        positions=positions,
                        return_logits=False,
                        scatter_state=False,
                    )
        except NotImplementedError:
            return None
        if batch_result is None:
            return None
        step_results = list(batch_result)
        if len(step_results) != len(chunk):
            raise RuntimeError(
                f"GGUF AR native batch decode returned {len(step_results)} result(s) "
                f"for {len(chunk)} live slot(s)"
            )
        if not record_tokens:
            return step_results
        decode_ms = _timing_ms_since(decode_start)
        for slot, step_result in zip(chunk, step_results, strict=True):
            _timing_add_ms(slot.timing, "decode_batch_ms", decode_ms)
            token = int(getattr(step_result, "token_id"))
            self._record_ar_serving_token(slot, token, request)
            slot.packed_decode_owner = first_session
            slot.native_decode_steps += 1
        return step_results

    def _flush_ar_packed_decode_owners_if_chunk_changed(self, chunk: list[_GGUFARServingSlot]) -> None:
        if not chunk:
            return
        owner = chunk[0].packed_decode_owner
        sessions = tuple(slot.session for slot in chunk)
        if owner is not None and all(slot.packed_decode_owner is owner for slot in chunk):
            owner_sessions = tuple(getattr(owner, "_packed_decode_sessions", ()))
            owner_dirty = bool(getattr(owner, "_packed_decode_state_dirty", False))
            if owner_dirty and owner_sessions == sessions:
                return
        self._flush_ar_packed_decode_owners(chunk)

    def _flush_ar_packed_decode_owners(self, slots: list[_GGUFARServingSlot]) -> None:
        owners: list[Any] = []
        for slot in slots:
            owner = slot.packed_decode_owner
            if owner is not None and not any(existing is owner for existing in owners):
                owners.append(owner)
        for owner in owners:
            flush = getattr(owner, "flush_packed_decode_state", None)
            if callable(flush):
                flush()
        if owners:
            for slot in slots:
                if any(slot.packed_decode_owner is owner for owner in owners):
                    slot.packed_decode_owner = None

    def _step_ar_serving_slot_serial(
        self,
        slot: _GGUFARServingSlot,
        request: GenerationRequest,
    ) -> None:
        if slot.done:
            return
        self._flush_ar_packed_decode_owners([slot])
        raise_if_generation_deadline_expired(request)
        decode_start = time.perf_counter()
        step = slot.session.step(slot.prev_token, return_logits=False)
        _timing_add(slot.timing, "decode_ms", decode_start)
        self._record_ar_serving_token(slot, int(step.token_id), request)
        slot.serial_decode_steps += 1

    def _record_ar_serving_token(
        self,
        slot: _GGUFARServingSlot,
        token_id: int,
        request: GenerationRequest,
    ) -> None:
        token = int(token_id)
        slot.generated_ids.append(token)
        slot.prev_token = token
        slot.seq_position += 1
        slot.done = (
            len(slot.generated_ids) >= int(request.max_tokens)
            or _gguf_finished(slot.generated_ids, self.tokenizer, request)
        )

    def _close_ar_serving_slots(self, slots: list[_GGUFARServingSlot], *, reuse: bool = True) -> None:
        for slot in reversed(slots):
            if slot.decode_stream:
                slot.session.runtime.stream_destroy(int(slot.decode_stream))
                slot.decode_stream = 0
            if reuse:
                self._release_shared_session(slot.session_pool_key, slot.session)
            else:
                slot.session.close()

    @_target_arch_scoped
    def _generate_dense_speculative_mtp_detailed(
        self,
        request: GenerationRequest,
        *,
        config: Qwen35GGUFConfig,
    ) -> list[GenerationOutput]:
        """Generate dense Qwen3.6 MTP through the shared transactional ABI."""

        from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession

        shared_runner = self._prepared_shared_runner()
        outputs: list[GenerationOutput] = []
        prompt_rows_by_request: dict[int, list[int]] = {}
        generated_ids_by_request: dict[int, list[int]] = {}
        cycles_by_request: dict[int, list[dict[str, Any]]] = {}
        request_started = time.perf_counter()
        encoded_prompts: dict[int, tuple[list[int], float]] = {
            row_index: _encode_prompt_timed(self.tokenizer, prompt)
            for row_index, prompt in enumerate(request.prompts)
        }
        requested_max_positions = max(
            int(getattr(self, "_prepared_max_sequence_length", 0) or 0),
            max(len(tokens) for tokens, _tokenize_ms in encoded_prompts.values())
            + int(request.max_tokens)
            + 4,
        )
        stop_request = _request_with_tokenizer_eos(request, self.tokenizer)
        eos_token_id = getattr(stop_request, "eos_token_id", None)
        stop_token_ids = tuple(getattr(stop_request, "stop_token_ids", ()) or ())
        candidate_budget = _gguf_mtp_server_candidate_budget()
        target_verify_mode = _gguf_mtp_server_target_verify_mode()
        with self._resident_session_scope(
            shared_runner=shared_runner,
            pool_name="mtp_target_dense",
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as (target, _session_reused):
            max_positions = max(
                requested_max_positions,
                int(target.target_layout.max_sequence_length),
            )
            provider, provider_pool_key, _provider_reused = self._acquire_dense_mtp_draft_provider(
                target,
                max_positions=max_positions,
                pool_enabled=shared_runner is not None,
            )
            release_provider_to_pool = False
            try:
                for row_index, _prompt in enumerate(request.prompts):
                    raise_if_generation_deadline_expired(request)
                    prompt_ids, tokenize_ms = encoded_prompts[row_index]
                    if not prompt_ids:
                        raise ValueError("GGUF prompt tokenization produced no token IDs")
                    decoder = Qwen35GGUFMTPDecodeSession(
                        target,
                        provider,
                        candidate_budget=candidate_budget,
                        quant="gguf_q4_k_m",
                        target_verify_mode=target_verify_mode,
                    )
                    try:
                        result = decoder.generate(
                            prompt_ids,
                            max_new_tokens=int(request.max_tokens),
                            request_id=row_index,
                            eos_token_id=eos_token_id,
                            stop_token_ids=stop_token_ids,
                            # RF3: poll the request cancellation token and
                            # deadline at every MTP cycle boundary so a
                            # timed-out/cancelled request stops before the next
                            # proposal/target mutation instead of letting GPU
                            # work continue after the HTTP client left.
                            checkpoint=lambda: raise_if_generation_deadline_expired(request),
                        )
                    finally:
                        decoder.close()
                    raise_if_generation_deadline_expired(request)
                    generated_ids = list(result.token_ids)
                    cycle_rows = [
                        {
                            "mode": "llama_compat_direct_commit",
                            "generated_draft_tokens": len(record.get("draft_tokens", ())),
                            "accepted_draft_tokens": int(record.get("accepted", 0)),
                            "visible_output_tokens": int(record.get("accepted", 0)) + 1,
                            "device_proposal_fallback_reason": record.get(
                                "device_proposal_fallback_reason"
                            ),
                            "target_native_graph_fallback_reason": record.get(
                                "target_native_graph_fallback_reason"
                            ),
                        }
                        for record in result.cycle_records
                    ]
                    timing = {
                        "tokenize_ms": tokenize_ms,
                        "prefill_ms": float(result.prefill_seconds) * 1000.0,
                        "decode_ms": float(result.decode_seconds) * 1000.0,
                        "draft_propose_ms": float(result.proposal_seconds) * 1000.0,
                        "target_verify_ms": float(result.verify_seconds) * 1000.0,
                    }
                    _add_mtp_cycle_timing_metrics(timing, cycle_rows)
                    _timing_set(timing, "request_total_ms", request_started)
                    prompt_rows_by_request[row_index] = prompt_ids
                    generated_ids_by_request[row_index] = generated_ids
                    cycles_by_request[row_index] = cycle_rows
                    outputs.append(
                        self._mtp_generation_output(
                            prompt_ids,
                            generated_ids,
                            request,
                            row_index=row_index,
                            resident_slot_count=1,
                            timing=timing,
                        )
                    )
                    provider.release_request(row_index)
                release_provider_to_pool = True
            finally:
                self._release_mtp_draft_runner(
                    provider_pool_key if release_provider_to_pool else None,
                    provider,
                )
        self.last_generation_outputs = tuple(outputs)
        self.last_batch_generation = _gguf_mtp_last_batch_generation(
            self.tokenizer,
            request,
            _gguf_sampler_plan(request),
            prompt_rows_by_request,
            generated_ids_by_request,
            {},
            outputs=self.last_generation_outputs,
            cycles_by_request=cycles_by_request,
            resident_slot_count=1,
            target_verify_batching=f"single_slot_transactional_{target_verify_mode}",
        )
        self.last_batch_generation["speculative_mtp"].update(
            {
                "draft_model": "architecture_shaped_nextn",
                "draft_n_max": candidate_budget,
                "nextn_block_id": int(config.ignored_block_ids[0]),
                "target_verify": f"transactional_{target_verify_mode}",
            }
        )
        return outputs

    @_target_arch_scoped
    def generate_speculative_mtp_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        """Generate through the llama.cpp-compatible GGUF MTP route.

        The c=1 path keeps the retained direct llama-compat hot loop. Coalesced
        c>1 requests use shared-weight resident slots with isolated target/MTP
        state and a phase-serial scheduler inside the server process.
        """

        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        if not self.supports_speculative_mtp:
            raise NotImplementedError("GGUF speculative MTP requires Qwen NextN tensors")
        config, _block_id, _required = _gguf_mtp_required_tensor_names(self.weight_index)
        plan = _gguf_sampler_plan(request)
        if plan.mode is not SamplingMode.GREEDY_FAST:
            raise NotImplementedError("GGUF speculative MTP currently supports only greedy-fast sampling")
        if request.max_tokens == 0:
            return self.generate_detailed(request)
        if not config.is_moe:
            return self._generate_dense_speculative_mtp_detailed(request, config=config)

        request_start = time.perf_counter()
        encoded_prompts: dict[int, list[int]] = {}
        tokenize_ms_by_request: dict[int, float] = {}
        for row_index, prompt in enumerate(request.prompts):
            (
                encoded_prompts[row_index],
                tokenize_ms_by_request[row_index],
            ) = _encode_prompt_timed(self.tokenizer, prompt)
        if any(
            len(prompt_ids) < _GGUF_MTP_CONTEXT_REPLAY_MIN_PROMPT_TOKENS
            for prompt_ids in encoded_prompts.values()
        ):
            with _temporary_env({"HIPENGINE_GGUF_DECODE_REPACK": "1"}):
                return self.generate_detailed(request)

        outputs: list[GenerationOutput] = []
        prompt_rows_by_request: dict[int, list[int]] = {}
        generated_ids_by_request: dict[int, list[int]] = {}
        token_logprobs_by_request: dict[int, list[TokenLogprob]] = {}
        mtp_cycles_by_request: dict[int, list[dict[str, Any]]] = {}

        with self._mtp_serving_lock, _temporary_env(_LLAMA_COMPAT_MTP_ENV) as base_env:
            assets_load_start = time.perf_counter()
            assets = self._load_mtp_serving_assets()
            assets_load_ms = _timing_ms_since(assets_load_start)
            if len(request.prompts) == 1:
                shared_runner = self._prepared_shared_runner()
                session_open_start = time.perf_counter()
                with self._resident_session_scope(
                    shared_runner=shared_runner,
                    pool_name="mtp_target",
                    use_wmma_prefill=_resident_session_wmma_prefill_default(),
                    use_gemv_decode=True,
                ) as (session, _session_reused):
                    session_open_ms = _timing_ms_since(session_open_start)
                    runtime = session.runtime
                    draft_open_start = time.perf_counter()
                    resident_draft, draft_pool_key, _draft_reused = self._acquire_mtp_draft_runner(
                        assets,
                        runtime=runtime,
                        pool_enabled=shared_runner is not None,
                    )
                    draft_open_ms = _timing_ms_since(draft_open_start)
                    release_draft_to_pool = False
                    try:
                        for row_index, prompt in enumerate(request.prompts):
                            raise_if_generation_deadline_expired(request)
                            prompt_ids = encoded_prompts[row_index]
                            prompt_rows_by_request[row_index] = prompt_ids
                            if not prompt_ids:
                                raise ValueError("GGUF prompt tokenization produced no token IDs")
                            run = self._generate_speculative_mtp_llama_compat(
                                session,
                                resident_draft,
                                assets,
                                prompt_ids,
                                request,
                                base_env=base_env,
                            )
                            generated_ids = list(run.generated_ids)
                            generated_ids_by_request[row_index] = generated_ids
                            mtp_cycles_by_request[row_index] = list(run.cycles)
                            timing = dict(run.timing)
                            timing["session_open_ms"] = session_open_ms
                            timing["draft_open_ms"] = draft_open_ms
                            timing["assets_load_ms"] = assets_load_ms
                            timing["tokenize_ms"] = tokenize_ms_by_request.get(row_index, 0.0)
                            _add_mtp_cycle_timing_metrics(timing, run.cycles)
                            _timing_set(timing, "request_total_ms", request_start)
                            outputs.append(
                                self._mtp_generation_output(
                                    prompt_ids,
                                    generated_ids,
                                    request,
                                    row_index=row_index,
                                    resident_slot_count=1,
                                    timing=timing,
                                )
                            )
                        release_draft_to_pool = True
                    finally:
                        self._release_mtp_draft_runner(
                            draft_pool_key if release_draft_to_pool else None,
                            resident_draft,
                        )
                resident_slot_count = 1
                target_verify_batching = "single_slot"
            else:
                shared_runner = self._prepared_shared_runner()
                if shared_runner is None:
                    with Qwen35GGUFFullStackRunner(self.model_path) as local_runner:
                        resident_slot_count, target_verify_batching = self._generate_prepared_mtp_serving_slots(
                            local_runner,
                            assets,
                            encoded_prompts,
                            request,
                            base_env=base_env,
                            prompt_rows_by_request=prompt_rows_by_request,
                            generated_ids_by_request=generated_ids_by_request,
                            mtp_cycles_by_request=mtp_cycles_by_request,
                            tokenize_ms_by_request=tokenize_ms_by_request,
                            assets_load_ms=assets_load_ms,
                            pool_sessions=False,
                            outputs=outputs,
                        )
                else:
                    resident_slot_count, target_verify_batching = self._generate_prepared_mtp_serving_slots(
                        shared_runner,
                        assets,
                        encoded_prompts,
                        request,
                        base_env=base_env,
                        prompt_rows_by_request=prompt_rows_by_request,
                        generated_ids_by_request=generated_ids_by_request,
                        mtp_cycles_by_request=mtp_cycles_by_request,
                        tokenize_ms_by_request=tokenize_ms_by_request,
                        assets_load_ms=assets_load_ms,
                        pool_sessions=True,
                        outputs=outputs,
                    )

        timing_batch_id: str | None = None
        if len(outputs) > 1:
            timing_batch_id = _new_gguf_timing_batch_id("mtp")
            outputs = _with_batch_timing_ownership(outputs, batch_id=timing_batch_id)
        self.last_generation_outputs = tuple(outputs)
        self.last_batch_generation = _gguf_mtp_last_batch_generation(
            self.tokenizer,
            request,
            plan,
            prompt_rows_by_request,
            generated_ids_by_request,
            token_logprobs_by_request,
            outputs=self.last_generation_outputs,
            cycles_by_request=mtp_cycles_by_request,
            resident_slot_count=resident_slot_count,
            target_verify_batching=target_verify_batching,
        )
        if timing_batch_id is not None:
            self.last_batch_generation.update(
                {
                    "batch_id": timing_batch_id,
                    "group_rows": len(outputs),
                    "timing_scope": "batch",
                    "timing_owner": True,
                }
            )
        return outputs

    def _generate_prepared_mtp_serving_slots(
        self,
        shared_runner: Qwen35GGUFFullStackRunner,
        assets: _GGUFMTPServingAssets,
        encoded_prompts: dict[int, list[int]],
        request: GenerationRequest,
        *,
        base_env: dict[str, str | None],
        prompt_rows_by_request: dict[int, list[int]],
        generated_ids_by_request: dict[int, list[int]],
        mtp_cycles_by_request: dict[int, list[dict[str, Any]]],
        tokenize_ms_by_request: dict[int, float],
        assets_load_ms: float,
        pool_sessions: bool,
        outputs: list[GenerationOutput],
    ) -> tuple[int, str]:
        slots_open_start = time.perf_counter()
        slots = self._open_mtp_serving_slots(
            shared_runner,
            assets,
            encoded_prompts,
            request,
            pool_sessions=pool_sessions,
        )
        slots_open_ms = _timing_ms_since(slots_open_start)
        resident_slot_count = len(slots)
        target_verify_batching = "per_slot_serial" if resident_slot_count > 1 else "single_slot"
        release_slots_to_pool = False
        try:
            slots_run_start = time.perf_counter()
            self._run_mtp_serving_slots(slots, assets, request, base_env=base_env)
            slots_run_ms = _timing_ms_since(slots_run_start)
            if resident_slot_count > 1 and any("target_verify_batch_ms" in slot.timing for slot in slots):
                target_verify_batching = "packed_slot_batch"
            for slot in slots:
                prompt_rows_by_request[slot.request_id] = list(slot.prompt_ids)
                generated_ids = list(slot.generated_ids)
                generated_ids_by_request[slot.request_id] = generated_ids
                mtp_cycles_by_request[slot.request_id] = list(slot.cycles)
                timing = dict(slot.timing)
                timing["tokenize_ms"] = tokenize_ms_by_request.get(slot.request_id, 0.0)
                timing["assets_load_ms"] = assets_load_ms
                timing["slots_open_ms"] = slots_open_ms
                timing["slots_run_ms"] = slots_run_ms
                _add_mtp_cycle_timing_metrics(timing, slot.cycles)
                outputs.append(
                    self._mtp_generation_output(
                        slot.prompt_ids,
                        generated_ids,
                        request,
                        row_index=slot.request_id,
                        resident_slot_count=resident_slot_count,
                        timing=timing,
                    )
                )
            release_slots_to_pool = True
        finally:
            self._close_mtp_serving_slots(slots, reuse=release_slots_to_pool)
        return resident_slot_count, target_verify_batching

    def _mtp_generation_output(
        self,
        prompt_ids: list[int],
        generated_ids: list[int],
        request: GenerationRequest,
        *,
        row_index: int,
        resident_slot_count: int,
        timing: dict[str, float] | None = None,
    ) -> GenerationOutput:
        return GenerationOutput(
            text=self.tokenizer.decode(generated_ids),
            generated_token_ids=generated_ids,
            finish_details=_gguf_finish_details(generated_ids, self.tokenizer, request),
            telemetry=_gguf_telemetry(
                prompt_ids,
                generated_ids,
                request,
                row_index=row_index,
                execution_path="gguf_llama_compat_mtp_server",
                native_compact_prefill=False,
                native_caware_decode=False,
                serial_decode_fallback=False,
                native_sampler_rows=False,
                timing=timing,
            ),
        )

    def _load_mtp_serving_assets(self) -> _GGUFMTPServingAssets:
        cached = self._mtp_serving_assets
        if cached is not None:
            return cached
        reader = GGUFReader(self.model_path)
        try:
            config, block_id, required_names = _gguf_mtp_required_tensor_names(reader.info)
        except ValueError as exc:
            raise NotImplementedError(str(exc)) from exc
        weights: dict[str, tuple[np.ndarray, int, tuple[int, ...]]] = {}
        required = set(required_names)
        for tensor in reader.info.tensors:
            if tensor.name in required:
                weights[tensor.name] = (
                    reader.tensor_data(tensor.name),
                    int(tensor.ggml_type),
                    tuple(tensor.shape),
                )
        missing = sorted(required.difference(weights))
        if missing:
            raise NotImplementedError(
                "GGUF speculative MTP requires missing tensor(s): " + ", ".join(missing[:8])
            )
        token_embd_f32 = dequantize_gguf_data(
            weights["token_embd.weight"][0],
            weights["token_embd.weight"][1],
        ).astype(np.float32, copy=False)
        rope_cos, rope_sin = _gguf_rope_tables(
            max_positions=262144,
            rotary_dim=int(config.rope_dimension_count),
            base=float(config.rope_freq_base),
        )
        assets = _GGUFMTPServingAssets(
            weights=weights,
            token_embd_f32=np.ascontiguousarray(token_embd_f32, dtype=np.float32),
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            config=config,
            nextn_block_id=block_id,
        )
        self._mtp_serving_assets = assets
        return assets

    def _open_mtp_serving_slots(
        self,
        shared_runner: Qwen35GGUFFullStackRunner,
        assets: _GGUFMTPServingAssets,
        encoded_prompts: dict[int, list[int]],
        request: GenerationRequest,
        *,
        pool_sessions: bool,
    ) -> list[_GGUFMTPServingSlot]:
        packed_slots = self._try_open_mtp_serving_slots_batch_prefill(
            shared_runner,
            assets,
            encoded_prompts,
            request,
            pool_sessions=pool_sessions,
        )
        if packed_slots is not None:
            return packed_slots
        slots: list[_GGUFMTPServingSlot] = []
        try:
            for row_index in range(len(request.prompts)):
                raise_if_generation_deadline_expired(request)
                prompt_ids = encoded_prompts[row_index]
                if not prompt_ids:
                    raise ValueError("GGUF prompt tokenization produced no token IDs")
                slots.append(
                    self._open_mtp_serving_slot(
                        shared_runner,
                        assets,
                        prompt_ids,
                        request,
                        request_id=row_index,
                        pool_sessions=pool_sessions,
                    )
                )
        except Exception:
            self._close_mtp_serving_slots(slots, reuse=False)
            raise
        return slots

    def _try_open_mtp_serving_slots_batch_prefill(
        self,
        shared_runner: Qwen35GGUFFullStackRunner,
        assets: _GGUFMTPServingAssets,
        encoded_prompts: dict[int, list[int]],
        request: GenerationRequest,
        *,
        pool_sessions: bool,
    ) -> list[_GGUFMTPServingSlot] | None:
        if len(request.prompts) <= 1 or not _gguf_mtp_server_packed_prefill_enabled():
            return None
        if len(request.prompts) > _MTP_SERVING_TARGET_BATCH_MAX_SLOTS:
            return None
        if not callable(getattr(Qwen35GGUFResidentSession, "prefill_batch_native", None)):
            return None

        acquired: list[dict[str, Any]] = []

        def close_acquired() -> None:
            for entry in reversed(acquired):
                slot = entry.get("slot")
                if isinstance(slot, _GGUFMTPServingSlot):
                    _free_mtp_buffers(slot.mtp_buffers, runtime=slot.session.runtime)
                    self._release_mtp_draft_runner(None, slot.resident_draft)
                    slot.session.close()
                    continue
                draft = entry.get("resident_draft")
                if draft is not None:
                    self._release_mtp_draft_runner(None, draft)
                session = entry.get("session")
                if isinstance(session, Qwen35GGUFResidentSession):
                    session.close()

        try:
            for row_index in range(len(request.prompts)):
                raise_if_generation_deadline_expired(request)
                prompt_ids = encoded_prompts[row_index]
                if not prompt_ids:
                    raise ValueError("GGUF prompt tokenization produced no token IDs")
                slot_open_start = time.perf_counter()
                timing: dict[str, float] = {}
                session_open_start = time.perf_counter()
                session, session_pool_key, _session_reused = self._acquire_shared_session(
                    shared_runner,
                    pool_name="mtp_target",
                    use_wmma_prefill=_resident_session_wmma_prefill_default(),
                    use_gemv_decode=True,
                ) if pool_sessions else (
                    Qwen35GGUFResidentSession(
                        self.model_path,
                        runtime=shared_runner.runtime,
                        shared_runner=shared_runner,
                        use_wmma_prefill=_resident_session_wmma_prefill_default(),
                        use_gemv_decode=True,
                    ),
                    None,
                    False,
                )
                _timing_set(timing, "session_open_ms", session_open_start)
                runtime = session.runtime
                draft_open_start = time.perf_counter()
                resident_draft, draft_pool_key, _draft_reused = self._acquire_mtp_draft_runner(
                    assets,
                    runtime=runtime,
                    pool_enabled=pool_sessions,
                )
                _timing_set(timing, "draft_open_ms", draft_open_start)
                acquired.append(
                    {
                        "request_id": int(row_index),
                        "prompt_ids": list(prompt_ids),
                        "slot_open_start": slot_open_start,
                        "timing": timing,
                        "session": session,
                        "session_pool_key": session_pool_key,
                        "resident_draft": resident_draft,
                        "draft_pool_key": draft_pool_key,
                    }
                )

            # The packed route can refuse an unsupported slab or resource
            # shape. That is a route decision, not a failure of the request: release
            # what was acquired and decline so the per-request slot path serves it.
            try:
                owner_session = acquired[0]["session"]
                prefill_batch = getattr(owner_session, "prefill_batch_native", None)
                if not callable(prefill_batch):
                    raise NotImplementedError("resident session has no packed prefill entry point")
                prefill_results_by_slot: list[Any | None] = [None] * len(acquired)
                prefill_ms_by_slot = [0.0] * len(acquired)
                chunk_start_index = 0
                with _temporary_env({"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}):
                    while chunk_start_index < len(acquired):
                        remaining = len(acquired) - chunk_start_index
                        take = min(_MTP_SERVING_TARGET_BATCH_MAX_SLOTS, remaining)
                        if remaining > _MTP_SERVING_TARGET_BATCH_MAX_SLOTS and remaining - take == 1:
                            take -= 1
                        chunk_entries = acquired[chunk_start_index:chunk_start_index + take]
                        chunk_owner = chunk_entries[0]["session"]
                        chunk_prefill_batch = getattr(chunk_owner, "prefill_batch_native", None)
                        if not callable(chunk_prefill_batch):
                            raise NotImplementedError("resident session has no packed prefill entry point")
                        prompt_batch = [tuple(entry["prompt_ids"]) for entry in chunk_entries]
                        session_batch = [entry["session"] for entry in chunk_entries]
                        prefill_start = time.perf_counter()
                        chunk_results = chunk_prefill_batch(
                            prompt_batch,
                            sessions=session_batch,
                            return_logits=False,
                            return_hidden_seeds=True,
                        )
                        prefill_ms = _timing_ms_since(prefill_start)
                        if chunk_results is None:
                            raise NotImplementedError("packed MTP prefill returned no results")
                        chunk_results = list(chunk_results)
                        if len(chunk_results) != len(chunk_entries):
                            raise RuntimeError(
                                f"packed MTP prefill returned {len(chunk_results)} result(s) "
                                f"for {len(chunk_entries)} slot(s)"
                            )
                        for offset, result in enumerate(chunk_results):
                            slot_index = chunk_start_index + offset
                            prefill_results_by_slot[slot_index] = result
                            prefill_ms_by_slot[slot_index] = prefill_ms
                        chunk_start_index += take
            except NotImplementedError:
                close_acquired()
                return None

            slots: list[_GGUFMTPServingSlot] = []
            hidden_size = int(assets.token_embd_f32.shape[1])
            qk_head_dim = int(
                np.asarray(
                    assets.weights[f"blk.{assets.nextn_block_id}.attn_q_norm.weight"][0]
                ).shape[0]
            )
            max_cycles = max(1, int(request.max_tokens))
            for entry, prefill_result, prefill_ms in zip(
                acquired,
                prefill_results_by_slot,
                prefill_ms_by_slot,
                strict=True,
            ):
                if prefill_result is None:
                    raise RuntimeError("packed MTP prefill did not populate every slot result")
                session = entry["session"]
                resident_draft = entry["resident_draft"]
                timing = entry["timing"]
                prompt_ids = entry["prompt_ids"]
                prompt_hidden_rows = np.ascontiguousarray(
                    getattr(prefill_result, "hidden_seeds"),
                    dtype=np.float32,
                )
                if prompt_hidden_rows.shape != (len(prompt_ids), hidden_size):
                    raise RuntimeError(
                        "packed MTP prefill returned hidden rows with shape "
                        f"{prompt_hidden_rows.shape}, expected {(len(prompt_ids), hidden_size)}"
                    )
                _timing_add_ms(timing, "prefill_ms", prefill_ms)
                _timing_add_ms(timing, "prefill_batch_ms", prefill_ms)
                mtp_context_tokens, mtp_context_hidden_rows = _llama_cpp_mtp_catchup_rows(
                    prompt_ids,
                    prompt_hidden_rows,
                )
                prev_token = int(getattr(prefill_result, "token_id"))
                generated_ids = [prev_token]
                seq_position = int(session.position)
                resident_context = _new_mtp_context(
                    session,
                    token_id=prev_token,
                    position=int(session.position) - 1,
                    mtp_block=resident_draft,
                )
                mtp_device_kv_capacity = max(
                    1,
                    len(mtp_context_tokens) + max_cycles * (2 * 2 + 2) + 4,
                )
                mtp_kv_alloc_start = time.perf_counter()
                mtp_key_cache, mtp_value_cache, mtp_buffers = _allocate_mtp_dense_kv(
                    runtime=session.runtime,
                    capacity=mtp_device_kv_capacity,
                    qk_head_dim=qk_head_dim,
                    kv_heads=2,
                )
                _timing_set(timing, "mtp_kv_alloc_ms", mtp_kv_alloc_start)
                slot = _GGUFMTPServingSlot(
                    request_id=int(entry["request_id"]),
                    prompt_ids=list(prompt_ids),
                    session=session,
                    resident_draft=resident_draft,
                    resident_context=resident_context,
                    mtp_key_cache=mtp_key_cache,
                    mtp_value_cache=mtp_value_cache,
                    mtp_buffers=mtp_buffers,
                    hidden_size=hidden_size,
                    prev_token=prev_token,
                    seq_position=seq_position,
                    generated_ids=generated_ids,
                    timing=timing,
                    session_pool_key=entry["session_pool_key"],
                    draft_pool_key=entry["draft_pool_key"],
                    done=(
                        len(generated_ids) >= int(request.max_tokens)
                        or _gguf_finished(generated_ids, self.tokenizer, request)
                    ),
                )
                if mtp_context_tokens:
                    context_positions = np.asarray(range(len(mtp_context_tokens)), dtype=np.int64)
                    context_tokens = np.asarray(mtp_context_tokens, dtype=np.int64)
                    context_write_start = time.perf_counter()
                    slot.mtp_device_kv_len = resident_draft.write_kv_rows(
                        mtp_context_hidden_rows,
                        context_tokens,
                        positions=context_positions,
                        rope_cos=assets.rope_cos,
                        rope_sin=assets.rope_sin,
                        dense_key_cache=mtp_key_cache,
                        dense_value_cache=mtp_value_cache,
                        dense_cache_len=0,
                    )
                    _timing_add(slot.timing, "mtp_context_write_ms", context_write_start)
                _timing_set(slot.timing, "slot_open_total_ms", float(entry["slot_open_start"]))
                entry["slot"] = slot
                slots.append(slot)
            return slots
        except NotImplementedError:
            close_acquired()
            return None
        except Exception:
            close_acquired()
            raise

    def _open_mtp_serving_slot(
        self,
        shared_runner: Qwen35GGUFFullStackRunner,
        assets: _GGUFMTPServingAssets,
        prompt_ids: list[int],
        request: GenerationRequest,
        *,
        request_id: int,
        pool_sessions: bool,
    ) -> _GGUFMTPServingSlot:
        from hipengine.core.hip import HipMemcpyKind

        slot_open_start = time.perf_counter()
        timing: dict[str, float] = {}
        session: Qwen35GGUFResidentSession | None = None
        resident_draft: Any | None = None
        mtp_buffers: list[Any] = []
        try:
            session_open_start = time.perf_counter()
            session, session_pool_key, _session_reused = self._acquire_shared_session(
                shared_runner,
                pool_name="mtp_target",
                use_wmma_prefill=_resident_session_wmma_prefill_default(),
                use_gemv_decode=True,
            ) if pool_sessions else (
                Qwen35GGUFResidentSession(
                    self.model_path,
                    runtime=shared_runner.runtime,
                    shared_runner=shared_runner,
                    use_wmma_prefill=_resident_session_wmma_prefill_default(),
                    use_gemv_decode=True,
                ),
                None,
                False,
            )
            _timing_set(timing, "session_open_ms", session_open_start)
            runtime = session.runtime
            draft_open_start = time.perf_counter()
            resident_draft, draft_pool_key, _draft_reused = self._acquire_mtp_draft_runner(
                assets,
                runtime=runtime,
                pool_enabled=pool_sessions,
            )
            _timing_set(timing, "draft_open_ms", draft_open_start)
            hidden_size = int(assets.token_embd_f32.shape[1])
            min_bulk_tokens = int(getattr(session.runner.weights.config, "ssm_conv_kernel", 4))
            if len(prompt_ids) >= min_bulk_tokens:
                prefill_start = time.perf_counter()
                prefill_result = session.prefill(
                    prompt_ids,
                    use_bulk=True,
                    bulk_attention_mode="bulk",
                    return_logits=False,
                    capture_hidden_seed_fp32=True,
                )
                _timing_add(timing, "prefill_ms", prefill_start)
                prompt_hidden_rows = np.empty((len(prompt_ids), hidden_size), dtype=np.float32)
                hidden_d2h_start = time.perf_counter()
                runtime.memcpy(
                    prompt_hidden_rows.ctypes.data,
                    session.fp32_verify_hidden_seed_ptr(0),
                    prompt_hidden_rows.nbytes,
                    HipMemcpyKind.DEVICE_TO_HOST,
                )
                _timing_add(timing, "prompt_hidden_d2h_ms", hidden_d2h_start)
                mtp_context_tokens, mtp_context_hidden_rows = _llama_cpp_mtp_catchup_rows(
                    prompt_ids,
                    prompt_hidden_rows,
                )
            else:
                prefill_start = time.perf_counter()
                prefill_result = session.prefill(
                    prompt_ids,
                    return_logits=False,
                    capture_hidden_seed_fp32=True,
                )
                _timing_add(timing, "prefill_ms", prefill_start)
                mtp_context_tokens = []
                mtp_context_hidden_rows = np.empty((0, hidden_size), dtype=np.float32)

            prev_token = int(prefill_result.token_id)
            generated_ids = [prev_token]
            seq_position = int(session.position)
            resident_context = _new_mtp_context(
                session,
                token_id=prev_token,
                position=int(session.position) - 1,
                mtp_block=resident_draft,
            )
            qk_head_dim = int(
                np.asarray(
                    assets.weights[f"blk.{assets.nextn_block_id}.attn_q_norm.weight"][0]
                ).shape[0]
            )
            max_cycles = max(1, int(request.max_tokens))
            mtp_device_kv_capacity = max(
                1,
                len(mtp_context_tokens) + max_cycles * (2 * 2 + 2) + 4,
            )
            mtp_kv_alloc_start = time.perf_counter()
            mtp_key_cache, mtp_value_cache, mtp_buffers = _allocate_mtp_dense_kv(
                runtime=runtime,
                capacity=mtp_device_kv_capacity,
                qk_head_dim=qk_head_dim,
                kv_heads=2,
            )
            _timing_set(timing, "mtp_kv_alloc_ms", mtp_kv_alloc_start)
            slot = _GGUFMTPServingSlot(
                request_id=int(request_id),
                prompt_ids=list(prompt_ids),
                session=session,
                resident_draft=resident_draft,
                resident_context=resident_context,
                mtp_key_cache=mtp_key_cache,
                mtp_value_cache=mtp_value_cache,
                mtp_buffers=mtp_buffers,
                hidden_size=hidden_size,
                prev_token=prev_token,
                seq_position=seq_position,
                generated_ids=generated_ids,
                timing=timing,
                session_pool_key=session_pool_key,
                draft_pool_key=draft_pool_key,
                done=(
                    len(generated_ids) >= int(request.max_tokens)
                    or _gguf_finished(generated_ids, self.tokenizer, request)
                ),
            )
            if mtp_context_tokens:
                context_positions = np.asarray(range(len(mtp_context_tokens)), dtype=np.int64)
                context_tokens = np.asarray(mtp_context_tokens, dtype=np.int64)
                context_write_start = time.perf_counter()
                slot.mtp_device_kv_len = resident_draft.write_kv_rows(
                    mtp_context_hidden_rows,
                    context_tokens,
                    positions=context_positions,
                    rope_cos=assets.rope_cos,
                    rope_sin=assets.rope_sin,
                    dense_key_cache=mtp_key_cache,
                    dense_value_cache=mtp_value_cache,
                    dense_cache_len=0,
                )
                _timing_add(slot.timing, "mtp_context_write_ms", context_write_start)
            _timing_set(slot.timing, "slot_open_total_ms", slot_open_start)
            return slot
        except Exception:
            if mtp_buffers and session is not None:
                _free_mtp_buffers(mtp_buffers, runtime=session.runtime)
            if resident_draft is not None:
                close = getattr(resident_draft, "close", None)
                if callable(close):
                    close()
            if session is not None:
                session.close()
            raise

    def _run_mtp_serving_slots(
        self,
        slots: list[_GGUFMTPServingSlot],
        assets: _GGUFMTPServingAssets,
        request: GenerationRequest,
        *,
        base_env: dict[str, str | None],
    ) -> None:
        while any(not slot.done for slot in slots):
            self._run_mtp_serving_slots_cycle(slots, assets, request, base_env=base_env)

    def _run_mtp_serving_slots_cycle(
        self,
        slots: list[_GGUFMTPServingSlot],
        assets: _GGUFMTPServingAssets,
        request: GenerationRequest,
        *,
        base_env: dict[str, str | None],
        verify_owner_session: Qwen35GGUFResidentSession | None = None,
    ) -> None:
        live_slots = [slot for slot in slots if not slot.done]
        if not live_slots:
            return
        cycle_start = time.perf_counter()
        drafted_cycles: list[_GGUFMTPDraftedCycle] = []

        draft_phase_start = time.perf_counter()
        stream_drafted = self._try_draft_mtp_serving_slots_streams(
            live_slots,
            assets,
            request,
            base_env=base_env,
        )
        if stream_drafted is None:
            for slot in live_slots:
                if slot.done:
                    continue
                raise_if_generation_deadline_expired(request)
                drafted = self._draft_mtp_serving_slot(slot, assets, request, base_env=base_env)
                if drafted is not None:
                    drafted_cycles.append(drafted)
        else:
            drafted_cycles.extend(stream_drafted)
        draft_phase_ms = _timing_ms_since(draft_phase_start)
        for slot in live_slots:
            _timing_add_ms(slot.timing, "slots_draft_phase_ms", draft_phase_ms)

        verify_phase_start = time.perf_counter()
        verified_cycles = self._verify_mtp_serving_cycles(
            drafted_cycles,
            request,
            verify_owner_session=verify_owner_session,
        )
        verify_phase_ms = _timing_ms_since(verify_phase_start)
        for slot in live_slots:
            _timing_add_ms(slot.timing, "slots_verify_phase_ms", verify_phase_ms)

        commit_phase_start = time.perf_counter()
        commit_index = 0
        try:
            for commit_index, verified in enumerate(verified_cycles):
                self._commit_mtp_serving_cycle(verified, assets, request)
        except Exception:
            for verified in verified_cycles[commit_index + 1:]:
                self._free_mtp_serving_cycle_snapshot(verified.drafted)
            raise
        commit_phase_ms = _timing_ms_since(commit_phase_start)
        for slot in live_slots:
            _timing_add_ms(slot.timing, "slots_commit_phase_ms", commit_phase_ms)
            _timing_add(slot.timing, "slots_cycle_wall_ms", cycle_start)

    def _advance_mtp_serving_slot(
        self,
        slot: _GGUFMTPServingSlot,
        assets: _GGUFMTPServingAssets,
        request: GenerationRequest,
        *,
        base_env: dict[str, str | None],
    ) -> None:
        drafted = self._draft_mtp_serving_slot(slot, assets, request, base_env=base_env)
        if drafted is None:
            return
        verified = self._verify_mtp_serving_cycle(drafted, request)
        self._commit_mtp_serving_cycle(verified, assets, request)

    def _try_draft_mtp_serving_slots_streams(
        self,
        live_slots: list[_GGUFMTPServingSlot],
        assets: _GGUFMTPServingAssets,
        request: GenerationRequest,
        *,
        base_env: dict[str, str | None],
    ) -> list[_GGUFMTPDraftedCycle] | None:
        if len(live_slots) <= 1 or not _gguf_mtp_server_stream_draft_enabled():
            return None
        for slot in live_slots:
            remaining = int(request.max_tokens) - len(slot.generated_ids)
            if remaining <= 1 or slot.resident_context.pending_seed is None:
                return None
            if not callable(getattr(slot.session.runtime, "stream_create", None)):
                return None
            if not callable(getattr(slot.session.runtime, "stream_synchronize", None)):
                return None
            if not callable(getattr(slot.session.runtime, "stream_destroy", None)):
                return None
        for slot in live_slots:
            if slot.draft_stream == 0:
                slot.draft_stream = int(slot.session.runtime.stream_create(nonblocking=True))

        def _draft(slot: _GGUFMTPServingSlot) -> _GGUFMTPDraftedCycle | None:
            raise_if_generation_deadline_expired(request)
            return self._draft_mtp_serving_slot(
                slot,
                assets,
                request,
                base_env=base_env,
                draft_stream=int(slot.draft_stream),
            )

        batch_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(live_slots)) as pool:
            futures = [(slot, pool.submit(_draft, slot)) for slot in live_slots]
            drafted: list[_GGUFMTPDraftedCycle] = []
            for _slot, future in futures:
                result = future.result()
                if result is not None:
                    drafted.append(result)
        batch_ms = _timing_ms_since(batch_start)
        for slot in live_slots:
            _timing_add_ms(slot.timing, "draft_stream_batch_ms", batch_ms)
        return drafted

    def _draft_mtp_serving_slot(
        self,
        slot: _GGUFMTPServingSlot,
        assets: _GGUFMTPServingAssets,
        request: GenerationRequest,
        *,
        base_env: dict[str, str | None],
        draft_stream: int = 0,
    ) -> _GGUFMTPDraftedCycle | None:
        advance_start = time.perf_counter()
        remaining = int(request.max_tokens) - len(slot.generated_ids)
        if remaining <= 0:
            slot.done = True
            _timing_add(slot.timing, "slot_advance_total_ms", advance_start)
            return None
        if remaining <= 1:
            ar_tail_start = time.perf_counter()
            with _exact_env(base_env):
                step = slot.session.step(
                    slot.prev_token,
                    return_logits=False,
                    capture_hidden_seed_fp32=True,
                )
            _timing_add(slot.timing, "ar_tail_ms", ar_tail_start)
            token = int(step.token_id)
            slot.generated_ids.append(token)
            slot.prev_token = token
            slot.seq_position += 1
            slot.cycles.append(
                {
                    "mode": "ar_tail",
                    "generated_draft_tokens": 0,
                    "accepted_draft_tokens": 0,
                    "visible_output_tokens": 1,
                }
            )
            slot.done = True
            _timing_add(slot.timing, "slot_advance_total_ms", advance_start)
            return None

        draft_n_max = min(2, remaining - 1)
        if slot.resident_context.pending_seed is None:
            raise RuntimeError("resident MTP context has no pending seed")
        cycle_mtp_kv_base_len = int(slot.mtp_device_kv_len)
        draft_start = time.perf_counter()
        draft_tokens, _draft_topk, slot.mtp_device_kv_len = slot.resident_draft.propose_chain_from_device_seed(
            int(slot.resident_context.pending_seed.hidden_ptr),
            start_token=slot.prev_token,
            start_position=slot.seq_position,
            draft_n_max=draft_n_max,
            top_k=1,
            rope_cos=assets.rope_cos,
            rope_sin=assets.rope_sin,
            dense_key_cache=slot.mtp_key_cache,
            dense_value_cache=slot.mtp_value_cache,
            dense_cache_len=slot.mtp_device_kv_len,
            draft_p_min=0.0,
            stream=draft_stream,
        )
        _timing_add(slot.timing, "draft_propose_ms", draft_start)
        draft_tokens = [int(token) for token in draft_tokens]
        if not draft_tokens:
            ar_tail_start = time.perf_counter()
            with _exact_env(base_env):
                step = slot.session.step(
                    slot.prev_token,
                    return_logits=False,
                    capture_hidden_seed_fp32=True,
                )
            _timing_add(slot.timing, "ar_tail_ms", ar_tail_start)
            token = int(step.token_id)
            slot.generated_ids.append(token)
            slot.prev_token = token
            slot.seq_position += 1
            slot.done = (
                len(slot.generated_ids) >= int(request.max_tokens)
                or _gguf_finished(slot.generated_ids, self.tokenizer, request)
            )
            _timing_add(slot.timing, "slot_advance_total_ms", advance_start)
            return None

        block_inputs = [int(slot.prev_token)] + draft_tokens
        block_start = int(slot.seq_position)
        direct_commit_exact = block_start + len(block_inputs) < 1024
        return _GGUFMTPDraftedCycle(
            slot=slot,
            advance_start=advance_start,
            cycle_mtp_kv_base_len=cycle_mtp_kv_base_len,
            draft_tokens=draft_tokens,
            block_inputs=block_inputs,
            block_start=block_start,
            direct_commit_exact=direct_commit_exact,
        )

    def _verify_mtp_serving_cycle(
        self,
        drafted: _GGUFMTPDraftedCycle,
        request: GenerationRequest,
    ) -> _GGUFMTPVerifiedCycle:
        return self._verify_mtp_serving_cycles([drafted], request)[0]

    def _verify_mtp_serving_cycles(
        self,
        drafted_cycles: list[_GGUFMTPDraftedCycle],
        request: GenerationRequest,
        *,
        verify_owner_session: Qwen35GGUFResidentSession | None = None,
    ) -> list[_GGUFMTPVerifiedCycle]:
        _ = request
        if not drafted_cycles:
            return []
        for drafted in drafted_cycles:
            slot = drafted.slot
            snapshot_start = time.perf_counter()
            snapshot = (
                slot.session._linear_state_snapshot()
                if not drafted.direct_commit_exact
                else None
            )
            drafted.snapshot = snapshot
            if snapshot is not None:
                _timing_add(slot.timing, "linear_state_snapshot_ms", snapshot_start)
        try:
            block_results = self._try_verify_mtp_serving_cycles_batch(
                drafted_cycles,
                verify_owner_session=verify_owner_session,
            )
            if block_results is None:
                block_results = []
                for drafted in drafted_cycles:
                    slot = drafted.slot
                    verify_start = time.perf_counter()
                    block_result = slot.session.verify_target_block(
                        drafted.block_inputs,
                        bulk_attention_mode="bulk",
                        use_wmma_prefill=_mtp_serving_target_wmma_for(self),
                        capture_linear_state_rows=True,
                        defer_linear_state_commit=True,
                    )
                    _timing_add(slot.timing, "target_verify_ms", verify_start)
                    block_results.append(block_result)
            if len(block_results) != len(drafted_cycles):
                raise RuntimeError(
                    f"MTP target batch verifier returned {len(block_results)} result(s) "
                    f"for {len(drafted_cycles)} drafted cycle(s)"
                )
            verified: list[_GGUFMTPVerifiedCycle] = []
            for drafted, block_result in zip(drafted_cycles, block_results, strict=True):
                block_target_tokens = [int(token) for token in block_result.token_ids]
                acceptance = _llama_cpp_acceptance_from_target_samples(
                    drafted.draft_tokens,
                    block_target_tokens,
                )
                verified.append(
                    _GGUFMTPVerifiedCycle(
                        drafted=drafted,
                        block_result=block_result,
                        block_target_tokens=block_target_tokens,
                        acceptance=acceptance,
                    )
                )
            return verified
        except Exception:
            for drafted in drafted_cycles:
                self._free_mtp_serving_cycle_snapshot(drafted)
            raise

    def _try_verify_mtp_serving_cycles_batch(
        self,
        drafted_cycles: list[_GGUFMTPDraftedCycle],
        *,
        verify_owner_session: Qwen35GGUFResidentSession | None = None,
    ) -> list[Any] | None:
        if len(drafted_cycles) <= 1:
            return None
        chunks: list[list[_GGUFMTPDraftedCycle]] = []
        index = 0
        while index < len(drafted_cycles):
            remaining = len(drafted_cycles) - index
            take = min(_MTP_SERVING_TARGET_BATCH_MAX_SLOTS, remaining)
            if remaining > _MTP_SERVING_TARGET_BATCH_MAX_SLOTS and remaining - take == 1:
                take -= 1
            chunks.append(drafted_cycles[index:index + take])
            index += take
        if len(chunks) > 1 and _gguf_mtp_server_stream_verify_enabled():
            streamed_results = self._try_verify_mtp_serving_cycle_chunks_streams(chunks)
            if streamed_results is not None:
                return streamed_results
        block_results: list[Any] = []
        for chunk in chunks:
            chunk_results = self._verify_mtp_serving_cycle_chunk(
                chunk,
                verify_owner_session=verify_owner_session,
            )
            if chunk_results is None:
                if block_results:
                    raise RuntimeError("packed target verifier chunk failed after a prior chunk advanced state")
                return None
            block_results.extend(chunk_results)
        return block_results

    def _try_verify_mtp_serving_cycle_chunks_streams(
        self,
        chunks: list[list[_GGUFMTPDraftedCycle]],
    ) -> list[Any] | None:
        if len(chunks) <= 1:
            return None
        for chunk in chunks:
            if not chunk:
                return None
            owner_slot = chunk[0].slot
            runtime = getattr(owner_slot.session, "runtime", None)
            if not callable(getattr(runtime, "stream_create", None)):
                return None
            if not callable(getattr(runtime, "stream_synchronize", None)):
                return None
            if not callable(getattr(runtime, "stream_destroy", None)):
                return None
            if not callable(getattr(owner_slot.session, "verify_target_blocks_batch", None)):
                return None
        for chunk in chunks:
            owner_slot = chunk[0].slot
            if owner_slot.verify_stream == 0:
                owner_slot.verify_stream = int(owner_slot.session.runtime.stream_create(nonblocking=True))

        def verify_chunk(chunk: list[_GGUFMTPDraftedCycle]) -> list[Any] | None:
            return self._verify_mtp_serving_cycle_chunk(
                chunk,
                stream=int(chunk[0].slot.verify_stream),
            )

        stream_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [(chunk, pool.submit(verify_chunk, chunk)) for chunk in chunks]
            chunk_results: list[tuple[list[_GGUFMTPDraftedCycle], list[Any]]] = []
            for chunk, future in futures:
                result = future.result()
                if result is None:
                    raise RuntimeError("streamed packed target verifier chunk failed after launch")
                chunk_results.append((chunk, result))
        stream_ms = _timing_ms_since(stream_start)
        block_results: list[Any] = []
        for chunk, result in chunk_results:
            for drafted in chunk:
                _timing_add_ms(drafted.slot.timing, "target_verify_stream_chunks_ms", stream_ms)
            block_results.extend(result)
        return block_results

    def _verify_mtp_serving_cycle_chunk(
        self,
        chunk: list[_GGUFMTPDraftedCycle],
        *,
        stream: int = 0,
        verify_owner_session: Qwen35GGUFResidentSession | None = None,
    ) -> list[Any] | None:
        first_session = verify_owner_session or chunk[0].slot.session
        verify_batch = getattr(first_session, "verify_target_blocks_batch", None)
        if not callable(verify_batch):
            return None
        defer_state_scatter = _gguf_mtp_server_defer_verify_scatter_enabled()
        jobs = [
            {
                "session": drafted.slot.session,
                "input_token_ids": tuple(int(token) for token in drafted.block_inputs),
                "bulk_attention_mode": "bulk",
                "use_wmma_prefill": _mtp_serving_target_wmma_for(self),
                "capture_linear_state_rows": True,
                "defer_linear_state_commit": True,
                "defer_state_scatter": defer_state_scatter,
            }
            for drafted in chunk
        ]
        verify_start = time.perf_counter()
        try:
            if stream:
                batch_result = verify_batch(jobs, stream=int(stream))
            else:
                batch_result = verify_batch(jobs)
        except NotImplementedError:
            return None
        if batch_result is None:
            return None
        chunk_results = list(batch_result)
        verify_ms = _timing_ms_since(verify_start)
        packed_stage_timings = getattr(first_session, "last_packed_verify_stage_timings_ms", {})
        for drafted in chunk:
            _timing_add_ms(drafted.slot.timing, "target_verify_ms", verify_ms)
            _timing_add_ms(drafted.slot.timing, "target_verify_batch_ms", verify_ms)
            if isinstance(packed_stage_timings, dict):
                for stage_name, stage_ms in packed_stage_timings.items():
                    _timing_add_ms(drafted.slot.timing, f"target_{stage_name}_ms", float(stage_ms))
        return chunk_results

    def _commit_mtp_serving_cycle(
        self,
        verified: _GGUFMTPVerifiedCycle,
        assets: _GGUFMTPServingAssets,
        request: GenerationRequest,
    ) -> None:
        drafted = verified.drafted
        slot = drafted.slot
        snapshot = drafted.snapshot
        block_inputs = drafted.block_inputs
        block_start = drafted.block_start
        block_target_tokens = verified.block_target_tokens
        acceptance = verified.acceptance
        accepted_draft_tokens = int(acceptance["accepted_draft_tokens"])
        consumed_rows = accepted_draft_tokens + 1
        try:
            state_commit_start = time.perf_counter()
            captured_rows = bool(
                getattr(verified.block_result, "linear_state_rows_captured", False)
            )
            deferred_packed_state = getattr(
                verified.block_result,
                "deferred_packed_state",
                None,
            )
            seed_row_count = (
                consumed_rows
                if deferred_packed_state is not None
                else len(block_target_tokens)
            )
            if consumed_rows < len(block_inputs) or drafted.direct_commit_exact:
                if not captured_rows:
                    raise RuntimeError(
                        "direct MTP commit requested without captured linear-state rows"
                    )
                if deferred_packed_state is not None:
                    owner = getattr(deferred_packed_state, "owner", None)
                    commit_deferred = getattr(
                        owner,
                        "_commit_deferred_packed_verify_state",
                        None,
                    )
                    if not callable(commit_deferred):
                        raise RuntimeError(
                            "deferred packed verifier state owner cannot commit rows"
                        )
                    commit_deferred(
                        deferred_packed_state,
                        slot.session,
                        commit_row_index=consumed_rows - 1,
                        position=block_start + consumed_rows,
                        hidden_rows=seed_row_count,
                    )
                else:
                    slot.session._commit_verify_linear_state_row(
                        consumed_rows - 1,
                        position=block_start + consumed_rows,
                    )
            else:
                if snapshot is None:
                    raise RuntimeError(
                        "MTP full-block replay requested without a linear-state snapshot"
                    )
                slot.session._restore_linear_state_snapshot(
                    snapshot,
                    position=block_start,
                )
                replay_result = slot.session.verify_target_block_serial_exact(block_inputs)
                replay_tokens = [int(token) for token in replay_result.token_ids]
                if replay_tokens != block_target_tokens:
                    raise RuntimeError(
                        "MTP serial-exact replay diverged from block verifier rows"
                    )
            _timing_add(slot.timing, "target_state_commit_ms", state_commit_start)
            target_verify_seed_rows = [
                _new_mtp_seed_row(
                    token_id=block_target_tokens[row],
                    position=block_start + row,
                    hidden_ptr=slot.session.fp32_verify_hidden_seed_ptr(row),
                    hidden_size=slot.hidden_size,
                    source=f"verify[{row}]",
                )
                for row in range(seed_row_count)
            ]
        finally:
            self._free_mtp_serving_cycle_snapshot(drafted)

        output_tokens = [int(token) for token in acceptance["output_tokens"]]
        slot.resident_context.record_verify_seeds(target_verify_seed_rows)
        slot.resident_context.accept(accepted_draft_tokens)
        slot.mtp_device_kv_len = min(slot.mtp_device_kv_len, drafted.cycle_mtp_kv_base_len + 1)
        if accepted_draft_tokens > 0:
            commit_tokens = np.asarray(output_tokens[:accepted_draft_tokens], dtype=np.int64)
            commit_positions = np.arange(
                slot.seq_position + 1,
                slot.seq_position + 1 + accepted_draft_tokens,
                dtype=np.int64,
            )
            kv_commit_start = time.perf_counter()
            slot.mtp_device_kv_len = slot.resident_draft.write_kv_rows_from_device_seed_base(
                int(target_verify_seed_rows[0].hidden_ptr),
                commit_tokens,
                positions=commit_positions,
                rope_cos=assets.rope_cos,
                rope_sin=assets.rope_sin,
                dense_key_cache=slot.mtp_key_cache,
                dense_value_cache=slot.mtp_value_cache,
                dense_cache_len=slot.mtp_device_kv_len,
            )
            _timing_add(slot.timing, "mtp_kv_commit_ms", kv_commit_start)

        slot.cycles.append(
            {
                "mode": "llama_compat_direct_commit",
                "generated_draft_tokens": len(drafted.draft_tokens),
                "accepted_draft_tokens": accepted_draft_tokens,
                "visible_output_tokens": len(output_tokens),
            }
        )
        slot.prev_token = int(output_tokens[-1])
        slot.seq_position += len(output_tokens)
        stop = False
        for token in output_tokens:
            if len(slot.generated_ids) >= int(request.max_tokens):
                break
            slot.generated_ids.append(int(token))
            if _gguf_finished(slot.generated_ids, self.tokenizer, request):
                stop = True
                break
        slot.done = stop or len(slot.generated_ids) >= int(request.max_tokens)
        _timing_add(slot.timing, "slot_advance_total_ms", drafted.advance_start)

    def _free_mtp_serving_cycle_snapshot(self, drafted: _GGUFMTPDraftedCycle) -> None:
        snapshot = drafted.snapshot
        if snapshot is None:
            return
        slot = drafted.slot
        snapshot_free_start = time.perf_counter()
        slot.session._free_linear_state_snapshot(snapshot)
        drafted.snapshot = None
        _timing_add(slot.timing, "linear_state_snapshot_free_ms", snapshot_free_start)

    def _close_mtp_serving_slots(self, slots: list[_GGUFMTPServingSlot], *, reuse: bool = True) -> None:
        for slot in reversed(slots):
            if slot.verify_stream:
                slot.session.runtime.stream_destroy(int(slot.verify_stream))
                slot.verify_stream = 0
            if slot.draft_stream:
                slot.session.runtime.stream_destroy(int(slot.draft_stream))
                slot.draft_stream = 0
            _free_mtp_buffers(slot.mtp_buffers, runtime=slot.session.runtime)
            self._release_mtp_draft_runner(
                slot.draft_pool_key if reuse else None,
                slot.resident_draft,
            )
            if reuse:
                self._release_shared_session(slot.session_pool_key, slot.session)
            else:
                slot.session.close()

    def _generate_speculative_mtp_llama_compat(
        self,
        session: Qwen35GGUFResidentSession,
        resident_draft: Any,
        assets: _GGUFMTPServingAssets,
        prompt_ids: list[int],
        request: GenerationRequest,
        *,
        base_env: dict[str, str | None],
    ) -> "_GGUFMTPServingRun":
        from hipengine.core.hip import HipMemcpyKind

        run_start = time.perf_counter()
        timing: dict[str, float] = {}
        runtime = session.runtime
        hidden_size = int(assets.token_embd_f32.shape[1])
        min_bulk_tokens = int(getattr(session.runner.weights.config, "ssm_conv_kernel", 4))
        if len(prompt_ids) >= min_bulk_tokens:
            prefill_start = time.perf_counter()
            prefill_result = session.prefill(
                prompt_ids,
                use_bulk=True,
                bulk_attention_mode="bulk",
                return_logits=False,
                capture_hidden_seed_fp32=True,
            )
            _timing_add(timing, "prefill_ms", prefill_start)
            prompt_hidden_rows = np.empty((len(prompt_ids), hidden_size), dtype=np.float32)
            hidden_d2h_start = time.perf_counter()
            runtime.memcpy(
                prompt_hidden_rows.ctypes.data,
                session.fp32_verify_hidden_seed_ptr(0),
                prompt_hidden_rows.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            _timing_add(timing, "prompt_hidden_d2h_ms", hidden_d2h_start)
            mtp_context_tokens, mtp_context_hidden_rows = _llama_cpp_mtp_catchup_rows(
                prompt_ids,
                prompt_hidden_rows,
            )
        else:
            prefill_start = time.perf_counter()
            prefill_result = session.prefill(
                prompt_ids,
                return_logits=False,
                capture_hidden_seed_fp32=True,
            )
            _timing_add(timing, "prefill_ms", prefill_start)
            mtp_context_tokens = []
            mtp_context_hidden_rows = np.empty((0, hidden_size), dtype=np.float32)

        prev_token = int(prefill_result.token_id)
        generated_ids = [prev_token]
        if _gguf_finished(generated_ids, self.tokenizer, request):
            _timing_set(timing, "mtp_run_total_ms", run_start)
            return _GGUFMTPServingRun(generated_ids=generated_ids, cycles=[], timing=timing)

        seq_position = int(session.position)
        resident_context = _new_mtp_context(
            session,
            token_id=prev_token,
            position=int(session.position) - 1,
            mtp_block=resident_draft,
        )
        qk_head_dim = int(
            np.asarray(
                assets.weights[f"blk.{assets.nextn_block_id}.attn_q_norm.weight"][0]
            ).shape[0]
        )
        max_cycles = max(1, int(request.max_tokens))
        mtp_device_kv_capacity = max(
            1,
            len(mtp_context_tokens) + max_cycles * (2 * 2 + 2) + 4,
        )
        mtp_kv_alloc_start = time.perf_counter()
        mtp_key_cache, mtp_value_cache, mtp_buffers = _allocate_mtp_dense_kv(
            runtime=runtime,
            capacity=mtp_device_kv_capacity,
            qk_head_dim=qk_head_dim,
            kv_heads=2,
        )
        _timing_set(timing, "mtp_kv_alloc_ms", mtp_kv_alloc_start)
        cycles: list[dict[str, Any]] = []
        mtp_device_kv_len = 0
        try:
            if mtp_context_tokens:
                context_positions = np.asarray(range(len(mtp_context_tokens)), dtype=np.int64)
                context_tokens = np.asarray(mtp_context_tokens, dtype=np.int64)
                context_write_start = time.perf_counter()
                mtp_device_kv_len = resident_draft.write_kv_rows(
                    mtp_context_hidden_rows,
                    context_tokens,
                    positions=context_positions,
                    rope_cos=assets.rope_cos,
                    rope_sin=assets.rope_sin,
                    dense_key_cache=mtp_key_cache,
                    dense_value_cache=mtp_value_cache,
                    dense_cache_len=0,
                )
                _timing_add(timing, "mtp_context_write_ms", context_write_start)

            while len(generated_ids) < int(request.max_tokens):
                raise_if_generation_deadline_expired(request)
                remaining = int(request.max_tokens) - len(generated_ids)
                if remaining <= 1:
                    ar_tail_start = time.perf_counter()
                    with _exact_env(base_env):
                        step = session.step(
                            prev_token,
                            return_logits=False,
                            capture_hidden_seed_fp32=True,
                        )
                    _timing_add(timing, "ar_tail_ms", ar_tail_start)
                    token = int(step.token_id)
                    generated_ids.append(token)
                    cycles.append(
                        {
                            "mode": "ar_tail",
                            "generated_draft_tokens": 0,
                            "accepted_draft_tokens": 0,
                            "visible_output_tokens": 1,
                        }
                    )
                    break

                draft_n_max = min(2, remaining - 1)
                if resident_context.pending_seed is None:
                    raise RuntimeError("resident MTP context has no pending seed")
                cycle_mtp_kv_base_len = int(mtp_device_kv_len)

                # N3 is the public c=1 adapter boundary: one call owns proposal,
                # target accept/commit, GGUF reseed, MTP-KV repair, and cursors.
                # Backends/shapes without the registered N2 target graph preserve
                # the established exact loop below.
                native_cycle = getattr(session, "run_native_spec_mtp_cycle", None)
                native_result = None
                if callable(native_cycle):
                    from hipengine.runtime.gguf_native_spec_cycle import (
                        NativeSpecTargetGraphUnsupportedError,
                    )

                    try:
                        native_result = native_cycle(
                            resident_draft,
                            resident_context,
                            root_token=prev_token,
                            root_position=seq_position,
                            candidate_budget=draft_n_max,
                            remaining_decode=remaining,
                            rope_cos=assets.rope_cos,
                            rope_sin=assets.rope_sin,
                            draft_key_cache=mtp_key_cache,
                            draft_value_cache=mtp_value_cache,
                            draft_cache_len=mtp_device_kv_len,
                            cycle_id=len(cycles),
                            transaction_id=len(cycles),
                        )
                    except NativeSpecTargetGraphUnsupportedError:
                        native_result = None
                    raise_if_generation_deadline_expired(request)
                if native_result is not None:
                    mtp_device_kv_len = int(native_result.draft_cache_len_after)
                    output_tokens = [int(token) for token in native_result.output_token_ids]
                    accepted_draft_tokens = int(native_result.accepted_draft_tokens)
                    _timing_add_ms(timing, "draft_propose_ms", native_result.proposal_wall_ms)
                    _timing_add_ms(timing, "target_verify_ms", native_result.target_wall_ms)
                    _timing_add_ms(
                        timing,
                        "mtp_kv_commit_ms",
                        native_result.mtp_kv_commit_wall_ms,
                    )
                    _timing_add_ms(timing, "native_complete_cycle_ms", native_result.call_wall_ms)
                    cycles.append(
                        {
                            "mode": "llama_compat_native_complete_cycle",
                            "generated_draft_tokens": len(native_result.draft_token_ids),
                            "accepted_draft_tokens": accepted_draft_tokens,
                            "visible_output_tokens": len(output_tokens),
                        }
                    )
                    prev_token = int(output_tokens[-1])
                    seq_position = int(native_result.end_position)
                    stop = False
                    for token in output_tokens:
                        if len(generated_ids) >= int(request.max_tokens):
                            break
                        generated_ids.append(int(token))
                        if _gguf_finished(generated_ids, self.tokenizer, request):
                            stop = True
                            break
                    if stop:
                        break
                    continue

                draft_start = time.perf_counter()
                draft_tokens, _draft_topk, mtp_device_kv_len = resident_draft.propose_chain_from_device_seed(
                    int(resident_context.pending_seed.hidden_ptr),
                    start_token=prev_token,
                    start_position=seq_position,
                    draft_n_max=draft_n_max,
                    top_k=1,
                    rope_cos=assets.rope_cos,
                    rope_sin=assets.rope_sin,
                    dense_key_cache=mtp_key_cache,
                    dense_value_cache=mtp_value_cache,
                    dense_cache_len=mtp_device_kv_len,
                    draft_p_min=0.0,
                )
                _timing_add(timing, "draft_propose_ms", draft_start)
                raise_if_generation_deadline_expired(request)
                draft_tokens = [int(token) for token in draft_tokens]
                if not draft_tokens:
                    ar_tail_start = time.perf_counter()
                    step = session.step(
                        prev_token,
                        return_logits=False,
                        capture_hidden_seed_fp32=True,
                    )
                    _timing_add(timing, "ar_tail_ms", ar_tail_start)
                    token = int(step.token_id)
                    generated_ids.append(token)
                    prev_token = token
                    seq_position += 1
                    continue

                block_inputs = [int(prev_token)] + draft_tokens
                block_start = int(seq_position)
                direct_commit_exact = block_start + len(block_inputs) < 1024
                snapshot_start = time.perf_counter()
                snapshot = None if direct_commit_exact else session._linear_state_snapshot()
                if snapshot is not None:
                    _timing_add(timing, "linear_state_snapshot_ms", snapshot_start)
                try:
                    verify_start = time.perf_counter()
                    block_result = session.verify_target_block(
                        block_inputs,
                        bulk_attention_mode="bulk",
                        use_wmma_prefill=_mtp_serving_target_wmma_for(self),
                        capture_linear_state_rows=True,
                        defer_linear_state_commit=True,
                    )
                    _timing_add(timing, "target_verify_ms", verify_start)
                    block_target_tokens = [int(token) for token in block_result.token_ids]
                    acceptance = _llama_cpp_acceptance_from_target_samples(
                        draft_tokens,
                        block_target_tokens,
                    )
                    accepted_draft_tokens = int(acceptance["accepted_draft_tokens"])
                    consumed_rows = accepted_draft_tokens + 1
                    state_commit_start = time.perf_counter()
                    captured_rows = bool(
                        getattr(block_result, "linear_state_rows_captured", False)
                    )
                    if consumed_rows < len(block_inputs) or direct_commit_exact:
                        if not captured_rows:
                            raise RuntimeError(
                                "direct MTP commit requested without captured linear-state rows"
                            )
                        session._commit_verify_linear_state_row(
                            consumed_rows - 1,
                            position=block_start + consumed_rows,
                        )
                    else:
                        if snapshot is None:
                            raise RuntimeError(
                                "MTP full-block replay requested without a linear-state snapshot"
                            )
                        session._restore_linear_state_snapshot(
                            snapshot,
                            position=block_start,
                        )
                        replay_result = session.verify_target_block_serial_exact(block_inputs)
                        replay_tokens = [int(token) for token in replay_result.token_ids]
                        if replay_tokens != block_target_tokens:
                            raise RuntimeError(
                                "MTP serial-exact replay diverged from block verifier rows"
                            )
                    _timing_add(timing, "target_state_commit_ms", state_commit_start)
                    seed_row_count = len(block_target_tokens)
                    target_verify_seed_rows = [
                        _new_mtp_seed_row(
                            token_id=block_target_tokens[row],
                            position=block_start + row,
                            hidden_ptr=session.fp32_verify_hidden_seed_ptr(row),
                            hidden_size=hidden_size,
                            source=f"verify[{row}]",
                        )
                        for row in range(seed_row_count)
                    ]
                finally:
                    if snapshot is not None:
                        snapshot_free_start = time.perf_counter()
                        session._free_linear_state_snapshot(snapshot)
                        _timing_add(timing, "linear_state_snapshot_free_ms", snapshot_free_start)

                output_tokens = [int(token) for token in acceptance["output_tokens"]]
                resident_context.record_verify_seeds(target_verify_seed_rows)
                resident_context.accept(accepted_draft_tokens)
                mtp_device_kv_len = min(mtp_device_kv_len, cycle_mtp_kv_base_len + 1)
                if accepted_draft_tokens > 0:
                    commit_tokens = np.asarray(output_tokens[:accepted_draft_tokens], dtype=np.int64)
                    commit_positions = np.arange(
                        seq_position + 1,
                        seq_position + 1 + accepted_draft_tokens,
                        dtype=np.int64,
                    )
                    kv_commit_start = time.perf_counter()
                    mtp_device_kv_len = resident_draft.write_kv_rows_from_device_seed_base(
                        int(target_verify_seed_rows[0].hidden_ptr),
                        commit_tokens,
                        positions=commit_positions,
                        rope_cos=assets.rope_cos,
                        rope_sin=assets.rope_sin,
                        dense_key_cache=mtp_key_cache,
                        dense_value_cache=mtp_value_cache,
                        dense_cache_len=mtp_device_kv_len,
                    )
                    _timing_add(timing, "mtp_kv_commit_ms", kv_commit_start)

                cycles.append(
                    {
                        "mode": "llama_compat_direct_commit",
                        "generated_draft_tokens": len(draft_tokens),
                        "accepted_draft_tokens": accepted_draft_tokens,
                        "visible_output_tokens": len(output_tokens),
                    }
                )
                prev_token = int(output_tokens[-1])
                seq_position += len(output_tokens)
                stop = False
                for token in output_tokens:
                    if len(generated_ids) >= int(request.max_tokens):
                        break
                    generated_ids.append(int(token))
                    if _gguf_finished(generated_ids, self.tokenizer, request):
                        stop = True
                        break
                if stop:
                    break
        finally:
            _free_mtp_buffers(mtp_buffers, runtime=runtime)

        _timing_set(timing, "mtp_run_total_ms", run_start)
        return _GGUFMTPServingRun(generated_ids=generated_ids, cycles=cycles, timing=timing)

    def _generate_greedy(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
        *,
        timing: dict[str, float] | None = None,
    ) -> list[int]:
        generated_ids: list[int] = []
        raise_if_generation_deadline_expired(request)
        prefill_start = time.perf_counter()
        result = session.prefill(prompt_ids, return_logits=False)
        if timing is not None:
            _timing_add(timing, "prefill_ms", prefill_start)
        raise_if_generation_deadline_expired(request)
        generated_ids.append(int(result.token_id))
        if request.ignore_eos or int(result.token_id) != self.tokenizer.eos_token_id:
            remaining = request.max_tokens - 1
            if remaining > 0:
                decode_start = time.perf_counter()
                minimum_fn = getattr(session, "decode_graph_min_replay_steps", None)
                minimum = minimum_fn() if callable(minimum_fn) else None
                use_graph = bool(
                    _gguf_decode_graph_enabled()
                    and minimum is not None
                    and remaining >= int(minimum)
                    and callable(getattr(session, "capture_decode_graph", None))
                )
                if use_graph:
                    graph = session.capture_decode_graph(
                        position=int(session.position),
                        steps_per_replay=1,
                        max_replay_steps=remaining,
                        attention_max_context_len=int(session.position) + remaining,
                    )
                    try:
                        for _ in range(remaining):
                            raise_if_generation_deadline_expired(request)
                            graph.replay(1)
                            step = graph.read_sample(return_logits=False)
                            raise_if_generation_deadline_expired(request)
                            generated_ids.append(int(step.token_id))
                            if (
                                not request.ignore_eos
                                and int(step.token_id) == self.tokenizer.eos_token_id
                            ):
                                break
                    finally:
                        graph.close()
                else:
                    for _ in range(remaining):
                        raise_if_generation_deadline_expired(request)
                        step = session.step(generated_ids[-1], return_logits=False)
                        raise_if_generation_deadline_expired(request)
                        generated_ids.append(int(step.token_id))
                        if (
                            not request.ignore_eos
                            and int(step.token_id) == self.tokenizer.eos_token_id
                        ):
                            break
                if timing is not None:
                    _timing_add(timing, "decode_ms", decode_start)
        return generated_ids

    def _generate_greedy_batch(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_rows_by_request: dict[int, list[int]],
        request: GenerationRequest,
        *,
        capacity: int,
    ) -> _GGUFNativeBatchRun:
        """Continuously admit and compact greedy rows around native c-aware decode."""

        if len(prompt_rows_by_request) <= 1:
            raise ValueError("native GGUF greedy batch requires at least two prompts")
        if int(capacity) < 2 or int(capacity) > 8:
            raise ValueError("native GGUF greedy batch capacity must be within [2, 8]")

        scheduler = ResidentBatchScheduler(capacity=int(capacity), context_bucket_size=256)
        for request_id in sorted(prompt_rows_by_request):
            scheduler.submit(
                prompt_rows_by_request[request_id],
                max_new_tokens=int(request.max_tokens),
                request_id=int(request_id),
                sampling_row_index=int(request_id),
            )

        native_decode_steps = 0
        single_row_tail_steps = 0
        eager_native_ready = False
        execution_paths: dict[str, str] = {}
        graph_objects: list[object] = []
        graph_bucket_labels: list[str] = []
        active_c_histogram: dict[int, int] = {}
        admission_history: list[dict[str, int]] = []
        admission_waves = 0
        reclaim_count = 0
        compaction_events = 0
        compacted_slot_moves = 0
        mixed_prefill_decode_admissions = 0
        peak_active_rows = 0

        config = getattr(getattr(getattr(session, "runner", None), "weights", None), "config", None)
        experts_per_token = int(getattr(config, "expert_used_count", 0) or 0)
        kv_storage = getattr(getattr(session, "kv_storage_dtype", None), "value", "bf16")
        resident_max_sequence_length = int(
            getattr(
                getattr(session, "target_layout", None),
                "max_sequence_length",
                max(len(tokens) for tokens in prompt_rows_by_request.values())
                + int(request.max_tokens),
            )
        )

        def compact_after_reclaim(completed_count: int) -> None:
            nonlocal compaction_events, compacted_slot_moves
            if completed_count <= 0:
                return
            moves = scheduler.compact()
            if scheduler.active_count or scheduler.pending_count:
                source_slots = tuple(int(move.old_slot) for move in moves)
                session.compact_target_slots(source_slots)
                compaction_events += 1
                compacted_slot_moves += sum(
                    int(move.old_slot) != int(move.new_slot) for move in moves
                )

        def admit_available() -> None:
            nonlocal admission_waves, reclaim_count, mixed_prefill_decode_admissions, peak_active_rows
            while scheduler.pending_count and scheduler.active_count < int(capacity):
                admitted = scheduler.admit_pending()
                if not admitted:
                    break
                admission_waves += 1
                if native_decode_steps or single_row_tail_steps:
                    mixed_prefill_decode_admissions += len(admitted)
                peak_active_rows = max(peak_active_rows, scheduler.active_count)
                completed_in_wave = 0
                for request_id in admitted:
                    work = scheduler.next_prefill_work(
                        chunk_size=resident_max_sequence_length,
                    )
                    if work is None or work.request_ids != (int(request_id),):
                        raise RuntimeError("GGUF scheduler prefill order diverged from admission order")
                    slot = scheduler.active_batch.slot_for(request_id)
                    admission_history.append(
                        {
                            "request_id": int(request_id),
                            "slot": int(slot),
                            "wave": int(admission_waves),
                        }
                    )
                    raise_if_generation_deadline_expired(request)
                    result = session.prefill_slot(
                        work.token_rows[0],
                        slot=slot,
                        return_logits=False,
                    )
                    first_token = int(result.token_id)
                    finished = (
                        not request.ignore_eos
                        and first_token == int(self.tokenizer.eos_token_id)
                    )
                    completed = scheduler.record_generated(
                        (GeneratedToken(int(request_id), first_token, finished=finished),)
                    )
                    completed_in_wave += len(completed)
                    reclaim_count += len(completed)
                compact_after_reclaim(completed_in_wave)

        def graph_for(key):
            def create_graph(_key):
                current_positions = tuple(
                    int(position)
                    for position in getattr(
                        session,
                        "row_positions",
                        (max(0, int(_key.context_bucket) - 1),) * int(_key.active_c),
                    )[: int(_key.active_c)]
                )
                required_bound = max(current_positions) + 1
                max_sequence_length = resident_max_sequence_length
                context_bound = min(
                    max_sequence_length,
                    max(required_bound, int(_key.context_bucket)),
                )
                graph = session.capture_native_rows_graph(
                    rows=int(_key.active_c),
                    max_context_len=context_bound,
                )
                graph_objects.append(graph)
                return graph

            return scheduler.graph_buckets.get_or_create(
                key,
                create_graph,
                miss_reason="gguf_native_shape_absent",
            )

        try:
            while scheduler.pending_count or scheduler.active_count:
                admit_available()
                if scheduler.active_count == 0:
                    continue
                work = scheduler.next_decode_work(
                    top_k=experts_per_token,
                    experts_per_token=experts_per_token,
                    replay_steps=1,
                    kv_storage_dtype=str(kv_storage),
                    layer_plan="qwen35_gguf_native",
                )
                if work is None:
                    raise RuntimeError("GGUF scheduler has active rows but no decode work")
                request_ids = tuple(int(request_id) for request_id in work.request_ids)
                current_tokens = tuple(
                    int(scheduler.active_batch.requests[request_id].generated_tokens[-1])
                    for request_id in request_ids
                )
                active_rows = len(request_ids)
                active_c_histogram[active_rows] = active_c_histogram.get(active_rows, 0) + 1
                key = scheduler.shape_key(
                    mode=WorkKind.DECODE,
                    top_k=experts_per_token,
                    experts_per_token=experts_per_token,
                    replay_steps=1,
                    kv_storage_dtype=str(kv_storage),
                    layer_plan="qwen35_gguf_native",
                )
                bucket_label = (
                    f"decode:c={key.active_c}:ctx={key.context_bucket}:"
                    f"mask={''.join('1' if active else '0' for active in key.active_mask)}:"
                    f"top_k={key.top_k}:experts={key.experts_per_token}"
                )
                if bucket_label not in graph_bucket_labels:
                    graph_bucket_labels.append(bucket_label)

                raise_if_generation_deadline_expired(request)
                if active_rows == 1:
                    step = session.step(current_tokens[0], return_logits=False)
                    next_tokens = (int(step.token_id),)
                    single_row_tail_steps += 1
                    execution_paths["single_row_tail"] = "resident_slot0_c1"
                elif not eager_native_ready or getattr(session, "host_token_embedding_enabled", False):
                    step = session.step_rows_native(current_tokens, return_logits=False)
                    next_tokens = tuple(int(token) for token in step.token_ids)
                    execution_paths.update(dict(step.execution_paths))
                    native_decode_steps += 1
                    eager_native_ready = True
                else:
                    graph = graph_for(key)
                    step = graph.step(current_tokens)
                    scheduler.graph_buckets.record_replay_kernel_hit()
                    next_tokens = tuple(int(token) for token in step.token_ids)
                    execution_paths.update(dict(step.execution_paths))
                    native_decode_steps += 1

                completed = scheduler.record_generated(
                    tuple(
                        GeneratedToken(
                            request_id,
                            token_id,
                            finished=(
                                not request.ignore_eos
                                and token_id == int(self.tokenizer.eos_token_id)
                            ),
                        )
                        for request_id, token_id in zip(request_ids, next_tokens, strict=True)
                    )
                )
                reclaim_count += len(completed)
                compact_after_reclaim(len(completed))
                peak_active_rows = max(peak_active_rows, scheduler.active_count)
        finally:
            for graph in reversed(graph_objects):
                close = getattr(graph, "close", None)
                if callable(close):
                    close()
                    continue
                exit_graph = getattr(graph, "__exit__", None)
                if callable(exit_graph):
                    exit_graph(None, None, None)

        generated = {
            request_id: list(scheduler.completed[request_id].generated_tokens)
            for request_id in sorted(prompt_rows_by_request)
        }
        graph_stats = scheduler.graph_buckets.stats.to_json_dict()
        scheduling = {
            "continuous_batching": True,
            "capacity": int(capacity),
            "admission_count": len(admission_history),
            "admission_waves": int(admission_waves),
            "admission_history": admission_history,
            "reclaim_count": int(reclaim_count),
            "compaction_events": int(compaction_events),
            "compacted_slot_moves": int(compacted_slot_moves),
            "mixed_prefill_decode_admissions": int(mixed_prefill_decode_admissions),
            "peak_active_rows": int(peak_active_rows),
            "active_c_histogram": {
                str(active_rows): int(steps)
                for active_rows, steps in sorted(active_c_histogram.items())
            },
            "graph_bucket_keys": graph_bucket_labels,
            "graph_bucket_stats": graph_stats,
            "single_row_tail_steps": int(single_row_tail_steps),
            "stable_request_ids": sorted(prompt_rows_by_request),
            "request_observability": {
                str(request_id): scheduler.completed[request_id].observability.to_json_dict()
                for request_id in sorted(prompt_rows_by_request)
            },
            "final_request_to_slot": scheduler.active_batch.request_to_slot,
            "serial_decode_fallback": False,
        }
        return _GGUFNativeBatchRun(
            generated_ids=generated,
            native_decode_steps=native_decode_steps,
            execution_paths=execution_paths,
            scheduling=scheduling,
        )

    def _generate_sampled(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
        *,
        row_index: int,
        timing: dict[str, float] | None = None,
    ) -> GenerationOutput:
        sampling_request = _request_with_tokenizer_eos(request, self.tokenizer)
        state = _gguf_row_sampling_state(sampling_request, prompt_ids, row_index=row_index)
        samples = []
        raise_if_generation_deadline_expired(request)
        result = session.prefill(prompt_ids, return_logits=True)
        raise_if_generation_deadline_expired(request)
        full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(result)
        sample = _select_from_gguf_logits(result, sampling_request, state, self.tokenizer)
        samples.append(sample)
        generated_ids = [int(sample.token_id)]
        _gguf_queue_json_object_close_if_needed(
            state,
            self.tokenizer,
            _gguf_token_text(self.tokenizer, sample),
            remaining_tokens=request.max_tokens - len(generated_ids),
        )
        if _gguf_finished(generated_ids, self.tokenizer, request):
            return _gguf_generation_output(
                self.tokenizer,
                samples,
                finish_details=_gguf_finish_details(generated_ids, self.tokenizer, request, state),
                telemetry=_gguf_telemetry(
                    prompt_ids,
                    generated_ids,
                    request,
                    row_index=row_index,
                    sampling_state=state,
                    forced_sample=sample,
                    full_vocab_logits_d2h=full_vocab_logits_d2h,
                    logits_d2h_bytes=logits_d2h_bytes,
                    timing=timing,
                ),
            )
        for _ in range(request.max_tokens - 1):
            raise_if_generation_deadline_expired(request)
            step = session.step(generated_ids[-1], return_logits=True)
            raise_if_generation_deadline_expired(request)
            step_full_vocab_logits_d2h, step_logits_d2h_bytes = _gguf_logits_d2h_metadata(step)
            if step_full_vocab_logits_d2h is not None:
                full_vocab_logits_d2h = step_full_vocab_logits_d2h
                logits_d2h_bytes = step_logits_d2h_bytes
            sample = _select_from_gguf_logits(step, sampling_request, state, self.tokenizer)
            samples.append(sample)
            generated_ids.append(int(sample.token_id))
            _gguf_queue_json_object_close_if_needed(
                state,
                self.tokenizer,
                _gguf_token_text(self.tokenizer, sample),
                remaining_tokens=request.max_tokens - len(generated_ids),
            )
            if _gguf_finished(generated_ids, self.tokenizer, request):
                break
        return _gguf_generation_output(
            self.tokenizer,
            samples,
            finish_details=_gguf_finish_details(generated_ids, self.tokenizer, request, state),
            telemetry=_gguf_telemetry(
                prompt_ids,
                generated_ids,
                request,
                row_index=row_index,
                sampling_state=state,
                forced_sample=samples[-1] if samples else None,
                full_vocab_logits_d2h=full_vocab_logits_d2h,
                logits_d2h_bytes=logits_d2h_bytes,
                timing=timing,
            ),
        )

    def _stream_greedy(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
        *,
        tokenize_ms: float,
    ) -> Iterator[GenerationStreamChunk]:
        generated_ids: list[int] = []
        telemetry_timing = {"tokenize_ms": max(0.0, float(tokenize_ms))}
        raise_if_generation_deadline_expired(request)
        result = session.prefill(prompt_ids, return_logits=False)
        raise_if_generation_deadline_expired(request)
        generated_ids.append(int(result.token_id))
        finished = _gguf_finished(generated_ids, self.tokenizer, request)
        yield GenerationStreamChunk(
            self.tokenizer.decode([generated_ids[-1]]),
            finish_details=(
                _gguf_finish_details(generated_ids, self.tokenizer, request)
                if finished or len(generated_ids) >= request.max_tokens
                else None
            ),
            telemetry=_gguf_telemetry(
                prompt_ids,
                generated_ids,
                request,
                row_index=0,
                phase="answer",
                timing=telemetry_timing,
            ),
            generated_token_ids=(
                tuple(generated_ids)
                if finished or len(generated_ids) >= request.max_tokens
                else None
            ),
        )
        if finished:
            return
        for _ in range(request.max_tokens - 1):
            raise_if_generation_deadline_expired(request)
            step = session.step(generated_ids[-1], return_logits=False)
            raise_if_generation_deadline_expired(request)
            generated_ids.append(int(step.token_id))
            finished = _gguf_finished(generated_ids, self.tokenizer, request)
            yield GenerationStreamChunk(
                self.tokenizer.decode([generated_ids[-1]]),
                finish_details=(
                    _gguf_finish_details(generated_ids, self.tokenizer, request)
                    if finished or len(generated_ids) >= request.max_tokens
                    else None
                ),
                telemetry=_gguf_telemetry(
                    prompt_ids,
                    generated_ids,
                    request,
                    row_index=0,
                    phase="answer",
                    timing=telemetry_timing,
                ),
                generated_token_ids=(
                    tuple(generated_ids)
                    if finished or len(generated_ids) >= request.max_tokens
                    else None
                ),
            )
            if finished:
                return

    def _stream_sampled(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
        *,
        row_index: int,
        tokenize_ms: float,
    ) -> Iterator[GenerationStreamChunk]:
        sampling_request = _request_with_tokenizer_eos(request, self.tokenizer)
        telemetry_timing = {"tokenize_ms": max(0.0, float(tokenize_ms))}
        state = _gguf_row_sampling_state(sampling_request, prompt_ids, row_index=row_index)
        generated_ids: list[int] = []
        live_phase = None if state.thinking_budget is not None else "answer"
        raise_if_generation_deadline_expired(request)
        result = session.prefill(prompt_ids, return_logits=True)
        raise_if_generation_deadline_expired(request)
        full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(result)
        sample = _select_from_gguf_logits(result, sampling_request, state, self.tokenizer)
        generated_ids.append(int(sample.token_id))
        _gguf_queue_json_object_close_if_needed(
            state,
            self.tokenizer,
            _gguf_token_text(self.tokenizer, sample),
            remaining_tokens=request.max_tokens - len(generated_ids),
        )
        finished = _gguf_finished(generated_ids, self.tokenizer, sampling_request)
        yield GenerationStreamChunk(
            self.tokenizer.decode([generated_ids[-1]]),
            token_logprobs=_gguf_stream_token_logprobs(self.tokenizer, sample, sampling_request),
            finish_details=(
                _gguf_finish_details(generated_ids, self.tokenizer, sampling_request, state)
                if finished or len(generated_ids) >= sampling_request.max_tokens
                else None
            ),
            telemetry=_gguf_telemetry(
                prompt_ids,
                generated_ids,
                sampling_request,
                row_index=row_index,
                sampling_state=state,
                phase=live_phase,
                forced_sample=sample,
                full_vocab_logits_d2h=full_vocab_logits_d2h,
                logits_d2h_bytes=logits_d2h_bytes,
                timing=telemetry_timing,
            ),
            generated_token_ids=(
                tuple(generated_ids)
                if finished or len(generated_ids) >= sampling_request.max_tokens
                else None
            ),
        )
        if finished:
            return
        for _ in range(request.max_tokens - 1):
            raise_if_generation_deadline_expired(request)
            step = session.step(generated_ids[-1], return_logits=True)
            raise_if_generation_deadline_expired(request)
            full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(step)
            sample = _select_from_gguf_logits(step, sampling_request, state, self.tokenizer)
            generated_ids.append(int(sample.token_id))
            _gguf_queue_json_object_close_if_needed(
                state,
                self.tokenizer,
                _gguf_token_text(self.tokenizer, sample),
                remaining_tokens=request.max_tokens - len(generated_ids),
            )
            finished = _gguf_finished(generated_ids, self.tokenizer, sampling_request)
            yield GenerationStreamChunk(
                self.tokenizer.decode([generated_ids[-1]]),
                token_logprobs=_gguf_stream_token_logprobs(self.tokenizer, sample, sampling_request),
                finish_details=(
                    _gguf_finish_details(generated_ids, self.tokenizer, sampling_request, state)
                    if finished or len(generated_ids) >= sampling_request.max_tokens
                    else None
                ),
                telemetry=_gguf_telemetry(
                    prompt_ids,
                    generated_ids,
                    sampling_request,
                    row_index=row_index,
                    sampling_state=state,
                    phase=live_phase,
                    forced_sample=sample,
                    full_vocab_logits_d2h=full_vocab_logits_d2h,
                    logits_d2h_bytes=logits_d2h_bytes,
                    timing=telemetry_timing,
                ),
                generated_token_ids=(
                    tuple(generated_ids)
                    if finished or len(generated_ids) >= sampling_request.max_tokens
                    else None
                ),
            )
            if finished:
                return


@dataclass(frozen=True, slots=True)
class _GGUFResidentSessionLease:
    session: Qwen35GGUFResidentSession
    pool_key: _GGUFSessionPoolKey


@dataclass(slots=True)
class _GGUFPrefixSnapshotEntry:
    tokens: tuple[int, ...]
    block_ids: tuple[int, ...]
    snapshot: Any
    owner_request_id: int | None
    retained: bool = False
    # True for the deepest 256-aligned boundary at or before the request's own
    # prompt end. Every following turn reaches it -- a resend matches it and a
    # cumulative turn passes through it first -- so it outranks a decode-time
    # boundary, which only a client resending this request's generated text
    # verbatim can reach.
    prompt_boundary: bool = False


@dataclass(frozen=True, slots=True)
class _GGUFPrefixReuseSource:
    matched_tokens: tuple[int, ...]
    block_ids: tuple[int, ...]
    source_row: _GGUFResidentLoopRow | None = None
    snapshot: Any | None = None


@dataclass(slots=True)
class _GGUFResidentLoopRow:
    request_id: int
    batch_id: int
    row_index: int
    request: GenerationRequest
    prompt_ids: tuple[int, ...]
    native_greedy: bool
    native_sampled: bool
    submitted_at: float
    tokenize_ms: float = 0.0
    prompt_encode_ms: float = 0.0
    render_ms: float = 0.0
    admission_prepare_ms: float = 0.0
    native_sampler: bool = False
    prefill_tokens_seen: int = 0
    incremental_prefill: bool | None = None
    resumable_prefill: Any | None = None
    prefill_chunk_count: int = 0
    prefill_ms: float = 0.0
    lease: _GGUFResidentSessionLease | None = None
    slot: _GGUFARServingSlot | None = None
    first_token_emitted: bool = False
    fallback_output: GenerationOutput | None = None
    kv_allocation: Any | None = None
    sampling_request: GenerationRequest | None = None
    sampler_plan: Any | None = None
    sampling_state: RowSamplingState | None = None
    samples: list[Any] = field(default_factory=list)
    full_vocab_logits_d2h: bool | None = None
    logits_d2h_bytes: int | None = None
    prefix_eligible: bool = False
    prefix_lookup: bool = False
    prefix_matched_tokens: int = 0
    prefix_reused_tokens: int = 0
    prefix_source_request_id: int | None = None
    prefix_source_kind: str | None = None
    prefix_state_clone_bytes: int = 0
    prefix_snapshot_hit: bool = False
    prefix_admission_fallback: bool = False
    prefix_fallback_reason: str | None = None
    mtp2_candidate_budget: int = 0
    mtp2_requested_budget: int = 0
    mtp2_prompt_streaming: bool = False
    mtp2_prompt_prime_rows: int = 0
    mtp2_prompt_carried_bytes: int = 0
    mtp2_prompt_fallback_reason: str | None = None
    # The four separately-published refusal facts. Declared on the row (a slots
    # dataclass) because the shared accounting helper cannot attach a new
    # attribute at runtime: why the row's provider was not ready, how wide the
    # group it was planned in was, and whether that group ran autoregressively.
    mtp2_provider_readiness: str | None = None
    mtp2_provider_decline_reason: str | None = None
    # Declared here because this row is a slotted dataclass: the accounting
    # layer writes these attributes, and an undeclared one raises inside the
    # engine loop rather than reporting a missing fact.
    mtp2_provider_state_present: bool | None = None
    mtp2_plan_group_rows: int | None = None
    mtp2_plan_ar_only: bool | None = None
    mtp2_plan_reason: str | None = None
    mtp2_mtp_output_tokens: int = 0
    mtp2_ar_output_tokens: int = 0
    mtp2_ar_step_output_tokens: int = 0
    mtp2_first_fallback_position: int | None = None
    mtp2_ar_step_reasons: dict[str, int] = field(default_factory=dict)
    mtp2_cycles: int = 0
    mtp2_candidate_counts: list[int] = field(default_factory=list)
    mtp2_accepted_counts: list[int] = field(default_factory=list)
    # Committed output spans (mode, reason, position, tokens), recorded only
    # while HIPENGINE_MTP2_OUTPUT_SPANS is enabled. Declared here because the
    # row is a slots dataclass: the shared accounting helper cannot attach a new
    # attribute at runtime.
    mtp2_output_spans: list[dict[str, Any]] = field(default_factory=list)
    mtp2_proposal_ms: float = 0.0
    mtp2_target_ms: float = 0.0
    mtp2_provider_update_ms: float = 0.0
    mtp2_accept_ms: float = 0.0
    mtp2_selected_commit_ms: float = 0.0
    mtp2_candidate_readback_ms: float = 0.0
    mtp2_target_readback_ms: float = 0.0
    mtp2_accept_upload_ms: float = 0.0
    mtp2_accept_tail_ms: float = 0.0
    mtp2_accept_enqueue_ms: float = 0.0
    mtp2_k0_catchups: int = 0
    mtp2_ngram_lookup_calls: int = 0
    mtp2_ngram_lookup_hits: int = 0
    mtp2_ngram_cycles: int = 0
    mtp2_ngram_probed_tokens: int = 0
    mtp2_ngram_accepted_tokens: int = 0
    mtp2_proposal_batch_calls: int = 0
    mtp2_proposal_physical_rows: list[int] = field(default_factory=list)
    mtp2_target_batch_calls: int = 0
    mtp2_target_physical_rows: list[int] = field(default_factory=list)
    mtp2_target_pass_ms: list[float] = field(default_factory=list)
    mtp2_target_pass_start_ns: list[int] = field(default_factory=list)
    mtp2_target_pass_end_ns: list[int] = field(default_factory=list)
    mtp2_cycle_profile_start_ns: list[int] = field(default_factory=list)
    mtp2_cycle_profile_end_ns: list[int] = field(default_factory=list)
    mtp2_accept_pass_ms: list[float] = field(default_factory=list)
    mtp2_provider_update_pass_ms: list[float] = field(default_factory=list)
    mtp2_candidate_device_handoffs: int = 0
    mtp2_candidate_d2h_after_target: int = 0
    mtp2_device_chain_oracle_trace: list[dict[str, Any]] = field(default_factory=list)
    mtp2_device_accept_calls: int = 0
    mtp2_selected_commit_batch_calls: int = 0
    mtp2_execution_routes: list[str] = field(default_factory=list)
    mtp2_recoverable_failures: int = 0
    mtp2_failure_reasons: list[str] = field(default_factory=list)


def _compact_live_execution_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Copy live route proof without PM4's per-dispatch diagnostic records."""

    compact = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "graph"
    }
    graph_source = manifest.get("graph")
    if not isinstance(graph_source, Mapping):
        if graph_source is not None:
            compact["graph"] = copy.deepcopy(graph_source)
        return compact
    graph = {
        key: copy.deepcopy(value)
        for key, value in graph_source.items()
        if key != "transport"
    }
    compact["graph"] = graph
    transport_source = graph_source.get("transport")
    if not isinstance(transport_source, Mapping):
        if transport_source is not None:
            graph["transport"] = copy.deepcopy(transport_source)
        return compact
    transport = {
        key: copy.deepcopy(value)
        for key, value in transport_source.items()
        if key != "executable"
    }
    graph["transport"] = transport
    executable_source = transport_source.get("executable")
    if not isinstance(executable_source, Mapping):
        if executable_source is not None:
            transport["executable"] = copy.deepcopy(executable_source)
        return compact
    executable = {
        key: copy.deepcopy(value)
        for key, value in executable_source.items()
        if key not in {"module_records", "dispatch_records"}
    }
    records_omitted = False
    for records_key, count_key in (
        ("module_records", "module_record_count"),
        ("dispatch_records", "dispatch_record_count"),
    ):
        records = executable_source.get(records_key)
        if isinstance(records, (list, tuple)):
            executable[count_key] = len(records)
            records_omitted = True
        elif records is not None:
            executable[records_key] = copy.deepcopy(records)
    if records_omitted:
        executable["records_omitted"] = True
    transport["executable"] = executable
    return compact


class Qwen35GGUFResidentModelRunner:
    """Long-lived scheduler-facing owner of GGUF model and session state.

    The owner reserves a fixed session pool once, keeps stable request identity
    separate from scheduler physical slots, and exposes one committed prefill or
    decode transition per engine-loop hook.  Greedy and host-sampled c>1 decode
    use the retained packed session primitive; host sampling remains explicit
    in telemetry without being mislabeled as a serial model-step fallback.
    """

    # The Generation-2 batch owner serializes its shared temporary workspaces
    # and keeps prefill/decode canonical state in independent target slots, so
    # one scheduler round may safely execute a prefill quantum followed by each
    # due decode row. Multiple prefill quanta stay disabled until independently
    # qualified; this capability alone prevents long-prefill ITL starvation.
    supports_prefill_decode_same_round = True
    supports_multiple_prefill_quanta_per_round = False

    def __init__(
        self,
        generator: Qwen35GGUFBringupGenerator,
        *,
        capacity: int = _GGUF_RESIDENT_MODEL_LOOP_DEFAULT_CAPACITY,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.generator = generator
        self.capacity = int(capacity)
        self.packed_prefill_max_rows = int(
            backend_package_capability(
                generator.backend,
                "GGUF_C2_PACKED_PREFILL_MAX_ROWS",
                1,
            )
        )
        if self.packed_prefill_max_rows <= 0:
            raise ValueError("GGUF_C2_PACKED_PREFILL_MAX_ROWS must be positive")
        self._shared_runner = generator._get_shared_runner()
        self._max_sequence_length = getattr(generator, "_prepared_max_sequence_length", None)
        self._available: list[_GGUFResidentSessionLease] = []
        self._resident_batch_owner: Qwen35GGUFResidentSession | None = None
        self._resident_batch_owner_pool_key: _GGUFSessionPoolKey | None = None
        self._rows: dict[int, _GGUFResidentLoopRow] = {}
        self._outputs: dict[int, GenerationOutput] = {}
        self._completed_metadata: dict[int, dict[str, Any]] = {}
        self._next_batch_id = 0
        self._kv_pool: Any | None = None
        self._kv_pool_generation = 0
        self._engine_loop_config: Any | None = None
        self._prefix_cache_mode = "off"
        self._prefix_cache: RadixCache | None = None
        self._prefix_state_snapshots: dict[tuple[int, ...], _GGUFPrefixSnapshotEntry] = {}
        self._prefix_snapshot_limit = max(1, int(capacity))
        # Retained snapshots are the durable, cross-request working set; transient
        # ones are the current request's captures. They are trimmed separately so
        # a request's own capture can never evict the boundary that served it.
        # The retained count defaults to the pre-fix effective budget; the wider
        # working set is opt-in (docs/REFACTOR.md records the regression).
        self._prefix_retained_limit = _gguf_prefix_retained_snapshot_limit(capacity)
        self._prefix_retained_state_bytes_limit = (
            _gguf_prefix_retained_state_bytes_limit()
        )
        self._prefix_retained_evictions_by_reason: dict[str, int] = {}
        self._prefix_snapshot_promotions = 0
        self._prefix_snapshot_hits = 0
        self._prefix_snapshot_evictions = 0
        self._prefix_snapshot_captures = 0
        self._prefix_snapshot_capture_bytes = 0
        self._prefix_usable_hits = 0
        self._prefix_unusable_hits = 0
        self._prefix_contiguous_admissions = 0
        self._prefix_gapped_admissions = 0
        self._prefix_admission_fallbacks = 0
        self._prefix_fallback_reasons: dict[str, int] = {}
        self._prefix_reused_tokens = 0
        self._prefix_state_clone_bytes = 0
        self._prefix_phase_ms: dict[str, float] = {}
        self._prefix_phase_calls: dict[str, int] = {}
        self._kv_hip_used_peak_sampled_bytes = 0
        self._kv_graph_invalidation_count = 0
        self._packed_workspace_release_events = 0
        self._packed_workspace_released_bytes = 0
        self._c1_shadow_resource_bundles: dict[int, dict[str, Any]] = {}
        self._route_counts: Counter[str] = Counter()
        self._fallback_reasons: Counter[str] = Counter()
        self._last_execution_manifest: dict[str, Any] = {}
        self._last_physical_group_plan: dict[str, Any] = {}
        self._recent_completed_routes: deque[dict[str, Any]] = deque(maxlen=1024)
        self._mtp2_adapter: Any | None = None
        self._mtp2_adapter_resolved = False
        self._closed = False
        # How much of the step currently executing could already have reached
        # device state.  ``"none"`` is set by a step's own pre-device
        # validation and cleared at its first device call; the conservative
        # default keeps a failure in any other phase fatal.
        self._execution_mutation_window = "unknown"
        self._execution_step_request_id: int | None = None
        # Rows whose device state may already have advanced inside the step or
        # speculative cycle currently executing, and rows whose canonical
        # commit this runner recorded inside it.  Containment reads both to
        # bound a post-device claim: a row is left unnamed only when the runner
        # can prove it never entered the device phase, and the mutation class
        # is ``committed`` exactly when a canonical commit was recorded before
        # the failure.
        self._execution_stepped_request_ids: set[int] = set()
        self._execution_committed_request_ids: set[int] = set()
        # The plan of the speculative cycle currently executing, so a failed
        # cycle can name the rows it owned.
        self._execution_speculative_plan: Any | None = None
        # True once the current step's device phase has completed without any
        # enumerated device work, so a failure after it is still pre-device for
        # every row.  A phase that raised before completing leaves it False, so
        # an unenumerated device path stays refused instead of contained.
        self._execution_device_phase_empty = False
        self._graph_handle_refs: dict[int, weakref.ReferenceType[Any]] = {}
        self._graph_handle_buckets: dict[int, str] = {}
        self._graph_handle_replays: dict[int, int] = {}
        self._graph_buckets: dict[str, dict[str, Any]] = {}
        # Real generators resolve the server/request KV policy in ``prepare``
        # before sessions exist. Lightweight fake owners used by host tests do
        # not expose that contract and retain eager reservation.
        if not bool(
            getattr(generator, "_defer_resident_session_policy_resolution", False)
        ):
            self._reserve_sessions()

    @property
    def active_request_ids(self) -> tuple[int, ...]:
        return tuple(self._rows)

    @property
    def available_session_count(self) -> int:
        return len(self._available)

    @property
    def kv_pool(self):
        return self._kv_pool

    @property
    def kv_pool_stats(self):
        pool = self._kv_pool
        return None if pool is None else pool.stats

    def _resident_sessions(self) -> tuple[Any, ...]:
        sessions = [lease.session for lease in self._available]
        sessions.extend(
            row.lease.session
            for row in self._rows.values()
            if row.lease is not None
        )
        unique: dict[int, Any] = {}
        for session in sessions:
            unique[id(session)] = session
        return tuple(unique.values())

    def _graph_handles_for_sessions(self, sessions: Sequence[Any]) -> tuple[Any, ...]:
        handles: dict[int, Any] = {}
        for session in sessions:
            for handle in tuple(getattr(session, "_decode_graphs", ())):
                handles[id(handle)] = handle
            device_handles = getattr(session, "_device_kv_graph_handles", {})
            if isinstance(device_handles, Mapping):
                for handle in device_handles.values():
                    handles[id(handle)] = handle
        return tuple(handles.values())

    def _graph_bucket_label(self, handle: Any) -> str:
        handle_id = id(handle)
        previous_ref = self._graph_handle_refs.get(handle_id)
        if previous_ref is not None and previous_ref() is not handle:
            self._graph_handle_buckets.pop(handle_id, None)
            self._graph_handle_replays.pop(handle_id, None)
        known = self._graph_handle_buckets.get(handle_id)
        if known is not None:
            return known
        key = getattr(handle, "bucket_key", None)
        as_dict = getattr(key, "as_dict", None)
        if callable(as_dict):
            payload = as_dict()
        elif isinstance(key, Mapping):
            payload = copy.deepcopy(dict(key))
        else:
            payload = {}
        label = str(
            payload.get("key_sha256")
            or getattr(key, "key_sha256", None)
            or "unkeyed"
        )
        self._graph_handle_buckets[handle_id] = label
        self._graph_buckets.setdefault(
            label,
            {
                "bucket_key": payload,
                "entries": 0,
                "captures": 0,
                "hits": 0,
                "replays": 0,
                "invalidations": 0,
            },
        )
        return label

    def _observe_graph_handles(self, sessions: Sequence[Any]) -> None:
        for handle in self._graph_handles_for_sessions(sessions):
            handle_id = id(handle)
            label = self._graph_bucket_label(handle)
            bucket = self._graph_buckets[label]
            previous_ref = self._graph_handle_refs.get(handle_id)
            is_new_handle = previous_ref is None or previous_ref() is not handle
            if is_new_handle:
                try:
                    self._graph_handle_refs[handle_id] = weakref.ref(handle)
                except TypeError:
                    # Runtime graph handles are weak-referenceable; retain a
                    # strong closure only for minimal third-party test doubles.
                    self._graph_handle_refs[handle_id] = lambda handle=handle: handle
                bucket["captures"] += 1
                self._graph_handle_replays[handle_id] = 0
            raw_replay_count = getattr(handle, "replay_count", None)
            if raw_replay_count is None:
                replayed_steps = max(0, int(getattr(handle, "replayed_steps", 0)))
                steps_per_replay = max(1, int(getattr(handle, "steps_per_replay", 1)))
                replay_count = replayed_steps // steps_per_replay
            else:
                replay_count = max(0, int(raw_replay_count))
            previous = self._graph_handle_replays.get(handle_id, 0)
            if replay_count > previous:
                delta = replay_count - previous
                bucket["hits"] += delta
                bucket["replays"] += delta
            self._graph_handle_replays[handle_id] = replay_count

    def _record_graph_invalidations(self, handles: Sequence[Any], count: int) -> None:
        remaining = max(0, int(count))
        for handle in handles:
            if remaining <= 0:
                break
            label = self._graph_bucket_label(handle)
            self._graph_buckets[label]["invalidations"] += 1
            remaining -= 1

    def observability_snapshot(self) -> dict[str, Any]:
        """Return real GGUF resource, graph, and route/fallback evidence."""

        sessions = self._resident_sessions()
        self._observe_graph_handles(sessions)
        pool = self._kv_pool
        workspace_pages_fn = getattr(pool, "workspace_pages", None)
        workspace_lease = (
            workspace_pages_fn(_GGUF_PACKED_WORKSPACE_LEASE_KEY)
            if callable(workspace_pages_fn) else None
        )
        pool_stats = None if pool is None else pool.stats.to_json_dict()
        if pool_stats is not None:
            pool_stats["max_pages"] = getattr(pool, "max_pages", None)
            pool_stats["budget_bytes"] = getattr(pool, "budget_bytes", None)
            pool_stats["private_workspace_bytes"] = int(getattr(pool, "private_workspace_bytes", 0))
            pool_stats["accounted_bytes"] = int(getattr(pool, "accounted_bytes", pool_stats["current_bytes"]))
        active_entries: Counter[str] = Counter()
        for handle in self._graph_handles_for_sessions(sessions):
            if bool(getattr(handle, "closed", False)):
                continue
            label = self._graph_bucket_label(handle)
            active_entries[label] += 1
        buckets = copy.deepcopy(self._graph_buckets)
        for label, row in buckets.items():
            row["entries"] = int(active_entries.get(label, 0))
        prefix_observability = self._prefix_cache_observability()
        workspace_owner_bytes, workspace_sessions = packed_workspace_owner_inventory(
            sessions
        )
        prefill_transients = prefill_transient_owner_inventory(sessions)
        pool_page_bytes = (
            0
            if pool_stats is None or not pool_stats.get("current_pages")
            else int(pool_stats["current_bytes"]) // int(pool_stats["current_pages"])
        )
        kv_layout_audits = [
            copy.deepcopy(audit())
            for session in sessions
            for audit in (getattr(session, "device_kv_layout_audit", None),)
            if callable(audit)
            and getattr(session, "device_kv_allocation", None) is not None
        ]
        return {
            "model_runner": {
                "capacity": int(self.capacity),
                "max_active_requests": int(self.capacity),
                "max_context_tokens": getattr(
                    self._available[-1].session,
                    "max_sequence_length",
                    None,
                )
                if self._available
                else None,
                "active_request_ids": list(self.active_request_ids),
                "active_requests": len(self._rows),
                "available_sessions": len(self._available),
                "packed_workspace_current_bytes": workspace_owner_bytes,
                "packed_workspace_owner_sessions": workspace_sessions,
                "packed_workspace_leased_pool_bytes": (
                    len(workspace_lease or ()) * pool_page_bytes
                ),
                "packed_kv_workspace_lease_skipped": bool(
                    getattr(self, "_packed_kv_workspace_lease_skipped", False)
                ),
                "packed_workspace_note": (
                    "current_bytes is owner-deduplicated unique allocation bytes"
                    " (shared slot views report their workspace once, split-growth"
                    " owners included); leased_pool_bytes counts the pool-plane"
                    " pages pinned by workspace leases - a different accounting"
                    " domain, never a substitute for either"
                ),
                "prefill_transients": prefill_transients,
                "packed_workspace_release_events": int(
                    getattr(self, "_packed_workspace_release_events", 0)
                ),
                "packed_workspace_released_bytes": int(
                    getattr(self, "_packed_workspace_released_bytes", 0)
                ),
                "kv_layout_audits": kv_layout_audits,
                "persistent_int8_payload_bytes": sum(
                    int(audit.get("persistent_int8_payload_bytes", 0))
                    for audit in kv_layout_audits
                ),
                "persistent_bf16_payload_bytes": sum(
                    int(audit.get("persistent_bf16_payload_bytes", 0))
                    for audit in kv_layout_audits
                ),
                "persistent_scale_bytes": sum(
                    int(audit.get("persistent_scale_bytes", 0))
                    for audit in kv_layout_audits
                ),
                "persistent_bf16_mirror_bytes": sum(
                    int(audit.get("persistent_bf16_mirror_bytes", 0))
                    for audit in kv_layout_audits
                ),
                "persistent_kv_total_bytes": sum(
                    int(audit.get("persistent_total_bytes", 0))
                    for audit in kv_layout_audits
                ),
            },
            "kv_pool": pool_stats,
            "prefix_cache": prefix_observability,
            "graph_buckets": {
                "entries": int(sum(active_entries.values())),
                "captures_total": int(sum(row["captures"] for row in buckets.values())),
                "hits_total": int(sum(row["hits"] for row in buckets.values())),
                "replays_total": int(sum(row["replays"] for row in buckets.values())),
                "invalidations_total": int(self._kv_graph_invalidation_count),
                "buckets": buckets,
            },
            "routes": {
                "counts": {
                    "native_full_prefill_rows": int(self._route_counts["native_full_prefill_rows"]),
                    "native_full_prefill_groups": int(
                        self._route_counts["native_full_prefill_groups"]
                    ),
                    "native_incremental_prefill_chunks": int(
                        self._route_counts["native_incremental_prefill_chunks"]
                    ),
                    "native_incremental_prefill_unsampled_chunks": int(
                        self._route_counts[
                            "native_incremental_prefill_unsampled_chunks"
                        ]
                    ),
                    "native_packed_decode_steps": int(
                        self._route_counts["native_packed_decode_steps"]
                    ),
                    "native_packed_graph_captures": int(
                        self._route_counts["native_packed_graph_captures"]
                    ),
                    "native_packed_graph_replays": int(
                        self._route_counts["native_packed_graph_replays"]
                    ),
                    "native_c1_decode_steps": int(self._route_counts["native_c1_decode_steps"]),
                    "native_sampled_prefill_rows": int(
                        self._route_counts["native_sampled_prefill_rows"]
                    ),
                    "native_sampler_requests": int(
                        self._route_counts["native_sampler_requests"]
                    ),
                    "native_sampler_batch_launches": int(
                        self._route_counts["native_sampler_batch_launches"]
                    ),
                    "native_sampler_row_launches": int(
                        self._route_counts["native_sampler_row_launches"]
                    ),
                    "host_sampler_requests": int(
                        self._route_counts["host_sampler_requests"]
                    ),
                    "serial_decode_fallback_steps": int(
                        self._route_counts["serial_decode_fallback_steps"]
                    ),
                    "serial_c1_row_steps": int(
                        self._route_counts["serial_c1_row_steps"]
                    ),
                    "resident_fallback_requests": int(
                        self._route_counts["resident_fallback_requests"]
                    ),
                },
                "physical_width_decode_steps": {
                    str(width): int(
                        self._route_counts[
                            "native_c1_decode_steps"
                            if width == 1
                            else f"native_c{width}_decode_steps"
                        ]
                    )
                    for width in _gguf_ar_physical_widths(
                        str(getattr(getattr(self, "_shared_runner", None), "backend", "hip_gfx1100")),
                        use_capability=getattr(self, "_resident_batch_owner", None) is not None,
                    )
                },
                "fallback_reasons": {
                    str(key): int(value)
                    for key, value in sorted(self._fallback_reasons.items())
                },
                "last_execution_manifest": copy.deepcopy(self._last_execution_manifest),
                "last_physical_group_plan": copy.deepcopy(
                    self._last_physical_group_plan
                ),
                "recent_completed": list(copy.deepcopy(self._recent_completed_routes)),
            },
        }

    def kv_pool_memory_snapshot(self) -> dict[str, Any]:
        """Return pool, tracked allocator, and sampled HIP current/peak evidence."""

        self._sample_kv_hip_memory()
        pool = self._kv_pool
        pool_stats = None if pool is None else pool.stats.to_json_dict()
        storage_view_fn = getattr(pool, "storage_view", None)
        storage_view = storage_view_fn() if callable(storage_view_fn) else None
        tracked = memory_stats()
        owner = self._resident_batch_owner
        workspace_backing = (
            None
            if owner is None
            else getattr(owner, "packed_workspace_backing", None)
        )
        workspace_pages_fn = getattr(pool, "workspace_pages", None)
        workspace_lease = (
            workspace_pages_fn(_GGUF_PACKED_WORKSPACE_LEASE_KEY)
            if callable(workspace_pages_fn)
            else None
        )
        return {
            "pool_contract": (
                None
                if pool is None
                else (
                    "global_generation2"
                    if bool(getattr(pool, "generation2_compatible", False))
                    else "legacy_single_backing"
                )
            ),
            "storage_view": (
                None
                if storage_view is None
                else {
                    "layout_key": str(storage_view.layout_key),
                    "generation": int(storage_view.generation),
                    "plane_count": len(storage_view.planes),
                    "metadata_descriptor_bytes": int(
                        storage_view.metadata_descriptor_bytes
                    ),
                }
            ),
            "dynamic_pool": pool_stats,
            "packed_workspace_backing": workspace_backing,
            "private_workspace_kv_bytes": (
                0 if pool is None else int(getattr(pool, "private_workspace_bytes", 0))
            ),
            "accounted_kv_bytes": (
                0 if pool is None else int(getattr(pool, "accounted_bytes", pool_stats["current_bytes"]))
            ),
            "kv_budget_bytes": None if pool is None else getattr(pool, "budget_bytes", None),
            "packed_workspace_lease_pages": (
                0 if workspace_lease is None else len(workspace_lease)
            ),
            "tracked_allocator": tracked,
            "hip_used_current_bytes": self._current_hip_used_bytes(),
            "hip_used_peak_sampled_bytes": int(self._kv_hip_used_peak_sampled_bytes),
            "graph_invalidation_count": int(self._kv_graph_invalidation_count),
        }

    def _teardown_kv_pool(self, *, release_workspace_state: bool) -> None:
        """Release the packed workspace lease, then close the KV pool.

        The workspace lease pins arena pages, so it must be released before
        ``close()`` (which rejects pinned pages). When the owner-shared packed
        workspace borrows arena planes it must also be freed first; the
        guarded release fails closed on unflushed state or live graphs.
        """

        pool = self._kv_pool
        if pool is None:
            return
        owner = self._resident_batch_owner
        if release_workspace_state and owner is not None:
            release = getattr(owner, "release_idle_packed_workspace", None)
            if callable(release):
                release()
        workspace_pages = getattr(pool, "workspace_pages", None)
        release_workspace = getattr(pool, "release_workspace", None)
        if callable(workspace_pages) and callable(release_workspace):
            if workspace_pages(_GGUF_PACKED_WORKSPACE_LEASE_KEY) is not None:
                release_workspace(_GGUF_PACKED_WORKSPACE_LEASE_KEY)
        pool.close()
        self._kv_pool = None
        if owner is not None:
            bind = getattr(owner, "bind_workspace_kv_pool", None)
            if callable(bind):
                bind(None)

    def configure_engine_loop(self, config: Any) -> None:
        """Bind engine-loop KV policy knobs to the real deferred session pool."""

        if self._rows:
            raise RuntimeError("cannot configure GGUF device KV pool while requests are active")
        self._clear_prefix_snapshots()
        self._engine_loop_config = config
        self._prefix_cache_mode = resolve_prefix_cache_mode(
            getattr(config, "prefix_cache", "off")
        )
        self._prefix_cache = (
            RadixCache(block_size=256)
            if self._prefix_cache_mode == "radix"
            else None
        )
        factory_session = self._available[-1].session if self._available else None
        create_global_pool = getattr(
            factory_session,
            "create_global_device_kv_pool",
            None,
        )
        create_legacy_pool = getattr(factory_session, "create_device_kv_pool", None)
        if not callable(create_global_pool) and not callable(create_legacy_pool):
            # Lightweight fake-session tests retain the D2 fixed-session path.
            return
        if self._kv_pool is not None:
            self._teardown_kv_pool(release_workspace_state=True)
        scratch = getattr(factory_session, "scratch", None)
        if scratch is None:
            raise RuntimeError("GGUF deferred session has no scratch capacity")
        setattr(
            factory_session,
            "kv_pool_memory_budget_mib",
            getattr(config, "kv_pool_memory_budget_mib", None),
        )
        # Pool pressure reclaims this runner's retained prefix snapshots, so the
        # pool-owning session is handed the handler explicitly. The session's own
        # `_resident_batch_owner` names the session that owns shared session-level
        # resources and has no prefix cache to reclaim from.
        setattr(factory_session, "_kv_pool_pressure_owner", self)
        initial_pages = int(config.kv_pool_initial_pages)
        self._kv_pool_memory_budget_mib = getattr(
            config, "kv_pool_memory_budget_mib", None
        )
        low_water_pages = min(int(config.kv_pool_low_water_pages), initial_pages)
        requested_high = getattr(config, "kv_pool_high_water_pages", None)
        high_water_pages = None if requested_high is None else int(requested_high)
        chunk_pages = max(1, int(config.kv_pool_chunk_pages))
        if callable(create_global_pool):
            # The initial device backing is a pool floor, not a
            # max_active_requests * max_context reservation.  The pool grows
            # on demand up to the runtime-derived memory budget.
            global_capacity = initial_pages
            if high_water_pages is not None:
                global_capacity = min(global_capacity, high_water_pages)
            if global_capacity <= 0:
                raise ValueError("GGUF global KV capacity must be positive")
            # Eager packed-execution workspace lease. This is an execution
            # scratch budget, not a full-context KV reservation per admitted
            # request: multi-row execution grows or falls back to
            # request-owned storage when this shared floor is insufficient.
            #
            # Use the same serving capacity and context floor as the packed
            # batch owner. Physical verifier slots may exceed serving capacity;
            # private fallback KV is charged against the same pool budget in
            # addition to this arena reservation.
            workspace_pages = packed_verify_workspace_lease_pages(
                int(self.capacity),
                int(scratch.max_positions),
            )
            # P4 (roadmap F2): the packed KV plane lease exists only for
            # plane consumers - non-slot-local packed prefill (prefix-cache
            # COW scatter), packed batch decode above one resident slot, and
            # the MTP verifier. A C1 server with prefix cache and MTP both
            # off has no such consumer: slot-local prefill reads request-
            # owned KV and singleton decode uses the request's direct
            # session. Creating the pool without the workspace share and
            # without the lease removes the second full-context KV
            # reservation outright instead of pinning pages nobody reads.
            # A plane consumer that appears anyway (misconfiguration, page
            # fragmentation forcing the non-slot-local fallback) fails safe:
            # _GGUFPackedTargetState.allocate falls back to a private KV
            # chunk allocation. HIPENGINE_GGUF_PACKED_KV_LEASE=1 forces the
            # eager lease for debugging or rollback.
            mtp_serving_policy = str(
                getattr(config, "speculative_mtp_serving", "auto")
            ).strip().lower().replace("-", "_")
            workspace_lease_needed = (
                int(self.capacity) > 1
                or self._prefix_cache_mode != "off"
                or mtp_serving_policy != "off"
                or os.environ.get("HIPENGINE_GGUF_PACKED_KV_LEASE", "0")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            )
            self._kv_pool_generation += 1
            if workspace_lease_needed:
                self._kv_pool = create_global_pool(
                    page_capacity=global_capacity + workspace_pages,
                    generation=self._kv_pool_generation,
                )
                self._kv_pool.lease_workspace(
                    _GGUF_PACKED_WORKSPACE_LEASE_KEY,
                    workspace_pages,
                )
            else:
                self._packed_kv_workspace_lease_skipped = True
                self._kv_pool = create_global_pool(
                    page_capacity=global_capacity,
                    generation=self._kv_pool_generation,
                )
            owner = self._resident_batch_owner
            if owner is not None:
                bind = getattr(owner, "bind_workspace_kv_pool", None)
                if callable(bind):
                    bind(self._kv_pool)
        else:
            assert callable(create_legacy_pool)
            self._kv_pool = create_legacy_pool(
                initial_pages=initial_pages,
                low_water_pages=low_water_pages,
                high_water_pages=high_water_pages,
                chunk_pages=chunk_pages,
                idle_grace_seconds=float(config.kv_pool_idle_grace_seconds),
            )
        self._sample_kv_hip_memory()

    def acquire_c1_shadow_resources(
        self,
        *,
        request_id: int,
        shadow_request_id: int,
        real_slot: int,
        shadow_slot: int,
    ) -> Mapping[str, Any]:
        """Reserve one unselected physical-C2 shadow resource bundle.

        This is an ownership-only ABI for B3. It borrows one available resident
        session, binds an independently keyed KV allocation, and owns one hidden
        row. It does not clone state or select physical target execution.
        """

        rid = int(request_id)
        shadow_id = int(shadow_request_id)
        if rid < 0 or shadow_id != -(rid + 1):
            raise ValueError("C1 shadow request ID does not match real ownership")
        if int(real_slot) < 0 or int(shadow_slot) < 0 or int(real_slot) == int(shadow_slot):
            raise ValueError("C1 shadow physical slots must be distinct and non-negative")
        bundles = getattr(self, "_c1_shadow_resource_bundles", None)
        if bundles is None:
            bundles = {}
            self._c1_shadow_resource_bundles = bundles
        if rid in bundles:
            raise RuntimeError(f"request_id {rid} already owns C1 shadow resources")
        pool = self._kv_pool
        if pool is None:
            raise RuntimeError("C1 shadow requires the dynamic KV pool")
        row = self._row(rid)
        if row.lease is None or row.kv_allocation is None:
            raise RuntimeError("C1 shadow requires admitted real target resources")
        lease = self._available[-1] if self._available else None
        if lease is None or lease is row.lease:
            raise RuntimeError("C1 shadow requires one additional resident session")
        shadow_target = lease.session
        scratch = getattr(shadow_target, "scratch", None)
        if scratch is None:
            raise RuntimeError("C1 shadow target has no recurrent-state owner")
        real_blocks = tuple(int(value) for value in row.kv_allocation.block_ids)
        if not real_blocks:
            raise RuntimeError("C1 shadow real KV allocation has no pages")
        allocation = None
        hidden_row = None
        bound = False
        try:
            allocation = pool.allocate(
                shadow_id,
                len(real_blocks),
                now_seconds=time.monotonic(),
            )
            shadow_target.bind_device_kv_allocation(pool, allocation)
            bound = True
            hidden_size = int(getattr(self._shared_runner, "hidden_size", 0))
            if hidden_size <= 0:
                raise RuntimeError("C1 shadow shared runner has no hidden size")
            hidden_row = malloc(
                hidden_size * DType.BF16.itemsize,
                runtime=shadow_target.runtime,
            )
            if not self._available or self._available[-1] is not lease:
                raise RuntimeError(
                    "GGUF available-session order changed during C1 shadow acquisition"
                )
            self._available.pop()
            bundle: dict[str, Any] = {
                "request_id": rid,
                "shadow_request_id": shadow_id,
                "real_slot": int(real_slot),
                "shadow_slot": int(shadow_slot),
                "lease": lease,
                "target_session": shadow_target,
                "hidden_row": hidden_row,
                "kv_owner": allocation,
                "recurrent_owner": scratch,
                "pool": pool,
                "reclaimed": set(),
                "finalized": False,
            }
            bundles[rid] = bundle
            return bundle
        except Exception:
            if hidden_row is not None:
                free(hidden_row, runtime=shadow_target.runtime)
            if bound:
                shadow_target.invalidate_device_kv_graphs()
                shadow_target.unbind_device_kv_allocation()
            if allocation is not None:
                pool.release(shadow_id, now_seconds=time.monotonic())
            raise

    @staticmethod
    def _c1_shadow_expected_reclaims() -> set[tuple[str, str]]:
        return {
            (surface, lane)
            for surface in (
                "target_session",
                "hidden_row",
                "kv_owner",
                "recurrent_owner",
            )
            for lane in ("real", "shadow")
        }

    def _finalize_c1_shadow_bundle(
        self,
        bundle: Mapping[str, Any],
        *,
        reason: str,
    ) -> None:
        mutable = bundle if isinstance(bundle, dict) else None
        if mutable is None:
            raise TypeError("C1 shadow resource bundle must be mutable")
        if bool(mutable.get("finalized", False)):
            return
        shadow_target = mutable["target_session"]
        allocation = mutable["kv_owner"]
        hidden_row = mutable["hidden_row"]
        pool = mutable["pool"]
        shadow_id = int(mutable["shadow_request_id"])
        lease = mutable["lease"]
        if hidden_row is not None:
            free(hidden_row, runtime=shadow_target.runtime)
            mutable["hidden_row"] = None
        if allocation is not None:
            shadow_target.invalidate_device_kv_graphs()
            shadow_target.unbind_device_kv_allocation()
            pool.release(shadow_id, now_seconds=time.monotonic())
            mutable["kv_owner"] = None
        reset = getattr(shadow_target, "reset", None)
        if callable(reset):
            reset()
        if lease in self._available:
            raise RuntimeError("C1 shadow session lease was already returned")
        self._available.append(lease)
        mutable["finalized"] = True
        mutable["finalize_reason"] = str(reason)
        self._c1_shadow_resource_bundles.pop(int(mutable["request_id"]), None)

    def reclaim_c1_shadow_resource(
        self,
        bundle: Mapping[str, Any],
        *,
        surface: str,
        lane: str,
        resource: Any,
        reason: str,
    ) -> None:
        """Return one lifecycle claim; finalize after all eight claims return."""

        del resource
        key = (str(surface), str(lane))
        expected = self._c1_shadow_expected_reclaims()
        if key not in expected:
            raise ValueError(f"unsupported C1 shadow reclaim surface: {key!r}")
        mutable = bundle if isinstance(bundle, dict) else None
        if mutable is None:
            raise TypeError("C1 shadow resource bundle must be mutable")
        reclaimed = mutable["reclaimed"]
        reclaimed.add(key)
        if reclaimed == expected:
            self._finalize_c1_shadow_bundle(mutable, reason=str(reason))

    def abort_c1_shadow_resources(
        self,
        bundle: Mapping[str, Any],
        *,
        reason: str,
    ) -> None:
        """Abort a partial adapter acquisition and return the whole bundle."""

        self._finalize_c1_shadow_bundle(bundle, reason=str(reason))

    def reserve_admission(self, request: RequestState) -> None:
        """Reserve and bind real device KV before scheduler slot publication."""

        pool = self._kv_pool
        if pool is None:
            return
        row = self._row(request.request_id)
        if row.lease is not None or row.kv_allocation is not None:
            raise RuntimeError(f"request_id {row.request_id} already has admission resources")
        if int(row.request.max_tokens) <= 0:
            return
        positions = len(row.prompt_ids) + max(0, int(row.request.max_tokens) - 1)
        if positions <= 0:
            return
        lease = self._available[-1] if self._available else None
        if lease is None:
            raise RuntimeError("GGUF resident model runner has no free session at admission")
        scratch = getattr(lease.session, "scratch", None)
        if scratch is None or positions > int(scratch.max_positions):
            capacity = 0 if scratch is None else int(scratch.max_positions)
            raise ValueError(
                f"GGUF request requires {positions} KV positions but resident capacity is {capacity}"
            )
        pages = (positions + 255) // 256
        start = time.perf_counter()
        prefix_source = self._prefix_source_for(row)
        self._prefix_phase_add("admission_lookup", start)
        if prefix_source is not None:
            matched_tokens = prefix_source.matched_tokens
            prefix_pages = len(matched_tokens) // 256
            # Placement decides which prefill route the hit gets, and the two
            # are not close. A gapped block table has no contiguous base row, so
            # `prefill_batch_native` drops to the packed paged route and imports
            # the session's history into the packed planes; a contiguous run
            # keeps the slot-local AOTriton route. Measured on Qwen3.5-0.8B, the
            # paged route does the same attention work about 2.5x slower, which
            # is what pushes a hit's break-even down to a suffix of roughly 12%
            # of the prompt.
            #
            # So ask for contiguity first and accept a gapped placement rather
            # than refusing: a slower hit still beats re-prefilling the whole
            # prompt. Requiring contiguity outright (the earlier policy) turned
            # fragmented placements into full-prefill misses, and never asking
            # (the policy this replaces) put every hit on the slow route.
            start = time.perf_counter()
            allocation = None
            try:
                allocation = pool.admit_with_shared_prefix(
                    row.request_id,
                    prefix_source.block_ids,
                    suffix_pages=pages - prefix_pages,
                    now_seconds=time.monotonic(),
                    require_contiguous=True,
                )
                self._prefix_contiguous_admissions += 1
            except (DeviceKVContiguityError, MemoryError):
                suffix_tokens = len(row.prompt_ids) - len(matched_tokens)
                if (
                    suffix_tokens > _gguf_prefix_gapped_suffix_max_tokens()
                    and not _gguf_prefix_gapped_fast_route_available(lease, positions)
                ):
                    # Only a gapped placement is available and the suffix is
                    # long enough that the slow paged route costs more than
                    # the full prefill it would replace. The fast gapped
                    # gather route is unavailable for this lease, so declining
                    # the hit is still the cheaper answer.
                    self._note_prefix_admission_fallback(
                        row, "gapped_suffix_exceeds_paged_budget"
                    )
                else:
                    try:
                        allocation = pool.admit_with_shared_prefix(
                            row.request_id,
                            prefix_source.block_ids,
                            suffix_pages=pages - prefix_pages,
                            now_seconds=time.monotonic(),
                            require_contiguous=False,
                        )
                        self._prefix_gapped_admissions += 1
                    except MemoryError:
                        self._note_prefix_admission_fallback(
                            row, "shared_admission_capacity"
                        )
            if allocation is not None:
                self._prefix_phase_add("admission_pool", start)
                try:
                    lease.session.bind_device_kv_allocation(pool, allocation)
                    restore_start = time.perf_counter()
                    if prefix_source.source_row is not None:
                        source_row = prefix_source.source_row
                        assert source_row.lease is not None
                        cloned_bytes = int(
                            lease.session.clone_prefix_state_from(
                                source_row.lease.session,
                                position=len(matched_tokens),
                            )
                        )
                    else:
                        cloned_bytes = int(
                            lease.session.clone_prefix_state_from_snapshot(
                                prefix_source.snapshot,
                            )
                        )
                    self._prefix_phase_add("admission_restore_state", restore_start)
                except Exception:
                    if getattr(lease.session, "device_kv_allocation", None) is not None or getattr(
                        lease.session, "allocation", None
                    ) is not None:
                        lease.session.unbind_device_kv_allocation()
                    pool.release(row.request_id, now_seconds=time.monotonic())
                    raise
                if not self._available or self._available[-1] is not lease:
                    lease.session.invalidate_device_kv_graphs()
                    lease.session.unbind_device_kv_allocation()
                    pool.release(row.request_id, now_seconds=time.monotonic())
                    raise RuntimeError("GGUF available-session order changed during shared admission")
                self._available.pop()
                row.lease = lease
                row.kv_allocation = allocation
                row.prefix_matched_tokens = len(matched_tokens)
                row.prefix_reused_tokens = len(matched_tokens)
                row.prefix_source_request_id = (
                    None
                    if prefix_source.source_row is None
                    else int(prefix_source.source_row.request_id)
                )
                row.prefix_source_kind = (
                    "completed_snapshot"
                    if prefix_source.snapshot is not None
                    else "active_current"
                )
                row.prefix_state_clone_bytes = cloned_bytes
                row.prefix_snapshot_hit = prefix_source.snapshot is not None
                row.prefix_fallback_reason = None
                if row.prefix_snapshot_hit:
                    self._prefix_snapshot_hits += 1
                self._prefix_usable_hits += 1
                self._prefix_reused_tokens += len(matched_tokens)
                self._prefix_state_clone_bytes += cloned_bytes
                refresh_start = time.perf_counter()
                self._refresh_prefix_cache(row)
                self._prefix_phase_add("admission_refresh", refresh_start)
                self._sample_kv_hip_memory()
                return

        try:
            start = time.perf_counter()
            allocation = pool.allocate(
                row.request_id,
                pages,
                now_seconds=time.monotonic(),
                require_contiguous=(
                    len(row.prompt_ids) >= PACKED_AR_PREFILL_CONTEXT_LIMIT
                ),
            )
        except MemoryError as exc:
            stats = pool.stats
            raise GenerationAdmissionRejected(
                str(exc),
                resource="device_kv_pool",
                request_id=int(row.request_id),
                requested_units=pages,
                current_units=int(stats.current_pages),
                capacity_units=(
                    int(pool.high_water_pages)
                    if pool.high_water_pages is not None
                    else None
                ),
            ) from exc
        try:
            lease.session.bind_device_kv_allocation(pool, allocation)
        except Exception:
            pool.release(row.request_id, now_seconds=time.monotonic())
            raise
        self._prefix_phase_add("admission_pool", start)
        if not self._available or self._available[-1] is not lease:
            lease.session.invalidate_device_kv_graphs()
            lease.session.unbind_device_kv_allocation()
            pool.release(row.request_id, now_seconds=time.monotonic())
            raise RuntimeError("GGUF available-session order changed during atomic admission")
        self._available.pop()
        row.lease = lease
        row.kv_allocation = allocation
        self._sample_kv_hip_memory()

    @staticmethod
    def _prefix_reuse_supported(row: _GGUFResidentLoopRow) -> bool:
        if row.native_greedy:
            return True
        plan = row.sampler_plan
        return bool(
            row.native_sampled
            and plan is not None
            and plan.mode is SamplingMode.PROCESSED_ARGMAX
        )

    def _prefix_phase_add(self, name: str, start: float) -> None:
        """Accumulate one named prefix-cache phase for serving diagnostics.

        The serving path pays for prefix reuse in several distinct places
        (admission lookup, trie maintenance, snapshot capture, state restore,
        suffix prefill). Wall-clock per phase is the only way to rank them,
        because the device work is asynchronous and the host work is not.
        """

        phases = self._prefix_phase_ms
        phases[name] = round(float(phases.get(name, 0.0)) + _timing_ms_since(start), 3)
        calls = self._prefix_phase_calls
        calls[name] = int(calls.get(name, 0)) + 1

    def _prefix_source_for(
        self,
        row: _GGUFResidentLoopRow,
    ) -> _GGUFPrefixReuseSource | None:
        cache = getattr(self, "_prefix_cache", None)
        if cache is None:
            self._note_prefix_fallback(row, "cache_off")
            return None
        if not self._prefix_reuse_supported(row):
            self._note_prefix_fallback(row, "sampling_unsupported")
            return None
        if len(row.prompt_ids) <= 256:
            self._note_prefix_fallback(row, "prompt_too_short")
            return None
        row.prefix_eligible = True
        start = time.perf_counter()
        self._flush_all_packed_owners()
        self._prefix_phase_add("lookup_flush_packed", start)
        start = time.perf_counter()
        for candidate in tuple(self._rows.values()):
            if candidate.request_id != row.request_id:
                self._refresh_prefix_cache(candidate)
        self._prefix_phase_add("lookup_refresh_others", start)
        row.prefix_lookup = True
        start = time.perf_counter()
        match = cache.match(row.prompt_ids)
        self._prefix_phase_add("lookup_match", start)
        row.prefix_matched_tokens = int(match.matched_token_count)
        if not match.hit:
            self._note_prefix_fallback(row, "miss")
            return None
        if match.matched_token_count >= len(row.prompt_ids):
            self._prefix_unusable_hits += 1
            self._note_prefix_fallback(row, "full_prompt_boundary_requires_suffix")
            return None
        if int(getattr(row, "mtp2_candidate_budget", 0)) > 0:
            provider_has_prefix = getattr(
                getattr(self, "_mtp2_adapter", None), "has_prefix_checkpoint", None,
            )
            if not callable(provider_has_prefix) or not provider_has_prefix(
                match.matched_tokens, match.block_ids,
            ):
                self._prefix_unusable_hits += 1
                self._note_prefix_fallback(row, "provider_checkpoint_unavailable")
                return None
        start = time.perf_counter()
        state = cache.entry_state(match.matched_tokens)
        for request_id in state.owner_request_ids:
            source = self._rows.get(int(request_id))
            if source is None or source.request_id == row.request_id:
                continue
            if source.lease is None or source.kv_allocation is None:
                continue
            processed_start = time.perf_counter()
            source_tokens = tuple(self._processed_tokens(source))
            self._prefix_phase_add("processed_tokens", processed_start)
            if source_tokens != match.matched_tokens:
                continue
            session = source.lease.session
            if int(getattr(session, "position", -1)) != match.matched_token_count:
                continue
            if tuple(source.kv_allocation.block_ids[: match.matched_block_count]) != match.block_ids:
                continue
            self._prefix_phase_add("lookup_resolve", start)
            return _GGUFPrefixReuseSource(
                matched_tokens=match.matched_tokens,
                block_ids=match.block_ids,
                source_row=source,
            )
        snapshot_entry = self._prefix_state_snapshots.get(match.matched_tokens)
        if snapshot_entry is not None:
            snapshot = snapshot_entry.snapshot
            valid = (
                not bool(getattr(snapshot, "closed", False))
                and snapshot_entry.block_ids == match.block_ids
                and int(getattr(snapshot, "position", -1)) == match.matched_token_count
            )
            if valid and snapshot_entry.retained:
                valid = (
                    self._kv_pool is not None
                    and all(
                        self._kv_pool.refcount(block_id) > 0
                        for block_id in snapshot_entry.block_ids
                    )
                )
            elif valid:
                owner = self._rows.get(int(snapshot_entry.owner_request_id or -1))
                valid = (
                    owner is not None
                    and owner.kv_allocation is not None
                    and tuple(
                        int(block_id)
                        for block_id in owner.kv_allocation.block_ids[: len(snapshot_entry.block_ids)]
                    )
                    == snapshot_entry.block_ids
                )
            if valid:
                self._prefix_state_snapshots.pop(match.matched_tokens)
                self._prefix_state_snapshots[match.matched_tokens] = snapshot_entry
                self._prefix_phase_add("lookup_resolve", start)
                return _GGUFPrefixReuseSource(
                    matched_tokens=match.matched_tokens,
                    block_ids=match.block_ids,
                    snapshot=snapshot,
                )
        self._prefix_phase_add("lookup_resolve", start)
        self._prefix_unusable_hits += 1
        self._note_prefix_fallback(row, "state_source_unavailable")
        return None

    @staticmethod
    def _processed_tokens(row: _GGUFResidentLoopRow) -> tuple[int, ...]:
        lease = row.lease
        if lease is None:
            return ()
        position = int(getattr(lease.session, "position", -1))
        if position < 0:
            return ()
        generated = () if row.slot is None else tuple(int(token) for token in row.slot.generated_ids)
        known = (*tuple(int(token) for token in row.prompt_ids), *generated)
        if position > len(known):
            return ()
        return tuple(known[:position])

    def _prefix_prompt_boundary(self, row: _GGUFResidentLoopRow) -> int:
        """Deepest 256-token boundary at or before the end of this row's prompt.

        This is the one boundary a following turn can reach: a cumulative
        client resends the whole transcript, and a client that rebuilds the
        transcript from its own normalized assistant/tool text still starts
        with the previous prompt verbatim.  Boundaries below it are superseded
        by it for both styles.
        """

        if self._prefix_cache is None:
            return 0
        return (len(row.prompt_ids) // 256) * 256

    def _refresh_prefix_cache(self, row: _GGUFResidentLoopRow) -> bool:
        cache = getattr(self, "_prefix_cache", None)
        if cache is None:
            return False
        if row.lease is None or row.kv_allocation is None:
            return False
        phase_start = time.perf_counter()
        processed_start = time.perf_counter()
        tokens = self._processed_tokens(row)
        self._prefix_phase_add("processed_tokens", processed_start)
        if not tokens or len(tokens) % 256 != 0:
            # Keep the latest exact aligned boundary live while the request
            # advances through a partial page. Normal completion can then
            # promote that historical snapshot before request ownership drops.
            self._prefix_phase_add("refresh_unaligned", phase_start)
            return False
        trie_start = time.perf_counter()
        cache.cancel(row.request_id)
        block_count = len(tokens) // 256
        block_ids = tuple(int(block_id) for block_id in row.kv_allocation.block_ids[:block_count])
        if len(block_ids) != block_count:
            self._prefix_phase_add("refresh_trie", trie_start)
            self._prefix_phase_add("refresh_unaligned", phase_start)
            return False
        try:
            cache.insert(row.request_id, tokens, block_ids)
        except ValueError as exc:
            self._prefix_phase_add("refresh_trie", trie_start)
            if "conflicting block ids" not in str(exc):
                raise
            self._prefix_unusable_hits += 1
            self._prefix_phase_add("refresh_conflict", phase_start)
            return False
        self._prefix_phase_add("refresh_trie", trie_start)
        self._capture_prefix_snapshot(row, tokens=tokens, block_ids=block_ids)
        self._prefix_phase_add("refresh_total", phase_start)
        return True

    def _prefix_cache_observability(self) -> dict[str, Any]:
        cache = getattr(self, "_prefix_cache", None)
        pool = getattr(self, "_kv_pool", None)
        entries = tuple(getattr(self, "_prefix_state_snapshots", {}).values())
        retained_entries = tuple(entry for entry in entries if entry.retained)
        retained_blocks = {
            int(block_id)
            for entry in retained_entries
            for block_id in entry.block_ids
        }
        page_bytes = 0 if pool is None else int(pool.page_bytes)
        snapshot_bytes = sum(
            int(getattr(entry.snapshot, "nbytes", 0)) for entry in entries
        )
        max_snapshot_bytes = max(
            (int(getattr(entry.snapshot, "nbytes", 0)) for entry in entries),
            default=0,
        )
        retained_kv_bytes = len(retained_blocks) * page_bytes
        pool_capacity_bytes = 0
        if pool is not None:
            stats = pool.stats
            capacity_pages = (
                int(pool.high_water_pages)
                if pool.high_water_pages is not None
                else int(stats.current_pages)
            )
            pool_capacity_bytes = capacity_pages * page_bytes
        return {
            "mode": getattr(self, "_prefix_cache_mode", "off"),
            "block_size_tokens": 256,
            "stats": None if cache is None else cache.stats.to_json_dict(),
            "usable_hits": int(getattr(self, "_prefix_usable_hits", 0)),
            "unusable_hits": int(getattr(self, "_prefix_unusable_hits", 0)),
            # Which prefill route the hits got: a contiguous run keeps the
            # slot-local AOTriton route, a gapped one falls to packed paged.
            "contiguous_admissions": int(
                getattr(self, "_prefix_contiguous_admissions", 0)
            ),
            "gapped_admissions": int(getattr(self, "_prefix_gapped_admissions", 0)),
            "admission_fallbacks": int(
                getattr(self, "_prefix_admission_fallbacks", 0)
            ),
            "fallback_reasons": dict(
                getattr(self, "_prefix_fallback_reasons", {})
            ),
            "reused_tokens": int(getattr(self, "_prefix_reused_tokens", 0)),
            "state_clone_bytes": int(
                getattr(self, "_prefix_state_clone_bytes", 0)
            ),
            "snapshot_entries": len(entries),
            "snapshot_limit": int(
                getattr(self, "_prefix_snapshot_limit", getattr(self, "capacity", 0))
            ),
            "retained_snapshot_limit": int(
                getattr(
                    self,
                    "_prefix_retained_limit",
                    getattr(self, "_prefix_snapshot_limit", 0),
                )
            ),
            "retained_snapshot_state_limit_bytes": int(
                getattr(
                    self,
                    "_prefix_retained_state_bytes_limit",
                    _PREFIX_RETAINED_STATE_BYTES_LIMIT,
                )
            ),
            "retained_snapshot_state_bytes": int(
                sum(
                    int(getattr(entry.snapshot, "nbytes", 0))
                    for entry in retained_entries
                )
            ),
            "retained_snapshot_evictions_by_reason": dict(
                getattr(self, "_prefix_retained_evictions_by_reason", {})
            ),
            "snapshot_promotions": int(
                getattr(self, "_prefix_snapshot_promotions", 0)
            ),
            "retained_snapshot_entries": len(retained_entries),
            "snapshot_hits": int(getattr(self, "_prefix_snapshot_hits", 0)),
            "snapshot_evictions": int(
                getattr(self, "_prefix_snapshot_evictions", 0)
            ),
            "snapshot_captures": int(
                getattr(self, "_prefix_snapshot_captures", 0)
            ),
            "snapshot_capture_bytes": int(
                getattr(self, "_prefix_snapshot_capture_bytes", 0)
            ),
            "phase_ms": dict(getattr(self, "_prefix_phase_ms", {})),
            "phase_calls": dict(getattr(self, "_prefix_phase_calls", {})),
            "snapshot_bytes": snapshot_bytes,
            "retained_kv_pages": len(retained_blocks),
            "retained_kv_bytes": retained_kv_bytes,
            "resident_bytes": snapshot_bytes + retained_kv_bytes,
            "resident_limit_bytes": (
                int(
                    getattr(
                        self,
                        "_prefix_snapshot_limit",
                        getattr(self, "capacity", 0),
                    )
                )
                * max_snapshot_bytes
                + pool_capacity_bytes
            ),
        }

    def _prefix_request_telemetry(
        self,
        row: _GGUFResidentLoopRow,
    ) -> dict[str, Any]:
        pool = getattr(self, "_kv_pool", None)
        page_bytes = 0 if pool is None else int(pool.page_bytes)
        reused_pages = (
            0
            if row.kv_allocation is None
            else len(row.kv_allocation.reused_block_ids)
        )
        residency = self._prefix_cache_observability()
        mode = getattr(self, "_prefix_cache_mode", "off")
        fallback_reason = row.prefix_fallback_reason
        if fallback_reason is None and mode == "off":
            fallback_reason = "cache_off"
        return {
            "mode": mode,
            "block_size_tokens": 256,
            "eligible": bool(row.prefix_eligible),
            "lookup": bool(row.prefix_lookup),
            "hit": bool(row.prefix_reused_tokens),
            "source": row.prefix_source_kind,
            "matched_tokens": int(row.prefix_matched_tokens),
            "reused_tokens": int(row.prefix_reused_tokens),
            "avoided_prefill_tokens": int(row.prefix_reused_tokens),
            "executed_prefill_tokens": max(
                0, len(row.prompt_ids) - int(row.prefix_reused_tokens)
            ),
            "reused_pages": int(reused_pages),
            "reused_page_bytes": int(reused_pages) * page_bytes,
            "state_clone_bytes": int(row.prefix_state_clone_bytes),
            "snapshot_hit": bool(row.prefix_snapshot_hit),
            "admission_fallback": bool(row.prefix_admission_fallback),
            "fallback_reason": fallback_reason,
            "cache_resident_entries": int(residency["snapshot_entries"]),
            "cache_resident_pages": int(residency["retained_kv_pages"]),
            "cache_resident_bytes": int(residency["resident_bytes"]),
        }

    def _request_diagnostics(
        self,
        row: _GGUFResidentLoopRow,
        *,
        include_kv_layout: bool = True,
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "prefix_cache": self._prefix_request_telemetry(row)
        }
        accounting = speculative_output_accounting(row)
        if accounting is not None:
            diagnostics["specdec2_mtp2"] = accounting
        slot = row.slot
        audit = (
            None
            if slot is None
            else getattr(slot.session, "device_kv_layout_audit", None)
        )
        if include_kv_layout and callable(audit):
            payload = audit()
            if isinstance(payload, Mapping):
                diagnostics["kv_layout"] = copy.deepcopy(dict(payload))
        return diagnostics

    def _refresh_prefix_cache_at_prompt_boundary(
        self,
        row: _GGUFResidentLoopRow,
        lease: _GGUFResidentSessionLease,
    ) -> bool:
        """Capture the single reusable boundary of this request.

        Each captured boundary clones the full hybrid Conv/GDN state and
        synchronizes the device.  Capturing every 256-token prefill chunk
        therefore costs one full state clone per chunk while only the deepest
        prompt-aligned boundary is reachable by the next turn, so the chunked
        greedy prefill path captures exactly that one.
        """

        boundary = self._prefix_prompt_boundary(row)
        if boundary <= 0:
            return False
        if boundary != int(getattr(lease.session, "position", -1)):
            return False
        return self._refresh_prefix_cache(row)

    def _capture_prefix_snapshot(
        self,
        row: _GGUFResidentLoopRow,
        *,
        tokens: tuple[int, ...],
        block_ids: tuple[int, ...],
    ) -> None:
        if not self._prefix_reuse_supported(row) or tokens in self._prefix_state_snapshots:
            return
        lease = row.lease
        if lease is None:
            return
        session = lease.session
        scratch = getattr(session, "scratch", None)
        if scratch is None or len(tokens) >= int(getattr(scratch, "max_positions", 0)):
            return
        capture = getattr(session, "capture_prefix_state_snapshot", None)
        if not callable(capture):
            return
        phase_start = time.perf_counter()
        clone_start = time.perf_counter()
        snapshot = capture(position=len(tokens))
        self._prefix_phase_add("capture_clone_state", clone_start)
        if int(getattr(snapshot, "position", -1)) != len(tokens):
            close = getattr(snapshot, "close", None)
            if callable(close):
                close()
            raise RuntimeError("GGUF prefix snapshot returned the wrong position")
        if tuple(int(block_id) for block_id in getattr(snapshot, "block_ids", ())) != block_ids:
            close = getattr(snapshot, "close", None)
            if callable(close):
                close()
            raise RuntimeError("GGUF prefix snapshot returned the wrong block ids")
        for prior_tokens, entry in tuple(self._prefix_state_snapshots.items()):
            if not entry.retained and entry.owner_request_id == row.request_id:
                if entry.prompt_boundary:
                    # The prompt-aligned boundary is the one every following turn
                    # reaches. A decode-time capture -- the decode path refreshes
                    # the cache after every token, so any reply long enough to
                    # cross the next 256-token boundary makes one -- is reachable
                    # only by a client that resends this request's generated text
                    # verbatim, so it may not evict the boundary entry: doing that
                    # left the next turn with no live prefix at all and made it
                    # re-prefill the whole prompt.
                    continue
                self._evict_prefix_snapshot(prior_tokens, reason="superseded_by_row")
        self._prefix_state_snapshots[tokens] = _GGUFPrefixSnapshotEntry(
            tokens=tokens,
            block_ids=block_ids,
            snapshot=snapshot,
            owner_request_id=int(row.request_id),
            prompt_boundary=len(tokens) == self._prefix_prompt_boundary(row),
        )
        self._prefix_snapshot_captures += 1
        self._prefix_snapshot_capture_bytes += int(
            getattr(snapshot, "nbytes", 0)
        )
        evict_start = time.perf_counter()
        self._trim_prefix_snapshots()
        self._prefix_phase_add("capture_evict", evict_start)
        self._prefix_phase_add("capture_total", phase_start)

    def _trim_prefix_snapshots(self) -> None:
        """Bound transient and retained snapshots independently.

        A transient snapshot belongs to the request that captured it and can be
        dropped freely. A retained one is the durable boundary another request
        may still match, so the transient trim must never take it: doing so made
        a request's own second capture evict the entry that had just served it
        and limited reuse to one hand-off per conversation.
        """

        transient = [
            tokens
            for tokens, entry in self._prefix_state_snapshots.items()
            if not entry.retained and not entry.prompt_boundary
        ]
        while len(transient) > self._prefix_snapshot_limit:
            self._evict_prefix_snapshot(transient.pop(0), reason="trim_transient")
        retained = [
            tokens
            for tokens, entry in self._prefix_state_snapshots.items()
            if entry.retained
        ]
        retained_bytes = sum(
            int(getattr(self._prefix_state_snapshots[tokens].snapshot, "nbytes", 0))
            for tokens in retained
        )
        while retained and (
            len(retained) > self._prefix_retained_limit
            or retained_bytes > self._prefix_retained_state_bytes_limit
        ):
            tokens = retained.pop(0)
            retained_bytes -= int(
                getattr(self._prefix_state_snapshots[tokens].snapshot, "nbytes", 0)
            )
            self._evict_prefix_snapshot(tokens, reason="trim_retained")

    def _promote_prefix_snapshots(self, row: _GGUFResidentLoopRow) -> None:
        cache = self._prefix_cache
        pool = self._kv_pool
        allocation = row.kv_allocation
        if cache is None or pool is None or allocation is None:
            self._drop_prefix_snapshots_for_row(row.request_id)
            return
        for tokens, entry in tuple(self._prefix_state_snapshots.items()):
            if entry.retained or entry.owner_request_id != row.request_id:
                continue
            prefix = tuple(
                int(block_id)
                for block_id in allocation.block_ids[: len(entry.block_ids)]
            )
            if prefix != entry.block_ids:
                self._evict_prefix_snapshot(tokens, reason="promote_mismatch")
                continue
            pool.retain_blocks(entry.block_ids)
            try:
                cache.retain_entry(tokens, entry.block_ids)
            except Exception:
                pool.release_blocks(entry.block_ids)
                self._evict_prefix_snapshot(tokens, reason="promote_failed")
                raise
            entry.owner_request_id = None
            entry.retained = True
            self._prefix_snapshot_promotions += 1
        # Retained entries are trimmed oldest-first, and a decode-time boundary
        # is inserted after the prompt-aligned one it must never outlive: move
        # the boundary entry last so a budget squeeze drops the boundary only a
        # verbatim continuation could reach first.
        for tokens, entry in tuple(self._prefix_state_snapshots.items()):
            if entry.retained and entry.prompt_boundary:
                self._prefix_state_snapshots.pop(tokens)
                self._prefix_state_snapshots[tokens] = entry
        self._trim_prefix_snapshots()

    def _drop_prefix_snapshots_for_row(self, request_id: int) -> None:
        rid = int(request_id)
        for tokens, entry in tuple(self._prefix_state_snapshots.items()):
            if not entry.retained and entry.owner_request_id == rid:
                self._evict_prefix_snapshot(tokens, reason="row_released")

    def _note_prefix_fallback(
        self, row: _GGUFResidentLoopRow, reason: str | None
    ) -> None:
        """Record why one row could not reuse a prefix, per reason.

        The served harness only sees process-wide counters, so a fallback that
        is not counted here is invisible in a serving artifact.
        """

        row.prefix_fallback_reason = reason
        if reason:
            counts = self._prefix_fallback_reasons
            counts[reason] = int(counts.get(reason, 0)) + 1

    def _note_prefix_admission_fallback(
        self, row: _GGUFResidentLoopRow, reason: str
    ) -> None:
        self._prefix_admission_fallbacks += 1
        row.prefix_admission_fallback = True
        self._note_prefix_fallback(row, reason)

    def _evict_prefix_snapshot(
        self,
        tokens: Sequence[int],
        *,
        reason: str = "unspecified",
    ) -> bool:
        token_tuple = tuple(int(token) for token in tokens)
        entry = self._prefix_state_snapshots.pop(token_tuple, None)
        if entry is None:
            return False
        discard_provider = getattr(
            getattr(self, "_mtp2_adapter", None), "discard_prefix_checkpoint", None,
        )
        if callable(discard_provider):
            discard_provider(token_tuple)
        if entry.retained:
            cache = self._prefix_cache
            pool = self._kv_pool
            if cache is None or pool is None:
                raise RuntimeError("GGUF retained prefix snapshot outlived cache ownership")
            cache.evict_entry(token_tuple)
            pool.release_blocks(entry.block_ids)
        close = getattr(entry.snapshot, "close", None)
        if callable(close):
            close()
        self._prefix_snapshot_evictions += 1
        if entry.retained:
            by_reason = self._prefix_retained_evictions_by_reason
            by_reason[reason] = int(by_reason.get(reason, 0)) + 1
        return True

    def _clear_prefix_snapshots(self) -> None:
        for tokens in tuple(self._prefix_state_snapshots):
            self._evict_prefix_snapshot(tokens, reason="clear")

    def evict_prefix_cache_for_pressure(self, required_pages: int) -> int:
        """Release reclaimable prefix pages before device-pool growth."""

        needed = max(1, int(required_pages))
        released = 0
        for tokens, entry in tuple(self._prefix_state_snapshots.items()):
            if not entry.retained:
                continue
            pages = len(tuple(entry.block_ids))
            if self._evict_prefix_snapshot(tokens, reason="pool_pressure"):
                released += pages
            if released >= needed:
                break
        return released

    def rollback_admission(self, request: RequestState) -> None:
        """Undo a bound KV/session lease that was never published active."""

        row = self._row(request.request_id)
        self._release_row_resources(row)

    def loop_barrier(self, *, active_count: int, pending_count: int) -> None:
        """Run allocator maintenance only between complete model transitions."""

        del active_count, pending_count
        pool = self._kv_pool
        if pool is None:
            return
        pool.shrink_idle(now_seconds=time.monotonic())
        self._sample_kv_hip_memory()

    def prompt_tokens(self, prompt: PromptInput) -> tuple[int, ...]:
        tokens = tuple(_encode_prompt(self.generator.tokenizer, prompt))
        if not tokens:
            raise ValueError("GGUF prompt tokenization produced no token IDs")
        return tokens

    def record_prompt_tokenize_ms(
        self,
        request_ids: Sequence[int],
        tokenize_ms: Sequence[float],
    ) -> None:
        ids = tuple(int(request_id) for request_id in request_ids)
        values = tuple(max(0.0, float(value)) for value in tokenize_ms)
        if len(ids) != len(values):
            raise ValueError("request_ids and tokenize_ms must have the same length")
        for request_id, value in zip(ids, values, strict=True):
            row = self._row(request_id)
            row.tokenize_ms = value
            row.prompt_encode_ms = value

    def scheduler_max_new_tokens(self, request: GenerationRequest) -> int:
        if int(request.max_tokens) > 0:
            return int(request.max_tokens)
        # Zero-token requests execute as one declared resident compatibility
        # transition so the scheduler can publish an empty completed output.
        return 1

    def _resolved_mtp2_adapter(self):
        if self._mtp2_adapter is not None:
            return self._mtp2_adapter
        if self._mtp2_adapter_resolved:
            return None
        self._mtp2_adapter_resolved = True
        capability_name = (
            "GGUF_SPECDEC2_MTP2_C1"
            if int(self.capacity) == 1
            else "GGUF_SPECDEC2_MTP2_PHYSICAL"
        )
        enabled = bool(
            backend_package_capability(
                self.generator.backend,
                capability_name,
                False,
            )
        )
        if not enabled or not self.generator.supports_speculative_mtp:
            _trace_mtp2_resolution(
                f"adapter unresolved: capability {capability_name}={enabled}, "
                f"supports_speculative_mtp="
                f"{bool(self.generator.supports_speculative_mtp)}"
            )
            return None
        adapter_key = str(
            getattr(self.generator.model_plugin, "speculative_mtp2_adapter", "")
            or ""
        ).strip()
        if not adapter_key:
            _trace_mtp2_resolution("adapter unresolved: model plugin has no adapter key")
            return None
        from hipengine.generation.qwen35_gguf_mtp2_registry import (
            register_builtin_gguf_mtp2_adapters,
            resolve_gguf_mtp2_adapter,
        )

        register_builtin_gguf_mtp2_adapters()
        try:
            factory = resolve_gguf_mtp2_adapter(adapter_key)
        except KeyError:
            return None
        quant_resolver = getattr(self.generator, "_kv_weight_quant_key", None)
        quant = (
            str(quant_resolver())
            if callable(quant_resolver)
            else str(getattr(self.generator.model_plugin, "default_quant", ""))
        )
        self._mtp2_adapter = factory(
            self,
            enabled=True,
            target_verify_mode=_gguf_mtp_server_target_verify_mode(),
            candidate_budget=min(
                _qwen35_gguf_mtp2_module.MTP2_MAX_CANDIDATE_DEPTH,
                int(
                    getattr(
                        self.generator,
                        "speculative_candidate_budget",
                        _gguf_mtp_server_candidate_budget(),
                    )
                ),
            ),
            quant=quant,
        )
        self._configure_mtp2_prefix_checkpoint_capture(self._mtp2_adapter)
        return self._mtp2_adapter

    def _configure_mtp2_prefix_checkpoint_capture(self, adapter: Any) -> None:
        """Publish the prefix-capture key to the adapter.

        The provider's checkpoint describes a prefix the target's cache holds,
        so both halves of its key are the runner's: the block size that decides
        which boundary is worth taking, and the block ids a later lookup
        validates against. With target prefix caching off there is nothing to
        restore, so provider checkpoints are disabled as well.
        """

        capacity = (
            _gguf_mtp2_prefix_checkpoint_entries()
            if getattr(self, "_prefix_cache_mode", "off") != "off" else 0
        )
        adapter.prefix_checkpoint_capacity = capacity
        adapter.prefix_checkpoint_block_size = int(_GGUF_PREFIX_CACHE_BLOCK_TOKENS)
        adapter.prefix_checkpoint_key = (
            self._prefix_checkpoint_block_ids if capacity > 0 else None
        )
        try:
            if os.environ.get("HIPENGINE_MTP2_TRACE_DECLINE", "").strip() not in {"", "0"}:
                print(
                    "[mtp2-prefix-checkpoint] configured "
                    f"entries={capacity} block={int(_GGUF_PREFIX_CACHE_BLOCK_TOKENS)}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception:
            pass

    def _prefix_checkpoint_block_ids(
        self, request_id: int, prefix_len: int
    ) -> tuple[int, ...] | None:
        """Return the target's block ids covering ``prefix_len`` tokens.

        The store validates a lookup against exactly these ids, so this must
        answer for the prefix in question rather than for the row's whole
        allocation: a later turn matches a shorter prefix than the row it
        arrived on, and ids past the match would fail a validation that should
        have succeeded.
        """

        row = self._row(int(request_id))
        if row is None:
            return None
        allocation = getattr(row, "kv_allocation", None)
        block_ids = getattr(allocation, "block_ids", None)
        if not block_ids:
            return None
        block_count = int(prefix_len) // int(_GGUF_PREFIX_CACHE_BLOCK_TOKENS)
        if block_count <= 0:
            return None
        available = tuple(int(block_id) for block_id in block_ids)
        if len(available) < block_count:
            return None
        return available[:block_count]

    def register_speculative_request(
        self,
        request_id: int,
        candidate_budget: int,
        *,
        static_eligibility=None,
    ) -> None:
        row = self._row(request_id)
        adapter = self._resolved_mtp2_adapter()
        adapter_budget = int(getattr(adapter, "candidate_budget", candidate_budget))
        evidence_budget = int(
            getattr(static_eligibility, "max_candidate_count", candidate_budget)
        )
        effective_budget = min(
            max(1, int(candidate_budget)),
            max(1, adapter_budget),
            max(1, evidence_budget),
        )
        row.mtp2_candidate_budget = effective_budget
        # Admission intent is never zeroed by a transient refusal: a prompt
        # activation that was refused while another was in flight keeps its
        # requested depth, because whether the row can speculate is decided by
        # its priming source rather than by that refusal. An *unsupported*
        # priming source is a property of the row on this route -- a reused
        # prefix, an unsupported target profile, a prompt past the context
        # window, an operator-disabled streaming path -- so those refusals do
        # zero the budget, and the response reports intent separately from
        # realized execution via ``mtp2_requested_budget``.
        row.mtp2_requested_budget = max(
            int(getattr(row, "mtp2_requested_budget", 0) or 0), effective_budget
        )
        if adapter is not None:
            adapter.register_request(
                request_id,
                effective_budget,
                static_eligibility=static_eligibility,
            )

    def speculative_desired_candidate_count(self, request: GenerationRequest) -> int:
        adapter = self._resolved_mtp2_adapter()
        max_budget = int(
            getattr(
                adapter,
                "candidate_budget",
                _qwen35_gguf_mtp2_module.MTP2_MAX_CANDIDATE_DEPTH,
            )
        )
        return min(max_budget, max(1, int(request.max_tokens)))

    def speculative_eos_supported(self, request_id: int) -> bool:
        """Whether the adapter owns EOS-aware state selection for this row."""

        adapter = self._resolved_mtp2_adapter()
        if adapter is None or not adapter.enabled:
            return False
        supports = getattr(adapter, "eos_finish_supported", None)
        if not callable(supports):
            return False
        return bool(supports(int(request_id)))

    def speculative_capability(self, request_semantics):
        adapter = self._resolved_mtp2_adapter()
        return None if adapter is None else adapter.capability(request_semantics)

    def speculative_post_reject_cooldown(self, request_ids):
        adapter = self._resolved_mtp2_adapter()
        resolve = (
            None if adapter is None else getattr(adapter, "post_reject_cooldown", None)
        )
        if not callable(resolve):
            return tuple(False for _ in request_ids)
        return tuple(
            bool(flag) for flag in resolve(tuple(int(value) for value in request_ids))
        )

    def speculative_graph_available(self, work) -> bool:
        del work
        adapter = self._resolved_mtp2_adapter()
        if adapter is None:
            return False
        # The plan is constructed after this cold hook, so use the current c1
        # intent and session graph cache directly in the adapter's later
        # telemetry. Uncaptured S3 shapes conservatively plan eager.
        return False

    def speculative_partition_max_requests(self, work) -> int:
        adapter = self._resolved_mtp2_adapter()
        resolve = None if adapter is None else getattr(adapter, "partition_max_requests", None)
        return 0 if not callable(resolve) else int(resolve(work.request_ids))

    @property
    def server_mtp_batch_max_active_requests(self) -> int | None:
        """Explicit-MTP batch-route width owned by the resolved adapter."""

        adapter = self._resolved_mtp2_adapter()
        return None if adapter is None else int(adapter.physical_request_bound)

    def speculative_claims_fit(self, plan) -> bool:
        adapter = self._resolved_mtp2_adapter()
        return bool(adapter is not None and adapter.claims_fit(plan))

    def speculative_frontier_available(self, plan) -> bool:
        adapter = self._resolved_mtp2_adapter()
        return bool(
            adapter is not None
            and getattr(adapter, "staged_frontier", True)
        )

    def execute_speculative_cycle(self, plan, *, commit: bool):
        self._begin_speculative_cycle(plan)
        adapter = self._resolved_mtp2_adapter()
        execute = None if adapter is None else getattr(adapter, "execute_cycle", None)
        if not callable(execute):
            raise NotImplementedError("GGUF MTP2 adapter has no bounded complete cycle")
        with hip_target_arch_environment(self.generator.target_arch):
            return execute(plan, commit=bool(commit))

    def prepare_speculative_k0(self, plan, request_semantics, *, stream=None) -> None:
        self._begin_speculative_cycle(plan)
        adapter = self._resolved_mtp2_adapter()
        if adapter is not None:
            with hip_target_arch_environment(self.generator.target_arch):
                adapter.prepare_k0(plan, request_semantics, stream=stream)

    def note_speculative_ar_commit(self, request_id, reason=None) -> None:
        """Forward one emitted autoregressive token to the resolved adapter."""

        adapter = self._resolved_mtp2_adapter()
        observe = (
            None
            if adapter is None
            else getattr(adapter, "note_speculative_ar_commit", None)
        )
        if callable(observe):
            with hip_target_arch_environment(self.generator.target_arch):
                observe(request_id, reason)

    def note_speculative_plan(self, plan) -> None:
        """Publish the group-level plan decision on every row it planned.

        The loop owns the plan; the row owns the per-request reporting. Without
        this the served response could only report the route's intent
        (``k0_class``) and a serving-key width, so a row that decoded
        autoregressively for its whole life still read as ``not_k0`` with no way
        to see the group it was actually planned in.
        """

        request_ids = tuple(int(value) for value in plan.request_ids)
        reasons = tuple(getattr(plan, "reasons", ()) or ())
        ar_only = bool(getattr(plan, "is_ar_only", False))
        for index, request_id in enumerate(request_ids):
            row = self._row(request_id)
            if row is None:
                continue
            reason = reasons[index] if index < len(reasons) else None
            record_speculative_plan(
                row,
                group_rows=len(request_ids),
                ar_only=ar_only,
                plan_reason=getattr(reason, "value", reason),
            )

    def note_provider_readiness(
        self, request_id, readiness, reason=None, state_present=None
    ) -> None:
        """Publish the row's own draft-provider readiness and decline reason."""

        row = self._row(int(request_id))
        if row is None:
            return
        record_provider_readiness(
            row,
            readiness=readiness,
            decline_reason=reason,
            state_present=state_present,
        )

    def speculative_component_claims(self, plan):
        adapter = self._resolved_mtp2_adapter()
        if adapter is None:
            raise RuntimeError("GGUF MTP2 adapter is unavailable")
        return adapter.component_claims(plan)

    def reserve_speculative_claims(self, claims):
        adapter = self._resolved_mtp2_adapter()
        if adapter is None:
            raise RuntimeError("GGUF MTP2 adapter is unavailable")
        return adapter.reserve_claims(claims)

    def release_speculative_claims(self, reservation) -> None:
        adapter = self._resolved_mtp2_adapter()
        if adapter is None:
            raise RuntimeError("GGUF MTP2 adapter is unavailable")
        with hip_target_arch_environment(self.generator.target_arch):
            adapter.release_claims(reservation)

    def prepare_speculative_requests(self, plan, request_semantics, *, stream=None) -> None:
        self._begin_speculative_cycle(plan)
        adapter = self._resolved_mtp2_adapter()
        if adapter is None:
            raise RuntimeError("GGUF MTP2 adapter is unavailable")
        with hip_target_arch_environment(self.generator.target_arch):
            adapter.prepare_requests(plan, request_semantics, stream=stream)

    def propose_speculative_batch(self, plan, request_semantics, *, stream=None):
        self._begin_speculative_cycle(plan)
        adapter = self._resolved_mtp2_adapter()
        if adapter is None:
            raise RuntimeError("GGUF MTP2 adapter is unavailable")
        with hip_target_arch_environment(self.generator.target_arch):
            return adapter.propose_batch(plan, request_semantics, stream=stream)

    def speculative_kv_live_spans_owner(self, plan) -> str:
        return f"gguf-resident:{id(self)}:{plan.operation_id}"

    def execute_target_frontier(
        self,
        plan,
        frontier,
        complete_claims,
        *,
        commit: bool,
        cancelled_request_ids,
    ):
        self._begin_speculative_cycle(plan)
        adapter = self._resolved_mtp2_adapter()
        if adapter is None:
            raise RuntimeError("GGUF MTP2 adapter is unavailable")
        with hip_target_arch_environment(self.generator.target_arch):
            return adapter.execute_target_frontier(
                plan,
                frontier,
                complete_claims,
                commit=commit,
                cancelled_request_ids=cancelled_request_ids,
            )

    def rollback_speculative_cycle(self, plan, candidate_graph, error) -> None:
        adapter = self._resolved_mtp2_adapter()
        if adapter is not None:
            with hip_target_arch_environment(self.generator.target_arch):
                adapter.rollback_cycle(plan, candidate_graph, error)

    def recover_speculative_cycle_failure(self, plan, error) -> bool:
        adapter = self._resolved_mtp2_adapter()
        if adapter is None:
            return False
        with hip_target_arch_environment(self.generator.target_arch):
            return bool(adapter.recover_cycle_failure(plan, error))

    def restore_speculative_target_rows(self, plan) -> bool:
        """Rebuild canonical target state after an uncertain selected commit."""

        return self.restore_speculative_target_request_ids(
            plan.speculative_request_ids
        )

    def restore_speculative_target_request_ids(
        self,
        request_ids: Sequence[int],
        *,
        require_token_match: bool = True,
    ) -> bool:
        """Rebuild named target rows from scheduler-authoritative token history."""

        request_ids = tuple(int(value) for value in request_ids)
        rows = tuple(self._rows.get(request_id) for request_id in request_ids)
        if not rows or any(row is None for row in rows):
            return False
        concrete = tuple(row for row in rows if row is not None)
        if any(row.slot is None or row.lease is None for row in concrete):
            return False
        token_rows = tuple(
            (
                *tuple(int(token) for token in row.prompt_ids),
                *tuple(int(token) for token in row.slot.generated_ids[:-1]),
            )
            for row in concrete
        )
        if any(
            not row.slot.generated_ids
            or len(tokens) != int(row.slot.seq_position)
            for row, tokens in zip(concrete, token_rows, strict=True)
        ):
            return False
        self._flush_rows(concrete)
        sessions = tuple(row.lease.session for row in concrete)
        for session in sessions:
            session.reset()
        owner = self._packed_execution_owner(sessions[0])
        prefill_batch = getattr(owner, "prefill_batch_native", None)
        if not callable(prefill_batch):
            return False
        try:
            with _temporary_env({"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}):
                results = prefill_batch(
                    token_rows,
                    sessions=sessions,
                    full_prompt_lengths=[len(tokens) for tokens in token_rows],
                    return_logits=False,
                    return_hidden_seeds=False,
                )
        except NotImplementedError:
            # This rebuild runs after the cycle already published committed
            # tokens, so a shape refusal from the packed route must not end a
            # live stream with unsupported_parameter. Rebuild every row on the
            # registered strict per-session route (the same tokens, the same
            # cursors the caller asserts below) and keep sessions that only the
            # block-table-aware route can represent fail-closed.
            if any(
                _gguf_single_row_block_table_prefill_required(session)
                for session in sessions
            ):
                return False
            results = [
                session.prefill(tokens, return_logits=False)
                for session, tokens in zip(sessions, token_rows, strict=True)
            ]
        result_rows = () if results is None else tuple(results)
        if len(result_rows) != len(concrete):
            raise RuntimeError(
                "SPECDEC2 postcommit target rebuild returned the wrong row count"
            )
        for row, session, result in zip(
            concrete,
            sessions,
            result_rows,
            strict=True,
        ):
            if (
                bool(require_token_match)
                and not bool(getattr(row, "native_sampled", False))
                and int(result.token_id) != int(row.slot.prev_token)
            ):
                raise RuntimeError(
                    "SPECDEC2 postcommit target rebuild changed the canonical token"
                )
            if int(session.position) != int(row.slot.seq_position):
                raise RuntimeError(
                    "SPECDEC2 postcommit target rebuild changed the canonical cursor"
                )
        return True

    def register_batch(
        self,
        request_ids: Sequence[int],
        request: GenerationRequest,
        *,
        prompt_rows: Sequence[Sequence[int]],
    ) -> None:
        prepare_kv_policy = getattr(self.generator, "_prepare_kv_policy", None)
        if callable(prepare_kv_policy):
            prepare_kv_policy(request)
        if not self._available and not self._rows:
            with hip_target_arch_environment(self.generator.target_arch):
                self._reserve_sessions()
                if self._engine_loop_config is not None:
                    self.configure_engine_loop(self._engine_loop_config)
        ids = tuple(int(request_id) for request_id in request_ids)
        prompts = tuple(tuple(int(token) for token in row) for row in prompt_rows)
        if len(ids) != len(request.prompts) or len(prompts) != len(ids):
            raise ValueError("request_ids, prompts, and prompt_rows must have the same length")
        if request.row_seeds and len(request.row_seeds) != len(request.prompts):
            raise ValueError("row_seeds must have one entry per prompt")
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        plan = _gguf_sampler_plan(
            request,
            native_gpu_available=_native_gpu_sampler_requested(),
        )
        native_greedy = (
            plan.mode is SamplingMode.GREEDY_FAST
            and int(request.max_tokens) > 0
        )
        native_sampled = (
            plan.mode is not SamplingMode.GREEDY_FAST
            and int(request.max_tokens) > 0
        )
        native_sampler = (
            native_sampled
            and _gguf_native_sampler_plan_enabled(request, plan)
        )
        now = time.perf_counter()
        for row_index, (request_id, prompt_ids) in enumerate(zip(ids, prompts, strict=True)):
            if request_id in self._rows or request_id in self._outputs:
                raise ValueError(f"request_id {request_id} is already registered")
            if not prompt_ids:
                raise ValueError("GGUF prompt tokenization produced no token IDs")
            source_prompt = request.prompts[row_index]
            prompt_encode_ms = max(
                0.0,
                float(getattr(source_prompt, "tokenize_ms", 0.0)),
            )
            self._rows[request_id] = _GGUFResidentLoopRow(
                request_id=request_id,
                batch_id=batch_id,
                row_index=row_index,
                request=request,
                prompt_ids=prompt_ids,
                native_greedy=native_greedy,
                native_sampled=native_sampled,
                submitted_at=now,
                tokenize_ms=prompt_encode_ms,
                prompt_encode_ms=prompt_encode_ms,
                render_ms=max(0.0, float(getattr(source_prompt, "render_ms", 0.0))),
                admission_prepare_ms=max(
                    0.0,
                    float(getattr(source_prompt, "admission_prepare_ms", 0.0)),
                ),
                native_sampler=native_sampler,
                sampler_plan=plan,
            )

    def prepare(self, *, max_sequence_length: int | None = None) -> None:
        requested = getattr(self.generator, "_prepared_max_sequence_length", None)
        if requested is None and max_sequence_length is not None:
            requested = int(max_sequence_length)
        if requested == self._max_sequence_length and (self._available or self._rows):
            return
        if self._rows:
            raise RuntimeError("cannot resize resident GGUF sessions while requests are active")
        with hip_target_arch_environment(self.generator.target_arch):
            config = self._engine_loop_config
            # The session scratch and the device KV pool are separate
            # allocations, and the pool is the larger one. The session-level
            # fallback in ``_acquire_shared_session`` cannot see a pool
            # failure, so the retry lives here where both are in scope. A pinned
            # context backs off too, and the warning names the size that was
            # asked for, so the substitution is never silent.
            # ``HIPENGINE_GGUF_AUTO_CONTEXT=0`` is the full rollback: it turns
            # off both the automatic sizing and the backoff, restoring the
            # historical hard failure.
            fallback_active = _gguf_auto_context_enabled()
            attempts = _gguf_auto_context_attempts() if fallback_active else 1
            for attempt in range(attempts):
                self._clear_prefix_snapshots()
                if self._kv_pool is not None:
                    self._teardown_kv_pool(release_workspace_state=True)
                self._release_available_sessions()
                self._max_sequence_length = requested
                try:
                    self._reserve_sessions()
                    if config is not None:
                        self.configure_engine_loop(config)
                except (HipError, MemoryError) as exc:
                    if (
                        not fallback_active
                        or not _gguf_allocation_failure(exc)
                        or attempt + 1 >= attempts
                    ):
                        raise
                    current = requested
                    if current is None:
                        current = getattr(
                            self.generator, "_auto_resolved_max_sequence_length", None
                        )
                    if current is None:
                        raise
                    next_context = self.generator._recalibrated_auto_context(
                        self._shared_runner,
                        failed_context=int(current),
                        max_batch_size=int(self.capacity),
                        defer_kv_allocation=True,
                    )
                    if next_context >= int(current):
                        raise
                    _LOGGER.warning(
                        "GGUF context request: requested %d tokens failed to allocate "
                        "(%s); retrying at %d tokens",
                        int(current),
                        exc,
                        next_context,
                    )
                    if requested is None:
                        # Automatic sizing owns the selection, so keep the
                        # generator's cache in step for the next attempt.
                        self.generator._record_auto_context_selection(
                            max_batch_size=int(self.capacity),
                            defer_kv_allocation=True,
                            context_tokens=int(next_context),
                        )
                    else:
                        requested = int(next_context)
                    continue
                return
            raise MemoryError(
                "GGUF resident context sizing exhausted its attempts"
            )  # pragma: no cover - the loop always returns or raises

    def _try_prefill_native_work_batch(self, work: WorkItem) -> frozenset[int]:
        """Run one full-prompt scheduler work item as native cN.

        Returns the request ids the grouped call actually consumed; callers must fall
        back per request for everything else, so the return type is a container and
        never a bool - an earlier `False` on the no-native-owner path made the caller's
        `request_id in handled` raise TypeError instead of prefilling serially.

        Prompt lengths may differ across rows: the call forwards ``full_prompt_lengths``
        per row, which is how the serving route already drives this entry point. The
        earlier equal-length requirement was stricter than that ABI and kept the C1-C8
        matrix on one-request-at-a-time prefill, because no realistic 8-wide wave has
        uniform prompt lengths.
        """

        if (
            len(work.request_ids) <= 1
            or len(work.request_ids) > self.packed_prefill_max_rows
            or not _gguf_ar_packed_prefill_enabled()
        ):
            return frozenset()
        rows = [self._row(request_id) for request_id in work.request_ids]
        chunks = [tuple(int(token) for token in token_row) for token_row in work.token_rows]
        # Group the compatible subset instead of refusing the whole item: one lane
        # whose chunk is truncated by the prefill chunk cap, or that reuses a prefix,
        # must not cost every other lane its grouped prefill. The caller prefills
        # whatever this returns-unhandled through the serial path.
        eligible = [
            (row, chunk)
            for row, chunk in zip(rows, chunks, strict=True)
            if chunk
            and row.native_greedy
            and row.slot is None
            and row.prefill_tokens_seen == 0
            and not row.prefix_reused_tokens
            and chunk == row.prompt_ids
        ]
        if len(eligible) <= 1:
            return frozenset()
        rows, chunks = [row for row, _ in eligible], [chunk for _, chunk in eligible]
        for row in rows:
            self._execution_step_request_id = int(row.request_id)
            raise_if_generation_deadline_expired(row.request)
        self._execution_step_request_id = None
        # From here the grouped call may reach device work for any of these
        # rows, so all of them join the containment scope of a failure.
        self._mark_execution_stepped(row.request_id for row in rows)
        leases: list[_GGUFResidentSessionLease] = []
        for row in rows:
            lease = row.lease or self._acquire_lease()
            row.lease = lease
            leases.append(lease)
        owner = self._packed_execution_owner(leases[0].session)
        prefill_batch = getattr(owner, "prefill_batch_native", None)
        if not callable(prefill_batch):
            return frozenset()
        started = time.perf_counter()
        streaming_sinks = self._begin_mtp2_prompt_streaming(rows)
        streaming = any(sink is not None for sink in streaming_sinks)
        capture_mtp2_hidden = bool(
            not streaming and any(row.mtp2_candidate_budget > 0 for row in rows)
        )
        streaming_kwargs = (
            {
                "target_hidden_chunk_sinks": streaming_sinks,
                "target_hidden_request_ids": tuple(row.request_id for row in rows),
                "target_hidden_chunk_starts": (0,) * len(rows),
            }
            if streaming
            else {}
        )
        try:
            with _temporary_env({"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}):
                results = prefill_batch(
                    chunks,
                    sessions=[lease.session for lease in leases],
                    full_prompt_lengths=[len(chunk) for chunk in chunks],
                    return_logits=False,
                    return_hidden_seeds=capture_mtp2_hidden,
                    **streaming_kwargs,
                )
        except NotImplementedError:
            # The packed route refuses this slab's operation or resource shape.
            # That is a route decision about this group, not a failure of the
            # request: report no rows as handled so the caller prefills them
            # through the serial path, which is the same contract this method
            # already uses for an ineligible lane or a missing packed owner.
            # Raising here instead retired every row of the item with an
            # ``execution_failed`` 500 for a prompt well inside the configured
            # context.
            self._finish_mtp2_prompt_streaming(
                rows,
                streaming_sinks,
                success=False,
            )
            for row, sink in zip(rows, streaming_sinks, strict=True):
                if sink is not None:
                    # The adapter already closed this row's streaming sink as
                    # failed; keep the budget consistent with that so the
                    # serial path does not open a second stream for the row.
                    row.mtp2_candidate_budget = 0
                    row.mtp2_prompt_fallback_reason = "packed_prefill_unsupported_k0"
            self._fallback_reasons["packed_prefill_unsupported_k0"] += 1
            return frozenset()
        except Exception:
            self._finish_mtp2_prompt_streaming(
                rows,
                streaming_sinks,
                success=False,
            )
            raise
        self._finish_mtp2_prompt_streaming(
            rows,
            streaming_sinks,
            success=True,
        )
        result_rows = [] if results is None else list(results)
        if len(result_rows) != len(rows):
            raise RuntimeError(
                "packed scheduler prefill must return one result per request"
            )
        elapsed_ms = _timing_ms_since(started)
        self._route_counts["native_full_prefill_rows"] += len(rows)
        # Distinct from the row counter: rows is also bumped by single-request prefill,
        # so without this a packet cannot prove a wave actually grouped.
        self._route_counts["native_full_prefill_groups"] += 1
        for row, chunk, result in zip(rows, chunks, result_rows, strict=True):
            row.prefill_tokens_seen = len(chunk)
            row.incremental_prefill = False
            row.prefill_ms += elapsed_ms
            row.prefill_chunk_count += 1
            self._refresh_prefix_cache(row)
            self._finish_native_prefill(
                row,
                result,
                native_compact_prefill=True,
            )
            raise_if_generation_deadline_expired(row.request)
        return frozenset(int(row.request_id) for row in rows)

    def contain_execution_failure(
        self,
        error: BaseException,
        *,
        phase: str,
        request_ids: tuple[int, ...],
        work_kind: str,
    ) -> ExecutionFailure | None:
        """Contain one GGUF step the runner can prove is safe to retire.

        Two claims are available for a prefill or decode step.  ``none`` means
        the step raised before any device call, so the failing row's scheduler
        bookkeeping is the whole blast radius.  ``partial`` means the step
        entered its device phase without claiming a canonical commit: the
        runner marks every row immediately before its first device call, so the
        claim names exactly the rows whose device state may have advanced, and
        the loop retires them through the same request-owned release path a
        single-row claim uses.  Every row the step never reached stays canonical
        and is left unnamed.

        A speculative cycle instead asks its resolved adapter for the mutation
        class and the rows it cannot prove canonical (a cycle that recorded a
        canonical commit is ``committed``; a cycle whose device cursors moved
        without one is ``partial``).

        Everything else stays fatal: a ``HipError`` (the device itself reported
        a fault, so shared device state is unproven), a quiesce that cannot be
        established, a phase that does not mark its device window, a device
        phase with no attributed row, and an adapter that cannot narrow the
        mutation window.  Those are ``unknown``, which is never a successful
        containment claim.
        """

        rows = tuple(int(request_id) for request_id in request_ids)
        if not rows:
            return None
        if str(phase) not in {"prefill", "decode"}:
            return self._contain_speculative_execution_failure(
                error,
                phase=phase,
                request_ids=rows,
                work_kind=work_kind,
            )
        if isinstance(error, HipError):
            return None
        stepped = {
            request_id
            for request_id in rows
            if request_id in self._execution_stepped_request_ids
        }
        step_request_id = self._execution_step_request_id
        trigger = (
            int(step_request_id)
            if step_request_id is not None and int(step_request_id) in rows
            else None
        )
        if stepped:
            # Device work is attributed to these rows; the row that raised is
            # part of the scope even when it never reached its own device call,
            # because its failure still has to be reported.
            affected = tuple(
                sorted(stepped | ({trigger} if trigger is not None else set()))
            )
            mutation = "partial"
        elif (
            trigger is not None
            or self._execution_mutation_window == "none"
            or self._execution_device_phase_empty
        ):
            # No row reached device work in this step, so the failure is
            # pre-device for everything the claim names.
            affected = (trigger,) if trigger is not None else rows
            mutation = "none"
        else:
            # The step entered its device phase without attributing device work
            # to any row: the window stays unresolved and the failure is fatal.
            return None
        if not self._quiesce_after_execution_failure(affected):
            return None
        return ExecutionFailure(
            request_ids=affected,
            phase=str(phase),
            work_kind=str(work_kind),
            mutation=mutation,
            reason=f"{type(error).__name__}: {error}",
            error=error,
        )

    def _contain_speculative_execution_failure(
        self,
        error: BaseException,
        *,
        phase: str,
        request_ids: tuple[int, ...],
        work_kind: str,
    ) -> ExecutionFailure | None:
        """Claim a failed speculative cycle the resolved adapter can scope."""

        if str(phase) not in {"speculative_prepare", "speculative_cycle"}:
            return None
        if isinstance(error, HipError):
            return None
        plan = self._execution_speculative_plan
        if plan is None:
            return None
        adapter = self._resolved_mtp2_adapter()
        evidence = (
            None if adapter is None else getattr(adapter, "containment_evidence", None)
        )
        if not callable(evidence):
            return None
        verdict = evidence(plan, error)
        if verdict is None:
            return None
        mutation, affected = verdict
        affected = tuple(int(request_id) for request_id in affected)
        if not affected:
            return None
        if any(request_id not in request_ids for request_id in affected):
            # A scope outside the failed work item is a runner bug, not a
            # containment: refuse it here so the loop never has to reject it.
            return None
        if not self._quiesce_after_execution_failure(affected):
            return None
        return ExecutionFailure(
            request_ids=affected,
            phase=str(phase),
            work_kind=str(work_kind),
            mutation=str(mutation),
            reason=f"{type(error).__name__}: {error}",
            error=error,
        )

    def _begin_execution_step(self) -> None:
        """Reset the per-step containment accounting."""

        self._execution_mutation_window = "none"
        self._execution_step_request_id = None
        self._execution_stepped_request_ids.clear()
        self._execution_committed_request_ids.clear()
        self._execution_device_phase_empty = False

    def _begin_speculative_cycle(self, plan) -> None:
        """Enter a speculative cycle: remember its plan and clear its commits."""

        self._execution_speculative_plan = plan
        self._execution_committed_request_ids.clear()

    def _execution_stepped(self) -> set[int]:
        """Return the stepped-row set, materializing it for lightweight runners."""

        stepped = getattr(self, "_execution_stepped_request_ids", None)
        if stepped is None:
            # Scheduling tests build runners with ``__new__`` and never run the
            # dataclass initializer, so the containment sets are absent until
            # the first step marks a row.
            stepped = set()
            self._execution_stepped_request_ids = stepped
        return stepped

    def _mark_execution_stepped(self, request_ids) -> None:
        """Record that these rows' device state may already have advanced."""

        self._execution_stepped().update(
            int(request_id) for request_id in request_ids
        )

    def note_execution_commit(self, request_id: int) -> None:
        """Record one canonical commit made inside the current cycle.

        A resolved adapter calls this when it publishes a cycle's committed
        tokens for a row.  Containment reads the set to distinguish a
        ``committed`` failure, where the row's device state is ahead of the
        outer scheduler and cannot fall back to autoregressive decoding, from a
        ``partial`` one where no commit was claimed.
        """

        self._execution_committed_request_ids.add(int(request_id))

    def execution_committed_request_ids(self) -> frozenset[int]:
        """Rows whose canonical commit this runner recorded in the current step."""

        return frozenset(self._execution_committed_request_ids)

    def _quiesce_after_execution_failure(self, request_ids: tuple[int, ...]) -> bool:
        """Establish device completion after a failed step, or report that it failed."""

        runtimes: list[Any] = []
        seen: set[int] = set()

        def remember(candidate: Any) -> None:
            if candidate is None or id(candidate) in seen:
                return
            seen.add(id(candidate))
            runtimes.append(candidate)

        for request_id in request_ids:
            row = self._rows.get(int(request_id))
            if row is None:
                continue
            slot = getattr(row, "slot", None)
            session = getattr(slot, "session", None)
            remember(getattr(session, "runtime", None))
            lease = getattr(row, "lease", None)
            remember(getattr(getattr(lease, "session", None), "runtime", None))
        remember(getattr(self._shared_runner, "_runtime", None))
        if not runtimes:
            # No reachable device runtime means no proof that the device is idle.
            return False
        for runtime in runtimes:
            synchronize = getattr(runtime, "device_synchronize", None)
            if not callable(synchronize):
                return False
            try:
                synchronize()
            except BaseException:
                return False
        return True

    def prefill_batch(self, work: WorkItem, *, commit: bool) -> None:
        self._begin_execution_step()
        if not commit:
            raise ValueError("GGUF resident prefill requires commit=True")
        with hip_target_arch_environment(self.generator.target_arch):
            handled = self._try_prefill_native_work_batch(work)
            self._execution_mutation_window = "unknown"
            for request_id, token_row in zip(work.request_ids, work.token_rows, strict=True):
                if int(request_id) in handled:
                    continue
                row = self._row(request_id)
                self._execution_step_request_id = int(request_id)
                start = int(row.prefill_tokens_seen)
                chunk = tuple(int(token) for token in token_row)
                expected = row.prompt_ids[start:start + len(chunk)]
                if chunk != expected:
                    raise RuntimeError(
                        f"GGUF prefill chunk drift for request_id {request_id}: "
                        f"expected {expected!r}, got {chunk!r}"
                    )
                row.prefill_tokens_seen += len(chunk)
                if row.prefill_tokens_seen > len(row.prompt_ids):
                    raise RuntimeError("GGUF prefill consumed beyond the registered prompt")
                final_chunk = row.prefill_tokens_seen == len(row.prompt_ids)
                reused_in_chunk = max(
                    0,
                    min(len(chunk), int(row.prefix_reused_tokens) - start),
                )
                model_chunk = chunk[reused_in_chunk:]
                # From here the row may reach a device call, so it joins the
                # containment scope of a failure later in this step.
                self._mark_execution_stepped((request_id,))
                if row.native_greedy:
                    if row.incremental_prefill is None:
                        row.incremental_prefill = bool(row.prefix_reused_tokens) or not (
                            start == 0 and final_chunk
                        )
                    raise_if_generation_deadline_expired(row.request)
                    if not model_chunk:
                        if final_chunk:
                            raise RuntimeError("GGUF prefix reuse requires an unmatched prompt suffix")
                    elif row.incremental_prefill:
                        self._prefill_native_chunk(row, model_chunk, final_chunk=final_chunk)
                    elif final_chunk:
                        self._prefill_native_row(row)
                    raise_if_generation_deadline_expired(row.request)
                elif (
                    row.native_sampled
                    and row.prefix_eligible
                    and self._prefix_reuse_supported(row)
                ):
                    raise_if_generation_deadline_expired(row.request)
                    if row.prefix_reused_tokens:
                        if not model_chunk:
                            if final_chunk:
                                raise RuntimeError(
                                    "GGUF prefix reuse requires an unmatched prompt suffix"
                                )
                        else:
                            self._prefill_processed_argmax_chunk(
                                row,
                                model_chunk,
                                final_chunk=final_chunk,
                            )
                    elif final_chunk:
                        self._prefill_processed_argmax_chunk(
                            row,
                            row.prompt_ids,
                            final_chunk=True,
                        )
                    raise_if_generation_deadline_expired(row.request)
                elif final_chunk:
                    raise_if_generation_deadline_expired(row.request)
                    if row.native_sampled:
                        self._prefill_sampled_row(row)
                    else:
                        self._run_resident_fallback(row)
                    raise_if_generation_deadline_expired(row.request)

    def decode_batch(self, work: WorkItem, *, commit: bool) -> tuple[GeneratedToken, ...]:
        self._begin_execution_step()
        if not commit:
            raise ValueError("GGUF resident decode requires commit=True")
        request_ids = tuple(int(request_id) for request_id in work.request_ids)
        with hip_target_arch_environment(self.generator.target_arch):
            rows = []
            for request_id in request_ids:
                self._execution_step_request_id = int(request_id)
                rows.append(self._row(request_id))
            for row in rows:
                self._execution_step_request_id = int(row.request_id)
                raise_if_generation_deadline_expired(row.request)
            self._execution_step_request_id = None
            self._execution_mutation_window = "unknown"
            step_rows = [
                row
                for row in rows
                if (row.native_greedy or row.native_sampled) and row.first_token_emitted
            ]
            if step_rows:
                step_request_ids = tuple(row.request_id for row in step_rows)
                if work.slot_ids and work.active_mask:
                    slot_by_request = dict(zip(request_ids, work.slot_ids, strict=True))
                    step_slot_ids = tuple(
                        slot_by_request[request_id] for request_id in step_request_ids
                    )
                    step_slot_set = set(step_slot_ids)
                    step_active_mask = tuple(
                        slot in step_slot_set for slot in range(len(work.active_mask))
                    )
                    step_work = WorkItem(
                        kind=work.kind,
                        request_ids=step_request_ids,
                        row_to_request=step_request_ids,
                        slot_ids=step_slot_ids,
                        active_mask=step_active_mask,
                    )
                else:
                    step_work = WorkItem(
                        kind=work.kind,
                        request_ids=step_request_ids,
                        row_to_request=step_request_ids,
                    )
                self._mark_execution_stepped(step_request_ids)
                self._step_native_rows(step_rows, work=step_work)
            else:
                # This tick's device phase has no packed work: every row's first
                # token came from prefill, so nothing here can have advanced a
                # row's device state.
                self._execution_device_phase_empty = True
            for row in rows:
                self._execution_step_request_id = int(row.request_id)
                raise_if_generation_deadline_expired(row.request)
            self._execution_step_request_id = None

            generated: list[GeneratedToken] = []
            for request_id, row in zip(request_ids, rows, strict=True):
                if not (row.native_greedy or row.native_sampled):
                    output = row.fallback_output
                    if output is None:
                        raise RuntimeError("GGUF resident fallback output is not ready")
                    token_ids = output.generated_token_ids or ()
                    token_id = int(token_ids[-1]) if token_ids else 0
                    generated.append(
                        GeneratedToken(
                            request_id,
                            token_id,
                            finished=True,
                            stream_chunk=GenerationStreamChunk(
                                text=output.text,
                                token_logprobs=output.token_logprobs,
                                finish_details=output.finish_details,
                                telemetry=output.telemetry,
                                generated_token_ids=output.generated_token_ids,
                            ),
                        )
                    )
                    continue
                slot = row.slot
                if slot is None or not slot.generated_ids:
                    raise RuntimeError("GGUF resident model row is not prefilled")
                if not row.first_token_emitted:
                    row.first_token_emitted = True
                generated.append(
                    GeneratedToken(
                        request_id,
                        int(slot.generated_ids[-1]),
                        finished=_gguf_finished(
                            slot.generated_ids,
                            self.generator.tokenizer,
                            row.sampling_request or row.request,
                        ),
                        stream_chunk=self._native_stream_chunk(row),
                    )
                )
            return tuple(generated)

    def compact_batch(self, moves: Sequence[SlotMove]) -> None:
        move_tuple = tuple(moves)
        moved = tuple(move for move in move_tuple if move.old_slot != move.new_slot)
        if moved:
            with hip_target_arch_environment(self.generator.target_arch):
                self._flush_all_packed_owners()
                sessions: list[Any] = []
                seen_sessions: set[int] = set()
                for move in moved:
                    row = self._row(move.request_id)
                    lease = row.lease
                    if lease is None or id(lease.session) in seen_sessions:
                        continue
                    seen_sessions.add(id(lease.session))
                    sessions.append(lease.session)
                session_tuple = tuple(sessions)
                self._observe_graph_handles(session_tuple)
                graph_handles = self._graph_handles_for_sessions(session_tuple)
                invalidated = 0
                if graph_handles:
                    for session in session_tuple:
                        invalidate = getattr(session, "invalidate_device_kv_graphs", None)
                        if callable(invalidate):
                            invalidated += int(invalidate())
                if invalidated:
                    self._record_graph_invalidations(graph_handles, invalidated)
                    self._kv_graph_invalidation_count += invalidated
        # Session state is request-owned, not physical-row-owned.  Compaction
        # changes only the scheduler slot map; state/KV pointers remain attached
        # to the request session after dirty state and slot-bound graphs retire.
        for move in move_tuple:
            self._row(move.request_id)

    def reclaim(self, completed: CompletedRequest) -> None:
        request_id = int(completed.request_id)
        row = self._rows.get(request_id)
        if row is None:
            return
        with hip_target_arch_environment(self.generator.target_arch):
            adapter = self._mtp2_adapter
            if adapter is not None:
                adapter.release_request(request_id)
            if row.native_greedy or row.native_sampled:
                self._flush_row_owner(row)
                output = self._native_output(row, completed)
            else:
                output = row.fallback_output or self._empty_output(row, completed)
            self._outputs[request_id] = output
            metadata = self._execution_metadata(row)
            self._completed_metadata[request_id] = metadata
            self._recent_completed_routes.append(
                {"request_id": request_id, **copy.deepcopy(metadata)}
            )
            self._release_row_resources(row, retain_prefix_snapshots=True)
            self._rows.pop(request_id, None)

    def has_outputs(self, request_ids: Sequence[int]) -> bool:
        return all(int(request_id) in self._outputs for request_id in request_ids)

    def missing_outputs(self, request_ids: Sequence[int]) -> list[int]:
        return [int(request_id) for request_id in request_ids if int(request_id) not in self._outputs]

    def take_outputs(self, request_ids: Sequence[int]) -> list[GenerationOutput]:
        return [self._outputs.pop(int(request_id)) for request_id in request_ids]

    def discard(self, request_ids: Sequence[int]) -> None:
        for request_id in request_ids:
            rid = int(request_id)
            row = self._rows.pop(rid, None)
            if row is not None:
                adapter = self._mtp2_adapter
                if adapter is not None:
                    adapter.release_request(rid)
                self._release_row_resources(row)
            self._outputs.pop(rid, None)
            self._completed_metadata.pop(rid, None)

    def finalize_batch(
        self,
        request: GenerationRequest,
        request_ids: Sequence[int],
        outputs: Sequence[GenerationOutput],
    ) -> None:
        ids = tuple(int(request_id) for request_id in request_ids)
        output_tuple = tuple(outputs)
        prompt_rows = {
            index: _encode_prompt(self.generator.tokenizer, prompt)
            for index, prompt in enumerate(request.prompts)
        }
        generated_rows = {
            index: list(output.generated_token_ids or ())
            for index, output in enumerate(output_tuple)
        }
        metadata = [self._completed_metadata.pop(request_id, {}) for request_id in ids]
        native_steps = max((int(item.get("native_decode_steps", 0)) for item in metadata), default=0)
        native_c1_steps = max(
            (int(item.get("native_c1_decode_steps", 0)) for item in metadata),
            default=0,
        )
        native_prefill = bool(metadata) and all(bool(item.get("native_compact_prefill", False)) for item in metadata)
        all_native_model = bool(metadata) and all(
            bool(item.get("native_greedy", False) or item.get("native_sampled", False))
            for item in metadata
        )
        any_native_sampled = any(
            bool(item.get("native_sampled", False)) for item in metadata
        )
        any_native_sampler = any(
            bool(item.get("native_sampler", False)) for item in metadata
        )
        serial_fallback = any(bool(item.get("serial_decode_fallback", False)) for item in metadata)
        self.generator.last_generation_outputs = output_tuple
        self.generator.last_batch_generation = _gguf_last_batch_generation(
            self.generator.tokenizer,
            request,
            _gguf_sampler_plan(
                request,
                native_gpu_available=any_native_sampler,
            ),
            prompt_rows,
            generated_rows,
            {index: list(output.token_logprobs) for index, output in enumerate(output_tuple)},
            outputs=output_tuple,
            execution_path=(
                (
                    (
                        "gguf_packed_ar_native_sampler_decode"
                        if any_native_sampler
                        else "gguf_packed_ar_host_sampler_decode"
                    )
                    if any_native_sampled
                    else "gguf_packed_ar_server_decode"
                )
                if all_native_model
                else "gguf_resident_model_loop"
            ),
            native_compact_prefill=native_prefill,
            native_decode_steps=native_steps,
            native_c1_decode_steps=native_c1_steps,
            native_caware_decode=native_steps > 0,
            serial_decode_fallback=serial_fallback,
            native_sampler_rows=any_native_sampler,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        with hip_target_arch_environment(self.generator.target_arch):
            try:
                self._flush_all_packed_owners()
                if self._mtp2_adapter is not None:
                    self._mtp2_adapter.close()
                for row in tuple(self._rows.values()):
                    self._release_row_resources(row)
                self._rows.clear()
                self._outputs.clear()
                self._completed_metadata.clear()
                self._clear_prefix_snapshots()
                if self._kv_pool is not None:
                    self._teardown_kv_pool(release_workspace_state=False)
                self._release_available_sessions()
            except BaseException as exc:  # pragma: no cover - defensive cleanup
                error = exc
            generator_close = getattr(self.generator, "close", None)
            if callable(generator_close):
                try:
                    generator_close()
                except BaseException as exc:  # pragma: no cover - defensive cleanup
                    if error is None:
                        error = exc
        if error is not None:
            raise error

    def _reserve_sessions(self) -> None:
        if not bool(
            getattr(self.generator, "_defer_resident_session_policy_resolution", False)
        ):
            self._reserve_legacy_test_sessions()
            return
        acquired: list[_GGUFResidentSessionLease] = []
        batch_owner: Qwen35GGUFResidentSession | None = None
        try:
            batch_owner, pool_key, _reused = self.generator._acquire_shared_session(
                self._shared_runner,
                pool_name="continuous_ar_dynamic_kv",
                use_wmma_prefill=_resident_session_wmma_prefill_default(),
                use_gemv_decode=True,
                defer_kv_allocation=True,
                max_batch_size=self.capacity,
            )
            batch_owner._reset_current_slot_only = True
            acquired.append(_GGUFResidentSessionLease(batch_owner, pool_key))
            slot_view = getattr(batch_owner, "resident_slot_view", None)
            if self.capacity > 1 and not callable(slot_view):
                raise RuntimeError("GGUF resident batch owner has no slot-view ABI")
            for slot_index in range(1, self.capacity):
                assert callable(slot_view)
                acquired.append(
                    _GGUFResidentSessionLease(slot_view(slot_index), pool_key)
                )
            sessions = tuple(lease.session for lease in acquired)
            validate_layout = getattr(
                batch_owner,
                "_resident_ar_kv_layout_for_sessions",
                None,
            )
            if callable(validate_layout):
                validate_layout(sessions)
        except Exception:
            if batch_owner is not None:
                batch_owner.close()
            raise
        self._resident_batch_owner = batch_owner
        self._resident_batch_owner_pool_key = pool_key
        self._available.extend(acquired)

    def _reserve_legacy_test_sessions(self) -> None:
        acquired: list[_GGUFResidentSessionLease] = []
        try:
            for _ in range(self.capacity):
                session, pool_key, _reused = self.generator._acquire_shared_session(
                    self._shared_runner,
                    pool_name="continuous_ar_dynamic_kv",
                    use_wmma_prefill=_resident_session_wmma_prefill_default(),
                    use_gemv_decode=True,
                    defer_kv_allocation=True,
                )
                acquired.append(_GGUFResidentSessionLease(session, pool_key))
        except Exception:
            for lease in reversed(acquired):
                lease.session.close()
            raise
        self._available.extend(acquired)

    def _release_available_sessions(self) -> None:
        owner = self._resident_batch_owner
        if owner is None:
            while self._available:
                lease = self._available.pop()
                self.generator._release_shared_session(lease.pool_key, lease.session)
            return
        self._available.clear()
        self._resident_batch_owner = None
        self._resident_batch_owner_pool_key = None
        owner.close()

    def _release_row_resources(
        self,
        row: _GGUFResidentLoopRow,
        *,
        retain_prefix_snapshots: bool = False,
    ) -> None:
        prefix_cache = getattr(self, "_prefix_cache", None)
        # A row cancelled or reclaimed mid-prefill owns suspended-state buffers
        # (P6b); they must not outlive the row. release() is idempotent.
        suspended = row.resumable_prefill
        if suspended is not None and suspended is not _RESUMABLE_PREFILL_DONE:
            scratch = getattr(suspended, "scratch", None)
            if scratch is not None:
                scratch.release()
                suspended.scratch = None
        row.resumable_prefill = None
        if retain_prefix_snapshots:
            self._promote_prefix_snapshots(row)
        else:
            self._drop_prefix_snapshots_for_row(row.request_id)
        if prefix_cache is not None:
            prefix_cache.cancel(row.request_id)
        lease = row.lease
        if lease is None:
            if row.kv_allocation is not None:
                raise RuntimeError("GGUF row retained KV without a session lease")
            return
        session = lease.session
        self._close_c1_decode_graph(row)
        graph_handles = tuple(
            handle
            for handle in self._graph_handles_for_sessions((session,))
            if not bool(getattr(handle, "closed", False))
        )
        self._observe_graph_handles((session,))
        invalidate = getattr(session, "invalidate_device_kv_graphs", None)
        if callable(invalidate):
            invalidated = int(invalidate())
            self._record_graph_invalidations(graph_handles, invalidated)
            self._kv_graph_invalidation_count += invalidated
        # Retain the owner-shared packed workspace across reclaim. The slab is
        # union-geometry and shared by all resident views; freeing it here
        # forces a same-size hot-path reallocation on the next packed step
        # (canonical C2-6 packet: 246 releases / 242.39 GiB cumulative churn),
        # violating the CONCURRENCY2 workspace-reuse / no-hot-path-allocation
        # invariants. Release remains a close-path operation via
        # session.close() / release_idle_packed_workspace().
        reset = getattr(session, "reset", None)
        if callable(reset):
            reset()
        if row.kv_allocation is not None:
            pool = self._kv_pool
            if pool is None:
                raise RuntimeError("GGUF row has dynamic KV but the pool is unavailable")
            detached = session.unbind_device_kv_allocation()
            if detached is not row.kv_allocation:
                raise RuntimeError("GGUF session detached a different KV allocation")
            released = pool.release(row.request_id, now_seconds=time.monotonic())
            if released is not row.kv_allocation:
                raise RuntimeError("GGUF pool released a different request allocation")
            row.kv_allocation = None
        self._available.append(lease)
        row.lease = None
        self._sample_kv_hip_memory()

    def _current_hip_used_bytes(self) -> int:
        runtime = getattr(self._shared_runner, "runtime", None)
        if runtime is None:
            return 0
        try:
            free_bytes, total_bytes = runtime.mem_get_info()
        except Exception:
            return 0
        return max(0, int(total_bytes) - int(free_bytes))

    def _sample_kv_hip_memory(self) -> None:
        self._kv_hip_used_peak_sampled_bytes = max(
            int(self._kv_hip_used_peak_sampled_bytes),
            self._current_hip_used_bytes(),
        )

    @property
    def _session(self) -> Any | None:
        """Representative resident session, for server reporting lookups.

        The server resolves resident sessions through the generator
        (``api._resident_session_for_engine``) to report the effective context,
        the KVCache summary, and /ready memory samples. The batch owner is the
        canonical session: every slot view shares its declared context.
        """

        owner = self._resident_batch_owner
        if owner is not None:
            return owner
        if self._available:
            return self._available[0].session
        for row in self._rows.values():
            session = getattr(getattr(row, "lease", None), "session", None)
            if session is not None:
                return session
        return None

    def _acquire_lease(self) -> _GGUFResidentSessionLease:
        if not self._available:
            raise RuntimeError("GGUF resident model runner has no free session")
        return self._available.pop()

    def _packed_execution_owner(
        self,
        fallback: Qwen35GGUFResidentSession,
    ) -> Qwen35GGUFResidentSession:
        return getattr(self, "_resident_batch_owner", None) or fallback

    def _row(self, request_id: int) -> _GGUFResidentLoopRow:
        rid = int(request_id)
        if rid not in self._rows:
            raise KeyError(f"request_id {rid} is not registered with the GGUF resident runner")
        return self._rows[rid]

    def prefill_activation_ready(self, request_id: int) -> bool:
        """Whether this row's next prefill chunk may run now.

        A speculative row whose prompt has not started prefill needs the MTP
        prompt-activation claim before its first chunk, because the sink
        captures hidden states from prompt position zero. While another row
        holds that claim (a chunked prefill holds it across ticks), running
        this row's first chunk would permanently forfeit its draft provider.
        The scheduler defers the row instead, so a concurrent arrival keeps its
        MTP eligibility at the cost of a bounded prefill start delay.
        """

        adapter = self._resolved_mtp2_adapter()
        if adapter is None:
            return True
        probe = getattr(adapter, "prompt_activation_available", None)
        if not callable(probe) or bool(probe()):
            return True
        row = self._row(int(request_id))
        if row.prefill_tokens_seen > 0 or row.mtp2_candidate_budget <= 0:
            return True
        # A prefix-reused row that has no usable checkpoint never primes a
        # provider (K0), so it can start whenever the scheduler picks it. One
        # that does prime streams its suffix like any other row and needs the
        # claim for the same reason: its first chunk would otherwise forfeit a
        # provider that is about to be restored.
        if not row.prefix_reused_tokens:
            return False
        return int(row.request_id) not in self._restorable_prefix_rows((row,))

    def prefill_needs_activation(self, request_id: int) -> bool:
        """Whether this row's next chunk would open the prompt-activation claim.

        Only a row whose prompt has not started prefill needs the claim; a row
        already carrying a sink is mid-activation and a row without speculative
        intent never opens one. The scheduler uses this to admit at most one
        claim-opening row per multi-row prefill item, because the serial prefill
        path activates each row immediately before its own first chunk and a
        second claim-opening row in the same item would forfeit its provider.
        """

        row = self._row(int(request_id))
        if row.mtp2_candidate_budget <= 0:
            return False
        if row.prefix_reused_tokens:
            # A restorable hit row opens a claim like any streaming row; a
            # refused one never opens a sink at all.
            return bool(
                int(row.request_id) in self._restorable_prefix_rows((row,))
                and row.prefill_tokens_seen == 0
                and row.slot is None
            )
        return bool(row.prefill_tokens_seen == 0 and row.slot is None)

    def _restorable_prefix_rows(
        self,
        rows: Sequence[_GGUFResidentLoopRow],
    ) -> set[int]:
        """Request ids whose reused prefix has a usable checkpoint.

        Empty when no adapter is resolved, so a row that cannot be restored
        keeps the pre-checkpoint behavior of never priming a provider.
        """

        if not hasattr(self, "_mtp2_adapter"):
            # A runner double that never initialized adapter state has none.
            return set()
        adapter = self._resolved_mtp2_adapter()
        if adapter is None:
            return set()
        probe = getattr(adapter, "restorable_prefix_rows", None)
        if not callable(probe):
            return set()
        return set(probe(rows))

    def _begin_mtp2_prompt_streaming(
        self,
        rows: Sequence[_GGUFResidentLoopRow],
    ) -> tuple[Any | None, ...]:
        resolver = getattr(self, "_resolved_mtp2_adapter", None)
        adapter = resolver() if callable(resolver) else None
        restorable: set[int] = set()
        # A runner double without adapter state has no restorable rows.
        probe = getattr(self, "_restorable_prefix_rows", None)
        if adapter is not None and callable(probe):
            restorable = set(probe(rows))
        selected = tuple(
            row
            for row in rows
            if row.mtp2_candidate_budget > 0
            and (
                not row.prefix_reused_tokens
                or int(row.request_id) in restorable
            )
        )
        # A reused prefix is never prefilled, so the prompt hidden rows a draft
        # provider is primed from cannot exist for such a row, and it is filtered
        # out above before any adapter gate sees it. Record that refusal here --
        # the adapter's own reuse gate is never reached for these rows -- because
        # a row refused without a reason is indistinguishable from a row whose
        # provider was merely still priming, and both used to publish only the
        # planner's ``no_provider``.
        for row in rows:
            if (
                row.mtp2_candidate_budget > 0
                and int(row.prefix_reused_tokens) > 0
                and int(row.request_id) not in restorable
                and row.mtp2_prompt_fallback_reason is None
            ):
                row.mtp2_prompt_fallback_reason = "prefix_reuse_k0"
        if not _gguf_specdec2_streaming_prompt_enabled():
            for row in selected:
                row.mtp2_candidate_budget = 0
                row.mtp2_prompt_fallback_reason = "operator_disabled_streaming_prompt_k0"
            return (None,) * len(tuple(rows))
        if not selected:
            return (None,) * len(tuple(rows))
        if adapter is None:
            return (None,) * len(tuple(rows))
        checkpoints = {
            int(row.request_id): (
                lambda row=row: raise_if_generation_deadline_expired(row.request)
            )
            for row in selected
        }
        sinks = adapter.begin_prompt_streaming(
            tuple(row.request_id for row in selected),
            checkpoints=checkpoints,
        )
        if sinks is None:
            return (None,) * len(tuple(rows))
        by_id = {
            int(row.request_id): sink
            for row, sink in zip(selected, sinks, strict=True)
        }
        return tuple(by_id.get(int(row.request_id)) for row in rows)

    def _finish_mtp2_prompt_streaming(
        self,
        rows: Sequence[_GGUFResidentLoopRow],
        sinks: Sequence[Any | None],
        *,
        success: bool,
        stream: int = 0,
    ) -> None:
        ids = tuple(
            int(row.request_id)
            for row, sink in zip(rows, sinks, strict=True)
            if sink is not None
        )
        if not ids:
            return
        adapter = self._resolved_mtp2_adapter()
        if adapter is None:
            raise RuntimeError("GGUF MTP2 streaming prompt adapter disappeared")
        adapter.finish_prompt_streaming(ids, success=bool(success), stream=int(stream))

    def _prefill_native_row(self, row: _GGUFResidentLoopRow) -> None:
        if row.slot is not None:
            return
        lease = row.lease or self._acquire_lease()
        row.lease = lease
        start = time.perf_counter()
        native_compact_prefill = False
        # Direct no-mirror INT8 uses one block-table-aware single-row prefill
        # route at every physical base. Keeping base-zero c1 on scalar bulk
        # prefill keeps the c1 control on the layer-outer executor (the
        # historical "different GDN state-capture arithmetic" divergence was
        # withdrawn 2026-09-10; the entries agree exactly with per-layer
        # oracles, but mixing executors across bases still compares two
        # schedules at c>N).
        packed_owner = self._packed_execution_owner(lease.session)
        if (
            getattr(self, "_resident_batch_owner", None) is None
            and not _gguf_single_row_block_table_prefill_required(lease.session)
            and row.mtp2_candidate_budget <= 0
        ):
            result = lease.session.prefill(row.prompt_ids, return_logits=False)
        else:
            prefill_batch = getattr(
                packed_owner,
                "prefill_batch_native",
                None,
            )
            if not callable(prefill_batch):
                raise RuntimeError(
                    "GGUF KV route requires block-table-aware single-row prefill"
                )
            streaming_sinks = self._begin_mtp2_prompt_streaming((row,))
            streaming = streaming_sinks[0] is not None
            mtp2_adapter = (
                self._resolved_mtp2_adapter()
                if row.mtp2_candidate_budget > 0
                else None
            )
            streaming_kwargs = (
                {
                    "target_hidden_chunk_sinks": streaming_sinks,
                    "target_hidden_request_ids": (row.request_id,),
                    "target_hidden_chunk_starts": (0,),
                }
                if streaming
                else {}
            )
            packed_prefill_declined = False
            try:
                with _temporary_env({"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}):
                    results = prefill_batch(
                        [row.prompt_ids],
                        sessions=[lease.session],
                        full_prompt_lengths=[len(row.prompt_ids)],
                        return_logits=False,
                        return_hidden_seeds=bool(
                            row.mtp2_candidate_budget > 0
                            and (
                                not streaming
                                or bool(
                                    getattr(
                                        mtp2_adapter,
                                        "requires_prefill_hidden_seeds",
                                        False,
                                    )
                                )
                            )
                        ),
                        **streaming_kwargs,
                    )
            except NotImplementedError:
                # A packed-route refusal is a shape limit, not a request
                # failure. The row may already own committed prompt chunks (a
                # declined incremental prefill resets the session and replays
                # the whole prompt here), so ending the response with an
                # unsupported_parameter error is both late and wrong: finish
                # the prompt on the registered strict per-session route and
                # keep decoding autoregressively. Sessions whose KV layout only
                # the block-table-aware route can represent still fail closed.
                self._finish_mtp2_prompt_streaming(
                    (row,),
                    streaming_sinks,
                    success=False,
                )
                if _gguf_single_row_block_table_prefill_required(lease.session):
                    raise
                row.mtp2_candidate_budget = 0
                row.mtp2_prompt_fallback_reason = "packed_prefill_unsupported_k0"
                self._fallback_reasons["packed_prefill_unsupported_k0"] += 1
                row.prefill_ms = 0.0
                row.prefill_chunk_count = 0
                result = lease.session.prefill(row.prompt_ids, return_logits=False)
                results = [result]
                packed_prefill_declined = True
            except Exception:
                self._finish_mtp2_prompt_streaming(
                    (row,),
                    streaming_sinks,
                    success=False,
                )
                raise
            self._finish_mtp2_prompt_streaming(
                (row,),
                streaming_sinks,
                success=True,
            )
            result_list = [] if results is None else list(results)
            if len(result_list) != 1:
                raise RuntimeError(
                    "shifted dynamic GGUF prefill did not return exactly one result"
                )
            result = result_list[0]
            native_compact_prefill = not packed_prefill_declined
        self._route_counts["native_full_prefill_rows"] += 1
        row.prefill_ms += _timing_ms_since(start)
        row.prefill_chunk_count += 1
        self._refresh_prefix_cache(row)
        self._finish_native_prefill(
            row,
            result,
            native_compact_prefill=native_compact_prefill,
        )

    def _prepare_sampled_prefill(
        self,
        row: _GGUFResidentLoopRow,
    ) -> tuple[GenerationRequest, RowSamplingState]:
        if row.sampling_request is None and row.sampling_state is None:
            sampling_request = _request_with_tokenizer_eos(
                row.request,
                self.generator.tokenizer,
            )
            sampling_state = _gguf_row_sampling_state(
                sampling_request,
                list(row.prompt_ids),
                row_index=row.row_index,
            )
            row.sampling_request = sampling_request
            row.sampler_plan = _gguf_sampler_plan(
                sampling_request,
                native_gpu_available=_native_gpu_sampler_requested(),
            )
            row.native_sampler = _gguf_native_sampler_plan_enabled(
                sampling_request,
                row.sampler_plan,
            )
            row.sampling_state = sampling_state
            return sampling_request, sampling_state
        if row.sampling_request is None or row.sampling_state is None:
            raise RuntimeError("GGUF sampled prefill has partial sampling state")
        return row.sampling_request, row.sampling_state

    def _finish_sampled_prefill(
        self,
        row: _GGUFResidentLoopRow,
        result: Any,
        *,
        native_compact_prefill: bool,
        native_sample: Any | None = None,
    ) -> None:
        if row.slot is not None:
            raise RuntimeError("GGUF sampled row was prefilled more than once")
        lease = row.lease
        if lease is None:
            raise RuntimeError("GGUF sampled prefill finished without a session lease")
        sampling_request, sampling_state = self._prepare_sampled_prefill(row)
        if native_sample is None:
            if row.native_sampler:
                raise RuntimeError("native GGUF prefill did not return a native sample")
            sample = _select_from_gguf_logits(
                result,
                sampling_request,
                sampling_state,
                self.generator.tokenizer,
            )
            full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(
                result
            )
        else:
            if not row.native_sampler:
                raise RuntimeError("host GGUF prefill received a native sample")
            sample = native_sample
            full_vocab_logits_d2h, logits_d2h_bytes = False, 0
        token = int(sample.token_id)
        _gguf_queue_json_object_close_if_needed(
            sampling_state,
            self.generator.tokenizer,
            _gguf_token_text(self.generator.tokenizer, sample),
            remaining_tokens=max(0, int(sampling_request.max_tokens) - 1),
        )
        self._route_counts["native_sampled_prefill_rows"] += 1
        plan = row.sampler_plan
        if plan is None:
            raise RuntimeError("GGUF sampled prefill has no sampler plan")
        if row.native_sampler:
            self._route_counts["native_sampler_requests"] += 1
        else:
            self._route_counts["host_sampler_requests"] += 1
            self._fallback_reasons[
                str(plan.fallback_reason or plan.mode.value)
            ] += 1
        row.samples.append(sample)
        row.full_vocab_logits_d2h = full_vocab_logits_d2h
        row.logits_d2h_bytes = logits_d2h_bytes
        self._refresh_prefix_cache(row)
        row.slot = _GGUFARServingSlot(
            request_id=row.request_id,
            prompt_ids=list(row.prompt_ids),
            session=lease.session,
            prev_token=token,
            seq_position=int(lease.session.position),
            generated_ids=[token],
            timing={
                "tokenize_ms": float(row.tokenize_ms),
                "prompt_encode_ms": float(row.prompt_encode_ms),
                "render_ms": float(row.render_ms),
                "admission_prepare_ms": float(row.admission_prepare_ms),
                "prefill_ms": float(row.prefill_ms),
                "prefill_chunk_count": float(row.prefill_chunk_count),
                "request_total_ms": _timing_ms_since(row.submitted_at),
            },
            session_pool_key=lease.pool_key,
            done=(
                int(sampling_request.max_tokens) <= 1
                or _gguf_finished(
                    (token,),
                    self.generator.tokenizer,
                    sampling_request,
                )
            ),
            native_compact_prefill=bool(native_compact_prefill),
        )
        self._observe_mtp2_prefill(row, result)

    def _prefill_sampled_row(self, row: _GGUFResidentLoopRow) -> None:
        if row.slot is not None:
            return
        lease = row.lease or self._acquire_lease()
        row.lease = lease
        sampling_request, sampling_state = self._prepare_sampled_prefill(row)
        start = time.perf_counter()
        native_compact_prefill = False
        packed_owner = self._packed_execution_owner(lease.session)
        # A sampled row needs the prompt's first-token logits for its host
        # sampler, and a packed prefill refuses to return logits and host hidden
        # rows together, so a row that owes a draft provider takes the streamed
        # target-hidden sink the greedy path uses: the sink carries the rows and
        # the call still returns logits. Without a sink the row keeps the shipped
        # scalar prefill and decodes without a provider.
        streaming_sinks = (
            self._begin_mtp2_prompt_streaming((row,))
            if row.mtp2_candidate_budget > 0
            else (None,)
        )
        streaming = streaming_sinks[0] is not None
        if (
            getattr(self, "_resident_batch_owner", None) is None
            and not _gguf_single_row_block_table_prefill_required(lease.session)
            and not streaming
        ):
            result = lease.session.prefill(
                row.prompt_ids,
                return_logits=not row.native_sampler,
            )
        else:
            prefill_batch = getattr(
                packed_owner,
                "prefill_batch_native",
                None,
            )
            if not callable(prefill_batch):
                raise RuntimeError(
                    "sampled GGUF KV requires block-table-aware single-row prefill"
                )
            native_logits_kwargs = (
                {"require_logits": True} if row.native_sampler else {}
            )
            streaming_kwargs = (
                {
                    "target_hidden_chunk_sinks": streaming_sinks,
                    "target_hidden_request_ids": (row.request_id,),
                    "target_hidden_chunk_starts": (0,),
                }
                if streaming
                else {}
            )
            try:
                with _temporary_env(
                    {"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}
                ):
                    results = prefill_batch(
                        [row.prompt_ids],
                        sessions=[lease.session],
                        full_prompt_lengths=[len(row.prompt_ids)],
                        return_logits=not row.native_sampler,
                        return_hidden_seeds=False,
                        **streaming_kwargs,
                        **native_logits_kwargs,
                    )
            except Exception:
                self._finish_mtp2_prompt_streaming(
                    (row,),
                    streaming_sinks,
                    success=False,
                )
                raise
            self._finish_mtp2_prompt_streaming(
                (row,),
                streaming_sinks,
                success=True,
            )
            result_list = [] if results is None else list(results)
            if len(result_list) != 1:
                raise RuntimeError(
                    "shifted sampled GGUF prefill did not return exactly one result"
                )
            result = result_list[0]
            native_compact_prefill = True
        row.prefill_ms += _timing_ms_since(start)
        row.prefill_chunk_count += 1
        native_sample = None
        if row.native_sampler:
            if native_compact_prefill:
                # Packed prefill leaves logits on its execution owner, while
                # the selected token still belongs to the request's slot.
                sample_native = getattr(
                    packed_owner,
                    "sample_native_from_packed_logits",
                    None,
                )
                if not callable(sample_native):
                    raise RuntimeError(
                        "GGUF packed prefill has no native sampler integration"
                    )
                native_sample = sample_native(
                    0,
                    sampling_request,
                    sampling_state,
                    output_session=lease.session,
                )
            else:
                sample_native = getattr(
                    lease.session,
                    "sample_native_from_last_logits",
                    None,
                )
                if not callable(sample_native):
                    raise RuntimeError(
                        "GGUF session has no native sampler integration"
                    )
                native_sample = sample_native(sampling_request, sampling_state)
        if native_sample is not None:
            self._route_counts["native_sampler_row_launches"] += 1
        self._finish_sampled_prefill(
            row,
            result,
            native_compact_prefill=native_compact_prefill,
            native_sample=native_sample,
        )

    def _prefill_processed_argmax_chunk(
        self,
        row: _GGUFResidentLoopRow,
        chunk: tuple[int, ...],
        *,
        final_chunk: bool,
    ) -> None:
        if row.slot is not None:
            raise RuntimeError("GGUF processed-argmax row was prefilled more than once")
        if not chunk:
            raise RuntimeError("GGUF processed-argmax prefill chunk must be non-empty")
        sampling_request, _ = self._prepare_sampled_prefill(row)
        plan = row.sampler_plan
        if plan is None or plan.mode is not SamplingMode.PROCESSED_ARGMAX:
            raise RuntimeError(
                "GGUF prefix reuse only supports deterministic processed-argmax sampling"
            )
        lease = row.lease or self._acquire_lease()
        row.lease = lease
        session = lease.session
        result = None
        native_compact_prefill = False
        final_prefix_boundary = (len(row.prompt_ids) // 256) * 256

        if row.prefix_reused_tokens:
            # A reused prefix is never prefilled, so this row's prompt activation
            # cannot produce the full-prompt hidden rows a provider needs. Run
            # the same admission the miss path runs, for its recorded reason:
            # it refuses a reused-prefix row (``prefix_reuse_k0``, or whichever
            # gate fires first) and would otherwise leave the row with no
            # published reason at all, so the served response reported the
            # planner's ``no_provider`` beside a ``not_k0`` route claim and no
            # reader could tell the row was refused at admission.
            self._begin_mtp2_prompt_streaming((row,))
            start = time.perf_counter()
            prefill_batch = (
                getattr(
                    self._packed_execution_owner(session),
                    "prefill_batch_native",
                    None,
                )
                if (
                    _gguf_prefix_batched_suffix_enabled()
                    and _gguf_prefix_batched_suffix_chunk_eligible(chunk)
                )
                else None
            )
            if callable(prefill_batch):
                segments = _gguf_prefix_suffix_segments(
                    int(getattr(session, "position", 0)),
                    len(row.prompt_ids),
                    chunk,
                )
                native_compact_prefill = True
                for segment_index, segment in enumerate(segments):
                    last_segment = segment_index == len(segments) - 1
                    want_output = bool(final_chunk and last_segment)
                    with _temporary_env(
                        {"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}
                    ):
                        results = prefill_batch(
                            [segment],
                            sessions=[session],
                            full_prompt_lengths=[len(row.prompt_ids)],
                            return_logits=want_output,
                            return_hidden_seeds=False,
                            sample_output=want_output,
                        )
                    if not last_segment:
                        # The segment ended exactly on the deepest prompt-
                        # aligned boundary; capture it before continuing.
                        self._refresh_prefix_cache(row)
                    elif final_chunk:
                        result_list = [] if results is None else list(results)
                        if len(result_list) != 1 or result_list[0] is None:
                            raise RuntimeError(
                                "GGUF batched reused-suffix prefill returned no result"
                            )
                        result = result_list[0]
                self._route_counts["prefix_batched_suffix_prefill_chunks"] += 1
                self._route_counts["prefix_batched_suffix_prefill_tokens"] += len(chunk)
                self._route_counts["processed_argmax_prefix_batched_suffix_chunks"] += 1
                self._route_counts["processed_argmax_prefix_batched_suffix_tokens"] += len(chunk)
                row.prefill_ms += _timing_ms_since(start)
                self._prefix_phase_add("suffix_prefill", start)
                row.prefill_chunk_count += 1
                self._refresh_prefix_cache_at_prompt_boundary(row, lease)
            else:
                for index, token_id in enumerate(chunk):
                    result = session.step(
                        int(token_id),
                        return_logits=bool(final_chunk and index == len(chunk) - 1),
                    )
                    if int(session.position) == final_prefix_boundary:
                        self._refresh_prefix_cache(row)
                self._route_counts["prefix_c1_suffix_prefill_chunks"] += 1
                self._route_counts["prefix_c1_suffix_prefill_tokens"] += len(chunk)
                self._route_counts["processed_argmax_prefix_c1_suffix_chunks"] += 1
                self._route_counts["processed_argmax_prefix_c1_suffix_tokens"] += len(chunk)
                self._fallback_reasons["prefix_batched_suffix_unavailable"] += 1
                row.prefill_ms += _timing_ms_since(start)
                self._prefix_phase_add("suffix_prefill", start)
                row.prefill_chunk_count += 1
        else:
            if not final_chunk or chunk != row.prompt_ids:
                raise RuntimeError(
                    "processed-argmax private radix prefill requires the complete prompt"
                )
            aligned_prompt = chunk[:final_prefix_boundary]
            tail = chunk[final_prefix_boundary:]
            if not aligned_prompt:
                raise RuntimeError(
                    "processed-argmax radix prefill requires an aligned prompt boundary"
                )
            operation_start = time.perf_counter()
            if not _gguf_single_row_block_table_prefill_required(session):
                result = session.prefill(
                    aligned_prompt,
                    return_logits=not tail,
                )
            else:
                prefill_batch = getattr(session, "prefill_batch_native", None)
                if not callable(prefill_batch):
                    raise RuntimeError(
                        "processed-argmax radix prefill requires block-table-aware prefill"
                    )
                sample_kwargs = {} if not tail else {"sample_output": False}
                try:
                    with _temporary_env(
                        {"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}
                    ):
                        results = prefill_batch(
                            [aligned_prompt],
                            sessions=[session],
                            full_prompt_lengths=[len(row.prompt_ids)],
                            return_logits=not tail,
                            return_hidden_seeds=False,
                            **sample_kwargs,
                        )
                except NotImplementedError as exc:
                    raise RuntimeError(
                        "processed-argmax radix prefill does not support private prefill"
                    ) from exc
                result_list = [] if results is None else list(results)
                if len(result_list) != 1:
                    raise RuntimeError(
                        "processed-argmax private prefill did not return exactly one result"
                    )
                result = result_list[0]
                native_compact_prefill = True
            if int(getattr(session, "position", -1)) != final_prefix_boundary:
                raise RuntimeError(
                    "processed-argmax private prefill advanced to the wrong boundary"
                )
            row.prefill_ms += _timing_ms_since(operation_start)
            row.prefill_chunk_count += 1
            self._route_counts[
                "processed_argmax_private_aligned_prefill_rows"
            ] += 1
            self._refresh_prefix_cache(row)
            if tail:
                tail_start = time.perf_counter()
                for index, token_id in enumerate(tail):
                    result = session.step(
                        int(token_id),
                        return_logits=index == len(tail) - 1,
                    )
                self._route_counts[
                    "processed_argmax_private_c1_tail_chunks"
                ] += 1
                self._route_counts[
                    "processed_argmax_private_c1_tail_tokens"
                ] += len(tail)
                row.prefill_ms += _timing_ms_since(tail_start)
                row.prefill_chunk_count += 1

        if final_chunk:
            if result is None or getattr(result, "logits", None) is None:
                raise RuntimeError(
                    "processed-argmax final prefill did not return full-vocabulary logits"
                )
            self._finish_sampled_prefill(
                row,
                result,
                native_compact_prefill=native_compact_prefill,
            )

    def _prefill_native_chunk(
        self,
        row: _GGUFResidentLoopRow,
        chunk: tuple[int, ...],
        *,
        final_chunk: bool,
    ) -> None:
        if row.resumable_prefill is _RESUMABLE_PREFILL_DONE:
            # The resumable layer-outer executor already produced this row's
            # first token; the scheduler's remaining chunks for this prompt are
            # bookkeeping only.
            return
        if row.slot is not None:
            raise RuntimeError("GGUF resident row was prefilled more than once")
        lease = row.lease or self._acquire_lease()
        row.lease = lease
        if row.prefix_reused_tokens:
            start = time.perf_counter()
            session = lease.session
            # A restorable hit row primes its provider from the checkpoint at
            # the boundary and then streams the suffix rows it is about to
            # prefill, which is the only source of provider state a reused
            # prefix can have: the reused tokens never run the target here.
            reused_prefix = int(row.prefix_reused_tokens)
            sink = None
            if row.mtp2_candidate_budget > 0:
                streaming_sinks = self._begin_mtp2_prompt_streaming((row,))
                sink = streaming_sinks[0] if streaming_sinks else None
            prefill_batch = (
                getattr(
                    self._packed_execution_owner(session),
                    "prefill_batch_native",
                    None,
                )
                if (
                    _gguf_prefix_batched_suffix_enabled()
                    and _gguf_prefix_batched_suffix_chunk_eligible(chunk)
                )
                else None
            )
            if callable(prefill_batch):
                chunk_abs_start = int(getattr(session, "position", 0))
                segments = _gguf_prefix_suffix_segments(
                    chunk_abs_start,
                    len(row.prompt_ids),
                    chunk,
                )
                result = None
                segment_offset = 0
                for segment_index, segment in enumerate(segments):
                    last_segment = segment_index == len(segments) - 1
                    # The sink carries the suffix, so its chunk start is the
                    # suffix-relative row index the engine reports as absolute.
                    streaming_kwargs = (
                        {
                            "target_hidden_chunk_sinks": (sink,),
                            "target_hidden_request_ids": (row.request_id,),
                            "target_hidden_chunk_starts": (
                                chunk_abs_start - reused_prefix + segment_offset,
                            ),
                            # A suffix is prefilled in chunks, so this call does
                            # not own the sink's whole timeline: the adapter
                            # closes it once the prompt is complete.
                            "finish_target_hidden_sinks": False,
                        }
                        if sink is not None
                        else {}
                    )
                    segment_offset += len(segment)
                    with _temporary_env(
                        {"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}
                    ):
                        results = prefill_batch(
                            [segment],
                            sessions=[session],
                            full_prompt_lengths=[len(row.prompt_ids)],
                            return_logits=False,
                            return_hidden_seeds=False,
                            sample_output=bool(final_chunk and last_segment),
                            **streaming_kwargs,
                        )
                    if not last_segment:
                        # The segment ended exactly on the deepest prompt-
                        # aligned boundary; capture it before continuing.
                        self._refresh_prefix_cache(row)
                    elif final_chunk:
                        result_list = [] if results is None else list(results)
                        if len(result_list) != 1 or result_list[0] is None:
                            raise RuntimeError(
                                "GGUF batched reused-suffix prefill returned no result"
                            )
                        result = result_list[0]
                self._route_counts["prefix_batched_suffix_prefill_chunks"] += 1
                self._route_counts["prefix_batched_suffix_prefill_tokens"] += len(chunk)
                row.prefill_ms += _timing_ms_since(start)
                self._prefix_phase_add("suffix_prefill", start)
                row.prefill_chunk_count += 1
                self._refresh_prefix_cache_at_prompt_boundary(row, lease)
                if final_chunk:
                    if sink is not None:
                        # Commit the streamed suffix: the provider is now at the
                        # prompt's end and the row owns it. A sink that is short
                        # is refused there and the row decodes autoregressively.
                        self._finish_mtp2_prompt_streaming(
                            (row,),
                            (sink,),
                            success=True,
                        )
                    self._finish_native_prefill(
                        row,
                        result,
                        native_compact_prefill=True,
                    )
                return
            result = None
            for token_id in chunk:
                result = session.step(int(token_id), return_logits=False)
            if result is None:
                raise RuntimeError("GGUF shared-prefix suffix chunk must be non-empty")
            self._route_counts["prefix_c1_suffix_prefill_chunks"] += 1
            self._route_counts["prefix_c1_suffix_prefill_tokens"] += len(chunk)
            self._fallback_reasons["prefix_batched_suffix_unavailable"] += 1
            row.prefill_ms += _timing_ms_since(start)
            self._prefix_phase_add("suffix_prefill", start)
            row.prefill_chunk_count += 1
            self._refresh_prefix_cache_at_prompt_boundary(row, lease)
            if final_chunk:
                self._finish_native_prefill(
                    row,
                    result,
                    native_compact_prefill=False,
                )
            return
        if getattr(lease.session, "kv_attention_source", None) == "int8_direct":
            if self._prefill_resumable_int8_chunk(
                row,
                chunk,
                final_chunk=final_chunk,
            ):
                return
            # Exact no-mirror prefill owns one bounded transient BF16 oracle.
            # Releasing it between scheduler chunks would lose prior BF16 K/V,
            # so IKV-C1 buffers scheduler work and executes the complete prompt
            # once through the shifted block-table-aware single-row route.
            # (P6 keeps this as the fail-closed path for shapes the resumable
            # layer-outer executor declines.)
            self._fallback_reasons["int8_direct_full_prompt_prefill"] += 1
            self._disable_incremental_prefill(row, final_chunk=final_chunk)
            return
        prefill_batch = getattr(
            self._packed_execution_owner(lease.session),
            "prefill_batch_native",
            None,
        )
        if not callable(prefill_batch):
            self._disable_incremental_prefill(row, final_chunk=final_chunk)
            return
        start = time.perf_counter()
        sample_kwargs = {} if final_chunk else {"sample_output": False}
        streaming_sinks = self._begin_mtp2_prompt_streaming((row,))
        streaming = streaming_sinks[0] is not None
        chunk_start = max(0, int(row.prefill_tokens_seen) - len(chunk))
        streaming_kwargs = (
            {
                "target_hidden_chunk_sinks": streaming_sinks,
                "target_hidden_request_ids": (row.request_id,),
                "target_hidden_chunk_starts": (chunk_start,),
                "finish_target_hidden_sinks": bool(final_chunk),
            }
            if streaming
            else {}
        )
        try:
            with _temporary_env({"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}):
                results = prefill_batch(
                    [chunk],
                    sessions=[lease.session],
                    full_prompt_lengths=[len(row.prompt_ids)],
                    return_logits=False,
                    return_hidden_seeds=False,
                    **streaming_kwargs,
                    **sample_kwargs,
                )
        except NotImplementedError:
            self._finish_mtp2_prompt_streaming(
                (row,),
                streaming_sinks,
                success=False,
            )
            row.mtp2_candidate_budget = 0
            row.mtp2_prompt_fallback_reason = "incremental_streaming_unsupported_k0"
            self._disable_incremental_prefill(row, final_chunk=final_chunk)
            return
        except Exception:
            self._finish_mtp2_prompt_streaming(
                (row,),
                streaming_sinks,
                success=False,
            )
            raise
        if final_chunk:
            self._finish_mtp2_prompt_streaming(
                (row,),
                streaming_sinks,
                success=True,
            )
        result_list = [] if results is None else list(results)
        if len(result_list) != 1:
            raise RuntimeError(
                f"GGUF incremental prefill returned {len(result_list)} result(s) for one row"
            )
        self._route_counts["native_incremental_prefill_chunks"] += 1
        if not final_chunk:
            self._route_counts["native_incremental_prefill_unsampled_chunks"] += 1
        row.prefill_ms += _timing_ms_since(start)
        self._prefix_phase_add("prefill_batched", start)
        row.prefill_chunk_count += 1
        self._refresh_prefix_cache_at_prompt_boundary(row, lease)
        if final_chunk:
            self._finish_native_prefill(
                row,
                result_list[0],
                native_compact_prefill=True,
            )

    def _prefill_resumable_int8_chunk(
        self,
        row: _GGUFResidentLoopRow,
        chunk: tuple[int, ...],
        *,
        final_chunk: bool,
    ) -> bool:
        """Advance one bounded layer segment of a compact-INT8 prefill (P6).

        Roadmap F5: the old route did no model work on any chunk but the last,
        so a long prompt blocked admission, cancellation, and interleaved
        decode for its whole duration. The layer-outer packed executor already
        makes every layer boundary a consistent state, so this runs at most
        ``budget`` layers per scheduler poll and hands control back with real
        GPU work done. Returns True when the resumable executor owned this
        chunk (including the segment that completes the prompt); False when the
        caller must use the fail-closed full-prompt path.
        """

        if row.prefix_reused_tokens:
            # Shared-prefix admission requires incremental prefill support that
            # the layer-outer executor does not provide.
            return False
        if not _gguf_packed_layer_outer_enabled():
            return False
        lease = row.lease
        if lease is None:
            return False
        owner = self._packed_execution_owner(lease.session)
        resume = getattr(
            owner,
            "prefill_batch_native_layer_outer_resumable",
            None,
        )
        if not callable(resume):
            return False
        layer_count = _gguf_resumable_layer_count(owner)
        if layer_count <= 0:
            return False
        state = row.resumable_prefill
        if state is None:
            remaining_layers = layer_count
            prompts: tuple[tuple[int, ...], ...] | None = (
                tuple(int(token) for token in row.prompt_ids),
            )
            sessions = (lease.session,)
        else:
            remaining_layers = max(0, layer_count - int(state.next_layer))
            prompts = None
            sessions = None
        chunk_len = max(1, len(chunk))
        # Polls still to come for this prompt, including this one. Future chunk
        # sizes are approximated by the current one; the final chunk always
        # finishes the remainder, so a misestimate only shifts work between
        # polls rather than dropping it.
        remaining_tokens = max(
            0,
            len(row.prompt_ids) - int(row.prefill_tokens_seen) + len(chunk),
        )
        remaining_polls = max(1, -(-remaining_tokens // chunk_len))
        if final_chunk:
            budget: int | None = None
        else:
            budget = max(1, -(-remaining_layers // remaining_polls))
        started = time.perf_counter()
        streaming_sinks = (
            self._begin_mtp2_prompt_streaming((row,))
            if state is None and int(getattr(row, "mtp2_candidate_budget", 0)) > 0
            else tuple(getattr(state, "target_hidden_chunk_sinks", ())) or (None,)
        )
        try:
            if state is None:
                streaming_kwargs = (
                    {"target_hidden_chunk_sinks": streaming_sinks}
                    if any(sink is not None for sink in streaming_sinks) else {}
                )
                result = resume(
                    prompts, sessions=sessions, layer_budget=budget, **streaming_kwargs,
                )
            else:
                result = resume(state=state, layer_budget=budget)
        except NotImplementedError:
            if any(sink is not None for sink in streaming_sinks):
                self._finish_mtp2_prompt_streaming((row,), streaming_sinks, success=False)
            if state is None:
                # The layer-outer executor declined this shape before doing any
                # device work; the caller falls back to the full-prompt route.
                self._fallback_reasons["resumable_int8_prefill_declined"] += 1
                return False
            raise
        except BaseException:
            if any(sink is not None for sink in streaming_sinks):
                self._finish_mtp2_prompt_streaming((row,), streaming_sinks, success=False)
            raise
        row.prefill_ms += _timing_ms_since(started)
        row.prefill_chunk_count += 1
        self._refresh_prefix_cache(row)
        if isinstance(result, _GGUFResumablePrefillState):
            row.resumable_prefill = result
            self._route_counts["resumable_int8_prefill_segments"] += 1
            return True
        result_list = [] if result is None else list(result)
        if len(result_list) != 1:
            raise RuntimeError(
                "resumable layer-outer prefill returned"
                f" {len(result_list)} result(s) for one row"
            )
        row.resumable_prefill = _RESUMABLE_PREFILL_DONE
        self._route_counts["resumable_int8_prefill_completions"] += 1
        if any(sink is not None for sink in streaming_sinks):
            self._finish_mtp2_prompt_streaming((row,), streaming_sinks, success=True)
        self._finish_native_prefill(
            row,
            result_list[0],
            native_compact_prefill=True,
        )
        return True

    def _disable_incremental_prefill(
        self,
        row: _GGUFResidentLoopRow,
        *,
        final_chunk: bool,
    ) -> None:
        if row.prefix_reused_tokens:
            raise RuntimeError("GGUF shared-prefix admission requires incremental prefill support")
        row.incremental_prefill = False
        self._fallback_reasons["incremental_prefill_unsupported"] += 1
        if row.lease is not None:
            row.lease.session.reset()
        row.prefill_chunk_count = 0
        row.prefill_ms = 0.0
        if final_chunk:
            self._prefill_native_row(row)

    def _finish_native_prefill(
        self,
        row: _GGUFResidentLoopRow,
        result: Any,
        *,
        native_compact_prefill: bool,
    ) -> None:
        lease = row.lease
        if lease is None:
            raise RuntimeError("GGUF resident prefill finished without a session lease")
        token = int(getattr(result, "token_id"))
        vocab_size = int(
            getattr(getattr(self, "_shared_runner", None), "vocab_size", 0) or 0
        )
        if token < 0 or (vocab_size > 0 and token >= vocab_size):
            session = lease.session
            raise RuntimeError(
                "GGUF prefill produced an invalid token: "
                f"request_id={row.request_id} token={token} vocab={vocab_size} "
                f"position={getattr(session, 'position', None)} "
                f"kv_attention_source={getattr(session, 'kv_attention_source', None)} "
                f"kv_base_row={_gguf_device_kv_contiguous_base_row(session)}"
            )
        timing = {
            "tokenize_ms": float(row.tokenize_ms),
            "prompt_encode_ms": float(row.prompt_encode_ms),
            "render_ms": float(row.render_ms),
            "admission_prepare_ms": float(row.admission_prepare_ms),
            "prefill_ms": float(row.prefill_ms),
            "prefill_chunk_count": float(row.prefill_chunk_count),
            "request_total_ms": _timing_ms_since(row.submitted_at),
        }
        row.slot = _GGUFARServingSlot(
            request_id=row.request_id,
            prompt_ids=list(row.prompt_ids),
            session=lease.session,
            prev_token=token,
            seq_position=int(lease.session.position),
            generated_ids=[token],
            timing=timing,
            session_pool_key=lease.pool_key,
            done=(
                int(row.request.max_tokens) <= 1
                or _gguf_finished((token,), self.generator.tokenizer, row.request)
            ),
            native_compact_prefill=bool(native_compact_prefill),
        )
        self._observe_mtp2_prefill(row, result)

    def _observe_mtp2_prefill(self, row: _GGUFResidentLoopRow, result: Any) -> None:
        """Hand a prefilled row's hidden rows to its MTP provider, if it has one.

        Both prefill finishes call this: the greedy path and the sampled path.
        A row that registered a candidate budget owns provider draft state, and
        the provider's capability needs the prompt's hidden rows before it can
        admit the row, so skipping this on the sampled path left a sampled row
        without a provider for its whole lifetime (planner reason
        ``no_provider`` on every cycle).
        """

        if row.mtp2_candidate_budget <= 0:
            return
        adapter = self._resolved_mtp2_adapter()
        if adapter is None:
            return
        adapter.observe_prefill_result(row.request_id, row.prompt_ids, result)

    def _run_resident_fallback(self, row: _GGUFResidentLoopRow) -> None:
        if row.fallback_output is not None:
            return
        if row.native_sampled:
            raise RuntimeError("sampled GGUF rows must use incremental resident model steps")
        self._route_counts["resident_fallback_requests"] += 1
        plan = _gguf_sampler_plan(row.request)
        fallback_reason = (
            "zero_max_tokens"
            if int(row.request.max_tokens) == 0
            else (plan.fallback_reason or plan.mode.value)
        )
        self._fallback_reasons[str(fallback_reason)] += 1
        if int(row.request.max_tokens) == 0:
            row.fallback_output = self._empty_output(row, None)
            return
        lease = row.lease or self._acquire_lease()
        row.lease = lease
        row.fallback_output = self.generator._generate_sampled(
            lease.session,
            list(row.prompt_ids),
            row.request,
            row_index=row.row_index,
        )

    def _step_native_rows(
        self,
        rows: Sequence[_GGUFResidentLoopRow],
        *,
        work: WorkItem | None = None,
    ) -> None:
        row_list = list(rows)
        if not row_list:
            return
        request_ids = tuple(int(row.request_id) for row in row_list)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("native decode rows must have unique request ids")
        if work is None:
            work = WorkItem(
                kind=WorkKind.DECODE,
                request_ids=request_ids,
                row_to_request=request_ids,
            )
        elif set(work.request_ids) != set(request_ids):
            raise ValueError("physical-group work must contain exactly the native decode rows")

        shared_runner = getattr(self, "_shared_runner", None)
        physical_bucket_widths = _gguf_ar_physical_widths(
            str(getattr(shared_runner, "backend", "hip_gfx1100")),
            use_capability=getattr(self, "_resident_batch_owner", None) is not None,
        )
        width_sequence = None
        cost_table = getattr(self, "_gguf_ar_cost_table", None)
        resident_owner = getattr(self, "_resident_batch_owner", None)
        generator = getattr(self, "generator", None)
        if (
            cost_table is None
            and resident_owner is not None
            and generator is not None
            and os.environ.get(_GGUF_AR_D2_COST_ARTIFACT_ENV, "").strip()
        ):
            kv_dtype = getattr(
                getattr(resident_owner, "kv_storage_dtype", None),
                "value",
                "bf16",
            )
            cost_table = _gguf_ar_resolve_cost_table(
                str(getattr(shared_runner, "backend", "hip_gfx1100")),
                target_arch=str(getattr(shared_runner, "target_arch", "gfx1100")),
                model_path=generator.model_path,
                quant=generator._kv_weight_quant_key(),
                kv_dtype=str(kv_dtype),
                physical_widths=physical_bucket_widths,
            )
        d2_metadata = None
        if cost_table is not None and resident_owner is not None:
            width_sequence = d2_partition(len(work.request_ids), cost_table)
            identity = getattr(cost_table, "identity", None)
            records = tuple(getattr(cost_table, "records", ()))
            d2_metadata = {
                "width_sequence": list(width_sequence),
                "estimated_serial_model_step_ms": sum(
                    float(cost_table.cost_ms(width)) for width in width_sequence
                ),
                "cost_source": None if not records else str(records[0].source),
                "identity": (
                    None
                    if identity is None
                    else identity.to_json_dict()
                ),
            }
        groups = plan_physical_batch_groups(
            work,
            physical_bucket_widths=physical_bucket_widths,
            compact_active_rows=True,
            width_sequence=width_sequence,
        )
        row_by_request = {int(row.request_id): row for row in row_list}
        group_payloads: list[dict[str, Any]] = []
        for group in groups:
            group_rows = [row_by_request[request_id] for request_id in group.request_ids]
            packed = False
            serial_fallback_reason = "packed_decode_unavailable"
            native_sampler_rows = any(
                bool(getattr(row, "native_sampler", False)) for row in group_rows
            )
            host_sampler_rows = any(
                bool(getattr(row, "native_sampled", False))
                and not bool(getattr(row, "native_sampler", False))
                for row in group_rows
            )
            group_slots = [getattr(row, "slot", None) for row in group_rows]
            packed_decode_limit = min(
                (
                    int(getattr(slot.session, "packed_decode_max_rows", 8))
                    if slot is not None
                    else max(_GGUF_AR_PHYSICAL_BUCKET_WIDTHS)
                )
                for slot in group_slots
            )
            if native_sampler_rows and host_sampler_rows:
                serial_fallback_reason = "mixed_sampler_routes"
            elif group.physical_rows > packed_decode_limit:
                serial_fallback_reason = "packed_decode_width_unqualified"
            elif _gguf_ar_packed_decode_enabled() and (
                group.active_rows > 1 or group.physical_rows > 1
            ):
                packed = self._step_native_chunk(
                    group_rows,
                    physical_rows=group.physical_rows,
                    active_slot_indices=group.active_slot_indices,
                    allow_graph=len(groups) == 1,
                )
            if packed:
                execution_path = "packed_native"
            else:
                self._step_native_serial(
                    group_rows,
                    fallback_reason=serial_fallback_reason,
                )
                if group.active_rows == 1 and group.physical_rows == 1:
                    slot = group_rows[0].slot
                    graph = None if slot is None else slot.c1_decode_graph
                    raw_replay_count = getattr(graph, "replay_count", None)
                    graph_replays = (
                        max(0, int(raw_replay_count))
                        if raw_replay_count is not None
                        else max(0, int(getattr(graph, "replayed_steps", 0)))
                        // max(1, int(getattr(graph, "steps_per_replay", 1)))
                    )
                    execution_path = (
                        "native_c1_graph" if graph_replays > 0 else "native_c1_eager"
                    )
                    self._last_execution_manifest = {
                        "schema": 1,
                        "kind": "gguf_ar_c1_execution_manifest",
                        "mode": execution_path,
                        "rows": 1,
                        "physical_rows": 1,
                        "active_rows": 1,
                        "active_mask": [True],
                        "model_step": {
                            "complete_c1_session_replays": 0,
                            "complete_c1_layer_replays": 0,
                            "host_model_row_loop_sites": 0,
                            "host_model_row_iterations": 0,
                        },
                        "graph": {
                            "captured": graph is not None,
                            "replay_count": graph_replays,
                        },
                    }
                else:
                    execution_path = "serial_fallback"
                    attention_sources = {
                        str(getattr(slot.session, "kv_attention_source", "unknown"))
                        for slot in group_slots
                        if slot is not None
                    }
                    attention_source = (
                        next(iter(attention_sources))
                        if len(attention_sources) == 1
                        else "mixed"
                    )
                    self._last_execution_manifest = {
                        "schema": 1,
                        "kind": "gguf_ar_serial_fallback_execution_manifest",
                        "mode": "serial_c1_per_row",
                        "rows": group.active_rows,
                        "physical_rows": 1,
                        "physical_execution_width": 1,
                        "active_rows": group.active_rows,
                        "active_mask": list(group.active_mask),
                        "kv_attention_source": attention_source,
                        "serial_decode_fallback": True,
                        "throughput_claim_eligible": False,
                        "fallback_reason": serial_fallback_reason,
                        "model_step": {
                            "complete_c1_session_replays": group.active_rows,
                            "complete_c1_layer_replays": 0,
                            "host_model_row_loop_sites": 1,
                            "host_model_row_iterations": group.active_rows,
                        },
                    }
            if isinstance(self._last_execution_manifest, Mapping):
                direct_manifest = copy.deepcopy(dict(self._last_execution_manifest))
                direct_manifest["logical_c"] = group.logical_c
                direct_manifest["physical_group"] = group.to_json_dict()
                self._last_execution_manifest = direct_manifest
            group_payload = group.to_json_dict()
            group_payload["execution_path"] = execution_path
            if execution_path == "serial_fallback":
                group_payload["planned_physical_rows"] = int(group.physical_rows)
                group_payload["physical_execution_width"] = 1
            group_payloads.append(group_payload)

        self._last_physical_group_plan = {
            "schema": 1,
            "kind": "gguf_ar_physical_group_plan",
            "logical_c": len(request_ids),
            "physical_bucket_widths": list(physical_bucket_widths),
            "policy": (
                "artifact_backed_d2"
                if d2_metadata is not None
                else "occupancy_adaptive_dense_execution"
            ),
            "group_count": len(groups),
            "groups": group_payloads,
        }
        if d2_metadata is not None:
            self._last_physical_group_plan["d2"] = d2_metadata

    def _packed_graph_capture_membership_stable(self) -> bool:
        """Require every registered native row to finish prefill before capture."""

        return all(
            row.slot is not None
            for row in self._rows.values()
            if row.native_greedy and not row.native_sampled
        )

    def _step_native_chunk(
        self,
        rows: Sequence[_GGUFResidentLoopRow],
        *,
        physical_rows: int | None = None,
        active_slot_indices: Sequence[int] = (),
        allow_graph: bool = True,
    ) -> bool:
        for row in rows:
            self._close_c1_decode_graph(row)
        slots = [row.slot for row in rows]
        if any(slot is None for slot in slots):
            raise RuntimeError("GGUF resident packed decode row is missing its session slot")
        concrete = [slot for slot in slots if slot is not None]
        owner_slot = concrete[0]
        execution_owner = self._packed_execution_owner(owner_slot.session)
        step_batch = getattr(execution_owner, "step_batch_native", None)
        if not callable(step_batch):
            return False
        width = int(physical_rows or len(concrete))
        active_indices = (
            tuple(range(len(concrete)))
            if not active_slot_indices
            else tuple(int(index) for index in active_slot_indices)
        )
        expected_sessions: list[Any | None] = [None] * width
        for slot, index in zip(concrete, active_indices, strict=True):
            expected_sessions[index] = slot.session
        expected_session_tuple = tuple(expected_sessions)
        owner_sessions = tuple(
            getattr(execution_owner, "_packed_decode_sessions", ())
        )
        if (
            bool(getattr(execution_owner, "_packed_decode_state_dirty", False))
            and owner_sessions != expected_session_tuple
        ):
            flush_owner = getattr(execution_owner, "flush_packed_decode_state", None)
            if not callable(flush_owner) or not bool(flush_owner()):
                raise RuntimeError(
                    "GGUF shared packed owner could not flush a changed session"
                    " tuple (dirty="
                    f"{bool(getattr(execution_owner, '_packed_decode_state_dirty', False))}"
                    ", state="
                    f"{getattr(execution_owner, '_packed_verify_state', None) is not None}"
                    ", layout="
                    f"{getattr(execution_owner, '_packed_decode_last_layout', None) is not None}"
                    f", recorded_sessions={len(owner_sessions)}"
                    f", expected_sessions={len(expected_session_tuple)})"
                )
        graphs = {
            id(graph): graph
            for slot in concrete
            for graph in (getattr(slot, "packed_decode_graph", None),)
            if graph is not None and not bool(getattr(graph, "closed", False))
        }
        graph = next(iter(graphs.values())) if len(graphs) == 1 else None
        if graph is not None and tuple(getattr(graph, "sessions", ())) != expected_session_tuple:
            graph = None
        if not bool(allow_graph):
            graph = None
        if graphs and graph is None:
            self._close_packed_decode_graphs(rows)

        self.generator._flush_ar_packed_decode_owners_if_chunk_changed(concrete)
        graph_eligible = bool(
            bool(allow_graph)
            and (
                graph is not None
                or (
                    _gguf_decode_graph_enabled()
                    and len(concrete) == width
                    and active_indices == tuple(range(width))
                    and all(
                        row.native_greedy and not row.native_sampled
                        for row in rows
                    )
                    and self._packed_graph_capture_membership_stable()
                    and not any(
                        bool(
                            getattr(
                                slot,
                                "packed_decode_graph_unavailable",
                                False,
                            )
                        )
                        for slot in concrete
                    )
                )
            )
        )
        if graph is None and graph_eligible:
            minimum_fn = getattr(execution_owner, "decode_graph_min_replay_steps", None)
            minimum = minimum_fn() if callable(minimum_fn) else None
            packed_minimum_fn = getattr(
                execution_owner,
                "packed_decode_graph_min_replay_steps",
                None,
            )
            remaining = min(
                max(0, int(row.request.max_tokens) - len(slot.generated_ids))
                for row, slot in zip(rows, concrete, strict=True)
            )
            scaled_minimum = (
                packed_minimum_fn(width)
                if callable(packed_minimum_fn)
                else (
                    None
                    if minimum is None
                    else max(1, (int(minimum) + width - 1) // width)
                )
            )
            capture = getattr(execution_owner, "capture_packed_decode_graph", None)
            if (
                scaled_minimum is not None
                and remaining >= scaled_minimum
                and callable(capture)
            ):
                try:
                    graph = capture(
                        [int(slot.prev_token) for slot in concrete],
                        sessions=tuple(slot.session for slot in concrete),
                        physical_rows=width,
                        active_slot_indices=active_indices,
                        steps_per_replay=1,
                        max_replay_steps=remaining,
                        record_steps=remaining,
                    )
                except NotImplementedError:
                    for slot in concrete:
                        slot.packed_decode_graph_unavailable = True
                else:
                    self._route_counts["native_packed_graph_captures"] += 1
                    for slot in concrete:
                        slot.packed_decode_graph = graph

        owner = execution_owner
        if graph is not None:
            graph.replay(1)
            physical_tokens = list(graph.read_latest_generated_token_ids())
            if len(physical_tokens) != width:
                raise RuntimeError(
                    f"GGUF resident packed graph returned {len(physical_tokens)} token(s) "
                    f"for physical width {width}"
                )
            self._route_counts["native_packed_graph_replays"] += 1
            self._observe_graph_handles(tuple(slot.session for slot in concrete))
            self._last_execution_manifest = _compact_live_execution_manifest(
                dict(getattr(graph, "execution_manifest", {}))
            )
            self._route_counts["native_packed_decode_steps"] += 1
            self._route_counts[f"native_c{width}_decode_steps"] += 1
            for row, slot, index in zip(rows, concrete, active_indices, strict=True):
                self._record_native_token(row, int(physical_tokens[index]))
                slot.packed_decode_owner = owner
                slot.native_decode_steps += 1
            return True

        native_sampler_rows = any(row.native_sampler for row in rows)
        sample_packed_native = getattr(
            owner,
            "sample_native_from_packed_logits",
            None,
        )
        sample_packed_native_rows = getattr(
            owner,
            "sample_native_from_packed_logits_rows",
            None,
        )
        if native_sampler_rows and not callable(sample_packed_native):
            raise RuntimeError("GGUF packed session has no native sampler integration")
        return_logits = any(
            row.native_sampled and not row.native_sampler for row in rows
        )
        batch_kwargs: dict[str, Any] = {
            "sessions": [slot.session for slot in concrete],
            "positions": [int(slot.seq_position) for slot in concrete],
            "return_logits": return_logits,
            "scatter_state": False,
        }
        if native_sampler_rows:
            batch_kwargs["require_logits"] = True
        if physical_rows is not None:
            batch_kwargs.update(
                {
                    "physical_rows": width,
                    "active_slot_indices": active_indices,
                }
            )
        try:
            with _temporary_env({"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}):
                results = step_batch(
                    [int(slot.prev_token) for slot in concrete],
                    **batch_kwargs,
                )
        except NotImplementedError:
            return False
        if results is None:
            return False
        result_list = list(results)
        if len(result_list) != len(concrete):
            raise RuntimeError(
                f"GGUF resident packed decode returned {len(result_list)} result(s) "
                f"for {len(concrete)} row(s)"
            )
        self._route_counts["native_packed_decode_steps"] += 1
        self._route_counts[f"native_c{width}_decode_steps"] += 1
        native_batch_samples: tuple[Any, ...] | None = None
        if (
            rows
            and all(row.native_sampler for row in rows)
            and callable(sample_packed_native_rows)
        ):
            if any(
                row.sampling_request is None or row.sampling_state is None
                for row in rows
            ):
                raise RuntimeError("GGUF native sampled batch has partial sampler state")
            try:
                native_batch_samples = tuple(
                    sample_packed_native_rows(
                        active_indices,
                        tuple(row.sampling_request for row in rows),
                        tuple(row.sampling_state for row in rows),
                    )
                )
            except NotImplementedError:
                native_batch_samples = None
            else:
                if len(native_batch_samples) != len(rows):
                    raise RuntimeError(
                        "GGUF native sampler batch returned the wrong row count"
                    )
                self._route_counts["native_sampler_batch_launches"] += 1
        for row_index, (row, slot, result, physical_index) in enumerate(
            zip(
                rows,
                concrete,
                result_list,
                active_indices,
                strict=True,
            )
        ):
            if row.native_sampler:
                if row.sampling_request is None or row.sampling_state is None:
                    raise RuntimeError("GGUF native sampled row has no sampler state")
                if native_batch_samples is None:
                    sample = sample_packed_native(
                        int(physical_index),
                        row.sampling_request,
                        row.sampling_state,
                        output_session=slot.session,
                    )
                    self._route_counts["native_sampler_row_launches"] += 1
                else:
                    sample = native_batch_samples[row_index]
                self._record_sampled_result(row, sample)
            elif row.native_sampled:
                self._record_sampled_result(row, result)
            else:
                self._record_native_token(row, int(getattr(result, "token_id")))
            slot.packed_decode_owner = owner
            slot.native_decode_steps += 1
        manifest = getattr(owner, "last_packed_execution_manifest", None)
        if isinstance(manifest, Mapping):
            self._last_execution_manifest = copy.deepcopy(dict(manifest))
        return True

    def _step_native_serial(
        self,
        rows: Sequence[_GGUFResidentLoopRow],
        *,
        fallback_reason: str = "packed_decode_unavailable",
    ) -> None:
        resident_owner = getattr(self, "_resident_batch_owner", None)
        if bool(getattr(resident_owner, "_packed_decode_state_dirty", False)):
            flush_owner = getattr(resident_owner, "flush_packed_decode_state", None)
            if not callable(flush_owner) or not bool(flush_owner()):
                raise RuntimeError(
                    "GGUF shared packed owner could not flush before serial decode"
                )
        self._flush_rows(rows)
        native_c1 = len(rows) == 1
        if native_c1:
            self._route_counts["native_c1_decode_steps"] += 1
        else:
            self._route_counts["serial_decode_fallback_steps"] += 1
            self._route_counts["serial_c1_row_steps"] += len(rows)
            self._fallback_reasons[str(fallback_reason)] += 1
        for row in rows:
            slot = row.slot
            if slot is None:
                raise RuntimeError("GGUF resident serial decode row is missing its session slot")
            result = (
                self._step_native_c1_graph(row)
                if native_c1 and row.native_greedy
                else slot.session.step(
                    int(slot.prev_token),
                    return_logits=bool(
                        row.native_sampled and not row.native_sampler
                    ),
                )
            )
            if row.native_sampler:
                if row.sampling_request is None or row.sampling_state is None:
                    raise RuntimeError("GGUF native sampled row has no sampler state")
                sample_native = getattr(
                    slot.session,
                    "sample_native_from_last_logits",
                    None,
                )
                if not callable(sample_native):
                    raise RuntimeError(
                        "GGUF session has no native sampler integration"
                    )
                sample = sample_native(row.sampling_request, row.sampling_state)
                self._route_counts["native_sampler_row_launches"] += 1
                self._record_sampled_result(row, sample)
            elif row.native_sampled:
                self._record_sampled_result(row, result)
            else:
                self._record_native_token(row, int(getattr(result, "token_id")))
            if native_c1:
                slot.native_c1_decode_steps += 1
            else:
                slot.serial_decode_steps += 1
            self._refresh_prefix_cache(row)

    def _step_native_c1_graph(self, row: _GGUFResidentLoopRow) -> Any:
        slot = row.slot
        if slot is None:
            raise RuntimeError("GGUF resident c1 decode row is missing its session slot")
        if len(getattr(self, "_rows", {})) > 1:
            # Slot views deliberately share the batch owner's execution buffers.
            # A scalar graph captured for the c1 edge of a wider logical round
            # would be overwritten by the peer packed group before replay.
            self._close_c1_decode_graph(row)
            return slot.session.step(int(slot.prev_token), return_logits=False)
        graph = slot.c1_decode_graph
        if graph is not None and bool(getattr(graph, "closed", False)):
            # A shared-buffer growth invalidated this graph; fall back to the
            # eager step and let the next eligible round re-capture.
            slot.c1_decode_graph = None
            graph = None
        if graph is None:
            minimum_fn = getattr(slot.session, "decode_graph_min_replay_steps", None)
            minimum = minimum_fn() if callable(minimum_fn) else None
            remaining = max(0, int(row.request.max_tokens) - len(slot.generated_ids))
            use_graph = bool(
                _gguf_decode_graph_enabled()
                and minimum is not None
                and remaining >= int(minimum)
                and callable(getattr(slot.session, "capture_decode_graph", None))
            )
            if use_graph:
                graph = slot.session.capture_decode_graph(
                    position=int(slot.seq_position),
                    steps_per_replay=1,
                    max_replay_steps=remaining,
                    attention_max_context_len=int(slot.seq_position) + remaining,
                    input_token_id=int(slot.prev_token),
                )
                slot.c1_decode_graph = graph
        if graph is None:
            return slot.session.step(int(slot.prev_token), return_logits=False)
        graph.replay(1)
        return graph.read_sample(return_logits=False)

    def _close_packed_decode_graphs(
        self,
        rows: Sequence[_GGUFResidentLoopRow],
        *,
        flush_state: bool = True,
    ) -> None:
        graphs = {
            id(graph): graph
            for row in rows
            for slot in (row.slot,)
            if slot is not None
            for graph in (getattr(slot, "packed_decode_graph", None),)
            if graph is not None
        }
        for graph in graphs.values():
            sessions = tuple(
                session
                for session in tuple(getattr(graph, "sessions", ()))
                if session is not None
            )
            if sessions:
                self._observe_graph_handles(sessions)
            was_open = not bool(getattr(graph, "closed", False))
            flush = getattr(graph, "flush_packed_state", None)
            if was_open and flush_state and callable(flush):
                flush()
            close = getattr(graph, "close", None)
            if was_open and callable(close):
                close()
            if was_open:
                self._record_graph_invalidations((graph,), 1)
                self._kv_graph_invalidation_count += 1
        if not graphs:
            return
        for candidate in self._rows.values():
            slot = candidate.slot
            if (
                slot is not None
                and id(getattr(slot, "packed_decode_graph", None)) in graphs
            ):
                slot.packed_decode_graph = None

    def _close_c1_decode_graph(self, row: _GGUFResidentLoopRow) -> None:
        slot = row.slot
        if slot is None or slot.c1_decode_graph is None:
            return
        graph = slot.c1_decode_graph
        lease = row.lease
        session_handles: tuple[Any, ...] = ()
        if lease is not None:
            session_handles = self._graph_handles_for_sessions((lease.session,))
            self._observe_graph_handles((lease.session,))
        was_open = not bool(getattr(graph, "closed", False))
        graph.close()
        if was_open and any(handle is graph for handle in session_handles):
            self._record_graph_invalidations((graph,), 1)
            self._kv_graph_invalidation_count += 1
        slot.c1_decode_graph = None

    def _record_sampled_result(self, row: _GGUFResidentLoopRow, result: Any) -> None:
        sampling_request = row.sampling_request
        sampling_state = row.sampling_state
        if sampling_request is None or sampling_state is None:
            raise RuntimeError("GGUF sampled row has no sampling request/state")
        if row.native_sampler:
            sample = result
            row.full_vocab_logits_d2h = False
            row.logits_d2h_bytes = 0
        else:
            sample = _select_from_gguf_logits(
                result,
                sampling_request,
                sampling_state,
                self.generator.tokenizer,
            )
            full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(
                result
            )
            if full_vocab_logits_d2h is not None:
                row.full_vocab_logits_d2h = full_vocab_logits_d2h
                row.logits_d2h_bytes = logits_d2h_bytes
        row.samples.append(sample)
        _gguf_queue_json_object_close_if_needed(
            sampling_state,
            self.generator.tokenizer,
            _gguf_token_text(self.generator.tokenizer, sample),
            remaining_tokens=max(
                0,
                int(sampling_request.max_tokens) - len(row.samples),
            ),
        )
        self._record_native_token(row, int(sample.token_id))

    def _record_native_token(self, row: _GGUFResidentLoopRow, token_id: int) -> None:
        slot = row.slot
        if slot is None:
            raise RuntimeError("GGUF resident row is missing its session slot")
        token = int(token_id)
        vocab_size = int(
            getattr(getattr(self, "_shared_runner", None), "vocab_size", 0) or 0
        )
        if token < 0 or (vocab_size > 0 and token >= vocab_size):
            raise RuntimeError(
                "GGUF decode produced an invalid token: "
                f"request_id={row.request_id} token={token} vocab={vocab_size} "
                f"position={slot.seq_position} "
                f"kv_attention_source={getattr(slot.session, 'kv_attention_source', None)} "
                f"kv_base_row={_gguf_device_kv_contiguous_base_row(slot.session)}"
            )
        slot.generated_ids.append(token)
        slot.prev_token = token
        slot.seq_position += 1
        finish_request = row.sampling_request or row.request
        slot.done = (
            len(slot.generated_ids) >= int(finish_request.max_tokens)
            or _gguf_finished(
                slot.generated_ids,
                self.generator.tokenizer,
                finish_request,
            )
        )

    def _flush_rows(self, rows: Sequence[_GGUFResidentLoopRow]) -> None:
        row_tuple = tuple(rows)
        self._close_packed_decode_graphs(row_tuple)
        slots = [row.slot for row in row_tuple if row.slot is not None]
        if slots:
            self.generator._flush_ar_packed_decode_owners(slots)

    def _flush_row_owner(self, row: _GGUFResidentLoopRow) -> None:
        slot = row.slot
        if slot is None or slot.packed_decode_owner is None:
            return
        owner = slot.packed_decode_owner
        related_rows = [
            candidate_row
            for candidate_row in self._rows.values()
            for candidate in (candidate_row.slot,)
            if candidate is not None and candidate.packed_decode_owner is owner
        ]
        all_done = bool(related_rows) and all(
            candidate.slot is not None and candidate.slot.done
            for candidate in related_rows
        )
        self._close_packed_decode_graphs(
            related_rows,
            flush_state=not all_done,
        )
        concrete = [
            candidate.slot for candidate in related_rows if candidate.slot is not None
        ]
        if all_done:
            # No session survives this physical group, so packed scratch state
            # has no future consumer. Closing the graph and invalidating the
            # owner's deferred binding is sufficient; scattering every layer
            # back to sessions only to reset them immediately adds a terminal
            # GPU synchronization.
            discard = getattr(owner, "discard_packed_decode_state", None)
            if not callable(discard):
                raise RuntimeError("GGUF packed decode owner cannot discard terminal state")
            discard()
            for slot in concrete:
                slot.packed_decode_owner = None
            return
        self.generator._flush_ar_packed_decode_owners(concrete)

    def _flush_all_packed_owners(self) -> None:
        rows = [row for row in self._rows.values() if row.slot is not None]
        self._close_packed_decode_graphs(rows)
        slots = [row.slot for row in rows if row.slot is not None]
        if slots:
            self.generator._flush_ar_packed_decode_owners(slots)

    def decorate_speculative_stream_events(
        self,
        events: Sequence[GeneratedTokenEvent],
    ) -> tuple[GeneratedTokenEvent, ...]:
        """Attach tokenizer-owned text to canonical speculative token events."""

        decorated: list[GeneratedTokenEvent] = []
        suppress_after_special = False
        for event in events:
            chunk = event.stream_chunk
            if chunk is None:
                decorated.append(event)
                continue
            raw_text = self.generator.tokenizer.decode((int(event.token_id),))
            visible_text = self.generator.tokenizer.decode(
                (int(event.token_id),), skip_special=True
            )
            text = "" if suppress_after_special else visible_text
            if raw_text and not visible_text:
                suppress_after_special = True
            decorated.append(
                replace(
                    event,
                    stream_chunk=replace(chunk, text=text),
                )
            )
        return tuple(decorated)

    def _native_stream_chunk(self, row: _GGUFResidentLoopRow) -> GenerationStreamChunk:
        slot = row.slot
        if slot is None or not slot.generated_ids:
            raise RuntimeError("GGUF resident model row has no token to stream")
        generated_ids = tuple(int(token) for token in slot.generated_ids)
        request = row.sampling_request or row.request
        sample = (
            row.samples[-1]
            if row.samples and len(row.samples) == len(generated_ids)
            else None
        )
        timing = dict(slot.timing)
        # Streaming keeps the same per-request MTP counters as the blocking
        # output, so a live/done chunk reports realized speculation rather than
        # only the admission intent.
        timing.update(accounting_timing_fields(speculative_output_accounting(row)))
        execution_path = (
            "gguf_specdec2_mtp2"
            if row.mtp2_cycles > 0
            else (
                "gguf_packed_ar_native_sampler_decode"
                if row.native_sampler
                else (
                    "gguf_packed_ar_host_sampler_decode"
                    if row.native_sampled
                    else "gguf_packed_ar_server_decode"
                )
            )
        )
        return GenerationStreamChunk(
            text=(
                _gguf_token_text(self.generator.tokenizer, sample)
                if sample is not None
                else self.generator.tokenizer.decode(
                    (generated_ids[-1],), skip_special=True
                )
            ),
            token_logprobs=(
                _gguf_stream_token_logprobs(self.generator.tokenizer, sample, request)
                if sample is not None
                else ()
            ),
            finish_details=(
                _gguf_finish_details(
                    generated_ids,
                    self.generator.tokenizer,
                    request,
                    row.sampling_state,
                    sampler_plan=row.sampler_plan,
                )
                if slot.done
                else None
            ),
            telemetry=_gguf_telemetry(
                row.prompt_ids,
                generated_ids,
                request,
                row_index=row.row_index,
                request_id=str(row.request_id),
                sampling_state=row.sampling_state,
                phase=(
                    None
                    if row.sampling_state is not None
                    and row.sampling_state.thinking_budget is not None
                    else "answer"
                ),
                forced_sample=sample,
                full_vocab_logits_d2h=row.full_vocab_logits_d2h,
                logits_d2h_bytes=row.logits_d2h_bytes,
                execution_path=execution_path,
                native_compact_prefill=slot.native_compact_prefill,
                native_caware_decode=slot.native_decode_steps > 0,
                serial_decode_fallback=slot.serial_decode_steps > 0,
                native_sampler_rows=row.native_sampler,
                timing=timing,
                sampler_plan=row.sampler_plan,
                diagnostics=self._request_diagnostics(
                    row,
                    include_kv_layout=slot.done,
                ),
            ),
            generated_token_ids=generated_ids if slot.done else None,
        )

    def _native_output(
        self,
        row: _GGUFResidentLoopRow,
        completed: CompletedRequest,
    ) -> GenerationOutput:
        slot = row.slot
        if slot is None:
            return self._empty_output(row, completed)
        generated_ids = tuple(int(token) for token in slot.generated_ids)
        request = row.sampling_request or row.request
        timing = dict(slot.timing)
        timing["request_total_ms"] = _timing_ms_since(row.submitted_at)
        finish_details = (
            completed.finish_details
            if completed.finish_reason in {"cancel", "disconnect", "timeout"}
            else _gguf_finish_details(
                generated_ids,
                self.generator.tokenizer,
                request,
                row.sampling_state,
                sampler_plan=row.sampler_plan,
            )
        )
        # Speculative outputs have token IDs but no per-token SampleResult.
        # A prefill-only sample list must not truncate the visible completion.
        token_logprobs = tuple(
            _gguf_token_logprob(self.generator.tokenizer, sample)
            for sample in row.samples
        ) if len(row.samples) == len(generated_ids) else ()
        execution_path = (
            "gguf_specdec2_mtp2"
            if row.mtp2_cycles > 0
            else (
                "gguf_packed_ar_native_sampler_decode"
                if row.native_sampler
                else (
                    "gguf_packed_ar_host_sampler_decode"
                    if row.native_sampled
                    else "gguf_packed_ar_server_decode"
                )
            )
        )
        if row.mtp2_candidate_budget > 0:
            timing.update(
                {
                    "specdec2_mtp2_prompt_streaming": float(
                        row.mtp2_prompt_streaming
                    ),
                    "specdec2_mtp2_prompt_prime_rows": float(
                        row.mtp2_prompt_prime_rows
                    ),
                    "specdec2_mtp2_prompt_carried_bytes": float(
                        row.mtp2_prompt_carried_bytes
                    ),
                }
            )
        if row.mtp2_cycles > 0:
            timing.update(accounting_timing_fields(speculative_output_accounting(row)))
            timing.update(
                {
                    "specdec2_mtp2_cycles": float(row.mtp2_cycles),
                    "specdec2_mtp2_proposal_ms": float(row.mtp2_proposal_ms),
                    "specdec2_mtp2_target_ms": float(row.mtp2_target_ms),
                    "specdec2_mtp2_provider_update_ms": float(
                        row.mtp2_provider_update_ms
                    ),
                    "specdec2_mtp2_accept_ms": float(row.mtp2_accept_ms),
                    "specdec2_mtp2_target_readback_ms": float(
                        row.mtp2_target_readback_ms
                    ),
                    "specdec2_mtp2_accept_upload_ms": float(
                        row.mtp2_accept_upload_ms
                    ),
                    "specdec2_mtp2_accept_tail_ms": float(
                        row.mtp2_accept_tail_ms
                    ),
                    "specdec2_mtp2_accept_enqueue_ms": float(
                        row.mtp2_accept_enqueue_ms
                    ),
                    "specdec2_mtp2_selected_commit_ms": float(
                        row.mtp2_selected_commit_ms
                    ),
                    "specdec2_mtp2_candidate_readback_ms": float(
                        row.mtp2_candidate_readback_ms
                    ),
                    "specdec2_mtp2_k0_catchups": float(row.mtp2_k0_catchups),
                    "specdec2_mtp2_ngram_lookup_calls": float(
                        row.mtp2_ngram_lookup_calls
                    ),
                    "specdec2_mtp2_ngram_lookup_hits": float(
                        row.mtp2_ngram_lookup_hits
                    ),
                    "specdec2_mtp2_ngram_cycles": float(row.mtp2_ngram_cycles),
                    "specdec2_mtp2_ngram_probed_tokens": float(
                        row.mtp2_ngram_probed_tokens
                    ),
                    "specdec2_mtp2_ngram_accepted_tokens": float(
                        row.mtp2_ngram_accepted_tokens
                    ),
                    "specdec2_mtp2_recoverable_failures": float(
                        row.mtp2_recoverable_failures
                    ),
                }
            )
        return GenerationOutput(
            text=(
                "".join(token.token_text for token in token_logprobs)
                if token_logprobs
                else self.generator.tokenizer.decode(generated_ids)
            ),
            token_logprobs=token_logprobs,
            generated_token_ids=generated_ids,
            finish_details=finish_details,
            telemetry=_gguf_telemetry(
                row.prompt_ids,
                generated_ids,
                request,
                row_index=row.row_index,
                request_id=str(row.request_id),
                sampling_state=row.sampling_state,
                forced_sample=row.samples[-1] if row.samples else None,
                full_vocab_logits_d2h=row.full_vocab_logits_d2h,
                logits_d2h_bytes=row.logits_d2h_bytes,
                execution_path=execution_path,
                native_compact_prefill=slot.native_compact_prefill,
                native_caware_decode=slot.native_decode_steps > 0,
                serial_decode_fallback=slot.serial_decode_steps > 0,
                native_sampler_rows=row.native_sampler,
                timing=timing,
                sampler_plan=row.sampler_plan,
                diagnostics=self._request_diagnostics(row),
            ),
        )

    def _empty_output(
        self,
        row: _GGUFResidentLoopRow,
        completed: CompletedRequest | None,
    ) -> GenerationOutput:
        finish_details = (
            completed.finish_details
            if completed is not None and completed.finish_reason in {"cancel", "disconnect", "timeout"}
            else _gguf_finish_details((), self.generator.tokenizer, row.request)
        )
        return GenerationOutput(
            text="",
            generated_token_ids=(),
            finish_details=finish_details,
            telemetry=_gguf_telemetry(
                row.prompt_ids,
                (),
                row.request,
                row_index=row.row_index,
                request_id=str(row.request_id),
                execution_path="gguf_resident_model_loop",
                native_compact_prefill=False,
                native_caware_decode=False,
                serial_decode_fallback=False,
                native_sampler_rows=False,
                timing={
                    "tokenize_ms": float(row.tokenize_ms),
                    "prompt_encode_ms": float(row.prompt_encode_ms),
                    "render_ms": float(row.render_ms),
                    "admission_prepare_ms": float(row.admission_prepare_ms),
                    "request_total_ms": _timing_ms_since(row.submitted_at),
                },
                diagnostics=self._request_diagnostics(row),
            ),
        )

    def _execution_metadata(self, row: _GGUFResidentLoopRow) -> dict[str, Any]:
        slot = row.slot
        return {
            "native_greedy": bool(row.native_greedy),
            "native_sampled": bool(row.native_sampled),
            "native_sampler": bool(row.native_sampler),
            "native_compact_prefill": bool(slot is not None and slot.native_compact_prefill),
            "native_decode_steps": 0 if slot is None else int(slot.native_decode_steps),
            "native_c1_decode_steps": (
                0 if slot is None else int(slot.native_c1_decode_steps)
            ),
            "serial_decode_fallback": bool(
                slot is not None and slot.serial_decode_steps > 0
            ),
            "specdec2_mtp2_used": bool(row.mtp2_cycles > 0),
            "specdec2_mtp2_prompt_streaming": bool(row.mtp2_prompt_streaming),
            "specdec2_mtp2_prompt_prime_rows": int(row.mtp2_prompt_prime_rows),
            "specdec2_mtp2_prompt_carried_bytes": int(
                row.mtp2_prompt_carried_bytes
            ),
            "specdec2_mtp2_prompt_fallback_reason": row.mtp2_prompt_fallback_reason,
            "specdec2_mtp2_cycles": int(row.mtp2_cycles),
            "specdec2_mtp2_candidate_counts": list(row.mtp2_candidate_counts),
            "specdec2_mtp2_accepted_counts": list(row.mtp2_accepted_counts),
            "specdec2_mtp2_proposal_ms": float(row.mtp2_proposal_ms),
            "specdec2_mtp2_target_ms": float(row.mtp2_target_ms),
            "specdec2_mtp2_provider_update_ms": float(
                row.mtp2_provider_update_ms
            ),
            "specdec2_mtp2_accept_ms": float(row.mtp2_accept_ms),
            "specdec2_mtp2_target_readback_ms": float(row.mtp2_target_readback_ms),
            "specdec2_mtp2_accept_upload_ms": float(row.mtp2_accept_upload_ms),
            "specdec2_mtp2_accept_tail_ms": float(row.mtp2_accept_tail_ms),
            "specdec2_mtp2_accept_enqueue_ms": float(row.mtp2_accept_enqueue_ms),
            "specdec2_mtp2_selected_commit_ms": float(
                row.mtp2_selected_commit_ms
            ),
            "specdec2_mtp2_candidate_readback_ms": float(
                row.mtp2_candidate_readback_ms
            ),
            "specdec2_mtp2_k0_catchups": int(row.mtp2_k0_catchups),
            "specdec2_mtp2_ngram_lookup_calls": int(
                row.mtp2_ngram_lookup_calls
            ),
            "specdec2_mtp2_ngram_lookup_hits": int(
                row.mtp2_ngram_lookup_hits
            ),
            "specdec2_mtp2_ngram_cycles": int(row.mtp2_ngram_cycles),
            "specdec2_mtp2_ngram_probed_tokens": int(
                row.mtp2_ngram_probed_tokens
            ),
            "specdec2_mtp2_ngram_accepted_tokens": int(
                row.mtp2_ngram_accepted_tokens
            ),
            "specdec2_mtp2_proposal_batch_calls": int(
                row.mtp2_proposal_batch_calls
            ),
            "specdec2_mtp2_proposal_physical_rows": list(
                row.mtp2_proposal_physical_rows
            ),
            "specdec2_mtp2_target_batch_calls": int(row.mtp2_target_batch_calls),
            "specdec2_mtp2_target_physical_rows": list(
                row.mtp2_target_physical_rows
            ),
            "specdec2_mtp2_target_pass_ms": list(row.mtp2_target_pass_ms),
            "specdec2_mtp2_target_pass_start_ns": list(
                row.mtp2_target_pass_start_ns
            ),
            "specdec2_mtp2_target_pass_end_ns": list(row.mtp2_target_pass_end_ns),
            "specdec2_mtp2_cycle_profile_start_ns": list(
                row.mtp2_cycle_profile_start_ns
            ),
            "specdec2_mtp2_cycle_profile_end_ns": list(
                row.mtp2_cycle_profile_end_ns
            ),
            "specdec2_mtp2_accept_pass_ms": list(row.mtp2_accept_pass_ms),
            "specdec2_mtp2_provider_update_pass_ms": list(
                row.mtp2_provider_update_pass_ms
            ),
            "specdec2_mtp2_candidate_device_handoffs": int(
                row.mtp2_candidate_device_handoffs
            ),
            "specdec2_mtp2_candidate_d2h_after_target": int(
                row.mtp2_candidate_d2h_after_target
            ),
            "specdec2_mtp2_device_chain_oracle_trace": copy.deepcopy(
                row.mtp2_device_chain_oracle_trace
            ),
            "specdec2_mtp2_device_accept_calls": int(
                row.mtp2_device_accept_calls
            ),
            "specdec2_mtp2_selected_commit_batch_calls": int(
                row.mtp2_selected_commit_batch_calls
            ),
            "specdec2_mtp2_execution_routes": list(row.mtp2_execution_routes),
            "specdec2_mtp2_recoverable_failures": int(
                row.mtp2_recoverable_failures
            ),
            "specdec2_mtp2_failure_reasons": list(row.mtp2_failure_reasons),
            "prefix_eligible": bool(row.prefix_eligible),
            "prefix_lookup": bool(row.prefix_lookup),
            "prefix_matched_tokens": int(row.prefix_matched_tokens),
            "prefix_reused_tokens": int(row.prefix_reused_tokens),
            "prefix_source_request_id": row.prefix_source_request_id,
            "prefix_source_kind": row.prefix_source_kind,
            "prefix_state_clone_bytes": int(row.prefix_state_clone_bytes),
            "prefix_snapshot_hit": bool(row.prefix_snapshot_hit),
            "prefix_admission_fallback": bool(row.prefix_admission_fallback),
            "prefix_fallback_reason": row.prefix_fallback_reason,
        }


def _select_from_gguf_logits(
    result: Any,
    request: GenerationRequest,
    state: RowSamplingState,
    tokenizer: Qwen35GGUFTokenizer | None = None,
):
    logits = getattr(result, "logits", None)
    if logits is None:
        raise RuntimeError("GGUF sampled generation requires logits from the resident session")
    return select_token(
        logits.reshape(-1),
        request,
        state,
        token_text_for_id=(
            None
            if tokenizer is None
            else lambda token_id: tokenizer.decode([int(token_id)])
        ),
    )


def _gguf_logits_d2h_metadata(result: Any) -> tuple[bool | None, int | None]:
    logits = getattr(result, "logits", None)
    if logits is None:
        return None, None
    size = getattr(logits, "size", None)
    itemsize = getattr(getattr(logits, "dtype", None), "itemsize", None)
    try:
        if int(size) > 0 and int(itemsize) > 0:
            return True, int(size) * int(itemsize)
    except (TypeError, ValueError):
        pass
    shape = getattr(logits, "shape", None)
    if shape:
        try:
            vocab_size = int(shape[-1])
        except (TypeError, ValueError):
            return True, None
        if vocab_size > 0:
            return True, vocab_size * 4
    return True, None


def _request_with_tokenizer_eos(
    request: GenerationRequest,
    tokenizer: Qwen35GGUFTokenizer,
) -> GenerationRequest:
    if request.eos_token_id is not None:
        return request
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return request
    return replace(request, eos_token_id=int(eos_token_id))


def _gguf_row_sampling_state(
    request: GenerationRequest,
    prompt_ids: list[int],
    *,
    row_index: int,
) -> RowSamplingState:
    return RowSamplingState(
        prompt_tokens=tuple(int(token) for token in prompt_ids),
        seed=row_seed_for_index(request, row_index),
        row_index=row_index,
        stop_token_sequences=request.stop_token_sequences,
        forced_tokens_pending=request.forced_tokens_pending,
        forced_token_reason=request.forced_token_reason,
        post_thinking_forced_tokens_pending=request.post_thinking_forced_tokens_pending,
        post_thinking_forced_token_reason=request.post_thinking_forced_token_reason,
        force_sequence_completion_token_sequences=request.force_sequence_completion_token_sequences,
        force_sequence_completion_reason=request.force_sequence_completion_reason,
        json_object_close_forcing=request.json_object_close_forcing,
        tool_call_constraint=request.tool_call_constraint,
        thinking_budget=thinking_budget_state_from_params(request),
    )


def _gguf_generation_output(
    tokenizer: Qwen35GGUFTokenizer,
    samples,
    *,
    finish_details: FinishDetails,
    telemetry: GenerationTelemetry | None = None,
) -> GenerationOutput:
    token_logprobs = tuple(_gguf_token_logprob(tokenizer, sample) for sample in samples)
    return GenerationOutput(
        text="".join(token.token_text for token in token_logprobs),
        token_logprobs=token_logprobs,
        generated_token_ids=tuple(token.token_id for token in token_logprobs),
        finish_details=finish_details,
        telemetry=telemetry,
    )


def _with_batch_timing_ownership(
    outputs: list[GenerationOutput],
    *,
    batch_id: str,
) -> list[GenerationOutput]:
    """Mark copied group timing once while preserving every row's decode state."""

    group_rows = len(outputs)
    owned_outputs: list[GenerationOutput] = []
    for output_index, output in enumerate(outputs):
        telemetry = output.telemetry
        if telemetry is None or telemetry.timing is None:
            owned_outputs.append(output)
            continue
        owned_outputs.append(
            replace(
                output,
                telemetry=replace(
                    telemetry,
                    timing_scope="batch",
                    batch_id=batch_id,
                    group_rows=group_rows,
                    timing_owner=output_index == 0,
                ),
            )
        )
    return owned_outputs


def _gguf_stream_token_logprobs(
    tokenizer: Qwen35GGUFTokenizer,
    sample: Any,
    request: GenerationRequest,
) -> tuple[TokenLogprob, ...]:
    if not request.logprobs and int(request.top_logprobs) <= 0:
        return ()
    return (_gguf_token_logprob(tokenizer, sample),)


def _gguf_token_logprob(tokenizer: Qwen35GGUFTokenizer, sample: Any) -> TokenLogprob:
    return TokenLogprob(
        token_id=sample.token_id,
        token_text=_gguf_token_text(tokenizer, sample),
        logprob=sample.logprob,
        top_logprobs=tuple(
            (token_id, tokenizer.decode([int(token_id)]), logprob)
            for token_id, logprob in sample.top_logprobs
        ),
    )


def _gguf_token_text(tokenizer: Qwen35GGUFTokenizer, sample: Any) -> str:
    token_text = getattr(sample, "token_text", None)
    if token_text is not None:
        return str(token_text)
    return tokenizer.decode([int(sample.token_id)])


def _gguf_last_batch_generation(
    tokenizer: Qwen35GGUFTokenizer,
    request: GenerationRequest,
    plan: Any,
    prompt_rows_by_request: dict[int, list[int]],
    generated_ids_by_request: dict[int, list[int]],
    token_logprobs_by_request: dict[int, list[TokenLogprob]],
    *,
    outputs: tuple[GenerationOutput, ...],
    execution_path: str | None = None,
    native_batch: bool = False,
    native_compact_prefill: bool = False,
    native_decode_steps: int = 0,
    native_c1_decode_steps: int = 0,
    native_caware_decode: bool = False,
    serial_decode_fallback: bool | None = None,
    native_sampler_rows: bool = False,
    execution_paths: dict[str, str] | None = None,
    scheduling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_ids = tuple(range(len(outputs)))
    path = (
        "gguf_native_continuous_decode"
        if native_batch
        else execution_path or _gguf_execution_path(plan)
    )
    prompt_lengths = [len(prompt_rows_by_request.get(request_id, ())) for request_id in request_ids]
    decode_steps = max((len(generated_ids_by_request.get(request_id, ())) for request_id in request_ids), default=0)
    native_caware_decode = bool(native_caware_decode or native_batch)
    native_sampler_rows = bool(native_sampler_rows or native_batch)
    serial_fallback = (
        False
        if native_batch
        else len(request_ids) > 1 if serial_decode_fallback is None else bool(serial_decode_fallback)
    )
    payload: dict[str, Any] = {
        "path": path,
        "batch_size": len(request_ids),
        "request_ids": list(request_ids),
        "prompt_lengths": prompt_lengths,
        "decode_steps": decode_steps,
        "native_decode_steps": int(native_decode_steps),
        "native_c1_decode_steps": int(native_c1_decode_steps),
        "serial_decode_fallback": serial_fallback,
        "native_compact_prefill": bool(native_compact_prefill),
        "native_caware_decode": native_caware_decode,
        "native_sampler_rows": native_sampler_rows,
        "throughput_claim_eligible": bool(native_batch and native_decode_steps > 0),
        "sampler_plan_metadata": [
            {
                "active_processors": list(plan.active_processors),
                "sampler_fast_path_blockers": list(plan.fast_path_blockers),
                "native_gpu_available": bool(plan.native_gpu_available or native_batch),
                **(
                    {"sampler_fallback_reason": plan.fallback_reason}
                    if plan.fallback_reason is not None
                    else {}
                ),
                "sampler_mode": plan.mode.value,
            }
            for _request_id in request_ids
        ],
    }
    if execution_paths:
        payload["native_execution_paths"] = dict(execution_paths)
    if scheduling:
        payload["continuous_scheduler"] = dict(scheduling)
    payload["scheduler_token_chunks"] = _gguf_scheduler_token_chunks(
        request_ids,
        prompt_rows_by_request,
        generated_ids_by_request,
        token_logprobs_by_request,
        tokenizer=tokenizer,
        request=request,
        plan=plan,
        execution_path=path,
        native_compact_prefill=bool(native_compact_prefill),
        native_caware_decode=native_caware_decode,
        serial_decode_fallback=serial_fallback,
        native_sampler_rows=native_sampler_rows,
    )
    return payload


def _gguf_mtp_last_batch_generation(
    tokenizer: Qwen35GGUFTokenizer,
    request: GenerationRequest,
    plan: Any,
    prompt_rows_by_request: dict[int, list[int]],
    generated_ids_by_request: dict[int, list[int]],
    token_logprobs_by_request: dict[int, list[TokenLogprob]],
    *,
    outputs: tuple[GenerationOutput, ...],
    cycles_by_request: dict[int, list[dict[str, Any]]],
    resident_slot_count: int = 1,
    target_verify_batching: str | None = None,
) -> dict[str, Any]:
    request_ids = tuple(range(len(outputs)))
    path = "gguf_llama_compat_mtp_server"
    cycles = [cycle for request_id in request_ids for cycle in cycles_by_request.get(request_id, ())]
    total_drafts = sum(int(cycle.get("generated_draft_tokens", 0)) for cycle in cycles)
    total_accepted = sum(int(cycle.get("accepted_draft_tokens", 0)) for cycle in cycles)
    visible_from_cycles = sum(int(cycle.get("visible_output_tokens", 0)) for cycle in cycles)
    mtp_summary = _mtp_cycle_summary(cycles)
    payload: dict[str, Any] = {
        "path": path,
        "batch_size": len(request_ids),
        "request_ids": list(request_ids),
        "prompt_lengths": [len(prompt_rows_by_request.get(request_id, ())) for request_id in request_ids],
        "decode_steps": max((len(generated_ids_by_request.get(request_id, ())) for request_id in request_ids), default=0),
        "native_decode_steps": 0,
        "serial_decode_fallback": False,
        "native_compact_prefill": False,
        "native_caware_decode": False,
        "native_sampler_rows": False,
        "throughput_claim_eligible": False,
        "speculative_mtp": {
            "serving_route": "llama_compat",
            "draft_n_max": 2,
            "target_verify": "bulk_direct_commit",
            "target_verify_batching": target_verify_batching or (
                "per_slot_serial"
                if int(resident_slot_count) > 1
                else "single_slot"
            ),
            "device_chain": True,
            "device_kv_cache": True,
            "resident_slot_count": int(resident_slot_count),
            "scheduler": (
                "resident_slots_phase_serial"
                if int(resident_slot_count) > 1
                else "single_resident_slot"
            ),
            "total_draft_tokens": total_drafts,
            "total_accepted_draft_tokens": total_accepted,
            "accept_per_draft": (total_accepted / total_drafts if total_drafts > 0 else 0.0),
            "visible_output_tokens_from_cycles": visible_from_cycles,
            "target_verify_rows": mtp_summary["linear_state_captured_rows"],
            "direct_cycles": mtp_summary["direct_cycles"],
            "full_accept_cycles": mtp_summary["full_accept_cycles"],
            "partial_accept_cycles": mtp_summary["partial_accept_cycles"],
            "reject_cycles": mtp_summary["reject_cycles"],
            "full_accept_rate": mtp_summary["full_accept_rate"],
            "accepted_draft_tokens_histogram": mtp_summary["accepted_draft_tokens_histogram"],
            "cycle_shape_histogram": mtp_summary["cycle_shape_histogram"],
            "linear_state_captured_rows": mtp_summary["linear_state_captured_rows"],
            "linear_state_commit_rows": mtp_summary["linear_state_commit_rows"],
            "linear_state_extra_rows": mtp_summary["linear_state_extra_rows"],
            "hidden_seed_captured_rows": mtp_summary["hidden_seed_captured_rows"],
            "hidden_seed_needed_rows": mtp_summary["hidden_seed_needed_rows"],
            "hidden_seed_extra_rows": mtp_summary["hidden_seed_extra_rows"],
            "cycles_by_request": {
                str(request_id): list(cycles_by_request.get(request_id, ()))
                for request_id in request_ids
            },
        },
        "sampler_plan_metadata": [
            {
                "active_processors": list(plan.active_processors),
                "sampler_fast_path_blockers": list(plan.fast_path_blockers),
                "native_gpu_available": False,
                **(
                    {"sampler_fallback_reason": plan.fallback_reason}
                    if plan.fallback_reason is not None
                    else {}
                ),
                "sampler_mode": plan.mode.value,
            }
            for _request_id in request_ids
        ],
    }
    payload["scheduler_token_chunks"] = _gguf_scheduler_token_chunks(
        request_ids,
        prompt_rows_by_request,
        generated_ids_by_request,
        token_logprobs_by_request,
        tokenizer=tokenizer,
        request=request,
        plan=plan,
        execution_path=path,
    )
    return payload


def _gguf_execution_path(plan: Any) -> str:
    if plan.mode is SamplingMode.GREEDY_FAST:
        return "gguf_serial_greedy_decode"
    return "gguf_serial_host_sampler_decode"


def _gguf_scheduler_token_chunks(
    request_ids: tuple[int, ...],
    prompt_rows_by_request: dict[int, list[int]],
    generated_ids_by_request: dict[int, list[int]],
    token_logprobs_by_request: dict[int, list[TokenLogprob]],
    *,
    tokenizer: Qwen35GGUFTokenizer,
    request: GenerationRequest,
    plan: Any,
    execution_path: str,
    native_compact_prefill: bool = False,
    native_caware_decode: bool = False,
    serial_decode_fallback: bool | None = None,
    native_sampler_rows: bool = False,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    serial_fallback = len(request_ids) > 1 if serial_decode_fallback is None else bool(serial_decode_fallback)
    for request_id in request_ids:
        generated_ids = generated_ids_by_request.get(request_id, [])
        token_logprobs = token_logprobs_by_request.get(request_id, [])
        prefix: list[int] = []
        for token_index, token_id in enumerate(generated_ids):
            prefix.append(int(token_id))
            final = token_index == len(generated_ids) - 1
            token_logprob = token_logprobs[token_index] if token_index < len(token_logprobs) else None
            token_text = (
                token_logprob.token_text
                if token_logprob is not None
                else tokenizer.decode([int(token_id)])
            )
            chunk = GenerationStreamChunk(
                text=token_text,
                token_logprobs=(
                    (token_logprob,)
                    if token_logprob is not None and (request.logprobs or int(request.top_logprobs) > 0)
                    else ()
                ),
                finish_details=(
                    _gguf_finish_details(prefix, tokenizer, request)
                    if final
                    else None
                ),
                telemetry=_gguf_telemetry(
                    prompt_rows_by_request.get(request_id, []),
                    prefix,
                    request,
                    row_index=request_id,
                    request_id=str(request_id),
                    phase="answer",
                    execution_path=execution_path,
                    native_compact_prefill=bool(native_compact_prefill),
                    native_caware_decode=bool(native_caware_decode),
                    serial_decode_fallback=serial_fallback,
                    native_sampler_rows=bool(native_sampler_rows),
                    sampler_plan=plan,
                ),
                generated_token_ids=tuple(prefix) if final else None,
            )
            chunks.append(_gguf_scheduler_token_chunk_payload(request_id, token_index, int(token_id), chunk))
    return chunks


def _gguf_scheduler_token_chunk_payload(
    request_id: int,
    token_index: int,
    token_id: int,
    chunk: GenerationStreamChunk,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": int(request_id),
        "token_index": int(token_index),
        "token_id": int(token_id),
        "finished": chunk.finish_details is not None,
        "chunk": {"text": chunk.text},
    }
    if chunk.token_logprobs:
        payload["chunk"]["token_logprobs"] = [
            {
                "token_id": token.token_id,
                "token_text": token.token_text,
                "logprob": token.logprob,
                "top_logprobs": [
                    {"token_id": top_id, "token_text": top_text, "logprob": top_logprob}
                    for top_id, top_text, top_logprob in token.top_logprobs
                ],
            }
            for token in chunk.token_logprobs
        ]
    if chunk.finish_details is not None:
        payload["chunk"]["finish_details"] = chunk.finish_details.to_json_dict()
    if chunk.generated_token_ids is not None:
        payload["chunk"]["generated_token_ids"] = list(chunk.generated_token_ids)
    if chunk.telemetry is not None:
        payload["chunk"]["telemetry"] = chunk.telemetry.to_json_dict()
    return payload


def _gguf_queue_json_object_close_if_needed(
    state: RowSamplingState,
    tokenizer: Qwen35GGUFTokenizer,
    token_text: str,
    *,
    remaining_tokens: int,
) -> None:
    state.observe_text_for_json_object_close(
        token_text,
        remaining_tokens=remaining_tokens,
        encode_text=lambda text: tuple(int(token) for token in tokenizer.encode(str(text))),
    )


def _gguf_telemetry(
    prompt_ids: list[int] | tuple[int, ...],
    generated_ids: list[int] | tuple[int, ...],
    request: GenerationRequest,
    *,
    row_index: int,
    request_id: str | None = None,
    sampling_state: RowSamplingState | None = None,
    phase: str | None = None,
    forced_sample: Any | None = None,
    full_vocab_logits_d2h: bool | None = None,
    logits_d2h_bytes: int | None = None,
    execution_path: str | None = None,
    native_compact_prefill: bool | None = None,
    native_caware_decode: bool | None = None,
    serial_decode_fallback: bool | None = None,
    native_sampler_rows: bool | None = None,
    timing: dict[str, float] | None = None,
    timing_scope: str | None = None,
    batch_id: str | None = None,
    group_rows: int | None = None,
    timing_owner: bool | None = None,
    sampler_plan: Any | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> GenerationTelemetry:
    if timing is not None and timing_scope is None:
        timing_scope = "choice"
        group_rows = 1 if group_rows is None else int(group_rows)
        timing_owner = True if timing_owner is None else bool(timing_owner)
    plan = _gguf_sampler_plan(request) if sampler_plan is None else sampler_plan
    state_payload = _gguf_decode_state_from_sampling_state(sampling_state)
    forced_token_id, forced_token_reason, forced_tokens_remaining = _gguf_forced_token_metadata(forced_sample)
    return GenerationTelemetry.from_decode_counts(
        request_id=request_id,
        row_index=row_index,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated_ids),
        phase=phase or state_payload.get("phase", "done"),
        reasoning_tokens=int(state_payload.get("reasoning_tokens", 0)),
        answer_tokens=int(state_payload.get("answer_tokens", 0)),
        forced_tokens_pending=tuple(state_payload.get("forced_tokens_pending", ())),
        forced_token_id=forced_token_id,
        forced_token_reason=forced_token_reason,
        forced_tokens_remaining=forced_tokens_remaining,
        post_thinking_forced_tokens_pending=tuple(state_payload.get("post_thinking_forced_tokens_pending", ())),
        post_thinking_forced_token_reason=state_payload.get("post_thinking_forced_token_reason"),
        force_sequence_completion_token_sequences=tuple(
            tuple(sequence) for sequence in state_payload.get("force_sequence_completion_token_sequences", ())
        ),
        force_sequence_completion_reason=state_payload.get("force_sequence_completion_reason"),
        budget_pressure=state_payload.get("budget_pressure"),
        sampler_mode=plan.mode.value,
        stop_suffix_state=_gguf_stop_suffix_state(generated_ids, request.stop_token_sequences),
        active_processors=plan.active_processors,
        sampler_fast_path_blockers=plan.fast_path_blockers,
        sampler_fallback_reason=plan.fallback_reason,
        full_vocab_logits_d2h=full_vocab_logits_d2h,
        logits_d2h_bytes=logits_d2h_bytes,
        execution_path=execution_path,
        native_compact_prefill=native_compact_prefill,
        native_caware_decode=native_caware_decode,
        serial_decode_fallback=serial_decode_fallback,
        native_sampler_rows=native_sampler_rows,
        timing=timing,
        timing_scope=timing_scope,
        batch_id=batch_id,
        group_rows=group_rows,
        timing_owner=timing_owner,
        diagnostics=diagnostics,
    )


def _gguf_forced_token_metadata(sample: Any | None) -> tuple[int | None, str | None, int | None]:
    if sample is None or not bool(getattr(sample, "forced", False)):
        return None, None, None
    return (
        int(getattr(sample, "token_id")),
        None if getattr(sample, "forced_reason", None) is None else str(getattr(sample, "forced_reason")),
        max(0, int(getattr(sample, "forced_tokens_remaining", 0))),
    )


def _gguf_decode_state_from_sampling_state(state: RowSamplingState | None) -> dict[str, Any]:
    if state is None:
        return {}
    payload: dict[str, Any] = {}
    if state.forced_tokens:
        payload["forced_tokens_pending"] = state.forced_tokens
    if state.post_thinking_forced_tokens_pending.pending_tokens:
        payload["post_thinking_forced_tokens_pending"] = state.post_thinking_forced_tokens_pending.pending_tokens
    if state.post_thinking_forced_token_reason is not None:
        payload["post_thinking_forced_token_reason"] = state.post_thinking_forced_token_reason
    if state.force_sequence_completion_token_sequences:
        payload["force_sequence_completion_token_sequences"] = state.force_sequence_completion_token_sequences
    if state.force_sequence_completion_reason is not None:
        payload["force_sequence_completion_reason"] = state.force_sequence_completion_reason
    budget = state.thinking_budget
    if budget is None:
        return payload
    payload["phase"] = str(budget.phase)
    payload["reasoning_tokens"] = int(budget.reasoning_tokens)
    payload["answer_tokens"] = int(budget.answer_tokens)
    forced_reason = getattr(budget.forced_tokens, "reason", None)
    pressure = "hard_close" if forced_reason == "thinking_hard_close" else budget.budget_pressure
    if pressure is not None:
        payload["budget_pressure"] = str(pressure)
    return payload


def _gguf_stop_suffix_state(
    generated_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> dict[str, Any] | None:
    if not stop_token_sequences:
        return None
    payload = token_sequence_state_for_tokens(generated_ids, stop_token_sequences).to_json_dict()
    return payload or None


def _gguf_finished(
    generated_ids: list[int] | tuple[int, ...],
    tokenizer: Qwen35GGUFTokenizer,
    request: GenerationRequest,
) -> bool:
    if not generated_ids:
        return False
    token_id = int(generated_ids[-1])
    eos_token_id = (
        getattr(tokenizer, "eos_token_id", None)
        if request.eos_token_id is None
        else request.eos_token_id
    )
    if (
        not request.ignore_eos
        and eos_token_id is not None
        and token_id == int(eos_token_id)
    ):
        return True
    if token_id in {int(stop_id) for stop_id in request.stop_token_ids}:
        return True
    for sequence in request.stop_token_sequences:
        if len(sequence) <= 0 or len(sequence) > len(generated_ids):
            continue
        if tuple(int(token) for token in generated_ids[-len(sequence) :]) == sequence:
            return True
    return False


def _gguf_finish_details(
    generated_ids: list[int] | tuple[int, ...],
    tokenizer: Qwen35GGUFTokenizer,
    request: GenerationRequest,
    state: RowSamplingState | None = None,
    *,
    sampler_plan: Any | None = None,
) -> FinishDetails:
    details: FinishDetails
    sampler_mode = (
        _sampler_mode_value(request)
        if sampler_plan is None
        else str(sampler_plan.mode.value)
    )
    if generated_ids:
        token_id = int(generated_ids[-1])
        eos_token_id = (
            getattr(tokenizer, "eos_token_id", None)
            if request.eos_token_id is None
            else request.eos_token_id
        )
        if (
            not request.ignore_eos
            and eos_token_id is not None
            and token_id == int(eos_token_id)
        ):
            details = FinishDetails(reason="eos", eos_token_id=token_id, sampler_mode=sampler_mode)
            return finish_details_with_sampling_state(details, state)
        if token_id in {int(stop_id) for stop_id in request.stop_token_ids}:
            details = FinishDetails(reason="stop", stop_sequence=(token_id,), sampler_mode=sampler_mode)
            return finish_details_with_sampling_state(details, state)
        sequence = _gguf_stop_sequence_match(generated_ids, request.stop_token_sequences)
        if sequence:
            details = FinishDetails(reason="stop", stop_sequence=sequence, sampler_mode=sampler_mode)
            return finish_details_with_sampling_state(details, state)
    if len(generated_ids) >= max(0, int(request.max_tokens)):
        details = FinishDetails(reason="length", length_limit=request.max_tokens, sampler_mode=sampler_mode)
        return finish_details_with_sampling_state(details, state)
    details = FinishDetails(reason="stop", sampler_mode=sampler_mode)
    return finish_details_with_sampling_state(details, state)


def _gguf_stop_sequence_match(
    generated_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    return token_sequence_state_for_tokens(generated_ids, stop_token_sequences).matched_sequence


def _sampler_mode_value(request: GenerationRequest) -> str:
    return _gguf_sampler_plan(request).mode.value


def _gguf_sampler_plan(
    request: GenerationRequest,
    *,
    native_gpu_available: bool = False,
):
    native_requested = _native_gpu_sampler_requested()
    return plan_sampler(
        request,
        native_gpu_available=bool(native_gpu_available and native_requested),
        native_gpu_requested=native_requested,
    )


def _gguf_native_sampler_plan_enabled(
    request: GenerationRequest,
    plan: Any,
) -> bool:
    return bool(
        plan.native_gpu_available
        and plan.mode is SamplingMode.GPU_SAMPLE
        and supports_native_gpu_sampling(request)
    )


def _native_gpu_sampler_requested() -> bool:
    value = os.environ.get("HIPENGINE_QWEN35_NATIVE_SAMPLER")
    return value is None or value.strip().lower() not in {"", "0", "false", "no", "off"}


def make_qwen35_gguf_bringup_generator(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
) -> Qwen35GGUFBringupGenerator:
    backend = "hip_gfx1100"
    return Qwen35GGUFBringupGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend=backend,
        server_plain_ar_max_active_requests=int(
            backend_package_capability(
                backend,
                "GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS",
                4,
            )
        ),
        server_plain_ar_max_active_requests_by_max_sequence_length=dict(
            backend_package_capability(
                backend,
                "GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS_BY_MAX_SEQUENCE_LENGTH",
                {},
            )
        ),
    )


def make_qwen35_gguf_q4_k_m_generator_gfx1100(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
) -> Qwen35GGUFBringupGenerator:
    """Create the gfx1100 Q4_K_M generator with the measured fair launch default.

    gfx1151's F4 packet retained the scoped ``fair:256`` default for the same
    (backend-family, quant) shape, and the W7900 Qwen3.8-27B Q4_K_M/BF16-KV
    16K server measured width-4 packed AR decode under fair/burst-1 while the
    protect_decode default serialized concurrent request decodes. The explicit
    env override remains the pin for configurations that must stay on
    protect_decode (the A4 frozen UD-Q4_K_M gates).
    """

    backend = "hip_gfx1100"
    return Qwen35GGUFBringupGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend=backend,
        engine_loop_config_defaults={
            "prefill_decode_policy": backend_package_capability(
                backend,
                "GGUF_Q4_K_M_PREFILL_DECODE_POLICY",
                "fair",
            ),
            "max_prefill_chunk_tokens": int(
                backend_package_capability(
                    backend,
                    "GGUF_Q4_K_M_MAX_PREFILL_CHUNK_TOKENS",
                    256,
                )
            ),
            "fair_prefill_burst_chunks": int(
                backend_package_capability(
                    backend,
                    "GGUF_Q4_K_M_FAIR_PREFILL_BURST_CHUNKS",
                    1,
                )
            ),
        },
        server_plain_ar_max_active_requests=int(
            backend_package_capability(
                backend,
                "GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS",
                4,
            )
        ),
        server_plain_ar_max_active_requests_by_max_sequence_length=dict(
            backend_package_capability(
                backend,
                "GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS_BY_MAX_SEQUENCE_LENGTH",
                {},
            )
        ),
    )


def make_qwen35_gguf_ud_q3_k_m_generator(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
) -> Qwen35GGUFBringupGenerator:
    """Select exact fully-bulk prefill and native c>N decode for UD-Q3_K_M."""

    return Qwen35GGUFBringupGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1100",
        bulk_prefill_attention_mode="bulk",
        prefill_quant="gguf_ud_q3_k_m",
        prefill_attn_aotriton_min_tokens=0,
        native_batch_decode=True,
    )


def make_qwen35_gguf_bringup_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
) -> Qwen35GGUFBringupGenerator:
    return Qwen35GGUFBringupGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1151",
    )


def make_qwen35_gguf_q4_k_m_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
) -> Qwen35GGUFBringupGenerator:
    """Create the gfx1151 Q4_K_M generator with F4-retained loop defaults."""

    backend = "hip_gfx1151"
    return Qwen35GGUFBringupGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend=backend,
        engine_loop_config_defaults={
            "prefill_decode_policy": backend_package_capability(
                backend,
                "GGUF_Q4_K_M_PREFILL_DECODE_POLICY",
                "protect_decode",
            ),
            "max_prefill_chunk_tokens": int(
                backend_package_capability(
                    backend,
                    "GGUF_Q4_K_M_MAX_PREFILL_CHUNK_TOKENS",
                    256,
                )
            ),
            "fair_prefill_burst_chunks": int(
                backend_package_capability(
                    backend,
                    "GGUF_Q4_K_M_FAIR_PREFILL_BURST_CHUNKS",
                    1,
                )
            ),
        },
        server_plain_ar_max_active_requests=int(
            backend_package_capability(
                backend,
                "GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS",
                4,
            )
        ),
    )


_GGUF_GENERATOR_FACTORIES_BY_BACKEND = {
    "hip_gfx1100": make_qwen35_gguf_bringup_generator,
    "hip_gfx1151": make_qwen35_gguf_bringup_generator_gfx1151,
}
_GGUF_GENERATOR_FACTORY_OVERRIDES = {
    ("hip_gfx1100", "gguf_ud_q3_k_m"): make_qwen35_gguf_ud_q3_k_m_generator,
    ("hip_gfx1100", "gguf_q4_k_m"): make_qwen35_gguf_q4_k_m_generator_gfx1100,
    ("hip_gfx1151", "gguf_q4_k_m"): make_qwen35_gguf_q4_k_m_generator_gfx1151,
}
for _model in ("qwen3_5_gguf", "qwen3_5_moe_gguf"):
    for _quant in (
        "gguf_q4_k_m",
        "gguf_q4_k_s",
        "gguf_q8_0",
        "gguf_q4_1",
        "gguf_ud_q4_k_xl",
        "gguf_ud_q3_k_m",
    ):
        for _backend, _default_factory in _GGUF_GENERATOR_FACTORIES_BY_BACKEND.items():
            register_text_generator(
                model=_model,
                backend=_backend,
                quant=_quant,
                factory=_GGUF_GENERATOR_FACTORY_OVERRIDES.get(
                    (_backend, _quant),
                    _default_factory,
                ),
            )


__all__ = [
    "Qwen35GGUFBringupGenerator",
    "make_qwen35_gguf_bringup_generator",
    "make_qwen35_gguf_bringup_generator_gfx1151",
    "make_qwen35_gguf_ud_q3_k_m_generator",
]
