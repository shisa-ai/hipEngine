"""Qwen3.5 GGUF generation path."""

from __future__ import annotations

import concurrent.futures
import copy
import os
import threading
import time
import uuid
import weakref
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar, Iterator, Mapping, Sequence

import numpy as np

from hipengine.core.memory import memory_stats
from hipengine.dispatch import (
    RequestState,
    SlotMove,
    WorkItem,
    WorkKind,
    plan_physical_batch_groups,
)
from hipengine.generation.batch_scheduler import CompletedRequest, GeneratedToken
from hipengine.generation.constraints import token_sequence_state_for_tokens
from hipengine.generation.deadline import raise_if_generation_deadline_expired
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
    thinking_budget_state_from_params,
)
from hipengine.loading.gguf import GGUFModelInfo, GGUFReader
from hipengine.kernels.backends import (
    backend_package_capability,
    hip_target_arch_environment,
    hip_target_arch_for_backend,
    resolve_backend,
)
from hipengine.quant.gguf import dequantize_gguf_data
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
    _rope_tables as _gguf_rope_tables,
)
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer


_GGUF_MTP_REQUIRED_TENSORS = (
    "token_embd.weight",
    "output.weight",
    "blk.40.nextn.eh_proj.weight",
    "blk.40.nextn.hnorm.weight",
    "blk.40.nextn.enorm.weight",
    "blk.40.nextn.shared_head_norm.weight",
    "blk.40.attn_norm.weight",
    "blk.40.attn_q.weight",
    "blk.40.attn_k.weight",
    "blk.40.attn_v.weight",
    "blk.40.attn_output.weight",
    "blk.40.attn_q_norm.weight",
    "blk.40.attn_k_norm.weight",
    "blk.40.post_attention_norm.weight",
    "blk.40.ffn_gate_inp.weight",
    "blk.40.ffn_gate_exps.weight",
    "blk.40.ffn_up_exps.weight",
    "blk.40.ffn_down_exps.weight",
    "blk.40.ffn_gate_inp_shexp.weight",
    "blk.40.ffn_gate_shexp.weight",
    "blk.40.ffn_up_shexp.weight",
    "blk.40.ffn_down_shexp.weight",
)


def _new_gguf_timing_batch_id(kind: str) -> str:
    return f"gguf-{str(kind)}-{uuid.uuid4().hex}"


def _encode_prompt(tokenizer: Any, prompt: PromptInput) -> list[int]:
    if not isinstance(prompt, str):
        return [int(token) for token in prompt]
    return [int(token) for token in tokenizer.encode(prompt)]


_LLAMA_COMPAT_MTP_ENV = {
    "HIPENGINE_GGUF_DECODE_REPACK": "1",
    "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1",
    "HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A": "1",
    "HIPENGINE_GGUF_T16_SELECTED_DP4A": "1",
    "HIPENGINE_GGUF_RAW_SELECTED_DP4A": "1",
    "HIPENGINE_GGUF_Q8_0_RAW_SIDECAR": "1",
    "HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL": "1",
    "HIPENGINE_GGUF_DENSE_Q8_DP4A_F32": "1",
    "HIPENGINE_GGUF_SELECTED_X8_REPACK": "q6",
    "HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A": "1",
    "HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE": "x8",
    "HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL": "1",
    "HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A": "1",
    "HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A_STAGES": "draft",
}
_GGUF_MTP_CONTEXT_REPLAY_MIN_PROMPT_TOKENS = 4
_MTP_SERVING_TARGET_BATCH_MAX_SLOTS = 4
_GGUF_AR_NATIVE_MAX_SLOTS = 8
_GGUF_RESIDENT_MODEL_LOOP_DEFAULT_CAPACITY = 4
_GGUF_AR_PHYSICAL_BUCKET_WIDTHS = (1, 2, 4, 8)
_GGUFSessionPoolKey = tuple[str, bool | None, bool | None, int | None]
_GGUF_AR_PACKED_DECODE_ENV = "HIPENGINE_GGUF_AR_PACKED_DECODE"
_GGUF_AR_PACKED_PREFILL_ENV = "HIPENGINE_GGUF_AR_PACKED_PREFILL"
_GGUF_AR_STREAM_DECODE_ENV = "HIPENGINE_GGUF_AR_STREAM_DECODE"
_GGUF_AR_STREAM_PREFILL_ENV = "HIPENGINE_GGUF_AR_STREAM_PREFILL"
_GGUF_DECODE_GRAPH_ENV = "HIPENGINE_GGUF_DECODE_GRAPH"
_GGUF_MTP_SERVER_PACKED_PREFILL_ENV = "HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL"
_GGUF_MTP_SERVER_STARTUP_WARMUP_ENV = "HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP"
_GGUF_MTP_SERVER_STREAM_DRAFT_ENV = "HIPENGINE_GGUF_MTP_SERVER_STREAM_DRAFT"
_GGUF_MTP_SERVER_STREAM_VERIFY_ENV = "HIPENGINE_GGUF_MTP_SERVER_STREAM_VERIFY"
_GGUF_MTP_SERVER_VERIFY_FINAL_STATE_FASTPATH_ENV = (
    "HIPENGINE_GGUF_MTP_SERVER_VERIFY_FINAL_STATE_FASTPATH"
)
_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER_ENV = "HIPENGINE_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER"
_GGUF_MTP_SERVER_ROLLING_SLOTS_ENV = "HIPENGINE_GGUF_MTP_SERVER_ROLLING_SLOTS"
_GGUF_PUBLIC_USE_WMMA_PREFILL = True
_GGUF_PUBLIC_USE_GEMV_DECODE = True
_MTP_SERVING_TARGET_USE_WMMA_PREFILL = False


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


def _gguf_ar_packed_prefill_enabled() -> bool:
    return os.environ.get(_GGUF_AR_PACKED_PREFILL_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_ar_stream_decode_enabled() -> bool:
    return os.environ.get(_GGUF_AR_STREAM_DECODE_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_ar_stream_prefill_enabled() -> bool:
    return os.environ.get(_GGUF_AR_STREAM_PREFILL_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_decode_graph_enabled() -> bool:
    return os.environ.get(_GGUF_DECODE_GRAPH_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_mtp_server_packed_prefill_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_PACKED_PREFILL_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_mtp_server_startup_warmup_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_STARTUP_WARMUP_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_mtp_server_stream_draft_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_STREAM_DRAFT_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_mtp_server_stream_verify_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_STREAM_VERIFY_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_mtp_server_verify_final_state_fastpath_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_VERIFY_FINAL_STATE_FASTPATH_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _gguf_mtp_server_defer_verify_scatter_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER_ENV, "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _gguf_mtp_server_rolling_slots_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_ROLLING_SLOTS_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class _GGUFMTPServingAssets:
    weights: dict[str, tuple[np.ndarray, int, tuple[int, ...]]]
    token_embd_f32: np.ndarray
    rope_cos: np.ndarray
    rope_sin: np.ndarray


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
    draft_pool_key: int | None = None
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


def _gguf_info_has_mtp_tensors(info: Any) -> bool:
    try:
        by_name = {tensor.name for tensor in info.tensors}
    except Exception:
        return False
    return all(name in by_name for name in _GGUF_MTP_REQUIRED_TENSORS)


def _timing_ms_since(start: float) -> float:
    return round(max(0.0, time.perf_counter() - start) * 1000.0, 3)


def _timing_add(timing: dict[str, float], key: str, start: float) -> None:
    timing[key] = round(float(timing.get(key, 0.0)) + _timing_ms_since(start), 3)


def _timing_add_ms(timing: dict[str, float], key: str, ms: float) -> None:
    timing[key] = round(float(timing.get(key, 0.0)) + max(0.0, float(ms)), 3)


def _timing_set(timing: dict[str, float], key: str, start: float) -> None:
    timing[key] = _timing_ms_since(start)


def _mtp_cycle_summary(cycles: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, Any]:
    direct_cycles = [
        cycle
        for cycle in cycles
        if str(cycle.get("mode", "")) == "llama_compat_direct_commit"
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
        if str(cycle.get("mode", "")) == "llama_compat_direct_commit"
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


@dataclass
class Qwen35GGUFBringupGenerator:
    """Public API GGUF greedy generator over a persistent resident session."""

    model_path: str | Path
    weight_index: GGUFModelInfo
    model_plugin: Any
    backend: str = "auto"
    engine_loop_config_defaults: Mapping[str, Any] = field(default_factory=dict, repr=False)
    tokenizer: Qwen35GGUFTokenizer = field(init=False)
    last_batch_generation: dict[str, Any] | None = field(default=None, init=False, repr=False)
    last_generation_outputs: tuple[GenerationOutput, ...] = field(default=(), init=False, repr=False)
    _mtp_serving_assets: _GGUFMTPServingAssets | None = field(default=None, init=False, repr=False)
    _mtp_serving_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _shared_runner: Qwen35GGUFFullStackRunner | None = field(default=None, init=False, repr=False)
    _shared_runner_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _prepared_max_sequence_length: int | None = field(default=None, init=False, repr=False)
    _shared_session_pool: dict[
        _GGUFSessionPoolKey,
        list[Qwen35GGUFResidentSession],
    ] = field(default_factory=dict, init=False, repr=False)
    _shared_session_pool_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _shared_mtp_draft_pool: dict[int, list[Any]] = field(default_factory=dict, init=False, repr=False)
    _shared_mtp_draft_pool_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    supports_stream_logprobs: ClassVar[bool] = True

    def __post_init__(self) -> None:
        self.backend = resolve_backend(self.backend)
        self.tokenizer = Qwen35GGUFTokenizer.from_gguf_info(self.weight_index)

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

        del sampling_params, max_new_tokens
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
            ar_widths = [width for width in (2, 4, 8) if width <= max_batch]
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
                                use_wmma_prefill=True,
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
            assets = self._load_mtp_serving_assets()
            for width in sorted(set(mtp_widths)):
                for target_len in warm_prompt_lengths:
                    sessions: list[Qwen35GGUFResidentSession] = []
                    session_keys: list[_GGUFSessionPoolKey | None] = []
                    drafts: list[Any] = []
                    draft_keys: list[int | None] = []
                    unsupported = False
                    try:
                        for _slot in range(width):
                            session, session_key, _session_reused = self._acquire_shared_session(
                                shared_runner,
                                pool_name="mtp_target",
                                use_wmma_prefill=True,
                                use_gemv_decode=True,
                            )
                            sessions.append(session)
                            session_keys.append(session_key)
                            draft, draft_key, _draft_reused = self._acquire_mtp_draft_runner(
                                assets,
                                runtime=session.runtime,
                                pool_enabled=True,
                            )
                            drafts.append(draft)
                            draft_keys.append(draft_key)
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
                                                    "use_wmma_prefill": _MTP_SERVING_TARGET_USE_WMMA_PREFILL,
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
                        for draft in drafts:
                            self._release_mtp_draft_runner(None, draft)
                        drafts = []
                        for session in sessions:
                            session.close()
                        sessions = []
                        raise
                    finally:
                        while drafts:
                            draft = drafts.pop()
                            draft_key = draft_keys.pop()
                            self._release_mtp_draft_runner(draft_key, draft)
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

        return Qwen35GGUFResidentModelRunner(
            self,
            capacity=(
                _GGUF_RESIDENT_MODEL_LOOP_DEFAULT_CAPACITY
                if capacity is None
                else int(capacity)
            ),
        )

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
                runner = Qwen35GGUFFullStackRunner(self.model_path, backend=self.backend)
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

    def _acquire_shared_session(
        self,
        shared_runner: Qwen35GGUFFullStackRunner,
        *,
        pool_name: str,
        use_wmma_prefill: bool | None = None,
        use_gemv_decode: bool | None = None,
        defer_kv_allocation: bool = False,
    ) -> tuple[Qwen35GGUFResidentSession, _GGUFSessionPoolKey, bool]:
        self._ensure_shared_pools()
        max_sequence_length = getattr(self, "_prepared_max_sequence_length", None)
        key = (str(pool_name), use_wmma_prefill, use_gemv_decode, max_sequence_length)
        with self._shared_session_pool_lock:
            pool = self._shared_session_pool.get(key)
            session = pool.pop() if pool else None
        if session is not None:
            reset = getattr(session, "reset", None)
            if callable(reset):
                reset()
            return session, key, True
        session_kwargs = (
            {}
            if max_sequence_length is None
            else {"max_sequence_length": int(max_sequence_length)}
        )
        return (
            Qwen35GGUFResidentSession(
                self.model_path,
                backend=self.backend,
                runtime=shared_runner.runtime,
                shared_runner=shared_runner,
                use_wmma_prefill=use_wmma_prefill,
                use_gemv_decode=use_gemv_decode,
                defer_kv_allocation=bool(defer_kv_allocation),
                **session_kwargs,
            ),
            key,
            False,
        )

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
            session_kwargs: dict[str, Any] = {"backend": self.backend}
            if use_wmma_prefill is not None:
                session_kwargs["use_wmma_prefill"] = bool(use_wmma_prefill)
            if use_gemv_decode is not None:
                session_kwargs["use_gemv_decode"] = bool(use_gemv_decode)
            with Qwen35GGUFResidentSession(self.model_path, **session_kwargs) as session:
                yield session, False
            return
        session, key, reused = self._acquire_shared_session(
            shared_runner,
            pool_name=pool_name,
            use_wmma_prefill=use_wmma_prefill,
            use_gemv_decode=use_gemv_decode,
        )
        try:
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

    def _release_mtp_draft_runner(self, key: int | None, draft: Any) -> None:
        self._ensure_shared_pools()
        if key is None:
            close = getattr(draft, "close", None)
            if callable(close):
                close()
            return
        with self._shared_mtp_draft_pool_lock:
            self._shared_mtp_draft_pool.setdefault(int(key), []).append(draft)

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(int(token) for token in self.tokenizer.encode(str(text)))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    @property
    def supports_speculative_mtp(self) -> bool:
        """Whether this GGUF inventory has the NextN tensors required for MTP."""

        return _gguf_info_has_mtp_tensors(self.weight_index)

    def generate(self, request: GenerationRequest) -> list[str]:
        outputs = self.generate_detailed(request)
        return [output.text for output in outputs]

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        for chunk in self.stream_detailed(request):
            yield chunk.text

    @_target_arch_scoped_stream
    def stream_detailed(self, request: GenerationRequest) -> Iterator[GenerationStreamChunk]:
        self.last_batch_generation = None
        if len(request.prompts) != 1:
            raise ValueError("streaming currently supports exactly one prompt")
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        if request.max_tokens == 0:
            return
        prompt_ids = _encode_prompt(self.tokenizer, request.prompts[0])
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
            }
            if shared_runner is not None
            else {
                "backend": self.backend,
                "use_wmma_prefill": _GGUF_PUBLIC_USE_WMMA_PREFILL,
                "use_gemv_decode": _GGUF_PUBLIC_USE_GEMV_DECODE,
            }
        )
        with Qwen35GGUFResidentSession(self.model_path, **session_kwargs) as session:
            if plan.mode is SamplingMode.GREEDY_FAST:
                yield from self._stream_greedy(session, prompt_ids, request)
                return
            yield from self._stream_sampled(
                session,
                prompt_ids,
                request,
                row_index=0,
            )

    @_target_arch_scoped
    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        plan = _gguf_sampler_plan(request)
        if request.max_tokens == 0:
            prompt_rows_by_request = {
                index: _encode_prompt(self.tokenizer, prompt)
                for index, prompt in enumerate(request.prompts)
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
                tokenize_start = time.perf_counter()
                prompt_ids = _encode_prompt(self.tokenizer, prompt)
                _timing_set(row_timing, "tokenize_ms", tokenize_start)
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
            tokenize_start = time.perf_counter()
            prompt_ids = _encode_prompt(self.tokenizer, prompt)
            tokenize_ms_by_request[row_index] = _timing_ms_since(tokenize_start)
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
                    use_wmma_prefill=True,
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
            if self._try_prefill_ar_serving_slots_streams(slots, request):
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
                return False
            raise
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
            slot.native_compact_prefill = True
            self._finish_ar_serving_slot_prefill(slot, int(getattr(result, "token_id")), request)
        return True

    def _try_prefill_ar_serving_slots_streams(
        self,
        slots: list[_GGUFARServingSlot],
        request: GenerationRequest,
    ) -> bool:
        if len(slots) <= 1 or not _gguf_ar_stream_prefill_enabled():
            return False
        for slot in slots:
            if not callable(getattr(slot.session, "prefill_async_top1", None)):
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
        batch_start = time.perf_counter()
        prefill_starts: dict[int, float] = {}
        for slot in slots:
            if slot.decode_stream == 0:
                slot.decode_stream = int(slot.session.runtime.stream_create(nonblocking=True))
            prefill_starts[int(slot.request_id)] = time.perf_counter()
            slot.session.prefill_async_top1(
                slot.prompt_ids,
                stream=int(slot.decode_stream),
            )
        for slot in slots:
            slot.session.runtime.stream_synchronize(int(slot.decode_stream))
            _timing_add(slot.timing, "prefill_ms", prefill_starts[int(slot.request_id)])
        batch_ms = _timing_ms_since(batch_start)
        for slot in slots:
            _timing_add_ms(slot.timing, "prefill_stream_batch_ms", batch_ms)
            result = slot.session.read_top1_sample()
            self._finish_ar_serving_slot_prefill(slot, int(result.token_id), request)
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
        plan = _gguf_sampler_plan(request)
        if plan.mode is not SamplingMode.GREEDY_FAST:
            raise NotImplementedError("GGUF speculative MTP currently supports only greedy-fast sampling")
        if request.max_tokens == 0:
            return self.generate_detailed(request)

        request_start = time.perf_counter()
        encoded_prompts: dict[int, list[int]] = {}
        tokenize_ms_by_request: dict[int, float] = {}
        for row_index, prompt in enumerate(request.prompts):
            tokenize_start = time.perf_counter()
            encoded_prompts[row_index] = _encode_prompt(self.tokenizer, prompt)
            tokenize_ms_by_request[row_index] = _timing_ms_since(tokenize_start)
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
                    use_wmma_prefill=True,
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
        if (
            len(request.prompts) > _MTP_SERVING_TARGET_BATCH_MAX_SLOTS
            and _gguf_mtp_server_rolling_slots_enabled()
        ):
            return self._generate_rolling_mtp_serving_slots(
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
                pool_sessions=pool_sessions,
                outputs=outputs,
            )

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

    def _generate_rolling_mtp_serving_slots(
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
        max_active_slots = min(_MTP_SERVING_TARGET_BATCH_MAX_SLOTS, len(request.prompts))
        pending_request_ids = list(range(len(request.prompts)))
        active_slots: list[_GGUFMTPServingSlot] = []
        completed_slots: list[_GGUFMTPServingSlot] = []
        deferred_close_slots: list[_GGUFMTPServingSlot] = []
        verify_owner_session: Qwen35GGUFResidentSession | None = None
        slots_open_ms = 0.0

        def open_more_slots() -> None:
            nonlocal slots_open_ms
            if len(active_slots) >= max_active_slots or not pending_request_ids:
                return
            capacity = max_active_slots - len(active_slots)
            take = min(capacity, len(pending_request_ids))
            if take == 3:
                take = 2
            if take == 1 and active_slots:
                return
            request_ids = [pending_request_ids.pop(0) for _ in range(take)]
            sub_request = replace(
                request,
                prompts=tuple(request.prompts[request_id] for request_id in request_ids),
            )
            sub_encoded = {
                local_index: encoded_prompts[request_id]
                for local_index, request_id in enumerate(request_ids)
            }
            open_start = time.perf_counter()
            new_slots = self._open_mtp_serving_slots(
                shared_runner,
                assets,
                sub_encoded,
                sub_request,
                pool_sessions=pool_sessions,
            )
            open_ms = _timing_ms_since(open_start)
            slots_open_ms += open_ms
            if len(new_slots) != len(request_ids):
                self._close_mtp_serving_slots(new_slots, reuse=False)
                raise RuntimeError(
                    f"rolling MTP opened {len(new_slots)} slot(s) for {len(request_ids)} request(s)"
                )
            for slot, request_id in zip(new_slots, request_ids, strict=True):
                slot.request_id = int(request_id)
                _timing_add_ms(slot.timing, "rolling_slot_open_batch_ms", open_ms)
            active_slots.extend(new_slots)

        slots_run_start = time.perf_counter()
        try:
            open_more_slots()
            if active_slots:
                verify_owner_session = active_slots[0].session
            while active_slots:
                if any(not slot.done for slot in active_slots):
                    self._run_mtp_serving_slots_cycle(
                        active_slots,
                        assets,
                        request,
                        base_env=base_env,
                        verify_owner_session=verify_owner_session,
                    )
                still_active: list[_GGUFMTPServingSlot] = []
                finished_now: list[_GGUFMTPServingSlot] = []
                for slot in active_slots:
                    if slot.done:
                        finished_now.append(slot)
                    else:
                        still_active.append(slot)
                active_slots = still_active
                for slot in finished_now:
                    completed_slots.append(slot)
                    if verify_owner_session is not None and slot.session is verify_owner_session:
                        deferred_close_slots.append(slot)
                    else:
                        self._close_mtp_serving_slots([slot], reuse=True)
                open_more_slots()
        except Exception:
            self._close_mtp_serving_slots(active_slots, reuse=False)
            self._close_mtp_serving_slots(deferred_close_slots, reuse=False)
            raise
        finally:
            if deferred_close_slots:
                self._close_mtp_serving_slots(deferred_close_slots, reuse=True)

        slots_run_ms = _timing_ms_since(slots_run_start)
        target_verify_batching = (
            "packed_slot_batch"
            if any("target_verify_batch_ms" in slot.timing for slot in completed_slots)
            else "per_slot_serial"
        )
        for slot in sorted(completed_slots, key=lambda item: int(item.request_id)):
            prompt_rows_by_request[slot.request_id] = list(slot.prompt_ids)
            generated_ids = list(slot.generated_ids)
            generated_ids_by_request[slot.request_id] = generated_ids
            mtp_cycles_by_request[slot.request_id] = list(slot.cycles)
            timing = dict(slot.timing)
            timing["tokenize_ms"] = tokenize_ms_by_request.get(slot.request_id, 0.0)
            timing["assets_load_ms"] = assets_load_ms
            timing["slots_open_ms"] = slots_open_ms
            timing["slots_run_ms"] = slots_run_ms
            timing["rolling_max_active_slots"] = float(max_active_slots)
            _add_mtp_cycle_timing_metrics(timing, slot.cycles)
            outputs.append(
                self._mtp_generation_output(
                    slot.prompt_ids,
                    generated_ids,
                    request,
                    row_index=slot.request_id,
                    resident_slot_count=max_active_slots,
                    timing=timing,
                )
            )
        return max_active_slots, target_verify_batching

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
        weights: dict[str, tuple[np.ndarray, int, tuple[int, ...]]] = {}
        required = set(_GGUF_MTP_REQUIRED_TENSORS)
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
        meta = reader.info.metadata
        rope_dim = int(meta.get("qwen35moe.rope.dimension_count", 64))
        rope_base = float(meta.get("qwen35moe.rope.freq_base", 10000000.0))
        rope_cos, rope_sin = _gguf_rope_tables(
            max_positions=262144,
            rotary_dim=rope_dim,
            base=rope_base,
        )
        assets = _GGUFMTPServingAssets(
            weights=weights,
            token_embd_f32=np.ascontiguousarray(token_embd_f32, dtype=np.float32),
            rope_cos=rope_cos,
            rope_sin=rope_sin,
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
                    use_wmma_prefill=True,
                    use_gemv_decode=True,
                ) if pool_sessions else (
                    Qwen35GGUFResidentSession(
                        self.model_path,
                        runtime=shared_runner.runtime,
                        shared_runner=shared_runner,
                        use_wmma_prefill=True,
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

            slots: list[_GGUFMTPServingSlot] = []
            hidden_size = int(assets.token_embd_f32.shape[1])
            qk_head_dim = int(np.asarray(assets.weights["blk.40.attn_q_norm.weight"][0]).shape[0])
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
                use_wmma_prefill=True,
                use_gemv_decode=True,
            ) if pool_sessions else (
                Qwen35GGUFResidentSession(
                    self.model_path,
                    runtime=shared_runner.runtime,
                    shared_runner=shared_runner,
                    use_wmma_prefill=True,
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
            qk_head_dim = int(np.asarray(assets.weights["blk.40.attn_q_norm.weight"][0]).shape[0])
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
        final_state_fastpath = _gguf_mtp_server_verify_final_state_fastpath_enabled()
        for drafted in drafted_cycles:
            slot = drafted.slot
            snapshot_start = time.perf_counter()
            snapshot = (
                slot.session._linear_state_snapshot()
                if final_state_fastpath or not drafted.direct_commit_exact
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
                        use_wmma_prefill=_MTP_SERVING_TARGET_USE_WMMA_PREFILL,
                        capture_linear_state_rows=not final_state_fastpath,
                        defer_linear_state_commit=not final_state_fastpath,
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
        final_state_fastpath = _gguf_mtp_server_verify_final_state_fastpath_enabled()
        defer_state_scatter = (
            _gguf_mtp_server_defer_verify_scatter_enabled()
            and not final_state_fastpath
        )
        jobs = [
            {
                "session": drafted.slot.session,
                "input_token_ids": tuple(int(token) for token in drafted.block_inputs),
                "bulk_attention_mode": "bulk",
                "use_wmma_prefill": _MTP_SERVING_TARGET_USE_WMMA_PREFILL,
                "capture_linear_state_rows": not final_state_fastpath,
                "defer_linear_state_commit": not final_state_fastpath,
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
            captured_rows = bool(getattr(verified.block_result, "linear_state_rows_captured", False))
            final_state_committed = bool(getattr(verified.block_result, "final_linear_state_committed", False))
            deferred_packed_state = getattr(verified.block_result, "deferred_packed_state", None)
            seed_row_count = (
                consumed_rows
                if deferred_packed_state is not None
                else len(block_target_tokens)
            )
            if consumed_rows < len(block_inputs):
                if captured_rows:
                    if deferred_packed_state is not None:
                        owner = getattr(deferred_packed_state, "owner", None)
                        commit_deferred = getattr(owner, "_commit_deferred_packed_verify_state", None)
                        if not callable(commit_deferred):
                            raise RuntimeError("deferred packed verifier state owner cannot commit rows")
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
                elif final_state_committed:
                    if snapshot is None:
                        raise RuntimeError("MTP partial fastpath replay requested without a linear-state snapshot")
                    slot.session._restore_linear_state_snapshot(snapshot, position=block_start)
                    slot.session.verify_target_block(
                        block_inputs[:consumed_rows],
                        bulk_attention_mode="bulk",
                        use_wmma_prefill=_MTP_SERVING_TARGET_USE_WMMA_PREFILL,
                        advance_state_only=True,
                    )
                else:
                    raise RuntimeError("direct MTP partial commit requested without captured or final state")
            elif drafted.direct_commit_exact:
                if captured_rows:
                    if deferred_packed_state is not None:
                        owner = getattr(deferred_packed_state, "owner", None)
                        commit_deferred = getattr(owner, "_commit_deferred_packed_verify_state", None)
                        if not callable(commit_deferred):
                            raise RuntimeError("deferred packed verifier state owner cannot commit rows")
                        commit_deferred(
                            deferred_packed_state,
                            slot.session,
                            commit_row_index=len(block_inputs) - 1,
                            position=block_start + len(block_inputs),
                            hidden_rows=seed_row_count,
                        )
                    else:
                        slot.session._commit_verify_linear_state_row(
                            len(block_inputs) - 1,
                            position=block_start + len(block_inputs),
                        )
                elif not final_state_committed:
                    raise RuntimeError("direct MTP full commit requested without captured or final state")
            else:
                if snapshot is None:
                    raise RuntimeError("MTP full-block replay requested without a linear-state snapshot")
                slot.session._restore_linear_state_snapshot(snapshot, position=block_start)
                replay_result = slot.session.verify_target_block_serial_exact(block_inputs)
                replay_tokens = [int(token) for token in replay_result.token_ids]
                if replay_tokens != block_target_tokens:
                    raise RuntimeError("MTP serial-exact replay diverged from block verifier rows")
            _timing_add(slot.timing, "target_state_commit_ms", state_commit_start)
            if (
                consumed_rows < len(block_inputs)
                and final_state_committed
                and not captured_rows
            ):
                seed_row_count = consumed_rows
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
        qk_head_dim = int(np.asarray(assets.weights["blk.40.attn_q_norm.weight"][0]).shape[0])
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
                final_state_fastpath = _gguf_mtp_server_verify_final_state_fastpath_enabled()
                if final_state_fastpath and snapshot is None:
                    snapshot = session._linear_state_snapshot()
                if snapshot is not None:
                    _timing_add(timing, "linear_state_snapshot_ms", snapshot_start)
                try:
                    verify_start = time.perf_counter()
                    block_result = session.verify_target_block(
                        block_inputs,
                        bulk_attention_mode="bulk",
                        use_wmma_prefill=_MTP_SERVING_TARGET_USE_WMMA_PREFILL,
                        capture_linear_state_rows=not final_state_fastpath,
                        defer_linear_state_commit=not final_state_fastpath,
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
                    captured_rows = bool(getattr(block_result, "linear_state_rows_captured", False))
                    final_state_committed = bool(getattr(block_result, "final_linear_state_committed", False))
                    if consumed_rows < len(block_inputs):
                        if captured_rows:
                            session._commit_verify_linear_state_row(
                                consumed_rows - 1,
                                position=block_start + consumed_rows,
                            )
                        elif final_state_committed:
                            if snapshot is None:
                                raise RuntimeError("MTP partial fastpath replay requested without a linear-state snapshot")
                            session._restore_linear_state_snapshot(snapshot, position=block_start)
                            session.verify_target_block(
                                block_inputs[:consumed_rows],
                                bulk_attention_mode="bulk",
                                use_wmma_prefill=_MTP_SERVING_TARGET_USE_WMMA_PREFILL,
                                advance_state_only=True,
                            )
                        else:
                            raise RuntimeError("direct MTP partial commit requested without captured or final state")
                    elif direct_commit_exact:
                        if captured_rows:
                            session._commit_verify_linear_state_row(
                                len(block_inputs) - 1,
                                position=block_start + len(block_inputs),
                            )
                        elif not final_state_committed:
                            raise RuntimeError("direct MTP full commit requested without captured or final state")
                    else:
                        if snapshot is None:
                            raise RuntimeError("MTP full-block replay requested without a linear-state snapshot")
                        session._restore_linear_state_snapshot(snapshot, position=block_start)
                        replay_result = session.verify_target_block_serial_exact(block_inputs)
                        replay_tokens = [int(token) for token in replay_result.token_ids]
                        if replay_tokens != block_target_tokens:
                            raise RuntimeError("MTP serial-exact replay diverged from block verifier rows")
                    _timing_add(timing, "target_state_commit_ms", state_commit_start)
                    seed_row_count = len(block_target_tokens)
                    if (
                        consumed_rows < len(block_inputs)
                        and final_state_committed
                        and not captured_rows
                    ):
                        seed_row_count = consumed_rows
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

    def _generate_sampled(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
        *,
        row_index: int,
    ) -> GenerationOutput:
        sampling_request = _request_with_tokenizer_eos(request, self.tokenizer)
        state = _gguf_row_sampling_state(sampling_request, prompt_ids, row_index=row_index)
        samples = []
        raise_if_generation_deadline_expired(request)
        result = session.prefill(prompt_ids, return_logits=True)
        raise_if_generation_deadline_expired(request)
        full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(result)
        sample = _select_from_gguf_logits(result, sampling_request, state)
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
            sample = _select_from_gguf_logits(step, sampling_request, state)
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
            ),
        )

    def _stream_greedy(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamChunk]:
        generated_ids: list[int] = []
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
    ) -> Iterator[GenerationStreamChunk]:
        sampling_request = _request_with_tokenizer_eos(request, self.tokenizer)
        state = _gguf_row_sampling_state(sampling_request, prompt_ids, row_index=row_index)
        generated_ids: list[int] = []
        live_phase = None if state.thinking_budget is not None else "answer"
        raise_if_generation_deadline_expired(request)
        result = session.prefill(prompt_ids, return_logits=True)
        raise_if_generation_deadline_expired(request)
        full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(result)
        sample = _select_from_gguf_logits(result, sampling_request, state)
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
            ),
        )
        if finished:
            return
        for _ in range(request.max_tokens - 1):
            raise_if_generation_deadline_expired(request)
            step = session.step(generated_ids[-1], return_logits=True)
            raise_if_generation_deadline_expired(request)
            full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(step)
            sample = _select_from_gguf_logits(step, sampling_request, state)
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
                ),
            )
            if finished:
                return


@dataclass(frozen=True, slots=True)
class _GGUFResidentSessionLease:
    session: Qwen35GGUFResidentSession
    pool_key: _GGUFSessionPoolKey


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
    prefill_tokens_seen: int = 0
    incremental_prefill: bool | None = None
    prefill_chunk_count: int = 0
    prefill_ms: float = 0.0
    lease: _GGUFResidentSessionLease | None = None
    slot: _GGUFARServingSlot | None = None
    first_token_emitted: bool = False
    fallback_output: GenerationOutput | None = None
    kv_allocation: Any | None = None
    sampling_request: GenerationRequest | None = None
    sampling_state: RowSamplingState | None = None
    samples: list[Any] = field(default_factory=list)
    full_vocab_logits_d2h: bool | None = None
    logits_d2h_bytes: int | None = None


class Qwen35GGUFResidentModelRunner:
    """Long-lived scheduler-facing owner of GGUF model and session state.

    The owner reserves a fixed session pool once, keeps stable request identity
    separate from scheduler physical slots, and exposes one committed prefill or
    decode transition per engine-loop hook.  Greedy and host-sampled c>1 decode
    use the retained packed session primitive; host sampling remains explicit
    in telemetry without being mislabeled as a serial model-step fallback.
    """

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
        self._shared_runner = generator._get_shared_runner()
        self._max_sequence_length = getattr(generator, "_prepared_max_sequence_length", None)
        self._available: list[_GGUFResidentSessionLease] = []
        self._rows: dict[int, _GGUFResidentLoopRow] = {}
        self._outputs: dict[int, GenerationOutput] = {}
        self._completed_metadata: dict[int, dict[str, Any]] = {}
        self._next_batch_id = 0
        self._kv_pool: Any | None = None
        self._engine_loop_config: Any | None = None
        self._kv_hip_used_peak_sampled_bytes = 0
        self._kv_graph_invalidation_count = 0
        self._route_counts: Counter[str] = Counter()
        self._fallback_reasons: Counter[str] = Counter()
        self._last_execution_manifest: dict[str, Any] = {}
        self._last_physical_group_plan: dict[str, Any] = {}
        self._recent_completed_routes: deque[dict[str, Any]] = deque(maxlen=1024)
        self._graph_handle_refs: dict[int, weakref.ReferenceType[Any]] = {}
        self._graph_handle_buckets: dict[int, str] = {}
        self._graph_handle_replays: dict[int, int] = {}
        self._graph_buckets: dict[str, dict[str, Any]] = {}
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
        pool_stats = None if pool is None else pool.stats.to_json_dict()
        active_entries: Counter[str] = Counter()
        for handle in self._graph_handles_for_sessions(sessions):
            if bool(getattr(handle, "closed", False)):
                continue
            label = self._graph_bucket_label(handle)
            active_entries[label] += 1
        buckets = copy.deepcopy(self._graph_buckets)
        for label, row in buckets.items():
            row["entries"] = int(active_entries.get(label, 0))
        return {
            "model_runner": {
                "capacity": int(self.capacity),
                "active_request_ids": list(self.active_request_ids),
                "active_requests": len(self._rows),
                "available_sessions": len(self._available),
            },
            "kv_pool": pool_stats,
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
                    "native_incremental_prefill_chunks": int(
                        self._route_counts["native_incremental_prefill_chunks"]
                    ),
                    "native_packed_decode_steps": int(
                        self._route_counts["native_packed_decode_steps"]
                    ),
                    "native_c1_decode_steps": int(self._route_counts["native_c1_decode_steps"]),
                    "native_sampled_prefill_rows": int(
                        self._route_counts["native_sampled_prefill_rows"]
                    ),
                    "host_sampler_requests": int(
                        self._route_counts["host_sampler_requests"]
                    ),
                    "serial_decode_fallback_steps": int(
                        self._route_counts["serial_decode_fallback_steps"]
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
                    for width in _GGUF_AR_PHYSICAL_BUCKET_WIDTHS
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
        tracked = memory_stats()
        return {
            "dynamic_pool": pool_stats,
            "tracked_allocator": tracked,
            "hip_used_current_bytes": self._current_hip_used_bytes(),
            "hip_used_peak_sampled_bytes": int(self._kv_hip_used_peak_sampled_bytes),
            "graph_invalidation_count": int(self._kv_graph_invalidation_count),
        }

    def configure_engine_loop(self, config: Any) -> None:
        """Bind engine-loop KV policy knobs to the real deferred session pool."""

        if self._rows:
            raise RuntimeError("cannot configure GGUF device KV pool while requests are active")
        self._engine_loop_config = config
        factory_session = self._available[-1].session if self._available else None
        create_pool = getattr(factory_session, "create_device_kv_pool", None)
        if not callable(create_pool):
            # Lightweight fake-session tests retain the D2 fixed-session path.
            return
        if self._kv_pool is not None:
            self._kv_pool.close()
            self._kv_pool = None
        scratch = getattr(factory_session, "scratch", None)
        if scratch is None:
            raise RuntimeError("GGUF deferred session has no scratch capacity")
        max_pages_per_request = max(1, (int(scratch.max_positions) + 255) // 256)
        total_pages = self.capacity * max_pages_per_request
        initial_pages = min(int(config.kv_pool_initial_pages), total_pages)
        low_water_pages = min(int(config.kv_pool_low_water_pages), initial_pages)
        requested_high = getattr(config, "kv_pool_high_water_pages", None)
        high_water_pages = None if requested_high is None else int(requested_high)
        chunk_pages = min(max(1, int(config.kv_pool_chunk_pages)), total_pages)
        self._kv_pool = create_pool(
            initial_pages=initial_pages,
            low_water_pages=low_water_pages,
            high_water_pages=high_water_pages,
            chunk_pages=chunk_pages,
            idle_grace_seconds=float(config.kv_pool_idle_grace_seconds),
        )
        self._sample_kv_hip_memory()

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
        allocation = pool.allocate(row.request_id, pages, now_seconds=time.monotonic())
        try:
            lease.session.bind_device_kv_allocation(pool, allocation)
        except Exception:
            pool.release(row.request_id, now_seconds=time.monotonic())
            raise
        if not self._available or self._available[-1] is not lease:
            lease.session.invalidate_device_kv_graphs()
            lease.session.unbind_device_kv_allocation()
            pool.release(row.request_id, now_seconds=time.monotonic())
            raise RuntimeError("GGUF available-session order changed during atomic admission")
        self._available.pop()
        row.lease = lease
        row.kv_allocation = allocation
        self._sample_kv_hip_memory()

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

    def scheduler_max_new_tokens(self, request: GenerationRequest) -> int:
        if int(request.max_tokens) > 0:
            return int(request.max_tokens)
        # Zero-token requests execute as one declared resident compatibility
        # transition so the scheduler can publish an empty completed output.
        return 1

    def register_batch(
        self,
        request_ids: Sequence[int],
        request: GenerationRequest,
        *,
        prompt_rows: Sequence[Sequence[int]],
    ) -> None:
        ids = tuple(int(request_id) for request_id in request_ids)
        prompts = tuple(tuple(int(token) for token in row) for row in prompt_rows)
        if len(ids) != len(request.prompts) or len(prompts) != len(ids):
            raise ValueError("request_ids, prompts, and prompt_rows must have the same length")
        if request.row_seeds and len(request.row_seeds) != len(request.prompts):
            raise ValueError("row_seeds must have one entry per prompt")
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        plan = _gguf_sampler_plan(request)
        native_greedy = (
            plan.mode is SamplingMode.GREEDY_FAST
            and int(request.max_tokens) > 0
        )
        native_sampled = (
            plan.mode is not SamplingMode.GREEDY_FAST
            and int(request.max_tokens) > 0
        )
        now = time.perf_counter()
        for row_index, (request_id, prompt_ids) in enumerate(zip(ids, prompts, strict=True)):
            if request_id in self._rows or request_id in self._outputs:
                raise ValueError(f"request_id {request_id} is already registered")
            if not prompt_ids:
                raise ValueError("GGUF prompt tokenization produced no token IDs")
            self._rows[request_id] = _GGUFResidentLoopRow(
                request_id=request_id,
                batch_id=batch_id,
                row_index=row_index,
                request=request,
                prompt_ids=prompt_ids,
                native_greedy=native_greedy,
                native_sampled=native_sampled,
                submitted_at=now,
            )

    def prepare(self, *, max_sequence_length: int | None = None) -> None:
        requested = getattr(self.generator, "_prepared_max_sequence_length", None)
        if requested is None and max_sequence_length is not None:
            requested = int(max_sequence_length)
        if requested == self._max_sequence_length:
            return
        if self._rows:
            raise RuntimeError("cannot resize resident GGUF sessions while requests are active")
        with hip_target_arch_environment(self.generator.target_arch):
            config = self._engine_loop_config
            if self._kv_pool is not None:
                self._kv_pool.close()
                self._kv_pool = None
            self._release_available_sessions()
            self._max_sequence_length = requested
            self._reserve_sessions()
            if config is not None:
                self.configure_engine_loop(config)

    def prefill_batch(self, work: WorkItem, *, commit: bool) -> None:
        if not commit:
            raise ValueError("GGUF resident prefill requires commit=True")
        with hip_target_arch_environment(self.generator.target_arch):
            for request_id, token_row in zip(work.request_ids, work.token_rows, strict=True):
                row = self._row(request_id)
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
                if row.native_greedy:
                    if row.incremental_prefill is None:
                        row.incremental_prefill = not (start == 0 and final_chunk)
                    raise_if_generation_deadline_expired(row.request)
                    if row.incremental_prefill:
                        self._prefill_native_chunk(row, chunk, final_chunk=final_chunk)
                    elif final_chunk:
                        self._prefill_native_row(row)
                    raise_if_generation_deadline_expired(row.request)
                elif final_chunk:
                    raise_if_generation_deadline_expired(row.request)
                    if row.native_sampled:
                        self._prefill_sampled_row(row)
                    else:
                        self._run_resident_fallback(row)
                    raise_if_generation_deadline_expired(row.request)

    def decode_batch(self, work: WorkItem, *, commit: bool) -> tuple[GeneratedToken, ...]:
        if not commit:
            raise ValueError("GGUF resident decode requires commit=True")
        request_ids = tuple(int(request_id) for request_id in work.request_ids)
        with hip_target_arch_environment(self.generator.target_arch):
            rows = [self._row(request_id) for request_id in request_ids]
            for row in rows:
                raise_if_generation_deadline_expired(row.request)
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
                self._step_native_rows(step_rows, work=step_work)
            for row in rows:
                raise_if_generation_deadline_expired(row.request)

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
            self._release_row_resources(row)
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
        any_native_sampled = any(bool(item.get("native_sampled", False)) for item in metadata)
        serial_fallback = any(bool(item.get("serial_decode_fallback", False)) for item in metadata)
        self.generator.last_generation_outputs = output_tuple
        self.generator.last_batch_generation = _gguf_last_batch_generation(
            self.generator.tokenizer,
            request,
            _gguf_sampler_plan(request),
            prompt_rows,
            generated_rows,
            {index: list(output.token_logprobs) for index, output in enumerate(output_tuple)},
            outputs=output_tuple,
            execution_path=(
                (
                    "gguf_packed_ar_host_sampler_decode"
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
        )

    def close(self) -> None:
        with hip_target_arch_environment(self.generator.target_arch):
            self._flush_all_packed_owners()
            for row in tuple(self._rows.values()):
                self._release_row_resources(row)
            self._rows.clear()
            if self._kv_pool is not None:
                self._kv_pool.close()
                self._kv_pool = None
            self._release_available_sessions()

    def _reserve_sessions(self) -> None:
        acquired: list[_GGUFResidentSessionLease] = []
        try:
            for _ in range(self.capacity):
                session, pool_key, _reused = self.generator._acquire_shared_session(
                    self._shared_runner,
                    pool_name="continuous_ar_dynamic_kv",
                    use_wmma_prefill=True,
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
        while self._available:
            lease = self._available.pop()
            self.generator._release_shared_session(lease.pool_key, lease.session)

    def _release_row_resources(self, row: _GGUFResidentLoopRow) -> None:
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

    def _acquire_lease(self) -> _GGUFResidentSessionLease:
        if not self._available:
            raise RuntimeError("GGUF resident model runner has no free session")
        return self._available.pop()

    def _row(self, request_id: int) -> _GGUFResidentLoopRow:
        rid = int(request_id)
        if rid not in self._rows:
            raise KeyError(f"request_id {rid} is not registered with the GGUF resident runner")
        return self._rows[rid]

    def _prefill_native_row(self, row: _GGUFResidentLoopRow) -> None:
        if row.slot is not None:
            return
        lease = row.lease or self._acquire_lease()
        row.lease = lease
        start = time.perf_counter()
        result = lease.session.prefill(row.prompt_ids, return_logits=False)
        self._route_counts["native_full_prefill_rows"] += 1
        row.prefill_ms += _timing_ms_since(start)
        row.prefill_chunk_count += 1
        self._finish_native_prefill(row, result, native_compact_prefill=False)

    def _prefill_sampled_row(self, row: _GGUFResidentLoopRow) -> None:
        if row.slot is not None:
            return
        lease = row.lease or self._acquire_lease()
        row.lease = lease
        sampling_request = _request_with_tokenizer_eos(
            row.request,
            self.generator.tokenizer,
        )
        sampling_state = _gguf_row_sampling_state(
            sampling_request,
            list(row.prompt_ids),
            row_index=row.row_index,
        )
        start = time.perf_counter()
        result = lease.session.prefill(row.prompt_ids, return_logits=True)
        sample = _select_from_gguf_logits(result, sampling_request, sampling_state)
        full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(result)
        token = int(sample.token_id)
        _gguf_queue_json_object_close_if_needed(
            sampling_state,
            self.generator.tokenizer,
            _gguf_token_text(self.generator.tokenizer, sample),
            remaining_tokens=max(0, int(sampling_request.max_tokens) - 1),
        )
        self._route_counts["native_sampled_prefill_rows"] += 1
        self._route_counts["host_sampler_requests"] += 1
        plan = _gguf_sampler_plan(sampling_request)
        self._fallback_reasons[str(plan.fallback_reason or plan.mode.value)] += 1
        row.sampling_request = sampling_request
        row.sampling_state = sampling_state
        row.samples.append(sample)
        row.full_vocab_logits_d2h = full_vocab_logits_d2h
        row.logits_d2h_bytes = logits_d2h_bytes
        prefill_ms = _timing_ms_since(start)
        row.prefill_ms += prefill_ms
        row.prefill_chunk_count += 1
        row.slot = _GGUFARServingSlot(
            request_id=row.request_id,
            prompt_ids=list(row.prompt_ids),
            session=lease.session,
            prev_token=token,
            seq_position=int(lease.session.position),
            generated_ids=[token],
            timing={
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
        )

    def _prefill_native_chunk(
        self,
        row: _GGUFResidentLoopRow,
        chunk: tuple[int, ...],
        *,
        final_chunk: bool,
    ) -> None:
        if row.slot is not None:
            raise RuntimeError("GGUF resident row was prefilled more than once")
        lease = row.lease or self._acquire_lease()
        row.lease = lease
        prefill_batch = getattr(lease.session, "prefill_batch_native", None)
        if not callable(prefill_batch):
            self._disable_incremental_prefill(row, final_chunk=final_chunk)
            return
        start = time.perf_counter()
        try:
            with _temporary_env({"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}):
                results = prefill_batch(
                    [chunk],
                    sessions=[lease.session],
                    full_prompt_lengths=[len(row.prompt_ids)],
                    return_logits=False,
                    return_hidden_seeds=False,
                )
        except NotImplementedError:
            self._disable_incremental_prefill(row, final_chunk=final_chunk)
            return
        result_list = [] if results is None else list(results)
        if len(result_list) != 1:
            raise RuntimeError(
                f"GGUF incremental prefill returned {len(result_list)} result(s) for one row"
            )
        self._route_counts["native_incremental_prefill_chunks"] += 1
        row.prefill_ms += _timing_ms_since(start)
        row.prefill_chunk_count += 1
        if final_chunk:
            self._finish_native_prefill(
                row,
                result_list[0],
                native_compact_prefill=True,
            )

    def _disable_incremental_prefill(
        self,
        row: _GGUFResidentLoopRow,
        *,
        final_chunk: bool,
    ) -> None:
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
        timing = {
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

        groups = plan_physical_batch_groups(
            work,
            physical_bucket_widths=_GGUF_AR_PHYSICAL_BUCKET_WIDTHS,
            compact_active_rows=True,
        )
        row_by_request = {int(row.request_id): row for row in row_list}
        group_payloads: list[dict[str, Any]] = []
        for group in groups:
            group_rows = [row_by_request[request_id] for request_id in group.request_ids]
            packed = False
            if _gguf_ar_packed_decode_enabled() and (
                group.active_rows > 1 or group.physical_rows > 1
            ):
                packed = self._step_native_chunk(
                    group_rows,
                    physical_rows=group.physical_rows,
                    active_slot_indices=group.active_slot_indices,
                )
            if packed:
                execution_path = "packed_native"
            else:
                self._step_native_serial(group_rows)
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
                    self._last_execution_manifest = {
                        "schema": 1,
                        "kind": "gguf_ar_serial_fallback_execution_manifest",
                        "mode": "serial_fallback",
                        "rows": group.active_rows,
                        "physical_rows": group.physical_rows,
                        "active_rows": group.active_rows,
                        "active_mask": list(group.active_mask),
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
            group_payloads.append(group_payload)

        self._last_physical_group_plan = {
            "schema": 1,
            "kind": "gguf_ar_physical_group_plan",
            "logical_c": len(request_ids),
            "physical_bucket_widths": list(_GGUF_AR_PHYSICAL_BUCKET_WIDTHS),
            "policy": "occupancy_adaptive_dense_execution",
            "group_count": len(groups),
            "groups": group_payloads,
        }

    def _step_native_chunk(
        self,
        rows: Sequence[_GGUFResidentLoopRow],
        *,
        physical_rows: int | None = None,
        active_slot_indices: Sequence[int] = (),
    ) -> bool:
        for row in rows:
            self._close_c1_decode_graph(row)
        slots = [row.slot for row in rows]
        if any(slot is None for slot in slots):
            raise RuntimeError("GGUF resident packed decode row is missing its session slot")
        concrete = [slot for slot in slots if slot is not None]
        owner_slot = concrete[0]
        step_batch = getattr(owner_slot.session, "step_batch_native", None)
        if not callable(step_batch):
            return False
        self.generator._flush_ar_packed_decode_owners_if_chunk_changed(concrete)
        return_logits = any(row.native_sampled for row in rows)
        batch_kwargs: dict[str, Any] = {
            "sessions": [slot.session for slot in concrete],
            "positions": [int(slot.seq_position) for slot in concrete],
            "return_logits": return_logits,
            "scatter_state": False,
        }
        if physical_rows is not None:
            batch_kwargs.update(
                {
                    "physical_rows": int(physical_rows),
                    "active_slot_indices": tuple(int(index) for index in active_slot_indices),
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
        owner = owner_slot.session
        self._route_counts["native_packed_decode_steps"] += 1
        self._route_counts[f"native_c{int(physical_rows or len(concrete))}_decode_steps"] += 1
        manifest = getattr(owner, "last_packed_execution_manifest", None)
        if isinstance(manifest, Mapping):
            self._last_execution_manifest = copy.deepcopy(dict(manifest))
        for row, slot, result in zip(rows, concrete, result_list, strict=True):
            if row.native_sampled:
                self._record_sampled_result(row, result)
            else:
                self._record_native_token(row, int(getattr(result, "token_id")))
            slot.packed_decode_owner = owner
            slot.native_decode_steps += 1
        return True

    def _step_native_serial(self, rows: Sequence[_GGUFResidentLoopRow]) -> None:
        self._flush_rows(rows)
        native_c1 = len(rows) == 1
        if native_c1:
            self._route_counts["native_c1_decode_steps"] += 1
        else:
            self._route_counts["serial_decode_fallback_steps"] += 1
            self._fallback_reasons["packed_decode_unavailable"] += 1
        for row in rows:
            slot = row.slot
            if slot is None:
                raise RuntimeError("GGUF resident serial decode row is missing its session slot")
            result = (
                self._step_native_c1_graph(row)
                if native_c1 and row.native_greedy
                else slot.session.step(
                    int(slot.prev_token),
                    return_logits=bool(row.native_sampled),
                )
            )
            if row.native_sampled:
                self._record_sampled_result(row, result)
            else:
                self._record_native_token(row, int(getattr(result, "token_id")))
            if native_c1:
                slot.native_c1_decode_steps += 1
            else:
                slot.serial_decode_steps += 1

    def _step_native_c1_graph(self, row: _GGUFResidentLoopRow) -> Any:
        slot = row.slot
        if slot is None:
            raise RuntimeError("GGUF resident c1 decode row is missing its session slot")
        graph = slot.c1_decode_graph
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
                )
                slot.c1_decode_graph = graph
        if graph is None:
            return slot.session.step(int(slot.prev_token), return_logits=False)
        graph.replay(1)
        return graph.read_sample(return_logits=False)

    def _close_c1_decode_graph(self, row: _GGUFResidentLoopRow) -> None:
        slot = row.slot
        if slot is None or slot.c1_decode_graph is None:
            return
        graph = slot.c1_decode_graph
        lease = row.lease
        if lease is not None:
            self._observe_graph_handles((lease.session,))
        graph.close()
        slot.c1_decode_graph = None

    def _record_sampled_result(self, row: _GGUFResidentLoopRow, result: Any) -> None:
        sampling_request = row.sampling_request
        sampling_state = row.sampling_state
        if sampling_request is None or sampling_state is None:
            raise RuntimeError("GGUF sampled row has no sampling request/state")
        sample = _select_from_gguf_logits(result, sampling_request, sampling_state)
        row.samples.append(sample)
        full_vocab_logits_d2h, logits_d2h_bytes = _gguf_logits_d2h_metadata(result)
        if full_vocab_logits_d2h is not None:
            row.full_vocab_logits_d2h = full_vocab_logits_d2h
            row.logits_d2h_bytes = logits_d2h_bytes
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
        slots = [row.slot for row in rows if row.slot is not None]
        if slots:
            self.generator._flush_ar_packed_decode_owners(slots)

    def _flush_row_owner(self, row: _GGUFResidentLoopRow) -> None:
        slot = row.slot
        if slot is None or slot.packed_decode_owner is None:
            return
        owner = slot.packed_decode_owner
        related = [
            candidate
            for candidate_row in self._rows.values()
            for candidate in (candidate_row.slot,)
            if candidate is not None and candidate.packed_decode_owner is owner
        ]
        self.generator._flush_ar_packed_decode_owners(related)

    def _flush_all_packed_owners(self) -> None:
        slots = [row.slot for row in self._rows.values() if row.slot is not None]
        if slots:
            self.generator._flush_ar_packed_decode_owners(slots)

    def _native_stream_chunk(self, row: _GGUFResidentLoopRow) -> GenerationStreamChunk:
        slot = row.slot
        if slot is None or not slot.generated_ids:
            raise RuntimeError("GGUF resident model row has no token to stream")
        generated_ids = tuple(int(token) for token in slot.generated_ids)
        request = row.sampling_request or row.request
        sample = row.samples[-1] if row.samples else None
        execution_path = (
            "gguf_packed_ar_host_sampler_decode"
            if row.native_sampled
            else "gguf_packed_ar_server_decode"
        )
        return GenerationStreamChunk(
            text=(
                _gguf_token_text(self.generator.tokenizer, sample)
                if sample is not None
                else self.generator.tokenizer.decode((generated_ids[-1],))
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
                native_sampler_rows=False,
            ),
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
            )
        )
        token_logprobs = tuple(
            _gguf_token_logprob(self.generator.tokenizer, sample)
            for sample in row.samples
        )
        execution_path = (
            "gguf_packed_ar_host_sampler_decode"
            if row.native_sampled
            else "gguf_packed_ar_server_decode"
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
                native_sampler_rows=False,
                timing=timing,
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
            ),
        )

    def _execution_metadata(self, row: _GGUFResidentLoopRow) -> dict[str, Any]:
        slot = row.slot
        return {
            "native_greedy": bool(row.native_greedy),
            "native_sampled": bool(row.native_sampled),
            "native_compact_prefill": bool(slot is not None and slot.native_compact_prefill),
            "native_decode_steps": 0 if slot is None else int(slot.native_decode_steps),
            "native_c1_decode_steps": (
                0 if slot is None else int(slot.native_c1_decode_steps)
            ),
            "serial_decode_fallback": bool(
                slot is not None and slot.serial_decode_steps > 0
            ),
        }


def _select_from_gguf_logits(
    result: Any,
    request: GenerationRequest,
    state: RowSamplingState,
):
    logits = getattr(result, "logits", None)
    if logits is None:
        raise RuntimeError("GGUF sampled generation requires logits from the resident session")
    return select_token(logits.reshape(-1), request, state)


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
    native_compact_prefill: bool = False,
    native_decode_steps: int = 0,
    native_c1_decode_steps: int = 0,
    native_caware_decode: bool = False,
    serial_decode_fallback: bool | None = None,
) -> dict[str, Any]:
    request_ids = tuple(range(len(outputs)))
    path = execution_path or _gguf_execution_path(plan)
    prompt_lengths = [len(prompt_rows_by_request.get(request_id, ())) for request_id in request_ids]
    decode_steps = max((len(generated_ids_by_request.get(request_id, ())) for request_id in request_ids), default=0)
    serial_fallback = len(request_ids) > 1 if serial_decode_fallback is None else bool(serial_decode_fallback)
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
        "native_caware_decode": bool(native_caware_decode),
        "native_sampler_rows": False,
        "throughput_claim_eligible": False,
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
        native_compact_prefill=bool(native_compact_prefill),
        native_caware_decode=bool(native_caware_decode),
        serial_decode_fallback=serial_fallback,
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
                    native_sampler_rows=False,
                ),
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
) -> GenerationTelemetry:
    if timing is not None and timing_scope is None:
        timing_scope = "choice"
        group_rows = 1 if group_rows is None else int(group_rows)
        timing_owner = True if timing_owner is None else bool(timing_owner)
    plan = _gguf_sampler_plan(request)
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
    if not request.ignore_eos and int(token_id) == int(tokenizer.eos_token_id):
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
) -> FinishDetails:
    details: FinishDetails
    if generated_ids:
        token_id = int(generated_ids[-1])
        if not request.ignore_eos and int(token_id) == int(tokenizer.eos_token_id):
            details = FinishDetails(reason="eos", eos_token_id=token_id, sampler_mode=_sampler_mode_value(request))
            return finish_details_with_sampling_state(details, state)
        if token_id in {int(stop_id) for stop_id in request.stop_token_ids}:
            details = FinishDetails(reason="stop", stop_sequence=(token_id,), sampler_mode=_sampler_mode_value(request))
            return finish_details_with_sampling_state(details, state)
        sequence = _gguf_stop_sequence_match(generated_ids, request.stop_token_sequences)
        if sequence:
            details = FinishDetails(reason="stop", stop_sequence=sequence, sampler_mode=_sampler_mode_value(request))
            return finish_details_with_sampling_state(details, state)
    if len(generated_ids) >= max(0, int(request.max_tokens)):
        details = FinishDetails(reason="length", length_limit=request.max_tokens, sampler_mode=_sampler_mode_value(request))
        return finish_details_with_sampling_state(details, state)
    details = FinishDetails(reason="stop", sampler_mode=_sampler_mode_value(request))
    return finish_details_with_sampling_state(details, state)


def _gguf_stop_sequence_match(
    generated_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    return token_sequence_state_for_tokens(generated_ids, stop_token_sequences).matched_sequence


def _sampler_mode_value(request: GenerationRequest) -> str:
    return _gguf_sampler_plan(request).mode.value


def _gguf_sampler_plan(request: GenerationRequest):
    return plan_sampler(request, native_gpu_requested=_native_gpu_sampler_requested())


def _native_gpu_sampler_requested() -> bool:
    value = os.environ.get("HIPENGINE_QWEN35_NATIVE_SAMPLER")
    return value is not None and value.strip().lower() not in {"", "0", "false", "no", "off"}


def make_qwen35_gguf_bringup_generator(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
) -> Qwen35GGUFBringupGenerator:
    return Qwen35GGUFBringupGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1100",
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
        },
    )


_GGUF_GENERATOR_FACTORIES_BY_BACKEND = {
    "hip_gfx1100": make_qwen35_gguf_bringup_generator,
    "hip_gfx1151": make_qwen35_gguf_bringup_generator_gfx1151,
}
_GGUF_GENERATOR_FACTORY_OVERRIDES = {
    ("hip_gfx1151", "gguf_q4_k_m"): make_qwen35_gguf_q4_k_m_generator_gfx1151,
}
for _model in ("qwen3_5_gguf", "qwen3_5_moe_gguf"):
    for _quant in ("gguf_q4_k_m", "gguf_q8_0", "gguf_q4_1", "gguf_ud_q4_k_xl"):
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
]
