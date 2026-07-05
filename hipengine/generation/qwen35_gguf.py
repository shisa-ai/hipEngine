"""Qwen3.5 GGUF generation path."""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Iterator

import numpy as np

from hipengine.generation.constraints import token_sequence_state_for_tokens
from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.finish import finish_details_with_sampling_state
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    GenerationTelemetry,
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
_GGUF_AR_PACKED_DECODE_ENV = "HIPENGINE_GGUF_AR_PACKED_DECODE"
_GGUF_AR_STREAM_DECODE_ENV = "HIPENGINE_GGUF_AR_STREAM_DECODE"
_GGUF_AR_STREAM_PREFILL_ENV = "HIPENGINE_GGUF_AR_STREAM_PREFILL"
_GGUF_MTP_SERVER_STREAM_DRAFT_ENV = "HIPENGINE_GGUF_MTP_SERVER_STREAM_DRAFT"


def _gguf_ar_packed_decode_enabled() -> bool:
    return os.environ.get(_GGUF_AR_PACKED_DECODE_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_ar_stream_decode_enabled() -> bool:
    return os.environ.get(_GGUF_AR_STREAM_DECODE_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_ar_stream_prefill_enabled() -> bool:
    return os.environ.get(_GGUF_AR_STREAM_PREFILL_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _gguf_mtp_server_stream_draft_enabled() -> bool:
    return os.environ.get(_GGUF_MTP_SERVER_STREAM_DRAFT_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


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
    session_pool_key: tuple[str, bool | None, bool | None] | None = None
    draft_pool_key: int | None = None
    mtp_device_kv_len: int = 0
    draft_stream: int = 0
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
    session_pool_key: tuple[str, bool | None, bool | None] | None = None
    done: bool = False
    native_decode_steps: int = 0
    serial_decode_steps: int = 0
    decode_stream: int = 0


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
    tokenizer: Qwen35GGUFTokenizer = field(init=False)
    last_batch_generation: dict[str, Any] | None = field(default=None, init=False, repr=False)
    last_generation_outputs: tuple[GenerationOutput, ...] = field(default=(), init=False, repr=False)
    _mtp_serving_assets: _GGUFMTPServingAssets | None = field(default=None, init=False, repr=False)
    _mtp_serving_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _shared_runner: Qwen35GGUFFullStackRunner | None = field(default=None, init=False, repr=False)
    _shared_runner_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _shared_session_pool: dict[
        tuple[str, bool | None, bool | None],
        list[Qwen35GGUFResidentSession],
    ] = field(default_factory=dict, init=False, repr=False)
    _shared_session_pool_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _shared_mtp_draft_pool: dict[int, list[Any]] = field(default_factory=dict, init=False, repr=False)
    _shared_mtp_draft_pool_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    supports_stream_logprobs: ClassVar[bool] = True

    def __post_init__(self) -> None:
        self.tokenizer = Qwen35GGUFTokenizer.from_gguf_info(self.weight_index)

    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: Any | None = None,
    ) -> int | None:
        """Materialize shared GGUF weights for server resident-session reuse."""

        if max_sequence_length is not None and int(max_sequence_length) <= 0:
            raise ValueError("max_sequence_length must be positive")
        self._get_shared_runner()
        return None if max_sequence_length is None else int(max_sequence_length)

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
                runner = Qwen35GGUFFullStackRunner(self.model_path)
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
    ) -> tuple[Qwen35GGUFResidentSession, tuple[str, bool | None, bool | None], bool]:
        self._ensure_shared_pools()
        key = (str(pool_name), use_wmma_prefill, use_gemv_decode)
        with self._shared_session_pool_lock:
            pool = self._shared_session_pool.get(key)
            session = pool.pop() if pool else None
        if session is not None:
            reset = getattr(session, "reset", None)
            if callable(reset):
                reset()
            return session, key, True
        return (
            Qwen35GGUFResidentSession(
                self.model_path,
                runtime=shared_runner.runtime,
                shared_runner=shared_runner,
                use_wmma_prefill=use_wmma_prefill,
                use_gemv_decode=use_gemv_decode,
            ),
            key,
            False,
        )

    def _release_shared_session(
        self,
        key: tuple[str, bool | None, bool | None] | None,
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
        use_wmma_prefill: bool | None = None,
        use_gemv_decode: bool | None = None,
    ):
        if shared_runner is None:
            session_kwargs: dict[str, bool] = {}
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

    def stream_detailed(self, request: GenerationRequest) -> Iterator[GenerationStreamChunk]:
        self.last_batch_generation = None
        if len(request.prompts) != 1:
            raise ValueError("streaming currently supports exactly one prompt")
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        if request.max_tokens == 0:
            return
        prompt_ids = self.tokenizer.encode(request.prompts[0])
        raise_if_generation_deadline_expired(request)
        if not prompt_ids:
            raise ValueError("GGUF prompt tokenization produced no token IDs")
        plan = _gguf_sampler_plan(request)
        shared_runner = self._prepared_shared_runner()
        session_kwargs = (
            {"runtime": shared_runner.runtime, "shared_runner": shared_runner}
            if shared_runner is not None
            else {}
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

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        plan = _gguf_sampler_plan(request)
        if request.max_tokens == 0:
            prompt_rows_by_request = {
                index: self.tokenizer.encode(prompt)
                for index, prompt in enumerate(request.prompts)
            }
            self.last_generation_outputs = tuple(
                GenerationOutput(
                    text="",
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
                prompt_ids = self.tokenizer.encode(prompt)
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
                    generated_ids_by_request[row_index] = [
                        int(token.token_id) for token in output.token_logprobs
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
            prompt_ids = self.tokenizer.encode(prompt)
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
                        finish_details=_gguf_finish_details(generated_ids, self.tokenizer, request),
                        telemetry=_gguf_telemetry(
                            slot.prompt_ids,
                            generated_ids,
                            request,
                            row_index=slot.request_id,
                            timing=row_timing,
                            execution_path="gguf_packed_ar_server_decode",
                            native_compact_prefill=False,
                            native_caware_decode=slot.native_decode_steps > 0,
                            serial_decode_fallback=slot.serial_decode_steps > 0,
                            native_sampler_rows=False,
                        ),
                    )
                )
            self.last_generation_outputs = tuple(outputs)
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
                native_decode_steps=native_decode_steps,
                native_caware_decode=native_decode_steps > 0,
                serial_decode_fallback=serial_decode_fallback,
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
            handled = self._try_step_ar_serving_slots_streams(live_slots, request)
            if not handled and _gguf_ar_packed_decode_enabled():
                handled = self._try_step_ar_serving_slots_batch(live_slots, request)
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
        index = 0
        while index < len(live_slots):
            remaining = len(live_slots) - index
            take = min(_MTP_SERVING_TARGET_BATCH_MAX_SLOTS, remaining)
            if remaining > _MTP_SERVING_TARGET_BATCH_MAX_SLOTS and remaining - take == 1:
                take -= 1
            chunk = live_slots[index:index + take]
            index += take
            if len(chunk) <= 1:
                for slot in chunk:
                    self._step_ar_serving_slot_serial(slot, request)
                continue
            first_session = chunk[0].session
            step_batch = getattr(first_session, "step_batch_native", None)
            if not callable(step_batch):
                for slot in chunk:
                    self._step_ar_serving_slot_serial(slot, request)
                continue
            token_ids = [int(slot.prev_token) for slot in chunk]
            sessions = [slot.session for slot in chunk]
            positions = [int(slot.seq_position) for slot in chunk]
            decode_start = time.perf_counter()
            try:
                with _temporary_env({"HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1"}):
                    batch_result = step_batch(
                        token_ids,
                        sessions=sessions,
                        positions=positions,
                        return_logits=False,
                    )
            except NotImplementedError:
                for slot in chunk:
                    self._step_ar_serving_slot_serial(slot, request)
                continue
            if batch_result is None:
                for slot in chunk:
                    self._step_ar_serving_slot_serial(slot, request)
                continue
            step_results = list(batch_result)
            if len(step_results) != len(chunk):
                raise RuntimeError(
                    f"GGUF AR native batch decode returned {len(step_results)} result(s) "
                    f"for {len(chunk)} live slot(s)"
                )
            decode_ms = _timing_ms_since(decode_start)
            for slot, step_result in zip(chunk, step_results, strict=True):
                _timing_add_ms(slot.timing, "decode_batch_ms", decode_ms)
                token = int(getattr(step_result, "token_id"))
                self._record_ar_serving_token(slot, token, request)
                slot.native_decode_steps += 1
        return True

    def _step_ar_serving_slot_serial(
        self,
        slot: _GGUFARServingSlot,
        request: GenerationRequest,
    ) -> None:
        if slot.done:
            return
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
            encoded_prompts[row_index] = self.tokenizer.encode(prompt)
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
            live_slots = [slot for slot in slots if not slot.done]
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
            try:
                verified_cycles = self._verify_mtp_serving_cycles(drafted_cycles, request)
            except Exception:
                raise
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
    ) -> list[_GGUFMTPVerifiedCycle]:
        _ = request
        if not drafted_cycles:
            return []
        for drafted in drafted_cycles:
            slot = drafted.slot
            snapshot_start = time.perf_counter()
            snapshot = None if drafted.direct_commit_exact else slot.session._linear_state_snapshot()
            drafted.snapshot = snapshot
            if snapshot is not None:
                _timing_add(slot.timing, "linear_state_snapshot_ms", snapshot_start)
        try:
            block_results = self._try_verify_mtp_serving_cycles_batch(drafted_cycles)
            if block_results is None:
                block_results = []
                for drafted in drafted_cycles:
                    slot = drafted.slot
                    verify_start = time.perf_counter()
                    block_result = slot.session.verify_target_block(
                        drafted.block_inputs,
                        bulk_attention_mode="bulk",
                        use_wmma_prefill=True,
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
        block_results: list[Any] = []
        for chunk in chunks:
            first_session = chunk[0].slot.session
            verify_batch = getattr(first_session, "verify_target_blocks_batch", None)
            if not callable(verify_batch):
                return None
            jobs = [
                {
                    "session": drafted.slot.session,
                    "input_token_ids": tuple(int(token) for token in drafted.block_inputs),
                    "bulk_attention_mode": "bulk",
                    "use_wmma_prefill": True,
                    "capture_linear_state_rows": True,
                    "defer_linear_state_commit": True,
                }
                for drafted in chunk
            ]
            verify_start = time.perf_counter()
            try:
                batch_result = verify_batch(jobs)
            except NotImplementedError:
                return None
            if batch_result is None:
                return None
            chunk_results = list(batch_result)
            verify_ms = _timing_ms_since(verify_start)
            for drafted in chunk:
                _timing_add_ms(drafted.slot.timing, "target_verify_ms", verify_ms)
                _timing_add_ms(drafted.slot.timing, "target_verify_batch_ms", verify_ms)
            block_results.extend(chunk_results)
        return block_results

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
            if consumed_rows < len(block_inputs):
                if not bool(getattr(verified.block_result, "linear_state_rows_captured", False)):
                    raise RuntimeError("direct MTP partial commit requested without captured state rows")
                slot.session._commit_verify_linear_state_row(
                    consumed_rows - 1,
                    position=block_start + consumed_rows,
                )
            elif drafted.direct_commit_exact:
                if not bool(getattr(verified.block_result, "linear_state_rows_captured", False)):
                    raise RuntimeError("direct MTP full commit requested without captured state rows")
                slot.session._commit_verify_linear_state_row(
                    len(block_inputs) - 1,
                    position=block_start + len(block_inputs),
                )
            else:
                if snapshot is None:
                    raise RuntimeError("MTP full-block replay requested without a linear-state snapshot")
                slot.session._restore_linear_state_snapshot(snapshot, position=block_start)
                replay_result = slot.session.verify_target_block_serial_exact(block_inputs)
                replay_tokens = [int(token) for token in replay_result.token_ids]
                if replay_tokens != block_target_tokens:
                    raise RuntimeError("MTP serial-exact replay diverged from block verifier rows")
            _timing_add(slot.timing, "target_state_commit_ms", state_commit_start)
            target_verify_seed_rows = [
                _new_mtp_seed_row(
                    token_id=block_target_tokens[row],
                    position=block_start + row,
                    hidden_ptr=slot.session.fp32_verify_hidden_seed_ptr(row),
                    hidden_size=slot.hidden_size,
                    source=f"verify[{row}]",
                )
                for row in range(len(block_target_tokens))
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
                if snapshot is not None:
                    _timing_add(timing, "linear_state_snapshot_ms", snapshot_start)
                try:
                    verify_start = time.perf_counter()
                    block_result = session.verify_target_block(
                        block_inputs,
                        bulk_attention_mode="bulk",
                        use_wmma_prefill=True,
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
                    if consumed_rows < len(block_inputs):
                        if not bool(getattr(block_result, "linear_state_rows_captured", False)):
                            raise RuntimeError("direct MTP partial commit requested without captured state rows")
                        session._commit_verify_linear_state_row(
                            consumed_rows - 1,
                            position=block_start + consumed_rows,
                        )
                    elif direct_commit_exact:
                        if not bool(getattr(block_result, "linear_state_rows_captured", False)):
                            raise RuntimeError("direct MTP full commit requested without captured state rows")
                        session._commit_verify_linear_state_row(
                            len(block_inputs) - 1,
                            position=block_start + len(block_inputs),
                        )
                    else:
                        if snapshot is None:
                            raise RuntimeError("MTP full-block replay requested without a linear-state snapshot")
                        session._restore_linear_state_snapshot(snapshot, position=block_start)
                        replay_result = session.verify_target_block_serial_exact(block_inputs)
                        replay_tokens = [int(token) for token in replay_result.token_ids]
                        if replay_tokens != block_target_tokens:
                            raise RuntimeError("MTP serial-exact replay diverged from block verifier rows")
                    _timing_add(timing, "target_state_commit_ms", state_commit_start)
                    target_verify_seed_rows = [
                        _new_mtp_seed_row(
                            token_id=block_target_tokens[row],
                            position=block_start + row,
                            hidden_ptr=session.fp32_verify_hidden_seed_ptr(row),
                            hidden_size=hidden_size,
                            source=f"verify[{row}]",
                        )
                        for row in range(len(block_target_tokens))
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
                # Eager per-token decode. The HIP decode graph provided no speed
                # benefit once build_hip loaded-library caching cut the per-launch
                # Python tax (~61 us -> ~12 us); eager == single-launch graph and
                # avoids the graph's 3rd-relaunch GDN corruption entirely. See
                # WORKLOG 2026-06-28 "#8 moot".
                decode_start = time.perf_counter()
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
        finish_details=finish_details,
        telemetry=telemetry,
    )


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
    native_decode_steps: int = 0,
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
        "serial_decode_fallback": serial_fallback,
        "native_compact_prefill": False,
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
    payload: dict[str, Any] = {
        "path": path,
        "batch_size": len(request_ids),
        "request_ids": list(request_ids),
        "prompt_lengths": [len(prompt_rows_by_request.get(request_id, ())) for request_id in request_ids],
        "decode_steps": max((len(generated_ids_by_request.get(request_id, ())) for request_id in request_ids), default=0),
        "native_decode_steps": 0,
        "serial_decode_fallback": len(request_ids) > int(resident_slot_count),
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
                    native_compact_prefill=False,
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
) -> GenerationTelemetry:
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
    )


for _model in ("qwen3_5_gguf", "qwen3_5_moe_gguf"):
    for _quant in ("gguf_q4_k_m", "gguf_q8_0", "gguf_q4_1", "gguf_ud_q4_k_xl"):
        register_text_generator(
            model=_model,
            backend="hip_gfx1100",
            quant=_quant,
            factory=make_qwen35_gguf_bringup_generator,
        )


__all__ = [
    "Qwen35GGUFBringupGenerator",
    "make_qwen35_gguf_bringup_generator",
]
