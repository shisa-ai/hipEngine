"""Qwen3.5/PARO text generation bring-up path."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from hipengine.generation.batch_scheduler import GeneratedToken, ResidentBatchScheduler
from hipengine.generation.registry import GenerationRequest, register_text_generator
from hipengine.kvcache import resolve_kv_policy
from hipengine.loading import WeightIndex
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoNextTokenRunner,
    Qwen35ParoResidentSession,
    _decode_token_cached,
    _select_token,
)


@dataclass
class Qwen35ParoOneTokenGenerator:
    """Greedy Qwen3.5/PARO generator backed by resident c=1 execution.

    The implementation is still serial across prompts, but each prompt uses the
    resident single-request native prefill path followed by multi-token
    autoregressive decode using the resident HIP layer chain.
    """

    model_path: str | Path
    weight_index: WeightIndex
    model_plugin: Any
    backend: str = "auto"
    lm_head_chunk: int = 4096
    _runner: Qwen35ParoNextTokenRunner | None = field(default=None, init=False, repr=False)
    _session: Qwen35ParoResidentSession | None = field(default=None, init=False, repr=False)
    _session_capacity: int = field(default=0, init=False, repr=False)
    _session_batch_size: int = field(default=0, init=False, repr=False)
    _session_kv_key: tuple[str, str, str, int] | None = field(default=None, init=False, repr=False)
    last_batch_generation: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def generate(self, request: GenerationRequest) -> list[str]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if request.temperature != 0.0 or request.top_p != 1.0:
            raise NotImplementedError(
                "Qwen3.5/PARO generator currently supports greedy sampling only"
            )
        if request.max_tokens == 0:
            self.last_batch_generation = None
            return ["" for _ in request.prompts]
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            request.kv_storage,
            scale_dtype=request.kv_scale_dtype,
            scale_granularity=request.kv_scale_granularity,
        )
        if len(request.prompts) == 1:
            self.last_batch_generation = None
            return [
                self._generate_one(
                    runner,
                    request.prompts[0],
                    request.max_tokens,
                    ignore_eos=request.ignore_eos,
                    kv_policy=kv_policy,
                )
            ]
        return self._generate_batch(
            runner,
            request.prompts,
            request.max_tokens,
            ignore_eos=request.ignore_eos,
            kv_policy=kv_policy,
        )

    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: Any | None = None,
    ) -> int:
        params = sampling_params
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            getattr(params, "kv_storage", "auto"),
            scale_dtype=getattr(params, "kv_scale_dtype", "fp16"),
            scale_granularity=getattr(params, "kv_scale_granularity", "per_token_head"),
        )
        auto_context_length = max_sequence_length is None
        if auto_context_length:
            requested_length = int(getattr(runner.config, "max_position_embeddings", 0) or 0)
            if requested_length <= 0:
                requested_length = _session_capacity_for(1)
        else:
            if int(max_sequence_length) <= 0:
                raise ValueError("max_sequence_length must be positive")
            requested_length = int(max_sequence_length)
        session_capacity = _session_capacity_for(requested_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
            auto_context_length=auto_context_length,
        )
        return int(getattr(session, "max_sequence_length", self._session_capacity))

    def count_tokens(self, text: str) -> int:
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), str(text), None)
        return len(prompt_ids)

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        if len(request.prompts) != 1:
            raise ValueError("streaming currently supports exactly one prompt")
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if request.temperature != 0.0 or request.top_p != 1.0:
            raise NotImplementedError(
                "Qwen3.5/PARO generator currently supports greedy sampling only"
            )
        if request.max_tokens == 0:
            return
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            request.kv_storage,
            scale_dtype=request.kv_scale_dtype,
            scale_granularity=request.kv_scale_granularity,
        )
        yield from self._stream_one(
            runner,
            request.prompts[0],
            request.max_tokens,
            ignore_eos=request.ignore_eos,
            kv_policy=kv_policy,
        )

    def _generate_one(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompt: str,
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
    ) -> str:
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        required_sequence_length = len(prompt_ids) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        generated_text: list[str] = []
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
        )
        next_result = session.prefill_native(prompt_ids, sample=True)
        if next_result is None:
            raise RuntimeError("native prefill did not produce next-token logits")
        generated_text.append(next_result.token_text)
        if not ignore_eos and _is_eos(session.tokenizer, next_result.token_id):
            return "".join(generated_text)

        remaining = max_tokens - 1
        if remaining:
            with session.capture_decode_graph(
                position=len(prompt_ids),
                steps_per_replay=1,
                max_replay_steps=remaining,
                record_steps=remaining,
            ) as graph:
                graph.replay(remaining)
                token_ids = graph.read_generated_token_ids(remaining)
            for token_id in token_ids:
                generated_text.append(_decode_token_cached(session.tokenizer, token_id))
                if not ignore_eos and _is_eos(session.tokenizer, token_id):
                    break
        return "".join(generated_text)

    def _generate_batch(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompts: tuple[str, ...],
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
    ) -> list[str]:
        """Generate a prompt list through the scheduler-owned c>N path.

        Native compact prefill runs all admitted rows together when their block
        table shapes permit it. Decode remains the explicit serial slot bridge
        until native c-aware replay lands; keep ``last_batch_generation`` clear
        about that so prompt-list batching is not mistaken for a retained c>N
        throughput path.
        """

        prompt_rows: list[list[int]] = []
        for prompt in prompts:
            _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
            if not prompt_ids:
                raise ValueError("prompt produced no tokens")
            prompt_rows.append([int(token) for token in prompt_ids])
        batch_size = len(prompt_rows)
        required_sequence_length = max(len(row) for row in prompt_rows) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            max_batch_size=batch_size,
            kv_policy=kv_policy,
        )
        scheduler = ResidentBatchScheduler(capacity=batch_size)
        request_ids = tuple(
            scheduler.submit(row, max_new_tokens=max(0, max_tokens - 1))
            for row in prompt_rows
        )
        admitted = scheduler.admit_pending()
        if admitted != request_ids:
            raise RuntimeError(f"unexpected admitted request ids {admitted!r}")

        output_parts: dict[int, list[str]] = {request_id: [] for request_id in request_ids}
        next_token_by_request: dict[int, int] = {}
        packed_slabs = scheduler.next_compact_prefill_slabs(
            chunk_size=max(len(row) for row in prompt_rows),
            block_size=getattr(session, "block_size", 256),
        )
        prefill_slab_shapes: list[dict[str, Any]] = []
        for slab in packed_slabs:
            prefill_slab_shapes.append(
                {
                    "request_ids": list(slab.request_ids),
                    "slot_ids": list(slab.physical_slot_ids),
                    "rows": slab.rows,
                    "request_count": slab.request_count,
                    "block_count": slab.block_count,
                }
            )
            results = session.prefill_native_packed(slab, sample=True)
            if len(results) != slab.request_count:
                raise RuntimeError(
                    "packed prefill returned "
                    f"{len(results)} results for {slab.request_count} requests"
                )
            for request_id, result in zip(slab.request_ids, results, strict=True):
                if result is None:
                    raise RuntimeError("packed native prefill did not produce next-token logits")
                output_parts[request_id].append(result.token_text)
                seed_finished = (
                    not ignore_eos and _is_eos(session.tokenizer, result.token_id)
                ) or max_tokens <= 1
                if seed_finished:
                    scheduler.record_generated(
                        (GeneratedToken(request_id, result.token_id, finished=True),)
                    )
                else:
                    next_token_by_request[request_id] = int(result.token_id)

        decode_steps = 0
        native_decode_steps = 0
        serial_decode_fallback = False
        while next_token_by_request:
            work = scheduler.next_decode_work()
            if work is None:
                raise RuntimeError("scheduler did not emit decode work")
            request_ids_for_step = tuple(
                request_id for request_id in work.request_ids if request_id in next_token_by_request
            )
            if not request_ids_for_step:
                raise RuntimeError("scheduler decode work did not include runnable requests")
            token_ids_for_step = [next_token_by_request[request_id] for request_id in request_ids_for_step]
            positions_for_step = [
                scheduler.active_batch.requests[request_id].context_len
                for request_id in request_ids_for_step
            ]
            slots_for_step = [
                scheduler.active_batch.slot_for(request_id)
                for request_id in request_ids_for_step
            ]
            compact_slots = tuple(slots_for_step) == tuple(range(len(slots_for_step)))
            use_native_decode = compact_slots and len(slots_for_step) > 1 and hasattr(session, "step_batch_native")
            if use_native_decode:
                try:
                    results = session.step_batch_native(
                        token_ids_for_step,
                        positions=positions_for_step,
                        slots=slots_for_step,
                        sample=True,
                    )
                    native_decode_steps += 1
                except NotImplementedError:
                    serial_decode_fallback = True
                    results = session.step_batch_serial(
                        token_ids_for_step,
                        positions=positions_for_step,
                        slots=slots_for_step,
                        sample=True,
                    )
            else:
                serial_decode_fallback = serial_decode_fallback or len(slots_for_step) > 1
                results = session.step_batch_serial(
                    token_ids_for_step,
                    positions=positions_for_step,
                    slots=slots_for_step,
                    sample=True,
                )
            generated: list[GeneratedToken] = []
            for request_id, result in zip(request_ids_for_step, results, strict=True):
                if result is None:
                    raise RuntimeError("decode step did not produce next-token logits")
                output_parts[request_id].append(result.token_text)
                next_token_by_request[request_id] = int(result.token_id)
                finished = not ignore_eos and _is_eos(session.tokenizer, result.token_id)
                generated.append(GeneratedToken(request_id, result.token_id, finished=finished))
            completed = scheduler.record_generated(generated)
            for done in completed:
                next_token_by_request.pop(done.request_id, None)
            decode_steps += 1

        native_decode_complete = decode_steps > 0 and native_decode_steps == decode_steps and not serial_decode_fallback
        batch_execution = session.batch_execution_metadata(
            scheduler_owned=True,
            native_decode=native_decode_complete,
        )
        self.last_batch_generation = {
            "path": "scheduler_native_packed_prefill_native_decode"
            if native_decode_complete
            else "scheduler_native_packed_prefill_serial_decode",
            "batch_size": batch_size,
            "request_ids": list(request_ids),
            "prompt_lengths": [len(row) for row in prompt_rows],
            "packed_prefill_slabs": prefill_slab_shapes,
            "decode_steps": decode_steps,
            "native_decode_steps": native_decode_steps,
            "serial_decode_fallback": serial_decode_fallback,
            "native_compact_prefill": bool(
                getattr(batch_execution, "native_compact_prefill", False)
            ),
            "native_caware_decode": bool(getattr(batch_execution, "native_caware_decode", False)),
            "throughput_claim_eligible": bool(
                getattr(batch_execution, "throughput_claim_eligible", False)
            ),
        }
        return ["".join(output_parts[request_id]) for request_id in request_ids]

    def _stream_one(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompt: str,
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
    ) -> Iterator[str]:
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        required_sequence_length = len(prompt_ids) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
        )
        next_result = session.prefill_native(prompt_ids, sample=True)
        if next_result is None:
            raise RuntimeError("native prefill did not produce next-token logits")
        yield next_result.token_text
        if not ignore_eos and _is_eos(session.tokenizer, next_result.token_id):
            return

        current_token_id = next_result.token_id
        for position in range(len(prompt_ids), len(prompt_ids) + max_tokens - 1):
            result = session.step(current_token_id, position=position, sample=True)
            if result is None:
                raise RuntimeError("decode step did not produce next-token logits")
            yield result.token_text
            current_token_id = result.token_id
            if not ignore_eos and _is_eos(session.tokenizer, result.token_id):
                return

    def _get_runner(self) -> Qwen35ParoNextTokenRunner:
        if self._runner is None:
            self._runner = Qwen35ParoNextTokenRunner(
                self.model_path,
                index=self.weight_index,
                backend=self.backend,
            )
        return self._runner

    def _get_session(
        self,
        runner: Qwen35ParoNextTokenRunner,
        *,
        max_sequence_length: int,
        kv_policy,
        auto_context_length: bool = False,
        max_batch_size: int = 1,
    ) -> Qwen35ParoResidentSession:
        kv_key = (
            kv_policy.storage_dtype.value,
            kv_policy.scale_dtype.value,
            kv_policy.scale_granularity,
            int(kv_policy.block_size),
        )
        batch_size = max(1, int(max_batch_size))
        capacity_ok = self._session_capacity >= max_sequence_length or bool(auto_context_length)
        batch_ok = self._session_batch_size >= batch_size
        if (
            self._session is None
            or not capacity_ok
            or not batch_ok
            or self._session_kv_key != kv_key
        ):
            self.close()
            session_kwargs = {
                "max_sequence_length": max_sequence_length,
                "kv_policy": kv_policy.create_policy(),
                "kv_scale_dtype": kv_policy.scale_dtype,
                "kv_scale_granularity": kv_policy.scale_granularity,
            }
            if auto_context_length:
                session_kwargs["auto_context_length"] = True
            if batch_size > 1:
                session_kwargs["max_batch_size"] = batch_size
            self._session = Qwen35ParoResidentSession(runner, **session_kwargs)
            self._session_capacity = int(
                getattr(self._session, "max_sequence_length", max_sequence_length)
            )
            self._session_batch_size = int(getattr(self._session, "max_batch_size", batch_size))
            self._session_kv_key = kv_key
        else:
            self._session.reset()
        return self._session

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._session_capacity = 0
        self._session_batch_size = 0
        self._session_kv_key = None


def _session_capacity_for(required_sequence_length: int) -> int:
    """Return a reusable session capacity for a request.

    Chat prompts grow after every turn, so allocating exactly the current
    prompt+decode length forces resident weights/KV buffers to be torn down and
    rebuilt on each request.  Keep a modest floor and bucket growth to preserve
    the resident session across normal local chat turns while still allowing
    larger explicit contexts to expand on demand.
    """

    required = int(required_sequence_length)
    if required <= 0:
        raise ValueError("required_sequence_length must be positive")
    floor = max(1, _env_int("HIPENGINE_SESSION_MIN_TOKENS", 4096))
    bucket = max(1, _env_int("HIPENGINE_SESSION_BUCKET_TOKENS", 1024))
    capacity = max(required, floor)
    return ((capacity + bucket - 1) // bucket) * bucket


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _is_eos(tokenizer: Any | None, token_id: int) -> bool:
    if tokenizer is None:
        return False
    try:
        eos_id = getattr(tokenizer, "token_to_id")("<|endoftext|>")
    except Exception:
        eos_id = None
    return eos_id is not None and int(token_id) == int(eos_id)


def make_qwen35_paro_one_token_generator(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> Qwen35ParoOneTokenGenerator:
    return Qwen35ParoOneTokenGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1100",
    )


def make_qwen35_paro_one_token_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> Qwen35ParoOneTokenGenerator:
    return Qwen35ParoOneTokenGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1151",
    )


register_text_generator(
    model="qwen3_5_moe_paro",
    backend="hip_gfx1100",
    quant="w4_paro",
    factory=make_qwen35_paro_one_token_generator,
)
register_text_generator(
    model="qwen3_5_moe_paro",
    backend="hip_gfx1151",
    quant="w4_paro",
    factory=make_qwen35_paro_one_token_generator_gfx1151,
)
