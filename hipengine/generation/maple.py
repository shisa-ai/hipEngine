"""Public greedy generation for deepgrove/maple-preview-2bit-mlx."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tokenizers.decoders import DecodeStream

from hipengine.dispatch import RequestState, SlotMove, WorkItem
from hipengine.generation.batch_scheduler import (
    CompletedRequest,
    GeneratedToken,
)
from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    register_text_generator,
)
from hipengine.kernels.backends import resolve_backend
from hipengine.loading.maple import MapleCheckpoint, validate_maple_weight_index
from hipengine.loading.safetensors import WeightIndex
from hipengine.runtime.maple import MapleBatchRunner, MapleRunner
from hipengine.tokenization.maple import MapleTokenizer

_MAPLE_QUANT = "maple_ternary2"
_MAPLE_DEFAULT_CONTEXT = 4_096
_MAPLE_PUBLIC_BATCH_CAPACITY = 8
_MAPLE_RESIDENT_EXECUTION_PATH = "maple_scheduler_native_prefill_batch_decode"


@dataclass
class MapleGenerator:
    """Greedy generator with bounded native prefill and resident packed weights."""

    model_path: str | Path
    weight_index: WeightIndex
    model_plugin: Any
    backend: str = "hip_gfx1151"
    context_length: int = _MAPLE_DEFAULT_CONTEXT
    runner_type: type[MapleRunner] = field(default=MapleRunner, repr=False)
    resident_batch_enabled: bool = field(default=True, repr=False)
    tokenizer: MapleTokenizer = field(init=False)
    checkpoint: MapleCheckpoint = field(init=False)
    last_generation_outputs: tuple[GenerationOutput, ...] = field(
        default=(), init=False, repr=False
    )
    last_generation_seconds: float | None = field(default=None, init=False, repr=False)
    last_batch_generation: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _runner: MapleRunner | None = field(default=None, init=False, repr=False)
    _resident_model_runner: MapleResidentModelRunner | None = field(
        default=None, init=False, repr=False
    )
    _load_seconds: float | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    supports_speculative_mtp = False
    supports_stream_many = False
    supports_resident_session_kv = False
    supports_stream_logprobs = False
    server_plain_ar_max_active_requests = _MAPLE_PUBLIC_BATCH_CAPACITY
    chat_template_family = "qwen"
    reasoning_parser_name = "qwen_tags"
    tool_parser_name = "qwen_tags"

    def __post_init__(self) -> None:
        self.model_path = Path(self.weight_index.model_path).expanduser().resolve()
        self.backend = resolve_backend(self.backend)
        self.context_length = int(self.context_length)
        self.checkpoint = MapleCheckpoint(
            index=self.weight_index,
            validation=validate_maple_weight_index(self.weight_index),
        )
        if self.context_length <= 0 or self.context_length > self.checkpoint.spec.max_position_embeddings:
            raise ValueError("Maple context_length is outside the checkpoint capacity")
        self.tokenizer = MapleTokenizer.from_model_path(
            self.model_path,
            model_vocab_size=self.checkpoint.spec.vocab_size,
            eos_token_id=self.checkpoint.spec.eos_token_id,
            bos_token_id=self.checkpoint.spec.bos_token_id,
        )

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Tokenize preformatted text; use ``tokenize_chat`` for a plain user message."""

        return self.tokenizer.encode(str(text))

    def tokenize_chat(self, user: str, *, system: str | None = None) -> tuple[int, ...]:
        return self.tokenizer.encode_chat(str(user), system=system)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def decode(self, token_ids, *, skip_special: bool = False) -> str:
        return self.tokenizer.decode(token_ids, skip_special=skip_special)

    @property
    def create_resident_model_runner(self):
        """Return the qualified resident-batch factory, or no capability."""

        if not getattr(self, "resident_batch_enabled", True):
            return None
        return self._create_resident_model_runner

    def _create_resident_model_runner(
        self,
        *,
        capacity: int | None = None,
    ) -> MapleResidentModelRunner:
        """Create the fixed-slot Maple owner used by the public engine loop."""

        if self._resident_model_runner is not None:
            raise RuntimeError("Maple resident model runner already exists")
        resolved = (
            _MAPLE_PUBLIC_BATCH_CAPACITY if capacity is None else int(capacity)
        )
        self._resident_model_runner = MapleResidentModelRunner(
            self, capacity=resolved
        )
        return self._resident_model_runner

    def generate(self, request: GenerationRequest) -> list[str]:
        return [output.text for output in self.generate_detailed(request)]

    def generate_detailed(self, request: GenerationRequest) -> tuple[GenerationOutput, ...]:
        self._validate_request(request)
        with self._lock:
            self._require_open()
            runner = self._ensure_runner()
            outputs: list[GenerationOutput] = []
            for row_index in range(len(request.prompts)):
                raise_if_generation_deadline_expired(request)
                prompt_ids = request.prompt_token_ids(row_index, self.tokenize)
                if not prompt_ids:
                    raise ValueError("Maple prompt produced no token IDs")
                if len(prompt_ids) + request.max_tokens > self.context_length:
                    raise ValueError("Maple prompt plus max_tokens exceeds context_length")
                started = time.perf_counter()
                runner.reset()
                next_step = runner.prefill_native(prompt_ids)
                generated: list[int] = []
                finish_reason = "length"
                eos_id: int | None = None
                stop_ids = set(request.stop_token_ids)
                configured_eos = (
                    self.checkpoint.spec.eos_token_id
                    if request.eos_token_id is None
                    else int(request.eos_token_id)
                )
                for _ in range(request.max_tokens):
                    raise_if_generation_deadline_expired(request)
                    token_id = int(next_step.token_id)
                    generated.append(token_id)
                    if token_id in stop_ids:
                        finish_reason = "stop"
                        eos_id = token_id
                        break
                    if not request.ignore_eos and token_id == configured_eos:
                        finish_reason = "stop"
                        eos_id = token_id
                        break
                    next_step = runner.step(token_id)
                text = self.tokenizer.decode(generated, skip_special=False)
                outputs.append(
                    GenerationOutput(
                        text=text,
                        generated_token_ids=tuple(generated),
                        finish_details=FinishDetails(
                            reason=finish_reason,
                            eos_token_id=eos_id,
                            length_limit=(
                                request.max_tokens if finish_reason == "length" else None
                            ),
                            sampler_mode="greedy",
                            phase="decode",
                        ),
                    )
                )
                self.last_generation_seconds = time.perf_counter() - started
            self.last_generation_outputs = tuple(outputs)
            return self.last_generation_outputs

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._resident_model_runner is not None:
                self._resident_model_runner.close()
                return
            self._closed = True
            if self._runner is not None:
                self._runner.close()
                self._runner = None

    def _ensure_runner(self) -> MapleRunner:
        if self._runner is not None:
            return self._runner
        started = time.perf_counter()
        self._runner = self.runner_type.load(
            self.checkpoint,
            backend=self.backend,
            max_context=self.context_length,
        )
        self._load_seconds = time.perf_counter() - started
        return self._runner

    def _validate_request(self, request: GenerationRequest) -> None:
        blockers: list[str] = []
        if request.temperature != 0.0:
            blockers.append("temperature must be 0")
        if request.top_p != 1.0 or request.top_k != 0 or request.min_p != 0.0:
            blockers.append("top-p/top-k/min-p sampling is not implemented")
        if (
            request.repetition_penalty != 1.0
            or request.presence_penalty != 0.0
            or request.frequency_penalty != 0.0
        ):
            blockers.append("logit penalties are not implemented")
        if request.logit_bias or request.suppress_token_ids:
            blockers.append("logit bias/suppression is not implemented")
        if request.stop_token_sequences:
            blockers.append("multi-token stop sequences are not implemented")
        if request.forced_tokens_pending or request.post_thinking_forced_tokens_pending:
            blockers.append("forced-token queues are not implemented")
        if request.tool_call_constraint is not None or request.json_object_close_forcing:
            blockers.append("structured constraints are not implemented")
        if request.min_tokens or request.logprobs or request.top_logprobs:
            blockers.append("min_tokens/logprobs are not implemented")
        if request.kv_storage not in {"auto", "bf16"}:
            blockers.append("only BF16 KV storage is implemented")
        if blockers:
            raise NotImplementedError("Maple basic runner: " + "; ".join(blockers))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Maple generator is closed")


@dataclass
class _MapleResidentRow:
    request_id: int
    request: GenerationRequest
    prompt_ids: tuple[int, ...]
    prefill_tokens_seen: int = 0
    slot: int | None = None
    next_token: int | None = None
    generated_ids: list[int] = field(default_factory=list)
    decoder: DecodeStream = field(
        default_factory=lambda: DecodeStream(skip_special_tokens=False)
    )
    streamed_text: str = ""


class MapleResidentModelRunner:
    """Scheduler-owned fixed-slot Maple prompt/decode/reclaim owner."""

    def __init__(self, generator: MapleGenerator, *, capacity: int) -> None:
        capacity = int(capacity)
        if capacity <= 0 or capacity > _MAPLE_PUBLIC_BATCH_CAPACITY:
            raise ValueError(
                "Maple resident capacity must be within "
                f"[1, {_MAPLE_PUBLIC_BATCH_CAPACITY}]"
            )
        self.generator = generator
        self.capacity = capacity
        self._batch: MapleBatchRunner | None = None
        self._rows: dict[int, _MapleResidentRow] = {}
        self._outputs: dict[int, GenerationOutput] = {}
        self._slot_to_request: list[int | None] = [None] * self.capacity
        self._prepared = False
        self._closed = False

    @property
    def active_request_ids(self) -> tuple[int, ...]:
        return tuple(self._rows)

    def prepare(self, *, max_sequence_length: int | None = None) -> None:
        if max_sequence_length is not None:
            required = int(max_sequence_length)
            if required <= 0 or required > self.generator.context_length:
                raise ValueError(
                    "Maple resident max_sequence_length exceeds public context"
                )
        if self._prepared:
            return
        with self.generator._lock:
            if self._closed or self.generator._closed:
                raise RuntimeError("Maple resident runner is closed")
            shared = self.generator._ensure_runner()
            self._batch = MapleBatchRunner.from_runner(
                shared,
                batch_size=self.capacity,
                per_capacity=self.generator.context_length,
            )
            self._prepared = True

    def prompt_tokens(self, prompt: Any) -> tuple[int, ...]:
        tokens = (
            self.generator.tokenize(prompt)
            if isinstance(prompt, str)
            else tuple(int(token) for token in prompt)
        )
        if not tokens:
            raise ValueError("Maple prompt produced no token IDs")
        return tokens

    def scheduler_max_new_tokens(self, request: GenerationRequest) -> int:
        return max(1, int(request.max_tokens))

    def register_batch(
        self,
        request_ids: Sequence[int],
        request: GenerationRequest,
        *,
        prompt_rows: Sequence[Sequence[int]],
    ) -> None:
        self.prepare()
        self.generator._validate_request(request)
        ids = tuple(int(request_id) for request_id in request_ids)
        prompts = tuple(tuple(int(token) for token in row) for row in prompt_rows)
        if len(ids) != len(request.prompts) or len(prompts) != len(ids):
            raise ValueError(
                "Maple request_ids, prompts, and prompt_rows must align"
            )
        for request_id, prompt_ids in zip(ids, prompts, strict=True):
            if request_id in self._rows or request_id in self._outputs:
                raise ValueError(f"request_id {request_id} is already registered")
            if len(prompt_ids) + int(request.max_tokens) > self.generator.context_length:
                raise ValueError(
                    "Maple prompt plus max_tokens exceeds context_length"
                )
            self._rows[request_id] = _MapleResidentRow(
                request_id=request_id,
                request=request,
                prompt_ids=prompt_ids,
            )

    def reserve_admission(self, request: RequestState) -> None:
        row = self._row(request.request_id)
        if row.slot is not None:
            return
        batch = self._require_batch()
        try:
            slot = self._slot_to_request.index(None)
        except ValueError as exc:
            raise RuntimeError("Maple resident batch has no free slot") from exc
        batch.reset_request(slot)
        self._slot_to_request[slot] = row.request_id
        row.slot = slot

    def rollback_admission(self, request: RequestState) -> None:
        row = self._row(request.request_id)
        self._release_slot(row)

    def prefill_batch(self, work: WorkItem, *, commit: bool) -> None:
        if not commit:
            raise ValueError("Maple resident prefill requires commit=True")
        batch = self._require_batch()
        slots = (
            tuple(int(slot) for slot in work.slot_ids)
            if work.slot_ids
            else tuple(
                int(self._row(request_id).slot)
                for request_id in work.request_ids
            )
        )
        for request_id, slot, token_row in zip(
            work.request_ids, slots, work.token_rows, strict=True
        ):
            row = self._row(request_id)
            raise_if_generation_deadline_expired(row.request)
            slot = int(slot)
            if row.slot is None:
                raise RuntimeError(
                    f"Maple request {request_id} has no reserved slot"
                )
            if row.slot != slot:
                raise RuntimeError(
                    f"Maple request {request_id} moved from slot "
                    f"{row.slot} to {slot} without compaction"
                )
            chunk = tuple(int(token) for token in token_row)
            start = row.prefill_tokens_seen
            if chunk != row.prompt_ids[start : start + len(chunk)]:
                raise RuntimeError(
                    f"Maple prefill chunk drift for request_id {request_id}"
                )
            row.prefill_tokens_seen += len(chunk)
            if int(row.request.max_tokens) == 0:
                continue
            result = batch.prefill_request(slot, chunk)
            row.next_token = int(result.token_id)
            raise_if_generation_deadline_expired(row.request)

    def decode_batch(
        self,
        work: WorkItem,
        *,
        commit: bool,
    ) -> tuple[GeneratedToken, ...]:
        if not commit:
            raise ValueError("Maple resident decode requires commit=True")
        batch = self._require_batch()
        ids = [0] * self.capacity
        active = [False] * self.capacity
        events: list[GeneratedToken] = []
        advancing_rows: list[_MapleResidentRow] = []
        for request_id in work.request_ids:
            row = self._row(request_id)
            raise_if_generation_deadline_expired(row.request)
            if int(row.request.max_tokens) == 0:
                finish = FinishDetails(
                    reason="length",
                    length_limit=0,
                    sampler_mode="greedy",
                )
                events.append(
                    GeneratedToken(
                        int(request_id),
                        0,
                        finished=True,
                        stream_chunk=GenerationStreamChunk(
                            text="",
                            finish_details=finish,
                            generated_token_ids=(),
                        ),
                    )
                )
                continue
            if row.next_token is None or row.slot is None:
                raise RuntimeError("Maple resident row is not fully prefilled")
            token_id = int(row.next_token)
            row.generated_ids.append(token_id)
            finish = _maple_finish_details(
                row.generated_ids,
                row.request,
                default_eos_token_id=self.generator.checkpoint.spec.eos_token_id,
            )
            piece = row.decoder.step(self.generator.tokenizer.encoder, token_id) or ""
            row.streamed_text += piece
            if finish is not None:
                final_text = self.generator.tokenizer.decode(
                    row.generated_ids, skip_special=False
                )
                if not final_text.startswith(row.streamed_text):
                    raise RuntimeError("Maple incremental decode diverged from full decode")
                piece += final_text[len(row.streamed_text) :]
                row.streamed_text = final_text
            else:
                ids[row.slot] = token_id
                active[row.slot] = True
                advancing_rows.append(row)
            events.append(
                GeneratedToken(
                    int(request_id),
                    token_id,
                    finished=finish is not None,
                    stream_chunk=GenerationStreamChunk(
                        text=piece,
                        finish_details=finish,
                        generated_token_ids=(
                            tuple(row.generated_ids) if finish is not None else None
                        ),
                    ),
                )
            )
        if len(advancing_rows) == 1:
            row = advancing_rows[0]
            assert row.slot is not None
            result = batch.step_request(row.slot, ids[row.slot])
            row.next_token = int(result.token_id)
            raise_if_generation_deadline_expired(row.request)
        elif advancing_rows:
            next_tokens = batch.batch_step(ids, active_mask=active)
            for row in advancing_rows:
                assert row.slot is not None
                row.next_token = int(next_tokens[row.slot])
                raise_if_generation_deadline_expired(row.request)
        return tuple(events)

    def compact_batch(self, moves: Sequence[SlotMove]) -> None:
        if moves:
            raise NotImplementedError(
                "Maple resident KV slot compaction is not implemented; "
                "fixed sparse slots remain supported"
            )

    def reclaim(self, completed: CompletedRequest) -> None:
        row = self._rows.pop(int(completed.request_id), None)
        if row is None:
            return
        finish = (
            completed.finish_details
            if completed.finish_reason in {"cancel", "disconnect", "timeout"}
            else _maple_finish_details(
                row.generated_ids,
                row.request,
                default_eos_token_id=self.generator.checkpoint.spec.eos_token_id,
            )
        )
        if finish is None:
            finish = completed.finish_details
        generated = tuple(row.generated_ids)
        self._outputs[row.request_id] = GenerationOutput(
            text=self.generator.tokenizer.decode(generated, skip_special=False),
            generated_token_ids=generated,
            finish_details=finish,
        )
        self._release_slot(row)

    def has_outputs(self, request_ids: Sequence[int]) -> bool:
        return all(int(request_id) in self._outputs for request_id in request_ids)

    def missing_outputs(self, request_ids: Sequence[int]) -> list[int]:
        return [
            int(request_id)
            for request_id in request_ids
            if int(request_id) not in self._outputs
        ]

    def take_outputs(self, request_ids: Sequence[int]) -> list[GenerationOutput]:
        return [self._outputs.pop(int(request_id)) for request_id in request_ids]

    def discard(self, request_ids: Sequence[int]) -> None:
        for request_id in request_ids:
            rid = int(request_id)
            row = self._rows.pop(rid, None)
            if row is not None:
                self._release_slot(row)
            self._outputs.pop(rid, None)

    def finalize_batch(
        self,
        request: GenerationRequest,
        request_ids: Sequence[int],
        outputs: Sequence[GenerationOutput],
    ) -> None:
        output_tuple = tuple(outputs)
        self.generator.last_generation_outputs = output_tuple
        self.generator.last_batch_generation = {
            "path": _MAPLE_RESIDENT_EXECUTION_PATH,
            "backend": self.generator.backend,
            "quant": _MAPLE_QUANT,
            "batch_size": len(output_tuple),
            "group_rows": len(output_tuple),
            "physical_batch_rows": self.capacity,
            "request_ids": [int(request_id) for request_id in request_ids],
            "prompt_lengths": [
                len(request.prompt_token_ids(index, self.generator.tokenize))
                for index in range(len(request.prompts))
            ],
            "decode_steps": max(
                (len(output.generated_token_ids or ()) for output in output_tuple),
                default=0,
            ),
            "native_compact_prefill": False,
            "native_packed_slot_admission": True,
            "native_caware_decode": True,
            "serial_decode_fallback": False,
            "throughput_claim_eligible": True,
        }

    def observability_snapshot(self) -> dict[str, object]:
        return {
            "kind": "maple_resident_model_runner",
            "capacity": self.capacity,
            "active_request_ids": list(self.active_request_ids),
            "outputs_buffered": len(self._outputs),
            "slot_to_request": list(self._slot_to_request),
            "prepared": self._prepared,
            "closed": self._closed,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        if self._batch is not None:
            try:
                self._batch.close()
            except BaseException as exc:  # pragma: no cover - defensive cleanup
                error = exc
            self._batch = None
        self._rows.clear()
        self._outputs.clear()
        self._slot_to_request = [None] * self.capacity
        with self.generator._lock:
            self.generator._resident_model_runner = None
            self.generator._closed = True
            if self.generator._runner is not None:
                try:
                    self.generator._runner.close()
                except BaseException as exc:  # pragma: no cover - defensive cleanup
                    if error is None:
                        error = exc
                self.generator._runner = None
        if error is not None:
            raise error

    def _row(self, request_id: int) -> _MapleResidentRow:
        try:
            return self._rows[int(request_id)]
        except KeyError as exc:
            raise KeyError(f"unknown Maple request_id {request_id}") from exc

    def _release_slot(self, row: _MapleResidentRow) -> None:
        slot, row.slot = row.slot, None
        if slot is None:
            return
        if self._slot_to_request[slot] == row.request_id:
            self._slot_to_request[slot] = None
        if self._batch is not None:
            self._batch.reset_request(slot)

    def _require_batch(self) -> MapleBatchRunner:
        if self._closed:
            raise RuntimeError("Maple resident runner is closed")
        self.prepare()
        assert self._batch is not None
        return self._batch


def _maple_finish_details(
    generated_ids: Sequence[int],
    request: GenerationRequest,
    *,
    default_eos_token_id: int,
) -> FinishDetails | None:
    if not generated_ids:
        return None
    token_id = int(generated_ids[-1])
    eos_id = (
        int(default_eos_token_id)
        if request.eos_token_id is None
        else int(request.eos_token_id)
    )
    if token_id in set(request.stop_token_ids):
        return FinishDetails(
            reason="stop",
            eos_token_id=token_id,
            sampler_mode="greedy",
        )
    if not request.ignore_eos and eos_id is not None and token_id == int(eos_id):
        return FinishDetails(
            reason="stop",
            eos_token_id=token_id,
            sampler_mode="greedy",
        )
    if len(generated_ids) >= int(request.max_tokens):
        return FinishDetails(
            reason="length",
            length_limit=int(request.max_tokens),
            sampler_mode="greedy",
        )
    return None


def make_maple_generator_cuda_sm120a(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> MapleGenerator:
    from hipengine.runtime.maple_cuda import MapleCudaRunner

    return MapleGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="cuda_sm120a",
        runner_type=MapleCudaRunner,
        resident_batch_enabled=False,
    )


def make_maple_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> MapleGenerator:
    return MapleGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1151",
    )


def make_maple_generator_gfx1100(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> MapleGenerator:
    return MapleGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1100",
    )


register_text_generator(
    model="maple",
    backend="cuda_sm120a",
    quant=_MAPLE_QUANT,
    factory=make_maple_generator_cuda_sm120a,
)
register_text_generator(
    model="maple",
    backend="hip_gfx1151",
    quant=_MAPLE_QUANT,
    factory=make_maple_generator_gfx1151,
)
register_text_generator(
    model="maple",
    backend="hip_gfx1100",
    quant=_MAPLE_QUANT,
    factory=make_maple_generator_gfx1100,
)


__all__ = [
    "MapleGenerator",
    "MapleResidentModelRunner",
    "make_maple_generator_cuda_sm120a",
    "make_maple_generator_gfx1100",
    "make_maple_generator_gfx1151",
]
