"""Strict Qwen3.8-Flash-Next qwen4exp GGUF text-generation plugin."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import threading
import time
from typing import Any, Sequence

from hipengine.dispatch import RequestState, SlotMove, WorkItem
from hipengine.generation.batch_scheduler import CompletedRequest, GeneratedToken
from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.qwen4_exp_multimodal import (
    qwen4_exp_multimodal_token_control,
    render_qwen4_exp_multimodal_prompt,
)
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    register_text_generator,
)
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.loading.qwen4_exp_gguf import build_qwen4_exp_gguf_tensor_map
from hipengine.loading.qwen4_exp_materialize import (
    materialize_qwen4_exp_weights,
    plan_qwen4_exp_residency,
)
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
from hipengine.tokenization.gguf import Qwen4ExpGGUFTokenizer


def _qwen4_exp_compact_prefill_kwargs() -> dict[str, bool]:
    return {"capture_logits": False, "capture_target_hidden": False}


def _qwen4_exp_compact_step_kwargs() -> dict[str, bool]:
    return {
        "capture_logits": False,
        "capture_target_hidden": False,
        "token_id_resident": True,
    }


class Qwen4ExpGGUFTextGenerator:
    """F5 strict c1/greedy text generator with serial multi-prompt ownership."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        weight_index: object,
        model_plugin: object,
        backend: str = "hip_gfx1151",
        tokenizer: Any | None = None,
        runner: Any | None = None,
        max_sequence_length: int | None = None,
        resident_capacity: int | None = None,
        prefill_chunk_size: int = 512,
        vision_model_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.weight_index = weight_index
        self.model_plugin = model_plugin
        self.backend = str(backend)
        self._configured_resident_capacity = (
            self.server_plain_ar_max_active_requests if resident_capacity is None else int(resident_capacity))
        if not 1 <= self._configured_resident_capacity <= self.server_plain_ar_max_active_requests:
            raise ValueError("Qwen4Exp resident capacity must be within 1..2")
        self.server_plain_ar_max_active_requests = self._configured_resident_capacity
        self.server_plain_ar_max_active_requests_by_max_sequence_length = {1024:self._configured_resident_capacity}
        self.context_admission = None
        self._resident = None
        self._vision_resident = None
        self._vision_runner = None
        self._speculative_provider = None
        self._resident_model_runner = None
        self._lock = threading.RLock()
        self._closed = False
        if tokenizer is None:
            metadata_info = getattr(weight_index, "metadata", None)
            if metadata_info is None:
                metadata_info = GGUFReader(discover_gguf_files(self.model_path)[0]).info
            else:
                metadata_info = weight_index
            tokenizer = Qwen4ExpGGUFTokenizer.from_gguf_info(metadata_info)
        self.tokenizer = tokenizer
        if runner is None:
            paths = discover_gguf_files(self.model_path)
            readers = tuple(GGUFReader(path) for path in paths)
            model_map = build_qwen4_exp_gguf_tensor_map(
                tuple(reader.info for reader in readers)
            )
            plan = plan_qwen4_exp_residency(
                model_map, staging_token_capacity=prefill_chunk_size
            )
            from hipengine.core.hip import get_hip_runtime
            from hipengine.loading.qwen4_exp_context import resolve_qwen4_exp_context
            runtime = get_hip_runtime()
            free_bytes,total_bytes = runtime.mem_get_info()
            vision_reserve = 0
            if vision_model_path is not None:
                vision_reserve = sum(
                    int(tensor.nbytes) for path in discover_gguf_files(Path(vision_model_path))
                    for tensor in GGUFReader(path).info.tensors)
            if free_bytes < vision_reserve:
                raise MemoryError("Qwen4Exp vision weights exceed available device memory")
            admission = resolve_qwen4_exp_context(
                plan,available_device_bytes=free_bytes-vision_reserve,
                requested_context=max_sequence_length,
                native_context_length=getattr(model_plugin,"native_context_length",model_map.config.context_length),
                resident_capacity=self._configured_resident_capacity)
            self.context_admission = {
                "mode": "auto" if max_sequence_length is None else "explicit",
                "device_free_bytes": free_bytes,"device_total_bytes": total_bytes,
                "vision_weight_reserve_bytes": vision_reserve,
                "plan": asdict(admission),
                "scratch_policy": "4GiB per runner includes current MMQ sidecars and prefill scratch",
            }
            self._resident = materialize_qwen4_exp_weights(
                readers,
                plan=plan,
                backend=self.backend,
                runtime=runtime,
            )
            try:
                runner = Qwen4ExpGGUFResidentModelRunner(
                    self._resident,
                    max_sequence_length=admission.context_tokens,
                    prefill_chunk_size=prefill_chunk_size,
                    backend=self.backend,
                )
            except Exception:
                self._resident.close()
                self._resident = None
                raise
        self.runner = runner
        if vision_model_path is not None:
            from hipengine.loading.qwen4_exp_vision_gguf import build_qwen4_exp_vision_gguf_map
            from hipengine.loading.qwen4_exp_vision_materialize import materialize_qwen4_exp_vision_weights, plan_qwen4_exp_vision_residency
            from hipengine.runtime.qwen4_exp_vision import Qwen4ExpVisionRunner
            vision_paths = discover_gguf_files(Path(vision_model_path))
            vision_readers = tuple(GGUFReader(path) for path in vision_paths)
            vision_map = build_qwen4_exp_vision_gguf_map(tuple(reader.info for reader in vision_readers))
            vision_plan = plan_qwen4_exp_vision_residency(vision_map)
            self._vision_resident = materialize_qwen4_exp_vision_weights(
                vision_readers, plan=vision_plan, backend=self.backend,
                runtime=self.runner.runtime,
            )
            try:
                source = vision_readers[0]
                self._vision_runner = Qwen4ExpVisionRunner(
                    self._vision_resident,
                    patch_weight0=source.tensor_data('v.patch_embd.weight'),
                    patch_weight1=source.tensor_data('v.patch_embd.weight.1'),
                    patch_bias=source.tensor_data('v.patch_embd.bias'),
                    position_embedding=source.tensor_data('v.position_embd.weight'),
                )
            except Exception:
                self._vision_resident.close()
                self._vision_resident = None
                raise

    server_plain_ar_max_active_requests = 2
    server_plain_ar_max_active_requests_by_max_sequence_length = {1024: 2}
    supports_stream_many = True
    supports_controlled_streaming = True

    @property
    def supports_vision(self) -> bool:
        return self._vision_runner is not None

    def create_resident_model_runner(
        self, *, capacity: int | None, config: Any | None = None
    ) -> "Qwen4ExpResidentServingRunner":
        del config
        resolved = (
            self._configured_resident_capacity
            if capacity is None
            else int(capacity)
        )
        if resolved <= 0 or resolved > self._configured_resident_capacity:
            raise ValueError(
                f"Qwen4Exp resident serving capacity must be within 1..{self._configured_resident_capacity}"
            )
        with self._lock:
            current = self._resident_model_runner
            if current is not None:
                if current.capacity != resolved:
                    raise RuntimeError(
                        "Qwen4Exp resident runner capacity cannot change while live"
                    )
                return current
            current = Qwen4ExpResidentServingRunner(self, capacity=resolved)
            self._resident_model_runner = current
            return current

    def prepare(
        self, *, max_sequence_length: int | None = None,
        sampling_params: Any | None = None,
    ) -> int:
        """Report the admitted context; never change the model's QSA budget."""
        with self._lock:
            self._require_open()
            if getattr(sampling_params,"kv_storage",None) not in (None,"auto","bf16"):
                raise NotImplementedError("Qwen4Exp context admission currently supports BF16 KV only")
            requested = self.runner.max_sequence_length if max_sequence_length is None else int(max_sequence_length)
            if not 0 < requested <= self.runner.max_sequence_length:
                raise ValueError("Qwen4Exp prepare exceeds admitted sequence capacity")
            return requested

    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        sampling_params: Any | None = None,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any]:
        del sampling_params, release_after_probe
        batch = int(max_batch_size)
        required = int(max_prompt_tokens) + max(0, int(max_new_tokens) - 1)
        if batch <= 0 or batch > self.server_plain_ar_max_active_requests:
            raise NotImplementedError("Qwen4Exp serving supports at most c2")
        if required <= 0 or required > self.runner.max_sequence_length:
            raise ValueError("Qwen4Exp serving scratch exceeds sequence capacity")
        resident = self.create_resident_model_runner(capacity=batch)
        resident.prepare(max_sequence_length=required)
        return {
            "schema": 1,
            "backend": self.backend,
            "execution_path": "qwen4exp_request_owned_runner_pool",
            "max_batch_size": batch,
            "max_sequence_length": required,
            "released_after_probe": False,
        }

    def count_tokens(self, text: str) -> int:
        self._require_open()
        return len(self.tokenizer.encode(str(text)))

    def tokenize(self, text: str) -> tuple[int, ...]:
        self._require_open()
        return tuple(int(token) for token in self.tokenizer.encode(str(text)))

    def detokenize(
        self, token_ids: Any, *, skip_special: bool = False
    ) -> str:
        self._require_open()
        return str(
            self.tokenizer.decode(
                tuple(int(token) for token in token_ids),
                skip_special=bool(skip_special),
            )
        )

    def generate(self, request: GenerationRequest) -> list[str]:
        return [output.text for output in self.generate_detailed(request)]

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        self._require_open()
        if request.temperature != 0.0 or request.top_k not in (0, 1):
            raise ValueError("Qwen4Exp F5 supports greedy generation only")
        outputs: list[GenerationOutput] = []
        for prompt in request.prompts:
            raise_if_generation_deadline_expired(request)
            token_ids = (
                [int(token) for token in prompt]
                if not isinstance(prompt, str)
                else [int(token) for token in self.tokenizer.encode(prompt)]
            )
            if len(token_ids) + request.max_tokens > self.runner.max_sequence_length:
                raise ValueError("Qwen4Exp request exceeds admitted sequence capacity")
            result = self.runner.prefill(
                token_ids,
                **_qwen4_exp_compact_prefill_kwargs(),
            )
            raise_if_generation_deadline_expired(request)
            generated: list[int] = []
            reason = "length"
            for index in range(request.max_tokens):
                raise_if_generation_deadline_expired(request)
                token = int(result.token_id)
                generated.append(token)
                if not request.ignore_eos and token == self.tokenizer.eos_token_id:
                    reason = "eos"
                    break
                if index + 1 < request.max_tokens:
                    result = self.runner.step(
                        token,
                        **_qwen4_exp_compact_step_kwargs(),
                    )
            outputs.append(
                GenerationOutput(
                    text=self.tokenizer.decode(generated, skip_special=False),
                    generated_token_ids=tuple(generated),
                    finish_details=FinishDetails(
                        reason=reason,
                        eos_token_id=(
                            self.tokenizer.eos_token_id if reason == "eos" else None
                        ),
                        length_limit=(request.max_tokens if reason == "length" else None),
                        sampler_mode="greedy",
                    ),
                )
            )
        return outputs

    @property
    def supports_speculative(self) -> bool:
        return self._speculative_provider is not None

    @property
    def supports_speculative_mtp(self) -> bool:
        return self.supports_speculative

    def attach_speculative_provider(self, provider: Any) -> None:
        self._require_open()
        if self._speculative_provider is not None:
            raise RuntimeError("Qwen4Exp generator already has a speculative provider")
        self._speculative_provider = provider

    def generate_speculative_detailed(
        self, request: GenerationRequest
    ) -> list[GenerationOutput]:
        self._require_open()
        if self._speculative_provider is None:
            raise NotImplementedError("Qwen4Exp speculative provider is not attached")
        return list(self._speculative_provider.generate_detailed(request))

    def generate_speculative_mtp_detailed(
        self, request: GenerationRequest
    ) -> list[GenerationOutput]:
        return self.generate_speculative_detailed(request)

    def stream_speculative_detailed(self, request: GenerationRequest):
        self._require_open()
        if self._speculative_provider is None:
            raise NotImplementedError("Qwen4Exp speculative provider is not attached")
        yield from self._speculative_provider.stream_detailed(request)

    def stream_speculative_mtp_detailed(self, request: GenerationRequest):
        yield from self.stream_speculative_detailed(request)

    def speculative_capabilities(self) -> dict[str, Any]:
        if self._speculative_provider is None:
            return {}
        return dict(self._speculative_provider.capabilities())

    def _encode_multimodal_input(self, media: Any):
        from hipengine.runtime.qwen4_exp_vision import Qwen4ExpVisionFeatures

        if isinstance(media, dict):
            if "items" in media:
                raw_items = tuple(media["items"])
            else:
                raw_items = tuple(
                    ({"type": "image", "data": value} for value in media.get("images", ()))
                ) + tuple(
                    ({"type": "video", "data": value} for value in media.get("videos", ()))
                )
        elif isinstance(media, (list, tuple)):
            raw_items = tuple({"type": "image", "data": value} for value in media)
        else:
            ndim = getattr(media, "ndim", None)
            kind = "video" if ndim == 4 else "image"
            raw_items = ({"type": kind, "data": media},)
        if not raw_items:
            raise ValueError("Qwen4Exp multimodal input is empty")
        encoded = []
        for item in raw_items:
            if not isinstance(item, dict) or item.get("type") not in {"image", "video"}:
                raise ValueError("Qwen4Exp media items require type=image|video and data")
            kind = str(item["type"])
            data = item.get("data")
            if kind == "video":
                method = getattr(self._vision_runner, "encode_video", None)
                if not callable(method):
                    raise NotImplementedError("attached Qwen4Exp vision runner lacks video support")
                feature = method(data)
            else:
                method = getattr(self._vision_runner, "encode_image", None)
                if callable(method):
                    feature = method(data)
                else:
                    values = self._vision_runner.encode(data)
                    feature = Qwen4ExpVisionFeatures(
                        values, (1, 2, int(values.shape[0]) * 2), "image"
                    )
            if not isinstance(feature, Qwen4ExpVisionFeatures):
                raise TypeError("Qwen4Exp vision runner returned an invalid feature object")
            encoded.append(feature)
        return tuple(encoded)

    def generate_multimodal_detailed(
        self,
        prompt: str,
        image: Any,
        request: GenerationRequest,
    ) -> GenerationOutput:
        self._require_open()
        raise_if_generation_deadline_expired(request)
        if self._vision_runner is None:
            raise NotImplementedError("Qwen4Exp vision model is not attached")
        if len(request.prompts) != 1:
            raise ValueError("Qwen4Exp multimodal generation supports one request")
        if request.temperature != 0.0 or request.top_k not in (0, 1):
            raise ValueError("Qwen4Exp multimodal generation supports greedy sampling")
        features = self._encode_multimodal_input(image)
        multimodal = render_qwen4_exp_multimodal_prompt(prompt, features)
        token_ids = [int(token) for token in self.tokenizer.encode(multimodal.rendered)]
        overrides, mrope_positions, next_rope_position = (
            qwen4_exp_multimodal_token_control(token_ids, multimodal.features)
        )
        if len(token_ids) + request.max_tokens > min(
            1024, self.runner.max_sequence_length
        ):
            raise ValueError("Qwen4Exp multimodal request exceeds 1K scope")
        raise_if_generation_deadline_expired(request)
        result = self.runner.prefill(
            token_ids,
            embedding_overrides=overrides,
            **_qwen4_exp_compact_prefill_kwargs(),
            mrope_positions=mrope_positions,
        )
        generated: list[int] = []
        reason = "length"
        for index in range(request.max_tokens):
            raise_if_generation_deadline_expired(request)
            token = int(result.token_id)
            generated.append(token)
            if not request.ignore_eos and token == self.tokenizer.eos_token_id:
                reason = "eos"
                break
            if index + 1 < request.max_tokens:
                position = next_rope_position + index
                result = self.runner.step(
                    token,
                    rope_positions=(position, position, position),
                    **_qwen4_exp_compact_step_kwargs(),
                )
        return GenerationOutput(
            text=self.tokenizer.decode(generated, skip_special=False),
            generated_token_ids=tuple(generated),
            finish_details=FinishDetails(
                reason=reason,
                eos_token_id=(
                    self.tokenizer.eos_token_id if reason == "eos" else None
                ),
                length_limit=(request.max_tokens if reason == "length" else None),
                sampler_mode="greedy_multimodal_mrope",
            ),
        )

    def close(self) -> None:
        if self._closed:
            return
        resident_runner = self._resident_model_runner
        if resident_runner is not None:
            self._resident_model_runner = None
            resident_runner.close()
            return
        if self._speculative_provider is not None:
            self._speculative_provider.close()
            self._speculative_provider = None
        if self._vision_runner is not None:
            self._vision_runner.close()
            self._vision_runner = None
        if self._vision_resident is not None:
            self._vision_resident.close()
            self._vision_resident = None
        self.runner.close()
        if self._resident is not None:
            self._resident.close()
            self._resident = None
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Qwen4Exp generator is closed")


@dataclass
class _Qwen4ExpServingRow:
    request_id: int
    row_index: int
    request: GenerationRequest
    prompt_ids: tuple[int, ...]
    submitted_at: float
    runner: Qwen4ExpGGUFResidentModelRunner | None = None
    prefill_tokens_seen: int = 0
    next_result: Any | None = None
    generated_ids: list[int] = field(default_factory=list)
    emitted_text: str = ""


class Qwen4ExpResidentServingRunner:
    """Request-owned c2 runner pool over one shared Qwen4Exp weight layout."""

    def __init__(self, generator: Qwen4ExpGGUFTextGenerator, *, capacity: int) -> None:
        self.generator = generator
        self.capacity = int(capacity)
        self._rows: dict[int, _Qwen4ExpServingRow] = {}
        self._outputs: dict[int, GenerationOutput] = {}
        self._all_runners = [generator.runner]
        self._available = [generator.runner]
        self._closed = False

    @property
    def active_request_ids(self) -> tuple[int, ...]:
        return tuple(self._rows)

    def prepare(self, *, max_sequence_length: int | None = None) -> int:
        if self._closed:
            raise RuntimeError("Qwen4Exp serving runner is closed")
        required = (
            self.generator.runner.max_sequence_length
            if max_sequence_length is None
            else int(max_sequence_length)
        )
        if required <= 0 or required > self.generator.runner.max_sequence_length:
            raise ValueError("Qwen4Exp serving prepare exceeds sequence capacity")
        if self.generator._resident is None and self.capacity > 1:
            raise RuntimeError("Qwen4Exp c2 requires model-owned resident weights")
        while len(self._all_runners) < self.capacity:
            runner = Qwen4ExpGGUFResidentModelRunner(
                self.generator._resident,
                max_sequence_length=self.generator.runner.max_sequence_length,
                prefill_chunk_size=self.generator.runner.prefill_chunk_size,
                backend=self.generator.backend,
                runtime=self.generator.runner.runtime,
            )
            self._all_runners.append(runner)
            self._available.append(runner)
        return required

    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        sampling_params: Any | None = None,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any]:
        del sampling_params, release_after_probe
        batch = int(max_batch_size)
        if batch <= 0 or batch > self.capacity:
            raise NotImplementedError(
                f"Qwen4Exp resident runner owns at most {self.capacity} rows"
            )
        required = int(max_prompt_tokens) + max(0, int(max_new_tokens) - 1)
        self.prepare(max_sequence_length=required)
        return {
            "schema": 1,
            "backend": self.generator.backend,
            "execution_path": "qwen4exp_request_owned_runner_pool",
            "max_batch_size": batch,
            "max_sequence_length": required,
            "resident_runners": batch,
            "released_after_probe": False,
        }

    def prompt_tokens(self, prompt: Any) -> tuple[int, ...]:
        if isinstance(prompt, str):
            values = tuple(int(token) for token in self.generator.tokenizer.encode(prompt))
        else:
            values = tuple(int(token) for token in prompt)
        if not values:
            raise ValueError("Qwen4Exp prompt tokenization produced no IDs")
        return values

    def scheduler_max_new_tokens(self, request: GenerationRequest) -> int:
        return max(1, int(request.max_tokens))

    def register_batch(
        self,
        request_ids: Sequence[int],
        request: GenerationRequest,
        *,
        prompt_rows: Sequence[Sequence[int]],
    ) -> None:
        if request.temperature != 0.0 or request.top_k not in (0, 1):
            raise ValueError("Qwen4Exp native serving supports greedy generation only")
        ids = tuple(int(value) for value in request_ids)
        prompts = tuple(tuple(int(token) for token in row) for row in prompt_rows)
        if len(ids) != len(request.prompts) or len(prompts) != len(ids):
            raise ValueError("Qwen4Exp request IDs/prompts must align")
        now = time.perf_counter()
        for row_index, (request_id, prompt_ids) in enumerate(
            zip(ids, prompts, strict=True)
        ):
            if request_id in self._rows or request_id in self._outputs:
                raise ValueError(f"request_id {request_id} is already registered")
            required = len(prompt_ids) + max(0, int(request.max_tokens) - 1)
            if required > self.generator.runner.max_sequence_length:
                raise ValueError("Qwen4Exp native request exceeds sequence capacity")
            self._rows[request_id] = _Qwen4ExpServingRow(
                request_id, row_index, request, prompt_ids, now
            )

    def reserve_admission(self, request: RequestState) -> None:
        self.prepare()
        row = self._row(request.request_id)
        if int(row.request.max_tokens) == 0:
            return
        if not self._available:
            raise RuntimeError("Qwen4Exp scheduler has no free request runner")
        row.runner = self._available.pop()
        row.runner.reset()

    def rollback_admission(self, request: RequestState) -> None:
        self._release_runner(self._row(request.request_id))

    def prefill_batch(self, work: WorkItem, *, commit: bool) -> None:
        if not commit:
            raise ValueError("Qwen4Exp native prefill requires commit=True")
        for request_id, token_row in zip(
            work.request_ids, work.token_rows, strict=True
        ):
            row = self._row(request_id)
            raise_if_generation_deadline_expired(row.request)
            chunk = tuple(int(token) for token in token_row)
            start = row.prefill_tokens_seen
            if chunk != row.prompt_ids[start : start + len(chunk)]:
                raise RuntimeError("Qwen4Exp native prefill chunk drift")
            row.prefill_tokens_seen += len(chunk)
            if int(row.request.max_tokens) == 0:
                continue
            if row.runner is None:
                raise RuntimeError("Qwen4Exp admitted row has no runner")
            if row.prefill_tokens_seen == len(row.prompt_ids):
                row.next_result = row.runner.prefill(
                    row.prompt_ids,
                    **_qwen4_exp_compact_prefill_kwargs(),
                )
                raise_if_generation_deadline_expired(row.request)

    def decode_batch(
        self, work: WorkItem, *, commit: bool
    ) -> tuple[GeneratedToken, ...]:
        if not commit:
            raise ValueError("Qwen4Exp native decode requires commit=True")
        generated: list[GeneratedToken] = []
        for request_id in work.request_ids:
            row = self._row(request_id)
            raise_if_generation_deadline_expired(row.request)
            if int(row.request.max_tokens) == 0:
                finish = FinishDetails(
                    reason="length", length_limit=0, sampler_mode="greedy"
                )
                generated.append(
                    GeneratedToken(
                        int(request_id), 0, finished=True,
                        stream_chunk=GenerationStreamChunk(
                            text="", finish_details=finish,
                            generated_token_ids=(),
                        ),
                    )
                )
                continue
            if row.next_result is None or row.runner is None:
                raise RuntimeError("Qwen4Exp native row is not prefilled")
            token = int(row.next_result.token_id)
            row.generated_ids.append(token)
            finish = self._finish(row)
            visible = self._visible_ids(row, finish)
            full_text = self.generator.tokenizer.decode(
                visible, skip_special=False
            )
            delta = (
                full_text[len(row.emitted_text) :]
                if full_text.startswith(row.emitted_text)
                else full_text
            )
            row.emitted_text = full_text
            generated.append(
                GeneratedToken(
                    int(request_id), token, finished=finish is not None,
                    stream_chunk=GenerationStreamChunk(
                        text=delta,
                        finish_details=finish,
                        generated_token_ids=(
                            tuple(row.generated_ids) if finish is not None else None
                        ),
                    ),
                )
            )
            if finish is None:
                row.next_result = row.runner.step(
                    token,
                    **_qwen4_exp_compact_step_kwargs(),
                )
                raise_if_generation_deadline_expired(row.request)
        return tuple(generated)

    def compact_batch(self, moves: Sequence[SlotMove]) -> None:
        for move in moves:
            self._row(move.request_id)

    def reclaim(self, completed: CompletedRequest) -> None:
        row = self._rows.pop(int(completed.request_id), None)
        if row is None:
            return
        finish = completed.finish_details
        if completed.finish_reason not in {"cancel", "disconnect", "timeout"}:
            finish = self._finish(row) or completed.finish_details
        visible = self._visible_ids(row, finish)
        self._outputs[row.request_id] = GenerationOutput(
            text=self.generator.tokenizer.decode(visible, skip_special=False),
            generated_token_ids=tuple(row.generated_ids),
            finish_details=finish,
        )
        self._release_runner(row)

    def has_outputs(self, request_ids: Sequence[int]) -> bool:
        return all(int(value) in self._outputs for value in request_ids)

    def missing_outputs(self, request_ids: Sequence[int]) -> list[int]:
        return [int(value) for value in request_ids if int(value) not in self._outputs]

    def take_outputs(self, request_ids: Sequence[int]) -> list[GenerationOutput]:
        return [self._outputs.pop(int(value)) for value in request_ids]

    def discard(self, request_ids: Sequence[int]) -> None:
        for request_id in request_ids:
            rid = int(request_id)
            row = self._rows.pop(rid, None)
            if row is not None:
                self._release_runner(row)
            self._outputs.pop(rid, None)

    def finalize_batch(
        self,
        request: GenerationRequest,
        request_ids: Sequence[int],
        outputs: Sequence[GenerationOutput],
    ) -> None:
        del request, request_ids, outputs

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for row in tuple(self._rows.values()):
            self._release_runner(row)
        self._rows.clear()
        self._outputs.clear()
        primary = self.generator.runner
        for runner in reversed(self._all_runners):
            if runner is not primary:
                runner.close()
        self._all_runners.clear()
        self._available.clear()
        self.generator._resident_model_runner = None
        self.generator.close()

    def _row(self, request_id: int) -> _Qwen4ExpServingRow:
        try:
            return self._rows[int(request_id)]
        except KeyError as exc:
            raise KeyError(f"unknown Qwen4Exp request_id {request_id}") from exc

    def _release_runner(self, row: _Qwen4ExpServingRow) -> None:
        runner, row.runner = row.runner, None
        if runner is not None and runner in self._all_runners and runner not in self._available:
            self._available.append(runner)

    def _finish(self, row: _Qwen4ExpServingRow) -> FinishDetails | None:
        ids = tuple(row.generated_ids)
        request = row.request
        enough = len(ids) >= int(request.min_tokens)
        if enough and ids and not request.ignore_eos and ids[-1] == self.generator.tokenizer.eos_token_id:
            return FinishDetails(
                reason="eos", eos_token_id=self.generator.tokenizer.eos_token_id,
                sampler_mode="greedy",
            )
        if enough and ids and ids[-1] in set(int(value) for value in request.stop_token_ids):
            return FinishDetails(
                reason="stop", stop_sequence=(ids[-1],), sampler_mode="greedy"
            )
        if enough:
            for sequence in request.stop_token_sequences:
                stop = tuple(int(value) for value in sequence)
                if stop and len(stop) <= len(ids) and ids[-len(stop) :] == stop:
                    return FinishDetails(reason="stop", stop_sequence=stop, sampler_mode="greedy")
        if len(ids) >= int(request.max_tokens):
            return FinishDetails(
                reason="length", length_limit=int(request.max_tokens),
                sampler_mode="greedy",
            )
        return None

    def _visible_ids(
        self, row: _Qwen4ExpServingRow, finish: FinishDetails | None
    ) -> tuple[int, ...]:
        ids = tuple(row.generated_ids)
        if finish is None or finish.reason == "length":
            return ids
        if finish.stop_sequence:
            return ids[: -len(finish.stop_sequence)]
        if finish.reason in {"stop", "eos"} and ids:
            return ids[:-1]
        return ids


def make_qwen4_exp_gguf_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: object,
    model_plugin: object,
    vision_model_path: str | Path | None = None,
    max_sequence_length: int | None = None,
    resident_capacity: int | None = None,
) -> Qwen4ExpGGUFTextGenerator:
    return Qwen4ExpGGUFTextGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1151",
        vision_model_path=vision_model_path,
        max_sequence_length=max_sequence_length,
        resident_capacity=resident_capacity,
    )


for _quant in ("gguf_q4_k_m", "gguf_ud_q4_k_xl"):
    register_text_generator(
        model="qwen4_exp_gguf",
        backend="hip_gfx1151",
        quant=_quant,
        factory=make_qwen4_exp_gguf_generator_gfx1151,
    )


__all__ = [
    "Qwen4ExpGGUFTextGenerator",
    "make_qwen4_exp_gguf_generator_gfx1151",
]
