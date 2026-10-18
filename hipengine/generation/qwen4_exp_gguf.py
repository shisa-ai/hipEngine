"""Strict Qwen3.8-Flash-Next qwen4exp GGUF text-generation plugin."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import threading
import time
from typing import Any, Sequence

from hipengine.dispatch import RequestState, SlotMove, WorkItem
from hipengine.generation.batch_scheduler import CompletedRequest, GeneratedToken
from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.finish import finish_details_with_sampling_state
from hipengine.generation.sampling import (
    RowSamplingState, SamplingMode, plan_sampler, row_seed_for_index,
    select_token, thinking_budget_state_from_params,
)
from hipengine.generation.qwen4_exp_serving_mtp import Qwen4ExpServingMTP, _RequestRetired
from hipengine.generation.qwen4_exp_ar_batch import (
    AR_BATCH_ENV,
    AR_BATCH_QUALIFIED_MAX_POSITIONS,
    AR_BATCH_QUALIFIED_MAX_SLOTS,
    Qwen4ExpARBatchOwner,
    ar_batch_capacity_ceiling,
    ar_batch_enabled,
    ar_batch_state_slots,
    make_qwen4_exp_ar_batch_runner,
)
from hipengine.generation.qwen4_exp_multimodal import (
    qwen4_exp_multimodal_token_control,
    render_qwen4_exp_multimodal_prompt,
)
from hipengine.generation.logprobs import token_bytes_for_id
from hipengine.generation.registry import (
    DecodeState,
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    GenerationTelemetry,
    TokenLogprob,
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

_T16_RESIDENT_ENV = "HIPENGINE_QWEN4_EXP_T16_RESIDENT"
_T16_RESIDENT_MAX_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_T16_RESIDENT_MAX_LAYERS"
_T16_RESIDENT_SLOTS_ENV = "HIPENGINE_QWEN4_EXP_T16_RESIDENT_SLOTS"
_T16_RESIDENT_SLOTS = ("expert_gate", "expert_up", "expert_down")


def _t16_resident_enabled() -> bool:
    """Default-off binding-owned T16 tile residency at load (design: the
    docs/QWEN4EXP-T16-RESIDENCY.md mechanism; the routes stay behind X-7)."""

    import os

    return os.environ.get(_T16_RESIDENT_ENV, "0") not in {
        "", "0", "false", "False",
    }


def _t16_resident_max_layers() -> int | None:
    import os

    raw = os.environ.get(_T16_RESIDENT_MAX_LAYERS_ENV, "")
    if not raw:
        return None
    return int(raw)


def _t16_resident_slot_suffixes() -> tuple[str, ...]:
    """Routed-expert slot roles to populate; empty or unset selects all."""

    import os

    raw = os.environ.get(_T16_RESIDENT_SLOTS_ENV, "")
    names = tuple(name.strip() for name in raw.split(",") if name.strip())
    unknown = sorted(set(names) - set(_T16_RESIDENT_SLOTS))
    if unknown:
        raise ValueError(
            f"{_T16_RESIDENT_SLOTS_ENV} names unknown slot roles: {unknown!r}"
        )
    return tuple("." + name for name in (names or _T16_RESIDENT_SLOTS))


def _populate_t16_resident_stages(
    resident,
    readers,
    *,
    runtime=None,
    slot_suffixes: tuple[str, ...] | None = None,
    max_layers: int | None = None,
) -> dict:
    """Populate routed-expert t16 binding stages from the GGUF readers.

    For every selected routed-expert slot in the residency plan, read the raw
    tensor, repack through the production T16 layout, upload, and attach as
    the weight's own ``t16`` allocation stage (raw stays; the free_raw detach
    is a separate explicit phase). The T16 routes then dispatch from the
    weights with no module-level sidecar cache. Slots that already carry a
    stage are skipped. Failures are fatal to the caller - a half-populated
    residency would dispatch half the layers.
    """

    from hipengine.loading.qwen4_exp_materialize import (
        materialize_qwen4_exp_t16_stages,
    )

    suffixes = slot_suffixes or tuple("." + name for name in _T16_RESIDENT_SLOTS)
    slots = [
        spec.slot_path
        for layer in resident.plan.layer_specs
        for spec in layer.values()
        if spec.slot_path.endswith(suffixes)
        and "t16" not in getattr(resident.weight(spec.slot_path), "allocations", {})
    ]
    if max_layers is not None:
        # Bound the populate window by MoE layer index: the raw stages stay
        # resident until the explicit free_raw phase, so an unbounded
        # populate is a transient dual residency the host may not fit. The
        # bound takes layers in plan order (numerical layer order).
        seen_layers: set[str] = set()
        bounded: list[str] = []
        for slot in slots:
            layer = slot.rsplit(".", 1)[0]
            if len(seen_layers) < max_layers or layer in seen_layers:
                seen_layers.add(layer)
                bounded.append(slot)
        slots = bounded
    by_name = {}
    for reader in readers:
        for info in reader.info.tensors:
            by_name[info.name] = reader

    def read_tensor(name):
        reader = by_name.get(name)
        if reader is None:
            raise KeyError(f"GGUF tensor not found for T16 residency: {name}")
        return reader.tensor_data(name)

    return materialize_qwen4_exp_t16_stages(
        resident,
        read_tensor=read_tensor,
        slots=slots,
        runtime=runtime,
    )


def _populate_t16_resident_stages_from_env(resident, readers, *, runtime=None) -> dict:
    """Default-off construction populate (HIPENGINE_QWEN4_EXP_T16_RESIDENT)."""

    if not _t16_resident_enabled():
        return {}
    return _populate_t16_resident_stages(
        resident,
        readers,
        runtime=runtime,
        slot_suffixes=_t16_resident_slot_suffixes(),
        max_layers=_t16_resident_max_layers(),
    )


def _qwen4_exp_compact_prefill_kwargs() -> dict[str, bool]:
    return {"capture_logits": False, "capture_target_hidden": False}


def _qwen4_exp_compact_step_kwargs() -> dict[str, bool]:
    return {
        "capture_logits": False,
        "capture_target_hidden": False,
        "token_id_resident": True,
    }


def _sampling_state(request, prompt_ids, row_index=0):
    if not plan_sampler(request).uses_host_logits and request.grammar is None:
        return None
    return RowSamplingState(
        prompt_tokens=prompt_ids, seed=row_seed_for_index(request, row_index),
        row_index=row_index, thinking_budget=thinking_budget_state_from_params(request),
        stop_token_sequences=request.stop_token_sequences,
        forced_tokens_pending=request.forced_tokens_pending,
        forced_token_reason=request.forced_token_reason,
        post_thinking_forced_tokens_pending=request.post_thinking_forced_tokens_pending,
        post_thinking_forced_token_reason=request.post_thinking_forced_token_reason,
        force_sequence_completion_token_sequences=request.force_sequence_completion_token_sequences,
        force_sequence_completion_reason=request.force_sequence_completion_reason,
        json_object_close_forcing=request.json_object_close_forcing,
        tool_call_constraint=request.tool_call_constraint,
    )


def _sampling_mode(request, grammar=None):
    if grammar is not None:
        return "grammar_greedy" if request.temperature == 0 else "grammar_host_logits_sample"
    mode = plan_sampler(request).mode
    return "greedy" if mode is SamplingMode.GREEDY_FAST else mode.value


def _validate_row_seeds(request):
    if request.row_seeds and len(request.row_seeds) != len(request.prompts):
        raise ValueError("row_seeds must have one entry per prompt")


def _token_finish(request, ids, grammar, default_eos):
    ids = tuple(ids)
    eos = default_eos if request.eos_token_id is None else request.eos_token_id
    if grammar is not None and grammar.done:
        is_eos = bool(ids and ids[-1] == eos)
        return FinishDetails(reason="eos" if is_eos else "stop",
                             eos_token_id=eos if is_eos else None,
                             sampler_mode="grammar_greedy")
    if len(ids) >= request.min_tokens and ids:
        if not request.ignore_eos and ids[-1] == eos:
            return FinishDetails(reason="eos", eos_token_id=eos, sampler_mode="greedy")
        if ids[-1] in request.stop_token_ids:
            return FinishDetails(reason="stop", stop_sequence=(ids[-1],), sampler_mode="greedy")
        for sequence in request.stop_token_sequences:
            if len(sequence) <= len(ids) and ids[-len(sequence):] == sequence:
                return FinishDetails(reason="stop", stop_sequence=sequence, sampler_mode="greedy")
    if len(ids) >= request.max_tokens:
        return FinishDetails(reason="length", length_limit=request.max_tokens, sampler_mode="greedy")
    return None


class Qwen4ExpGGUFTextGenerator:
    """Flash-Next text generation with request-owned host sampling controls."""

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
        prefill_chunk_size: int | None = 512,
        vision_model_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.weight_index = weight_index
        self.model_plugin = model_plugin
        self.backend = str(backend)
        production_capacity = type(self).server_plain_ar_max_active_requests
        capacity_ceiling = ar_batch_capacity_ceiling(
            max_positions=max_sequence_length,
            default=production_capacity,
        )
        self._configured_resident_capacity = (
            production_capacity if resident_capacity is None else int(resident_capacity))
        if not 1 <= self._configured_resident_capacity <= capacity_ceiling:
            raise ValueError(
                "Qwen4Exp resident capacity must be within "
                f"1..{capacity_ceiling} for max_sequence_length="
                f"{max_sequence_length}; the production bound is "
                f"{production_capacity}, and {AR_BATCH_ENV}=1 qualifies up to "
                f"{AR_BATCH_QUALIFIED_MAX_SLOTS} slots at <= "
                f"{AR_BATCH_QUALIFIED_MAX_POSITIONS} positions")
        # Publish the resolved capability on this instance only. The class
        # attribute stays the production bound, so a second generator in the
        # same process cannot inherit the qualification width.
        self.server_plain_ar_max_active_requests = self._configured_resident_capacity
        self.server_plain_ar_max_active_requests_by_max_sequence_length = {1024:self._configured_resident_capacity}
        self.context_admission = None
        # The chunk the admission was actually computed for. `None` means the
        # caller supplied a resident runner, so no admission ran and the binder
        # keeps its own default.
        self.admitted_prefill_chunk_size = None
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
        from hipengine.chat.qwen4_exp import Qwen4ExpToolParser,Qwen4ExpReasoningParser
        self.chat_tool_parser = Qwen4ExpToolParser()
        self.chat_reasoning_parser = Qwen4ExpReasoningParser(tokenizer)
        self._grammar_tokenizer = None
        vision_readers: tuple[GGUFReader, ...] = ()
        vision_plan = None
        if vision_model_path is not None:
            from hipengine.loading.qwen4_exp_vision_gguf import build_qwen4_exp_vision_gguf_map
            from hipengine.loading.qwen4_exp_vision_materialize import plan_qwen4_exp_vision_residency
            vision_readers = tuple(
                GGUFReader(path) for path in discover_gguf_files(Path(vision_model_path)))
            vision_plan = plan_qwen4_exp_vision_residency(
                build_qwen4_exp_vision_gguf_map(tuple(reader.info for reader in vision_readers)))
        if runner is None:
            paths = discover_gguf_files(self.model_path)
            readers = tuple(GGUFReader(path) for path in paths)
            model_map = build_qwen4_exp_gguf_tensor_map(
                tuple(reader.info for reader in readers)
            )
            # A `None` chunk means the capacity guard below chooses it, so the
            # plan is built there for the rung that wins.
            plan = (
                None if prefill_chunk_size is None
                else plan_qwen4_exp_residency(
                    model_map, staging_token_capacity=prefill_chunk_size)
            )
            from hipengine.core.hip import get_hip_runtime
            from hipengine.loading.qwen4_exp_context import resolve_qwen4_exp_context
            runtime = get_hip_runtime()
            free_bytes,total_bytes = runtime.mem_get_info()
            # Reserve what the vision weights actually occupy on device: only
            # mapped, device-resident specs are materialized, the same way the
            # text plan excludes the host-mmap PLE table. Summing every tensor
            # in the file happens to match today because vision map validation
            # rejects unexpected tensors, but it would over-reserve the moment
            # a non-resident vision tensor exists.
            vision_reserve = 0 if vision_plan is None else int(vision_plan.device_weight_bytes)
            if free_bytes < vision_reserve:
                raise MemoryError("Qwen4Exp vision weights exceed available device memory")
            native_context = getattr(
                model_plugin,"native_context_length",model_map.config.context_length)
            if prefill_chunk_size is None:
                # Capacity guard (ledger RT-1/RT-3): take the largest chunk whose
                # admission still yields the requested context, and fall back a
                # step at a time when it does not. The guard exists because the
                # workspace is CONTEXT-aware, not prompt-aware - rows are
                # min(chunk, native_context), so in a serving session the larger
                # chunk is paid at construction for every session regardless of
                # traffic, and a blanket flip is the wrong form.
                #
                # The admission below is pure arithmetic on the residency plan,
                # so the whole ladder is evaluated before any weight is
                # materialized and the fallback costs no device memory or time.
                prefill_chunk_size, plan, admission = _select_prefill_chunk(
                    model_map,
                    ladder=_AUTO_PREFILL_CHUNK_LADDER,
                    available_device_bytes=free_bytes-vision_reserve,
                    requested_context=max_sequence_length,
                    native_context_length=native_context,
                    resident_capacity=self._configured_resident_capacity,
                    plan_fn=plan_qwen4_exp_residency,
                    admit_fn=resolve_qwen4_exp_context)
            else:
                admission = resolve_qwen4_exp_context(
                    plan,available_device_bytes=free_bytes-vision_reserve,
                    requested_context=max_sequence_length,
                    native_context_length=native_context,
                    resident_capacity=self._configured_resident_capacity,
                    prefill_chunk_size=prefill_chunk_size)
            self.admitted_prefill_chunk_size = int(prefill_chunk_size)
            self.context_admission = {
                "mode": "auto" if max_sequence_length is None else "explicit",
                "device_free_bytes": free_bytes,"device_total_bytes": total_bytes,
                "vision_weight_reserve_bytes": vision_reserve,
                "plan": asdict(admission),
                "scratch_policy": "max(4GiB floor, mandatory runner scratch and repair queues) per runner; optional resources require separate qualification",
            }
            self._resident = materialize_qwen4_exp_weights(
                readers,
                plan=plan,
                backend=self.backend,
                runtime=runtime,
            )
            try:
                _populate_t16_resident_stages_from_env(
                    self._resident, readers, runtime=runtime
                )
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
        if vision_plan is not None:
            from hipengine.loading.qwen4_exp_vision_materialize import materialize_qwen4_exp_vision_weights
            from hipengine.runtime.qwen4_exp_vision import Qwen4ExpVisionRunner
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
    chat_template_family = "qwen4exp_embedded"
    supports_grammar = True
    model_owned_tool_grammar = True
    chat_template_reasoning_effort = True

    def _grammar_session(self, request):
        if request.grammar is None:
            return None
        from hipengine.generation.grammar import GrammarSession
        return GrammarSession(self.grammar_tokenizer,request.grammar)

    @property
    def grammar_tokenizer(self):
        from hipengine.generation.grammar import make_tokenizer
        with self._lock:
            if self._grammar_tokenizer is None:
                self._grammar_tokenizer = make_tokenizer(self.tokenizer)
            return self._grammar_tokenizer

    def _controlled_token(self, result, request, grammar, sampling_state=None, token_logprobs=None):
        if sampling_state is not None:
            params = request if request.eos_token_id is not None else replace(
                request, eos_token_id=self.tokenizer.eos_token_id,
            )
            scores = result.logits if grammar is None else grammar.mask_logits(result.logits)
            token_text = lambda token: self.tokenizer.decode((int(token),), skip_special=False)
            sampled = select_token(scores, params, sampling_state, token_text_for_id=token_text)
            if grammar is not None:
                import numpy as np
                if not np.isfinite(scores[sampled.token_id]):
                    raise ValueError("thinking close token violates output grammar")
                grammar.accept_token(sampled.token_id)
            if sampling_state.has_token_text_constraint:
                sampling_state.observe_selected_token_text(
                    token_text(sampled.token_id),
                    remaining_tokens=request.max_tokens - sampling_state.step_index,
                    encode_text=self.tokenizer.encode,
                )
            if token_logprobs is not None and (request.logprobs or request.top_logprobs):
                token_logprobs.append(TokenLogprob(
                    token_id=sampled.token_id, token_text=token_text(sampled.token_id),
                    logprob=sampled.logprob,
                    top_logprobs=tuple((token, token_text(token), lp) for token, lp in sampled.top_logprobs),
                    token_bytes=token_bytes_for_id(self.tokenizer, sampled.token_id),
                    top_token_bytes=tuple(token_bytes_for_id(self.tokenizer, token) for token, _ in sampled.top_logprobs),
                ))
            return sampled.token_id
        return int(result.token_id)

    def render_chat_prompt(self, messages, *, tools=None, enable_thinking=False,
                           add_generation_prompt=True, reasoning_effort=None):
        from hipengine.chat.qwen4_exp import render_qwen4_exp_chat
        return render_qwen4_exp_chat(
            self.tokenizer.chat_template,messages,tools=tools,
            enable_thinking=enable_thinking,add_generation_prompt=add_generation_prompt,
            reasoning_effort=reasoning_effort)

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
            "decode": resident.decode_telemetry(),
        }

    def decode_telemetry(self) -> dict[str, Any]:
        """Report the resident serving decode path and its physical batch state.

        Read-only and safe to call while requests are in flight: the resident
        runner snapshots its counters under its own lock. Returns
        ``decode_path = uninitialized`` before the engine loop builds the
        resident runner, and ``unknown`` for a resident runner that does not
        report decode telemetry.
        """

        with self._lock:
            resident = self._resident_model_runner
        if resident is None:
            return {"decode_path": "uninitialized", "ar_batch": None}
        probe = getattr(resident, "decode_telemetry", None)
        if not callable(probe):
            return {"decode_path": "unknown", "ar_batch": None}
        payload = probe()
        if not isinstance(payload, dict):
            return {"decode_path": "unknown", "ar_batch": None}
        return dict(payload)

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
        _validate_row_seeds(request)
        outputs: list[GenerationOutput] = []
        for row_index, prompt in enumerate(request.prompts):
            raise_if_generation_deadline_expired(request)
            token_ids = (
                [int(token) for token in prompt]
                if not isinstance(prompt, str)
                else [int(token) for token in self.tokenizer.encode(prompt)]
            )
            if len(token_ids) + request.max_tokens > self.runner.max_sequence_length:
                raise ValueError("Qwen4Exp request exceeds admitted sequence capacity")
            grammar = self._grammar_session(request)
            sampling_state = _sampling_state(request, token_ids, row_index)
            token_logprobs: list[TokenLogprob] = []
            controlled = sampling_state is not None or grammar is not None or bool(request.suppress_token_ids or request.logit_bias)
            prefill_kwargs = _qwen4_exp_compact_prefill_kwargs()
            step_kwargs = _qwen4_exp_compact_step_kwargs()
            if controlled:
                prefill_kwargs["capture_logits"] = True
                step_kwargs.update(capture_logits=True,token_id_resident=False)
            result = None if request.max_tokens == 0 else self.runner.prefill(
                token_ids,
                **prefill_kwargs,
            )
            raise_if_generation_deadline_expired(request)
            generated: list[int] = []
            finish = FinishDetails(reason="length", length_limit=request.max_tokens)
            for index in range(request.max_tokens):
                raise_if_generation_deadline_expired(request)
                token = self._controlled_token(result, request, grammar, sampling_state, token_logprobs)
                generated.append(token)
                finish = _token_finish(request, generated, grammar, self.tokenizer.eos_token_id)
                if finish is not None:
                    break
                if index + 1 < request.max_tokens:
                    result = self.runner.step(
                        token,
                        **step_kwargs,
                    )
            outputs.append(
                GenerationOutput(
                    text=self.tokenizer.decode(generated, skip_special=True),
                    generated_token_ids=tuple(generated),
                    token_logprobs=tuple(token_logprobs),
                    finish_details=finish_details_with_sampling_state(
                        replace(finish, sampler_mode=_sampling_mode(request, grammar)), sampling_state,
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

    @property
    def supports_speculative_grammar(self) -> bool:
        return bool(getattr(self._speculative_provider, "supports_grammar", False))

    def attach_speculative_provider(self, provider: Any) -> None:
        self._require_open()
        if self._speculative_provider is not None:
            raise RuntimeError("Qwen4Exp generator already has a speculative provider")
        self._speculative_provider = provider

    def generate_speculative_detailed(
        self, request: GenerationRequest
    ) -> list[GenerationOutput]:
        self._require_open()
        if request.grammar is not None and not self.supports_speculative_grammar:
            raise NotImplementedError("grammar-constrained speculative decoding is not supported")
        if self._speculative_provider is None:
            raise NotImplementedError("Qwen4Exp speculative provider is not attached")
        return list(self._speculative_provider.generate_detailed(request))

    def generate_speculative_mtp_detailed(
        self, request: GenerationRequest
    ) -> list[GenerationOutput]:
        return self.generate_speculative_detailed(request)

    def stream_speculative_detailed(self, request: GenerationRequest):
        self._require_open()
        if request.grammar is not None and not self.supports_speculative_grammar:
            raise NotImplementedError("grammar-constrained speculative decoding is not supported")
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
        if request.grammar is not None:
            raise NotImplementedError("multimodal grammar decoding is not supported")
        raise_if_generation_deadline_expired(request)
        if self._vision_runner is None:
            raise NotImplementedError("Qwen4Exp vision model is not attached")
        if len(request.prompts) != 1:
            raise ValueError("Qwen4Exp multimodal generation supports one request")
        if plan_sampler(request).uses_host_logits:
            raise NotImplementedError("Qwen4Exp multimodal sampling and logprobs are not supported")
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
            text=self.tokenizer.decode(generated, skip_special=True),
            generated_token_ids=tuple(generated),
            finish_details=FinishDetails(
                reason=reason,
                eos_token_id=(
                    self.tokenizer.eos_token_id if reason == "eos" else None
                ),
                length_limit=(request.max_tokens if reason == "length" else None),
                sampler_mode="greedy_multimodal_mrope",
            ),
            telemetry=GenerationTelemetry(
                decode_state=DecodeState(
                    prompt_tokens=len(token_ids),
                    generated_tokens=len(generated),
                ),
                # The rendered prompt carries the chat template and one vision
                # pad per encoded patch, so counting the caller's raw text would
                # understate what the model actually read.
                usage={"prompt_tokens": len(token_ids)},
            ),
        )

    def populate_t16_resident_stages(self, slots: Sequence[str]) -> dict:
        """Make the named routed-expert roles' T16 tiles resident.

        The profile binder's entry point: a profile whose plan selects T16
        routes calls this once at bind time. ``slots`` names roles
        (``expert_gate``/``expert_up``/``expert_down``); roles whose tiles are
        already resident are left alone.
        """

        unknown = sorted(set(slots) - set(_T16_RESIDENT_SLOTS))
        if unknown:
            raise ValueError(f"unknown T16 residency slot roles: {unknown!r}")
        if self._resident is None or not slots:
            return {}
        readers = tuple(
            GGUFReader(path) for path in discover_gguf_files(self.model_path)
        )
        populated = _populate_t16_resident_stages(
            self._resident,
            readers,
            runtime=self.runner.runtime,
            slot_suffixes=tuple("." + name for name in slots),
        )
        if populated:
            # The runner's layer bindings hold weight objects; rebind so the
            # new stages are the ones dispatch sees.
            self.runner.rebind_resident_weights()
        return populated

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
    grammar: Any | None = None
    lifetime_request: GenerationRequest | None = None
    mtp_budget: int = 0
    draft: Any | None = None
    mtp_steps: Any | None = None
    mtp_output: GenerationOutput | None = None
    previous_hidden: Any | None = None
    sampling_state: RowSamplingState | None = None
    token_logprobs: list[TokenLogprob] = field(default_factory=list)

    @property
    def controlled(self):
        return self.sampling_state is not None or self.grammar is not None or bool(self.request.suppress_token_ids or self.request.logit_bias)


class Qwen4ExpResidentServingRunner(Qwen4ExpServingMTP):
    """Request-owned c2 runner pool over one shared Qwen4Exp weight layout."""

    supports_stream_token_logprobs = True

    def __init__(self, generator: Qwen4ExpGGUFTextGenerator, *, capacity: int) -> None:
        self.generator = generator
        self.capacity = int(capacity)
        self._rows: dict[int, _Qwen4ExpServingRow] = {}
        self._outputs: dict[int, GenerationOutput] = {}
        self._all_runners = [generator.runner]
        self._available = [generator.runner]
        # Opt-in full-model AR decode batch. ``_ar_batch_mode`` is pinned at the
        # first prepare() so one serving runner never mixes the c1 step path and
        # the shared N-row step path across requests.
        self._ar_batch: Qwen4ExpARBatchOwner | None = None
        self._ar_batch_mode: str | None = None
        self._closed = False
        self._init_mtp()

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
        self._prepare_ar_batch()
        self._prepare_mtp()
        return required

    def _prepare_ar_batch(self) -> None:
        """Select and build the opt-in AR decode path exactly once.

        A build failure is not cached: the next prepare() retries and raises
        again, so a caller that ignored the first error cannot fall through to
        the c1 path with the switch still on.
        """

        if self._ar_batch is not None or self._ar_batch_mode is not None:
            return
        if not ar_batch_enabled():
            self._ar_batch_mode = "qwen4exp_c1_request_owned_step"
            return
        if self.generator._resident is None:
            # Fail closed instead of silently staying on c1: the operator asked
            # for the batch path and this generator cannot own its weights.
            raise RuntimeError(
                "Qwen4Exp AR batch decode requires model-owned resident weights; "
                "this generator was built with an external runner"
            )
        slots = ar_batch_state_slots(self.capacity)
        owner = Qwen4ExpARBatchOwner(
            batch_factory=lambda: make_qwen4_exp_ar_batch_runner(
                self.generator, state_slots=slots
            ),
            state_slots=slots,
            max_positions=self.generator.runner.max_sequence_length,
        )
        owner.ensure()
        self._ar_batch = owner
        self._ar_batch_mode = owner.label

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
            "decode": self.decode_telemetry(),
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
        _validate_row_seeds(request)
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
                request_id, row_index, request, prompt_ids, now,
                grammar=self.generator._grammar_session(request),
                sampling_state=_sampling_state(request, prompt_ids, row_index),
            )

    def reserve_admission(self, request: RequestState) -> None:
        self.prepare()
        row = self._row(request.request_id)
        if int(row.request.max_tokens) == 0:
            return
        if not self._available:
            raise RuntimeError("Qwen4Exp scheduler has no free request runner")
        row.runner = self._available.pop()
        try:
            row.runner.reset()
            self._reserve_mtp(row)
            self._reserve_ar_batch(row)
        except BaseException:
            self._release_runner(row)
            raise

    def plan_admission(self, pending, *, max_items: int):
        """Admit only as many AR rows as the shared batch can hold.

        With the default width (one state slot per serving request) this is a
        pure prefix. A narrower ``HIPENGINE_QWEN4_EXP_AR_BATCH_SLOTS`` leaves
        the extra rows pending instead of failing admission inside the tick.
        """

        owner = self._ar_batch
        selected: list[int] = []
        free = 0 if owner is None else owner.free_slot_count
        for request in tuple(pending)[: int(max_items)]:
            request_id = int(request.request_id)
            row = self._rows.get(request_id)
            needs_slot = owner is not None and row is not None and not row.mtp_budget
            if needs_slot:
                if free <= 0:
                    continue
                free -= 1
            selected.append(request_id)
        return tuple(selected)

    def _reserve_ar_batch(self, row) -> None:
        owner = self._ar_batch
        if owner is None or row.mtp_budget:
            # MTP rows keep their request-owned c1 runner at width 1; a
            # speculative candidate is never a batch row.
            return
        required = len(row.prompt_ids) + max(0, int(row.request.max_tokens) - 1)
        owner.reserve(row.request_id, positions=required)

    def rollback_admission(self, request: RequestState) -> None:
        self._release_runner(self._row(request.request_id))

    def prefill_batch(self, work: WorkItem, *, commit: bool) -> None:
        if not commit:
            raise ValueError("Qwen4Exp native prefill requires commit=True")
        for request_id, token_row in zip(
            work.request_ids, work.token_rows, strict=True
        ):
            row = self._row(request_id)
            try:
                self._prefill_row(row, token_row)
            except _RequestRetired:
                # The engine probes request_cancel_reason and reclaims this row
                # after the bounded work item; a peer remains runnable.
                continue

    def _prefill_row(self, row, token_row):
        self._check_request_live(row)
        chunk = tuple(int(token) for token in token_row)
        start = row.prefill_tokens_seen
        if chunk != row.prompt_ids[start : start + len(chunk)]:
            raise RuntimeError("Qwen4Exp native prefill chunk drift")
        if int(row.request.max_tokens) == 0:
            row.prefill_tokens_seen += len(chunk)
            return
        if row.runner is None:
            raise RuntimeError("Qwen4Exp admitted row has no runner")
        prefill_kwargs = _qwen4_exp_compact_prefill_kwargs()
        if row.controlled:
            prefill_kwargs["capture_logits"] = True
        if row.mtp_budget:
            prefill_kwargs["capture_hidden_seeds"] = True
        result = row.runner.prefill_append(chunk, **prefill_kwargs)
        self._check_request_live(row)
        if row.mtp_budget:
            self._prefill_mtp(row, chunk, result)
        row.prefill_tokens_seen += len(chunk)
        if row.prefill_tokens_seen == len(row.prompt_ids):
            row.next_result = result
            self._import_ar_batch(row)

    def _import_ar_batch(self, row) -> None:
        """Move a completed c1 prefill into the row's batch slot, once."""

        owner = self._ar_batch
        if owner is None or not owner.owns(row.request_id):
            return
        owner.import_prefilled(row.request_id, row.runner)

    def decode_batch(
        self, work: WorkItem, *, commit: bool
    ) -> tuple[GeneratedToken, ...]:
        if not commit:
            raise ValueError("Qwen4Exp native decode requires commit=True")
        generated: list[GeneratedToken | None] = []
        # Rows whose state lives in the shared AR batch. Their next token is
        # sampled first and then stepped once for the whole row set, so
        # publication waits until that single call returns.
        batched: list[tuple[int, _Qwen4ExpServingRow, int]] = []
        for request_id in work.request_ids:
            row = self._row(request_id)
            if self.request_cancel_reason(request_id) is not None:
                continue
            if self._ar_batch is not None and self._ar_batch.failed(request_id):
                raise RuntimeError(
                    f"Qwen4Exp request {request_id} lost its AR batch slot to a "
                    "failed step; it cannot continue on the c1 path"
                )
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
            if row.mtp_budget:
                progress, _ = self._advance_mtp(row, 0)
                if progress is None:
                    continue
                if len(progress.token_ids) != 1:
                    raise RuntimeError("Flash-Next root/K0 must emit one token")
                generated.append(
                    self._published_token(row, progress.token_ids[0], self._finish(row))
                )
                continue
            token = self.generator._controlled_token(
                row.next_result, row.request, row.grammar, row.sampling_state, row.token_logprobs,
            )
            row.generated_ids.append(token)
            finish = self._finish(row)
            if finish is None and self._ar_batch_owns(row):
                # A placeholder keeps the published order equal to the work
                # item's request order; it is filled or dropped below.
                generated.append(None)
                batched.append((len(generated) - 1, row, token))
                continue
            if finish is None:
                step_kwargs = _qwen4_exp_compact_step_kwargs()
                if row.controlled:
                    step_kwargs.update(capture_logits=True, token_id_resident=False)
                row.next_result = row.runner.step(token, **step_kwargs)
                if self.request_cancel_reason(request_id) is not None:
                    row.generated_ids.pop()
                    continue
            generated.append(self._published_token(row, token, finish))
        if batched:
            self._step_ar_batch([(row, token) for _, row, token in batched])
            for index, row, token in batched:
                if self.request_cancel_reason(row.request_id) is not None:
                    # The shared step advanced this slot, but the row cancelled
                    # while it ran. Publish nothing; reclaim releases the slot.
                    row.generated_ids.pop()
                    continue
                generated[index] = self._published_token(row, token, None)
        return tuple(item for item in generated if item is not None)

    def _published_token(self, row, token, finish):
        visible = self._visible_ids(row, finish)
        delta = self._stream_text_delta(row, visible, final=finish is not None)
        return GeneratedToken(
            int(row.request_id), token, finished=finish is not None,
            stream_chunk=GenerationStreamChunk(
                text=delta,
                token_logprobs=tuple(row.token_logprobs[-1:]) if len(visible) == len(row.generated_ids) else (),
                finish_details=finish,
                generated_token_ids=(
                    tuple(row.generated_ids) if finish is not None else None
                ),
            ),
        )

    def _ar_batch_owns(self, row) -> bool:
        owner = self._ar_batch
        return owner is not None and owner.owns(row.request_id)

    def _step_ar_batch(self, rows) -> None:
        """Advance every batched row with exactly one N-row batch call."""

        owner = self._ar_batch
        if owner is None:
            raise RuntimeError("Qwen4Exp AR batch step without a batch owner")
        results = owner.step(
            [(row.request_id, token) for row, token in rows],
            # One readback serves the whole row set: a controlled row needs its
            # full logits, an uncontrolled row only needs the committed argmax.
            capture_logits=any(row.controlled for row, _ in rows),
        )
        for row, _ in rows:
            row.next_result = results[row.request_id]

    def _stream_text_delta(self, row, visible, *, final):
        text = self.generator.tokenizer.decode(visible, skip_special=True)
        if not final:
            # Byte-fallback tokens may end inside a UTF-8 codepoint. Do not
            # publish a replacement character that the next token will repair.
            text = text.rstrip("\ufffd")
        if text.startswith(row.emitted_text):
            delta = text[len(row.emitted_text):]
        elif final and row.emitted_text.startswith(text):
            # A terminal stop suffix is held by the service's token filter.
            delta = ""
        else:
            raise RuntimeError("Flash-Next tokenizer rewrote published text")
        row.emitted_text = text
        return delta

    def prepare_speculative_stream_events(self, request_ids):
        rows = {rid: self._row(rid) for rid in request_ids}
        return lambda events: self._decorate_speculative_stream_events(events, rows)

    def _decorate_speculative_stream_events(self, events, rows):
        """Decode only scheduler-committed IDs, never provisional candidates."""
        remaining = {}
        for event in events:
            remaining[event.request_id] = remaining.get(event.request_id, 0) + 1
        decorated = []
        for event in events:
            row = rows[event.request_id]
            index = len(row.generated_ids) - remaining[event.request_id]
            remaining[event.request_id] -= 1
            finish = self._finish(row) if event.finished else None
            visible = self._visible_ids(row, finish) if event.finished else tuple(row.generated_ids[:index + 1])
            delta = self._stream_text_delta(row, visible, final=event.finished)
            chunk = replace(
                event.stream_chunk,
                text=delta,
                token_logprobs=tuple(row.token_logprobs[index:index + 1]) if index < len(visible) else (),
                finish_details=finish,
            )
            decorated.append(replace(event, stream_chunk=chunk))
        return tuple(decorated)

    def compact_batch(self, moves: Sequence[SlotMove]) -> None:
        # Physical batch slots are pinned per request at admission and never
        # follow the scheduler's slot indices, so compaction moves no state.
        for move in moves:
            row = self._row(move.request_id)
            if row.mtp_budget and self._ar_batch is not None and self._ar_batch.owns(
                move.request_id
            ):
                raise RuntimeError(
                    "a Flash-Next MTP row must not hold an AR batch slot"
                )

    def reclaim(self, completed: CompletedRequest) -> None:
        row = self._rows.pop(int(completed.request_id), None)
        if row is None:
            return
        finish = completed.finish_details
        if completed.finish_reason in {"cancel", "disconnect", "timeout"}:
            # A cancellation can arrive between native work and the engine's
            # publication check. Only scheduler-committed IDs are observable.
            row.generated_ids[:] = completed.generated_tokens
            row.mtp_output = None
        else:
            finish = self._finish(row) or completed.finish_details
        visible = self._visible_ids(row, finish)
        telemetry = None if row.mtp_output is None else row.mtp_output.telemetry
        if telemetry is None:
            telemetry = self._ar_batch_telemetry(row)
        self._outputs[row.request_id] = GenerationOutput(
            text=self.generator.tokenizer.decode(visible, skip_special=True),
            generated_token_ids=tuple(row.generated_ids),
            token_logprobs=tuple(row.token_logprobs[:len(visible)]),
            finish_details=finish,
            telemetry=telemetry,
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
        if self._ar_batch is not None:
            self._ar_batch.close()
            self._ar_batch = None
        self._close_mtp()
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

    def _ar_batch_telemetry(self, row: _Qwen4ExpServingRow) -> GenerationTelemetry | None:
        """Report the physical widths this row's own decode steps ran at.

        Read before ``_release_runner`` discards the lease. A row that never
        held a batch slot reports nothing here rather than a fabricated width,
        and an MTP row keeps its own provider telemetry.
        """

        owner = self._ar_batch
        if owner is None:
            return None
        batch = owner.request_telemetry(row.request_id)
        if batch is None:
            return None
        return GenerationTelemetry(
            decode_state=DecodeState(
                request_id=str(row.request_id),
                prompt_tokens=len(row.prompt_ids),
                generated_tokens=len(row.generated_ids),
                step_index=len(row.generated_ids),
                execution_path=str(batch["execution_path"]),
            ),
            event="generation_complete",
            diagnostics={"decode_batch": batch},
        )

    def _release_runner(self, row: _Qwen4ExpServingRow) -> None:
        self._release_mtp(row)
        owner = self._ar_batch
        if owner is not None:
            owner.release(row.request_id)
        runner, row.runner = row.runner, None
        if runner is not None and runner in self._all_runners and runner not in self._available:
            self._available.append(runner)

    def decode_telemetry(self) -> dict[str, Any]:
        """Report which decode path this runner selected, and its width."""

        owner = self._ar_batch
        return {
            "decode_path": self._ar_batch_mode or "qwen4exp_c1_request_owned_step",
            "ar_batch": None if owner is None else owner.telemetry(),
        }

    def _finish(self, row: _Qwen4ExpServingRow) -> FinishDetails | None:
        if row.mtp_output is not None:
            return row.mtp_output.finish_details
        details = self._raw_finish(row)
        if details is None:
            return None
        details = replace(details, sampler_mode=_sampling_mode(row.request, row.grammar))
        return finish_details_with_sampling_state(details, row.sampling_state)

    def _raw_finish(self, row: _Qwen4ExpServingRow) -> FinishDetails | None:
        return _token_finish(row.request, row.generated_ids, row.grammar, self.generator.tokenizer.eos_token_id)

    def _visible_ids(
        self, row: _Qwen4ExpServingRow, finish: FinishDetails | None
    ) -> tuple[int, ...]:
        ids = tuple(row.generated_ids)
        if finish is None or finish.reason == "length":
            return ids
        if row.grammar is not None and row.grammar.done and finish.reason == "stop" and not finish.stop_sequence:
            return ids
        if finish.stop_sequence:
            return ids[: -len(finish.stop_sequence)]
        if finish.reason in {"stop", "eos"} and ids:
            return ids[:-1]
        return ids


_AUTO_PREFILL_CHUNK_LADDER = (4096, 2048, 1024)


def _select_prefill_chunk(
    model_map: object,
    *,
    ladder: tuple[int, ...],
    available_device_bytes: int,
    requested_context: int | None,
    native_context_length: int,
    resident_capacity: int,
    plan_fn: object,
    admit_fn: object,
) -> tuple[int, object, object]:
    """Capacity guard for the prefill chunk (ledger `RT-1` / `RT-3`).

    Take the largest chunk in ``ladder`` whose admission still yields the
    requested context, falling back one step at a time when it does not.

    The workspace is context-aware rather than prompt-aware: ``allocate_inputs``
    sizes rows as ``min(prefill_chunk_size, max_sequence_length)`` and the
    runner's ``max_sequence_length`` comes from the model's native context, so
    in a serving session the larger chunk is paid at construction for every
    session regardless of traffic. A blanket default flip therefore spends
    device memory on requests that never use it; this picks the largest chunk
    the device can actually afford at the context being served.

    Both callbacks are pure arithmetic on the residency plan, so the whole
    ladder is evaluated before any weight is materialized.

    The requested context is what the caller asked for, or the native context
    when the caller left it open, so a fallback is only taken when the larger
    chunk would genuinely cost context rather than merely cost headroom.
    """
    wanted = int(native_context_length if requested_context is None else requested_context)
    failure: Exception | None = None
    for chunk in ladder:
        try:
            plan = plan_fn(model_map, staging_token_capacity=chunk)
            admission = admit_fn(
                plan, available_device_bytes=available_device_bytes,
                requested_context=requested_context,
                native_context_length=native_context_length,
                resident_capacity=resident_capacity,
                prefill_chunk_size=chunk)
        except MemoryError as error:  # chunk too large for this device
            failure = error
            continue
        if int(admission.context_tokens) >= wanted:
            return int(chunk), plan, admission
        failure = None
    if failure is not None:
        raise failure
    # No rung preserved the requested context; take the smallest and let the
    # admission report whatever context it can actually serve.
    chunk = int(ladder[-1])
    plan = plan_fn(model_map, staging_token_capacity=chunk)
    admission = admit_fn(
        plan, available_device_bytes=available_device_bytes,
        requested_context=requested_context,
        native_context_length=native_context_length,
        resident_capacity=resident_capacity,
        prefill_chunk_size=chunk)
    return chunk, plan, admission


def make_qwen4_exp_gguf_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: object,
    model_plugin: object,
    vision_model_path: str | Path | None = None,
    max_sequence_length: int | None = None,
    resident_capacity: int | None = None,
    prefill_chunk_size: int = 512,
) -> Qwen4ExpGGUFTextGenerator:
    return Qwen4ExpGGUFTextGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1151",
        vision_model_path=vision_model_path,
        max_sequence_length=max_sequence_length,
        resident_capacity=resident_capacity,
        prefill_chunk_size=prefill_chunk_size,
    )


def make_qwen4_exp_ud_q4_k_xl_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: object,
    model_plugin: object,
    vision_model_path: str | Path | None = None,
    max_sequence_length: int | None = None,
    resident_capacity: int | None = None,
    prefill_chunk_size: int | None = None,
) -> Qwen4ExpGGUFTextGenerator:
    generator = make_qwen4_exp_gguf_generator_gfx1151(
        model_path=model_path,weight_index=weight_index,model_plugin=model_plugin,
        vision_model_path=vision_model_path,max_sequence_length=max_sequence_length,
        resident_capacity=resident_capacity,
        # `None` asks the generator's capacity guard to take the largest chunk
        # whose admission still yields the requested context (ledger RT-1/RT-3),
        # falling back 4096 -> 2048 -> 1024 when the device cannot afford it.
        # The guard runs on the residency plan before any weight is
        # materialized, so it costs no device memory and no time, and it keeps
        # the pre-binding allocation consistent with what the profile will
        # bind - the binder can only lower the runner's chunk afterwards.
        prefill_chunk_size=None if prefill_chunk_size is None else prefill_chunk_size,
    )
    generator._implicit_profile_chunk_size = prefill_chunk_size is None
    return generator


register_text_generator(
    model="qwen4_exp_gguf",backend="hip_gfx1151",quant="gguf_q4_k_m",
    factory=make_qwen4_exp_gguf_generator_gfx1151,
)
register_text_generator(
    model="qwen4_exp_gguf",backend="hip_gfx1151",quant="gguf_ud_q4_k_xl",
    factory=make_qwen4_exp_ud_q4_k_xl_generator_gfx1151,
)


__all__ = [
    "Qwen4ExpGGUFTextGenerator",
    "make_qwen4_exp_gguf_generator_gfx1151",
    "make_qwen4_exp_ud_q4_k_xl_generator_gfx1151",
]
