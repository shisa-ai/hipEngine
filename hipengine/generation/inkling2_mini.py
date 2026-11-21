"""Public text generation adapter for the Inkling 2 Mini CPU reference."""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import replace

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import (
    DecodeState, FinishDetails, GenerationOutput, GenerationRequest, GenerationTelemetry, register_text_generator,
)
from hipengine.generation.sampling import RowSamplingState, row_seed_for_index, select_token
from hipengine.loading.inkling2_mini import Inkling2TextWeights
from hipengine.models.inkling2_mini import INKLING2_MINI
from hipengine.runtime.inkling2_mini_dspark import Inkling2DSpark
from hipengine.runtime.inkling2_mini_reference import Inkling2TextReference
from hipengine.runtime.inkling2_mini_vision import Inkling2VisionReference


class Inkling2ReferenceGenerator:
    def __init__(self, *, model_path, weight_index, model_plugin, max_sequence_length=None):
        self.model = Inkling2TextReference(weight_index.config, Inkling2TextWeights(weight_index))
        self.tokenizer = Tokenizer.from_file(str(weight_index.model_path / "tokenizer.json"))
        self.eos_token_id = int(weight_index.config["eos_token_id"])
        self.supports_vision = "vision_config" in weight_index.config
        self.vision = Inkling2VisionReference(weight_index) if self.supports_vision else None
        self.model_path = weight_index.model_path
        self.draft_path = self.model_path / "mtp.safetensors"
        self.supports_speculative_mtp = "mtp_config" in weight_index.config and self.draft_path.is_file()
        limit = self.model.config["model_max_length"]
        if max_sequence_length is not None and not 0 < max_sequence_length <= limit:
            raise ValueError(f"Inkling 2 reference context must be between 1 and {limit}")
        self.max_sequence_length = limit if max_sequence_length is None else max_sequence_length

    def tokenize(self, text):
        return tuple(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def detokenize(self, token_ids, *, skip_special=True):
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special)

    def generate(self, request):
        return [item.text for item in self.generate_detailed(request)]

    def generate_detailed(self, request: GenerationRequest):
        return self._generate_batch(request)

    def generate_speculative_mtp_detailed(self, request: GenerationRequest):
        return self._generate_batch(request, drafter=self._load_drafter())

    def _load_drafter(self):
        if not self.supports_speculative_mtp:
            raise NotImplementedError("Inkling 2 DSpark needs mtp_config and mtp.safetensors")
        return Inkling2DSpark(self.model, self.draft_path)

    def _generate_batch(self, request, *, drafter=None):
        self._validate_request(request, speculative=drafter is not None)
        if request.eos_token_id is None:
            request = replace(request, eos_token_id=self.eos_token_id)
        rows = tuple(request.prompt_token_ids(i, self.tokenize) for i in range(len(request.prompts)))
        for ids in rows:
            if not ids:
                raise ValueError("the Inkling 2 prompt produced no token IDs")
            if len(ids) + request.max_tokens > self.max_sequence_length:
                raise ValueError(f"Inkling 2 prompt and completion exceed the {self.max_sequence_length}-token context")
        return tuple(self._generate_row(request, i, ids, drafter=drafter) for i, ids in enumerate(rows))

    def generate_multimodal_detailed(self, prompt, image, request):
        return self._generate_image(prompt, image, request)

    def generate_multimodal_speculative_mtp_detailed(self, prompt, image, request):
        return self._generate_image(prompt, image, request, drafter=self._load_drafter())

    def _generate_image(self, prompt, image, request, *, drafter=None):
        if self.vision is None:
            raise NotImplementedError("Inkling 2 checkpoint does not declare a vision tower")
        self._validate_request(request, speculative=drafter is not None)
        raise_if_generation_deadline_expired(request)
        processor = json.loads((self.model_path / "processor_config.json").read_text(encoding="utf-8"))
        marker = processor["image_token"]
        marker_id = self.tokenizer.token_to_id(marker)
        if marker_id is None:
            raise ValueError("the Inkling 2 tokenizer has no declared image placeholder")
        if marker not in prompt:
            prompt = self._format_vision_prompt(prompt)
        prompt_ids = self.tokenize(prompt)
        if prompt_ids.count(marker_id) != 1:
            raise ValueError("Inkling 2 single-image generation requires exactly one image placeholder")
        start = prompt_ids.index(marker_id)
        features = self.vision.encode(image, check=lambda: raise_if_generation_deadline_expired(request))
        ids = prompt_ids[:start] + (marker_id,) * len(features) + prompt_ids[start + 1:]
        if len(ids) + request.max_tokens > self.max_sequence_length:
            raise ValueError(f"Inkling 2 image prompt and completion exceed the {self.max_sequence_length}-token context")
        if request.eos_token_id is None:
            request = replace(request, eos_token_id=self.eos_token_id)
        return self._generate_row(request, 0, ids, image_start=start, features=features, drafter=drafter)

    def _format_vision_prompt(self, text):
        source = (self.model_path / "chat_template.jinja").read_text(encoding="utf-8")
        # Hugging Face's generation tags annotate output spans but emit no text.
        source = re.sub(r"\{%-?\s*(?:end)?generation\s*-?%\}", "", source)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        return env.from_string(source).render(
            messages=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            tools=(), add_generation_prompt=True,
        )

    @staticmethod
    def _validate_request(request, *, speculative=False):
        unsupported = (
            (request.grammar is not None, "grammar"),
            (request.tool_call_constraint is not None, "tool_call_constraint"),
            (request.json_object_close_forcing, "json_object_close_forcing"),
            (request.thinking_hard_token_cap is not None or request.thinking_close_token_ids, "thinking budget"),
            (request.forced_tokens_pending or request.post_thinking_forced_tokens_pending, "forced token queues"),
            (request.force_sequence_completion_token_sequences, "forced sequence completion"),
            (request.logprobs or request.top_logprobs, "logprobs"),
            (request.resident_session_key is not None, "resident session reuse"),
            (not speculative and (request.speculative_mtp_static_eligibility is not None or request.speculative_mtp_automatic), "speculative decoding"),
            (request.kv_storage != "auto", f"kv_storage={request.kv_storage!r} (CPU reference stores FP32)"),
        )
        blockers = [name for active, name in unsupported if active]
        if blockers:
            raise NotImplementedError("Inkling 2 CPU reference does not implement: " + ", ".join(blockers))

    def _generate_row(self, request, row, ids, *, image_start=0, features=None, drafter=None):
        raise_if_generation_deadline_expired(request)
        generated = []
        reason, eos, stop_sequence, mode = "length", None, (), None
        proposed = accepted = 0
        candidates = deque()
        if request.max_tokens:
            capacity = len(ids) + request.max_tokens - 1
            session = self.model.session(capacity)
            draft = drafter.session(capacity) if drafter is not None else None
            taps = frozenset(drafter.taps) if draft is not None else ()

            def advance(token, **options):
                if draft is None:
                    return session.step(token, **options)
                captured = {}

                def capture(boundary, hidden):
                    if boundary in taps:
                        captured[boundary] = hidden

                result = session.step(token, capture=capture, **options)
                draft.append(captured)
                return result

            for i, token in enumerate(ids):
                raise_if_generation_deadline_expired(request)
                options = {"logits": i == len(ids) - 1}
                if features is not None and 0 <= i - image_start < len(features):
                    options["embedding"] = features[i - image_start]
                logits = advance(token, **options)
            state = RowSamplingState(prompt_tokens=ids, seed=row_seed_for_index(request, row), row_index=row)
            for i in range(request.max_tokens):
                raise_if_generation_deadline_expired(request)
                sampled = select_token(logits, request, state)
                token = sampled.token_id
                if candidates:
                    if candidates.popleft() == token:
                        accepted += 1
                    else:
                        candidates.clear()
                generated.append(token)
                mode = str(sampled.mode.value)
                if token in request.stop_token_ids or (not request.ignore_eos and token == request.eos_token_id):
                    reason, eos = "stop", token
                    break
                stop_sequence = next((seq for seq in request.stop_token_sequences
                                      if tuple(generated[-len(seq):]) == seq), ())
                if stop_sequence:
                    reason = "stop"
                    break
                remaining = request.max_tokens - i - 1
                if remaining:
                    if draft is not None and not candidates:
                        depth = min(remaining, drafter.cfg["block_size"] - 1)
                        candidates.extend(draft.propose(token, depth))
                        proposed += len(candidates)
                    logits = advance(token)
        telemetry = None if drafter is None else GenerationTelemetry(
            decode_state=DecodeState(prompt_tokens=len(ids), generated_tokens=len(generated)),
            diagnostics={"speculative_mtp": {"provider": "dspark", "verification": "serial_target",
                                             "proposed_tokens": proposed, "accepted_tokens": accepted}},
        )
        return GenerationOutput(
            self.detokenize(generated), generated_token_ids=tuple(generated), prompt_tokens=len(ids), telemetry=telemetry,
            finish_details=FinishDetails(reason=reason, eos_token_id=eos, stop_sequence=stop_sequence,
                                         length_limit=request.max_tokens if reason == "length" else None,
                                         sampler_mode=mode, phase="decode"),
        )


register_text_generator(model=INKLING2_MINI.name, backend="cpu_reference", quant=INKLING2_MINI.default_quant,
                        factory=Inkling2ReferenceGenerator)
