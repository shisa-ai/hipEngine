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
    packed_verification = False
    prefill_block_width = 1
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
        return Inkling2DSpark.from_safetensors(self.model, self.draft_path)

    def _generate_batch(self, request, *, drafter=None):
        request, rows = self._prepare_text_rows(request, speculative=drafter is not None)
        return tuple(self._generate_row(request, i, ids, drafter=drafter) for i, ids in enumerate(rows))

    def _prepare_text_rows(self, request, *, speculative=False):
        self._validate_request(request, speculative=speculative)
        if request.eos_token_id is None:
            request = replace(request, eos_token_id=self.eos_token_id)
        rows = tuple(request.prompt_token_ids(i, self.tokenize) for i in range(len(request.prompts)))
        for ids in rows:
            if not ids:
                raise ValueError("the Inkling 2 prompt produced no token IDs")
            if len(ids) + request.max_tokens > self.max_sequence_length:
                raise ValueError(f"Inkling 2 prompt and completion exceed the {self.max_sequence_length}-token context")
        return request, rows

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

    def _generate_row(self, request, row, ids, **options):
        steps = self._iterate_row(request, row, ids, **options)
        while True:
            try:
                next(steps)
            except StopIteration as done:
                return done.value

    def _iterate_row(self, request, row, ids, *, image_start=0, features=None, drafter=None):
        raise_if_generation_deadline_expired(request)
        capacity = len(ids) + request.max_tokens - 1
        session = self.model.session(capacity) if request.max_tokens else None
        draft = None
        try:
            if session is not None and drafter is not None:
                draft = drafter.session(capacity)
            output = yield from self._iterate_row_with_session(request, row, ids, session, draft,
                                                               image_start=image_start, features=features, drafter=drafter)
            return self._decorate_output(output)
        finally:
            try:
                close = getattr(draft, "close", None)
                if callable(close):
                    close()
            finally:
                close = getattr(session, "close", None)
                if callable(close):
                    close()

    def _decorate_output(self, output):
        return output

    def _iterate_row_with_session(self, request, row, ids, session, draft, *, image_start, features, drafter):
        generated = []
        reason, eos, stop_sequence, mode = "length", None, (), None
        proposed = accepted = blocks = 0
        packed = draft is not None and self.packed_verification
        candidates = deque()

        def emit(sampled):
            nonlocal reason, eos, stop_sequence, mode
            token = sampled.token_id
            generated.append(token)
            mode = str(sampled.mode.value)
            if token in request.stop_token_ids or (not request.ignore_eos and token == request.eos_token_id):
                reason, eos = "stop", token
                return True
            stop_sequence = next((seq for seq in request.stop_token_sequences
                                  if tuple(generated[-len(seq):]) == seq), ())
            if stop_sequence:
                reason = "stop"
                return True
            return len(generated) == request.max_tokens
        if session is not None:
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

            # The text block API accepts token IDs, not image embeddings.
            width = self.prefill_block_width if features is None else 1
            for start in range(0, len(ids), width):
                raise_if_generation_deadline_expired(request)
                block = ids[start:start + width]
                final = start + len(block) == len(ids)
                if len(block) == 1:
                    options = {"logits": final}
                    if features is not None and 0 <= start - image_start < len(features):
                        options["embedding"] = features[start - image_start]
                    logits = advance(block[0], **options)
                elif draft is None:
                    logits = session.prefill_block(block, logits=final)
                else:
                    captured = {}

                    def capture_block(boundary, hidden):
                        if boundary in taps:
                            captured[boundary] = hidden

                    logits = session.prefill_block(block, logits=final, capture=capture_block)
                    for i in range(len(block)):
                        raise_if_generation_deadline_expired(request)
                        draft.append({boundary: hidden[i] for boundary, hidden in captured.items()})
            state = RowSamplingState(prompt_tokens=ids, seed=row_seed_for_index(request, row), row_index=row)
            if packed:
                proposed, accepted, blocks = self._verify_packed(
                    session, draft, drafter, request, state, logits, taps, generated, emit,
                )
            for i in range(0 if packed else request.max_tokens):
                raise_if_generation_deadline_expired(request)
                sampled = select_token(logits, request, state)
                token = sampled.token_id
                if candidates:
                    if candidates.popleft() == token:
                        accepted += 1
                    else:
                        candidates.clear()
                finished = emit(sampled)
                if not finished:
                    yield token
                    raise_if_generation_deadline_expired(request)
                else:
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
            diagnostics={"speculative_mtp": {"provider": "dspark",
                                             "verification": "packed_target" if packed else "serial_target",
                                             "proposed_tokens": proposed, "accepted_tokens": accepted,
                                             **({"target_blocks": blocks} if packed else {})}},
        )
        return GenerationOutput(
            self.detokenize(generated), generated_token_ids=tuple(generated), prompt_tokens=len(ids), telemetry=telemetry,
            finish_details=FinishDetails(reason=reason, eos_token_id=eos, stop_sequence=stop_sequence,
                                         length_limit=request.max_tokens if reason == "length" else None,
                                         sampler_mode=mode, phase="decode"),
        )

    @staticmethod
    def _verify_packed(session, draft, drafter, request, state, logits, taps, generated, emit):
        proposed = accepted = blocks = 0
        depth = drafter.cfg["block_size"] - 1

        def sample(scores):
            raise_if_generation_deadline_expired(request)
            return select_token(scores, request, state)

        pending = sample(logits)
        while not emit(pending):
            anchor = pending.token_id
            remaining = request.max_tokens - len(generated)
            candidates = draft.propose(anchor, min(depth, remaining))
            proposed += len(candidates)
            # The last output token does not need to be forwarded into the cache.
            tokens = (anchor, *candidates[:remaining - 1])
            captured = {}

            def capture(boundary, hidden):
                if boundary in taps:
                    captured[boundary] = hidden

            matched = 0
            done = False
            with session.provisional_block(tokens, capture=capture) as (scores, keep):
                blocks += 1
                for i, candidate in enumerate(candidates):
                    pending = sample(scores[i])
                    if pending.token_id != candidate:
                        break
                    matched += 1
                    if emit(pending):
                        done = True
                        break
                else:
                    pending = sample(scores[matched])
                committed = min(len(tokens), 1 + matched)
                keep(committed)
            accepted += matched
            if done:
                break
            for i in range(committed):
                draft.append({boundary: hidden[i] for boundary, hidden in captured.items()})
        return proposed, accepted, blocks


register_text_generator(model=INKLING2_MINI.name, backend="cpu_reference", quant=INKLING2_MINI.default_quant,
                        factory=Inkling2ReferenceGenerator)
