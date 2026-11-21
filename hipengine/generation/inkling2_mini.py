"""Public text generation adapter for the Inkling 2 Mini CPU reference."""

from __future__ import annotations

from dataclasses import replace

from tokenizers import Tokenizer

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import FinishDetails, GenerationOutput, GenerationRequest, register_text_generator
from hipengine.generation.sampling import RowSamplingState, row_seed_for_index, select_token
from hipengine.loading.inkling2_mini import Inkling2TextWeights
from hipengine.models.inkling2_mini import INKLING2_MINI
from hipengine.runtime.inkling2_mini_reference import Inkling2TextReference


class Inkling2ReferenceGenerator:
    def __init__(self, *, model_path, weight_index, model_plugin, max_sequence_length=None):
        self.model = Inkling2TextReference(weight_index.config, Inkling2TextWeights(weight_index))
        self.tokenizer = Tokenizer.from_file(str(weight_index.model_path / "tokenizer.json"))
        self.eos_token_id = int(weight_index.config["eos_token_id"])
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
        self._validate_request(request)
        if request.eos_token_id is None:
            request = replace(request, eos_token_id=self.eos_token_id)
        rows = tuple(request.prompt_token_ids(i, self.tokenize) for i in range(len(request.prompts)))
        for ids in rows:
            if not ids:
                raise ValueError("the Inkling 2 prompt produced no token IDs")
            if len(ids) + request.max_tokens > self.max_sequence_length:
                raise ValueError(f"Inkling 2 prompt and completion exceed the {self.max_sequence_length}-token context")
        return tuple(self._generate_row(request, i, ids) for i, ids in enumerate(rows))

    @staticmethod
    def _validate_request(request):
        unsupported = (
            (request.grammar is not None, "grammar"),
            (request.tool_call_constraint is not None, "tool_call_constraint"),
            (request.json_object_close_forcing, "json_object_close_forcing"),
            (request.thinking_hard_token_cap is not None or request.thinking_close_token_ids, "thinking budget"),
            (request.forced_tokens_pending or request.post_thinking_forced_tokens_pending, "forced token queues"),
            (request.force_sequence_completion_token_sequences, "forced sequence completion"),
            (request.logprobs or request.top_logprobs, "logprobs"),
            (request.resident_session_key is not None, "resident session reuse"),
            (request.speculative_mtp_static_eligibility is not None or request.speculative_mtp_automatic, "speculative decoding"),
            (request.kv_storage != "auto", f"kv_storage={request.kv_storage!r} (CPU reference stores FP32)"),
        )
        blockers = [name for active, name in unsupported if active]
        if blockers:
            raise NotImplementedError("Inkling 2 CPU reference does not implement: " + ", ".join(blockers))

    def _generate_row(self, request, row, ids):
        raise_if_generation_deadline_expired(request)
        generated = []
        reason, eos, stop_sequence, mode = "length", None, (), None
        if request.max_tokens:
            session = self.model.session(len(ids) + request.max_tokens - 1)
            for i, token in enumerate(ids):
                raise_if_generation_deadline_expired(request)
                logits = session.step(token, logits=i == len(ids) - 1)
            state = RowSamplingState(prompt_tokens=ids, seed=row_seed_for_index(request, row), row_index=row)
            for i in range(request.max_tokens):
                raise_if_generation_deadline_expired(request)
                sampled = select_token(logits, request, state)
                token = sampled.token_id
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
                if i + 1 < request.max_tokens:
                    logits = session.step(token)
        return GenerationOutput(
            self.detokenize(generated), generated_token_ids=tuple(generated), prompt_tokens=len(ids),
            finish_details=FinishDetails(reason=reason, eos_token_id=eos, stop_sequence=stop_sequence,
                                         length_limit=request.max_tokens if reason == "length" else None,
                                         sampler_mode=mode, phase="decode"),
        )


register_text_generator(model=INKLING2_MINI.name, backend="cpu_reference", quant=INKLING2_MINI.default_quant,
                        factory=Inkling2ReferenceGenerator)
