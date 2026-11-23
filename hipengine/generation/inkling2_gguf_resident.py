"""Run independent Inkling GGUF request sessions on the resident driver."""

from dataclasses import dataclass, field

from tokenizers.decoders import DecodeStream

from hipengine.generation.batch_scheduler import GeneratedToken
from hipengine.generation.deadline import generation_deadline_expired
from hipengine.generation.registry import FinishDetails, GenerationOutput, GenerationStreamChunk
from hipengine.generation.sampling import RowSamplingState, row_seed_for_index, select_token


@dataclass(slots=True)
class _Row:
    request: object
    prompt: tuple[int, ...]
    sampling: RowSamplingState
    decoder: DecodeStream = field(default_factory=lambda: DecodeStream(skip_special_tokens=True))
    generated: list[int] = field(default_factory=list)
    published: str = ""
    submitted: int = 0
    computed: int = 0
    lifetime: object = None
    session: object = None
    logits: object = None
    output: GenerationOutput | None = None

    def release(self):
        session, self.session = self.session, None
        if session is not None:
            session.close()


class Inkling2ResidentRunner:
    """Own one packed HIP session per admitted row; a decode tick is still serial."""

    def __init__(self, generator, *, capacity):
        self.generator = generator
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("Inkling 2 resident capacity must be positive")
        self.rows = {}
        self.outputs = {}

    @property
    def active_request_ids(self):
        return tuple(self.rows)

    def prompt_tokens(self, prompt):
        return self.generator.tokenize(prompt) if isinstance(prompt, str) else tuple(map(int, prompt))

    def scheduler_max_new_tokens(self, request):
        return int(request.max_tokens)

    def register_batch(self, request_ids, request, *, prompt_rows):
        request, prompts = self.generator._prepare_text_rows(request)
        if tuple(prompt_rows) != prompts:
            raise ValueError("Inkling 2 scheduler and generator token rows disagree")
        for i, (rid, prompt) in enumerate(zip(request_ids, prompts, strict=True)):
            if rid in self.rows or rid in self.outputs:
                raise ValueError(f"Inkling 2 request {rid} is already registered")
            self.rows[rid] = _Row(request, prompt, RowSamplingState(
                prompt_tokens=prompt, seed=row_seed_for_index(request, i), row_index=i))

    def bind_request_lifetime(self, request_ids, request):
        for rid in request_ids:
            self.rows[rid].lifetime = request

    def request_cancel_reason(self, request_id):
        row = self.rows.get(request_id)
        if row is None or row.lifetime is None:
            return None
        token = row.lifetime.cancellation_token
        if token is not None and (token.cancel_requested or token.cancelled):
            details = token.finish_details
            return "timeout" if details.deadline_exceeded else "disconnect" if details.reason == "disconnect" else "cancel"
        if generation_deadline_expired(row.lifetime.deadline_at):
            return "timeout"
        return None

    def reserve_admission(self, request):
        row = self.rows[request.request_id]
        if row.request.max_tokens and row.session is None:
            row.session = self.generator.model.session(len(row.prompt) + row.request.max_tokens - 1)

    def rollback_admission(self, request):
        self.rows[request.request_id].release()

    def prefill_batch(self, work, *, commit):
        if not commit:
            raise ValueError("Inkling 2 resident prefill requires commit=True")
        width = self.generator.prefill_block_width
        for rid, chunk in zip(work.request_ids, work.token_rows, strict=True):
            row = self.rows[rid]
            limit = row.submitted + len(chunk)
            if tuple(chunk) != row.prompt[row.submitted:limit]:
                raise RuntimeError(f"Inkling 2 scheduler prompt drift for request {rid}")
            row.submitted = limit
            if not row.request.max_tokens:
                continue
            if row.session is None:
                raise RuntimeError(f"Inkling 2 request {rid} does not own a session")
            # Scheduler budgets may split a model block. Compute only after
            # collecting its canonical width so batch composition cannot change math.
            while row.computed < limit:
                end = min(len(row.prompt), row.computed + width)
                if end > limit:
                    break
                tokens = row.prompt[row.computed:end]
                last = end == len(row.prompt)
                row.logits = (row.session.step(tokens[0], logits=last) if len(tokens) == 1
                              else row.session.prefill_block(tokens, logits=last))
                row.computed = end

    def decode_batch(self, work, *, commit):
        if not commit:
            raise ValueError("Inkling 2 resident decode requires commit=True")
        return tuple(self._decode(rid, self.rows[rid]) for rid in work.request_ids)

    def _decode(self, rid, row):
        if row.computed != len(row.prompt) or row.session is None:
            raise RuntimeError(f"Inkling 2 request {rid} is not prefilled")
        if row.generated:
            row.logits = row.session.step(row.generated[-1])
        sampled = select_token(row.logits, row.request, row.sampling)
        token = sampled.token_id
        row.generated.append(token)
        stop = token in row.request.stop_token_ids or (not row.request.ignore_eos and token == row.request.eos_token_id)
        sequence = next((seq for seq in row.request.stop_token_sequences
                         if tuple(row.generated[-len(seq):]) == seq), ()) if not stop else ()
        at_limit = len(row.generated) == row.request.max_tokens
        reason = "stop" if stop or sequence else "length"
        piece = row.decoder.step(self.generator.tokenizer, token) or ""
        row.published += piece
        finish = None
        if stop or sequence or at_limit:
            finish = FinishDetails(reason=reason, eos_token_id=token if stop else None, stop_sequence=sequence,
                                   length_limit=row.request.max_tokens if reason == "length" else None,
                                   sampler_mode=str(sampled.mode.value), phase="decode")
            row.output = self._output(row, finish)
            if not row.output.text.startswith(row.published):
                raise RuntimeError("Inkling 2 resident incremental decode disagrees with full decoding")
            piece += row.output.text[len(row.published):]
        return GeneratedToken(rid, token, finished=bool(stop or sequence), stream_chunk=GenerationStreamChunk(
            text=piece, finish_details=finish, telemetry=None if row.output is None else row.output.telemetry,
            generated_token_ids=None if finish is None else tuple(row.generated)))

    def _output(self, row, finish):
        return self.generator._decorate_output(GenerationOutput(
            self.generator.detokenize(row.generated), generated_token_ids=tuple(row.generated),
            prompt_tokens=len(row.prompt), finish_details=finish))

    def compact_batch(self, moves):
        # Session ownership uses request IDs, never scheduler slot indices.
        del moves

    def reclaim(self, completed):
        rid = completed.request_id
        row = self.rows.pop(rid, None)
        if row is not None:
            try:
                finish = completed.finish_details
                if row.request.max_tokens == 0 and finish.reason == "length":
                    finish = FinishDetails(reason="length", length_limit=0, phase="decode")
                self.outputs[rid] = row.output if row.output is not None else self._output(row, finish)
            finally:
                row.release()

    def has_outputs(self, request_ids):
        return not self.missing_outputs(request_ids)

    def missing_outputs(self, request_ids):
        return [rid for rid in request_ids if rid not in self.outputs]

    def take_outputs(self, request_ids):
        return [self.outputs.pop(rid) for rid in request_ids]

    def discard(self, request_ids):
        for rid in request_ids:
            row = self.rows.pop(rid, None)
            if row is not None:
                row.release()
            self.outputs.pop(rid, None)

    def close(self):
        self.discard(tuple(self.rows))
        self.outputs.clear()
