"""Public correctness-first Gemma 4 MTP provider.

The cycle is the batched-verify one measured in
``scripts/gemma4_mtp_batched_verify.py``: the whole draft goes through the
backbone in **one** call with ``logits_rows=len(drafts) + 1``, so one backbone
forward commits up to ``len(drafts) + 1`` tokens.

What makes that legal is a property ``Gemma4Runner.forward`` documents and this
provider depends on: *"Each returned row is the same computation a single-token
forward would produce for that position: the mask is built from absolute
positions, so a row never depends on how many rows accompany it."* The batched
verify is therefore **bit-identical** to verifying one token at a time, and the
measured cycle reproduces plain greedy decoding exactly. That is why this
provider is arithmetic-preserving rather than a reassociating candidate.

Two details that are easy to get wrong and are load-bearing:

* ``drafts[0]`` predicts the token *after* ``token``, not ``token`` itself. The
  drafter is fed the sampled token together with the hidden row that produced
  it, so its first output is a guess for the following position.
* Accepting every draft still needs one more row, because row ``i`` is the
  logits after the forward has processed row ``i``. The verify therefore
  forwards ``[token, *drafts]`` and reads ``len(drafts) + 1`` rows.

The seed row comes from the verify pass rather than from a later forward.
``Gemma4Runner.rewind`` documents that contract from the other side: *"the hidden
rows of the forward that was just rewound are still in the scratch buffer...
That is what lets a caller draft from the last accepted position after
rewinding: the row that produced an accepted token is indexed from the verify
pass, not from a forward that has not happened yet."*
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import (
    DecodeState,
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    GenerationTelemetry,
)
from hipengine.loading.gguf import scan_gguf
from hipengine.loading.gemma4_assistant_device import load_gemma4_assistant_device_weights
from hipengine.loading.gemma4_gguf import gemma4_gguf_config_from_metadata
from hipengine.runtime.gemma4_assistant import Gemma4AssistantHead, Gemma4MtpDrafter
from hipengine.speculative.registry import (
    QUANT_AGNOSTIC,
    SpeculativeProviderCapabilities,
    SpeculativeProviderConfig,
    SpeculativeProviderKey,
    register_speculative_provider,
)

_PROVIDER = "gemma4_mtp"
_TARGET_MODEL = "gemma4_gguf"
_EPS = 1e-6

# The assistant head is a small sidecar, so its scratch is sized by the same
# context the backbone runs at rather than by a bring-up ceiling.
_MAX_CANDIDATES = 8


@dataclass(frozen=True)
class Gemma4MTPCycle:
    """One draft/verify round, recorded for the acceptance telemetry."""

    start_position: int
    candidates: tuple[int, ...]
    accepted: int
    committed: int


class Gemma4MTPTextProvider:
    """Greedy MTP with a batched exact verify over the backbone's own logits."""

    provider_name = _PROVIDER

    def __init__(
        self,
        *,
        target_generator: Any,
        config: SpeculativeProviderConfig,
        drafter: Gemma4MtpDrafter | None = None,
    ) -> None:
        self.target_generator = target_generator
        self.config = config
        self.candidate_budget = int(config.candidate_budget)
        if not 1 <= self.candidate_budget <= _MAX_CANDIDATES:
            raise ValueError(
                f"Gemma 4 MTP candidate budget must be in 1..{_MAX_CANDIDATES}"
            )
        # The head is built lazily: the provider attaches before the target is
        # materialized, and the head needs the backbone's embedding and output
        # norm, which do not exist until it is.
        self._drafter = drafter
        self._head: Gemma4AssistantHead | None = None
        self._head_weights: Any | None = None
        self.closed = False
        self.last_cycles: tuple[Gemma4MTPCycle, ...] = ()

    # --- construction ---------------------------------------------------

    def _ensure_drafter(self) -> Gemma4MtpDrafter:
        if self._drafter is not None:
            return self._drafter
        generator = self.target_generator
        runner = generator._ensure_runner()
        backbone_weights = generator._weights
        if backbone_weights is None:
            raise RuntimeError("Gemma 4 MTP requires a materialized target")
        weights = load_gemma4_assistant_device_weights(str(self.config.draft_model))
        try:
            head = Gemma4AssistantHead(
                weights=weights,
                backbone=gemma4_gguf_config_from_metadata(
                    scan_gguf(generator.model_path)
                ),
                backbone_embedding=backbone_weights.embed_tokens,
                backbone_output_norm=backbone_weights.final_norm.buffer,
                capacity=int(runner.capacity),
                eps=_EPS,
            )
        except BaseException:
            weights.free()
            raise
        self._head = head
        self._head_weights = weights
        self._drafter = Gemma4MtpDrafter(
            head=head,
            runner=runner,
            max_drafts=self.candidate_budget,
        )
        return self._drafter

    def capabilities(self) -> dict[str, Any]:
        row = SpeculativeProviderCapabilities(
            provider_name=self.provider_name,
            artifact_fingerprint="model_attached_sidecar",
            attachment_mode="model_attached",
            supported_modes=("verify_chain",),
            # The verify forwards the sampled token plus every draft.
            max_verifier_rows=self.candidate_budget + 1,
            transaction_mode="batched_exact_target_verify",
            provider_state_key="gemma4_mtp_assistant_state",
            provider_kv_key="gemma4_mtp_assistant_kv",
            fixed_transaction_units=(("gemma4_mtp.request", 1),),
            per_candidate_units=(("gemma4_mtp.candidate", 1),),
            strict_fallback="target_ar",
        )
        return {
            "provider_name": row.provider_name,
            "artifact_fingerprint": row.artifact_fingerprint,
            "attachment_mode": row.attachment_mode,
            "supported_modes": list(row.supported_modes),
            "max_verifier_rows": row.max_verifier_rows,
            "transaction_mode": row.transaction_mode,
            "strict_fallback": row.strict_fallback,
            "candidate_budget": self.candidate_budget,
            "streaming_mode": "buffered_public",
            "verify_equivalence": "bit_identical_to_target_ar_greedy",
        }

    # --- generation -----------------------------------------------------

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        self._require_open()
        if request.temperature != 0.0 or request.top_k not in (0, 1):
            raise NotImplementedError(
                "Gemma 4 MTP supports greedy generation only"
            )
        return [self._generate_one(prompt, request) for prompt in request.prompts]

    def _generate_one(self, prompt: Any, request: GenerationRequest) -> GenerationOutput:
        raise_if_generation_deadline_expired(request)
        generator = self.target_generator
        tokenizer = generator.tokenizer
        token_ids = (
            [int(token) for token in prompt]
            if not isinstance(prompt, str)
            else [int(token) for token in tokenizer.encode(prompt)]
        )
        if not token_ids:
            raise ValueError("Gemma 4 MTP prompt produced no token IDs")

        if request.max_tokens == 0:
            return GenerationOutput(
                text="",
                generated_token_ids=(),
                finish_details=FinishDetails(
                    reason="length",
                    length_limit=0,
                    sampler_mode="greedy_speculative_mtp",
                ),
            )

        drafter = self._ensure_drafter()
        runner = generator._ensure_runner()
        eos = tokenizer.eos_token_id

        phase_ms: dict[str, float] = {}
        phase_calls: dict[str, int] = {}

        def record(name: str, started: float) -> None:
            phase_ms[name] = phase_ms.get(name, 0.0) + (
                time.perf_counter() - started
            ) * 1_000.0
            phase_calls[name] = phase_calls.get(name, 0) + 1

        prefill_started = time.perf_counter()
        # A runner carries the previous request's position and KV, and this route
        # has to give that back before its own prefill, the way the AR route does
        # per prompt (``gemma4_gguf.py`` ``generate_detailed`` calls
        # ``runner.reset()``). Without it the prefill appends this prompt after
        # the last request's tokens and attends over them: the output becomes a
        # function of what ran before it rather than of the prompt, which is both
        # wrong for a server and enough to break the comparison this route's
        # whole design rests on -- the batched verify reproduces a single-token
        # forward *for this request's own context*.
        runner.reset()
        logits = runner.forward(token_ids, apply_softcap=False)
        record("target_prefill", prefill_started)

        token = int(np.argmax(logits))
        generated: list[int] = [token]
        # The prefill's own last row produced `token`.
        seed_row = -1
        cycles: list[Gemma4MTPCycle] = []
        reason = "length"
        if not request.ignore_eos and token == eos:
            reason = "eos"

        while len(generated) < request.max_tokens and reason != "eos":
            raise_if_generation_deadline_expired(request)
            remaining = request.max_tokens - len(generated)
            # A cycle commits `accepted + 1` tokens: the accepted drafts plus the
            # row at `accepted`, which is always kept. Capping only the drafts at
            # `remaining` therefore lets the final cycle commit one token past
            # `max_tokens`. Leaving a row of headroom bounds the commit by
            # construction, and `remaining == 1` falls through to a zero-draft
            # cycle, which is a plain single-token verify.
            budget = min(self.candidate_budget, remaining - 1)
            drafter.max_drafts = budget if budget > 0 else 1
            start_position = int(runner.position)

            proposal_started = time.perf_counter()
            drafts = drafter.draft(token, hidden_row=seed_row) if budget > 0 else []
            record("proposal", proposal_started)
            if not drafts and budget > 0:
                break

            # One forward for the whole draft. `drafts[i]` is tested against
            # `rows[i]`, and the token after `accepted` drafts comes from
            # `rows[accepted]`, so the row count is len(drafts) + 1.
            verify_started = time.perf_counter()
            rows = runner.forward(
                [token, *drafts],
                apply_softcap=False,
                logits_rows=len(drafts) + 1,
            )
            if rows.ndim == 1:
                # ``Gemma4Runner.forward`` returns a flat ``(vocab,)`` array for
                # the one-row count and a ``(rows, vocab)`` array above it, and a
                # zero-draft cycle is exactly that one-row case. Row 0 is read
                # like any other row here, so give it the shape every other row
                # count produces: ``rows[0]`` on a flat array is the scalar logit
                # of vocabulary entry 0, whose argmax is 0, and a cycle that
                # commits it emits ``<pad>`` instead of the model's next token.
                rows = rows.reshape(1, -1)
            record("target_verify", verify_started)

            accepted = 0
            for index, draft in enumerate(drafts):
                if int(np.argmax(rows[index])) != draft:
                    break
                accepted += 1

            token = int(np.argmax(rows[accepted]))
            generated.extend(drafts[:accepted])
            generated.append(token)

            # The verify consumed `token` plus every draft; the target keeps
            # `token` and the accepted prefix, so the rejected tail goes back.
            runner.rewind(start_position + 1 + accepted)
            seed_row = accepted

            cycles.append(
                Gemma4MTPCycle(
                    start_position=start_position,
                    candidates=tuple(int(draft) for draft in drafts),
                    accepted=accepted,
                    committed=accepted + 1,
                )
            )
            if not request.ignore_eos and token == eos:
                reason = "eos"

        self.last_cycles = tuple(cycles)
        proposed = sum(len(cycle.candidates) for cycle in cycles)
        accepted_total = sum(cycle.accepted for cycle in cycles)
        telemetry = GenerationTelemetry(
            decode_state=DecodeState(
                prompt_tokens=len(token_ids),
                generated_tokens=len(generated),
                step_index=len(cycles),
                sampler_mode="greedy_speculative_mtp",
                execution_path="gemma4_mtp_batched_exact_verify",
            ),
            event="generation_complete",
            diagnostics={
                "speculative_provider": self.provider_name,
                "candidate_budget": self.candidate_budget,
                "proposed_draft_tokens": proposed,
                "accepted_draft_tokens": accepted_total,
                "draft_acceptance": (
                    float(accepted_total / proposed) if proposed else 0.0
                ),
                "tokens_per_cycle": (
                    float(len(generated) / len(cycles)) if cycles else 1.0
                ),
                "cycles": [
                    {
                        "start_position": cycle.start_position,
                        "candidates": list(cycle.candidates),
                        "accepted": cycle.accepted,
                        "committed": cycle.committed,
                    }
                    for cycle in cycles
                ],
                "target_verify": "batched_exact",
                "draft_rollback": "cursor_rewind",
                "verify_equivalence": "bit_identical_to_target_ar_greedy",
                "phase_census": {
                    "cycles": len(cycles),
                    "target_prefill": {
                        "calls": phase_calls.get("target_prefill", 0),
                        "ms": phase_ms.get("target_prefill", 0.0),
                    },
                    "proposal": {
                        "calls": phase_calls.get("proposal", 0),
                        "ms": phase_ms.get("proposal", 0.0),
                    },
                    "target_verify": {
                        "calls": phase_calls.get("target_verify", 0),
                        "rows": phase_calls.get("target_verify", 0)
                        * (self.candidate_budget + 1),
                        "ms": phase_ms.get("target_verify", 0.0),
                    },
                },
            },
        )
        return GenerationOutput(
            text=tokenizer.decode(generated, skip_special=False),
            generated_token_ids=tuple(generated),
            telemetry=telemetry,
            finish_details=FinishDetails(
                reason=reason,
                eos_token_id=eos if reason == "eos" else None,
                length_limit=request.max_tokens if reason == "length" else None,
                sampler_mode="greedy_speculative_mtp",
            ),
        )

    def stream_detailed(self, request: GenerationRequest):
        self._require_open()
        if len(request.prompts) != 1:
            raise ValueError("Gemma 4 MTP streaming requires exactly one prompt")
        output = self.generate_detailed(request)[0]
        yield GenerationStreamChunk(
            text=output.text,
            finish_details=output.finish_details,
            telemetry=output.telemetry,
            generated_token_ids=output.generated_token_ids,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        head, self._head = self._head, None
        weights, self._head_weights = self._head_weights, None
        if head is not None:
            head.close()
        if weights is not None:
            weights.free()
        self._drafter = None

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Gemma 4 MTP provider is closed")


def make_gemma4_mtp_provider(**kwargs: Any) -> Gemma4MTPTextProvider:
    return Gemma4MTPTextProvider(**kwargs)


def register_gemma4_mtp_providers(*, replace: bool = True) -> None:
    """Register the Gemma 4 MTP provider as quantization-independent.

    The provider has no quantization dependence to declare: the assistant head is
    a separate sidecar and the verify pass reads whatever logits the backbone
    produces, so the target's quantization is not an input to it. Registering it
    under :data:`QUANT_AGNOSTIC` says that, instead of writing out a list of
    known-good quantizations -- which would refuse a newly resolvable
    quantization for a reason that is not a capability miss.

    What the provider does require is a Gemma 4 backbone, which the
    ``target_model`` key names, and an assistant sidecar, which it refuses by
    name if the file is missing.
    """

    register_speculative_provider(
        SpeculativeProviderKey(
            provider=_PROVIDER,
            target_model=_TARGET_MODEL,
            backend="hip_gfx1151",
            quant=QUANT_AGNOSTIC,
        ),
        make_gemma4_mtp_provider,
        replace=replace,
    )


register_gemma4_mtp_providers()


__all__ = [
    "Gemma4MTPCycle",
    "Gemma4MTPTextProvider",
    "make_gemma4_mtp_provider",
    "register_gemma4_mtp_providers",
]
