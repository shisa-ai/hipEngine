"""Muse Glimmer DFlash speculative provider (block-diffusion drafting).

Capability-gated attachment: the draft artifact must declare the ``dflash``
architecture with a hidden size matching the target and readable
``target_layers``/``block_size`` metadata — no model-name, path, or hash
allowlists. The cycle is target-corrected greedy: draft a masked block, verify
it with one target forward, accept the matching prefix plus the corrected
token, and roll both KV caches back to the committed length. Output is
bit-identical to true greedy AR by construction (the exactness gate); the
drafter only changes how many tokens each target forward commits.

Bring-up scope: greedy (temperature <= 0), one request, chain proposals.
Sampled speculative acceptance is a named refusal until its rejection-sampling
gate lands.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
)
from hipengine.speculative.registry import (
    SpeculativeProviderConfig,
    SpeculativeProviderKey,
    register_speculative_provider,
)

MUSE_GLIMMER_DFLASH_EXECUTION_PATH = "muse_glimmer_dflash_block16_greedy"


@dataclass
class _CycleCounters:
    target_forwards: int = 0
    draft_forwards: int = 0
    proposed: int = 0
    accepted: int = 0
    committed: int = 0


class MuseGlimmerDFlashTextProvider:
    """DFlash owner for one Muse Glimmer target generator."""

    provider_name = "dflash"

    def __init__(self, target_generator: Any, config: SpeculativeProviderConfig) -> None:
        if str(config.provider) != self.provider_name:
            raise ValueError("Muse Glimmer DFlash provider config must select dflash")
        self.target = target_generator
        self.config = config
        self.draft_model = Path(config.draft_model).expanduser().resolve()
        self.candidate_budget = int(config.candidate_budget)
        self._drafter: Any = None
        self._dcfg: Any = None
        self._closed = False
        self.counters = _CycleCounters()
        self.last_load_seconds: float | None = None
        self.last_prefill_seconds: float | None = None
        self.last_decode_seconds: float | None = None

    # -- attachment --------------------------------------------------------
    def _ensure_drafter(self):
        if self._drafter is not None:
            return self._drafter
        from hipengine.kernels.cpu_reference.muse_glimmer_dflash import MuseGlimmerDFlashConfig
        from hipengine.loading.gguf import GGUFReader
        from hipengine.runtime.muse_glimmer_dflash import MuseGlimmerDFlashDrafter

        started = time.perf_counter()
        reader = GGUFReader(self.draft_model)
        md = reader.info.metadata
        mask_token = md.get("tokenizer.ggml.mask_token_id")
        if mask_token is None:
            raise ValueError("the DFlash draft artifact declares no mask_token_id")
        dcfg = MuseGlimmerDFlashConfig.from_gguf_info(reader.info, mask_token_id=int(mask_token))
        session = self.target._ensure_session()
        if dcfg.n_embd != session.config.n_embd:
            raise ValueError(
                f"draft hidden size {dcfg.n_embd} does not match target {session.config.n_embd}"
            )
        if dcfg.head_dim != session.config.head_dim:
            raise ValueError("draft head_dim does not match the target")
        if max(dcfg.target_layers) > session.config.n_layers:
            raise ValueError("draft target_layers exceed the target depth")
        if self.candidate_budget > dcfg.block_size - 1:
            raise ValueError(
                f"candidate budget {self.candidate_budget} exceeds the trained "
                f"block capacity {dcfg.block_size - 1}"
            )
        self._dcfg = dcfg
        self._drafter = MuseGlimmerDFlashDrafter(
            reader, dcfg, target_weights=session.loaded,
            backend=self.target.backend, max_context=session.max_context,
        )
        self.last_load_seconds = time.perf_counter() - started
        return self._drafter

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "attachment_mode": "independent",
            "supported_modes": ["verify_chain"],
            "tree_supported": False,
            "transaction_mode": "kv_truncate",
            "resource_ownership": "provider draft weights/KV; target session shared",
            "policy": "explicit_attach",
            "streaming_compatible": True,
            "candidate_budget": self.candidate_budget,
            "exactness_mode": "target_corrected_greedy",
            "execution_path": MUSE_GLIMMER_DFLASH_EXECUTION_PATH,
            "sampling": "greedy_only",
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._drafter is not None:
            self._drafter.close()
            self._drafter = None

    # -- request validation ------------------------------------------------
    def _prepare_request(self, request: GenerationRequest) -> tuple[int, ...]:
        if len(request.prompts) != 1:
            raise ValueError("Muse Glimmer DFlash serves one prompt per request")
        if request.temperature > 0.0:
            raise NotImplementedError(
                "Muse Glimmer DFlash bring-up implements greedy (temperature=0); "
                "sampled speculative acceptance is not implemented"
            )
        blockers = []
        if request.grammar is not None:
            blockers.append("grammar-constrained decoding")
        if request.logprobs or request.top_logprobs:
            blockers.append("logprobs")
        if request.stop_token_sequences:
            blockers.append("multi-token stop sequences")
        if request.forced_tokens_pending or request.post_thinking_forced_tokens_pending:
            blockers.append("forced-token queues")
        if blockers:
            raise NotImplementedError(
                "Muse Glimmer DFlash does not implement: " + ", ".join(blockers)
            )
        prompt_ids = request.prompt_token_ids(0, self.target.tokenize)
        if not prompt_ids:
            raise ValueError("the prompt produced no token ids")
        return tuple(int(t) for t in prompt_ids)

    # -- cycle -------------------------------------------------------------
    def _generate_row(
        self, request: GenerationRequest, prompt_ids: tuple[int, ...]
    ) -> GenerationOutput:
        session = self.target._ensure_session()
        drafter = self._ensure_drafter()
        dcfg = self._dcfg
        tap_layers = dcfg.target_layers
        n_draft = min(self.candidate_budget, dcfg.block_size - 1)
        stop_ids = set(request.stop_token_ids) | set(self.target.tokenizer.stop_token_ids)
        configured_eos = (
            self.target.tokenizer.eos_token_id if request.eos_token_id is None
            else int(request.eos_token_id)
        )

        started = time.perf_counter()
        session.reset()
        drafter.reset()
        logits, taps = session.prefill_with_taps(prompt_ids, tap_layers)
        self.counters.target_forwards += 1
        drafter.inject(taps, 0)
        length = len(prompt_ids)
        last_committed = prompt_ids[-1]
        prev_argmax = int(np.argmax(logits))
        # Deferred boundary token: the bonus/corrected token from the previous
        # cycle whose target KV is not yet written. It is processed as row 0 of
        # the next verify (amortized with the draft block) instead of a separate
        # decode forward, mirroring llama.cpp's deferred boundary pair.
        pending_root: int | None = None

        generated: list[int] = []
        finish_reason = "length"
        eos_token_id: int | None = None

        def _stop_check(token: int) -> bool:
            nonlocal finish_reason, eos_token_id
            if token in stop_ids or (not request.ignore_eos and configured_eos is not None and token == configured_eos):
                finish_reason = "stop"
                eos_token_id = token
                return True
            return False

        decode_started = time.perf_counter()
        self.last_prefill_seconds = decode_started - started
        while len(generated) < request.max_tokens:
            raise_if_generation_deadline_expired(request)
            root_rows = 1 if pending_root is not None else 0
            if length + root_rows + n_draft + 1 > session.max_context:
                finish_reason = "length"
                break
            # Draft only the rows we will verify (anchor + n_draft masks): the
            # draft FFN/head are full-width, so unused mask rows cost projection
            # time for no acceptance.
            block = np.asarray(
                [last_committed] + [dcfg.mask_token_id] * n_draft,
                dtype=np.int64,
            )
            draft_logits = drafter.draft_block(block)
            self.counters.draft_forwards += 1
            draft_ids = [int(np.argmax(draft_logits[i])) for i in range(1, n_draft + 1)]
            self.counters.proposed += len(draft_ids)

            # Verify the deferred root (if any) plus the draft block in one
            # forward. Row 0 of a pending-root verify writes the root's KV and
            # yields its argmax (the prediction that accepts draft_ids[0]).
            verify_ids = ([pending_root] + draft_ids) if pending_root is not None else draft_ids
            vlogits, vtaps = session.verify_with_taps(verify_ids, tap_layers)
            self.counters.target_forwards += 1
            vargmax = np.argmax(vlogits, axis=1)
            if pending_root is None:
                predictors = [prev_argmax] + [int(v) for v in vargmax[:-1]]
                draft_base = length
            else:
                predictors = [int(v) for v in vargmax[:-1]]
                draft_base = length + 1
            accepted = 0
            while accepted < len(draft_ids) and draft_ids[accepted] == predictors[accepted]:
                accepted += 1
            self.counters.accepted += accepted
            # Inject the deferred root's taps first (contiguous draft context).
            if pending_root is not None:
                drafter.inject(vtaps[0:1], length)
            if accepted < len(draft_ids):
                corrected = predictors[accepted]
                commit = draft_ids[:accepted] + [corrected]
                # The verify row at the mismatch processed the rejected draft
                # token; roll back to the accepted prefix and defer the
                # corrected token to the next verify.
                session.truncate(draft_base + accepted)
                drafter.inject(vtaps[root_rows:root_rows + accepted], draft_base)
                length = draft_base + accepted
                pending_root = corrected
                last_committed = corrected
            else:
                bonus = int(vargmax[-1])
                commit = draft_ids + [bonus]
                drafter.inject(vtaps[root_rows:root_rows + len(draft_ids)], draft_base)
                length = draft_base + len(draft_ids)
                pending_root = bonus
                last_committed = bonus

            for token in commit:
                if len(generated) >= request.max_tokens:
                    break
                generated.append(token)
                self.counters.committed += 1
                if _stop_check(token):
                    break
            if finish_reason == "stop":
                break

        text = self.target.tokenizer.decode(generated, skip_special=False)
        self.last_decode_seconds = time.perf_counter() - decode_started
        self.target.last_generation_seconds = time.perf_counter() - started
        return GenerationOutput(
            text=text,
            generated_token_ids=tuple(generated),
            prompt_tokens=len(prompt_ids),
            finish_details=FinishDetails(
                reason=finish_reason,
                eos_token_id=eos_token_id,
                length_limit=request.max_tokens if finish_reason == "length" else None,
                sampler_mode="greedy",
                phase="decode",
            ),
        )

    # -- provider protocol -------------------------------------------------
    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        prompt_ids = self._prepare_request(request)
        if request.max_tokens == 0:
            return [GenerationOutput(
                text="", generated_token_ids=(), prompt_tokens=len(prompt_ids),
                finish_details=FinishDetails(reason="length", length_limit=0, sampler_mode="greedy"),
            )]
        with self.target._lock:
            self.target._require_open()
            return [self._generate_row(request, prompt_ids)]

    def stream_detailed(self, request: GenerationRequest) -> Iterator[GenerationStreamChunk]:
        output = self.generate_detailed(request)[0]
        yield GenerationStreamChunk(
            text=output.text,
            finish_details=output.finish_details,
            generated_token_ids=output.generated_token_ids,
        )


def make_muse_glimmer_dflash_text_provider(
    *,
    target_generator: Any,
    config: SpeculativeProviderConfig,
) -> MuseGlimmerDFlashTextProvider:
    return MuseGlimmerDFlashTextProvider(target_generator, config)


register_speculative_provider(
    SpeculativeProviderKey(
        provider="dflash",
        target_model="muse_glimmer_gguf",
        backend="hip_gfx1151",
        quant="gguf_q4_k_m",
    ),
    make_muse_glimmer_dflash_text_provider,
    candidate_budget=15,
)


__all__ = [
    "MUSE_GLIMMER_DFLASH_EXECUTION_PATH",
    "MuseGlimmerDFlashTextProvider",
    "make_muse_glimmer_dflash_text_provider",
]