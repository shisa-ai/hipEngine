"""Qwen3.5/3.8 GGUF EAGLE-4 speculative provider (feature-autoregression drafting).

Capability-gated attachment on the draft ``eagle_config``/``efficient_lm_head_config``
metadata + hidden-size match (no identity allowlists). The cycle is target-corrected
greedy: the native EAGLE-4 drafter autoregressively proposes a token chain from the
target's aux hidden states, the target verifies it, and the accepted prefix plus the
corrected token are committed — bit-identical to true greedy AR (the lossless gate).

Bring-up scope: greedy (temperature <= 0), one request, chain proposals (a dynamic
tree is a speed follow-up), a dedicated resident session per request with a
persistent drafter. Sampled acceptance is a named refusal until its gate lands.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

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

QWEN35_EAGLE4_EXECUTION_PATH = "qwen35_gguf_eagle4_chain_greedy"


class Qwen35EAGLE4TextProvider:
    """EAGLE-4 owner for one Qwen3.5/3.8 GGUF target generator."""

    provider_name = "eagle4"

    def __init__(self, target_generator: Any, config: SpeculativeProviderConfig) -> None:
        if str(config.provider) != self.provider_name:
            raise ValueError("Qwen3.5 EAGLE-4 provider config must select eagle4")
        self.target = target_generator
        self.config = config
        self.draft_model = Path(config.draft_model).expanduser().resolve()
        self.candidate_budget = int(config.candidate_budget)
        self._loaded = False
        self._closed = False
        self._native = None
        self.last_load_seconds: float | None = None
        self.last_decode_seconds: float | None = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from hipengine.kernels.cpu_reference.eagle4 import EAGLE4NumpyDrafter
        from hipengine.speculative.eagle4_cycle import load_eagle4_weights

        started = time.perf_counter()
        config, weights = load_eagle4_weights(self.draft_model)
        hidden = int(config["hidden_size"])
        target_hidden = int(self.target.weight_index.metadata.get("qwen35.embedding_length", 0) or 0)
        if target_hidden and hidden != target_hidden:
            raise ValueError(f"EAGLE-4 draft hidden {hidden} does not match target {target_hidden}")
        if "eagle_config" not in config or "efficient_lm_head_config" not in config:
            raise ValueError("the draft artifact declares no EAGLE-4 eagle_config/efficient_lm_head_config")
        self._config = config
        self._weights = weights
        self._nd = EAGLE4NumpyDrafter(config, weights)
        self._loaded = True
        self.last_load_seconds = time.perf_counter() - started

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "attachment_mode": "independent",
            "supported_modes": ["verify_chain"],
            "tree_supported": False,
            "transaction_mode": "kv_truncate",
            "resource_ownership": "provider draft weights + dedicated target session",
            "policy": "explicit_attach",
            "streaming_compatible": True,
            "candidate_budget": self.candidate_budget,
            "exactness_mode": "target_corrected_greedy",
            "execution_path": QWEN35_EAGLE4_EXECUTION_PATH,
            "sampling": "greedy_only",
        }

    def close(self) -> None:
        self._closed = True
        if getattr(self, "_native", None) is not None:
            self._native.close()
            self._native = None

    def _prepare_request(self, request: GenerationRequest) -> tuple[int, ...]:
        if len(request.prompts) != 1:
            raise ValueError("Qwen3.5 EAGLE-4 serves one prompt per request")
        if request.temperature > 0.0:
            raise NotImplementedError(
                "Qwen3.5 EAGLE-4 bring-up implements greedy (temperature=0); "
                "sampled speculative acceptance is not implemented"
            )
        blockers = []
        if request.grammar is not None:
            blockers.append("grammar-constrained decoding")
        if request.logprobs or request.top_logprobs:
            blockers.append("logprobs")
        if request.stop_token_sequences:
            blockers.append("multi-token stop sequences")
        if blockers:
            raise NotImplementedError("Qwen3.5 EAGLE-4 does not implement: " + ", ".join(blockers))
        prompt_ids = request.prompt_token_ids(0, self.target.tokenize)
        if not prompt_ids:
            raise ValueError("the prompt produced no token ids")
        return tuple(int(t) for t in prompt_ids)

    def _generate_row(self, request: GenerationRequest, prompt_ids: tuple[int, ...]) -> GenerationOutput:
        from hipengine.runtime.eagle4 import EAGLE4NativeDrafter
        from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
        from hipengine.speculative.eagle4_cycle import run_eagle4_cycle

        self._ensure_loaded()
        max_seq = len(prompt_ids) + request.max_tokens + 16
        n_draft = max(1, min(int(self.candidate_budget), 12))
        stop_ids = set(request.stop_token_ids) | set(getattr(self.target.tokenizer, "stop_token_ids", ()))
        eos = (self.target.tokenizer.eos_token_id if request.eos_token_id is None
               else int(request.eos_token_id))
        started = time.perf_counter()
        with Qwen35GGUFResidentSession(
            self.target.model_path, backend=self.target.backend, compiler_version=None,
            require_cached_build=False, max_sequence_length=max_seq, max_batch_size=1,
        ) as session:
            if self._native is None:
                self._native = EAGLE4NativeDrafter(self._config, self._weights,
                                                   runtime=session.runtime, max_seq=n_draft + 2)
            ds = run_eagle4_cycle(
                session, self._native, self._nd, prompt_ids=prompt_ids,
                max_new_tokens=request.max_tokens, runtime=session.runtime,
                n_draft=n_draft, stop_ids=stop_ids, eos_token_id=eos,
                ignore_eos=request.ignore_eos)
        self.last_decode_seconds = ds["elapsed_s"]
        ids = ds["output_ids"]
        text = self.target.tokenizer.decode(ids, skip_special=False)
        return GenerationOutput(
            text=text,
            generated_token_ids=tuple(ids),
            prompt_tokens=len(prompt_ids),
            finish_details=FinishDetails(
                reason=ds["finish_reason"],
                eos_token_id=ds["eos_token_id"],
                length_limit=request.max_tokens if ds["finish_reason"] == "length" else None,
                sampler_mode="greedy",
                phase="decode",
            ),
            telemetry={
                "execution_path": QWEN35_EAGLE4_EXECUTION_PATH,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(ids),
                "draft_forwards": ds["draft_forwards"],
                "mean_acceptance": ds["mean_acceptance"],
                "decode_seconds": ds["elapsed_s"],
                "load_seconds": self.last_load_seconds,
                "total_seconds": time.perf_counter() - started,
            },
        )

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        prompt_ids = self._prepare_request(request)
        if request.max_tokens == 0:
            return [GenerationOutput(
                text="", generated_token_ids=(), prompt_tokens=len(prompt_ids),
                finish_details=FinishDetails(reason="length", length_limit=0, sampler_mode="greedy"),
            )]
        return [self._generate_row(request, prompt_ids)]

    def stream_detailed(self, request: GenerationRequest) -> Iterator[GenerationStreamChunk]:
        output = self.generate_detailed(request)[0]
        yield GenerationStreamChunk(
            text=output.text,
            finish_details=output.finish_details,
            generated_token_ids=output.generated_token_ids,
        )


def make_qwen35_eagle4_text_provider(*, target_generator: Any, config: SpeculativeProviderConfig):
    return Qwen35EAGLE4TextProvider(target_generator, config)


register_speculative_provider(
    SpeculativeProviderKey(
        provider="eagle4",
        target_model="qwen3_5_gguf",
        backend="hip_gfx1151",
        quant="gguf_q4_k_m",
    ),
    make_qwen35_eagle4_text_provider,
    # Measured chain optimum on the full mtpbench suite (gfx1151): n_draft 1/2/3/4/6/8
    # -> 8.92/7.53/6.66/5.61/4.34/2.31 tok/s. Acceptance falls with depth (the first
    # draft is accepted ~0.445, marginal deeper drafts not enough to amortize their
    # per-step cost), so the chain peaks at n_draft=1 (worklog 20261109T012249 sweep).
    candidate_budget=1,
)


__all__ = [
    "QWEN35_EAGLE4_EXECUTION_PATH",
    "Qwen35EAGLE4TextProvider",
    "make_qwen35_eagle4_text_provider",
]