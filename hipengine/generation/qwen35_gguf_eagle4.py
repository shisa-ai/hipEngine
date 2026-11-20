"""Qwen GGUF EAGLE-4 feature drafting with target-corrected greedy verification."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.feature_draft import FeatureDraftTextProvider
from hipengine.generation.registry import GenerationRequest
from hipengine.speculative.registry import (
    SpeculativeProviderConfig,
    SpeculativeProviderKey,
    register_speculative_provider,
)

QWEN35_EAGLE4_EXECUTION_PATH = "qwen35_gguf_eagle4_chain_greedy"


class Qwen35EAGLE4TextProvider(FeatureDraftTextProvider):
    """EAGLE-4 owner for one Qwen3.5/3.8 GGUF target generator."""

    provider_name = "eagle4"
    execution_path = QWEN35_EAGLE4_EXECUTION_PATH

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
        self._init_request_owner()

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
            "resource_ownership": "request-private draft state + target-owned pooled session",
            "policy": "explicit_attach",
            "streaming_compatible": True,
            "candidate_budget": self.candidate_budget,
            "exactness_mode": "target_corrected_greedy",
            "execution_path": QWEN35_EAGLE4_EXECUTION_PATH,
            "sampling": "greedy_only",
        }

    def _prepare_request(self, request: GenerationRequest) -> tuple[int, ...]:
        if self._closed:
            raise RuntimeError("EAGLE-4 provider is closed")
        raise_if_generation_deadline_expired(request)
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
        if blockers:
            raise NotImplementedError("Qwen3.5 EAGLE-4 does not implement: " + ", ".join(blockers))
        prompt_ids = request.prompt_token_ids(0, self.target.tokenize)
        if not prompt_ids:
            raise ValueError("the prompt produced no token ids")
        return tuple(int(t) for t in prompt_ids)

    def _workspace_tokens(self):
        return 16

    def _make_native(self, session, max_seq):
        from hipengine.runtime.eagle4 import EAGLE4NativeDrafter
        return EAGLE4NativeDrafter(
            self._config, self._weights, runtime=session.runtime,
            max_seq=max(1, min(self.candidate_budget, 12)) + 2, max_context_len=max_seq,
        )

    def _cycle_steps(self, session, native, prompt_ids, request):
        from hipengine.speculative.eagle4_cycle import iter_eagle4_cycle
        return iter_eagle4_cycle(
            session, native, self._nd, prompt_ids=prompt_ids,
            max_new_tokens=request.max_tokens, runtime=session.runtime,
            n_draft=max(1, min(self.candidate_budget, 12)), ignore_eos=True,
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
    # Keep depth 3 pending a longer, matched public-path comparison. The short
    # post-Q8 sweep does not establish a depth win: token-weighted reconstruction
    # of per-prompt medians puts n2 and n3 effectively level. See the aggregation
    # correction in benchmarks/results/2026-11-12-qwen38-specdec-reaggregation.json.
    candidate_budget=3,
)


__all__ = [
    "QWEN35_EAGLE4_EXECUTION_PATH",
    "Qwen35EAGLE4TextProvider",
    "make_qwen35_eagle4_text_provider",
]
