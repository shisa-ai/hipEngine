"""Qwen3.5/3.8 GGUF DSpine speculative provider (adjacent-injection drafting).

Capability-gated attachment: the draft artifact must declare the DSpine
``dflash_config`` bridge metadata with a hidden size matching the target — no
model-name, path, or hash allowlists. The cycle is target-corrected greedy: the
native GPU DSpine drafter proposes a block, the target verifies it with one
forward, and the accepted prefix plus the corrected token are committed, so the
output is bit-identical to true greedy AR (the lossless gate).

Bring-up scope: greedy (temperature <= 0), one request, chain proposals, a
dedicated resident session per request. Sampled acceptance and shared-session
prefix reuse are named refusals until their gates land.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

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

QWEN35_DSPINE_EXECUTION_PATH = "qwen35_gguf_dspine_block16_greedy"


class Qwen35DSpineTextProvider:
    """DSpine owner for one Qwen3.5/3.8 GGUF target generator."""

    provider_name = "dspine"

    def __init__(self, target_generator: Any, config: SpeculativeProviderConfig) -> None:
        if str(config.provider) != self.provider_name:
            raise ValueError("Qwen3.5 DSpine provider config must select dspine")
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
        from hipengine.kernels.cpu_reference.dspine import DSpineNumpyDrafter
        from hipengine.speculative.dspine_cycle import load_dspine_weights, load_target_arrays

        started = time.perf_counter()
        config, weights = load_dspine_weights(self.draft_model)
        hidden = int(config["hidden_size"])
        target_hidden = int(self.target.weight_index.metadata.get("qwen35.embedding_length", 0) or 0)
        # Capability check against the target's declared hidden size when present.
        if target_hidden and hidden != target_hidden:
            raise ValueError(f"DSpine draft hidden {hidden} does not match target {target_hidden}")
        if self.candidate_budget > int(config["block_size"]) - 1:
            raise ValueError(
                f"candidate budget {self.candidate_budget} exceeds trained block capacity "
                f"{int(config['block_size']) - 1}"
            )
        self._config = config
        self._weights = weights
        self._nd = DSpineNumpyDrafter(config, weights)
        _, self._token_embd, self._head = load_target_arrays(self.target.model_path)
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
            "execution_path": QWEN35_DSPINE_EXECUTION_PATH,
            "sampling": "greedy_only",
        }

    def close(self) -> None:
        self._closed = True
        if getattr(self, "_native", None) is not None:
            self._native.close()
            self._native = None

    def _prepare_request(self, request: GenerationRequest) -> tuple[int, ...]:
        if len(request.prompts) != 1:
            raise ValueError("Qwen3.5 DSpine serves one prompt per request")
        if request.temperature > 0.0:
            raise NotImplementedError(
                "Qwen3.5 DSpine bring-up implements greedy (temperature=0); "
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
            raise NotImplementedError("Qwen3.5 DSpine does not implement: " + ", ".join(blockers))
        prompt_ids = request.prompt_token_ids(0, self.target.tokenize)
        if not prompt_ids:
            raise ValueError("the prompt produced no token ids")
        return tuple(int(t) for t in prompt_ids)

    def _generate_row(self, request: GenerationRequest, prompt_ids: tuple[int, ...]) -> GenerationOutput:
        from hipengine.runtime.dspine import DSpineNativeDrafter
        from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
        from hipengine.speculative.dspine_cycle import run_dspine_cycle

        self._ensure_loaded()
        block_size = int(self._config["block_size"])
        max_seq = len(prompt_ids) + request.max_tokens + block_size + 2
        stop_ids = set(request.stop_token_ids) | set(getattr(self.target.tokenizer, "stop_token_ids", ()))
        eos = (self.target.tokenizer.eos_token_id if request.eos_token_id is None
               else int(request.eos_token_id))
        started = time.perf_counter()
        with Qwen35GGUFResidentSession(
            self.target.model_path, backend=self.target.backend, compiler_version=None,
            require_cached_build=False, max_sequence_length=max_seq, max_batch_size=1,
        ) as session:
            # Persist the drafter across requests so the refine head upload and
            # weight uploads amortize; recreate only if a longer context is needed.
            if self._native is None or self._native.max_context_len < max_seq:
                if self._native is not None:
                    self._native.close()
                self._native = DSpineNativeDrafter(self._config, self._weights,
                                                   runtime=session.runtime, max_context_len=max_seq)
            ds = run_dspine_cycle(
                session, self._native, self._nd, self._token_embd, self._head,
                prompt_ids=prompt_ids, max_new_tokens=request.max_tokens,
                runtime=session.runtime, stop_ids=stop_ids, eos_token_id=eos,
                ignore_eos=request.ignore_eos, use_gpu_refine=True,
                candidate_budget=self.candidate_budget)
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
                "execution_path": QWEN35_DSPINE_EXECUTION_PATH,
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


def make_qwen35_dspine_text_provider(*, target_generator: Any, config: SpeculativeProviderConfig):
    return Qwen35DSpineTextProvider(target_generator, config)


register_speculative_provider(
    SpeculativeProviderKey(
        provider="dspine",
        target_model="qwen3_5_gguf",
        backend="hip_gfx1151",
        quant="gguf_q4_k_m",
    ),
    make_qwen35_dspine_text_provider,
    # Measured prefix-verify optimum on the full mtpbench suite (gfx1151): draft
    # the full 16-row block, verify the first n drafts. n 1/2/3/4/5/6 ->
    # 6.92/8.53/10.10/10.60/10.81/10.24 tok/s; peaks at n=5 (0.95x AR). The budget
    # truncates the verify prefix, not the drafter input (worklog 20261109 prefix-
    # verify sweep; the earlier input-block sweep was invalid).
    candidate_budget=5,
)


__all__ = [
    "QWEN35_DSPINE_EXECUTION_PATH",
    "Qwen35DSpineTextProvider",
    "make_qwen35_dspine_text_provider",
]