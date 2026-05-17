"""Qwen3.5 GGUF generation path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hipengine.generation.registry import GenerationRequest, register_text_generator
from hipengine.loading.gguf import GGUFModelInfo
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer


@dataclass
class Qwen35GGUFBringupGenerator:
    """Public API GGUF greedy generator over a persistent resident session."""

    model_path: str | Path
    weight_index: GGUFModelInfo
    model_plugin: Any
    tokenizer: Qwen35GGUFTokenizer = field(init=False)

    def __post_init__(self) -> None:
        self.tokenizer = Qwen35GGUFTokenizer.from_gguf_info(self.weight_index)

    def generate(self, request: GenerationRequest) -> list[str]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if request.temperature != 0.0 or request.top_p != 1.0:
            raise NotImplementedError("Qwen3.5 GGUF generator currently supports greedy sampling only")
        if request.max_tokens == 0:
            return ["" for _ in request.prompts]
        outputs: list[str] = []
        with Qwen35GGUFResidentSession(self.model_path) as session:
            for prompt in request.prompts:
                prompt_ids = self.tokenizer.encode(prompt)
                if not prompt_ids:
                    raise ValueError("GGUF prompt tokenization produced no token IDs")
                generated_ids: list[int] = []
                result = session.prefill(prompt_ids, return_logits=False)
                generated_ids.append(result.token_id)
                if request.ignore_eos or result.token_id != self.tokenizer.eos_token_id:
                    remaining = request.max_tokens - 1
                    if remaining > 0:
                        if _session_uses_host_routed_decode(session):
                            for _ in range(remaining):
                                step = session.step(generated_ids[-1], return_logits=False)
                                generated_ids.append(step.token_id)
                                if not request.ignore_eos and step.token_id == self.tokenizer.eos_token_id:
                                    break
                        else:
                            with session.capture_decode_graph(
                                position=len(prompt_ids),
                                steps_per_replay=1,
                                max_replay_steps=remaining,
                                record_steps=remaining,
                            ) as graph:
                                graph.replay(remaining)
                                for token_id in graph.read_generated_token_ids(remaining):
                                    generated_ids.append(token_id)
                                    if not request.ignore_eos and token_id == self.tokenizer.eos_token_id:
                                        break
                outputs.append(self.tokenizer.decode(generated_ids))
        return outputs


def _session_uses_host_routed_decode(session: Qwen35GGUFResidentSession) -> bool:
    """Return True for GGUF paths whose decode step cannot be graph-captured yet."""

    runner = getattr(session, "runner", None)
    weights = getattr(runner, "weights", None)
    config = getattr(weights, "config", None)
    return bool(getattr(config, "is_moe", False))


def make_qwen35_gguf_bringup_generator(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
) -> Qwen35GGUFBringupGenerator:
    return Qwen35GGUFBringupGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
    )


for _model in ("qwen3_5_gguf", "qwen3_5_moe_gguf"):
    for _quant in ("gguf_q4_k_m", "gguf_q8_0", "gguf_q4_1", "gguf_ud_q4_k_xl"):
        register_text_generator(
            model=_model,
            backend="hip_gfx1100",
            quant=_quant,
            factory=make_qwen35_gguf_bringup_generator,
        )


__all__ = [
    "Qwen35GGUFBringupGenerator",
    "make_qwen35_gguf_bringup_generator",
]
