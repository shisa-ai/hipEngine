"""Strict Qwen3.8-Flash-Next qwen4exp GGUF text-generation plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    register_text_generator,
)
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.loading.qwen4_exp_gguf import build_qwen4_exp_gguf_tensor_map
from hipengine.loading.qwen4_exp_materialize import (
    materialize_qwen4_exp_weights,
    plan_qwen4_exp_residency,
)
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
from hipengine.tokenization.gguf import Qwen4ExpGGUFTokenizer


class Qwen4ExpGGUFTextGenerator:
    """F5 strict c1/greedy text generator with serial multi-prompt ownership."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        weight_index: object,
        model_plugin: object,
        backend: str = "hip_gfx1151",
        tokenizer: Any | None = None,
        runner: Any | None = None,
        max_sequence_length: int = 2_051,
        prefill_chunk_size: int = 64,
    ) -> None:
        self.model_path = Path(model_path)
        self.weight_index = weight_index
        self.model_plugin = model_plugin
        self.backend = str(backend)
        self._resident = None
        self._speculative_provider = None
        self._closed = False
        if tokenizer is None:
            metadata_info = getattr(weight_index, "metadata", None)
            if metadata_info is None:
                metadata_info = GGUFReader(discover_gguf_files(self.model_path)[0]).info
            else:
                metadata_info = weight_index
            tokenizer = Qwen4ExpGGUFTokenizer.from_gguf_info(metadata_info)
        self.tokenizer = tokenizer
        if runner is None:
            paths = discover_gguf_files(self.model_path)
            readers = tuple(GGUFReader(path) for path in paths)
            model_map = build_qwen4_exp_gguf_tensor_map(
                tuple(reader.info for reader in readers)
            )
            plan = plan_qwen4_exp_residency(model_map)
            self._resident = materialize_qwen4_exp_weights(
                readers,
                plan=plan,
                backend=self.backend,
            )
            try:
                runner = Qwen4ExpGGUFResidentModelRunner(
                    self._resident,
                    max_sequence_length=max_sequence_length,
                    prefill_chunk_size=prefill_chunk_size,
                    backend=self.backend,
                )
            except Exception:
                self._resident.close()
                self._resident = None
                raise
        self.runner = runner

    def generate(self, request: GenerationRequest) -> list[str]:
        return [output.text for output in self.generate_detailed(request)]

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        self._require_open()
        if request.temperature != 0.0 or request.top_k not in (0, 1):
            raise ValueError("Qwen4Exp F5 supports greedy generation only")
        outputs: list[GenerationOutput] = []
        for prompt in request.prompts:
            token_ids = (
                [int(token) for token in prompt]
                if not isinstance(prompt, str)
                else [int(token) for token in self.tokenizer.encode(prompt)]
            )
            if len(token_ids) + request.max_tokens > self.runner.max_sequence_length:
                raise ValueError("Qwen4Exp request exceeds dense runner capacity")
            result = self.runner.prefill(token_ids)
            generated: list[int] = []
            reason = "length"
            for index in range(request.max_tokens):
                token = int(result.token_id)
                generated.append(token)
                if not request.ignore_eos and token == self.tokenizer.eos_token_id:
                    reason = "eos"
                    break
                if index + 1 < request.max_tokens:
                    result = self.runner.step(token)
            outputs.append(
                GenerationOutput(
                    text=self.tokenizer.decode(generated, skip_special=False),
                    generated_token_ids=tuple(generated),
                    finish_details=FinishDetails(
                        reason=reason,
                        eos_token_id=(
                            self.tokenizer.eos_token_id if reason == "eos" else None
                        ),
                        length_limit=(request.max_tokens if reason == "length" else None),
                        sampler_mode="greedy",
                    ),
                )
            )
        return outputs

    @property
    def supports_speculative(self) -> bool:
        return self._speculative_provider is not None

    @property
    def supports_speculative_mtp(self) -> bool:
        return self.supports_speculative

    def attach_speculative_provider(self, provider: Any) -> None:
        self._require_open()
        if self._speculative_provider is not None:
            raise RuntimeError("Qwen4Exp generator already has a speculative provider")
        self._speculative_provider = provider

    def generate_speculative_detailed(
        self, request: GenerationRequest
    ) -> list[GenerationOutput]:
        self._require_open()
        if self._speculative_provider is None:
            raise NotImplementedError("Qwen4Exp speculative provider is not attached")
        return list(self._speculative_provider.generate_detailed(request))

    def generate_speculative_mtp_detailed(
        self, request: GenerationRequest
    ) -> list[GenerationOutput]:
        return self.generate_speculative_detailed(request)

    def stream_speculative_detailed(self, request: GenerationRequest):
        self._require_open()
        if self._speculative_provider is None:
            raise NotImplementedError("Qwen4Exp speculative provider is not attached")
        yield from self._speculative_provider.stream_detailed(request)

    def stream_speculative_mtp_detailed(self, request: GenerationRequest):
        yield from self.stream_speculative_detailed(request)

    def speculative_capabilities(self) -> dict[str, Any]:
        if self._speculative_provider is None:
            return {}
        return dict(self._speculative_provider.capabilities())

    def close(self) -> None:
        if self._closed:
            return
        if self._speculative_provider is not None:
            self._speculative_provider.close()
            self._speculative_provider = None
        self.runner.close()
        if self._resident is not None:
            self._resident.close()
            self._resident = None
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Qwen4Exp generator is closed")


def make_qwen4_exp_gguf_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: object,
    model_plugin: object,
) -> Qwen4ExpGGUFTextGenerator:
    return Qwen4ExpGGUFTextGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1151",
    )


for _quant in ("gguf_q4_k_m", "gguf_ud_q4_k_xl"):
    register_text_generator(
        model="qwen4_exp_gguf",
        backend="hip_gfx1151",
        quant=_quant,
        factory=make_qwen4_exp_gguf_generator_gfx1151,
    )


__all__ = [
    "Qwen4ExpGGUFTextGenerator",
    "make_qwen4_exp_gguf_generator_gfx1151",
]
