"""Torch-free StepFun GGUF short-context decode planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Mapping, Sequence

from hipengine.kernels.registry import KernelKey, resolve
from hipengine.loading.gguf import GGUFSplitModelInfo, scan_gguf_splits
from hipengine.loading.stepfun_gguf import StepFunGGUFModelMap, build_stepfun_gguf_tensor_map
from hipengine.tokenization import StepFunGGUFTokenizer

DEFAULT_STEPFUN_SHORT_CONTEXT = 512
DEFAULT_STEPFUN_MAX_NEW_TOKENS = 1


@dataclass(frozen=True)
class StepFunDecodePlan:
    """Validated prompt-side plan for StepFun c=1 bring-up."""

    input_ids: tuple[int, ...]
    rendered_prompt: str
    stop_token_ids: tuple[int, ...]
    max_context: int
    max_new_tokens: int
    backend: str
    quant_dispatch_keys: Mapping[str, KernelKey]

    @property
    def prompt_length(self) -> int:
        return len(self.input_ids)

    def should_stop(self, token_id: int) -> bool:
        return int(token_id) in self.stop_token_ids


@dataclass(frozen=True)
class StepFunShortContextDecodePlanner:
    """Pre-run planner for StepFun text-only c=1 decode.

    This is intentionally not the full model runner. It binds the pieces that
    must be stable before streaming decode: split GGUF metadata, tokenizer/chat
    rendering, short-context limits, multi-EOS stopping, and mixed-quant kernel
    registry keys. The full P11 runner can consume this plan once resident weight
    materialization and all layer dispatch paths are wired.
    """

    info: GGUFSplitModelInfo
    model_map: StepFunGGUFModelMap
    tokenizer: StepFunGGUFTokenizer
    backend: str = "hip_gfx1151"
    max_context: int = DEFAULT_STEPFUN_SHORT_CONTEXT
    max_new_tokens: int = DEFAULT_STEPFUN_MAX_NEW_TOKENS

    @classmethod
    def from_gguf_paths(
        cls,
        paths: Sequence[str | Path],
        *,
        backend: str = "hip_gfx1151",
        max_context: int = DEFAULT_STEPFUN_SHORT_CONTEXT,
        max_new_tokens: int = DEFAULT_STEPFUN_MAX_NEW_TOKENS,
    ) -> "StepFunShortContextDecodePlanner":
        if max_context <= 0:
            raise ValueError("max_context must be positive")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        info = scan_gguf_splits(tuple(Path(path) for path in paths))
        model_map = build_stepfun_gguf_tensor_map(info)
        tokenizer = StepFunGGUFTokenizer.from_gguf_info(info)
        return cls(
            info=info,
            model_map=model_map,
            tokenizer=tokenizer,
            backend=backend,
            max_context=int(max_context),
            max_new_tokens=int(max_new_tokens),
        )

    def plan_chat(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        reasoning_effort: str | None = "low",
        add_generation_prompt: bool = True,
    ) -> StepFunDecodePlan:
        rendered = self.tokenizer.render_chat(
            messages,
            add_generation_prompt=add_generation_prompt,
            reasoning_effort=reasoning_effort,
        )
        input_ids = tuple(self.tokenizer.encode(rendered, add_bos=False))
        self._validate_short_context(input_ids)
        return StepFunDecodePlan(
            input_ids=input_ids,
            rendered_prompt=rendered,
            stop_token_ids=self.tokenizer.eos_token_ids,
            max_context=self.max_context,
            max_new_tokens=self.max_new_tokens,
            backend=self.backend,
            quant_dispatch_keys=self.resolve_quant_dispatch_keys(),
        )

    def resolve_quant_dispatch_keys(self) -> Mapping[str, KernelKey]:
        """Return representative mixed-GGUF linear dispatch keys for this model."""

        # Import-time backend plugins populate aliases/registrations. Resolve by
        # backend module name instead of branching on a concrete backend key.
        backend_module = import_module(f"hipengine.kernels.{self.backend}")
        registrar_name = f"register_{self.backend.removeprefix('hip_')}_kernels"
        registrar = getattr(backend_module, registrar_name, None)
        if callable(registrar):
            registrar()
        required = {
            "gguf_q3_k": KernelKey(self.backend, "linear", "gguf_q3_k", "gemv_bf16_bf16_out"),
            "gguf_q5_k": KernelKey(self.backend, "linear", "gguf_q5_k", "gemv_bf16_bf16_out"),
            "gguf_q8_0": KernelKey(self.backend, "linear", "gguf_q8_0", "gemv_bf16_bf16_out"),
        }
        missing = [
            key
            for key in required.values()
            if resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
                missing="none",
            )
            is None
        ]
        if missing:
            joined = ", ".join(str(key) for key in missing)
            raise RuntimeError(f"missing StepFun mixed-quant dispatch keys: {joined}")
        return required

    def _validate_short_context(self, input_ids: tuple[int, ...]) -> None:
        if len(input_ids) + self.max_new_tokens > self.max_context:
            raise ValueError(
                "StepFun short-context bring-up exceeded max_context: "
                f"prompt={len(input_ids)} max_new_tokens={self.max_new_tokens} "
                f"max_context={self.max_context}"
            )


__all__ = [
    "DEFAULT_STEPFUN_MAX_NEW_TOKENS",
    "DEFAULT_STEPFUN_SHORT_CONTEXT",
    "StepFunDecodePlan",
    "StepFunShortContextDecodePlanner",
]
