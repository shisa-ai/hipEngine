"""Torch-free StepFun GGUF short-context decode planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Mapping, Sequence

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.loading.gguf import GGUFSplitModelInfo, scan_gguf_splits
from hipengine.loading.stepfun_gguf import StepFunGGUFModelMap, build_stepfun_gguf_tensor_map
from hipengine.loading.stepfun_gguf_materialize import (
    StepFunGGUFResidentWeights,
    materialize_stepfun_gguf_weights,
)
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import GGUF_OUTPUT_BF16, GGUF_OUTPUT_F32, launch_gguf_linear
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

        _register_backend_plugin(self.backend)
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


@dataclass
class StepFunResidentSession:
    """Owned resident StepFun state for incremental GGUF decode bring-up.

    The session is intentionally still below the full streaming runner: it owns
    materialized split-GGUF weights and exposes prompt embedding execution as the
    first real resident operation. Layer execution, KV allocation, and logits are
    wired in later P11 iterations.
    """

    info: GGUFSplitModelInfo
    model_map: StepFunGGUFModelMap
    tokenizer: StepFunGGUFTokenizer
    weights: StepFunGGUFResidentWeights
    backend: str = "hip_gfx1151"
    _closed: bool = False

    @classmethod
    def from_gguf_paths(
        cls,
        paths: Sequence[str | Path],
        *,
        backend: str = "hip_gfx1151",
        selected_slots: Sequence[str] | None = None,
        runtime: HipRuntime | None = None,
    ) -> "StepFunResidentSession":
        info = scan_gguf_splits(tuple(Path(path) for path in paths))
        model_map = build_stepfun_gguf_tensor_map(info)
        tokenizer = StepFunGGUFTokenizer.from_gguf_info(info)
        weights = materialize_stepfun_gguf_weights(
            info,
            selected_slots=selected_slots,
            runtime=runtime,
        )
        return cls(
            info=info,
            model_map=model_map,
            tokenizer=tokenizer,
            weights=weights,
            backend=backend,
        )

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        if self._closed:
            return
        self.weights.free(runtime=runtime)
        self._closed = True

    def __enter__(self) -> "StepFunResidentSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.free()

    def weight_for_slot(self, slot_path: str):
        """Return a resident weight by StepFun materialization slot path."""

        if slot_path.startswith("root."):
            return self.weights.root(slot_path.removeprefix("root."))
        if slot_path.startswith("layers."):
            parts = slot_path.split(".", 2)
            if len(parts) != 3:
                raise ValueError(f"invalid StepFun layer slot path: {slot_path!r}")
            return self.weights.layer(int(parts[1])).weight(parts[2])
        raise ValueError(f"invalid StepFun materialization slot path: {slot_path!r}")

    def embed_token_ids_bf16(
        self,
        token_ids: Sequence[int],
        *,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ):
        """Launch resident Q8_0 token embedding and return BF16 bit rows."""

        import numpy as np

        if self._closed:
            raise RuntimeError("StepFun resident session is closed")
        if "token_embedding" not in self.weights.root_weights:
            raise RuntimeError("token_embedding weight is not resident in this session")
        runtime = runtime or get_hip_runtime()
        _register_backend_plugin(self.backend)
        ids = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("token_ids must not be empty")
        vocab_size = int(self.model_map.config.vocab_size)
        if np.any(ids < 0) or np.any(ids >= vocab_size):
            raise ValueError("token_ids contain out-of-range StepFun token IDs")
        rows = int(ids.shape[0])
        hidden_size = int(self.model_map.config.hidden_size)
        out = np.empty((rows, hidden_size), dtype=np.uint16)
        token_buf = malloc(ids.nbytes, runtime=runtime)
        out_buf = malloc(out.nbytes, runtime=runtime)
        try:
            copy_host_to_device(token_buf, host_array_ptr(ids), runtime=runtime)
            launch_gguf_embedding(
                self.weights.root("token_embedding"),
                token_buf.ptr,
                out_buf.ptr,
                rows=rows,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                backend=self.backend,
                stream=stream,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(out), out_buf, runtime=runtime)
        finally:
            free(out_buf, runtime=runtime)
            free(token_buf, runtime=runtime)
        return out

    def linear_slot_bf16(
        self,
        slot_path: str,
        x_bf16_bits,
        *,
        output_dtype: str = GGUF_OUTPUT_F32,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ):
        """Launch a resident GGUF linear slot with BF16-bit activations."""

        import numpy as np

        if self._closed:
            raise RuntimeError("StepFun resident session is closed")
        if output_dtype not in {GGUF_OUTPUT_BF16, GGUF_OUTPUT_F32}:
            raise ValueError(f"unsupported StepFun resident linear output dtype {output_dtype!r}")
        runtime = runtime or get_hip_runtime()
        _register_backend_plugin(self.backend)
        weight = self.weight_for_slot(slot_path)
        if len(weight.spec.source.shape) != 2:
            raise ValueError(f"StepFun linear slot must be rank-2, got {slot_path!r}")
        out_features, in_features = (int(dim) for dim in weight.spec.source.shape)
        x = np.ascontiguousarray(x_bf16_bits, dtype=np.uint16)
        if x.ndim != 2:
            raise ValueError("x_bf16_bits must have shape [rows, in_features]")
        rows = int(x.shape[0])
        if rows <= 0:
            raise ValueError("x_bf16_bits must have at least one row")
        if int(x.shape[1]) != in_features:
            raise ValueError(
                f"x_bf16_bits.shape[1]={x.shape[1]} does not match {slot_path} in_features={in_features}"
            )
        out_dtype = np.uint16 if output_dtype == GGUF_OUTPUT_BF16 else np.float32
        out = np.empty((rows, out_features), dtype=out_dtype)
        x_buf = malloc(x.nbytes, runtime=runtime)
        out_buf = malloc(out.nbytes, runtime=runtime)
        try:
            copy_host_to_device(x_buf, host_array_ptr(x), runtime=runtime)
            launch_gguf_linear(
                weight,
                x_buf.ptr,
                out_buf.ptr,
                rows=rows,
                in_features=in_features,
                out_features=out_features,
                output_dtype=output_dtype,
                backend=self.backend,
                stream=stream,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(out), out_buf, runtime=runtime)
        finally:
            free(out_buf, runtime=runtime)
            free(x_buf, runtime=runtime)
        return out


def _register_backend_plugin(backend: str) -> None:
    # Import-time backend plugins populate aliases/registrations. Resolve by
    # backend module name instead of branching on a concrete backend key.
    backend_module = import_module(f"hipengine.kernels.{backend}")
    registrar_name = f"register_{backend.removeprefix('hip_')}_kernels"
    registrar = getattr(backend_module, registrar_name, None)
    if callable(registrar):
        registrar()


__all__ = [
    "DEFAULT_STEPFUN_MAX_NEW_TOKENS",
    "DEFAULT_STEPFUN_SHORT_CONTEXT",
    "StepFunDecodePlan",
    "StepFunResidentSession",
    "StepFunShortContextDecodePlanner",
]
