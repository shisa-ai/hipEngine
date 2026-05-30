"""Torch-free StepFun GGUF short-context decode planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Mapping, Sequence

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
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
BF16_BYTES = 2


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
class StepFunKVCacheAllocation:
    """Owned synthetic BF16 KV-cache buffers for StepFun decode bring-up."""

    buffers: tuple[DeviceBuffer, ...]
    context_pages: int
    page_size: int
    layer_nbytes: tuple[tuple[int, int], ...]

    @property
    def tokens(self) -> int:
        return self.context_pages * self.page_size

    @property
    def nbytes(self) -> int:
        return sum(key_bytes + value_bytes for key_bytes, value_bytes in self.layer_nbytes)

    @property
    def buffer_count(self) -> int:
        return len(self.buffers)

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for buffer in reversed(self.buffers):
            free(buffer, runtime=runtime)


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

    def allocate_kv_cache(
        self,
        *,
        context_pages: int,
        page_size: int = DEFAULT_STEPFUN_SHORT_CONTEXT,
        runtime: HipRuntime | None = None,
    ) -> StepFunKVCacheAllocation:
        """Allocate per-layer BF16 K/V buffers for StepFun decode bring-up."""

        if self._closed:
            raise RuntimeError("StepFun resident session is closed")
        if context_pages <= 0:
            raise ValueError("context_pages must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        runtime = runtime or get_hip_runtime()
        tokens = int(context_pages) * int(page_size)
        buffers: list[DeviceBuffer] = []
        layer_nbytes: list[tuple[int, int]] = []
        try:
            for kv_heads in self.model_map.config.kv_head_counts:
                key_nbytes = tokens * int(kv_heads) * int(self.model_map.config.head_dim) * BF16_BYTES
                value_nbytes = tokens * int(kv_heads) * int(self.model_map.config.value_dim) * BF16_BYTES
                buffers.append(malloc(key_nbytes, runtime=runtime))
                buffers.append(malloc(value_nbytes, runtime=runtime))
                layer_nbytes.append((key_nbytes, value_nbytes))
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
            raise
        return StepFunKVCacheAllocation(
            buffers=tuple(buffers),
            context_pages=int(context_pages),
            page_size=int(page_size),
            layer_nbytes=tuple(layer_nbytes),
        )

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

    def project_attention_inputs_bf16(
        self,
        layer_id: int,
        x_bf16_bits,
        *,
        output_dtype: str = GGUF_OUTPUT_F32,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ) -> Mapping[str, object]:
        """Launch resident StepFun Q/K/V/gate input projections for one layer."""

        self._validate_layer_id(layer_id)
        prefix = f"layers.{layer_id}"
        return {
            "q": self.linear_slot_bf16(
                f"{prefix}.attn_q",
                x_bf16_bits,
                output_dtype=output_dtype,
                runtime=runtime,
                stream=stream,
            ),
            "k": self.linear_slot_bf16(
                f"{prefix}.attn_k",
                x_bf16_bits,
                output_dtype=output_dtype,
                runtime=runtime,
                stream=stream,
            ),
            "v": self.linear_slot_bf16(
                f"{prefix}.attn_v",
                x_bf16_bits,
                output_dtype=output_dtype,
                runtime=runtime,
                stream=stream,
            ),
            "gate": self.linear_slot_bf16(
                f"{prefix}.attn_gate",
                x_bf16_bits,
                output_dtype=output_dtype,
                runtime=runtime,
                stream=stream,
            ),
        }

    def project_dense_mlp_inputs_bf16(
        self,
        layer_id: int,
        x_bf16_bits,
        *,
        output_dtype: str = GGUF_OUTPUT_F32,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ) -> Mapping[str, object]:
        """Launch resident dense-SwiGLU gate/up projections for one layer."""

        self._validate_layer_id(layer_id)
        layer = self.model_map.layer(layer_id)
        if "ffn_gate" not in layer.tensors or "ffn_up" not in layer.tensors:
            raise RuntimeError(f"layer {layer_id} does not expose dense ffn_gate/ffn_up weights")
        prefix = f"layers.{layer_id}"
        return {
            "gate": self.linear_slot_bf16(
                f"{prefix}.ffn_gate",
                x_bf16_bits,
                output_dtype=output_dtype,
                runtime=runtime,
                stream=stream,
            ),
            "up": self.linear_slot_bf16(
                f"{prefix}.ffn_up",
                x_bf16_bits,
                output_dtype=output_dtype,
                runtime=runtime,
                stream=stream,
            ),
        }

    def dense_mlp_probe_bf16(
        self,
        layer_id: int,
        x_bf16_bits,
        *,
        output_dtype: str = GGUF_OUTPUT_F32,
        runtime: HipRuntime | None = None,
        stream: int = 0,
    ):
        """Correctness probe for a resident dense SwiGLU MLP layer.

        Gate/up/down projections run through resident GGUF linears. SwiGLU and
        BF16 rounding happen on the host until a device-side fused MLP path is
        available, so this is not the final streaming hot path.
        """

        import numpy as np
        from hipengine.loading.materialize import float_array_to_bf16_bits

        projections = self.project_dense_mlp_inputs_bf16(
            layer_id,
            x_bf16_bits,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=runtime,
            stream=stream,
        )
        gate = np.asarray(projections["gate"], dtype=np.float32)
        up = np.asarray(projections["up"], dtype=np.float32)
        activated_gate = gate / (np.float32(1.0) + np.exp(-gate).astype(np.float32))
        limit = float(self.model_map.config.swiglu_clamp_exp[layer_id])
        if limit > 0.0:
            activated_gate = np.minimum(activated_gate, np.float32(limit))
            up = np.clip(up, np.float32(-limit), np.float32(limit))
        fused_bits = float_array_to_bf16_bits(activated_gate * up)
        return self.linear_slot_bf16(
            f"layers.{layer_id}.ffn_down",
            fused_bits,
            output_dtype=output_dtype,
            runtime=runtime,
            stream=stream,
        )

    def _validate_layer_id(self, layer_id: int) -> None:
        if layer_id < 0 or layer_id >= self.model_map.config.block_count:
            raise ValueError(f"layer_id out of range: {layer_id}")


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
    "StepFunKVCacheAllocation",
    "StepFunResidentSession",
    "StepFunShortContextDecodePlanner",
]
