"""Torch-free eager c=1 resident runner for Poolside Laguna S 2.1 GGUF.

This is the correctness-first L6 path: token-serial BF16 execution, mixed
full/SWA ``KVLiveSpans`` attention, deterministic owned scratch, greedy top-1,
and optional caller-owned post-layer hidden taps for the matched DFlash model.
"""

from __future__ import annotations

import ctypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.kernels.backends import (
    hip_target_arch_environment,
    hip_target_arch_for_backend,
    load_backend_kernel_package,
    resolve_backend,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import lm_head_argmax_stage1_blocks
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.laguna_gguf import (
    DENSE_MLP,
    FULL_ATTENTION,
    PER_HEAD_GATE,
    SLIDING_ATTENTION,
    SPARSE_MOE,
    LagunaGGUFConfig,
    laguna_gguf_config_from_metadata,
)
from hipengine.loading.laguna_gguf_materialize import (
    LAYOUT_DENSE_F16,
    LAYOUT_DENSE_F32,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    LagunaGGUFRepackedCache,
    LagunaGGUFResidentLayerWeights,
    LagunaGGUFResidentWeights,
    materialize_laguna_gguf_weights,
)
from hipengine.runtime.f16_weight_linear import (
    launch_f16_weight_linear,
    launch_f16_weight_linear_triple,
)
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import GGUF_OUTPUT_F32, launch_gguf_linear
from hipengine.runtime.laguna_kv import LagunaKVCache, allocate_laguna_kv_cache
from hipengine.runtime.laguna_moe import (
    LagunaMoEKernelPlan,
    LagunaMoEScratch,
    allocate_laguna_moe_scratch,
    resolve_laguna_moe_plan,
    run_laguna_moe_c1,
    validate_laguna_moe_layer,
)
from hipengine.runtime.laguna_rope import (
    LagunaDeviceRoPETables,
    launch_laguna_head_rmsnorm_rope,
    materialize_laguna_rope_tables,
)

LAGUNA_DFLASH_CAPTURE_DEPTHS = (2, 11, 20, 30, 39, 48)
_INITIAL_MAX_CONTEXT = 4_096
_EXPECTED_HEAD_COUNTS = tuple([48, 72, 72, 72] * 12)
_EXPECTED_LAYER_TYPES = tuple(
    FULL_ATTENTION if layer_id % 4 == 0 else SLIDING_ATTENTION for layer_id in range(48)
)
_BF16_NBYTES = DType.BF16.itemsize
_F32_NBYTES = DType.FP32.itemsize
_I64_NBYTES = DType.INT64.itemsize


@dataclass(frozen=True)
class LagunaHiddenCaptureTargets:
    """Caller-owned BF16 destinations for configured post-layer hidden rows."""

    hidden_size: int
    buffers: Mapping[int, DeviceBuffer]

    def __post_init__(self) -> None:
        hidden_size = int(self.hidden_size)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        expected_nbytes = hidden_size * _BF16_NBYTES
        normalized: dict[int, DeviceBuffer] = {}
        for raw_depth, buffer in self.buffers.items():
            depth = int(raw_depth)
            if depth not in LAGUNA_DFLASH_CAPTURE_DEPTHS:
                raise ValueError(
                    "Laguna hidden captures are limited to the configured DFlash depths "
                    f"{LAGUNA_DFLASH_CAPTURE_DEPTHS}; got {depth}"
                )
            if not isinstance(buffer, DeviceBuffer):
                raise TypeError("Laguna hidden capture destinations must be DeviceBuffer views")
            if buffer.nbytes != expected_nbytes:
                raise ValueError(
                    "each Laguna hidden capture target must hold exactly one BF16 hidden row; "
                    f"depth={depth} expected={expected_nbytes} actual={buffer.nbytes}"
                )
            normalized[depth] = buffer
        object.__setattr__(self, "hidden_size", hidden_size)
        object.__setattr__(self, "buffers", MappingProxyType(normalized))


@dataclass(frozen=True)
class LagunaEagerKernelPlan:
    """Exact registry keys and callables used by the eager session."""

    backend: str
    rmsnorm_key: KernelKey
    add_rmsnorm_key: KernelKey
    add_key: KernelKey
    attention_gate_key: KernelKey
    dense_silu_key: KernelKey
    argmax_key: KernelKey
    f16_triple_key: KernelKey
    f16_f32_key: KernelKey
    f16_bf16_key: KernelKey
    rope_key: KernelKey
    rmsnorm: Callable
    add_rmsnorm: Callable
    add: Callable
    attention_gate: Callable
    dense_silu: Callable
    argmax: Callable

    @property
    def kernel_keys(self) -> tuple[KernelKey, ...]:
        return (
            self.rmsnorm_key,
            self.add_rmsnorm_key,
            self.add_key,
            self.attention_gate_key,
            self.dense_silu_key,
            self.argmax_key,
            self.f16_triple_key,
            self.f16_f32_key,
            self.f16_bf16_key,
            self.rope_key,
        )


@dataclass
class LagunaEagerScratch:
    """Deterministic c=1 scratch owner sized for Laguna's widest layer."""

    max_query_width: int
    max_query_heads: int
    token_id: DeviceBuffer
    position: DeviceBuffer
    hidden: DeviceBuffer
    norm: DeviceBuffer
    query: DeviceBuffer
    key: DeviceBuffer
    value: DeviceBuffer
    query_rotated: DeviceBuffer
    key_rotated: DeviceBuffer
    gate_logits: DeviceBuffer
    context: DeviceBuffer
    gated_context: DeviceBuffer
    attention_output: DeviceBuffer
    post_attention: DeviceBuffer
    dense_gate: DeviceBuffer
    dense_up: DeviceBuffer
    dense_intermediate: DeviceBuffer
    dense_output: DeviceBuffer
    final_norm: DeviceBuffer
    logits: DeviceBuffer
    argmax_block_values: DeviceBuffer
    argmax_block_indices: DeviceBuffer
    argmax_id: DeviceBuffer
    argmax_value: DeviceBuffer
    _closed: bool = False

    @classmethod
    def allocate(
        cls,
        config: LagunaGGUFConfig,
        *,
        runtime: HipRuntime | None = None,
    ) -> "LagunaEagerScratch":
        max_heads = max(int(value) for value in config.head_counts)
        max_query_width = max_heads * int(config.key_length)
        kv_width = int(config.head_count_kv) * int(config.key_length)
        hidden = int(config.hidden_size)
        dense_ffn = int(config.feed_forward_length)
        vocab = int(config.vocab_size)
        argmax_blocks = lm_head_argmax_stage1_blocks(vocab)
        sizes = (
            _I64_NBYTES,
            _I64_NBYTES,
            hidden * _BF16_NBYTES,
            hidden * _BF16_NBYTES,
            max_query_width * _F32_NBYTES,
            kv_width * _F32_NBYTES,
            kv_width * _F32_NBYTES,
            max_query_width * _F32_NBYTES,
            kv_width * _F32_NBYTES,
            max_heads * _F32_NBYTES,
            max_query_width * _F32_NBYTES,
            max_query_width * _BF16_NBYTES,
            hidden * _BF16_NBYTES,
            hidden * _BF16_NBYTES,
            dense_ffn * _BF16_NBYTES,
            dense_ffn * _BF16_NBYTES,
            dense_ffn * _BF16_NBYTES,
            hidden * _BF16_NBYTES,
            hidden * _BF16_NBYTES,
            vocab * _F32_NBYTES,
            argmax_blocks * _F32_NBYTES,
            argmax_blocks * _I64_NBYTES,
            _I64_NBYTES,
            _F32_NBYTES,
        )
        buffers: list[DeviceBuffer] = []
        try:
            buffers.extend(malloc(nbytes, runtime=runtime) for nbytes in sizes)
        except Exception:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
            raise
        return cls(max_query_width, max_heads, *buffers)

    @property
    def buffers(self) -> tuple[DeviceBuffer, ...]:
        return (
            self.token_id,
            self.position,
            self.hidden,
            self.norm,
            self.query,
            self.key,
            self.value,
            self.query_rotated,
            self.key_rotated,
            self.gate_logits,
            self.context,
            self.gated_context,
            self.attention_output,
            self.post_attention,
            self.dense_gate,
            self.dense_up,
            self.dense_intermediate,
            self.dense_output,
            self.final_norm,
            self.logits,
            self.argmax_block_values,
            self.argmax_block_indices,
            self.argmax_id,
            self.argmax_value,
        )

    @property
    def nbytes(self) -> int:
        return sum(buffer.nbytes for buffer in self.buffers)

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        for buffer in reversed(self.buffers):
            free(buffer, runtime=runtime)


@dataclass(frozen=True)
class LagunaEagerLibraries:
    """Loaded JIT libraries, held once for the whole resident session."""

    embedding: object
    gguf_ops: object
    f16_projection: object
    attention_gate: object
    kv_attention: object
    dense_silu: object
    argmax: object
    q4_linear: object
    q6_linear: object
    q6_t16_linear: object
    router_logits: object
    router_select: object
    selected_experts: object
    routed_sum: object

    @property
    def linear(self) -> Mapping[str, object]:
        return {
            "gguf_q4_k": self.q4_linear,
            "gguf_q6_k": self.q6_linear,
            "gguf_q6_k_t16_v1": self.q6_t16_linear,
        }

    @property
    def moe(self) -> Mapping[str, object]:
        return {
            **self.linear,
            "router_logits": self.router_logits,
            "router_select": self.router_select,
            "selected_gate_up": self.selected_experts,
            "selected_silu": self.dense_silu,
            "selected_down": self.selected_experts,
            "routed_sum": self.routed_sum,
            "shared_silu": self.dense_silu,
            "add": self.gguf_ops,
        }


@dataclass(frozen=True)
class LagunaEagerTokenResult:
    """One eager token result; device buffers remain owned by the session."""

    position: int
    input_token_id: int
    next_token_id: int
    next_token_logit: float
    logits: DeviceBuffer
    final_hidden: DeviceBuffer
    post_layer_hidden: DeviceBuffer


def resolve_laguna_eager_kernel_plan(
    config: LagunaGGUFConfig,
    *,
    backend: str,
) -> LagunaEagerKernelPlan:
    """Validate the S 2.1 eager contract and resolve only exact registry keys."""

    if not str(backend).startswith("hip_"):
        raise ValueError("Laguna eager execution requires a concrete HIP backend")
    if config.block_count != 48:
        raise ValueError("Laguna S 2.1 eager execution requires exactly 48 layers")
    if config.hidden_size != 3_072 or config.vocab_size != 100_352:
        raise ValueError("Laguna S 2.1 eager execution requires hidden=3072 and vocab=100352")
    if config.head_count_kv != 8 or config.key_length != 128 or config.value_length != 128:
        raise ValueError("Laguna S 2.1 eager execution requires 8 KV heads of dimension 128")
    if tuple(config.head_counts) != _EXPECTED_HEAD_COUNTS:
        raise ValueError("Laguna S 2.1 eager query-head sequence must repeat 48/72/72/72")
    if tuple(config.layer_types) != _EXPECTED_LAYER_TYPES:
        raise ValueError("Laguna S 2.1 eager layer sequence must repeat global/SWA/SWA/SWA")
    if config.sliding_window != 512:
        raise ValueError("Laguna S 2.1 eager execution requires a 512-token SWA ring")
    if config.leading_dense_block_count != 1:
        raise ValueError("Laguna S 2.1 eager execution requires one leading dense layer")

    load_backend_kernel_package(backend)
    keys = {
        "rmsnorm": KernelKey(backend, "rmsnorm", "gguf_f32_weight", "bf16_out"),
        "add_rmsnorm": KernelKey(backend, "add_rmsnorm", "gguf_f32_weight", "bf16_out"),
        "add": KernelKey(backend, "elementwise", "bf16", "add"),
        "attention_gate": KernelKey(
            backend, "attention_gate", "f32", "softplus_broadcast_bf16_out"
        ),
        "dense_silu": KernelKey(backend, "silu_mul_separate", "bf16", "out"),
        "argmax": KernelKey(backend, "argmax", "f32", "top1_i64"),
        "f16_triple": KernelKey(backend, "linear_triple", "fp16_weight", "bf16_f32_out"),
        "f16_f32": KernelKey(backend, "linear", "fp16_weight", "bf16_f32_out"),
        "f16_bf16": KernelKey(backend, "linear", "fp16_weight", "bf16_bf16_out"),
        "rope": KernelKey(
            backend,
            "head_rmsnorm+partial_rotary",
            "laguna_f32_weight",
            "positions_f32",
        ),
    }
    functions = {name: _resolve_exact(key) for name, key in keys.items()}
    return LagunaEagerKernelPlan(
        backend=backend,
        rmsnorm_key=keys["rmsnorm"],
        add_rmsnorm_key=keys["add_rmsnorm"],
        add_key=keys["add"],
        attention_gate_key=keys["attention_gate"],
        dense_silu_key=keys["dense_silu"],
        argmax_key=keys["argmax"],
        f16_triple_key=keys["f16_triple"],
        f16_f32_key=keys["f16_f32"],
        f16_bf16_key=keys["f16_bf16"],
        rope_key=keys["rope"],
        rmsnorm=functions["rmsnorm"],
        add_rmsnorm=functions["add_rmsnorm"],
        add=functions["add"],
        attention_gate=functions["attention_gate"],
        dense_silu=functions["dense_silu"],
        argmax=functions["argmax"],
    )


def capture_laguna_hidden_tap(
    source_bf16_ptr: int,
    *,
    depth: int,
    targets: LagunaHiddenCaptureTargets | None,
    hidden_size: int,
    runtime: HipRuntime,
    stream: int = 0,
) -> None:
    """Copy one requested tap; the ``None`` path performs no runtime call."""

    if targets is None:
        return
    if int(targets.hidden_size) != int(hidden_size):
        raise ValueError("Laguna hidden capture hidden_size does not match the session")
    target = targets.buffers.get(int(depth))
    if target is None:
        return
    runtime.memcpy_async(
        target.ptr,
        int(source_bf16_ptr),
        int(hidden_size) * _BF16_NBYTES,
        HipMemcpyKind.DEVICE_TO_DEVICE,
        int(stream),
    )


def load_laguna_eager_libraries(
    *,
    backend: str,
    compiler_version: str | None = None,
    require_cached: bool = False,
) -> LagunaEagerLibraries:
    """Build/load every library used by one eager session exactly once."""

    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
    )
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import build_gguf_ops
    from hipengine.kernels.hip_gfx1100.fused.laguna_attention import (
        build_laguna_attention,
    )
    from hipengine.kernels.hip_gfx1100.fused.paro_combine import build_paro_combine
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import build_paro_silu
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        build_laguna_f16_projection,
    )
    from hipengine.kernels.hip_gfx1100.linear.lm_head import build_lm_head
    from hipengine.kernels.hip_gfx1100.moe.laguna_router import build_laguna_router
    from hipengine.kernels.hip_gfx1100.moe.router import build_qwen35_router
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import build_gguf_k_gemv
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import (
        build_gguf_q6_k_embedding,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        build_gguf_q6_k_t16_gemv,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        build_gguf_t16_selected_gemv,
    )

    kwargs = {
        "compiler_version": compiler_version,
        "require_cached": require_cached,
        "load": True,
    }
    target_arch = hip_target_arch_for_backend(backend)
    with hip_target_arch_environment(target_arch):
        return LagunaEagerLibraries(
            embedding=build_gguf_q6_k_embedding(**kwargs),
            gguf_ops=build_gguf_ops(**kwargs),
            f16_projection=build_laguna_f16_projection(**kwargs),
            attention_gate=build_laguna_attention(**kwargs),
            kv_attention=build_laguna_kv_attention(**kwargs),
            dense_silu=build_paro_silu(**kwargs),
            argmax=build_lm_head(**kwargs),
            q4_linear=build_gguf_q4_k_gemv(**kwargs),
            q6_linear=build_gguf_k_gemv(**kwargs),
            q6_t16_linear=build_gguf_q6_k_t16_gemv(**kwargs),
            router_logits=build_qwen35_router(**kwargs),
            router_select=build_laguna_router(**kwargs),
            selected_experts=build_gguf_t16_selected_gemv(**kwargs),
            routed_sum=build_paro_combine(**kwargs),
        )


class LagunaGGUFResidentSession:
    """All-resident eager Laguna S 2.1 c=1 session with BF16 KV state."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        resident_weights: LagunaGGUFResidentWeights | None = None,
        context_length: int = _INITIAL_MAX_CONTEXT,
        backend: str = "hip_gfx1151",
        runtime: HipRuntime | None = None,
        device: Device | None = None,
        compiler_version: str | None = None,
        require_cached_build: bool = False,
        available_bytes: int | None = None,
        progress: Callable | None = None,
        repacked_cache: LagunaGGUFRepackedCache | str | Path | None = None,
        model_sha256: str | None = None,
    ) -> None:
        self.runtime = runtime or get_hip_runtime()
        self.device = device or Device("hip", 0)
        self.backend = resolve_backend(backend)
        self.context_length = int(context_length)
        self.position = -1
        self.last_result: LagunaEagerTokenResult | None = None
        self.weights: LagunaGGUFResidentWeights | None = None
        self.kv_cache: LagunaKVCache | None = None
        self.scratch: LagunaEagerScratch | None = None
        self.moe_scratch: LagunaMoEScratch | None = None
        self.full_rope: LagunaDeviceRoPETables | None = None
        self.swa_rope: LagunaDeviceRoPETables | None = None
        self.libraries: LagunaEagerLibraries | None = None
        self.kernel_plan: LagunaEagerKernelPlan | None = None
        self.moe_plan: LagunaMoEKernelPlan | None = None
        self._owns_weights = resident_weights is None
        self._closed = False

        if self.context_length <= 0 or self.context_length > _INITIAL_MAX_CONTEXT:
            raise ValueError(
                f"initial Laguna eager context_length must be within [1, {_INITIAL_MAX_CONTEXT}]"
            )
        if resident_weights is not None and (
            repacked_cache is not None or model_sha256 is not None
        ):
            raise ValueError(
                "repacked_cache/model_sha256 apply only when the session owns model loading"
            )
        try:
            if resident_weights is None:
                if model_path is None:
                    raise ValueError("model_path is required without resident_weights")
                reader = GGUFReader(model_path)
                config = laguna_gguf_config_from_metadata(reader.info)
            else:
                config = resident_weights.config
                if resident_weights.backend != self.backend:
                    raise ValueError("resident Laguna backend does not match the session backend")

            self.kernel_plan = resolve_laguna_eager_kernel_plan(
                config,
                backend=self.backend,
            )
            self.libraries = load_laguna_eager_libraries(
                backend=self.backend,
                compiler_version=compiler_version,
                require_cached=require_cached_build,
            )
            if resident_weights is None:
                self.weights = materialize_laguna_gguf_weights(
                    reader,
                    context_length=self.context_length,
                    available_bytes=available_bytes,
                    device=self.device,
                    runtime=self.runtime,
                    backend=self.backend,
                    progress=progress,
                    repacked_cache=repacked_cache,
                    repacked_cache_source_sha256=model_sha256,
                )
            else:
                self.weights = resident_weights
            self.moe_plan = resolve_laguna_moe_plan(config, backend=self.backend)
            self._validate_resident_weights()
            self.full_rope = materialize_laguna_rope_tables(
                self.context_length,
                config.full_rope,
                device=self.device,
                runtime=self.runtime,
            )
            if config.swa_rope is None:
                raise ValueError("Laguna eager session requires SWA RoPE metadata")
            self.swa_rope = materialize_laguna_rope_tables(
                self.context_length,
                config.swa_rope,
                device=self.device,
                runtime=self.runtime,
            )
            self.kv_cache = allocate_laguna_kv_cache(
                config,
                context_length=self.context_length,
                backend=self.backend,
                device=self.device,
                runtime=self.runtime,
            )
            self.scratch = LagunaEagerScratch.allocate(config, runtime=self.runtime)
            self.moe_scratch = allocate_laguna_moe_scratch(
                self.moe_plan,
                runtime=self.runtime,
            )
        except BaseException:
            self._close(suppress_errors=True)
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def config(self) -> LagunaGGUFConfig:
        self._check_open()
        assert self.weights is not None
        return self.weights.config

    @property
    def resident_nbytes(self) -> int:
        self._check_open()
        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.scratch is not None
        assert self.moe_scratch is not None
        assert self.full_rope is not None
        assert self.swa_rope is not None
        return (
            self.weights.resident_nbytes
            + self.kv_cache.resident_nbytes
            + self.scratch.nbytes
            + self.moe_scratch.nbytes
            + self.full_rope.cos.buffer.nbytes
            + self.full_rope.sin.buffer.nbytes
            + self.swa_rope.cos.buffer.nbytes
            + self.swa_rope.sin.buffer.nbytes
        )

    def forward_token(
        self,
        token_id: int,
        *,
        captures: LagunaHiddenCaptureTargets | None = None,
        stream: int = 0,
    ) -> LagunaEagerTokenResult:
        """Append and execute one token, then return the borrowed top-1 result."""

        self._check_open()
        config = self.config
        token = int(token_id)
        if token < 0 or token >= config.vocab_size:
            raise ValueError(f"token_id must be within [0, {config.vocab_size})")
        if captures is not None and captures.hidden_size != config.hidden_size:
            raise ValueError("Laguna hidden capture hidden_size does not match the session")
        next_position = self.position + 1
        if next_position >= self.context_length:
            raise ValueError("Laguna eager session exhausted its admitted context")

        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        try:
            _copy_i64(self.scratch.token_id, token, self.runtime)
            _copy_i64(self.scratch.position, next_position, self.runtime)
            self.kv_cache.prepare_position(next_position)
            launch_gguf_embedding(
                self.weights.root("token_embedding"),
                self.scratch.token_id.ptr,
                self.scratch.hidden.ptr,
                1,
                config.hidden_size,
                config.vocab_size,
                backend=self.backend,
                stream=stream,
                libraries={"gguf_q4_k": self.libraries.embedding},
                runtime=self.runtime,
            )
            for layer_id in range(config.block_count):
                self._run_layer(layer_id, stream=stream)
                capture_laguna_hidden_tap(
                    self.scratch.hidden.ptr,
                    depth=layer_id + 1,
                    targets=captures,
                    hidden_size=config.hidden_size,
                    runtime=self.runtime,
                    stream=stream,
                )
            result = self._project_and_sample(
                input_token_id=token,
                position=next_position,
                stream=stream,
            )
            self.position = next_position
            self.last_result = result
            return result
        except BaseException:
            self._close(suppress_errors=True)
            raise

    def prefill(
        self,
        token_ids: Sequence[int],
        *,
        capture_last: LagunaHiddenCaptureTargets | None = None,
        stream: int = 0,
    ) -> LagunaEagerTokenResult:
        """Token-serial correctness fallback; only the final row may emit taps."""

        tokens = tuple(int(value) for value in token_ids)
        if not tokens:
            raise ValueError("Laguna eager prefill requires at least one token")
        result: LagunaEagerTokenResult | None = None
        for index, token in enumerate(tokens):
            result = self.forward_token(
                token,
                captures=capture_last if index == len(tokens) - 1 else None,
                stream=stream,
            )
        assert result is not None
        return result

    def generate_greedy(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        stop_token_ids: Sequence[int] = (),
        stream: int = 0,
    ) -> tuple[int, ...]:
        """Diagnostic target-only greedy loop over the resident eager state."""

        count = int(max_new_tokens)
        if count <= 0:
            raise ValueError("max_new_tokens must be positive")
        result = self.prefill(prompt_token_ids, stream=stream)
        stops = {int(value) for value in stop_token_ids}
        generated: list[int] = []
        for _ in range(count):
            token = int(result.next_token_id)
            generated.append(token)
            if token in stops:
                break
            if len(generated) == count:
                break
            result = self.forward_token(token, stream=stream)
        return tuple(generated)

    def close(self) -> None:
        self._close(suppress_errors=False)

    def __enter__(self) -> "LagunaGGUFResidentSession":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _run_layer(self, layer_id: int, *, stream: int) -> None:
        assert self.weights is not None
        assert self.kv_cache is not None
        assert self.scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        assert self.full_rope is not None
        assert self.swa_rope is not None
        config = self.weights.config
        layer = self.weights.layer(layer_id)
        scratch = self.scratch
        heads = config.head_count(layer_id)
        q_width = heads * config.key_length
        kv_width = config.head_count_kv * config.key_length
        f16_libraries = {"fp16_weight": self.libraries.f16_projection}

        self.kernel_plan.rmsnorm(
            scratch.hidden.ptr,
            layer.weight("attn_norm").allocation("raw").tensor.ptr,
            scratch.norm.ptr,
            1,
            config.hidden_size,
            config.rms_norm_eps,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )
        launch_f16_weight_linear_triple(
            layer.weight("attn_q"),
            layer.weight("attn_k"),
            layer.weight("attn_v"),
            scratch.norm.ptr,
            scratch.query.ptr,
            scratch.key.ptr,
            scratch.value.ptr,
            1,
            config.hidden_size,
            q_width,
            kv_width,
            kv_width,
            backend=self.backend,
            stream=stream,
            libraries=f16_libraries,
            runtime=self.runtime,
        )
        launch_f16_weight_linear(
            layer.weight("attn_gate"),
            scratch.norm.ptr,
            scratch.gate_logits.ptr,
            1,
            config.hidden_size,
            heads,
            activation_dtype="bf16",
            output_dtype="f32",
            backend=self.backend,
            stream=stream,
            libraries=f16_libraries,
            runtime=self.runtime,
        )
        rope = self.full_rope if layer.attention_type == FULL_ATTENTION else self.swa_rope
        launch_laguna_head_rmsnorm_rope(
            scratch.query.ptr,
            scratch.key.ptr,
            layer.weight("attn_q_norm").allocation("raw").tensor.ptr,
            layer.weight("attn_k_norm").allocation("raw").tensor.ptr,
            scratch.position.ptr,
            scratch.query_rotated.ptr,
            scratch.key_rotated.ptr,
            config.rms_norm_eps,
            1,
            heads,
            config.head_count_kv,
            config.key_length,
            rope,
            backend=self.backend,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )
        self.kv_cache.append(
            layer_id,
            scratch.key_rotated.ptr,
            scratch.value.ptr,
            stream=stream,
            library=self.libraries.kv_attention,
        )
        self.kv_cache.attend(
            layer_id,
            scratch.query_rotated.ptr,
            scratch.context.ptr,
            stream=stream,
            library=self.libraries.kv_attention,
        )
        self.kernel_plan.attention_gate(
            scratch.context.ptr,
            scratch.gate_logits.ptr,
            scratch.gated_context.ptr,
            1,
            heads,
            config.value_length,
            stream=stream,
            library=self.libraries.attention_gate,
            runtime=self.runtime,
        )
        launch_f16_weight_linear(
            layer.weight("attn_output"),
            scratch.gated_context.ptr,
            scratch.attention_output.ptr,
            1,
            q_width,
            config.hidden_size,
            activation_dtype="bf16",
            output_dtype="bf16",
            backend=self.backend,
            stream=stream,
            libraries=f16_libraries,
            runtime=self.runtime,
        )
        self.kernel_plan.add_rmsnorm(
            scratch.hidden.ptr,
            scratch.attention_output.ptr,
            layer.weight("ffn_norm").allocation("raw").tensor.ptr,
            scratch.norm.ptr,
            scratch.post_attention.ptr,
            1,
            config.hidden_size,
            config.rms_norm_eps,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )
        if layer.mlp_type == DENSE_MLP:
            self._run_dense_ffn(layer, stream=stream)
        elif layer.mlp_type == SPARSE_MOE:
            self._run_sparse_ffn(layer, stream=stream)
        else:
            raise ValueError(f"unsupported Laguna MLP type {layer.mlp_type!r}")

    def _run_dense_ffn(
        self,
        layer: LagunaGGUFResidentLayerWeights,
        *,
        stream: int,
    ) -> None:
        assert self.weights is not None
        assert self.scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        config = self.weights.config
        scratch = self.scratch
        linear_libraries = self.libraries.linear
        for slot, output in (
            ("ffn_gate", scratch.dense_gate),
            ("ffn_up", scratch.dense_up),
        ):
            launch_gguf_linear(
                layer.weight(slot),
                scratch.norm.ptr,
                output.ptr,
                1,
                config.hidden_size,
                config.feed_forward_length,
                backend=self.backend,
                stream=stream,
                libraries=linear_libraries,
                runtime=self.runtime,
                use_wmma_prefill=False,
                use_gemv_decode=False,
            )
        self.kernel_plan.dense_silu(
            scratch.dense_gate.ptr,
            scratch.dense_up.ptr,
            scratch.dense_intermediate.ptr,
            1,
            config.feed_forward_length,
            stream=stream,
            library=self.libraries.dense_silu,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            layer.weight("ffn_down"),
            scratch.dense_intermediate.ptr,
            scratch.dense_output.ptr,
            1,
            config.feed_forward_length,
            config.hidden_size,
            backend=self.backend,
            stream=stream,
            libraries=linear_libraries,
            runtime=self.runtime,
            use_wmma_prefill=False,
            use_gemv_decode=False,
        )
        self.kernel_plan.add(
            scratch.post_attention.ptr,
            scratch.dense_output.ptr,
            scratch.hidden.ptr,
            config.hidden_size,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )

    def _run_sparse_ffn(
        self,
        layer: LagunaGGUFResidentLayerWeights,
        *,
        stream: int,
    ) -> None:
        assert self.weights is not None
        assert self.scratch is not None
        assert self.moe_scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        output = run_laguna_moe_c1(
            self.scratch.norm.ptr,
            layer,
            self.moe_scratch,
            stream=stream,
            runtime=self.runtime,
            libraries=self.libraries.moe,
        )
        self.kernel_plan.add(
            self.scratch.post_attention.ptr,
            output.ptr,
            self.scratch.hidden.ptr,
            self.weights.config.hidden_size,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )

    def _project_and_sample(
        self,
        *,
        input_token_id: int,
        position: int,
        stream: int,
    ) -> LagunaEagerTokenResult:
        assert self.weights is not None
        assert self.scratch is not None
        assert self.kernel_plan is not None
        assert self.libraries is not None
        config = self.weights.config
        scratch = self.scratch
        self.kernel_plan.rmsnorm(
            scratch.hidden.ptr,
            self.weights.root("output_norm").allocation("raw").tensor.ptr,
            scratch.final_norm.ptr,
            1,
            config.hidden_size,
            config.rms_norm_eps,
            stream=stream,
            library=self.libraries.gguf_ops,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            self.weights.root("lm_head"),
            scratch.final_norm.ptr,
            scratch.logits.ptr,
            1,
            config.hidden_size,
            config.vocab_size,
            output_dtype=GGUF_OUTPUT_F32,
            backend=self.backend,
            stream=stream,
            libraries=self.libraries.linear,
            runtime=self.runtime,
            use_wmma_prefill=False,
            use_gemv_decode=False,
        )
        self.kernel_plan.argmax(
            scratch.logits.ptr,
            scratch.argmax_block_values.ptr,
            scratch.argmax_block_indices.ptr,
            scratch.argmax_id.ptr,
            scratch.argmax_value.ptr,
            config.vocab_size,
            stream=stream,
            library=self.libraries.argmax,
            runtime=self.runtime,
        )
        if stream:
            self.runtime.stream_synchronize(stream)
        else:
            self.runtime.device_synchronize()
        next_id = _read_i64(scratch.argmax_id, self.runtime)
        next_value = _read_f32(scratch.argmax_value, self.runtime)
        return LagunaEagerTokenResult(
            position=position,
            input_token_id=input_token_id,
            next_token_id=next_id,
            next_token_logit=next_value,
            logits=scratch.logits,
            final_hidden=scratch.final_norm,
            post_layer_hidden=scratch.hidden,
        )

    def _validate_resident_weights(self) -> None:
        assert self.weights is not None
        config = self.weights.config
        if self.weights.backend != self.backend:
            raise ValueError("Laguna resident weights must share the session backend")
        for slot in ("token_embedding", "output_norm", "lm_head"):
            self.weights.root(slot)
        if len(self.weights.layers) != config.block_count:
            raise ValueError("Laguna resident layer count does not match GGUF metadata")
        for layer_id, layer in enumerate(self.weights.layers):
            heads = config.head_count(layer_id)
            expected_attention = {
                "attn_norm": ((config.hidden_size,), LAYOUT_DENSE_F32),
                "attn_q": ((heads * config.key_length, config.hidden_size), LAYOUT_DENSE_F16),
                "attn_k": (
                    (config.head_count_kv * config.key_length, config.hidden_size),
                    LAYOUT_DENSE_F16,
                ),
                "attn_v": (
                    (config.head_count_kv * config.value_length, config.hidden_size),
                    LAYOUT_DENSE_F16,
                ),
                "attn_gate": ((heads, config.hidden_size), LAYOUT_DENSE_F16),
                "attn_q_norm": ((config.key_length,), LAYOUT_DENSE_F32),
                "attn_k_norm": ((config.key_length,), LAYOUT_DENSE_F32),
                "attn_output": (
                    (config.hidden_size, heads * config.value_length),
                    LAYOUT_DENSE_F16,
                ),
                "ffn_norm": ((config.hidden_size,), LAYOUT_DENSE_F32),
            }
            for slot, (shape, layout) in expected_attention.items():
                weight = layer.weight(slot)
                if weight.spec.source.shape != shape or weight.spec.layout != layout:
                    raise ValueError(f"Laguna layer {layer_id} {slot} resident contract mismatch")
            if layer.weight("attn_gate").spec.source.shape != (
                heads,
                config.hidden_size,
            ):
                raise ValueError(
                    f"Laguna layer {layer_id} requires {PER_HEAD_GATE!r} attention gating"
                )
            if layer.mlp_type == DENSE_MLP:
                dense_expected = {
                    "ffn_gate": LAYOUT_Q4_K_PACK8,
                    "ffn_up": LAYOUT_Q4_K_PACK8,
                    "ffn_down": LAYOUT_RAW_GGUF,
                }
                for slot, layout in dense_expected.items():
                    if layer.weight(slot).spec.layout != layout:
                        raise ValueError(f"Laguna dense layer {layer_id} {slot} layout mismatch")
            elif layer.mlp_type != SPARSE_MOE:
                raise ValueError(f"unsupported Laguna MLP type {layer.mlp_type!r}")
            else:
                continue
        assert self.moe_plan is not None
        for layer in self.weights.layers[config.leading_dense_block_count :]:
            validate_laguna_moe_layer(layer, self.moe_plan)

    def _close(self, *, suppress_errors: bool) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []

        def release(action: Callable[[], None]) -> None:
            try:
                action()
            except BaseException as exc:  # best-effort teardown after HIP failures
                errors.append(exc)

        if self.moe_scratch is not None:
            scratch = self.moe_scratch
            self.moe_scratch = None
            release(lambda: scratch.free(runtime=self.runtime))
        if self.scratch is not None:
            scratch = self.scratch
            self.scratch = None
            release(lambda: scratch.free(runtime=self.runtime))
        if self.kv_cache is not None:
            cache = self.kv_cache
            self.kv_cache = None
            release(cache.free)
        if self.swa_rope is not None:
            tables = self.swa_rope
            self.swa_rope = None
            release(lambda: tables.free(runtime=self.runtime))
        if self.full_rope is not None:
            tables = self.full_rope
            self.full_rope = None
            release(lambda: tables.free(runtime=self.runtime))
        if self.weights is not None:
            weights = self.weights
            self.weights = None
            if self._owns_weights:
                release(lambda: weights.free(runtime=self.runtime))
        self.kernel_plan = None
        self.moe_plan = None
        self.libraries = None
        if errors and not suppress_errors:
            raise RuntimeError("one or more Laguna session resources failed to free") from errors[0]

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Laguna GGUF resident session is closed")


def _copy_i64(buffer: DeviceBuffer, value: int, runtime: HipRuntime) -> None:
    host = ctypes.c_int64(int(value))
    runtime.memcpy(
        buffer.ptr,
        ctypes.addressof(host),
        ctypes.sizeof(host),
        HipMemcpyKind.HOST_TO_DEVICE,
    )


def _read_i64(buffer: DeviceBuffer, runtime: HipRuntime) -> int:
    host = ctypes.c_int64()
    runtime.memcpy(
        ctypes.addressof(host),
        buffer.ptr,
        ctypes.sizeof(host),
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return int(host.value)


def _read_f32(buffer: DeviceBuffer, runtime: HipRuntime) -> float:
    host = ctypes.c_float()
    runtime.memcpy(
        ctypes.addressof(host),
        buffer.ptr,
        ctypes.sizeof(host),
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return float(host.value)


def _resolve_exact(key: KernelKey) -> Callable:
    if not is_registered(key):
        raise LookupError(f"required Laguna eager kernel is not registered: {key.display()}")
    function = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    assert function is not None
    return function


__all__ = [
    "LAGUNA_DFLASH_CAPTURE_DEPTHS",
    "LagunaEagerKernelPlan",
    "LagunaEagerLibraries",
    "LagunaEagerScratch",
    "LagunaEagerTokenResult",
    "LagunaGGUFResidentSession",
    "LagunaHiddenCaptureTargets",
    "capture_laguna_hidden_tap",
    "load_laguna_eager_libraries",
    "resolve_laguna_eager_kernel_plan",
]
