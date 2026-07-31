"""Torch-free fixed-address Moonshine resident runtime and FP16 decoder."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Iterator

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
    memory_stats,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference.moonshine import moonshine_rope_tables
from hipengine.loading.moonshine import MoonshineLoadedModel, load_moonshine_model
from hipengine.runtime.workspace import RuntimeWorkspace


MOONSHINE_CROSS_KV_THREADS = 32
MOONSHINE_CROSS_ATTENTION_THREADS = 256
MOONSHINE_TRIPLE_QKV_THREADS = 32
MOONSHINE_SINGLE_PROJECTION_THREADS = 64
MOONSHINE_MLP_FC1_THREADS = 32
MOONSHINE_MLP_FC2_THREADS = 64


def _moonshine_self_attention_threads(position: int) -> int:
    if position <= 0:
        raise ValueError("parallel self-attention position must be positive")
    if position == 1:
        return 64
    if position <= 3:
        return 128
    return 256


def _moonshine_token_graph_bucket(
    position: int,
    *,
    capacity: int = 194,
) -> tuple[str, int, int]:
    """Return the general self-attention launch bucket for one decode position."""

    if isinstance(position, bool) or not isinstance(position, int):
        raise ValueError("position must be an integer")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if position < 0 or position >= capacity:
        raise ValueError("position is outside self-cache capacity")
    if position == 0:
        return ("position_0", 0, 0)
    if position == 1:
        return ("position_1", 1, 1)
    if position <= 3:
        return ("positions_2_3", 2, min(3, capacity - 1))
    return (f"positions_4_{capacity - 1}", 4, capacity - 1)


class NoAllocationError(RuntimeError):
    """Raised when a future timed region allocates through hipEngine memory."""


@dataclass(frozen=True)
class MoonshineCacheView:
    key: Tensor
    value: Tensor


@dataclass(frozen=True)
class MoonshineDecoderLibraries:
    """Prebuilt code objects used by the resident decoder chain."""

    projection: object
    dense_projection: object
    layernorm: object
    glue: object
    mlp: object
    attention: object


@dataclass
class MoonshineTokenGraph:
    """One captured fixed-address token DAG for a self-cache launch bucket."""

    owner: "MoonshineResidentRuntime"
    bucket: str
    minimum_position: int
    maximum_position: int
    capture_position: int
    graph: int
    graph_exec: int
    capture_wall_ms: float
    instantiate_wall_ms: float
    replay_count: int = 0
    closed: bool = False

    @property
    def position_range(self) -> tuple[int, int]:
        return (self.minimum_position, self.maximum_position)

    def accepts(self, position: int) -> bool:
        return self.minimum_position <= int(position) <= self.maximum_position

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.owner.runtime.graph_exec_destroy(self.graph_exec)
        self.owner.runtime.graph_destroy(self.graph)
        current = self.owner._token_graphs.get(self.bucket)
        if current is self:
            del self.owner._token_graphs[self.bucket]


class MoonshineResidentRuntime:
    """Own every fixed-address object needed by a batch-one Moonshine decoder."""

    _SCRATCH_NAMES = (
        "hidden",
        "residual",
        "normalized",
        "query",
        "key",
        "value",
        "attention",
        "projection",
        "mlp_fc1",
        "mlp_intermediate",
        "logits",
        "token",
        "position",
        "argmax",
    )

    def __init__(
        self,
        *,
        encoder_frames: int,
        model_path: str | Path | None = None,
        loaded_model: MoonshineLoadedModel | None = None,
        device: Device | None = None,
        runtime: HipRuntime | None = None,
    ) -> None:
        if (model_path is None) == (loaded_model is None):
            raise ValueError("provide exactly one of model_path or loaded_model")
        self.runtime = runtime or get_hip_runtime()
        self.device = device or Device("hip", 0)
        self.loaded_model = loaded_model
        self.weights = loaded_model.weights if loaded_model is not None else None
        self.spec = loaded_model.spec if loaded_model is not None else None
        self.encoder_frames = int(encoder_frames)
        self.workspace = RuntimeWorkspace(device=self.device, runtime=self.runtime)
        self.stream = 0
        self.start_event = 0
        self.stop_event = 0
        self.self_cache_length = 0
        self.cross_cache_valid = False
        self.encoder_state_valid = False
        self.decode_position: int | None = None
        self.decoder_libraries: MoonshineDecoderLibraries | None = None
        self._token_graphs: dict[str, MoonshineTokenGraph] = {}
        self.closed = False
        self.teardown_returned_to_baseline: bool | None = None
        try:
            if self.loaded_model is None:
                self.loaded_model = load_moonshine_model(
                    model_path,
                    device=self.device,
                    runtime=self.runtime,
                )
                self.weights = self.loaded_model.weights
                self.spec = self.loaded_model.spec
            assert self.loaded_model is not None
            assert self.weights is not None
            assert self.spec is not None
            valid_frames = {frames for _, frames in self.spec.encoder_buckets}
            if self.encoder_frames not in valid_frames:
                expected = ", ".join(str(value) for value in sorted(valid_frames))
                raise ValueError(
                    f"encoder frame bucket {self.encoder_frames} is unsupported; expected one of {expected}"
                )
            self.stream = self.runtime.stream_create(nonblocking=True)
            self.start_event = self.runtime.event_create()
            self.stop_event = self.runtime.event_create()
            self._reserve_workspace()
            self._initialize_workspace()
        except Exception:
            self.close()
            raise

    def _reserve_workspace(self) -> None:
        assert self.spec is not None
        spec = self.spec
        reserve = self.workspace.reserve_tensor
        reserve("rope_cos", (spec.max_positions, spec.rotary_dim // 2), DType.FP16)
        reserve("rope_sin", (spec.max_positions, spec.rotary_dim // 2), DType.FP16)
        reserve("encoder_hidden", (1, self.encoder_frames, spec.hidden_size), DType.FP16)
        reserve("encoder_attention_mask", (1, self.encoder_frames), DType.INT32)
        reserve(
            "self_kv",
            (
                spec.decoder_layers,
                2,
                1,
                spec.decoder_kv_heads,
                spec.self_cache_capacity,
                spec.head_dim,
            ),
            DType.FP16,
        )
        reserve(
            "cross_kv",
            (
                spec.decoder_layers,
                2,
                1,
                spec.decoder_kv_heads,
                self.encoder_frames,
                spec.head_dim,
            ),
            DType.FP16,
        )
        reserve("hidden", (1, 1, spec.hidden_size), DType.FP16)
        reserve("residual", (1, 1, spec.hidden_size), DType.FP16)
        reserve("normalized", (1, 1, spec.hidden_size), DType.FP16)
        for name in ("query", "key", "value", "attention"):
            reserve(
                name,
                (1, spec.decoder_attention_heads, 1, spec.head_dim),
                DType.FP16,
            )
        reserve("projection", (1, 1, spec.hidden_size), DType.FP16)
        reserve("mlp_fc1", (1, 1, 2 * spec.intermediate_size), DType.FP16)
        reserve("mlp_intermediate", (1, 1, spec.intermediate_size), DType.FP16)
        reserve("logits", (1, spec.vocab_size), DType.FP16)
        reserve("token", (1,), DType.INT64)
        reserve("position", (1,), DType.INT64)
        reserve("argmax", (1,), DType.INT64)

    def _initialize_workspace(self) -> None:
        for name in self.workspace.names:
            allocation = self.workspace.allocation(name)
            self.runtime.memset_async(
                allocation.buffer.ptr,
                0,
                allocation.buffer.nbytes,
                self.stream,
            )
        self.runtime.stream_synchronize(self.stream)
        assert self.spec is not None
        cos, sin = moonshine_rope_tables(
            self.spec.max_positions,
            rotary_dim=self.spec.rotary_dim,
            theta=self.spec.rope_theta,
        )
        copy_host_to_device(
            self.workspace.allocation("rope_cos").buffer,
            host_array_ptr(cos),
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.workspace.allocation("rope_sin").buffer,
            host_array_ptr(sin),
            runtime=self.runtime,
        )

    def tensor(self, name: str) -> Tensor:
        return self.workspace.allocation(name).tensor

    @property
    def resident_nbytes(self) -> int:
        if self.loaded_model is None:
            return sum(
                self.workspace.allocation(name).buffer.nbytes for name in self.workspace.names
            )
        return self.loaded_model.owned_weight_bytes + sum(
            self.workspace.allocation(name).buffer.nbytes for name in self.workspace.names
        )

    def _cache_view(self, name: str, layer: int) -> MoonshineCacheView:
        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine runtime is closed")
        if layer < 0 or layer >= self.spec.decoder_layers:
            raise ValueError("cache layer is outside the decoder")
        allocation = self.workspace.allocation(name)
        capacity = self.spec.self_cache_capacity if name == "self_kv" else self.encoder_frames
        shape = (1, self.spec.decoder_kv_heads, capacity, self.spec.head_dim)
        elements = self.spec.decoder_kv_heads * capacity * self.spec.head_dim
        strides = (
            self.spec.decoder_kv_heads * capacity * self.spec.head_dim,
            capacity * self.spec.head_dim,
            self.spec.head_dim,
            1,
        )

        def view(slot: int) -> Tensor:
            offset = (layer * 2 + slot) * elements * DType.FP16.itemsize
            return Tensor.from_handle(
                allocation.buffer.ptr + offset,
                shape,
                DType.FP16,
                self.device,
                strides=strides,
            )

        return MoonshineCacheView(view(0), view(1))

    def self_cache(self, layer: int) -> MoonshineCacheView:
        return self._cache_view("self_kv", layer)

    def cross_cache(self, layer: int) -> MoonshineCacheView:
        return self._cache_view("cross_kv", layer)

    def set_self_cache_length(self, length: int) -> None:
        if self.spec is None or length < 0 or length > self.spec.self_cache_capacity:
            raise ValueError("self cache length must be in 0..capacity")
        self.self_cache_length = int(length)

    def mark_cross_cache_ready(self, encoder_frames: int) -> None:
        if int(encoder_frames) != self.encoder_frames:
            raise ValueError(
                f"encoder_frames {encoder_frames} does not match resident bucket {self.encoder_frames}"
            )
        self.cross_cache_valid = True

    def _zero(self, names: tuple[str, ...]) -> None:
        for name in names:
            allocation = self.workspace.allocation(name)
            self.runtime.memset_async(
                allocation.buffer.ptr,
                0,
                allocation.buffer.nbytes,
                self.stream,
            )

    def reset_generation(self, *, clear_cross_cache: bool = False) -> None:
        """Reset generation state without changing any allocation or address."""

        if self.closed:
            raise RuntimeError("Moonshine runtime is closed")
        names = ("self_kv", *self._SCRATCH_NAMES)
        if clear_cross_cache:
            names = (
                *names,
                "cross_kv",
                "encoder_hidden",
                "encoder_attention_mask",
            )
        self._zero(names)
        self.runtime.stream_synchronize(self.stream)
        self.self_cache_length = 0
        self.decode_position = None
        if clear_cross_cache:
            self.cross_cache_valid = False
            self.encoder_state_valid = False

    @contextmanager
    def no_allocation_region(self, name: str) -> Iterator[None]:
        """Fail if a region allocates even when it frees before returning."""

        if not name:
            raise ValueError("allocation region name must be non-empty")
        before = memory_stats()
        try:
            yield
        finally:
            after = memory_stats()
            allocated = after["total_allocated_bytes"] - before["total_allocated_bytes"]
            current_delta = after["current_allocated_bytes"] - before["current_allocated_bytes"]
            active_delta = after["active_allocations"] - before["active_allocations"]
            if allocated or current_delta or active_delta:
                raise NoAllocationError(
                    f"region {name!r} allocated {allocated} bytes "
                    f"(current_delta={current_delta}, active_delta={active_delta})"
                )

    def prepare_decoder_kernels(
        self,
        *,
        libraries: MoonshineDecoderLibraries | None = None,
        compiler_version: str | None = None,
        require_cached: bool = False,
    ) -> MoonshineDecoderLibraries:
        """Load every code object before any no-allocation/timed token region."""

        if self.closed:
            raise RuntimeError("Moonshine runtime is closed")
        if libraries is None:
            from hipengine.kernels.hip_gfx1100.attention.moonshine_attention import (
                build_moonshine_attention,
            )
            from hipengine.kernels.hip_gfx1100.fused.moonshine_glue import (
                build_moonshine_glue,
            )
            from hipengine.kernels.hip_gfx1100.fused.moonshine_mlp import (
                build_moonshine_mlp,
            )
            from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
                build_dense_gemv,
            )
            from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
                build_moonshine_projection,
            )
            from hipengine.kernels.hip_gfx1100.norm.moonshine_layernorm import (
                build_moonshine_layernorm,
            )

            arguments = {
                "compiler_version": compiler_version,
                "load": True,
                "require_cached": require_cached,
            }
            libraries = MoonshineDecoderLibraries(
                projection=build_moonshine_projection(**arguments),
                dense_projection=build_dense_gemv(**arguments),
                layernorm=build_moonshine_layernorm(**arguments),
                glue=build_moonshine_glue(**arguments),
                mlp=build_moonshine_mlp(**arguments),
                attention=build_moonshine_attention(**arguments),
            )
        if self._token_graphs:
            self._close_token_graphs()
        self.decoder_libraries = libraries
        return libraries

    def set_encoder_state(
        self,
        encoder_hidden: np.ndarray,
        attention_mask: np.ndarray,
    ) -> None:
        """Upload one certified encoder bucket and invalidate stale cross K/V."""

        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine runtime is closed")
        if self.self_cache_length or self.decode_position is not None:
            raise RuntimeError("reset generation before replacing encoder state")
        hidden = np.asarray(encoder_hidden)
        mask = np.asarray(attention_mask)
        expected_hidden = (1, self.encoder_frames, self.spec.hidden_size)
        expected_mask = (1, self.encoder_frames)
        if hidden.shape != expected_hidden or hidden.dtype != np.float16:
            raise ValueError(
                f"encoder hidden must be float16 with shape {expected_hidden}"
            )
        if mask.shape != expected_mask or mask.dtype != np.int32:
            raise ValueError(f"encoder attention mask must be int32 with shape {expected_mask}")
        if not bool(np.isfinite(hidden).all()):
            raise ValueError("encoder hidden must contain only finite values")
        if not bool(np.isin(mask, (0, 1)).all()) or not bool(mask.any()):
            raise ValueError("encoder attention mask must contain visible 0/1 entries")
        copy_host_to_device(
            self.workspace.allocation("encoder_hidden").buffer,
            host_array_ptr(np.ascontiguousarray(hidden)),
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.workspace.allocation("encoder_attention_mask").buffer,
            host_array_ptr(np.ascontiguousarray(mask)),
            runtime=self.runtime,
        )
        self.encoder_state_valid = True
        self.cross_cache_valid = False

    def precompute_cross_kv(self) -> None:
        """Project encoder rows once into all eight resident head-major caches."""

        if self.closed or self.spec is None or self.weights is None:
            raise RuntimeError("Moonshine runtime is closed")
        libraries = self.decoder_libraries
        if libraries is None:
            raise RuntimeError("Moonshine decoder kernels are not prepared")
        if not self.encoder_state_valid:
            raise RuntimeError("Moonshine encoder state is not loaded")
        from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
            moonshine_f16_projection_pair_head_major,
        )

        encoder_ptr = self.tensor("encoder_hidden").ptr
        for layer in range(self.spec.decoder_layers):
            prefix = f"model.decoder.layers.{layer}.encoder_attn"
            cache = self.cross_cache(layer)
            moonshine_f16_projection_pair_head_major(
                encoder_ptr,
                self.weights[f"{prefix}.k_proj.weight"].ptr,
                self.weights[f"{prefix}.v_proj.weight"].ptr,
                cache.key.ptr,
                cache.value.ptr,
                self.encoder_frames,
                self.spec.hidden_size,
                self.spec.decoder_kv_heads * self.spec.head_dim,
                self.spec.decoder_kv_heads * self.spec.head_dim,
                self.spec.head_dim,
                threads=MOONSHINE_CROSS_KV_THREADS,
                stream=self.stream,
                library=libraries.projection,
                runtime=self.runtime,
            )
        self.runtime.stream_synchronize(self.stream)
        self.mark_cross_cache_ready(self.encoder_frames)

    def set_decode_state(self, *, token_id: int, position: int) -> None:
        """Set the next device token/position under strict sequential-cache ownership."""

        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine runtime is closed")
        if self.decoder_libraries is None:
            raise RuntimeError("Moonshine decoder kernels are not prepared")
        if not self.cross_cache_valid or not self.encoder_state_valid:
            raise RuntimeError("Moonshine cross cache is not ready")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError("token_id must be an integer")
        if token_id < 0 or token_id >= self.spec.vocab_size:
            raise ValueError("token_id is outside the Moonshine vocabulary")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("position must be an integer")
        if position != self.self_cache_length:
            raise ValueError(
                f"decode positions must be sequential: expected {self.self_cache_length}, "
                f"got {position}"
            )
        if position < 0 or position >= self.spec.self_cache_capacity:
            raise ValueError("decode position is outside self-cache capacity")
        token = np.asarray([token_id], dtype=np.int64)
        position_array = np.asarray([position], dtype=np.int64)
        copy_host_to_device(
            self.workspace.allocation("token").buffer,
            host_array_ptr(token),
            runtime=self.runtime,
        )
        copy_host_to_device(
            self.workspace.allocation("position").buffer,
            host_array_ptr(position_array),
            runtime=self.runtime,
        )
        self.decode_position = position

    def _require_token_step_ready(self) -> int:
        if self.closed or self.spec is None or self.weights is None:
            raise RuntimeError("Moonshine runtime is closed")
        if self.decoder_libraries is None:
            raise RuntimeError("Moonshine decoder kernels are not prepared")
        if not self.cross_cache_valid or not self.encoder_state_valid:
            raise RuntimeError("Moonshine cross cache is not ready")
        if self.decode_position != self.self_cache_length:
            raise RuntimeError("Moonshine device token/position state is not set")
        return int(self.decode_position)

    def token_step(
        self,
        *,
        boundary_callback: Callable[[str, Tensor], None] | None = None,
    ) -> None:
        """Enqueue one complete FP16 decoder step through eager Python dispatch."""

        route_position = self._require_token_step_ready()
        self._enqueue_token_step(
            route_position=route_position,
            stream=self.stream,
            boundary_callback=boundary_callback,
        )
        self.self_cache_length += 1
        self.decode_position = None

    def _enqueue_token_step(
        self,
        *,
        route_position: int,
        stream: int,
        boundary_callback: Callable[[str, Tensor], None] | None = None,
    ) -> None:
        """Enqueue the fixed-address token DAG without changing host-owned state."""

        if self.closed or self.spec is None or self.weights is None:
            raise RuntimeError("Moonshine runtime is closed")
        libraries = self.decoder_libraries
        if libraries is None:
            raise RuntimeError("Moonshine decoder kernels are not prepared")
        if not self.cross_cache_valid or not self.encoder_state_valid:
            raise RuntimeError("Moonshine cross cache is not ready")
        _moonshine_token_graph_bucket(
            route_position,
            capacity=self.spec.self_cache_capacity,
        )
        from hipengine.kernels.hip_gfx1100.attention.moonshine_attention import (
            moonshine_cross_attention_parallel_fp16,
            moonshine_self_attention_fp16,
            moonshine_self_attention_parallel_fp16,
        )
        from hipengine.kernels.hip_gfx1100.fused.moonshine_glue import (
            moonshine_argmax_fp16,
            moonshine_embedding_lookup_fp16,
            moonshine_partial_rope_cache_append_fp16,
        )
        from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
            dense_gemv_out_fp16,
        )
        from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
            moonshine_f16_lm_head_projection_wave8,
            moonshine_f16_projection_bias_gated_silu,
            moonshine_f16_projection_bias_residual,
            moonshine_f16_projection_triple,
        )
        from hipengine.kernels.hip_gfx1100.norm.moonshine_layernorm import (
            moonshine_layernorm_fp16,
            moonshine_residual_layernorm_fp16,
        )

        spec = self.spec
        hidden = self.tensor("hidden")
        normalized = self.tensor("normalized")
        query = self.tensor("query")
        key = self.tensor("key")
        value = self.tensor("value")
        attention = self.tensor("attention")
        projection = self.tensor("projection")
        position = self.tensor("position")
        common = {"stream": stream, "runtime": self.runtime}
        moonshine_embedding_lookup_fp16(
            self.weights[spec.embedding_weight_name].ptr,
            self.tensor("token").ptr,
            hidden.ptr,
            spec.hidden_size,
            spec.vocab_size,
            library=libraries.glue,
            **common,
        )
        for layer in range(spec.decoder_layers):
            prefix = f"model.decoder.layers.{layer}"
            moonshine_layernorm_fp16(
                hidden.ptr,
                self.weights[f"{prefix}.input_layernorm.weight"].ptr,
                normalized.ptr,
                1,
                spec.hidden_size,
                eps=spec.layer_norm_epsilon,
                library=libraries.layernorm,
                **common,
            )
            moonshine_f16_projection_triple(
                normalized.ptr,
                self.weights[f"{prefix}.self_attn.q_proj.weight"].ptr,
                self.weights[f"{prefix}.self_attn.k_proj.weight"].ptr,
                self.weights[f"{prefix}.self_attn.v_proj.weight"].ptr,
                query.ptr,
                key.ptr,
                value.ptr,
                1,
                spec.hidden_size,
                spec.hidden_size,
                spec.hidden_size,
                spec.hidden_size,
                threads=MOONSHINE_TRIPLE_QKV_THREADS,
                library=libraries.projection,
                **common,
            )
            self_cache = self.self_cache(layer)
            moonshine_partial_rope_cache_append_fp16(
                query.ptr,
                key.ptr,
                value.ptr,
                self.tensor("rope_cos").ptr,
                self.tensor("rope_sin").ptr,
                position.ptr,
                attention.ptr,
                projection.ptr,
                self_cache.key.ptr,
                self_cache.value.ptr,
                spec.decoder_attention_heads,
                spec.head_dim,
                spec.rotary_dim,
                spec.self_cache_capacity,
                spec.max_positions,
                library=libraries.glue,
                **common,
            )
            self_attention = (
                moonshine_self_attention_fp16
                if route_position == 0
                else moonshine_self_attention_parallel_fp16
            )
            self_attention_options = (
                {}
                if route_position == 0
                else {"threads": _moonshine_self_attention_threads(route_position)}
            )
            self_attention(
                attention.ptr,
                self_cache.key.ptr,
                self_cache.value.ptr,
                position.ptr,
                attention.ptr,
                spec.decoder_attention_heads,
                spec.head_dim,
                spec.self_cache_capacity,
                library=libraries.attention,
                **self_attention_options,
                **common,
            )
            dense_gemv_out_fp16(
                attention.ptr,
                self.weights[f"{prefix}.self_attn.o_proj.weight"].ptr,
                projection.ptr,
                1,
                spec.hidden_size,
                spec.hidden_size,
                threads=MOONSHINE_SINGLE_PROJECTION_THREADS,
                library=libraries.dense_projection,
                **common,
            )
            moonshine_residual_layernorm_fp16(
                hidden.ptr,
                projection.ptr,
                self.weights[f"{prefix}.post_attention_layernorm.weight"].ptr,
                hidden.ptr,
                normalized.ptr,
                1,
                spec.hidden_size,
                eps=spec.layer_norm_epsilon,
                library=libraries.layernorm,
                **common,
            )
            if boundary_callback is not None:
                boundary_callback(f"layer_{layer}.after_self_attention", hidden)
            dense_gemv_out_fp16(
                normalized.ptr,
                self.weights[f"{prefix}.encoder_attn.q_proj.weight"].ptr,
                query.ptr,
                1,
                spec.hidden_size,
                spec.hidden_size,
                threads=MOONSHINE_SINGLE_PROJECTION_THREADS,
                library=libraries.dense_projection,
                **common,
            )
            cross_cache = self.cross_cache(layer)
            moonshine_cross_attention_parallel_fp16(
                query.ptr,
                cross_cache.key.ptr,
                cross_cache.value.ptr,
                self.tensor("encoder_attention_mask").ptr,
                attention.ptr,
                spec.decoder_attention_heads,
                spec.head_dim,
                self.encoder_frames,
                threads=MOONSHINE_CROSS_ATTENTION_THREADS,
                library=libraries.attention,
                **common,
            )
            dense_gemv_out_fp16(
                attention.ptr,
                self.weights[f"{prefix}.encoder_attn.o_proj.weight"].ptr,
                projection.ptr,
                1,
                spec.hidden_size,
                spec.hidden_size,
                threads=MOONSHINE_SINGLE_PROJECTION_THREADS,
                library=libraries.dense_projection,
                **common,
            )
            moonshine_residual_layernorm_fp16(
                hidden.ptr,
                projection.ptr,
                self.weights[f"{prefix}.final_layernorm.weight"].ptr,
                hidden.ptr,
                normalized.ptr,
                1,
                spec.hidden_size,
                eps=spec.layer_norm_epsilon,
                library=libraries.layernorm,
                **common,
            )
            if boundary_callback is not None:
                boundary_callback(f"layer_{layer}.after_cross_attention", hidden)
            moonshine_f16_projection_bias_gated_silu(
                normalized.ptr,
                self.weights[f"{prefix}.mlp.fc1.weight"].ptr,
                self.weights[f"{prefix}.mlp.fc1.bias"].ptr,
                self.tensor("mlp_intermediate").ptr,
                1,
                spec.hidden_size,
                spec.intermediate_size,
                threads=MOONSHINE_MLP_FC1_THREADS,
                library=libraries.projection,
                **common,
            )
            moonshine_f16_projection_bias_residual(
                self.tensor("mlp_intermediate").ptr,
                self.weights[f"{prefix}.mlp.fc2.weight"].ptr,
                self.weights[f"{prefix}.mlp.fc2.bias"].ptr,
                hidden.ptr,
                hidden.ptr,
                1,
                spec.intermediate_size,
                spec.hidden_size,
                threads=MOONSHINE_MLP_FC2_THREADS,
                library=libraries.projection,
                **common,
            )
            if boundary_callback is not None:
                boundary_callback(f"layer_{layer}.after_mlp", hidden)
        moonshine_layernorm_fp16(
            hidden.ptr,
            self.weights["model.decoder.norm.weight"].ptr,
            normalized.ptr,
            1,
            spec.hidden_size,
            eps=spec.layer_norm_epsilon,
            library=libraries.layernorm,
            **common,
        )
        if boundary_callback is not None:
            boundary_callback("final_hidden", normalized)
        moonshine_f16_lm_head_projection_wave8(
            normalized.ptr,
            self.weights[spec.lm_head_alias_name].ptr,
            self.tensor("logits").ptr,
            1,
            spec.hidden_size,
            spec.vocab_size,
            library=libraries.projection,
            **common,
        )
        moonshine_argmax_fp16(
            self.tensor("logits").ptr,
            self.tensor("token").ptr,
            spec.vocab_size,
            library=libraries.glue,
            **common,
        )

    def capture_token_graphs(self) -> tuple[MoonshineTokenGraph, ...]:
        """Capture one reusable token DAG for each general self-cache launch bucket."""

        if self.closed or self.spec is None or self.weights is None:
            raise RuntimeError("Moonshine runtime is closed")
        if self.decoder_libraries is None:
            raise RuntimeError("Moonshine decoder kernels are not prepared")
        if not self.cross_cache_valid or not self.encoder_state_valid:
            raise RuntimeError("Moonshine cross cache is not ready")
        if self.decode_position is not None:
            raise RuntimeError("capture token graphs before setting device decode state")
        if self._token_graphs:
            return tuple(self._token_graphs.values())

        representatives = tuple(
            position
            for position in (0, 1, 2, 4)
            if position < self.spec.self_cache_capacity
        )
        captures: list[MoonshineTokenGraph] = []
        try:
            for position in representatives:
                bucket, minimum, maximum = _moonshine_token_graph_bucket(
                    position,
                    capacity=self.spec.self_cache_capacity,
                )
                self.runtime.stream_synchronize(self.stream)
                graph = 0
                capture_start = time.perf_counter_ns()
                self.runtime.stream_begin_capture(self.stream)
                try:
                    self._enqueue_token_step(
                        route_position=position,
                        stream=self.stream,
                    )
                    graph = self.runtime.stream_end_capture(self.stream)
                except Exception:
                    try:
                        self.runtime.stream_end_capture(self.stream)
                    except Exception:
                        pass
                    raise
                capture_wall_ms = (time.perf_counter_ns() - capture_start) / 1.0e6
                instantiate_start = time.perf_counter_ns()
                try:
                    graph_exec = self.runtime.graph_instantiate(graph)
                except Exception:
                    self.runtime.graph_destroy(graph)
                    raise
                instantiate_wall_ms = (time.perf_counter_ns() - instantiate_start) / 1.0e6
                captures.append(
                    MoonshineTokenGraph(
                        owner=self,
                        bucket=bucket,
                        minimum_position=minimum,
                        maximum_position=maximum,
                        capture_position=position,
                        graph=graph,
                        graph_exec=graph_exec,
                        capture_wall_ms=float(capture_wall_ms),
                        instantiate_wall_ms=float(instantiate_wall_ms),
                    )
                )
        except Exception:
            for capture in reversed(captures):
                capture.close()
            raise
        self._token_graphs = {capture.bucket: capture for capture in captures}
        return tuple(captures)

    def graph_token_step(self) -> None:
        """Launch the captured token DAG selected by the current cache position."""

        route_position = self._require_token_step_ready()
        assert self.spec is not None
        bucket, _, _ = _moonshine_token_graph_bucket(
            route_position,
            capacity=self.spec.self_cache_capacity,
        )
        capture = self._token_graphs.get(bucket)
        if capture is None or capture.closed:
            raise RuntimeError(f"Moonshine token graph bucket {bucket!r} is not captured")
        if not capture.accepts(route_position):
            raise RuntimeError("Moonshine token graph bucket does not cover decode position")
        self.runtime.graph_launch(capture.graph_exec, self.stream)
        capture.replay_count += 1
        self.self_cache_length += 1
        self.decode_position = None

    def token_graph_contract(self) -> dict[str, object]:
        captures = tuple(self._token_graphs.values())
        return {
            "captured": bool(captures),
            "graph_count": len(captures),
            "buckets": [capture.bucket for capture in captures],
            "capture_positions": [capture.capture_position for capture in captures],
            "capture_wall_ms": sum(capture.capture_wall_ms for capture in captures),
            "instantiate_wall_ms": sum(capture.instantiate_wall_ms for capture in captures),
            "replay_count": sum(capture.replay_count for capture in captures),
        }

    def _close_token_graphs(self) -> None:
        for capture in reversed(tuple(self._token_graphs.values())):
            capture.close()
        self._token_graphs.clear()

    def read_token(self) -> int:
        """Synchronize the resident stream and read the selected int64 token."""

        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine runtime is closed")
        self.runtime.stream_synchronize(self.stream)
        output = np.empty((1,), dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(output),
            self.workspace.allocation("token").buffer,
            runtime=self.runtime,
        )
        token_id = int(output[0])
        if token_id < 0 or token_id >= self.spec.vocab_size:
            raise RuntimeError(f"Moonshine decoder returned invalid token ID {token_id}")
        return token_id

    def allocation_contract(self) -> dict[str, int | bool | None]:
        current = memory_stats()
        baseline_bytes = (
            0 if self.loaded_model is None else self.loaded_model.baseline_allocated_bytes
        )
        baseline_allocations = (
            0 if self.loaded_model is None else self.loaded_model.baseline_active_allocations
        )
        return {
            "baseline_allocated_bytes": baseline_bytes,
            "baseline_active_allocations": baseline_allocations,
            "resident_nbytes": self.resident_nbytes,
            "current_allocated_bytes": current["current_allocated_bytes"],
            "current_active_allocations": current["active_allocations"],
            "decoder_kernels_prepared": self.decoder_libraries is not None,
            "encoder_state_valid": self.encoder_state_valid,
            "cross_cache_valid": self.cross_cache_valid,
            "teardown_returned_to_baseline": self.teardown_returned_to_baseline,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._close_token_graphs()
        self.decoder_libraries = None
        loaded = self.loaded_model
        try:
            if self.stream:
                self.runtime.stream_synchronize(self.stream)
        finally:
            try:
                self.workspace.free()
            finally:
                if loaded is not None:
                    loaded.free(runtime=self.runtime)
                for event in (self.stop_event, self.start_event):
                    if event:
                        self.runtime.event_destroy(event)
                if self.stream:
                    self.runtime.stream_destroy(self.stream)
                self.stop_event = 0
                self.start_event = 0
                self.stream = 0
                if loaded is not None:
                    current = memory_stats()
                    self.teardown_returned_to_baseline = bool(
                        current["current_allocated_bytes"]
                        == loaded.baseline_allocated_bytes
                        and current["active_allocations"]
                        == loaded.baseline_active_allocations
                    )

    def __enter__(self) -> "MoonshineResidentRuntime":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


__all__ = [
    "MoonshineCacheView",
    "MoonshineDecoderLibraries",
    "MoonshineResidentRuntime",
    "MoonshineTokenGraph",
    "NoAllocationError",
]
