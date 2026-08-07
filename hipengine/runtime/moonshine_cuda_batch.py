"""Torch-free fixed-address static-batch Moonshine resident decoder (C8 phase 1).

``MoonshineCudaBatchRuntime`` processes ``max_batch`` requests in lockstep on
one stream with batch-strided state/caches and zero timed allocation.  It is
the c=N counterpart of ``MoonshineCudaResidentRuntime``: every dense kernel is
reused at ``rows=B``, attention/glue/LM-head use the static-B batch kernels,
and the token stream is published per row so the host can stop rows at EOS.

Exact-equality contract: each row of the batch must be **bit-exact** to B
independent c=1 sessions on the same cross caches and seed tokens.  Because the
one-wave t32 and parallel t256 self-attention variants are *not* bit-identical
(they diverge in the last FP16 ULP on some long-position inputs), the batch
runs in lockstep: all rows share one route position each step, and the
attention schedule is chosen from that shared position exactly as a c=1 session
would.  Per-row positions are supported by the batch kernels, but mixing
positions across the visible-8 boundary in one launch requires an explicit
``threads=`` override (advanced use; not the phase-1 static model).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

from hipengine.core.cuda import CudaRuntime, get_cuda_runtime
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import (
    MemcpyKind,
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
    memory_stats,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference.moonshine import moonshine_rope_tables
from hipengine.loading.moonshine import MoonshineLoadedModel, load_moonshine_model
from hipengine.runtime.moonshine_cuda import (
    MoonshineCudaDecoderLibraries,
    NoAllocationError,
    _moonshine_cuda_token_graph_bucket,
    _self_attention_threads,
)
from hipengine.runtime.workspace import RuntimeWorkspace

# C1e: cross attention is flat parallel t256 at every production encoder length.
_CROSS_THREADS = 256
# C1f fused LM-head rows-per-block measured bucket.
_LM_HEAD_ROWS_PER_BLOCK = 8


@dataclass(frozen=True)
class MoonshineCudaBatchCacheView:
    key: Tensor
    value: Tensor


@dataclass
class MoonshineCudaBatchTokenGraph:
    """One captured fixed-address batch token DAG for a decode-position bucket.

    Keyed by the self-attention position topology (t32 below 8 visible tokens,
    parallel t256 at/above) within this runtime's fixed ``(max_batch,
    encoder_frames)`` bucket: a captured graph is position-generic inside its
    bucket because every kernel reads the device position tensor at launch.
    """

    owner: "MoonshineCudaBatchRuntime"
    bucket: str
    minimum_position: int
    maximum_position: int
    capture_position: int
    threads: int
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


class MoonshineCudaBatchRuntime:
    """Own every fixed-address object needed by a static-B CUDA Moonshine decoder."""

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
        "token",
        "position",
        "argmax",
        "lm_head_values",
        "lm_head_indices",
        "lm_head_out_value",
    )

    def __init__(
        self,
        *,
        max_batch: int,
        encoder_frames: int,
        model_path: str | Path | None = None,
        loaded_model: MoonshineLoadedModel | None = None,
        device: Device | None = None,
        runtime: CudaRuntime | None = None,
        owns_weights: bool = True,
    ) -> None:
        if (model_path is None) == (loaded_model is None):
            raise ValueError("provide exactly one of model_path or loaded_model")
        if isinstance(max_batch, bool) or not isinstance(max_batch, int):
            raise ValueError("max_batch must be a positive integer")
        if max_batch <= 0:
            raise ValueError("max_batch must be a positive integer")
        self.runtime = runtime or get_cuda_runtime()
        self.device = device or Device("cuda", 0)
        self.loaded_model = loaded_model
        self.weights = loaded_model.weights if loaded_model is not None else None
        self.spec = loaded_model.spec if loaded_model is not None else None
        self.owns_weights = bool(owns_weights)
        self.max_batch = int(max_batch)
        self.encoder_frames = int(encoder_frames)
        self.workspace = RuntimeWorkspace(device=self.device, runtime=self.runtime)
        self.stream = 0
        self.self_cache_length = 0
        self.cross_cache_valid = False
        self.encoder_state_valid = False
        self.decode_position: int | None = None
        self._device_owned_decode = False
        self.decoder_libraries: MoonshineCudaDecoderLibraries | None = None
        self._token_graphs: dict[str, MoonshineCudaBatchTokenGraph] = {}
        self.closed = False
        self.teardown_returned_to_baseline: bool | None = None
        self._allocation_baseline = memory_stats()["current_allocated_bytes"]
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
            if self.encoder_frames <= 0:
                raise ValueError("encoder_frames must be positive")
            self.stream = self.runtime.stream_create(nonblocking=True)
            self._reserve_workspace()
            self._initialize_workspace()
        except Exception:
            self.close()
            raise

    def _reserve_workspace(self) -> None:
        assert self.spec is not None
        spec = self.spec
        batch = self.max_batch
        reserve = self.workspace.reserve_tensor
        reserve("rope_cos", (spec.max_positions, spec.rotary_dim // 2), DType.FP16)
        reserve("rope_sin", (spec.max_positions, spec.rotary_dim // 2), DType.FP16)
        reserve(
            "self_kv",
            (
                spec.decoder_layers,
                2,
                batch,
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
                batch,
                spec.decoder_kv_heads,
                self.encoder_frames,
                spec.head_dim,
            ),
            DType.FP16,
        )
        reserve("encoder_attention_mask", (batch, self.encoder_frames), DType.INT32)
        reserve("hidden", (batch, 1, spec.hidden_size), DType.FP16)
        reserve("residual", (batch, 1, spec.hidden_size), DType.FP16)
        reserve("normalized", (batch, 1, spec.hidden_size), DType.FP16)
        for name in ("query", "key", "value", "attention"):
            reserve(
                name,
                (batch, spec.decoder_attention_heads, 1, spec.head_dim),
                DType.FP16,
            )
        reserve("projection", (batch, 1, spec.hidden_size), DType.FP16)
        reserve("mlp_fc1", (batch, 1, 2 * spec.intermediate_size), DType.FP16)
        reserve("mlp_intermediate", (batch, 1, spec.intermediate_size), DType.FP16)
        reserve("token", (batch,), DType.INT64)
        reserve("position", (batch,), DType.INT64)
        reserve("argmax", (batch,), DType.INT64)
        from hipengine.kernels.cuda_sm120a.linear.lm_head import (
            lm_head_argmax_scratch_elements,
        )

        num_blocks = lm_head_argmax_scratch_elements(
            spec.vocab_size, _LM_HEAD_ROWS_PER_BLOCK
        )
        reserve("lm_head_values", (batch, num_blocks), DType.FP32)
        reserve("lm_head_indices", (batch, num_blocks), DType.INT64)
        reserve("lm_head_out_value", (batch,), DType.FP32)

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

    def prepare_decoder_kernels(
        self,
        *,
        libraries: MoonshineCudaDecoderLibraries | None = None,
        compiler_version: str | None = None,
        require_cached: bool = False,
    ) -> MoonshineCudaDecoderLibraries:
        """Load every code object before any no-allocation/timed token region."""

        if self.closed:
            raise RuntimeError("Moonshine batch runtime is closed")
        if libraries is None:
            from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
                build_moonshine_attention,
            )
            from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
                build_moonshine_glue,
            )
            from hipengine.kernels.cuda_sm120a.linear.lm_head import (
                build_moonshine_lm_head,
            )
            from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
                build_moonshine_projection,
            )
            from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
                build_moonshine_layernorm,
            )

            arguments = {
                "compiler_version": compiler_version,
                "load": True,
                "require_cached": require_cached,
            }
            libraries = MoonshineCudaDecoderLibraries(
                glue=build_moonshine_glue(**arguments),
                layernorm=build_moonshine_layernorm(**arguments),
                projection=build_moonshine_projection(**arguments),
                attention=build_moonshine_attention(**arguments),
                lm_head=build_moonshine_lm_head(**arguments),
            )
        self.decoder_libraries = libraries
        return libraries

    def tensor(self, name: str) -> Tensor:
        return self.workspace.allocation(name).tensor

    @property
    def resident_nbytes(self) -> int:
        if self.loaded_model is None:
            return 0
        loaded = self.loaded_model
        owned = getattr(loaded, "owned_weight_bytes", None)
        if owned is not None:
            return owned
        weights = loaded.weights
        resident = getattr(weights, "resident_bytes", None)
        if resident is not None:
            return resident
        return sum(
            allocation.buffer.nbytes
            for allocation in weights.tensors.values()
            if allocation.owns_buffer
        )

    def _batch_cache_view(
        self, name: str, layer: int
    ) -> MoonshineCudaBatchCacheView:
        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine batch runtime is closed")
        if layer < 0 or layer >= self.spec.decoder_layers:
            raise ValueError("cache layer is outside the decoder")
        allocation = self.workspace.allocation(name)
        capacity = (
            self.spec.self_cache_capacity
            if name == "self_kv"
            else self.encoder_frames
        )
        shape = (self.max_batch, self.spec.decoder_kv_heads, capacity, self.spec.head_dim)
        elements = (
            self.max_batch * self.spec.decoder_kv_heads * capacity * self.spec.head_dim
        )
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

        return MoonshineCudaBatchCacheView(key=view(0), value=view(1))

    def self_cache(self, layer: int) -> MoonshineCudaBatchCacheView:
        return self._batch_cache_view("self_kv", layer)

    def cross_cache(self, layer: int) -> MoonshineCudaBatchCacheView:
        return self._batch_cache_view("cross_kv", layer)

    def load_cross_cache_batch(
        self,
        keys: Sequence[np.ndarray],
        values: Sequence[np.ndarray],
        *,
        masks: np.ndarray | None = None,
    ) -> None:
        """Copy per-row head-major cross K/V into the batch-strided caches.

        Each layer's array is ``[B, kv_heads, frames, head_dim]`` FP16.  A
        batch ``[B, frames]`` int32 mask is optional (omitted means all frames
        valid).  Must be called before any token step; resets generation state.
        """
        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine batch runtime is closed")
        spec = self.spec
        if len(keys) != spec.decoder_layers or len(values) != spec.decoder_layers:
            raise ValueError(
                f"cross cache needs {spec.decoder_layers} layers, "
                f"got {len(keys)} keys / {len(values)} values"
            )
        expected = (
            self.max_batch,
            spec.decoder_kv_heads,
            self.encoder_frames,
            spec.head_dim,
        )
        for layer in range(spec.decoder_layers):
            key = np.ascontiguousarray(keys[layer], dtype=np.float16)
            value = np.ascontiguousarray(values[layer], dtype=np.float16)
            if key.shape != expected or value.shape != expected:
                raise ValueError(
                    f"layer {layer} batch cross cache shape {key.shape} != {expected}"
                )
            view = self.cross_cache(layer)
            self.runtime.memcpy(
                view.key.ptr,
                host_array_ptr(key),
                key.nbytes,
                MemcpyKind.HOST_TO_DEVICE,
            )
            self.runtime.memcpy(
                view.value.ptr,
                host_array_ptr(value),
                value.nbytes,
                MemcpyKind.HOST_TO_DEVICE,
            )
        if masks is not None:
            flat = np.ascontiguousarray(masks, dtype=np.int32)
            if flat.shape != (self.max_batch, self.encoder_frames):
                raise ValueError(
                    f"batch mask shape {flat.shape} != "
                    f"{(self.max_batch, self.encoder_frames)}"
                )
            copy_host_to_device(
                self.workspace.allocation("encoder_attention_mask").buffer,
                host_array_ptr(flat),
                runtime=self.runtime,
            )
        else:
            ones = np.ones(
                (self.max_batch, self.encoder_frames), dtype=np.int32
            )
            copy_host_to_device(
                self.workspace.allocation("encoder_attention_mask").buffer,
                host_array_ptr(ones),
                runtime=self.runtime,
            )
        self.runtime.stream_synchronize(self.stream)
        self.cross_cache_valid = True
        self.encoder_state_valid = True
        self.reset_generation(clear_cross_cache=False)

    def reset_generation(self, *, clear_cross_cache: bool = False) -> None:
        """Reset generation state without changing any allocation or address."""

        if self.closed:
            raise RuntimeError("Moonshine batch runtime is closed")
        names = ("self_kv", *self._SCRATCH_NAMES)
        if clear_cross_cache:
            names = (*names, "cross_kv", "encoder_attention_mask")
        for name in names:
            allocation = self.workspace.allocation(name)
            self.runtime.memset_async(
                allocation.buffer.ptr,
                0,
                allocation.buffer.nbytes,
                self.stream,
            )
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

    def set_batch_decode_state(self, *, tokens: Sequence[int], position: int) -> None:
        """Set the next device token/position for all rows at one lockstep position."""

        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine batch runtime is closed")
        if self.decoder_libraries is None:
            raise RuntimeError("Moonshine batch decoder kernels are not prepared")
        if not self.cross_cache_valid or not self.encoder_state_valid:
            raise RuntimeError("Moonshine batch cross cache is not ready")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("position must be an integer")
        if position != self.self_cache_length:
            raise ValueError(
                f"decode positions must be sequential: expected {self.self_cache_length}, "
                f"got {position}"
            )
        if position < 0 or position >= self.spec.self_cache_capacity:
            raise ValueError("decode position is outside self-cache capacity")
        token_list = list(tokens)
        if len(token_list) != self.max_batch:
            raise ValueError(
                f"expected {self.max_batch} tokens, got {len(token_list)}"
            )
        for token_id in token_list:
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or token_id >= self.spec.vocab_size
            ):
                raise ValueError("token_id is outside the Moonshine vocabulary")
        self.runtime.stream_synchronize(self.stream)
        token = np.asarray(token_list, dtype=np.int64)
        position_array = np.full(self.max_batch, position, dtype=np.int64)
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

    def set_batch_device_owned_decode(self, enabled: bool = True) -> None:
        """Enable device-owned decode state for all rows (token/position on device)."""

        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine batch runtime is closed")
        self._device_owned_decode = bool(enabled)

    def set_batch_decode_seed(self, *, tokens: Sequence[int]) -> None:
        """Seed device token/position buffers for a device-owned batch decode run."""

        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine batch runtime is closed")
        if not self._device_owned_decode:
            raise RuntimeError("device-owned batch decode is not enabled")
        if not self.cross_cache_valid or not self.encoder_state_valid:
            raise RuntimeError("Moonshine batch cross cache is not ready")
        token_list = list(tokens)
        if len(token_list) != self.max_batch:
            raise ValueError(
                f"expected {self.max_batch} tokens, got {len(token_list)}"
            )
        for token_id in token_list:
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or token_id >= self.spec.vocab_size
            ):
                raise ValueError("token_id is outside the Moonshine vocabulary")
        if self.self_cache_length != 0:
            raise ValueError("set_batch_decode_seed requires a fresh generation")
        token = np.asarray(token_list, dtype=np.int64)
        position_array = np.zeros(self.max_batch, dtype=np.int64)
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
        self.runtime.stream_synchronize(self.stream)
        self.decode_position = None
        self.self_cache_length = 0

    def _require_token_step_ready(self) -> int:
        if self.closed or self.spec is None or self.weights is None:
            raise RuntimeError("Moonshine batch runtime is closed")
        if self.decoder_libraries is None:
            raise RuntimeError("Moonshine batch decoder kernels are not prepared")
        if not self.cross_cache_valid or not self.encoder_state_valid:
            raise RuntimeError("Moonshine batch cross cache is not ready")
        if self._device_owned_decode:
            if self.decode_position is not None:
                raise RuntimeError("device-owned batch decode position is not consumed")
            return int(self.self_cache_length)
        if self.decode_position != self.self_cache_length:
            raise RuntimeError("Moonshine batch device token/position state is not set")
        return int(self.decode_position)

    def batch_token_step(
        self,
        *,
        threads: int | None = None,
        boundary_callback: Callable[[str, Tensor], None] | None = None,
    ) -> None:
        """Enqueue one complete lockstep batch decoder step through eager dispatch."""

        route_position = self._require_token_step_ready()
        if threads is None:
            threads = _self_attention_threads(route_position + 1)
        with self.no_allocation_region("batch_token_step"):
            self._enqueue_batch_token_step(
                route_position=route_position,
                threads=threads,
                stream=self.stream,
                boundary_callback=boundary_callback,
            )
        self.self_cache_length += 1
        self.decode_position = None

    def _enqueue_batch_token_step(
        self,
        *,
        route_position: int,
        threads: int,
        stream: int,
        boundary_callback: Callable[[str, Tensor], None] | None = None,
    ) -> None:
        """Enqueue the fixed-address batch token DAG without host-owned state change."""

        if self.closed or self.spec is None or self.weights is None:
            raise RuntimeError("Moonshine batch runtime is closed")
        libraries = self.decoder_libraries
        if libraries is None:
            raise RuntimeError("Moonshine batch decoder kernels are not prepared")
        if not self.cross_cache_valid or not self.encoder_state_valid:
            raise RuntimeError("Moonshine batch cross cache is not ready")
        from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
            moonshine_cross_attention_parallel_batch_fp16,
            moonshine_self_attention_batch_fp16,
            moonshine_self_attention_parallel_batch_fp16,
        )
        from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
            moonshine_embedding_lookup_batch_fp16,
            moonshine_partial_rope_cache_append_batch_fp16,
        )
        from hipengine.kernels.cuda_sm120a.linear.lm_head import (
            moonshine_lm_head_argmax_batch_fp16,
        )
        from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
            moonshine_f16_projection,
            moonshine_f16_projection_bias_gated_silu,
            moonshine_f16_projection_bias_residual,
            moonshine_f16_projection_triple,
        )
        from hipengine.kernels.cuda_sm120a.norm.moonshine_layernorm import (
            moonshine_layernorm_fp16,
            moonshine_residual_layernorm_fp16,
        )

        spec = self.spec
        batch = self.max_batch
        hidden = self.tensor("hidden")
        normalized = self.tensor("normalized")
        query = self.tensor("query")
        key = self.tensor("key")
        value = self.tensor("value")
        attention = self.tensor("attention")
        projection = self.tensor("projection")
        position = self.tensor("position")
        common = {"stream": stream, "runtime": self.runtime}

        def attention_projection(
            input_ptr: int,
            weight_name: str,
            output_ptr: int,
        ) -> None:
            moonshine_f16_projection(
                input_ptr,
                self.weights[weight_name].ptr,
                output_ptr,
                batch,
                spec.hidden_size,
                spec.hidden_size,
                library=libraries.projection,
                **common,
            )

        moonshine_embedding_lookup_batch_fp16(
            self.weights[spec.embedding_weight_name].ptr,
            self.tensor("token").ptr,
            hidden.ptr,
            spec.hidden_size,
            spec.vocab_size,
            batch,
            library=libraries.glue,
            **common,
        )
        for layer in range(spec.decoder_layers):
            prefix = f"model.decoder.layers.{layer}"
            moonshine_layernorm_fp16(
                hidden.ptr,
                self.weights[f"{prefix}.input_layernorm.weight"].ptr,
                normalized.ptr,
                batch,
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
                batch,
                spec.hidden_size,
                spec.hidden_size,
                spec.hidden_size,
                spec.hidden_size,
                library=libraries.projection,
                **common,
            )
            self_cache = self.self_cache(layer)
            moonshine_partial_rope_cache_append_batch_fp16(
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
                batch,
                library=libraries.glue,
                **common,
            )
            if threads == 32:
                moonshine_self_attention_batch_fp16(
                    attention.ptr,
                    self_cache.key.ptr,
                    self_cache.value.ptr,
                    position.ptr,
                    attention.ptr,
                    spec.decoder_attention_heads,
                    spec.head_dim,
                    spec.self_cache_capacity,
                    threads=32,
                    batch=batch,
                    library=libraries.attention,
                    **common,
                )
            else:
                moonshine_self_attention_parallel_batch_fp16(
                    attention.ptr,
                    self_cache.key.ptr,
                    self_cache.value.ptr,
                    position.ptr,
                    attention.ptr,
                    spec.decoder_attention_heads,
                    spec.head_dim,
                    spec.self_cache_capacity,
                    threads=threads,
                    batch=batch,
                    library=libraries.attention,
                    **common,
                )
            attention_projection(
                attention.ptr,
                f"{prefix}.self_attn.o_proj.weight",
                projection.ptr,
            )
            moonshine_residual_layernorm_fp16(
                hidden.ptr,
                projection.ptr,
                self.weights[f"{prefix}.post_attention_layernorm.weight"].ptr,
                hidden.ptr,
                normalized.ptr,
                batch,
                spec.hidden_size,
                eps=spec.layer_norm_epsilon,
                library=libraries.layernorm,
                **common,
            )
            if boundary_callback is not None:
                boundary_callback(f"layer_{layer}.after_self_attention", hidden)
            attention_projection(
                normalized.ptr,
                f"{prefix}.encoder_attn.q_proj.weight",
                query.ptr,
            )
            cross_cache = self.cross_cache(layer)
            moonshine_cross_attention_parallel_batch_fp16(
                query.ptr,
                cross_cache.key.ptr,
                cross_cache.value.ptr,
                self.tensor("encoder_attention_mask").ptr,
                attention.ptr,
                spec.decoder_attention_heads,
                spec.head_dim,
                self.encoder_frames,
                threads=_CROSS_THREADS,
                batch=batch,
                library=libraries.attention,
                **common,
            )
            attention_projection(
                attention.ptr,
                f"{prefix}.encoder_attn.o_proj.weight",
                projection.ptr,
            )
            moonshine_residual_layernorm_fp16(
                hidden.ptr,
                projection.ptr,
                self.weights[f"{prefix}.final_layernorm.weight"].ptr,
                hidden.ptr,
                normalized.ptr,
                batch,
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
                batch,
                spec.hidden_size,
                spec.intermediate_size,
                library=libraries.projection,
                **common,
            )
            moonshine_f16_projection_bias_residual(
                self.tensor("mlp_intermediate").ptr,
                self.weights[f"{prefix}.mlp.fc2.weight"].ptr,
                self.weights[f"{prefix}.mlp.fc2.bias"].ptr,
                hidden.ptr,
                hidden.ptr,
                batch,
                spec.intermediate_size,
                spec.hidden_size,
                # C8 phase-1 exact-equality contract: the fused fc2 boundary
                # must run the same 256-thread per-row reduction schedule as a
                # c=1 session (the auto-default would select the 64-thread batch
                # schedule at rows>1 and diverge in the last FP16 ULP).  The
                # faster 64-thread schedule is an explicit performance override.
                threads=256,
                library=libraries.projection,
                **common,
            )
            if boundary_callback is not None:
                boundary_callback(f"layer_{layer}.after_mlp", hidden)
        moonshine_layernorm_fp16(
            hidden.ptr,
            self.weights["model.decoder.norm.weight"].ptr,
            normalized.ptr,
            batch,
            spec.hidden_size,
            eps=spec.layer_norm_epsilon,
            library=libraries.layernorm,
            **common,
        )
        if boundary_callback is not None:
            boundary_callback("final_hidden", normalized)
        moonshine_lm_head_argmax_batch_fp16(
            normalized.ptr,
            self.weights[spec.lm_head_alias_name].ptr,
            self.tensor("lm_head_values").ptr,
            self.tensor("lm_head_indices").ptr,
            self.tensor("token").ptr,
            self.tensor("lm_head_out_value").ptr,
            spec.hidden_size,
            spec.vocab_size,
            batch,
            rows_per_block=_LM_HEAD_ROWS_PER_BLOCK,
            library=libraries.lm_head,
            **common,
        )
        if self._device_owned_decode:
            from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
                moonshine_advance_position_batch_fp16,
            )

            moonshine_advance_position_batch_fp16(
                self.tensor("position").ptr,
                spec.self_cache_capacity,
                batch,
                stream=stream,
                library=libraries.glue,
                runtime=self.runtime,
            )

    def capture_batch_token_graphs(self) -> tuple[MoonshineCudaBatchTokenGraph, ...]:
        """Capture one reusable batch token DAG per self-attention bucket.

        Mirrors the c=1 runtime: one graph at route position 0 (visible 1,
        one-wave t32) and one at position 7 (visible 8, parallel t256).  Every
        kernel reads the fixed device position tensor at launch, so each graph
        is position-generic within its bucket and is replayed after
        ``set_batch_decode_state`` / ``set_batch_decode_seed`` update token
        and position under fixed addresses.  The graph identity is bound to
        this runtime's fixed ``(max_batch, encoder_frames)`` bucket.
        """

        import time

        if self.closed or self.spec is None or self.weights is None:
            raise RuntimeError("Moonshine batch runtime is closed")
        if self.decoder_libraries is None:
            raise RuntimeError("Moonshine batch decoder kernels are not prepared")
        if not self.cross_cache_valid or not self.encoder_state_valid:
            raise RuntimeError("Moonshine batch cross cache is not ready")
        if self.decode_position is not None:
            raise RuntimeError("capture batch token graphs before setting device state")
        if self._token_graphs:
            return tuple(self._token_graphs.values())

        representatives = tuple(
            position for position in (0, 7) if position < self.spec.self_cache_capacity
        )
        captures: list[MoonshineCudaBatchTokenGraph] = []
        try:
            for position in representatives:
                bucket, minimum, maximum = _moonshine_cuda_token_graph_bucket(
                    position,
                    capacity=self.spec.self_cache_capacity,
                )
                threads = _self_attention_threads(position + 1)
                self.runtime.stream_synchronize(self.stream)
                graph = 0
                capture_start = time.perf_counter_ns()
                self.runtime.stream_begin_capture(self.stream)
                try:
                    self._enqueue_batch_token_step(
                        route_position=position,
                        threads=threads,
                        stream=self.stream,
                    )
                    graph = self.runtime.stream_end_capture(self.stream)
                except Exception:
                    try:
                        leaked = self.runtime.stream_end_capture(self.stream)
                        if leaked:
                            self.runtime.graph_destroy(leaked)
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
                    MoonshineCudaBatchTokenGraph(
                        owner=self,
                        bucket=bucket,
                        minimum_position=minimum,
                        maximum_position=maximum,
                        capture_position=position,
                        threads=threads,
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

    def graph_batch_token_step(self) -> None:
        """Launch the captured batch token DAG for the current cache position."""

        route_position = self._require_token_step_ready()
        assert self.spec is not None
        bucket, _, _ = _moonshine_cuda_token_graph_bucket(
            route_position,
            capacity=self.spec.self_cache_capacity,
        )
        capture = self._token_graphs.get(bucket)
        if capture is None or capture.closed:
            raise RuntimeError(
                f"Moonshine batch token graph bucket {bucket!r} is not captured"
            )
        if not capture.accepts(route_position):
            raise RuntimeError(
                "Moonshine batch token graph bucket does not cover decode position"
            )
        self.runtime.graph_launch(capture.graph_exec, self.stream)
        capture.replay_count += 1
        self.self_cache_length += 1
        self.decode_position = None

    def batch_token_graph_contract(self) -> dict[str, object]:
        captures = tuple(self._token_graphs.values())
        return {
            "captured": bool(captures),
            "graph_count": len(captures),
            "buckets": [capture.bucket for capture in captures],
            "capture_positions": [capture.capture_position for capture in captures],
            "threads": [capture.threads for capture in captures],
            "capture_wall_ms": sum(capture.capture_wall_ms for capture in captures),
            "instantiate_wall_ms": sum(
                capture.instantiate_wall_ms for capture in captures
            ),
            "replay_count": sum(capture.replay_count for capture in captures),
        }

    def _close_token_graphs(self) -> None:
        for capture in reversed(tuple(self._token_graphs.values())):
            capture.close()
        self._token_graphs.clear()

    def read_tokens(self) -> np.ndarray:
        """Synchronize the resident stream and read the per-row int64 tokens."""

        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine batch runtime is closed")
        self.runtime.stream_synchronize(self.stream)
        host = np.empty(self.max_batch, dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(host),
            self.workspace.allocation("token").buffer,
            runtime=self.runtime,
        )
        for token_id in host:
            if token_id < 0 or token_id >= self.spec.vocab_size:
                raise RuntimeError(
                    f"Moonshine batch decoder returned invalid token ID {int(token_id)}"
                )
        return host

    def allocation_contract(self) -> dict[str, int | bool | None]:
        """Fixed-address contract for the batch decoder."""

        return {
            "max_batch": self.max_batch,
            "cross_cache_valid": self.cross_cache_valid,
            "encoder_state_valid": self.encoder_state_valid,
            "self_cache_length": self.self_cache_length,
            "decode_position": self.decode_position,
            "resident_nbytes": self.resident_nbytes,
            "workspace_nbytes": sum(
                allocation.buffer.nbytes
                for allocation in self.workspace._allocations.values()
            ),
            "baseline_current_allocated_bytes": self._allocation_baseline,
        }

    def close(self) -> None:
        """Free weights, workspace, stream, and report teardown parity."""

        if self.closed:
            return
        self.closed = True
        try:
            if self._token_graphs:
                self._close_token_graphs()
            if self.workspace is not None:
                self.workspace.free()
            if self.owns_weights and self.loaded_model is not None and self.weights is not None:
                self.loaded_model.weights.free(runtime=self.runtime)
            if self.stream:
                self.runtime.stream_destroy(self.stream)
                self.stream = 0
        finally:
            after = memory_stats()["current_allocated_bytes"]
            self.teardown_returned_to_baseline = after <= self._allocation_baseline

    def __enter__(self) -> "MoonshineCudaBatchRuntime":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
