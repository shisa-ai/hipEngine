"""Torch-free fixed-address Moonshine resident eager decoder for CUDA ``sm_120a`` (C2).

C2 mirrors the HIP ``MoonshineResidentRuntime`` composition contract for the
peer CUDA backend: every pinned FP16 weight is materialized once on the CUDA
device (with the tied embedding/LM head aliased to a single owner), all
self/cross caches, RoPE tables, masks, logits, token, position, and scratch are
reserved once before timed execution, and sequential decode positions ``0..193``
run eagerly with zero timed allocation.  Cross attention consumes the retained
head-major cross K/V (fixture or encoder-handoff path), and the LM head uses the
bounded fused projection+stable-argmax candidate from C1f.  Kernel schedules are
the measured C1e/C1f defaults: cross attention is flat parallel t256, self
attention is one-wave t32 below 8 visible tokens and parallel t256 at/above, and
the fused LM head uses ``rows_per_block=8``.
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
from hipengine.runtime.workspace import RuntimeWorkspace

# Measured C1e schedule: cross attention is flat parallel t256 at every
# production encoder length (40/207/1248) on exclusive GPU0.
_CROSS_THREADS = 256
# C1f fused LM-head rows-per-block measured bucket.
_LM_HEAD_ROWS_PER_BLOCK = 8


def _self_attention_threads(visible: int) -> int:
    """Measured C1e general cache-position bucket for self attention.

    One-wave t32 below 8 visible tokens, parallel t256 at/above.
    """
    from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
        _default_self_threads,
    )

    return _default_self_threads(visible)


def _moonshine_cuda_token_graph_bucket(
    position: int,
    *,
    capacity: int = 194,
) -> tuple[str, int, int]:
    """Return the CUDA self-attention launch bucket for one decode position.

    The measured CUDA schedule has exactly two self-attention variants (t32
    below 8 visible tokens, parallel t256 at/above), so there are two CUDA
    graph buckets: route positions 0-6 (visible 1-7, one-wave t32) and route
    positions 7+ (visible 8+, parallel t256).  This is the CUDA-measured
    counterpart to the HIP ``_moonshine_token_graph_bucket`` hypotheses, which
    C3 re-evaluates rather than importing.
    """

    if isinstance(position, bool) or not isinstance(position, int):
        raise ValueError("position must be an integer")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if position < 0 or position >= capacity:
        raise ValueError("position is outside self-cache capacity")
    if position <= 6:
        return ("positions_0_6", 0, min(6, capacity - 1))
    return (f"positions_7_{capacity - 1}", 7, capacity - 1)


class NoAllocationError(RuntimeError):
    """Raised when a future timed region allocates through hipEngine memory."""


@dataclass
class MoonshineCudaTokenGraph:
    """One captured fixed-address token DAG for a CUDA self-cache launch bucket."""

    owner: "MoonshineCudaResidentRuntime"
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


@dataclass(frozen=True)
class MoonshineCudaCacheView:
    key: Tensor
    value: Tensor


@dataclass(frozen=True)
class MoonshineCudaDecoderLibraries:
    """Prebuilt code objects used by the CUDA resident decoder chain."""

    glue: object
    layernorm: object
    projection: object
    attention: object
    lm_head: object


class MoonshineCudaResidentRuntime:
    """Own every fixed-address object needed by a batch-one CUDA Moonshine decoder."""

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
        "lm_head_values",
        "lm_head_indices",
        "lm_head_out_value",
    )

    def __init__(
        self,
        *,
        encoder_frames: int,
        model_path: str | Path | None = None,
        loaded_model: MoonshineLoadedModel | None = None,
        device: Device | None = None,
        runtime: CudaRuntime | None = None,
        owns_weights: bool = True,
    ) -> None:
        if (model_path is None) == (loaded_model is None):
            raise ValueError("provide exactly one of model_path or loaded_model")
        self.runtime = runtime or get_cuda_runtime()
        self.device = device or Device("cuda", 0)
        self.loaded_model = loaded_model
        self.weights = loaded_model.weights if loaded_model is not None else None
        self.spec = loaded_model.spec if loaded_model is not None else None
        self.owns_weights = bool(owns_weights)
        self.encoder_frames = int(encoder_frames)
        self.workspace = RuntimeWorkspace(device=self.device, runtime=self.runtime)
        self.stream = 0
        self.self_cache_length = 0
        self.cross_cache_valid = False
        self.encoder_state_valid = False
        self.decode_position: int | None = None
        self.decoder_libraries: MoonshineCudaDecoderLibraries | None = None
        self._token_graphs: dict[str, MoonshineCudaTokenGraph] = {}
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
            # The decoder kernels are length-generic (cross attention measured at
            # the 40/207/1248 production buckets but correct for any frame count).
            # C2 fixture gates use the actual retained frame counts (24-105), so
            # the strict production-bucket check is left to the C4 encoder
            # handoff rather than the resident decoder.
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
        from hipengine.kernels.cuda_sm120a.linear.lm_head import (
            lm_head_argmax_scratch_elements,
        )

        num_blocks = lm_head_argmax_scratch_elements(
            spec.vocab_size, _LM_HEAD_ROWS_PER_BLOCK
        )
        reserve("lm_head_values", (num_blocks,), DType.FP32)
        reserve("lm_head_indices", (num_blocks,), DType.INT64)
        reserve("lm_head_out_value", (1,), DType.FP32)

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
        """Load every code object before any no-allocation/timed token region.

        Mirrors the HIP runtime contract: kernel preparation is an explicit step
        (not part of ``__init__``) so callers can inject fake libraries in tests
        and keep construction free of any code-object build or load.
        """

        if self.closed:
            raise RuntimeError("Moonshine runtime is closed")
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
        return (
            self.loaded_model.weights.resident_bytes
            if hasattr(self.loaded_model.weights, "resident_bytes")
            else sum(
                allocation.buffer.nbytes
                for allocation in self.loaded_model.weights.tensors.values()
            )
        )

    def _cache_view(self, name: str, layer: int) -> MoonshineCudaCacheView:
        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine runtime is closed")
        if layer < 0 or layer >= self.spec.decoder_layers:
            raise ValueError("cache layer is outside the decoder")
        allocation = self.workspace.allocation(name)
        capacity = (
            self.spec.self_cache_capacity
            if name == "self_kv"
            else self.encoder_frames
        )
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

        return MoonshineCudaCacheView(key=view(0), value=view(1))

    def self_cache(self, layer: int) -> MoonshineCudaCacheView:
        return self._cache_view("self_kv", layer)

    def cross_cache(self, layer: int) -> MoonshineCudaCacheView:
        return self._cache_view("cross_kv", layer)

    def load_cross_cache(
        self,
        keys: Sequence[np.ndarray],
        values: Sequence[np.ndarray],
        *,
        mask: np.ndarray | None = None,
    ) -> None:
        """Copy retained head-major cross K/V (and the encoder mask) into fixed buffers.

        Each array is ``[1, kv_heads, frames, head_dim]`` FP16 (fixture layout).
        Must be called before any token step; resets generation state.
        """
        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine runtime is closed")
        spec = self.spec
        if len(keys) != spec.decoder_layers or len(values) != spec.decoder_layers:
            raise ValueError(
                f"cross cache needs {spec.decoder_layers} layers, "
                f"got {len(keys)} keys / {len(values)} values"
            )
        for layer in range(spec.decoder_layers):
            key = np.ascontiguousarray(keys[layer], dtype=np.float16)
            value = np.ascontiguousarray(values[layer], dtype=np.float16)
            expected = (
                1,
                spec.decoder_kv_heads,
                self.encoder_frames,
                spec.head_dim,
            )
            if key.shape != expected or value.shape != expected:
                raise ValueError(
                    f"layer {layer} cross cache shape {key.shape} != {expected}"
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
        if mask is not None:
            flat = np.ascontiguousarray(mask, dtype=np.int32).reshape(-1)
            if flat.size != self.encoder_frames:
                raise ValueError(
                    f"encoder mask size {flat.size} != encoder frames {self.encoder_frames}"
                )
            copy_host_to_device(
                self.workspace.allocation("encoder_attention_mask").buffer,
                host_array_ptr(flat),
                runtime=self.runtime,
            )
        self.runtime.stream_synchronize(self.stream)
        self.cross_cache_valid = True
        self.encoder_state_valid = True
        self.reset_generation(clear_cross_cache=False)

    def set_encoder_state_from_device(
        self,
        *,
        hidden_fp16_ptr: int,
        attention_mask_int32_ptr: int,
        source_frames: int,
    ) -> None:
        """Copy a contiguous device encoder prefix into the fixed padded bucket.

        The producing runtime owns and validates finite FP16 hidden values plus a
        binary int32 mask, and must synchronize its producer stream before this
        handoff.  The source tensors remain caller-owned (C4 D2D bring-up).
        """

        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine runtime is closed")
        if self.self_cache_length or self.decode_position is not None:
            raise RuntimeError("reset generation before replacing encoder state")
        if (
            isinstance(hidden_fp16_ptr, bool)
            or not isinstance(hidden_fp16_ptr, int)
            or isinstance(attention_mask_int32_ptr, bool)
            or not isinstance(attention_mask_int32_ptr, int)
            or hidden_fp16_ptr <= 0
            or attention_mask_int32_ptr <= 0
        ):
            raise ValueError("device encoder pointers must be positive integers")
        if (
            isinstance(source_frames, bool)
            or not isinstance(source_frames, int)
            or source_frames <= 0
            or source_frames > self.encoder_frames
        ):
            raise ValueError(
                f"source_frames must be in 1..{self.encoder_frames} for the resident bucket"
            )

        hidden = self.workspace.allocation("encoder_hidden").buffer
        mask = self.workspace.allocation("encoder_attention_mask").buffer
        self.runtime.memset_async(hidden.ptr, 0, hidden.nbytes, self.stream)
        self.runtime.memset_async(mask.ptr, 0, mask.nbytes, self.stream)
        self.runtime.memcpy_async(
            hidden.ptr,
            hidden_fp16_ptr,
            source_frames * self.spec.hidden_size * DType.FP16.itemsize,
            MemcpyKind.DEVICE_TO_DEVICE,
            self.stream,
        )
        self.runtime.memcpy_async(
            mask.ptr,
            attention_mask_int32_ptr,
            source_frames * DType.INT32.itemsize,
            MemcpyKind.DEVICE_TO_DEVICE,
            self.stream,
        )
        self.runtime.stream_synchronize(self.stream)
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
        from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
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
                stream=self.stream,
                library=libraries.projection,
                runtime=self.runtime,
            )
        self.runtime.stream_synchronize(self.stream)
        self.cross_cache_valid = True
        self.encoder_state_valid = True
        self.reset_generation(clear_cross_cache=False)

    def reset_generation(self, *, clear_cross_cache: bool = False) -> None:
        """Reset generation state without changing any allocation or address."""

        if self.closed:
            raise RuntimeError("Moonshine runtime is closed")
        names = ("self_kv", *self._SCRATCH_NAMES)
        if clear_cross_cache:
            names = (*names, "cross_kv", "encoder_hidden", "encoder_attention_mask")
        self._zero(names)
        self.runtime.stream_synchronize(self.stream)
        self.self_cache_length = 0
        self.decode_position = None
        if clear_cross_cache:
            self.cross_cache_valid = False
            self.encoder_state_valid = False

    def _zero(self, names: tuple[str, ...]) -> None:
        for name in names:
            allocation = self.workspace.allocation(name)
            self.runtime.memset_async(
                allocation.buffer.ptr,
                0,
                allocation.buffer.nbytes,
                self.stream,
            )

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
        # Ensure the previous step's fused LM-head write to the token buffer has
        # completed before overwriting it with the next input token.
        self.runtime.stream_synchronize(self.stream)
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
        """Enqueue one complete resident decoder step through eager Python dispatch."""

        route_position = self._require_token_step_ready()
        with self.no_allocation_region("token_step"):
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
        from hipengine.kernels.cuda_sm120a.attention.moonshine_attention import (
            moonshine_cross_attention_parallel_fp16,
            moonshine_self_attention_fp16,
            moonshine_self_attention_parallel_fp16,
        )
        from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
            moonshine_embedding_lookup_fp16,
            moonshine_partial_rope_cache_append_fp16,
        )
        from hipengine.kernels.cuda_sm120a.linear.lm_head import (
            moonshine_lm_head_argmax_fp16,
        )
        from hipengine.kernels.cuda_sm120a.linear.moonshine_projection import (
            moonshine_f16_lm_head_projection_wave8,
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
                1,
                spec.hidden_size,
                spec.hidden_size,
                library=libraries.projection,
                **common,
            )

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
            visible = route_position + 1
            threads = _self_attention_threads(visible)
            if threads == 32:
                moonshine_self_attention_fp16(
                    attention.ptr,
                    self_cache.key.ptr,
                    self_cache.value.ptr,
                    position.ptr,
                    attention.ptr,
                    spec.decoder_attention_heads,
                    spec.head_dim,
                    spec.self_cache_capacity,
                    library=libraries.attention,
                    **common,
                )
            else:
                moonshine_self_attention_parallel_fp16(
                    attention.ptr,
                    self_cache.key.ptr,
                    self_cache.value.ptr,
                    position.ptr,
                    attention.ptr,
                    spec.decoder_attention_heads,
                    spec.head_dim,
                    spec.self_cache_capacity,
                    threads=threads,
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
                1,
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
            moonshine_cross_attention_parallel_fp16(
                query.ptr,
                cross_cache.key.ptr,
                cross_cache.value.ptr,
                self.tensor("encoder_attention_mask").ptr,
                attention.ptr,
                spec.decoder_attention_heads,
                spec.head_dim,
                self.encoder_frames,
                threads=_CROSS_THREADS,
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
        # Bounded C1f fused LM head + stable argmax (rows_per_block=8, byte-exact
        # with the two-step projection+argmax path). Writes the token into the
        # fixed token tensor so read_token()/next-step embedding reuse it.
        moonshine_lm_head_argmax_fp16(
            normalized.ptr,
            self.weights[spec.lm_head_alias_name].ptr,
            self.tensor("lm_head_values").ptr,
            self.tensor("lm_head_indices").ptr,
            self.tensor("token").ptr,
            self.tensor("lm_head_out_value").ptr,
            spec.hidden_size,
            spec.vocab_size,
            rows_per_block=_LM_HEAD_ROWS_PER_BLOCK,
            library=libraries.lm_head,
            **common,
        )

    def read_token(self) -> int:
        """Synchronize the resident stream and read the selected int64 token."""

        if self.closed or self.spec is None:
            raise RuntimeError("Moonshine runtime is closed")
        self.runtime.stream_synchronize(self.stream)
        host = np.empty(1, dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(host),
            self.workspace.allocation("token").buffer,
            runtime=self.runtime,
        )
        token_id = int(host[0])
        if token_id < 0 or token_id >= self.spec.vocab_size:
            raise RuntimeError(f"Moonshine decoder returned invalid token ID {token_id}")
        return token_id

    def capture_token_graphs(self) -> tuple[MoonshineCudaTokenGraph, ...]:
        """Capture one reusable token DAG for each CUDA self-attention bucket.

        The measured CUDA schedule has exactly two variants (t32 below 8
        visible tokens, parallel t256 at/above), so two graphs are captured: at
        route positions 0 (visible 1, one-wave t32) and 7 (visible 8, parallel
        t256).  Every kernel reads the current position from the fixed position
        tensor at launch, so a captured graph is position-generic within its
        bucket and is replayed after ``set_decode_state`` updates token and
        position under fixed addresses.
        """

        import time

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
            position for position in (0, 7) if position < self.spec.self_cache_capacity
        )
        captures: list[MoonshineCudaTokenGraph] = []
        try:
            for position in representatives:
                bucket, minimum, maximum = _moonshine_cuda_token_graph_bucket(
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
                    MoonshineCudaTokenGraph(
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
        bucket, _, _ = _moonshine_cuda_token_graph_bucket(
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
            "instantiate_wall_ms": sum(
                capture.instantiate_wall_ms for capture in captures
            ),
            "replay_count": sum(capture.replay_count for capture in captures),
        }

    def _close_token_graphs(self) -> None:
        for capture in reversed(tuple(self._token_graphs.values())):
            capture.close()
        self._token_graphs.clear()

    def allocation_contract(self) -> dict[str, int | bool | None]:
        """Fixed-address contract for the resident decoder."""

        return {
            "cross_cache_valid": self.cross_cache_valid,
            "encoder_state_valid": self.encoder_state_valid,
            "self_cache_length": self.self_cache_length,
            "decode_position": self.decode_position,
            "resident_nbytes": self.resident_nbytes,
            "workspace_nbytes": sum(
                allocation.buffer.nbytes for allocation in self.workspace._allocations.values()
            ),
            "baseline_current_allocated_bytes": self._allocation_baseline,
        }

    def close(self) -> None:
        """Free weights, workspace, graphs, events/stream, and report teardown parity."""

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
            self.teardown_returned_to_baseline = (
                after <= self._allocation_baseline
            )

    def __enter__(self) -> "MoonshineCudaResidentRuntime":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
