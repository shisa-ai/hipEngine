"""Torch-free fixed-address Moonshine resident runtime skeleton.

Phase 1 owns memory, streams, events, caches, and lifecycle only.  Decoder
kernels intentionally begin in Phase 2.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import copy_host_to_device, host_array_ptr, memory_stats
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference.moonshine import moonshine_rope_tables
from hipengine.loading.moonshine import MoonshineLoadedModel, load_moonshine_model
from hipengine.runtime.workspace import RuntimeWorkspace


class NoAllocationError(RuntimeError):
    """Raised when a future timed region allocates through hipEngine memory."""


@dataclass(frozen=True)
class MoonshineCacheView:
    key: Tensor
    value: Tensor


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
        if clear_cross_cache:
            self.cross_cache_valid = False

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

    def token_step(self) -> None:
        raise NotImplementedError("Moonshine decoder token kernels begin in Phase 2")

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
            "teardown_returned_to_baseline": self.teardown_returned_to_baseline,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
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
    "MoonshineResidentRuntime",
    "NoAllocationError",
]
