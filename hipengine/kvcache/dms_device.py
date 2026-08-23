"""Device payload store for the compact DMS backend (C2-7 U6).

Owns the per-layer device slot buffers (BF16 K/V, int32 token positions,
uint8 evict flags) and launches the registered ``dms_compact`` HIP
kernels. The host backend keeps only the O(rows x heads x capacity)
extent metadata; the BF16 K/V payload exists solely here, so the
no-shadow property holds on the host. The host parent implementation
remains the registered fallback: when the HIP runtime is unavailable the
backend constructs without a store and runs entirely host-side.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.registry import resolve

ENV_ENABLE = "HIPENGINE_DMS_DEVICE_PAYLOADS"
ENV_TRIPWIRE = "HIPENGINE_DMS_DEVICE_TRIPWIRE"


class DMSDeviceUnavailable(RuntimeError):
    """The HIP runtime is not available for the device payload path."""


def device_payloads_requested(explicit: bool | None) -> bool:
    """Resolve the opt-in flag: explicit argument, else the environment."""
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(ENV_ENABLE, "").strip().lower() in {"1", "true", "yes", "on"}


def tripwire_enabled() -> bool:
    """Post-append status readback (default off; enable for soak runs)."""
    return os.environ.get(ENV_TRIPWIRE, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DMSLayerView:
    """Readback of one layer's device slot buffers (test/observability only)."""

    k_bits: np.ndarray  # [slots, dim] uint16
    v_bits: np.ndarray  # [slots, dim] uint16
    positions: np.ndarray  # [slots] int32
    evict: np.ndarray  # [slots] uint8


@dataclass(frozen=True)
class DMSDeviceExtentSnapshot:
    layer: int
    head: int
    start: int
    length: int
    k_bits: np.ndarray
    v_bits: np.ndarray
    positions: np.ndarray
    evict: np.ndarray


@dataclass(frozen=True)
class DMSDevicePayloadSnapshot:
    """Temporary request-owned compact journal; never a persistent dense shadow."""

    extents: tuple[DMSDeviceExtentSnapshot, ...]


class DMSDevicePayloadStore:
    """Per-layer device slot pools + shared staging for the dms_compact kernels."""

    def __init__(self, *, retrofit, slots_per_layer: int, max_pack_rows: int) -> None:
        from hipengine.core.hip import get_hip_runtime
        from hipengine.kernels.hip_gfx1100.attention import build_dms_compact

        get_hip_runtime()  # raises when HIP is unavailable
        self._library = build_dms_compact(load=True)
        try:
            self._pack_fn = resolve(
                backend="hip_gfx1100",
                layer="dms_streaming_pack",
                quant="bf16",
                variant="count_rank_scatter",
            )
            self._append_fn = resolve(
                backend="hip_gfx1100",
                layer="dms_append_decode",
                quant="bf16",
                variant="compact_append_evict",
            )
            self._attn_fn = resolve(
                backend="hip_gfx1100",
                layer="dms_compact_attn_decode",
                quant="bf16",
                variant="grouped_gqa",
            )
        except Exception as exc:  # unregistered kernels = unavailable device path
            raise DMSDeviceUnavailable(f"dms_compact kernels not registered: {exc}") from exc

        self._layers = int(retrofit.num_layers)
        self._heads = int(retrofit.num_kv_heads)
        self._q_heads = int(retrofit.num_q_heads)
        self._dim = int(retrofit.head_dim)
        self._window = int(retrofit.window_size)
        self._slots = int(slots_per_layer)
        if self._q_heads % self._heads != 0:
            raise ValueError("retrofit GQA geometry must divide evenly")

        self._buffers: list[DeviceBuffer] = []
        self._k_slot = [self._alloc(self._slots * self._dim * 2) for _ in range(self._layers)]
        self._v_slot = [self._alloc(self._slots * self._dim * 2) for _ in range(self._layers)]
        self._positions = [self._alloc(self._slots * 4) for _ in range(self._layers)]
        self._slot_evict = [self._alloc(self._slots) for _ in range(self._layers)]

        h, d = self._heads, self._dim
        self._stg = {
            "pack_k": self._alloc(int(max_pack_rows) * h * d * 2),
            "pack_v": self._alloc(int(max_pack_rows) * h * d * 2),
            "pack_evict": self._alloc(int(max_pack_rows) * h),
            "append_k": self._alloc(h * d * 2),
            "append_v": self._alloc(h * d * 2),
            "append_evict": self._alloc(h),
            "row_starts": self._alloc(4),
            "row_tokens": self._alloc(4),
            "row_positions": self._alloc(4),
            "base": self._alloc(h * 4),
            "capacity": self._alloc(h * 4),
            "live": self._alloc(h * 4),
            "status": self._alloc(h * 4),
            "q": self._alloc(self._q_heads * d * 4),
            "out": self._alloc(self._q_heads * d * 4),
        }
        self._closed = False

    def _alloc(self, nbytes: int) -> DeviceBuffer:
        buf = malloc(nbytes)
        self._buffers.append(buf)
        return buf

    def _upload(self, name: str, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        copy_host_to_device(self._stg[name], host_array_ptr(array), array.nbytes)

    def pack_layer(
        self,
        layer: int,
        k_bits: np.ndarray,
        v_bits: np.ndarray,
        evict: np.ndarray,
        base: np.ndarray,
        capacity: np.ndarray,
    ) -> None:
        """Launch the streaming pack for one row (one request) on one layer.

        ``k_bits``/``v_bits`` are ``[tokens, heads, dim]`` uint16, ``evict``
        is ``[tokens, heads]`` uint8, ``base``/``capacity`` are ``[heads]``
        int32 (the request's per-head extents on this layer).
        """
        self._check_closed()
        tokens = int(k_bits.shape[0])
        if k_bits.shape != (tokens, self._heads, self._dim) or v_bits.shape != k_bits.shape:
            raise ValueError("DMS device pack expects K/V [tokens,heads,dim]")
        if evict.shape != (tokens, self._heads):
            raise ValueError("DMS device pack evict shape mismatch")
        self._upload("pack_k", k_bits)
        self._upload("pack_v", v_bits)
        self._upload("pack_evict", evict.astype(np.uint8))
        self._upload("base", base)
        self._upload("capacity", capacity)
        self._upload("row_starts", np.zeros(1, dtype=np.int32))
        self._upload("row_tokens", np.asarray([tokens], dtype=np.int32))
        self._pack_fn(
            self._stg["pack_k"].ptr,
            self._stg["pack_v"].ptr,
            self._stg["pack_evict"].ptr,
            self._stg["base"].ptr,
            self._stg["capacity"].ptr,
            self._stg["live"].ptr,
            self._stg["row_starts"].ptr,
            self._stg["row_tokens"].ptr,
            self._k_slot[layer].ptr,
            self._v_slot[layer].ptr,
            self._positions[layer].ptr,
            self._slot_evict[layer].ptr,
            1,
            self._heads,
            self._dim,
            self._window,
            library=self._library,
        )

    def append_layer(
        self,
        layer: int,
        k_new_bits: np.ndarray,
        v_new_bits: np.ndarray,
        evict_new: np.ndarray,
        row_position: int,
        base: np.ndarray,
        capacity: np.ndarray,
        live: np.ndarray,
    ) -> None:
        """Launch the append/decode for one row (one request) on one layer.

        The caller must have already verified (host pre-check) that the
        append cannot overflow; the kernel's own fail-closed status is
        read back as a tripwire when ``ENV_TRIPWIRE`` is set.
        """
        self._check_closed()
        if k_new_bits.shape != (self._heads, self._dim):
            raise ValueError("DMS device append expects K/V [heads,dim]")
        self._upload("append_k", k_new_bits)
        self._upload("append_v", v_new_bits)
        self._upload("append_evict", evict_new.astype(np.uint8))
        self._upload("row_positions", np.asarray([int(row_position)], dtype=np.int32))
        self._upload("base", base)
        self._upload("capacity", capacity)
        self._upload("live", live)
        self._upload("status", np.zeros(self._heads, dtype=np.int32))
        self._append_fn(
            self._stg["append_k"].ptr,
            self._stg["append_v"].ptr,
            self._stg["append_evict"].ptr,
            self._stg["row_positions"].ptr,
            self._stg["base"].ptr,
            self._stg["capacity"].ptr,
            self._stg["live"].ptr,
            self._k_slot[layer].ptr,
            self._v_slot[layer].ptr,
            self._positions[layer].ptr,
            self._slot_evict[layer].ptr,
            self._stg["status"].ptr,
            1,
            self._heads,
            self._dim,
            self._window,
            library=self._library,
        )
        if tripwire_enabled():
            status = np.zeros(self._heads, dtype=np.int32)
            copy_device_to_host(host_array_ptr(status), self._stg["status"], status.nbytes)
            if int(status.max()) != 0:
                raise RuntimeError(
                    f"DMS device append tripwire: kernel reported overflow on "
                    f"layer {layer} (host pre-check missed it): {status.tolist()}"
                )

    def attention_layer(
        self,
        layer: int,
        *,
        q: np.ndarray | None = None,
        q_ptr: int | None = None,
        out: np.ndarray | None = None,
        out_ptr: int | None = None,
        base: np.ndarray,
        live: np.ndarray,
        scale: float | None = None,
    ) -> None:
        """Launch GQA decode attention for one row (one request) on one layer.

        Either a host ``q`` (``[q_heads, dim]`` FP32, uploaded) or a device
        ``q_ptr`` is accepted; likewise ``out`` (host readback) or
        ``out_ptr`` (device). ``base``/``live`` are ``[heads]`` int32.
        """
        self._check_closed()
        if (q is None) == (q_ptr is None):
            raise ValueError("provide exactly one of q / q_ptr")
        if (out is None) == (out_ptr is None):
            raise ValueError("provide exactly one of out / out_ptr")
        if q is not None:
            q = np.ascontiguousarray(q, dtype=np.float32)
            if q.shape != (self._q_heads, self._dim):
                raise ValueError("DMS device attention expects Q [q_heads, dim]")
            self._upload("q", q)
            q_dev = self._stg["q"].ptr
        else:
            q_dev = int(q_ptr)
        self._upload("base", base)
        self._upload("live", live)
        if out is not None:
            out_dev = self._stg["out"].ptr
        else:
            out_dev = int(out_ptr)
        self._attn_fn(
            q_dev,
            self._k_slot[layer].ptr,
            self._v_slot[layer].ptr,
            self._stg["base"].ptr,
            self._stg["live"].ptr,
            out_dev,
            1,
            self._q_heads,
            self._heads,
            self._dim,
            float(scale) if scale is not None else float(self._dim**-0.5),
            max(1, int(np.max(live))),
            library=self._library,
        )
        if out is not None:
            copy_device_to_host(host_array_ptr(out), self._stg["out"], out.nbytes)

    def layer_view(self, layer: int) -> DMSLayerView:
        self._check_closed()
        dim = self._dim
        k_bits = np.zeros((self._slots, dim), dtype=np.uint16)
        v_bits = np.zeros_like(k_bits)
        positions = np.zeros(self._slots, dtype=np.int32)
        evict = np.zeros(self._slots, dtype=np.uint8)
        for array, buf in (
            (k_bits, self._k_slot[layer]),
            (v_bits, self._v_slot[layer]),
            (positions, self._positions[layer]),
            (evict, self._slot_evict[layer]),
        ):
            copy_device_to_host(host_array_ptr(array), buf, array.nbytes)
        return DMSLayerView(
            k_bits=k_bits, v_bits=v_bits, positions=positions, evict=evict
        )

    def snapshot(
        self,
        base_offsets: np.ndarray,
        range_capacity: np.ndarray,
    ) -> DMSDevicePayloadSnapshot:
        """Read only one request's compact extents at a commit barrier."""

        self._check_closed()
        bases = np.asarray(base_offsets, dtype=np.int32)
        capacities = np.asarray(range_capacity, dtype=np.int32)
        expected = (self._layers, self._heads)
        if bases.shape != expected or capacities.shape != expected:
            raise ValueError("DMS device snapshot extent metadata shape mismatch")
        extents: list[DMSDeviceExtentSnapshot] = []
        for layer in range(self._layers):
            for head in range(self._heads):
                start = int(bases[layer, head])
                length = int(capacities[layer, head])
                if start < 0 or length <= 0 or start + length > self._slots:
                    raise ValueError("DMS device snapshot extent is out of range")
                k_bits = np.empty((length, self._dim), dtype=np.uint16)
                v_bits = np.empty_like(k_bits)
                positions = np.empty((length,), dtype=np.int32)
                evict = np.empty((length,), dtype=np.uint8)
                for array, source, byte_offset in (
                    (k_bits, self._k_slot[layer], start * self._dim * 2),
                    (v_bits, self._v_slot[layer], start * self._dim * 2),
                    (positions, self._positions[layer], start * 4),
                    (evict, self._slot_evict[layer], start),
                ):
                    copy_device_to_host(
                        host_array_ptr(array),
                        DeviceBuffer(source.ptr + byte_offset, array.nbytes),
                        array.nbytes,
                    )
                extents.append(
                    DMSDeviceExtentSnapshot(
                        layer=layer,
                        head=head,
                        start=start,
                        length=length,
                        k_bits=k_bits,
                        v_bits=v_bits,
                        positions=positions,
                        evict=evict,
                    )
                )
        return DMSDevicePayloadSnapshot(extents=tuple(extents))

    def restore(self, snapshot: DMSDevicePayloadSnapshot) -> None:
        """Restore request-owned compact extents byte-for-byte after failure."""

        self._check_closed()
        if not isinstance(snapshot, DMSDevicePayloadSnapshot):
            raise TypeError("DMS device restore requires DMSDevicePayloadSnapshot")
        if len(snapshot.extents) != self._layers * self._heads:
            raise ValueError("DMS device snapshot extent count mismatch")
        seen: set[tuple[int, int]] = set()
        for extent in snapshot.extents:
            key = (int(extent.layer), int(extent.head))
            if key in seen:
                raise ValueError("DMS device snapshot has duplicate layer/head extent")
            seen.add(key)
            layer, head = key
            start = int(extent.start)
            length = int(extent.length)
            if not (0 <= layer < self._layers and 0 <= head < self._heads):
                raise ValueError("DMS device snapshot layer/head is out of range")
            if start < 0 or length <= 0 or start + length > self._slots:
                raise ValueError("DMS device snapshot extent is out of range")
            expected_payload = (length, self._dim)
            if extent.k_bits.shape != expected_payload or extent.k_bits.dtype != np.uint16:
                raise ValueError("DMS device snapshot K shape/dtype mismatch")
            if extent.v_bits.shape != expected_payload or extent.v_bits.dtype != np.uint16:
                raise ValueError("DMS device snapshot V shape/dtype mismatch")
            if extent.positions.shape != (length,) or extent.positions.dtype != np.int32:
                raise ValueError("DMS device snapshot position shape/dtype mismatch")
            if extent.evict.shape != (length,) or extent.evict.dtype != np.uint8:
                raise ValueError("DMS device snapshot eviction shape/dtype mismatch")
            for array, destination, byte_offset in (
                (extent.k_bits, self._k_slot[layer], start * self._dim * 2),
                (extent.v_bits, self._v_slot[layer], start * self._dim * 2),
                (extent.positions, self._positions[layer], start * 4),
                (extent.evict, self._slot_evict[layer], start),
            ):
                contiguous = np.ascontiguousarray(array)
                copy_host_to_device(
                    DeviceBuffer(destination.ptr + byte_offset, contiguous.nbytes),
                    host_array_ptr(contiguous),
                    contiguous.nbytes,
                )

    def close(self) -> None:
        if self._closed:
            return
        for buf in self._buffers:
            free(buf)
        self._buffers.clear()
        self._closed = True

    def _check_closed(self) -> None:
        if self._closed:
            raise RuntimeError("DMS device payload store is closed")
