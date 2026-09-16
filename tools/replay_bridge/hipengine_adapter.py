#!/usr/bin/env python3
"""hipEngine side of the identical-operand cross-engine replay.

Loads a packet captured by ``capture_packet.py``, re-resolves the dispatch key
that produced it, and runs that kernel on the packet's exact weight and
activation bytes. Nothing here reconstructs or approximates an operand: the
bytes on the device are the bytes the packet recorded.

The dispatch guard is deliberately strict. ``resolve`` applies generic
backend/quant/variant fallbacks, so "resolve returned a callable" is not
evidence that the recorded path was used. The adapter therefore requires an
exact-key registration through ``is_registered`` (which performs no fallback)
before it will resolve anything, and refuses to report a timing if that check
fails.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DTYPE_BYTES = {"f32": 4, "bf16": 2, "fp16": 2}


class DispatchGuardError(RuntimeError):
    """The recorded dispatch path is not available, so no timing is valid."""


@dataclass
class Packet:
    stem: Path
    manifest: dict
    w_raw: np.ndarray
    x: np.ndarray
    out: np.ndarray
    key: dict = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return int(self.manifest["geometry"]["rows"])

    @property
    def in_features(self) -> int:
        return int(self.manifest["geometry"]["in_features"])

    @property
    def out_features(self) -> int:
        return int(self.manifest["geometry"]["out_features"])

    @property
    def activation_dtype(self) -> str:
        return str(self.manifest["arrays"]["x"]["dtype"])

    @property
    def output_dtype(self) -> str:
        return str(self.manifest["arrays"]["out"]["dtype"])

    def x_float(self) -> np.ndarray:
        return to_float32(self.x, self.activation_dtype)

    def out_float(self) -> np.ndarray:
        return to_float32(self.out, self.output_dtype)

    def weight_matrix(self) -> np.ndarray:
        """Dequantize the packet's Q8_0 weight bytes to float32 (M, K)."""
        if str(self.manifest["quant"]) != "gguf_q8_0":
            raise ValueError(f"unsupported packet quant {self.manifest['quant']!r}")
        blocks = self.w_raw.reshape(self.out_features, self.in_features // 32, 34)
        scales = (
            np.ascontiguousarray(blocks[:, :, :2])
            .view(np.float16)
            .reshape(blocks.shape[:2])
            .astype(np.float32)
        )
        codes = blocks[:, :, 2:].view(np.int8).astype(np.float32)
        return (scales[:, :, None] * codes).reshape(self.out_features, self.in_features)


def to_float32(array: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "f32":
        return array.astype(np.float32, copy=False)
    if dtype == "bf16":
        return (array.astype(np.uint32) << 16).view(np.float32)
    if dtype == "fp16":
        return array.view(np.float16).astype(np.float32)
    raise ValueError(f"unsupported dtype {dtype!r}")


def load_packet(stem: Path, *, verify: bool = True) -> Packet:
    stem = Path(stem)
    npz_path = stem.with_suffix(".npz")
    json_path = stem.with_suffix(".json")
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"packet not found: {npz_path} / {json_path}")
    manifest = json.loads(json_path.read_text())
    with np.load(npz_path) as data:
        arrays = {name: data[name] for name in ("w_raw", "x", "out")}

    if verify:
        for name, array in arrays.items():
            recorded = manifest["arrays"][name]
            if list(array.shape) != list(recorded["shape"]):
                raise ValueError(
                    f"packet {name} shape {list(array.shape)} does not match "
                    f"manifest {recorded['shape']}"
                )
            digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            if digest != recorded["sha256"]:
                raise ValueError(f"packet {name} sha256 mismatch: packet is corrupted")

    geometry = manifest["geometry"]
    if arrays["x"].shape[0] != geometry["rows"] or arrays["x"].shape[1] != geometry["in_features"]:
        raise ValueError("packet activation shape does not match recorded geometry")
    if arrays["out"].shape != (geometry["rows"], geometry["out_features"]):
        raise ValueError("packet output shape does not match recorded geometry")

    return Packet(
        stem=stem,
        manifest=manifest,
        w_raw=arrays["w_raw"],
        x=arrays["x"],
        out=arrays["out"],
        key=dict(manifest["hipengine_variant"]),
    )


@dataclass
class ReplayResult:
    key: dict
    registered: bool
    wrapper_name: str
    event_ms: float
    wall_ms: float
    spread_ms: float
    samples: int
    output: np.ndarray
    wall_total_ms: float = 0.0


class HipEngineAdapter:
    """Runs one packet through hipEngine's recorded dispatch path."""

    def __init__(self, packet: Packet, *, strict: bool = True):
        self.packet = packet
        self.strict = strict
        self._buffers: list[int] = []
        self._runtime = None
        self._fn = None
        self._wrapper_name = ""
        self._registered = False
        self._key = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "HipEngineAdapter":
        self.prepare()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def prepare(self) -> None:
        from hipengine.core.hip import MemcpyKind, get_hip_runtime
        from hipengine.kernels.registry import KernelKey, is_registered, resolve

        runtime = get_hip_runtime()
        self._runtime = runtime
        packet = self.packet
        key = KernelKey(
            str(packet.key["backend"]),
            str(packet.key["layer"]),
            str(packet.key["quant"]),
            str(packet.key["variant"]),
        )
        self._key = key

        # Exact-key registration only. resolve() would silently accept a
        # broader fallback, which is precisely the failure this guards against.
        self._registered = is_registered(key)
        if not self._registered:
            if self.strict:
                raise DispatchGuardError(
                    "recorded hipEngine dispatch key is not registered, so any "
                    f"timing would be of a fallback kernel: {key}"
                )
            return
        self._fn = resolve(
            backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant
        )
        self._wrapper_name = _attribute_name(self._fn)

        # Upload the packet's exact bytes. Nothing is regenerated.
        self._buffers.append(self._upload(runtime, packet.w_raw, MemcpyKind.HOST_TO_DEVICE))
        self._buffers.append(self._upload(runtime, packet.x, MemcpyKind.HOST_TO_DEVICE))
        out_nbytes = packet.rows * packet.out_features * _DTYPE_BYTES[packet.output_dtype]
        self._buffers.append(runtime.malloc(out_nbytes))

    def close(self) -> None:
        if self._runtime is not None:
            for ptr in self._buffers:
                try:
                    self._runtime.free(ptr)
                except Exception:  # noqa: BLE001 - teardown must not mask the result
                    pass
        self._buffers = []

    # -- execution ---------------------------------------------------------

    def _upload(self, runtime, array: np.ndarray, kind) -> int:
        contiguous = np.ascontiguousarray(array)
        ptr = runtime.malloc(int(contiguous.nbytes))
        runtime.memcpy(ptr, contiguous.ctypes.data, int(contiguous.nbytes), kind)
        return ptr

    def _call(self) -> None:
        self._fn(
            self._buffers[1],
            self._buffers[0],
            self._buffers[2],
            self.packet.rows,
            self.packet.in_features,
            self.packet.out_features,
            stream=0,
            runtime=self._runtime,
        )

    def replay(self, *, reps: int = 20, warmup: int = 5, read_output: bool = True) -> ReplayResult:
        if not self._registered or self._fn is None:
            raise DispatchGuardError(
                f"recorded dispatch key is not registered: {self._key}"
            )
        from hipengine.core.hip import MemcpyKind

        runtime = self._runtime
        packet = self.packet

        for _ in range(int(warmup)):
            self._call()
        runtime.device_synchronize()

        start = runtime.event_create()
        stop = runtime.event_create()
        samples: list[float] = []
        try:
            runtime.event_record(start, 0)
            wall_start = time.perf_counter()
            for _ in range(int(reps)):
                self._call()
            runtime.event_record(stop, 0)
            runtime.event_synchronize(stop)
            wall_total = (time.perf_counter() - wall_start) * 1e3
            event_total = runtime.event_elapsed_time_ms(start, stop)

            # Per-sample spread on the same kernel, so a bimodal schedule shows up.
            for _ in range(int(reps)):
                runtime.event_record(start, 0)
                self._call()
                runtime.event_record(stop, 0)
                runtime.event_synchronize(stop)
                samples.append(runtime.event_elapsed_time_ms(start, stop))
        finally:
            runtime.event_destroy(start)
            runtime.event_destroy(stop)

        output = np.empty(0, dtype=np.float32)
        if read_output:
            if packet.output_dtype == "f32":
                output = np.empty((packet.rows, packet.out_features), dtype=np.float32)
            else:
                output = np.empty((packet.rows, packet.out_features), dtype=np.uint16)
            runtime.memcpy(
                output.ctypes.data, self._buffers[2], int(output.nbytes), MemcpyKind.DEVICE_TO_HOST
            )
            output = to_float32(output, packet.output_dtype)

        return ReplayResult(
            key=dict(packet.key),
            registered=True,
            wrapper_name=self._wrapper_name,
            event_ms=event_total / int(reps),
            wall_ms=wall_total / int(reps),
            wall_total_ms=wall_total,
            spread_ms=max(samples) - min(samples) if samples else 0.0,
            samples=len(samples),
            output=output,
        )


def _attribute_name(function: object) -> str:
    """Best-effort human name for a registered wrapper.

    The registry stores plain callables, so the informative name is the module
    attribute the registration used, not ``__qualname__`` (which is the closure
    name shared by every wrapper in the family).
    """
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv

    for name, value in vars(gguf_k_gemv).items():
        if value is function and not name.startswith("_"):
            return f"gguf_k_gemv.{name}"
    return getattr(function, "__qualname__", repr(function))
