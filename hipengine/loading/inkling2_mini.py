"""Read original Inkling 2 Mini text weights without expanding expert banks."""

from __future__ import annotations

from math import prod

import numpy as np

from hipengine.loading.safetensors import WeightIndex, _parse_safetensors_header_and_data_start
from hipengine.quant.inkling2_experts import decode_mxfp8_expert, decode_nvfp4_expert

_STORAGE = {"BF16": "<u2", "F32": "<f4", "U8": "u1", "F8_E4M3": "u1"}


class Inkling2TextWeights:
    """Keep read-only mappings, and decode only the requested expert or dense slice."""

    def __init__(self, index: WeightIndex, *, namespace: str = "model.llm"):
        self._index = index
        self._namespace = namespace
        self._headers = {p: _parse_safetensors_header_and_data_start(p) for p in index.shards}
        self._file_sizes = {p: p.stat().st_size for p in index.shards}
        self._mapped: dict[str, np.ndarray] = {}

    def _info(self, name):
        return self._index.tensors[f"{self._namespace}.{name}"]

    def raw(self, name: str) -> np.ndarray:
        if name not in self._mapped:
            info = self._info(name)
            header, base = self._headers[info.shard_path]
            dtype_name = _STORAGE.get(info.dtype)
            if dtype_name is None:
                raise ValueError(f"unsupported Inkling 2 tensor dtype for {name}: {info.dtype}")
            dtype = np.dtype(dtype_name)
            begin, end = header[info.name]["data_offsets"]
            if begin < 0 or end - begin != prod(info.shape) * dtype.itemsize or base + end > self._file_sizes[info.shard_path]:
                raise ValueError(f"invalid Inkling 2 tensor byte range for {name}")
            self._mapped[name] = np.memmap(info.shard_path, mode="r", offset=base + begin, dtype=dtype, shape=info.shape)
        return self._mapped[name]

    def dense(self, name: str, selector=...) -> np.ndarray:
        dtype = self._info(name).dtype
        if dtype not in ("BF16", "F32"):
            raise ValueError(f"Inkling 2 dense tensor {name} requires BF16 or F32, got {dtype}")
        data = self.raw(name)[selector]
        if dtype == "BF16":
            return (data.astype(np.uint32) << 16).view(np.float32)
        return np.asarray(data)

    def expert(self, layer: int, projection: str, expert: int) -> np.ndarray:
        prefix = f"layers.{layer}.mlp.experts.{projection}_weight"
        weight_info = self._info(prefix)
        scale_info = self._info(prefix + "_scale")
        layout = (weight_info.dtype, scale_info.dtype)
        if layout not in (("U8", "F8_E4M3"), ("F8_E4M3", "U8")):
            raise ValueError(f"unsupported Inkling 2 expert layout for {prefix}: {layout}")
        if (len(weight_info.shape) != 3 or len(scale_info.shape) != 3
                or weight_info.shape[:2] != scale_info.shape[:2]):
            raise ValueError(f"incompatible Inkling 2 expert bank shape for {prefix}")
        if not 0 <= expert < weight_info.shape[0]:
            raise IndexError(f"Inkling 2 expert index out of range: {expert}")
        values, scales = self.raw(prefix)[expert], self.raw(prefix + "_scale")[expert]
        if weight_info.dtype == "U8":
            amax = self.dense(prefix + "_amax", expert)
            denominator = self.dense(prefix + "_outer_scale_denominator", 0)
            return decode_nvfp4_expert(values, scales, amax=amax, denominator=denominator)
        return decode_mxfp8_expert(values, scales)
