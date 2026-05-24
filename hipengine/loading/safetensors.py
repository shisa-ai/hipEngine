"""Torch-free Hugging Face safetensors/config discovery helpers."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hipengine.loading.hf_cache import resolve_model_path

_DTYPE_NBYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


@dataclass(frozen=True)
class TensorInfo:
    """Metadata for one tensor inside a safetensors shard."""

    name: str
    shard_path: Path
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int | None:
        itemsize = _DTYPE_NBYTES.get(self.dtype)
        if itemsize is None:
            return None
        count = 1
        for dim in self.shape:
            count *= dim
        return count * itemsize


@dataclass(frozen=True)
class WeightIndex:
    """Resolved safetensors shards and tensor metadata for a model directory."""

    model_path: Path
    config: dict[str, Any]
    tensors: dict[str, TensorInfo]
    shards: tuple[Path, ...]

    def require(self, names: Iterable[str]) -> tuple[TensorInfo, ...]:
        missing = [name for name in names if name not in self.tensors]
        if missing:
            preview = ", ".join(missing[:8])
            more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
            raise MissingTensorError(f"missing required tensors: {preview}{more}")
        return tuple(self.tensors[name] for name in names)

    def names_with_prefix(self, prefix: str) -> tuple[str, ...]:
        return tuple(name for name in sorted(self.tensors) if name.startswith(prefix))


class MissingConfigError(FileNotFoundError):
    pass


class MissingWeightsError(FileNotFoundError):
    pass


class MissingTensorError(KeyError):
    pass


def read_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.exists():
        raise MissingConfigError(f"config.json not found under {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    return data


def discover_safetensor_shards(model_path: str | Path) -> tuple[Path, ...]:
    path = Path(model_path)
    if path.is_file() and path.suffix == ".safetensors":
        return (path,)
    if not path.is_dir():
        raise MissingWeightsError(f"model path is not a directory or safetensors file: {path}")

    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors index weight_map: {index_path}")
        return tuple(sorted({(path / str(shard)).resolve() for shard in weight_map.values()}))

    shards = tuple(sorted(path.glob("*.safetensors")))
    if not shards:
        raise MissingWeightsError(f"no .safetensors files found under {path}")
    return tuple(shard.resolve() for shard in shards)


def _parse_safetensors_header(shard: Path) -> dict[str, dict]:
    """Parse the safetensors JSON header without opening the full file.

    Returns ``{tensor_name: {"dtype": str, "shape": list[int], "data_offsets": [int, int]}}``,
    with the ``__metadata__`` key (if present) stripped.

    This is **O(header_size)** regardless of the data payload size — ~0.2 s for
    a 12 MB / 94 K-tensor header vs 120 s+ via ``safe_open`` + per-tensor
    ``get_slice``.
    """

    with open(shard, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header_bytes = fh.read(header_len)
    header = json.loads(header_bytes)
    header.pop("__metadata__", None)
    return header


def load_weight_index(model_path: str | Path) -> WeightIndex:
    path = resolve_model_path(model_path)
    model_dir = path if path.is_dir() else path.parent
    config = read_config(model_dir)
    shards = discover_safetensor_shards(path)
    tensors: dict[str, TensorInfo] = {}
    for shard in shards:
        if not shard.exists():
            raise MissingWeightsError(f"safetensors shard not found: {shard}")
        header = _parse_safetensors_header(shard)
        for name, meta in header.items():
            if name in tensors:
                raise ValueError(f"duplicate tensor {name!r} in {shard} and {tensors[name].shard_path}")
            tensors[name] = TensorInfo(
                name=name,
                shard_path=shard,
                dtype=str(meta["dtype"]),
                shape=tuple(int(d) for d in meta["shape"]),
            )
    return WeightIndex(
        model_path=model_dir.resolve(),
        config=config,
        tensors=dict(sorted(tensors.items())),
        shards=shards,
    )
