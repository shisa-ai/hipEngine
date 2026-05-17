"""Explicit qwen35moe GGUF expert pack8 sidecar builder.

The sidecar is intentionally separate from normal GGUF materialization: runtime
callers must opt in to building/loading these files before future grouped MoE
kernels consume them.  The public ``LLM.generate`` hot path remains torch-free and
continues to work from raw GGUF expert tensors when no sidecar is requested.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from hipengine.loading.gguf import GGUFModelInfo, GGUFReader, GGUFTensorInfo
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.quant.gguf import GGMLQuantizationType, QK_K, dequantize_gguf_data, quant_layout, unpack_q4_k_scale_min
from hipengine.quant.gguf_q4_k import awq_pack8_shift_for_lane

EXPERT_SIDECAR_FORMAT = "hipengine.qwen35moe.gguf_expert_pack8.v1"
EXPERT_SIDECAR_VERSION = 1
EXPERT_SIDECAR_LAYOUT = "gguf_expert_pack8_v1"
EXPERT_SIDECAR_PACK = 8
Q4_K_BLOCK_BYTES = 144
Q5_K_BLOCK_BYTES = 176
Q6_K_BLOCK_BYTES = 210
Q4_Q5_GROUP_VALUES = 32
Q6_GROUP_VALUES = 16
QW_HIGH_NONE = "none"
QW_HIGH_Q5_BIT1 = "uint8_lane_bit1"
QW_HIGH_Q6_BIT2 = "uint16_lane_bit2"
MIN_KIND_NONE = "none"
MIN_KIND_SCALE_MIN = "scale_min"
_SUPPORTED_EXPERT_QTYPES = {
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
}
DEFAULT_EXPERT_SIDECAR_SLOTS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


@dataclass(frozen=True)
class GGUFExpertPackedTensor:
    """Host pack8 sidecar arrays for one rank-3 qwen35moe GGUF expert tensor.

    ``qweight_low`` packs the low four quant bits for eight adjacent output
    channels into one int32 using the same lane order as the existing Q4_K pack8
    GEMV layout.  Q5_K adds one high bit per lane in ``qweight_high``; Q6_K adds
    two high bits per lane.  ``scales`` stores precomputed FP32 scale terms per
    input group and output channel.  ``mins`` is present for Q4_K/Q5_K, which use
    GGML scale/min math, and absent for Q6_K.
    """

    tensor_name: str
    slot: str
    quant_key: str
    ggml_type: int
    shape: tuple[int, int, int]
    byte_shape: tuple[int, int, int]
    qweight_low: np.ndarray
    scales: np.ndarray
    qweight_high: np.ndarray | None = None
    mins: np.ndarray | None = None
    format: str = EXPERT_SIDECAR_FORMAT
    version: int = EXPERT_SIDECAR_VERSION

    @property
    def num_experts(self) -> int:
        return int(self.shape[0])

    @property
    def out_features(self) -> int:
        return int(self.shape[1])

    @property
    def in_features(self) -> int:
        return int(self.shape[2])

    @property
    def out_packed(self) -> int:
        return self.out_features // EXPERT_SIDECAR_PACK

    @property
    def qtype(self) -> GGMLQuantizationType:
        return GGMLQuantizationType(self.ggml_type)

    @property
    def qweight_high_kind(self) -> str:
        if self.qweight_high is None:
            return QW_HIGH_NONE
        if self.qtype == GGMLQuantizationType.Q5_K:
            return QW_HIGH_Q5_BIT1
        if self.qtype == GGMLQuantizationType.Q6_K:
            return QW_HIGH_Q6_BIT2
        raise ValueError(f"unexpected high-bit tensor for {self.qtype.name}")

    @property
    def min_kind(self) -> str:
        return MIN_KIND_NONE if self.mins is None else MIN_KIND_SCALE_MIN

    @property
    def nbytes(self) -> int:
        total = int(self.qweight_low.nbytes + self.scales.nbytes)
        if self.qweight_high is not None:
            total += int(self.qweight_high.nbytes)
        if self.mins is not None:
            total += int(self.mins.nbytes)
        return total

    def metadata(self) -> dict[str, object]:
        return {
            "format": self.format,
            "version": self.version,
            "layout": EXPERT_SIDECAR_LAYOUT,
            "tensor_name": self.tensor_name,
            "slot": self.slot,
            "quant_key": self.quant_key,
            "ggml_type": int(self.ggml_type),
            "ggml_type_name": self.qtype.name,
            "shape": list(self.shape),
            "byte_shape": list(self.byte_shape),
            "num_experts": self.num_experts,
            "out_features": self.out_features,
            "in_features": self.in_features,
            "out_packed": self.out_packed,
            "pack": EXPERT_SIDECAR_PACK,
            "qweight_low_dtype": str(self.qweight_low.dtype),
            "qweight_high_kind": self.qweight_high_kind,
            "scales_dtype": str(self.scales.dtype),
            "min_kind": self.min_kind,
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True)
class Qwen35MoeGGUFExpertLayerSidecar:
    """Packed sidecar tensors for selected qwen35moe expert slots in one layer."""

    model_path: Path
    layer_id: int
    tensors: Mapping[str, GGUFExpertPackedTensor]
    cache_paths: Mapping[str, Path]

    def tensor(self, slot: str) -> GGUFExpertPackedTensor:
        return self.tensors[slot]

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(self.tensors)

    @property
    def nbytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors.values())

    def metadata(self) -> dict[str, object]:
        return {
            "format": EXPERT_SIDECAR_FORMAT,
            "version": EXPERT_SIDECAR_VERSION,
            "layout": EXPERT_SIDECAR_LAYOUT,
            "model_path": str(self.model_path),
            "layer_id": self.layer_id,
            "slots": list(self.slots),
            "nbytes": self.nbytes,
            "tensors": {slot: tensor.metadata() for slot, tensor in self.tensors.items()},
            "cache_paths": {slot: str(path) for slot, path in self.cache_paths.items()},
        }


def pack_gguf_expert_tensor(
    raw_qweight: object,
    qtype: int | GGMLQuantizationType,
    *,
    tensor_name: str,
    slot: str = "",
) -> GGUFExpertPackedTensor:
    """Pack a rank-3 raw GGUF expert tensor into the sidecar layout."""

    qtype = GGMLQuantizationType(qtype)
    if qtype not in _SUPPORTED_EXPERT_QTYPES:
        supported = ", ".join(q.name for q in sorted(_SUPPORTED_EXPERT_QTYPES, key=int))
        raise ValueError(f"unsupported GGUF expert sidecar type {qtype.name}; expected one of: {supported}")
    raw = np.asarray(raw_qweight, dtype=np.uint8)
    if raw.ndim != 3:
        raise ValueError("raw_qweight must have GGUF expert byte shape [experts, out_features, bytes_per_row]")
    experts, out_features, bytes_per_row = (int(raw.shape[0]), int(raw.shape[1]), int(raw.shape[2]))
    if experts <= 0 or out_features <= 0 or bytes_per_row <= 0:
        raise ValueError("expert sidecar dimensions must be positive")
    if out_features % EXPERT_SIDECAR_PACK != 0:
        raise ValueError(f"out_features must be divisible by {EXPERT_SIDECAR_PACK}")
    layout = quant_layout(qtype)
    if bytes_per_row % layout.type_size != 0:
        raise ValueError(f"bytes_per_row {bytes_per_row} is not a multiple of {qtype.name} type size {layout.type_size}")
    blocks_per_row = bytes_per_row // layout.type_size
    in_features = blocks_per_row * QK_K
    byte_shape = (experts, out_features, bytes_per_row)
    shape = (experts, out_features, in_features)
    blocks = raw.reshape(experts, out_features, blocks_per_row, layout.type_size)

    if qtype == GGMLQuantizationType.Q4_K:
        qweight_low, scales, mins = _pack_q4_k_blocks(blocks)
        return GGUFExpertPackedTensor(
            tensor_name=tensor_name,
            slot=slot,
            quant_key="gguf_q4_k",
            ggml_type=int(qtype),
            shape=shape,
            byte_shape=byte_shape,
            qweight_low=qweight_low,
            scales=scales,
            mins=mins,
        )
    if qtype == GGMLQuantizationType.Q5_K:
        qweight_low, qweight_high, scales, mins = _pack_q5_k_blocks(blocks)
        return GGUFExpertPackedTensor(
            tensor_name=tensor_name,
            slot=slot,
            quant_key="gguf_q5_k",
            ggml_type=int(qtype),
            shape=shape,
            byte_shape=byte_shape,
            qweight_low=qweight_low,
            qweight_high=qweight_high,
            scales=scales,
            mins=mins,
        )
    qweight_low, qweight_high, scales = _pack_q6_k_blocks(blocks)
    return GGUFExpertPackedTensor(
        tensor_name=tensor_name,
        slot=slot,
        quant_key="gguf_q6_k",
        ggml_type=int(qtype),
        shape=shape,
        byte_shape=byte_shape,
        qweight_low=qweight_low,
        qweight_high=qweight_high,
        scales=scales,
    )


def dequantize_packed_expert_tensor(packed: GGUFExpertPackedTensor) -> np.ndarray:
    """CPU oracle dequantization from the sidecar layout to float32."""

    qtype = packed.qtype
    if qtype in (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K):
        return _dequantize_packed_q4_q5(packed)
    if qtype == GGMLQuantizationType.Q6_K:
        return _dequantize_packed_q6(packed)
    raise ValueError(f"unsupported packed expert qtype {qtype.name}")


def reference_dequantize_expert_tensor(raw_qweight: object, qtype: int | GGMLQuantizationType) -> np.ndarray:
    """Dequantize a rank-3 raw GGUF expert tensor with the generic CPU oracle."""

    raw = np.asarray(raw_qweight, dtype=np.uint8)
    if raw.ndim != 3:
        raise ValueError("raw_qweight must have GGUF expert byte shape [experts, out_features, bytes_per_row]")
    experts, out_features, _ = (int(raw.shape[0]), int(raw.shape[1]), int(raw.shape[2]))
    flat = dequantize_gguf_data(raw.reshape(experts * out_features, raw.shape[2]), qtype)
    return flat.reshape(experts, out_features, flat.shape[-1])


def default_expert_sidecar_cache_dir() -> Path:
    """Return the default explicit sidecar cache directory."""

    root = os.environ.get("HIPENGINE_GGUF_SIDECAR_CACHE")
    if root:
        return Path(root).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "hipengine" / "gguf_sidecars"


def expert_sidecar_cache_key(info: GGUFModelInfo, tensor: GGUFTensorInfo) -> str:
    """Stable cache key for one tensor sidecar, invalidated by file metadata."""

    stat = info.path.stat()
    payload = {
        "format": EXPERT_SIDECAR_FORMAT,
        "version": EXPERT_SIDECAR_VERSION,
        "path": str(info.path),
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "tensor_name": tensor.name,
        "ggml_type": tensor.ggml_type,
        "shape": tensor.shape,
        "byte_shape": tensor.byte_shape,
        "data_offset": tensor.data_offset,
        "nbytes": tensor.nbytes,
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(data).hexdigest()[:24]


def expert_sidecar_cache_path(
    info: GGUFModelInfo,
    tensor: GGUFTensorInfo,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    root = default_expert_sidecar_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
    safe_name = tensor.name.replace("/", "_").replace(".", "_")
    return root / EXPERT_SIDECAR_LAYOUT / f"{safe_name}-{expert_sidecar_cache_key(info, tensor)}.npz"


def build_or_load_qwen35moe_expert_sidecar(
    reader_or_path: GGUFReader | str | Path,
    *,
    layer_id: int,
    slots: Iterable[str] = DEFAULT_EXPERT_SIDECAR_SLOTS,
    cache_dir: str | Path | None = None,
    overwrite: bool = False,
    require_cached: bool = False,
) -> Qwen35MoeGGUFExpertLayerSidecar:
    """Explicitly build or load cached qwen35moe GGUF expert sidecars.

    This function is opt-in by design.  It may create large ``.npz`` files under
    the requested cache directory, but normal model loading does not call it.
    """

    reader = reader_or_path if isinstance(reader_or_path, GGUFReader) else GGUFReader(reader_or_path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    if not model_map.config.is_moe:
        raise ValueError("qwen35moe expert sidecars require a qwen35moe GGUF model")
    layer = model_map.layer(layer_id)
    tensors: dict[str, GGUFExpertPackedTensor] = {}
    cache_paths: dict[str, Path] = {}
    for slot in tuple(slots):
        tensor = layer.tensor(slot)
        path = expert_sidecar_cache_path(reader.info, tensor, cache_dir=cache_dir)
        if path.exists() and not overwrite:
            packed = load_packed_expert_tensor(path)
        else:
            if require_cached:
                raise FileNotFoundError(f"missing cached GGUF expert sidecar for {tensor.name}: {path}")
            packed = build_packed_expert_tensor_from_reader(reader, tensor, slot=slot)
            save_packed_expert_tensor(path, packed)
        tensors[slot] = packed
        cache_paths[slot] = path
    return Qwen35MoeGGUFExpertLayerSidecar(
        model_path=reader.info.path,
        layer_id=int(layer_id),
        tensors=MappingProxyType(tensors),
        cache_paths=MappingProxyType(cache_paths),
    )


def build_packed_expert_tensor_from_reader(
    reader: GGUFReader,
    tensor: GGUFTensorInfo,
    *,
    slot: str = "",
) -> GGUFExpertPackedTensor:
    if len(tensor.shape) != 3 or len(tensor.byte_shape) != 3:
        raise ValueError(f"GGUF expert tensor {tensor.name!r} must be rank-3, got {tensor.shape}")
    return pack_gguf_expert_tensor(
        np.ascontiguousarray(reader.tensor_data(tensor.name)),
        tensor.ggml_type,
        tensor_name=tensor.name,
        slot=slot,
    )


def save_packed_expert_tensor(path: str | Path, packed: GGUFExpertPackedTensor) -> Path:
    """Write one packed tensor sidecar atomically as ``.npz``."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(packed.metadata(), sort_keys=True)
    arrays: dict[str, object] = {
        "metadata_json": np.asarray(metadata),
        "qweight_low": packed.qweight_low,
        "scales": packed.scales,
    }
    if packed.qweight_high is not None:
        arrays["qweight_high"] = packed.qweight_high
    if packed.mins is not None:
        arrays["mins"] = packed.mins
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, target)
    return target


def load_packed_expert_tensor(path: str | Path) -> GGUFExpertPackedTensor:
    """Load and validate one packed tensor sidecar from ``.npz``."""

    source = Path(path).expanduser()
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("format") != EXPERT_SIDECAR_FORMAT or int(metadata.get("version", -1)) != EXPERT_SIDECAR_VERSION:
            raise ValueError(f"unsupported GGUF expert sidecar format in {source}: {metadata!r}")
        qweight_high = np.ascontiguousarray(data["qweight_high"]) if "qweight_high" in data.files else None
        mins = np.ascontiguousarray(data["mins"]) if "mins" in data.files else None
        return GGUFExpertPackedTensor(
            tensor_name=str(metadata["tensor_name"]),
            slot=str(metadata.get("slot", "")),
            quant_key=str(metadata["quant_key"]),
            ggml_type=int(metadata["ggml_type"]),
            shape=tuple(int(x) for x in metadata["shape"]),  # type: ignore[arg-type]
            byte_shape=tuple(int(x) for x in metadata["byte_shape"]),  # type: ignore[arg-type]
            qweight_low=np.ascontiguousarray(data["qweight_low"]),
            qweight_high=qweight_high,
            scales=np.ascontiguousarray(data["scales"]),
            mins=mins,
        )


def _pack_q4_k_blocks(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    experts, out_features, blocks_per_row, block_bytes = blocks.shape
    if block_bytes != Q4_K_BLOCK_BYTES:
        raise ValueError(f"Q4_K blocks must be {Q4_K_BLOCK_BYTES} bytes, got {block_bytes}")
    d = blocks[..., 0:2].copy().view(np.float16).astype(np.float32).reshape(experts, out_features, blocks_per_row)
    dmin = blocks[..., 2:4].copy().view(np.float16).astype(np.float32).reshape(experts, out_features, blocks_per_row)
    sc, minv = unpack_q4_k_scale_min(blocks[..., 4:16].reshape(-1, 12))
    sc = sc.reshape(experts, out_features, blocks_per_row, 8)
    minv = minv.reshape(experts, out_features, blocks_per_row, 8)
    scales = _transpose_group_terms(d[..., None] * sc.astype(np.float32))
    mins = _transpose_group_terms(dmin[..., None] * minv.astype(np.float32))
    q_values = _q4_low_values(blocks[..., 16:144])
    qweight_low = _pack_low4_lanes(q_values)
    return qweight_low, scales, mins


def _pack_q5_k_blocks(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    experts, out_features, blocks_per_row, block_bytes = blocks.shape
    if block_bytes != Q5_K_BLOCK_BYTES:
        raise ValueError(f"Q5_K blocks must be {Q5_K_BLOCK_BYTES} bytes, got {block_bytes}")
    d = blocks[..., 0:2].copy().view(np.float16).astype(np.float32).reshape(experts, out_features, blocks_per_row)
    dmin = blocks[..., 2:4].copy().view(np.float16).astype(np.float32).reshape(experts, out_features, blocks_per_row)
    sc, minv = unpack_q4_k_scale_min(blocks[..., 4:16].reshape(-1, 12))
    sc = sc.reshape(experts, out_features, blocks_per_row, 8)
    minv = minv.reshape(experts, out_features, blocks_per_row, 8)
    scales = _transpose_group_terms(d[..., None] * sc.astype(np.float32))
    mins = _transpose_group_terms(dmin[..., None] * minv.astype(np.float32))
    q_low = _q4_low_values(blocks[..., 48:176])
    q_high = _q5_high_values(blocks[..., 16:48])
    return _pack_low4_lanes(q_low), _pack_high1_lanes(q_high), scales, mins


def _pack_q6_k_blocks(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    experts, out_features, blocks_per_row, block_bytes = blocks.shape
    if block_bytes != Q6_K_BLOCK_BYTES:
        raise ValueError(f"Q6_K blocks must be {Q6_K_BLOCK_BYTES} bytes, got {block_bytes}")
    q_low = _q6_low_values(blocks[..., 0:128])
    q_high = _q6_high_values(blocks[..., 128:192])
    scales_i8 = blocks[..., 192:208].copy().view(np.int8).astype(np.float32)
    d = blocks[..., 208:210].copy().view(np.float16).astype(np.float32).reshape(experts, out_features, blocks_per_row)
    scales = _transpose_group_terms(d[..., None] * scales_i8.reshape(experts, out_features, blocks_per_row, 16))
    return _pack_low4_lanes(q_low), _pack_high2_lanes(q_high), scales


def _transpose_group_terms(values: np.ndarray) -> np.ndarray:
    # [experts, out_features, blocks, groups_per_block] -> [experts, groups, out_features]
    experts, out_features, blocks_per_row, groups_per_block = values.shape
    return np.ascontiguousarray(values.transpose(0, 2, 3, 1).reshape(experts, blocks_per_row * groups_per_block, out_features))


def _q4_low_values(raw_qs: np.ndarray) -> np.ndarray:
    experts, out_features, blocks_per_row, _ = raw_qs.shape
    qs = raw_qs.reshape(experts, out_features, blocks_per_row, 4, 1, 32)
    qs = (qs >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 1, 1, 2, 1)) & np.uint8(0x0F)
    return np.ascontiguousarray(qs.reshape(experts, out_features, blocks_per_row * QK_K).astype(np.uint8))


def _q5_high_values(raw_qh: np.ndarray) -> np.ndarray:
    experts, out_features, blocks_per_row, _ = raw_qh.shape
    qh = raw_qh.reshape(experts, out_features, blocks_per_row, 1, 1, 32)
    qh = (qh >> np.arange(8, dtype=np.uint8).reshape(1, 1, 1, 1, 8, 1)) & np.uint8(0x01)
    return np.ascontiguousarray(qh.reshape(experts, out_features, blocks_per_row * QK_K).astype(np.uint8))


def _q6_low_values(raw_ql: np.ndarray) -> np.ndarray:
    experts, out_features, blocks_per_row, _ = raw_ql.shape
    ql = raw_ql.reshape(experts, out_features, blocks_per_row, 2, 1, 64)
    ql = (ql >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 1, 1, 2, 1)) & np.uint8(0x0F)
    return np.ascontiguousarray(ql.reshape(experts, out_features, blocks_per_row * QK_K).astype(np.uint8))


def _q6_high_values(raw_qh: np.ndarray) -> np.ndarray:
    experts, out_features, blocks_per_row, _ = raw_qh.shape
    qh = raw_qh.reshape(experts, out_features, blocks_per_row, 2, 1, 32)
    qh = (qh >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 1, 1, 4, 1)) & np.uint8(0x03)
    return np.ascontiguousarray(qh.reshape(experts, out_features, blocks_per_row * QK_K).astype(np.uint8))


def _pack_low4_lanes(q_values: np.ndarray) -> np.ndarray:
    experts, out_features, in_features = q_values.shape
    out_packed = out_features // EXPERT_SIDECAR_PACK
    packed = np.zeros((experts, out_packed, in_features), dtype=np.uint32)
    q_values = q_values.astype(np.uint32, copy=False)
    for lane in range(EXPERT_SIDECAR_PACK):
        packed |= (q_values[:, lane::EXPERT_SIDECAR_PACK, :] & np.uint32(0x0F)) << np.uint32(awq_pack8_shift_for_lane(lane))
    return np.ascontiguousarray(packed.view(np.int32))


def _pack_high1_lanes(q_high: np.ndarray) -> np.ndarray:
    experts, out_features, in_features = q_high.shape
    out_packed = out_features // EXPERT_SIDECAR_PACK
    packed = np.zeros((experts, out_packed, in_features), dtype=np.uint8)
    for lane in range(EXPERT_SIDECAR_PACK):
        packed |= (q_high[:, lane::EXPERT_SIDECAR_PACK, :] & np.uint8(0x01)) << np.uint8(lane)
    return np.ascontiguousarray(packed)


def _pack_high2_lanes(q_high: np.ndarray) -> np.ndarray:
    experts, out_features, in_features = q_high.shape
    out_packed = out_features // EXPERT_SIDECAR_PACK
    packed = np.zeros((experts, out_packed, in_features), dtype=np.uint16)
    q_high_u16 = q_high.astype(np.uint16, copy=False)
    for lane in range(EXPERT_SIDECAR_PACK):
        packed |= (q_high_u16[:, lane::EXPERT_SIDECAR_PACK, :] & np.uint16(0x03)) << np.uint16(2 * lane)
    return np.ascontiguousarray(packed)


def _unpack_low_lane(qweight_low: np.ndarray, lane: int) -> np.ndarray:
    low_u32 = qweight_low.view(np.uint32)
    return ((low_u32 >> np.uint32(awq_pack8_shift_for_lane(lane))) & np.uint32(0x0F)).astype(np.float32)


def _dequantize_packed_q4_q5(packed: GGUFExpertPackedTensor) -> np.ndarray:
    if packed.mins is None:
        raise ValueError("Q4_K/Q5_K packed expert tensor requires mins")
    out = np.empty(packed.shape, dtype=np.float32)
    high = packed.qweight_high
    group = np.arange(packed.in_features, dtype=np.int64) // Q4_Q5_GROUP_VALUES
    scale_by_k = packed.scales[:, group, :]
    min_by_k = packed.mins[:, group, :]
    for lane in range(EXPERT_SIDECAR_PACK):
        q = _unpack_low_lane(packed.qweight_low, lane)
        if high is not None:
            q += (((high >> np.uint8(lane)) & np.uint8(0x01)).astype(np.float32) * np.float32(16.0))
        lane_scales = scale_by_k[:, :, lane::EXPERT_SIDECAR_PACK].transpose(0, 2, 1)
        lane_mins = min_by_k[:, :, lane::EXPERT_SIDECAR_PACK].transpose(0, 2, 1)
        out[:, lane::EXPERT_SIDECAR_PACK, :] = q * lane_scales - lane_mins
    return out


def _dequantize_packed_q6(packed: GGUFExpertPackedTensor) -> np.ndarray:
    if packed.qweight_high is None:
        raise ValueError("Q6_K packed expert tensor requires high-bit tensor")
    out = np.empty(packed.shape, dtype=np.float32)
    group = np.arange(packed.in_features, dtype=np.int64) // Q6_GROUP_VALUES
    scale_by_k = packed.scales[:, group, :]
    for lane in range(EXPERT_SIDECAR_PACK):
        low = _unpack_low_lane(packed.qweight_low, lane).astype(np.int16)
        high = ((packed.qweight_high >> np.uint16(2 * lane)) & np.uint16(0x03)).astype(np.int16)
        q = (low | (high << np.int16(4))).astype(np.int16) - np.int16(32)
        lane_scales = scale_by_k[:, :, lane::EXPERT_SIDECAR_PACK].transpose(0, 2, 1)
        out[:, lane::EXPERT_SIDECAR_PACK, :] = q.astype(np.float32) * lane_scales
    return out


__all__ = [
    "DEFAULT_EXPERT_SIDECAR_SLOTS",
    "EXPERT_SIDECAR_FORMAT",
    "EXPERT_SIDECAR_LAYOUT",
    "EXPERT_SIDECAR_PACK",
    "EXPERT_SIDECAR_VERSION",
    "GGUFExpertPackedTensor",
    "Qwen35MoeGGUFExpertLayerSidecar",
    "build_or_load_qwen35moe_expert_sidecar",
    "build_packed_expert_tensor_from_reader",
    "default_expert_sidecar_cache_dir",
    "dequantize_packed_expert_tensor",
    "expert_sidecar_cache_key",
    "expert_sidecar_cache_path",
    "load_packed_expert_tensor",
    "pack_gguf_expert_tensor",
    "reference_dequantize_expert_tensor",
    "save_packed_expert_tensor",
]
