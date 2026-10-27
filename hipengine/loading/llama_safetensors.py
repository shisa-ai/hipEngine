"""Torch-free materialization for dense Llama safetensors checkpoints.

Two storage formats are read:

* ``BF16`` / ``F16`` / ``F32`` plain tensors, converted to BF16 on the host.
* ``F8_E4M3`` weights paired with a ``weight_scale`` tensor, as written by
  ``compressed-tensors`` for per-output-channel float-quantized checkpoints.

The float8 path dequantizes once at load (``weight * scale`` in FP32, rounded to
BF16) and keeps BF16 resident. That is a storage-format conversion, not a
requantization: no second rounding of the scale product beyond the BF16 the
runtime already uses, and the source shards are never modified.

Nothing here imports torch. FP8 E4M3FN decoding uses a 256-entry lookup table
validated bit-exactly against ``torch.float8_e4m3fn``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.core.tensor import Tensor
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    float_array_to_bf16_bits,
    load_host_array_to_device_as_dtype,
)
from hipengine.loading.safetensors import (
    TensorInfo,
    WeightIndex,
    read_tensor_storage_bytes,
)
from hipengine.models.llama import (
    LlamaCapabilityMiss,
    LlamaSpec,
    parse_llama_spec,
)

#: Source dtypes this loader can convert to a BF16 runtime tensor.
_PLAIN_SOURCE_DTYPES = frozenset({"BF16", "F16", "F32"})
_FP8_SOURCE_DTYPE = "F8_E4M3"
_SCALE_SUFFIX = "weight_scale"

ProgressFn = Callable[[str, int, int], None]


def _fp8_e4m3fn_decode_table() -> np.ndarray:
    """Return the 256-entry E4M3FN code-to-float table (bias 7, NaN at 0x7F/0xFF)."""

    codes = np.arange(256, dtype=np.uint16)
    sign = np.where((codes >> 7) & 1, -1.0, 1.0).astype(np.float32)
    exponent = ((codes >> 3) & 0xF).astype(np.int32)
    mantissa = (codes & 0x7).astype(np.float32)
    magnitude = np.where(
        exponent == 0,
        mantissa / 8.0 * np.float32(2.0**-6),
        (1.0 + mantissa / 8.0) * np.power(2.0, (exponent - 7).astype(np.float32)),
    )
    values = sign * magnitude.astype(np.float32)
    values[(exponent == 0xF) & (mantissa == 7)] = np.nan
    return np.ascontiguousarray(values, dtype=np.float32)


_FP8_DECODE = _fp8_e4m3fn_decode_table()


def fp8_e4m3fn_to_float32(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Decode raw E4M3FN storage bytes to FP32."""

    codes = np.frombuffer(payload, dtype=np.uint8)
    return _FP8_DECODE[codes].reshape(shape)


def _plain_source_to_float32(info: TensorInfo) -> np.ndarray:
    payload = read_tensor_storage_bytes(info)
    if info.dtype == "BF16":
        bits = np.frombuffer(payload, dtype=np.uint16).astype(np.uint32)
        return (bits << np.uint32(16)).view(np.float32).reshape(info.shape)
    if info.dtype == "F16":
        return np.frombuffer(payload, dtype=np.float16).astype(np.float32).reshape(info.shape)
    if info.dtype == "F32":
        return np.frombuffer(payload, dtype=np.float32).reshape(info.shape)
    raise LlamaCapabilityMiss(
        f"tensor {info.name!r} has source dtype {info.dtype!r}; the dense Llama loader "
        f"reads {sorted(_PLAIN_SOURCE_DTYPES)} and {_FP8_SOURCE_DTYPE} only"
    )


@dataclass(frozen=True)
class LlamaTensorSource:
    """One runtime tensor plus how it is derived from checkpoint storage.

    ``runtime_dtype`` is the resident dtype. Linear weights and embeddings are
    BF16; RMSNorm weights are FP32 because the registered norm kernel reads a
    plain FP32 multiplier, and a per-row scale is worth more precision than the
    BF16 storage it arrived in.
    """

    runtime_name: str
    info: TensorInfo
    scale: TensorInfo | None
    runtime_dtype: DType = DType.BF16

    @property
    def storage(self) -> str:
        if self.scale is not None:
            return "f8_e4m3_per_channel"
        return self.info.dtype.lower()

    @property
    def is_norm_weight(self) -> bool:
        return self.runtime_dtype == DType.FP32


@dataclass(frozen=True)
class LlamaLayerWeights:
    attn_norm: Tensor
    q_proj: Tensor
    k_proj: Tensor
    v_proj: Tensor
    o_proj: Tensor
    ffn_norm: Tensor
    gate_proj: Tensor
    up_proj: Tensor
    down_proj: Tensor


@dataclass(frozen=True)
class LlamaLoadedModel:
    """Validated spec, resident BF16 weights, and load provenance."""

    spec: LlamaSpec
    model_path: Path
    embed_tokens: Tensor
    final_norm: Tensor
    lm_head: Tensor
    layers: tuple[LlamaLayerWeights, ...]
    storage_formats: dict[str, str]
    device_allocations: tuple[DeviceTensorAllocation, ...]
    checkpoint_sha256: str

    @property
    def hidden_size(self) -> int:
        return self.spec.hidden_size

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for allocation in reversed(self.device_allocations):
            allocation.free(runtime=runtime)


def load_llama_spec(index: WeightIndex) -> LlamaSpec:
    return parse_llama_spec(index.config)


def _scale_name_for(weight_name: str) -> str:
    stem, _, leaf = weight_name.rpartition(".")
    if leaf != "weight":
        raise LlamaCapabilityMiss(f"cannot derive a scale name for {weight_name!r}")
    return f"{stem}.{_SCALE_SUFFIX}"


def plan_llama_tensor_sources(index: WeightIndex, spec: LlamaSpec) -> tuple[LlamaTensorSource, ...]:
    """Resolve every runtime tensor to its checkpoint storage and validate it.

    Validation is shape-and-dtype based. A checkpoint that declares a decoder this
    implementation cannot execute fails here with a named capability miss rather
    than loading a partially correct model.
    """

    expected = spec.expected_weight_shapes(lm_head=not spec.tie_word_embeddings)
    present = set(index.tensors)
    if spec.tie_word_embeddings:
        present.discard("lm_head.weight")

    missing = sorted(name for name in expected if name not in index.tensors)
    if missing:
        preview = ", ".join(missing[:8])
        more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        raise LlamaCapabilityMiss(
            f"checkpoint is missing {len(missing)} dense Llama tensor(s): {preview}{more}"
        )

    unexpected_layernorm_bias = sorted(
        name
        for name in present
        if name.endswith("layernorm.bias") or name.endswith("norm.bias")
    )
    if unexpected_layernorm_bias:
        raise LlamaCapabilityMiss(
            "checkpoint carries RMSNorm biases, which this decoder does not apply: "
            + ", ".join(unexpected_layernorm_bias[:4])
        )

    sources: list[LlamaTensorSource] = []
    for name, shape in expected.items():
        info = index.tensors[name]
        if tuple(info.shape) != tuple(shape):
            raise LlamaCapabilityMiss(
                f"tensor {name!r} has shape {info.shape}, config declares {shape}"
            )
        if info.dtype in _PLAIN_SOURCE_DTYPES:
            sources.append(
                LlamaTensorSource(
                    runtime_name=name,
                    info=info,
                    scale=None,
                    runtime_dtype=DType.FP32 if len(shape) == 1 else DType.BF16,
                )
            )
            continue
        if info.dtype != _FP8_SOURCE_DTYPE:
            raise LlamaCapabilityMiss(
                f"tensor {name!r} has source dtype {info.dtype!r}; the dense Llama loader "
                f"reads {sorted(_PLAIN_SOURCE_DTYPES)} and {_FP8_SOURCE_DTYPE} only"
            )
        scale_name = _scale_name_for(name)
        scale_info = index.tensors.get(scale_name)
        if scale_info is None:
            raise LlamaCapabilityMiss(
                f"float8 tensor {name!r} has no {scale_name!r}; per-channel float quantization "
                "requires one scale per output channel"
            )
        expected_scale = (shape[0], 1)
        if tuple(scale_info.shape) not in {expected_scale, (shape[0],)}:
            raise LlamaCapabilityMiss(
                f"float8 scale {scale_name!r} has shape {scale_info.shape}; expected "
                f"one scale per output channel {expected_scale}"
            )
        if scale_info.dtype not in _PLAIN_SOURCE_DTYPES:
            raise LlamaCapabilityMiss(
                f"float8 scale {scale_name!r} has dtype {scale_info.dtype!r}; "
                "expected BF16, F16, or F32"
            )
        sources.append(
            LlamaTensorSource(
                runtime_name=name,
                info=info,
                scale=scale_info,
                runtime_dtype=DType.BF16,
            )
        )
    return tuple(sources)


def _runtime_host_array(source: LlamaTensorSource) -> np.ndarray:
    """Produce the contiguous host array for one runtime tensor."""

    if source.scale is None:
        values = _plain_source_to_float32(source.info)
        if source.is_norm_weight:
            return np.ascontiguousarray(values, dtype=np.float32).reshape(source.info.shape)
        return float_array_to_bf16_bits(values).reshape(source.info.shape)

    weights = fp8_e4m3fn_to_float32(
        read_tensor_storage_bytes(source.info),
        source.info.shape,
    )
    scale = _plain_source_to_float32(source.scale).reshape(-1)
    if scale.shape[0] != weights.shape[0]:
        raise LlamaCapabilityMiss(
            f"float8 scale for {source.info.name!r} has {scale.shape[0]} entries; "
            f"expected {weights.shape[0]}"
        )
    if not np.all(np.isfinite(scale)) or np.any(scale == 0.0):
        raise LlamaCapabilityMiss(
            f"float8 scale for {source.info.name!r} contains non-finite or zero values"
        )
    dequantized = weights.astype(np.float32) * scale[:, None].astype(np.float32)
    return float_array_to_bf16_bits(dequantized).reshape(source.info.shape)

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def llama_checkpoint_sha256(index: WeightIndex) -> str:
    """Digest the artifact identity: config bytes plus every shard's bytes.

    A multi-shard checkpoint has no single file hash, so the binding covers the
    shard names, sizes, and contents together with ``config.json``. The DMS
    sidecar fails closed on any change to this digest.
    """

    digest = hashlib.sha256()
    digest.update(b"hipengine-llama-checkpoint-v1\n")
    config_bytes = json.dumps(index.config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest.update(f"config:{len(config_bytes)}:".encode("ascii"))
    digest.update(config_bytes)
    for shard in index.shards:
        digest.update(f"\nshard:{shard.name}:{shard.stat().st_size}:".encode("ascii"))
        digest.update(_sha256_file(shard).encode("ascii"))
    return digest.hexdigest()


def materialize_llama_weights(
    index: WeightIndex,
    spec: LlamaSpec,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
    progress: ProgressFn | None = None,
    sources: tuple[LlamaTensorSource, ...] | None = None,
) -> LlamaLoadedModel:
    """Stream every checkpoint tensor to BF16 device memory.

    Tensors are converted and uploaded one at a time, so peak host staging is one
    tensor's FP32 image rather than the whole checkpoint.
    """

    planned = sources or plan_llama_tensor_sources(index, spec)
    allocations: dict[str, DeviceTensorAllocation] = {}
    storage_formats: dict[str, str] = {}
    total = len(planned)
    try:
        for position, source in enumerate(planned, start=1):
            host = _runtime_host_array(source)
            allocation = load_host_array_to_device_as_dtype(
                source.runtime_name,
                host,
                source.runtime_dtype,
                source_dtype="F32" if source.is_norm_weight else "BF16",
                device=device,
                runtime=runtime,
            )
            allocations[source.runtime_name] = allocation
            storage_formats[source.runtime_name] = source.storage
            del host
            if progress is not None:
                progress(source.runtime_name, position, total)
    except Exception:
        for allocation in reversed(tuple(allocations.values())):
            allocation.free(runtime=runtime)
        raise

    def tensor(name: str) -> Tensor:
        return allocations[name].tensor

    lm_head_name = "lm_head.weight" if not spec.tie_word_embeddings else "model.embed_tokens.weight"
    layers = tuple(
        LlamaLayerWeights(
            attn_norm=tensor(f"model.layers.{layer}.input_layernorm.weight"),
            q_proj=tensor(f"model.layers.{layer}.self_attn.q_proj.weight"),
            k_proj=tensor(f"model.layers.{layer}.self_attn.k_proj.weight"),
            v_proj=tensor(f"model.layers.{layer}.self_attn.v_proj.weight"),
            o_proj=tensor(f"model.layers.{layer}.self_attn.o_proj.weight"),
            ffn_norm=tensor(f"model.layers.{layer}.post_attention_layernorm.weight"),
            gate_proj=tensor(f"model.layers.{layer}.mlp.gate_proj.weight"),
            up_proj=tensor(f"model.layers.{layer}.mlp.up_proj.weight"),
            down_proj=tensor(f"model.layers.{layer}.mlp.down_proj.weight"),
        )
        for layer in range(spec.num_hidden_layers)
    )
    return LlamaLoadedModel(
        spec=spec,
        model_path=index.model_path,
        embed_tokens=tensor("model.embed_tokens.weight"),
        final_norm=tensor("model.norm.weight"),
        lm_head=tensor(lm_head_name),
        layers=layers,
        storage_formats=storage_formats,
        device_allocations=tuple(allocations.values()),
        checkpoint_sha256=llama_checkpoint_sha256(index),
    )


def llama_load_provenance(loaded: LlamaLoadedModel, index: WeightIndex) -> dict[str, Any]:
    """Load-time provenance recorded with DMS captures and quality evidence."""

    formats = sorted(set(loaded.storage_formats.values()))
    return {
        "runtime_quant": "bf16",
        "source_storage_formats": formats,
        "source_dtypes": sorted({info.dtype for info in index.tensors.values()}),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "config": {
            "hidden_size": loaded.spec.hidden_size,
            "num_hidden_layers": loaded.spec.num_hidden_layers,
            "num_attention_heads": loaded.spec.num_attention_heads,
            "num_key_value_heads": loaded.spec.num_key_value_heads,
            "head_dim": loaded.spec.head_dim,
            "intermediate_size": loaded.spec.intermediate_size,
            "vocab_size": loaded.spec.vocab_size,
            "rms_norm_eps": loaded.spec.rms_norm_eps,
            "rope_theta": loaded.spec.rope_theta,
            "tie_word_embeddings": loaded.spec.tie_word_embeddings,
        },
    }


__all__ = [
    "LlamaLayerWeights",
    "LlamaLoadedModel",
    "LlamaTensorSource",
    "fp8_e4m3fn_to_float32",
    "llama_checkpoint_sha256",
    "llama_load_provenance",
    "load_llama_spec",
    "materialize_llama_weights",
    "plan_llama_tensor_sources",
]
