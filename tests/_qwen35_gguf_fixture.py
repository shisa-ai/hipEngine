"""Tiny real-GGUF writer for Qwen3.5 admission/loader/runner CPU integration tests.

The U1 review repairs must be exercised through the real loader and runner
entry points, not only isolated preflight helpers.  This module writes a
minimal but *structurally real* qwen35 GGUF file (header, metadata, tensor
descriptor table, aligned payload blob) that ``scan_gguf`` /
``build_qwen35_gguf_tensor_map`` / ``materialize_qwen35_gguf_weights`` accept
on a CPU-only host.

Block payloads reuse the byte-exact helpers in ``tests/_gguf_synthetic_weights.py``
(Q4_K/Q5_K/Q6_K/Q8_0) plus a valid all-zero IQ4_XS row builder (136-byte blocks,
decode d=0).  Shapes follow ``required_qwen35_gguf_tensor_names`` /
``_shape_errors`` for a linear-attention Qwen3.5 dense model.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hipengine.quant.gguf import GGMLQuantizationType, LlamaFileType
from tests._gguf_synthetic_weights import (
    make_q4_k_weight,
    make_q6_k_weight,
    make_q8_0_weight,
)

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# Linear-attention Qwen3.5 geometry shared by every fixture writer here.
FIXTURE_HIDDEN = 256
FIXTURE_VOCAB = 64
FIXTURE_FFN = 512
FIXTURE_SSM_INNER = 64
FIXTURE_SSM_GROUP = 2
FIXTURE_SSM_STATE = 32
FIXTURE_SSM_CONV_KERNEL = 4
FIXTURE_SSM_TIME_STEP_RANK = 2
FIXTURE_HEADS = 2
FIXTURE_HEADS_KV = 1
FIXTURE_KEY_LENGTH = 64
FIXTURE_VALUE_LENGTH = 64
FIXTURE_QKV_WIDTH = 2 * FIXTURE_SSM_GROUP * FIXTURE_SSM_STATE + FIXTURE_SSM_INNER


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_scalar(value_type: int, value: object) -> bytes:
    if value_type == 4:  # UINT32
        return struct.pack("<I", int(value))
    if value_type == 5:  # INT32
        return struct.pack("<i", int(value))
    if value_type == 6:  # FLOAT32
        return struct.pack("<f", float(value))
    if value_type == 10:  # UINT64
        return struct.pack("<Q", int(value))
    raise ValueError(f"unsupported GGUF scalar type {value_type}")


def _gguf_value(value_type: int, value: object) -> bytes:
    if value_type == 8:  # STRING
        return _gguf_string(str(value))
    return _gguf_scalar(value_type, value)


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


def iq4_xs_row_bytes(in_features: int) -> int:
    """Byte length of one IQ4_XS row (136-byte blocks per 256 values)."""

    if in_features % 256:
        raise ValueError("IQ4_XS fixture rows require in_features % 256 == 0")
    return in_features // 256 * 136


def payload_bytes(
    shape: Sequence[int],
    qtype: GGMLQuantizationType,
) -> bytes:
    """Return a valid GGUF payload for one fixture tensor."""

    if qtype == GGMLQuantizationType.F32:
        elements = 1
        for dim in shape:
            elements *= int(dim)
        return (
            (np.arange(elements, dtype=np.float32) * 0.125 - 1.0)
            .reshape(tuple(int(dim) for dim in shape))
            .tobytes()
        )
    if qtype == GGMLQuantizationType.F16:
        elements = 1
        for dim in shape:
            elements *= int(dim)
        return np.zeros(elements, dtype=np.float16).tobytes()
    if qtype == GGMLQuantizationType.BF16:
        elements = 1
        for dim in shape:
            elements *= int(dim)
        return np.zeros(elements, dtype=np.uint16).tobytes()
    if len(shape) != 2:
        raise ValueError(f"block-quant fixtures require rank-2 tensors, got {shape}")
    out_features, in_features = int(shape[0]), int(shape[1])
    if qtype == GGMLQuantizationType.Q8_0:
        return make_q8_0_weight(out_features, in_features).tobytes()
    if qtype == GGMLQuantizationType.Q4_K:
        return make_q4_k_weight(out_features, in_features).tobytes()
    if qtype == GGMLQuantizationType.Q6_K:
        return make_q6_k_weight(out_features, in_features).tobytes()
    if qtype == GGMLQuantizationType.IQ4_XS:
        return np.zeros((out_features, iq4_xs_row_bytes(in_features)), dtype=np.uint8).tobytes()
    raise ValueError(f"no fixture payload builder for {qtype.name}")


def linear_attention_layer_slots(
    layer_id: int,
    *,
    projection_type: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
    alpha_beta_type: GGMLQuantizationType = GGMLQuantizationType.F32,
    gate_type: GGMLQuantizationType | None = None,
    attn_qkv_type: GGMLQuantizationType | None = None,
    ssm_out_type: GGMLQuantizationType | None = None,
) -> list[tuple[str, tuple[int, ...], GGMLQuantizationType]]:
    """Ordered ``(name, shape, qtype)`` records for one AR linear-attention layer.

    ``ssm_out`` defaults to Q8_0 because its fixture K width (64) is not a
    Q4_K/Q6_K block multiple.
    """

    gate = gate_type if gate_type is not None else projection_type
    qkv = attn_qkv_type if attn_qkv_type is not None else projection_type
    ssm_out = ssm_out_type if ssm_out_type is not None else GGMLQuantizationType.Q8_0
    prefix = f"blk.{layer_id}"
    return [
        (f"{prefix}.attn_norm.weight", (FIXTURE_HIDDEN,), GGMLQuantizationType.F32),
        (f"{prefix}.post_attention_norm.weight", (FIXTURE_HIDDEN,), GGMLQuantizationType.F32),
        (f"{prefix}.attn_gate.weight", (FIXTURE_SSM_INNER, FIXTURE_HIDDEN), gate),
        (f"{prefix}.attn_qkv.weight", (FIXTURE_QKV_WIDTH, FIXTURE_HIDDEN), qkv),
        (f"{prefix}.ssm_a", (FIXTURE_SSM_TIME_STEP_RANK,), GGMLQuantizationType.F32),
        (f"{prefix}.ssm_alpha.weight", (FIXTURE_SSM_TIME_STEP_RANK, FIXTURE_HIDDEN), alpha_beta_type),
        (f"{prefix}.ssm_beta.weight", (FIXTURE_SSM_TIME_STEP_RANK, FIXTURE_HIDDEN), alpha_beta_type),
        (f"{prefix}.ssm_conv1d.weight", (FIXTURE_QKV_WIDTH, FIXTURE_SSM_CONV_KERNEL), GGMLQuantizationType.F32),
        (f"{prefix}.ssm_dt.bias", (FIXTURE_SSM_TIME_STEP_RANK,), GGMLQuantizationType.F32),
        (f"{prefix}.ssm_norm.weight", (FIXTURE_SSM_STATE,), GGMLQuantizationType.F32),
        (f"{prefix}.ssm_out.weight", (FIXTURE_HIDDEN, FIXTURE_SSM_INNER), ssm_out),
        (f"{prefix}.ffn_gate.weight", (FIXTURE_FFN, FIXTURE_HIDDEN), projection_type),
        (f"{prefix}.ffn_up.weight", (FIXTURE_FFN, FIXTURE_HIDDEN), projection_type),
        (f"{prefix}.ffn_down.weight", (FIXTURE_HIDDEN, FIXTURE_FFN), projection_type),
    ]


def default_fixture_tensors(
    layer_count: int = 1,
    *,
    embedding_type: GGMLQuantizationType = GGMLQuantizationType.Q8_0,
    projection_type: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
    alpha_beta_type: GGMLQuantizationType = GGMLQuantizationType.F32,
) -> list[tuple[str, tuple[int, ...], GGMLQuantizationType]]:
    tensors: list[tuple[str, tuple[int, ...], GGMLQuantizationType]] = [
        ("token_embd.weight", (FIXTURE_VOCAB, FIXTURE_HIDDEN), embedding_type),
        ("output_norm.weight", (FIXTURE_HIDDEN,), GGMLQuantizationType.F32),
    ]
    for layer_id in range(layer_count):
        tensors.extend(linear_attention_layer_slots(layer_id, projection_type=projection_type, alpha_beta_type=alpha_beta_type))
    return tensors


def fixture_metadata(
    layer_count: int,
    *,
    file_type: int = int(LlamaFileType.MOSTLY_Q4_K_M),
    extra: Mapping[str, object] | None = None,
) -> list[tuple[str, int, object]]:
    metadata: list[tuple[str, int, object]] = [
        ("general.architecture", 8, "qwen35"),
        ("general.alignment", 4, DEFAULT_ALIGNMENT),
        ("general.file_type", 4, int(file_type)),
        ("qwen35.block_count", 4, layer_count),
        ("qwen35.embedding_length", 4, FIXTURE_HIDDEN),
        ("qwen35.feed_forward_length", 4, FIXTURE_FFN),
        ("qwen35.context_length", 4, 64),
        ("qwen35.attention.head_count", 4, FIXTURE_HEADS),
        ("qwen35.attention.head_count_kv", 4, FIXTURE_HEADS_KV),
        ("qwen35.attention.key_length", 4, FIXTURE_KEY_LENGTH),
        ("qwen35.attention.value_length", 4, FIXTURE_VALUE_LENGTH),
        ("qwen35.rope.dimension_count", 4, FIXTURE_KEY_LENGTH),
        ("qwen35.ssm.inner_size", 4, FIXTURE_SSM_INNER),
        ("qwen35.ssm.group_count", 4, FIXTURE_SSM_GROUP),
        ("qwen35.ssm.state_size", 4, FIXTURE_SSM_STATE),
        ("qwen35.ssm.conv_kernel", 4, FIXTURE_SSM_CONV_KERNEL),
        ("qwen35.ssm.time_step_rank", 4, FIXTURE_SSM_TIME_STEP_RANK),
        ("qwen35.attention.layer_norm_rms_epsilon", 6, 1.0e-6),
        ("qwen35.rope.freq_base", 6, 10000000.0),
    ]
    if extra:
        metadata.extend((key, 8 if isinstance(value, str) else 4, value) for key, value in extra.items())
    return metadata


def write_qwen35_gguf(
    path: str | Path,
    tensors: Sequence[tuple[str, Sequence[int], GGMLQuantizationType]],
    metadata: Sequence[tuple[str, int, object]],
    *,
    alignment: int = DEFAULT_ALIGNMENT,
) -> Path:
    """Write one structurally real GGUF file and return its path."""

    tensor_blob = bytearray()
    records: list[tuple[str, tuple[int, ...], GGMLQuantizationType, int, int]] = []
    for name, shape, qtype in tensors:
        payload = payload_bytes(shape, qtype)
        offset = _align_up(len(tensor_blob), alignment)
        tensor_blob += b"\x00" * (offset - len(tensor_blob))
        ggml_shape = tuple(reversed(tuple(int(dim) for dim in shape)))
        records.append((name, ggml_shape, qtype, offset, len(payload)))
        tensor_blob += payload

    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<IQQ", GGUF_VERSION, len(records), len(metadata))
    for key, value_type, value in metadata:
        header += _gguf_string(key)
        header += struct.pack("<I", int(value_type))
        header += _gguf_value(value_type, value)
    for name, ggml_shape, qtype, offset, _nbytes in records:
        header += _gguf_string(name)
        header += struct.pack("<I", len(ggml_shape))
        header += struct.pack(f"<{len(ggml_shape)}Q", *ggml_shape)
        header += struct.pack("<IQ", int(qtype), offset)
    data_start = _align_up(len(header), alignment)
    header += b"\x00" * (data_start - len(header))
    Path(path).write_bytes(bytes(header) + bytes(tensor_blob))
    return Path(path)
