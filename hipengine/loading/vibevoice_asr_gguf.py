"""Load a quantized VibeVoice-ASR GGUF into Qwen2 backbone device buffers.

Reads the Q4_K_M GGUF produced by scripts/vibevoice_asr_to_gguf.py +
llama-quantize + scripts/vibevoice_asr_gguf_merge.py:

- spec geometry from the ``vibevoice.text.*`` metadata keys
- ``blk.N.*`` backbone tensors: Q4_K blocks for the six GEMM weights
  (raw GGUF byte layout, consumed directly by the q4_k kernels),
  BF16 for norms, biases, ``token_embd`` (Q4_K/Q6_K in file),
  and a BF16 ``output`` (untied lm_head)
- encoder/projector tensors are not part of this loader; the front-end
  keeps loading BF16 weights from the HF artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.core.memory import DeviceBuffer, copy_host_array_to_device, malloc
from hipengine.loading.gguf import GGUFReader, scan_gguf


@dataclass(frozen=True)
class Qwen2Q4Spec:
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float

    @classmethod
    def from_gguf(cls, metadata: Any) -> "Qwen2Q4Spec":
        get = metadata.get
        hidden = int(get("vibevoice.text.hidden_size"))
        heads = int(get("vibevoice.text.num_attention_heads"))
        return cls(
            hidden_size=hidden,
            num_layers=int(get("vibevoice.text.num_hidden_layers")),
            num_attention_heads=heads,
            num_key_value_heads=int(get("vibevoice.text.num_key_value_heads")),
            head_dim=hidden // heads,
            intermediate_size=int(get("vibevoice.text.intermediate_size")),
            vocab_size=int(get("vibevoice.text.vocab_size")),
            rope_theta=float(get("vibevoice.text.rope_theta", 1000000.0)),
            rms_norm_eps=float(get("vibevoice.text.rms_norm_eps", 1e-6)),
        )


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return ((bits.astype(np.uint32) << 16).view(np.float32))


@dataclass
class Qwen2Q4Layer:
    input_ln: DeviceBuffer
    q_w: DeviceBuffer
    q_b: DeviceBuffer
    k_w: DeviceBuffer
    k_b: DeviceBuffer
    v_w: DeviceBuffer
    v_b: DeviceBuffer
    o_w: DeviceBuffer
    post_ln: DeviceBuffer
    gate_w: DeviceBuffer
    up_w: DeviceBuffer
    down_w: DeviceBuffer
    # host-side metadata for kernel launches
    gemm_shapes: dict = field(default_factory=dict)


@dataclass
class Qwen2Q4Weights:
    spec: Qwen2Q4Spec
    embed: DeviceBuffer
    embed_host_bf16: np.ndarray
    final_norm: DeviceBuffer
    lm_head: DeviceBuffer
    lm_head_host_bf16: np.ndarray
    layers: list[Qwen2Q4Layer]
    buffers: list[DeviceBuffer] = field(default_factory=list)


GGUF_WEIGHT_TYPES: dict[int, int] = {}
"""Device pointer -> GGML tensor type, published by the loader so the
q4 linear wrappers can route Q4_K vs Q6_K blocks per weight."""


def _upload(reader: GGUFReader, name: str, keep: list) -> DeviceBuffer:
    """Raw storage upload: BF16 as uint16, Q4_K/Q6_K as their byte blocks."""
    data = np.asarray(reader.tensor_data(name))
    buf = malloc(data.nbytes)
    copy_host_array_to_device(buf, data)
    keep.append(buf)
    GGUF_WEIGHT_TYPES[buf.ptr] = int(reader.tensor_info(name).ggml_type)
    return buf


def load_vibevoice_qwen2_q4(gguf_path: str | Path) -> Qwen2Q4Weights:
    path = Path(gguf_path)
    info = scan_gguf(path)
    if info.architecture != "vibevoice-asr":
        raise ValueError(f"expected vibevoice-asr GGUF, got {info.architecture!r}")
    spec = Qwen2Q4Spec.from_gguf(info.metadata)
    reader = GGUFReader(path)
    keep: list[DeviceBuffer] = []

    def up(name: str) -> DeviceBuffer:
        buf = _upload(reader, name, keep)
        return buf

    def up_f32(name: str) -> DeviceBuffer:
        """BF16 storage widened to fp32 (biases are consumed as f32)."""
        bits = np.asarray(reader.tensor_data(name)).reshape(-1)
        data = _bf16_bits_to_f32(bits).astype(np.float32)
        buf = malloc(data.nbytes)
        copy_host_array_to_device(buf, data)
        keep.append(buf)
        return buf

    def host_bf16(name: str) -> np.ndarray:
        data = np.asarray(reader.tensor_data(name)).reshape(-1)
        return _bf16_bits_to_f32(data)

    embed_host = host_bf16("token_embd.weight")
    # token_embd may be quantized in file; fall back to dequantized upload
    embed_info = reader.tensor_info("token_embd.weight")
    if embed_info.ggml_type != 30:
        embed_host = np.asarray(reader.dequantize_tensor("token_embd.weight")).reshape(-1)
    embed_buf = malloc(embed_host.nbytes * 2)
    copy_host_array_to_device(embed_buf, _f32_to_bf16_bits(embed_host))
    keep.append(embed_buf)

    final_host = host_bf16("output_norm.weight")
    final_buf = malloc(final_host.nbytes * 2)
    copy_host_array_to_device(final_buf, _f32_to_bf16_bits(final_host))
    keep.append(final_buf)

    lm_host = host_bf16("output.weight")
    lm_buf = malloc(lm_host.nbytes * 2)
    copy_host_array_to_device(lm_buf, _f32_to_bf16_bits(lm_host))
    keep.append(lm_buf)

    layers: list[Qwen2Q4Layer] = []
    hidden = spec.hidden_size
    kv_dim = spec.num_key_value_heads * spec.head_dim
    ffn = spec.intermediate_size
    for i in range(spec.num_layers):
        p = f"blk.{i}."
        for gemm in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
            t = reader.tensor_info(p + gemm + ".weight")
            if t.ggml_type not in (12, 13, 14):  # Q4_K, Q5_K, Q6_K
                raise ValueError(f"{p}{gemm}.weight not quantized (type {t.ggml_type})")
        layers.append(
            Qwen2Q4Layer(
                input_ln=up(p + "attn_norm.weight"),
                q_w=up(p + "attn_q.weight"),
                q_b=up_f32(p + "attn_q.bias"),
                k_w=up(p + "attn_k.weight"),
                k_b=up_f32(p + "attn_k.bias"),
                v_w=up(p + "attn_v.weight"),
                v_b=up_f32(p + "attn_v.bias"),
                o_w=up(p + "attn_output.weight"),
                post_ln=up(p + "ffn_norm.weight"),
                gate_w=up(p + "ffn_gate.weight"),
                up_w=up(p + "ffn_up.weight"),
                down_w=up(p + "ffn_down.weight"),
                gemm_shapes={
                    "q": (hidden, hidden),
                    "k": (hidden, kv_dim),
                    "v": (hidden, kv_dim),
                    "o": (kv_dim, spec.num_attention_heads * spec.head_dim),
                    "gate": (hidden, ffn),
                    "up": (hidden, ffn),
                    "down": (ffn, hidden),
                },
            )
        )
    return Qwen2Q4Weights(
        spec=spec,
        embed=embed_buf,
        embed_host_bf16=embed_host,
        final_norm=final_buf,
        lm_head=lm_buf,
        lm_head_host_bf16=lm_host,
        layers=layers,
        buffers=keep,
    )


def _f32_to_bf16_bits(host: np.ndarray) -> np.ndarray:
    bits = np.asarray(host, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return (rounded >> 16).astype(np.uint16)
