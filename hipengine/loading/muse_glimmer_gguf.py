"""Muse Glimmer GGUF tensor map and resident weight materialization.

The loader reads every dimension from the artifact (no identity gating) and
uploads quantized projections byte-for-byte in the raw GGUF layout; dispatch
resolves the per-quant kernel owners at launch.

One load-time layout transform is applied: the conversion stored Q/K in ggml
NORM (interleaved) RoPE layout, while the registered head-norm+rotary kernels
rotate NEOX pairs ``(i, i + d/2)``. Permuting the Q/K projection *rows* (and the
per-head QK-norm weight vectors) from NORM into NEOX order makes the existing
NEOX kernels compute exactly the interleaved rotation: each quantized row is
independently quantized along ``in_features``, so a row reorder is a pure byte
permutation with no requantization and no arithmetic change. Attention dot
products are invariant under the same permutation applied to Q and K.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.cpu_reference.muse_glimmer import MuseGlimmerConfig
from hipengine.loading.gguf import GGUFModelInfo, GGUFReader
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    load_host_array_to_device,
    load_host_array_to_device_as_dtype,
)
from hipengine.runtime.gguf_weight import GGUFDeviceWeight, GGUFWeightSpec

LAYOUT_RAW_GGUF = "raw_gguf"

#: Raw-layout quant keys this loader uploads. The runtime dispatch surface
#: resolves GEMV/prefill owners per key; admission never consults this map.
RAW_QUANT_KEYS: Mapping[str, str] = {
    "Q4_K": "gguf_q4_k",
    "Q5_K": "gguf_q5_k",
    "Q6_K": "gguf_q6_k",
    "Q8_0": "gguf_q8_0",
}

_LAYER_SUFFIXES = (
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_gate.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
    "post_attention_norm.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "post_ffw_norm.weight",
)


def required_tensor_names(config: MuseGlimmerConfig) -> tuple[str, ...]:
    names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    for layer in range(config.n_layers):
        names.extend(f"blk.{layer}.{suffix}" for suffix in _LAYER_SUFFIXES)
    return tuple(names)


def config_from_gguf_info(info: GGUFModelInfo) -> MuseGlimmerConfig:
    """Decode the architecture contract from the artifact metadata."""

    vocab_size = int(info.tensor("token_embd.weight").shape[0])
    return MuseGlimmerConfig.from_gguf_info(info, vocab_size=vocab_size)


def validate_tensor_map(info: GGUFModelInfo, config: MuseGlimmerConfig) -> None:
    """Fail with a named capability miss when a required tensor is absent/mis-shaped."""

    present = {t.name for t in info.tensors}
    d, n_heads, n_kv_heads = config.head_dim, config.n_heads, config.n_kv_heads
    expected_shapes = {
        "token_embd.weight": (config.vocab_size, config.n_embd),
        "output_norm.weight": (config.n_embd,),
        "output.weight": (config.vocab_size, config.n_embd),
    }
    for layer in range(config.n_layers):
        p = f"blk.{layer}."
        expected_shapes.update({
            p + "attn_norm.weight": (config.n_embd,),
            p + "attn_q.weight": (n_heads * d, config.n_embd),
            p + "attn_k.weight": (n_kv_heads * d, config.n_embd),
            p + "attn_v.weight": (n_kv_heads * d, config.n_embd),
            p + "attn_gate.weight": (n_heads * d, config.n_embd),
            p + "attn_q_norm.weight": (d,),
            p + "attn_k_norm.weight": (d,),
            p + "attn_output.weight": (config.n_embd, n_heads * d),
            p + "post_attention_norm.weight": (config.n_embd,),
            p + "ffn_norm.weight": (config.n_embd,),
            p + "ffn_gate.weight": (config.n_ff, config.n_embd),
            p + "ffn_up.weight": (config.n_ff, config.n_embd),
            p + "ffn_down.weight": (config.n_embd, config.n_ff),
            p + "post_ffw_norm.weight": (config.n_embd,),
        })
    for name, shape in expected_shapes.items():
        if name not in present:
            raise ValueError(f"Muse Glimmer GGUF is missing required tensor {name!r}")
        actual = tuple(info.tensor(name).shape)
        if actual != shape:
            raise ValueError(
                f"Muse Glimmer tensor {name!r} shape {actual} does not match "
                f"the declared architecture shape {shape}"
            )


def norm_to_neox_row_permutation(n_heads: int, head_dim: int) -> np.ndarray:
    """Gather indices mapping NORM-layout rows to NEOX order, per head.

    ``result[dst] = src`` with ``src = h*d + j`` (NORM: pair ``j//2``, side
    ``j%2``) and ``dst = h*d + (j%2)*(d//2) + j//2`` (NEOX: half then pair).
    Applying it to Q and K rows makes NEOX rotation equal the ggml NORM
    interleaved rotation on the original rows.
    """

    if head_dim % 2:
        raise ValueError("head_dim must be even for paired rotary")
    src = np.arange(n_heads * head_dim, dtype=np.int64)
    head = src // head_dim
    j = src % head_dim
    dst = head * head_dim + (j % 2) * (head_dim // 2) + j // 2
    perm = np.empty_like(src)
    perm[dst] = src
    return perm


@dataclass(frozen=True)
class MuseGlimmerWeightSpec:
    """Resident-layout metadata for one Glimmer weight (dispatch protocol)."""

    layout: str
    quant_key: str


@dataclass(frozen=True)
class MuseGlimmerDeviceWeight:
    """One resident Glimmer weight satisfying the GGUF dispatch protocol."""

    spec: MuseGlimmerWeightSpec
    backend: str
    _allocation: DeviceTensorAllocation

    def allocation(self, name: str | None = None) -> DeviceTensorAllocation:
        if name not in (None, "raw"):
            raise ValueError(f"Glimmer raw weights expose only 'raw', not {name!r}")
        return self._allocation


@dataclass(frozen=True)
class MuseGlimmerLoadedWeights:
    """Device-resident Glimmer weight set; owns every buffer it materialized."""

    config: MuseGlimmerConfig
    backend: str
    weights: Mapping[str, MuseGlimmerDeviceWeight]
    _map: DeviceWeightMap

    def weight(self, name: str) -> GGUFDeviceWeight:
        return self.weights[name]

    def ptr(self, name: str) -> int:
        return self.weights[name].allocation().tensor.ptr

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        self._map.free(runtime=runtime)


def materialize_muse_glimmer_weights(
    reader: GGUFReader,
    *,
    backend: str,
    runtime: HipRuntime | None = None,
) -> MuseGlimmerLoadedWeights:
    """Upload every Glimmer tensor; Q/K rows permuted NORM->NEOX on SWA layers.

    Quantized projections keep their exact GGUF bytes (row reorder only).
    F32 norm weights upload as-is, except the per-head QK-norm vectors on SWA
    layers, which take the same element permutation as their projection rows.
    """

    runtime = runtime or get_hip_runtime()
    config = config_from_gguf_info(reader.info)
    validate_tensor_map(reader.info, config)

    allocations: dict[str, DeviceTensorAllocation] = {}
    weights: dict[str, MuseGlimmerDeviceWeight] = {}
    try:
        for name in required_tensor_names(config):
            tensor = reader.tensor_info(name)
            raw = np.ascontiguousarray(reader.tensor_data(name))
            quant_key = ""
            if tensor.ggml_type_name == "F32":
                if name.endswith(("attn_q_norm.weight", "attn_k_norm.weight")):
                    layer = int(name.split(".")[1])
                    if config.swa_pattern[layer]:
                        perm = norm_to_neox_row_permutation(1, config.head_dim)
                        raw = np.ascontiguousarray(raw[perm])
                allocation = load_host_array_to_device(name, raw, runtime=runtime)
            elif tensor.ggml_type_name in RAW_QUANT_KEYS:
                quant_key = RAW_QUANT_KEYS[tensor.ggml_type_name]
                if name.endswith(("attn_q.weight", "attn_k.weight")):
                    layer = int(name.split(".")[1])
                    if config.swa_pattern[layer]:
                        n_heads = (
                            config.n_heads if name.endswith("attn_q.weight")
                            else config.n_kv_heads
                        )
                        perm = norm_to_neox_row_permutation(n_heads, config.head_dim)
                        raw = np.ascontiguousarray(raw[perm])
                allocation = load_host_array_to_device_as_dtype(
                    name, raw, DType.INT8, source_dtype="I8", runtime=runtime
                )
            else:
                raise ValueError(
                    f"Muse Glimmer tensor {name!r} has unsupported storage type "
                    f"{tensor.ggml_type_name!r}; no raw resident route is registered"
                )
            allocations[name] = allocation
            weights[name] = MuseGlimmerDeviceWeight(
                spec=MuseGlimmerWeightSpec(
                    layout=LAYOUT_RAW_GGUF if quant_key else "dense_f32",
                    quant_key=quant_key or "f32",
                ),
                backend=backend,
                _allocation=allocation,
            )
    except Exception:
        DeviceWeightMap(tensors=allocations).free(runtime=runtime)
        raise
    return MuseGlimmerLoadedWeights(
        config=config,
        backend=backend,
        weights=weights,
        _map=DeviceWeightMap(tensors=allocations),
    )


__all__ = [
    "LAYOUT_RAW_GGUF",
    "MuseGlimmerDeviceWeight",
    "MuseGlimmerLoadedWeights",
    "MuseGlimmerWeightSpec",
    "config_from_gguf_info",
    "materialize_muse_glimmer_weights",
    "norm_to_neox_row_permutation",
    "required_tensor_names",
    "validate_tensor_map",
]