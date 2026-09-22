"""Dump one real Gemma 4 layer's weights from the GGUF as float32 for HF comparison.

Writes ``/tmp/g4_layer<LAYER>/weights.npz`` with the layer's dequantized tensors,
the real embedding rows, the layer input, and the hipEngine layer output for a
fixed token block. A separate process running the reference HuggingFace
implementation on the *same* weights compares against it; see
``scripts/gemma4_real_layer_hf_check.py``.

Usage::

    G4_LAYER=5 python scripts/gemma4_real_layer_dump.py

Layer 0 is a sliding-attention layer (``head_dim`` 256, one RoPE table, no
``k_eq_v``). Layer 5 is the first global layer (``global_head_dim`` 512,
proportional partial RoPE, ``V = K``), so running both covers every structural
branch the real model uses.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from hipengine.kernels.cpu_reference.gemma4 import gemma4_rmsnorm
from hipengine.kernels.cpu_reference.gemma4_streaming import (
    Gemma4GGUFStreamingWeights,
    _streaming_layer_forward,
)
from hipengine.loading.gguf import GGUFReader

ARTIFACT = Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")

PROMPT = [2, 818, 5279, 529, 7001, 563]
LAYER = int(os.environ.get("G4_LAYER", "0"))
OUT = Path(f"/tmp/g4_layer{LAYER}")

SLOT_TO_HF = {
    "input_layernorm": "input_layernorm.weight",
    "post_attention_layernorm": "post_attention_layernorm.weight",
    "pre_feedforward_layernorm": "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm": "post_feedforward_layernorm.weight",
    "post_feedforward_layernorm_1": "post_feedforward_layernorm_1.weight",
    "post_feedforward_layernorm_2": "post_feedforward_layernorm_2.weight",
    "pre_feedforward_layernorm_2": "pre_feedforward_layernorm_2.weight",
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    "mlp_gate_proj": "mlp.gate_proj.weight",
    "mlp_up_proj": "mlp.up_proj.weight",
    "mlp_down_proj": "mlp.down_proj.weight",
    "router_scale": "router.scale",
    "router_proj": "router.proj.weight",
    "router_per_expert_scale": "router.per_expert_scale",
    "experts_gate_up_proj": "experts.gate_up_proj",
    "experts_down_proj": "experts.down_proj",
    "layer_scalar": "layer_scalar",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reader = GGUFReader(ARTIFACT)
    streaming = Gemma4GGUFStreamingWeights(reader, dtype=np.float32)
    config = streaming.config
    geometry = config.geometry(LAYER)

    started = time.time()
    layer = streaming.layer_weights(LAYER)
    payload: dict[str, np.ndarray] = {}
    for attribute, hf_name in SLOT_TO_HF.items():
        if attribute in ("experts_gate_up_proj", "experts_down_proj"):
            continue
        value = getattr(layer, attribute)
        if value is None:
            continue
        payload[hf_name] = np.asarray(value, dtype=np.float32)
    all_experts = list(range(config.num_experts))
    gate_up, down = streaming.expert_weights(LAYER, all_experts)
    payload["experts.gate_up_proj"] = np.asarray(gate_up, dtype=np.float32)
    payload["experts.down_proj"] = np.asarray(down, dtype=np.float32)
    print(f"weights dequantized in {time.time() - started:.0f}s", flush=True)

    embedding = streaming.embed_tokens()
    payload["embed_rows"] = np.asarray(embedding[PROMPT], dtype=np.float32)
    del embedding

    hidden = payload["embed_rows"] * np.float32(config.embed_scale)
    payload["layer_input"] = np.asarray(hidden, dtype=np.float32)
    positions = np.arange(len(PROMPT), dtype=np.int64)
    started = time.time()
    output = _streaming_layer_forward(
        hidden, streaming, LAYER, geometry, config, positions=positions
    )
    payload["hipengine_output"] = np.asarray(output, dtype=np.float32)
    print(f"hipEngine layer {LAYER} forward in {time.time() - started:.0f}s", flush=True)

    final = gemma4_rmsnorm(output, streaming.final_norm(), config.rms_norm_eps)
    payload["hipengine_after_final_norm"] = np.asarray(final, dtype=np.float32)

    np.savez(OUT / "weights.npz", **payload)
    print("wrote", OUT / "weights.npz", flush=True)
    for name, value in payload.items():
        print(f"  {name:36s} {value.shape} {value.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
