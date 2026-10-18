"""Check hipEngine's Gemma 4 layer forward against HuggingFace on real weights.

Loads the float32 tensors dumped by ``scripts/gemma4_real_layer_dump.py`` into a
single ``Gemma4TextDecoderLayer`` built from the real ``text_config``, runs the
same input, and compares against the hipEngine CPU reference output. Both sides
consume the *same* dequantized weights in float32, so any disagreement is a
difference in the arithmetic rather than in the artifact.

The layer is called with an explicit additive causal mask. A standalone
``Gemma4TextDecoderLayer`` with ``attention_mask=None`` falls back to full
attention, which makes every row but the last differ for a reason that has
nothing to do with the reference.

Usage::

    G4_LAYER=5 python scripts/gemma4_real_layer_hf_check.py

Run it with the ``therock`` environment, which carries ``transformers``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextDecoderLayer,
    Gemma4TextRotaryEmbedding,
)

LAYER = int(os.environ.get("G4_LAYER", "0"))
DUMP = Path(f"/tmp/g4_layer{LAYER}/weights.npz")
CONFIG_JSON = Path("/tmp/g4_config.json")

# Absolute tolerance relative to the layer's largest output magnitude. The two
# sides agree to float32 rounding on the same weights, so this is a tight gate.
RELATIVE_TOLERANCE = 1e-4


def main() -> int:
    payload = dict(np.load(DUMP))
    text_config = json.loads(CONFIG_JSON.read_text())["text_config"]
    text_config.pop("model_type", None)
    config = Gemma4TextConfig(**text_config)
    config._attn_implementation = "eager"

    torch.manual_seed(0)
    layer = Gemma4TextDecoderLayer(config, LAYER).to(torch.float32)
    layer.eval()

    prefixes = (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "self_attn",
        "mlp.",
        "router.",
        "experts.",
        "layer_scalar",
    )
    state = {
        name: torch.from_numpy(np.ascontiguousarray(value))
        for name, value in payload.items()
        if name.startswith(prefixes)
    }
    missing, unexpected = layer.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("state dict mismatch:", missing, unexpected)
        return 1

    hidden = torch.from_numpy(payload["layer_input"])[None]
    tokens = hidden.shape[1]
    position_ids = torch.arange(tokens)[None]
    rotary = Gemma4TextRotaryEmbedding(config)
    position_embeddings = rotary(hidden, position_ids, layer_type=config.layer_types[LAYER])
    mask = torch.triu(
        torch.full((1, 1, tokens, tokens), torch.finfo(torch.float32).min),
        diagonal=1,
    )

    with torch.no_grad():
        out = layer(hidden, position_embeddings=position_embeddings, attention_mask=mask)
    out = out[0].float().numpy()

    reference = payload["hipengine_output"]
    scale = float(np.abs(reference).max())
    worst = float(np.abs(out - reference).max())
    relative = worst / scale
    print(f"layer {LAYER} ({config.layer_types[LAYER]})")
    print(f"  HF  absmax {np.abs(out).max():.6f}  mean {np.abs(out).mean():.6f}")
    print(f"  hip absmax {scale:.6f}  mean {np.abs(reference).mean():.6f}")
    print(f"  max abs diff {worst:.6g}   relative to absmax {relative:.3e}")
    print(f"  per-row max abs diff {np.array2string(np.abs(out - reference).max(axis=1))}")

    np.save(f"/tmp/g4_layer{LAYER}/hf_output.npy", out)
    if relative >= RELATIVE_TOLERANCE:
        print(f"FAIL: relative difference {relative:.3e} >= {RELATIVE_TOLERANCE:.0e}")
        return 1
    print(f"PASS: relative difference {relative:.3e} < {RELATIVE_TOLERANCE:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
