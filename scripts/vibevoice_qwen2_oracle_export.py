#!/usr/bin/env python3
"""Export HF Qwen2 backbone outputs from the pinned fork (oracle venv).

Runs the community fork's language model (HF ``Qwen2Model``) in float32 on a
fixed token sequence and dumps the final-norm hidden states plus tied-embedding
logits for the hipEngine CPU-reference comparison. Run inside the pinned
oracle environment (~/venvs/vibevoice-oracle), never the engine venv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

DEFAULT_FORK = "/home/lhl/VibeVoice-community"
DEFAULT_MODEL = "/models/vibevoice/VibeVoice-1.5B"

# Fixed prompt-ish sequence: system marker, text, speech start, in-range ids.
TOKENS = [151644, 872, 198, 9707, 151645, 151652, 100, 200, 300, 400]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fork", default=DEFAULT_FORK)
    parser.add_argument("--out", default="benchmarks/results/2026-09-15-vibevoice-qwen2-oracle.npz")
    args = parser.parse_args()

    sys.path.insert(0, args.fork)
    from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration

    model = VibeVoiceForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    lm = model.model.language_model

    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        out = lm(ids)
        hidden = out.last_hidden_state[0]  # [tokens, hidden] final-norm output
        embed = model.get_input_embeddings().weight
        logits = hidden @ embed.T

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        token_ids=np.array(TOKENS, dtype=np.int64),
        hidden=hidden.numpy().astype(np.float64),
        logits=logits.numpy().astype(np.float64),
    )
    meta = {
        "model_type": lm.config.model_type,
        "hidden_size": lm.config.hidden_size,
        "num_hidden_layers": lm.config.num_hidden_layers,
        "num_attention_heads": lm.config.num_attention_heads,
        "num_key_value_heads": lm.config.num_key_value_heads,
        "head_dim": getattr(lm.config, "head_dim", None) or lm.config.hidden_size // lm.config.num_attention_heads,
        "rope_theta": lm.config.rope_theta,
        "rms_norm_eps": lm.config.rms_norm_eps,
        "tie_word_embeddings": lm.config.tie_word_embeddings,
        "attn_implementation": "eager",
        "tokens": TOKENS,
    }
    Path(str(out_path).replace(".npz", ".json")).write_text(json.dumps(meta, indent=1))
    print("exported", out_path, "hidden", tuple(hidden.shape), "logits", tuple(logits.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
