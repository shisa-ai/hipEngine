"""Capture a framework oracle for the Gemma 4 text decoder.

This is optional test tooling, not part of the torch-free runtime. It builds a
tiny randomly-initialized ``Gemma4ForCausalLM`` whose config exercises every
structural feature the real ``google/gemma-4-26B-A4B-it`` text tower uses:

* mixed ``sliding_attention`` / ``full_attention`` layer types;
* per-layer attention geometry (``head_dim`` vs ``global_head_dim``);
* ``attention_k_eq_v`` on global layers (no ``v_proj``, ``V = K``);
* parallel dense MLP and routed-expert branch with separate post-norms;
* ``gelu_pytorch_tanh`` activation in both branches;
* per-layer ``layer_scalar``, embedding scale, and final logit softcapping.

The captured tensors are the reference the NumPy CPU reference in
``hipengine/kernels/cpu_reference/gemma4.py`` is gated against. Run it with an
environment that has torch and transformers installed, for example::

    ~/mambaforge/envs/therock/bin/python scripts/gemma4_hf_tiny_oracle.py \\
        --out tests/fixtures/gemma4_tiny_hf_oracle

The script deliberately avoids the multimodal wrapper so it needs no vision
weights and stays CPU-cheap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM


# A small config that keeps the real model's *structure* and shrinks only the
# widths. Every field that changes the arithmetic is set explicitly.
TINY_CONFIG: dict[str, object] = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 32,
    "global_head_dim": 64,
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attention_k_eq_v": True,
    "sliding_window": 4,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "final_logit_softcapping": 30.0,
    "use_bidirectional_attention": "vision",
    "hidden_size_per_layer_input": 0,
    "num_kv_shared_layers": 0,
    "enable_moe_block": True,
    "use_double_wide_mlp": False,
    "num_experts": 6,
    "top_k_experts": 2,
    "moe_intermediate_size": 16,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
        },
    },
}


def build_model(seed: int) -> Gemma4ForCausalLM:
    torch.manual_seed(seed)
    config = Gemma4TextConfig(**TINY_CONFIG)
    # Pin eager attention so the oracle is the plain scaled-dot-product form the
    # CPU reference implements, with no fused-kernel reassociation.
    config._attn_implementation = "eager"
    model = Gemma4ForCausalLM(config)
    model.eval()
    return model


def capture(seed: int, prompt: list[int]) -> dict[str, np.ndarray]:
    """Capture per-stage activations with explicit, hook-defined boundaries.

    ``transformers`` reports ``hidden_states[-1]`` already passed through the
    final norm, which makes the list ambiguous at its end. Recording the
    boundaries directly keeps the fixture readable: the embedding output, each
    decoder layer's output, and the final norm's output.
    """

    model = build_model(seed)
    input_ids = torch.tensor([prompt], dtype=torch.long)

    recorded: dict[str, np.ndarray] = {}

    def record(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            recorded[name] = tensor[0].detach().numpy().copy()

        return hook

    handles = []
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(record(f"layer.{index}")))
    handles.append(model.model.norm.register_forward_hook(record("final_norm")))
    handles.append(model.model.embed_tokens.register_forward_hook(record("embedding")))
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    tensors: dict[str, np.ndarray] = {
        "logits": outputs.logits[0].numpy(),
        "last_hidden_state": outputs.hidden_states[-1][0].numpy(),
    }
    tensors.update(recorded)
    return tensors


def weight_digest(model: Gemma4ForCausalLM) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().numpy()).tobytes())
    return digest.hexdigest()


def state_dict_tensors(model: Gemma4ForCausalLM) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(tensor.detach().numpy())
        for name, tensor in sorted(model.state_dict().items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260401)
    parser.add_argument(
        "--prompt",
        type=int,
        nargs="+",
        default=[3, 17, 42, 7, 91, 12, 64, 5, 23, 88],
    )
    args = parser.parse_args()

    model = build_model(args.seed)
    tensors = capture(args.seed, args.prompt)
    weights = state_dict_tensors(model)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "weights.npz", **weights)
    np.savez_compressed(args.out / "activations.npz", **tensors)
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "tiny_config": TINY_CONFIG,
                "seed": args.seed,
                "prompt": args.prompt,
                "torch_version": torch.__version__,
                "transformers_version": _transformers_version(),
                "weight_digest": weight_digest(model),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {len(weights)} weight tensors and {len(tensors)} activations to {args.out}")
    return 0


def _transformers_version() -> str:
    import transformers

    return str(transformers.__version__)


if __name__ == "__main__":
    raise SystemExit(main())
