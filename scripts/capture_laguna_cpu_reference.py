#!/usr/bin/env python3
"""Capture a tiny deterministic Laguna CPU oracle with Hugging Face Transformers.

This is optional fixture-generation tooling. Run it in an environment containing
PyTorch and Transformers with native Laguna support; neither dependency is part
of hipEngine's torch-free runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import LagunaConfig, LagunaForCausalLM
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.laguna.modeling_laguna import LagunaRotaryEmbedding

POOL_SIDE_REVISION = "179ee67cf0fff5391c67fe1a392ea849fa6d643f"
DEFAULT_OUTPUT = Path("tests/fixtures/laguna_cpu_reference.json")
ROPE_PARAMETERS = {
    "full_attention": {
        "rope_type": "yarn",
        "rope_theta": 500_000.0,
        "factor": 32.0,
        "original_max_position_embeddings": 8_192,
        "beta_slow": 1.0,
        "beta_fast": 32.0,
        "partial_rotary_factor": 0.5,
    },
    "sliding_attention": {
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 1.0,
    },
}


def _config(*, head_dim: int, hidden_size: int, vocab_size: int) -> LagunaConfig:
    config = LagunaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=head_dim,
        max_position_embeddings=262_144,
        rms_norm_eps=1.0e-6,
        sliding_window=512,
        rope_parameters=ROPE_PARAMETERS,
        layer_types=["full_attention", "sliding_attention"],
        num_attention_heads_per_layer=[48, 72],
        mlp_layer_types=["dense", "sparse"],
        moe_intermediate_size=1,
        shared_expert_intermediate_size=1,
        num_experts=256,
        num_experts_per_tok=10,
        moe_routed_scaling_factor=2.5,
        moe_router_logit_softcapping=0.0,
        attention_bias=False,
        tie_word_embeddings=False,
        pad_token_id=max(vocab_size - 1, 0),
        bos_token_id=min(2, vocab_size - 1),
        eos_token_id=[min(2, vocab_size - 1)],
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


def _parameter_array(name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Return exactly specified deterministic FP32 fixture values."""

    count = int(np.prod(shape, dtype=np.int64))
    index = np.arange(1, count + 1, dtype=np.float64)
    phase = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little") % 10_007
    if name.endswith("norm.weight") or name == "model.norm.weight":
        values = 0.9 + 0.08 * np.sin(index * 0.071 + phase * 0.003)
    elif name.endswith("e_score_correction_bias"):
        values = 0.01 * np.sin(index * 0.13 + phase * 0.001)
        favored = (3, 17, 29, 47, 71, 101, 139, 173, 211, 251)
        for rank, expert_id in enumerate(favored):
            values[expert_id] = 0.9 - rank * 0.035
    else:
        values = 0.055 * np.sin(index * ((phase % 19) + 1) * 0.017) + 0.025 * np.cos(
            (index + phase) * 0.013
        )
    return np.ascontiguousarray(values.reshape(shape).astype(np.float32))


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _to_json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().to(torch.float32).numpy()
        return {
            "dtype": "float32",
            "shape": list(array.shape),
            "data": array.tolist(),
        }
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot serialize oracle value of type {type(value)!r}")


def _capture_tiny_model() -> tuple[dict[str, Any], dict[str, str]]:
    config = _config(head_dim=4, hidden_size=4, vocab_size=7)
    model = LagunaForCausalLM(config).float().eval()
    model.config._attn_implementation = "eager"
    model.config._experts_implementation = "eager"

    parameter_sha256: dict[str, str] = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            values = _parameter_array(name, tuple(parameter.shape))
            parameter.copy_(torch.from_numpy(values))
            parameter_sha256[name] = _sha256_array(values)

    captures: dict[str, Any] = {}
    handles = []

    def capture(name: str, *, tuple_index: int | None = None):
        def hook(_module, _inputs, output):
            selected = output if tuple_index is None else output[tuple_index]
            captures[name] = _to_json_value(selected)

        return hook

    layer0 = model.model.layers[0]
    layer1 = model.model.layers[1]
    hook_specs = (
        (layer0.input_layernorm, "layer0.attention_norm", None),
        (layer0.self_attn.q_norm, "layer0.q_head_norm", None),
        (layer0.self_attn.k_norm, "layer0.k_head_norm", None),
        (layer0.self_attn.g_proj, "layer0.gate_logits", None),
        (layer0.self_attn, "layer0.attention_output", 0),
        (layer0.post_attention_layernorm, "layer0.ffn_norm", None),
        (layer0.mlp, "layer0.ffn_output", None),
        (layer0, "layer0.output", None),
        (layer1.input_layernorm, "layer1.attention_norm", None),
        (layer1.self_attn.q_norm, "layer1.q_head_norm", None),
        (layer1.self_attn.k_norm, "layer1.k_head_norm", None),
        (layer1.self_attn.g_proj, "layer1.gate_logits", None),
        (layer1.self_attn, "layer1.attention_output", 0),
        (layer1.post_attention_layernorm, "layer1.ffn_norm", None),
        (layer1.mlp.gate, "layer1.router", None),
        (layer1.mlp.experts, "layer1.routed_output_unscaled", None),
        (layer1.mlp.shared_experts, "layer1.shared_output", None),
        (layer1.mlp, "layer1.ffn_output", None),
        (layer1, "layer1.output", None),
        (model.model.norm, "model.final_norm", None),
        (model.lm_head, "model.logits", None),
    )
    for module, name, tuple_index in hook_specs:
        handles.append(module.register_forward_hook(capture(name, tuple_index=tuple_index)))

    input_ids = torch.tensor([[1, 3, 5]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            output_router_logits=True,
        )
    for handle in handles:
        handle.remove()

    captures["model.hidden_states"] = _to_json_value(outputs.hidden_states)
    captures["model.router_logits"] = _to_json_value(outputs.router_logits)
    return (
        {
            "input_ids": input_ids.tolist(),
            "position_ids": position_ids.tolist(),
            "captures": captures,
        },
        parameter_sha256,
    )


def _capture_rope() -> dict[str, Any]:
    config = _config(head_dim=128, hidden_size=3_072, vocab_size=7)
    rotary = LagunaRotaryEmbedding(config).float().eval()
    positions = torch.tensor([[0, 1, 8_191, 8_192, 8_193, 262_143]], dtype=torch.long)
    probe = torch.zeros((1, positions.shape[1], 3_072), dtype=torch.float32)
    result: dict[str, Any] = {"positions": positions.tolist()[0]}
    with torch.no_grad():
        for layer_type in ("full_attention", "sliding_attention"):
            cos, sin = rotary(probe, positions, layer_type=layer_type)
            result[layer_type] = {
                "rotary_dim": int(cos.shape[-1]),
                "cos": _to_json_value(cos[0]),
                "sin": _to_json_value(sin[0]),
            }
    return result


def _capture_masks() -> dict[str, Any]:
    config = _config(head_dim=4, hidden_size=4, vocab_size=7)
    result: dict[str, Any] = {}
    for length in (511, 512, 513):
        inputs = torch.zeros((1, length, 4), dtype=torch.float32)
        positions = torch.arange(length, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones((1, length), dtype=torch.long)
        entry: dict[str, Any] = {}
        for name, factory in (
            ("full_attention", create_causal_mask),
            ("sliding_attention", create_sliding_window_causal_mask),
        ):
            mask = factory(
                config=config,
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=positions,
            )
            assert isinstance(mask, torch.Tensor)
            visible = torch.where(mask[0, 0, -1] == 0)[0].tolist()
            entry[name] = {
                "query_position": length - 1,
                "visible_key_positions": visible,
            }
        result[str(length)] = entry
    return result


def capture() -> dict[str, Any]:
    tiny_model, parameter_sha256 = _capture_tiny_model()
    modeling_path = Path(inspect.getfile(LagunaForCausalLM))
    modeling_sha256 = hashlib.sha256(modeling_path.read_bytes()).hexdigest()
    return {
        "schema": 1,
        "name": "laguna-transformers-tiny-two-layer-cpu-reference",
        "provenance": {
            "framework": "Hugging Face Transformers",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "poolside_model_revision": POOL_SIDE_REVISION,
            "poolside_modeling_url": (
                "https://huggingface.co/poolside/Laguna-S-2.1/blob/"
                f"{POOL_SIDE_REVISION}/modeling_laguna.py"
            ),
            "poolside_config_url": (
                "https://huggingface.co/poolside/Laguna-S-2.1/blob/"
                f"{POOL_SIDE_REVISION}/config.json"
            ),
            "installed_modeling_path": str(modeling_path),
            "installed_modeling_sha256": modeling_sha256,
            "command": (
                "/home/lhl/miniforge3/envs/vllm/bin/python scripts/capture_laguna_cpu_reference.py"
            ),
        },
        "config": {
            "hidden_size": 4,
            "head_dim": 4,
            "num_attention_heads_per_layer": [48, 72],
            "num_key_value_heads": 8,
            "layer_types": ["full_attention", "sliding_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "num_experts": 256,
            "num_experts_per_tok": 10,
            "dense_intermediate_size": 6,
            "moe_intermediate_size": 1,
            "shared_expert_intermediate_size": 1,
            "routed_scaling_factor": 2.5,
            "sliding_window": 512,
            "rms_norm_eps": 1.0e-6,
            "rope_parameters": ROPE_PARAMETERS,
            "vocab_size": 7,
            "tie_word_embeddings": False,
        },
        "weight_generator": {
            "algorithm": "capture_laguna_cpu_reference._parameter_array/v1",
            "dtype": "float32",
            "parameter_sha256": parameter_sha256,
        },
        "tiny_model": tiny_model,
        "production_rope": _capture_rope(),
        "mask_boundaries": _capture_masks(),
        "absolute_position_ring": {
            "query_position": 513,
            "physical_slot_token_positions": [512, 513, *range(2, 512)],
            "sliding_window": 512,
            "expected_visible_physical_slots": list(range(512)),
        },
        "tolerances": {
            "atol": 2.0e-5,
            "rtol": 2.0e-5,
            "rope_atol": 2.0e-5,
            "rope_rtol": 2.0e-5,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = capture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
