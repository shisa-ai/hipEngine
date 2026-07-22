from __future__ import annotations

from pathlib import Path
from typing import Any

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo


def laguna_metadata() -> dict[str, Any]:
    return {
        "general.architecture": "laguna",
        "general.file_type": 15,
        "general.quantization_version": 2,
        "laguna.attention.head_count": [48, 72, 72, 72] * 12,
        "laguna.attention.head_count_kv": 8,
        "laguna.attention.key_length": 128,
        "laguna.attention.layer_norm_rms_epsilon": 1.0e-6,
        "laguna.attention.sliding_window": 512,
        "laguna.attention.value_length": 128,
        "laguna.block_count": 48,
        "laguna.context_length": 262_144,
        "laguna.embedding_length": 3_072,
        "laguna.expert_count": 256,
        "laguna.expert_feed_forward_length": 1_024,
        "laguna.expert_gating_func": 2,
        "laguna.expert_shared_feed_forward_length": 1_024,
        "laguna.expert_used_count": 10,
        "laguna.expert_weights_norm": True,
        "laguna.expert_weights_scale": 2.5,
        "laguna.feed_forward_length": 12_288,
        "laguna.leading_dense_block_count": 1,
        "laguna.rope.dimension_count": 64,
        "laguna.rope.dimension_count_swa": 128,
        "laguna.rope.freq_base": 500_000.0,
        "laguna.rope.freq_base_swa": 10_000.0,
        "laguna.rope.scaling.factor": 32.0,
        "laguna.rope.scaling.original_context_length": 8_192,
        "laguna.rope.scaling.type": "yarn",
        "laguna.rope.scaling.yarn_attn_factor": 1.0,
        "laguna.rope.scaling.yarn_beta_fast": 32.0,
        "laguna.rope.scaling.yarn_beta_slow": 1.0,
        "laguna.vocab_size": 100_352,
    }


def make_laguna_info(
    *,
    metadata: dict[str, Any] | None = None,
    tensors: tuple[GGUFTensorInfo, ...] = (),
) -> GGUFModelInfo:
    return GGUFModelInfo(
        path=Path("/synthetic/laguna-s-2.1-Q4_K_M.gguf"),
        version=3,
        alignment=32,
        metadata=laguna_metadata() if metadata is None else metadata,
        tensors=tensors,
        tensor_data_offset=0,
    )
