"""Shared CPU-only fixture builders for the Qwen4-Exp GGUF config tests.

Kept out of ``test_*`` modules so ``test_unit_*`` files can import them without
depending on another test module (see docs/TESTING.md, "Test Naming and
Discovery": keep shared fixture builders in non-test helper modules).
"""

from __future__ import annotations

from pathlib import Path

from hipengine.loading.gguf import GGUFModelInfo


def _metadata() -> dict[str, object]:
    head_sizes = (
        20_000_003,
        20_000_023,
        20_000_033,
        20_000_047,
        20_000_059,
        20_000_063,
        20_000_069,
        20_000_077,
        20_000_081,
        20_000_093,
        20_000_107,
        20_000_147,
        20_000_153,
        20_000_159,
        20_000_161,
        20_000_171,
    )
    offsets: list[int] = []
    offset = 0
    for size in head_sizes:
        offsets.append(offset)
        offset += size
    return {
        "general.architecture": "qwen4exp",
        "qwen4exp.block_count": 48,
        "qwen4exp.context_length": 262_144,
        "qwen4exp.embedding_length": 2_560,
        "qwen4exp.hyper_connection.count": 4,
        "qwen4exp.hyper_connection.low_rank": 320,
        "qwen4exp.full_attention_interval": 4,
        "qwen4exp.attention.compress_ratios": [0, 0, 0, 4] * 12,
        "qwen4exp.attention.head_count": 24,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 256,
        "qwen4exp.attention.value_length": 256,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen4exp.attention.indexer.head_count": 4,
        "qwen4exp.attention.indexer.key_length": 128,
        "qwen4exp.attention.indexer.top_k": 2_048,
        "qwen4exp.rope.dimension_count": 64,
        "qwen4exp.rope.dimension_sections": [11, 11, 10, 0],
        "qwen4exp.rope.freq_base": 10_000_000.0,
        "qwen4exp.ssm.conv_kernel": 4,
        "qwen4exp.ssm.group_count": 16,
        "qwen4exp.ssm.inner_size": 6_144,
        "qwen4exp.ssm.state_size": 128,
        "qwen4exp.ssm.time_step_rank": 48,
        "qwen4exp.expert_count": 512,
        "qwen4exp.expert_used_count": 10,
        "qwen4exp.expert_feed_forward_length": 640,
        "qwen4exp.expert_shared_feed_forward_length": 640,
        "qwen4exp.ple.layers": [1],
        "qwen4exp.ple.ngram_size": 3,
        "qwen4exp.ple.heads_per_ngram": 8,
        "qwen4exp.ple.conv_kernel": 4,
        "qwen4exp.ple.eos_token_id": 248_044,
        "qwen4exp.ple.image_token_id": 248_056,
        "qwen4exp.embedding_length_per_layer_input": 160,
        "qwen4exp.ple.layer_multipliers": [
            23_703_573_157_769,
            20_109_073_645_365,
            8_052_911_324_071,
        ],
        "qwen4exp.ple.head_offsets": offsets,
        "qwen4exp.ple.head_vocab_sizes": list(head_sizes),
        "tokenizer.ggml.tokens": [None] * 248_320,
        "tokenizer.ggml.bos_token_id": 248_044,
        "tokenizer.ggml.eos_token_id": 248_046,
        "tokenizer.ggml.padding_token_id": 248_044,
    }


def _info(metadata: dict[str, object] | None = None) -> GGUFModelInfo:
    return GGUFModelInfo(
        path=Path("synthetic-qwen4exp.gguf"),
        version=3,
        alignment=32,
        metadata=_metadata() if metadata is None else metadata,
        tensors=(),
        tensor_data_offset=0,
    )
