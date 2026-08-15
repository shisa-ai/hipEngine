"""Architecture/geometry identities shared by backend capability policies.

Human-facing model names are provenance, not dispatch keys.  Backend packages
use these immutable identities to admit only model families with the exact
weight/state topology that was gated, while compatible finetunes and renamed
exports inherit the same policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GGUFModelGeometry:
    """GGUF execution topology relevant to layout, scratch, and graph policy."""

    architecture: str
    block_count: int
    hidden_size: int
    vocab_size: int
    feed_forward_length: int
    head_count: int
    head_count_kv: int
    key_length: int
    value_length: int
    full_attention_interval: int
    layer_types: tuple[str, ...]
    ssm_inner_size: int
    ssm_group_count: int
    ssm_state_size: int
    ssm_conv_kernel: int
    ssm_time_step_rank: int
    expert_count: int
    expert_used_count: int
    expert_feed_forward_length: int
    expert_shared_feed_forward_length: int

    @classmethod
    def from_config(cls, config: object) -> "GGUFModelGeometry":
        """Build a fail-closed policy key from validated GGUF configuration."""

        return cls(
            architecture=str(getattr(config, "architecture")),
            block_count=int(getattr(config, "block_count")),
            hidden_size=int(getattr(config, "hidden_size")),
            vocab_size=int(getattr(config, "vocab_size")),
            feed_forward_length=int(getattr(config, "feed_forward_length")),
            head_count=int(getattr(config, "head_count")),
            head_count_kv=int(getattr(config, "head_count_kv")),
            key_length=int(getattr(config, "key_length")),
            value_length=int(getattr(config, "value_length")),
            full_attention_interval=int(getattr(config, "full_attention_interval")),
            layer_types=tuple(str(value) for value in getattr(config, "layer_types")),
            ssm_inner_size=int(getattr(config, "ssm_inner_size")),
            ssm_group_count=int(getattr(config, "ssm_group_count")),
            ssm_state_size=int(getattr(config, "ssm_state_size")),
            ssm_conv_kernel=int(getattr(config, "ssm_conv_kernel")),
            ssm_time_step_rank=int(getattr(config, "ssm_time_step_rank")),
            expert_count=int(getattr(config, "expert_count", 0)),
            expert_used_count=int(getattr(config, "expert_used_count", 0)),
            expert_feed_forward_length=int(
                getattr(config, "expert_feed_forward_length", 0)
            ),
            expert_shared_feed_forward_length=int(
                getattr(config, "expert_shared_feed_forward_length", 0)
            ),
        )

    @classmethod
    def try_from_config(cls, config: object) -> "GGUFModelGeometry | None":
        """Return no policy identity for incomplete compatibility fixtures."""

        try:
            return cls.from_config(config)
        except (AttributeError, TypeError, ValueError):
            return None


def _qwen35_layer_types(block_count: int, full_attention_interval: int) -> tuple[str, ...]:
    return tuple(
        "full_attention"
        if (index + 1) % int(full_attention_interval) == 0
        else "linear_attention"
        for index in range(int(block_count))
    )


QWEN35_DENSE_H5120_GEOMETRY = GGUFModelGeometry(
    architecture="qwen35",
    block_count=64,
    hidden_size=5_120,
    vocab_size=248_320,
    feed_forward_length=17_408,
    head_count=24,
    head_count_kv=4,
    key_length=256,
    value_length=256,
    full_attention_interval=4,
    layer_types=_qwen35_layer_types(64, 4),
    ssm_inner_size=6_144,
    ssm_group_count=16,
    ssm_state_size=128,
    ssm_conv_kernel=4,
    ssm_time_step_rank=48,
    expert_count=0,
    expert_used_count=0,
    expert_feed_forward_length=0,
    expert_shared_feed_forward_length=0,
)

QWEN35_MOE_H2048_E256_GEOMETRY = GGUFModelGeometry(
    architecture="qwen35moe",
    block_count=40,
    hidden_size=2_048,
    vocab_size=248_320,
    feed_forward_length=512,
    head_count=16,
    head_count_kv=2,
    key_length=256,
    value_length=256,
    full_attention_interval=4,
    layer_types=_qwen35_layer_types(40, 4),
    ssm_inner_size=4_096,
    ssm_group_count=16,
    ssm_state_size=128,
    ssm_conv_kernel=4,
    ssm_time_step_rank=32,
    expert_count=256,
    expert_used_count=8,
    expert_feed_forward_length=512,
    expert_shared_feed_forward_length=512,
)


__all__ = [
    "GGUFModelGeometry",
    "QWEN35_DENSE_H5120_GEOMETRY",
    "QWEN35_MOE_H2048_E256_GEOMETRY",
]
