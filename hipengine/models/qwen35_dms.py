"""Qwen3.5/3.8 model-plugin capability for external DMS decisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from hipengine.kvcache.dms import DMSRetrofitConfig


@dataclass(frozen=True, slots=True)
class Qwen35DMSDecisionCapability:
    model_family: str
    decision_source: str
    input_stage: str
    physical_layer_ids: tuple[int, ...]
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    prefix_mode: str = "unsupported"
    speculative_modes: tuple[str, ...] = ()
    strict_fallback: str = "paged_dense_bf16"
    span_roles: tuple[str, ...] = ("prefill", "decode")

    @property
    def fingerprint(self) -> str:
        payload = {
            "model_family": self.model_family,
            "decision_source": self.decision_source,
            "input_stage": self.input_stage,
            "physical_layer_ids": list(self.physical_layer_ids),
            "hidden_size": self.hidden_size,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "prefix_mode": self.prefix_mode,
            "speculative_modes": list(self.speculative_modes),
            "strict_fallback": self.strict_fallback,
            "span_roles": list(self.span_roles),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def require_span_role(self, span_role: str) -> None:
        role = str(span_role)
        if role not in self.span_roles:
            raise ValueError(
                f"external DMS decision source has unsupported span role {role!r}; "
                f"strict fallback is {self.strict_fallback}"
            )


_CAPABILITIES: dict[tuple[str, str], Qwen35DMSDecisionCapability] = {}


def register_qwen35_dms_decision_capability(
    capability: Qwen35DMSDecisionCapability,
    *,
    replace: bool = False,
) -> None:
    key = (capability.model_family, capability.decision_source)
    if key in _CAPABILITIES and not replace:
        raise ValueError(f"Qwen DMS decision capability already registered: {key}")
    _CAPABILITIES[key] = capability


def resolve_qwen35_dms_decision_capability(
    config: DMSRetrofitConfig,
    *,
    layer_types: Sequence[str],
    hidden_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Qwen35DMSDecisionCapability:
    key = (config.model_family, config.decision_source)
    try:
        template = _CAPABILITIES[key]
    except KeyError as exc:
        raise ValueError(f"no Qwen DMS decision capability for {key}") from exc
    physical_layer_ids = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if str(layer_type) == "full_attention"
    )
    if physical_layer_ids != config.physical_layer_ids:
        raise ValueError(
            "Qwen DMS physical layer map does not match the model plugin capability"
        )
    if config.input_stage != template.input_stage:
        raise ValueError("Qwen DMS input stage does not match the model plugin capability")
    model_geometry = (
        int(hidden_size),
        int(num_q_heads),
        int(num_kv_heads),
        int(head_dim),
    )
    sidecar_geometry = (
        int(config.hidden_size),
        config.num_q_heads,
        config.num_kv_heads,
        config.head_dim,
    )
    if model_geometry != sidecar_geometry:
        raise ValueError(
            f"Qwen DMS sidecar geometry {sidecar_geometry} does not match model {model_geometry}"
        )
    if config.borrowed_query_channel is not None or config.zero_borrowed_query_channel:
        raise ValueError("external Qwen DMS capability must preserve ordinary Q channels")
    return Qwen35DMSDecisionCapability(
        model_family=template.model_family,
        decision_source=template.decision_source,
        input_stage=template.input_stage,
        physical_layer_ids=physical_layer_ids,
        hidden_size=int(hidden_size),
        num_q_heads=int(num_q_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        prefix_mode=template.prefix_mode,
        speculative_modes=template.speculative_modes,
        strict_fallback=template.strict_fallback,
        span_roles=template.span_roles,
    )


register_qwen35_dms_decision_capability(
    Qwen35DMSDecisionCapability(
        model_family="qwen35_dense_hybrid",
        decision_source="external_linear_sidecar_v1",
        input_stage="post_attn_rmsnorm_pre_q_projection",
        physical_layer_ids=(),
        hidden_size=0,
        num_q_heads=0,
        num_kv_heads=0,
        head_dim=0,
    )
)


__all__ = [
    "Qwen35DMSDecisionCapability",
    "register_qwen35_dms_decision_capability",
    "resolve_qwen35_dms_decision_capability",
]
