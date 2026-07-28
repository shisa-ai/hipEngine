from __future__ import annotations

import numpy as np

from scripts.laguna_f16_decode_q8_full_gate import (
    _classify_source_name,
    _kl_divergence,
    _scope_layers,
    _summarize_records,
)


def test_laguna_f16_decode_q8_full_gate_classifies_projection_slots() -> None:
    assert _classify_source_name("blk.0.attn_q.weight") == "qkv"
    assert _classify_source_name("blk.47.attn_gate.weight") == "gate"
    assert _classify_source_name("blk.23.attn_output.weight") == "output"
    assert _classify_source_name("blk.1.ffn_gate_exps.weight") is None


def test_laguna_f16_decode_q8_full_gate_metrics_enforce_max_kl_and_top1() -> None:
    logits = np.asarray([0.0, 2.0, -1.0], dtype=np.float32)
    assert _kl_divergence(logits, logits) == 0.0
    records = [
        {"step": 0, "kl": 0.01, "top1_match": True},
        {"step": 1, "kl": 0.051, "top1_match": True},
    ]
    summary = _summarize_records(records, kl_limit=0.05, top1_minimum=0.9)

    assert summary["maximum_kl"] == 0.051
    assert summary["top1_agreement"] == 1.0
    assert summary["passed"] is False


def test_laguna_f16_decode_q8_full_gate_scopes_are_structural() -> None:
    attention_types = tuple(
        "full_attention" if layer % 4 == 0 else "sliding_attention"
        for layer in range(48)
    )

    assert _scope_layers("full", attention_types) == frozenset(range(0, 48, 4))
    assert _scope_layers("swa", attention_types) == (
        frozenset(range(48)) - frozenset(range(0, 48, 4))
    )
    assert _scope_layers("even", attention_types) == frozenset(range(0, 48, 2))
    assert _scope_layers("first24", attention_types) == frozenset(range(24))
    assert _scope_layers("mod3", attention_types) == frozenset(range(3, 48, 4))
