from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.laguna_f16_decode_q8_full_gate import (
    _classify_source_name,
    _kl_divergence,
    _mode_owns,
    _q8_weight_only_decode_owner,
    _scope_layers,
    _summarize_records,
)


def test_laguna_f16_decode_q8_full_gate_classifies_projection_slots() -> None:
    assert _classify_source_name("blk.0.attn_q.weight") == "qkv"
    assert _classify_source_name("blk.47.attn_gate.weight") == "gate"
    assert _classify_source_name("blk.23.attn_output.weight") == "output"
    assert _classify_source_name("blk.1.ffn_gate_exps.weight") is None


def test_laguna_f16_decode_q8_full_gate_modes_isolate_qkv_roles() -> None:
    assert _mode_owns("all", "q")
    assert _mode_owns("qkv_gate", "q")
    assert _mode_owns("qkv_gate", "gate")
    assert _mode_owns("q", "q")
    assert not _mode_owns("q", "k")
    assert not _mode_owns("gate", "q")


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


def test_laguna_f16_decode_q8_weight_only_owner_keeps_activations_exact(
    monkeypatch,
) -> None:
    import hipengine.runtime.laguna_gguf_runner as runner
    import scripts.laguna_f16_decode_q8_full_gate as gate

    calls: list[tuple[object, ...]] = []

    def exact_single(*args, **kwargs):
        calls.append(("exact_single", args, kwargs))

    def exact_triple(*args, **kwargs):
        calls.append(("exact_triple", args, kwargs))

    def raw_q8(*args, **kwargs):
        calls.append(("raw_q8", args, kwargs))

    def t16_q8(*args, **kwargs):
        calls.append(("t16_q8", args, kwargs))

    monkeypatch.setattr(runner, "launch_f16_weight_linear", exact_single)
    monkeypatch.setattr(runner, "launch_f16_weight_linear_triple", exact_triple)
    monkeypatch.setattr(gate, "gguf_q8_0_gemv_bf16_f32_out", raw_q8)
    monkeypatch.setattr(
        gate,
        "gguf_q8_0_t16_gemv_decode_f32_bf16_out",
        t16_q8,
    )

    def weight(name: str):
        return SimpleNamespace(
            spec=SimpleNamespace(source=SimpleNamespace(name=name))
        )

    q = weight("blk.0.attn_q.weight")
    k = weight("blk.0.attn_k.weight")
    v = weight("blk.0.attn_v.weight")
    gate_weight = weight("blk.0.attn_gate.weight")
    output = weight("blk.0.attn_output.weight")
    sidecar = SimpleNamespace(
        raw_weights={
            item.spec.source.name: SimpleNamespace(ptr=ptr)
            for item, ptr in (
                (q, 101),
                (k, 102),
                (v, 103),
                (gate_weight, 104),
                (output, 105),
            )
        },
        output_t16_weights={
            output.spec.source.name: SimpleNamespace(ptr=205)
        },
    )
    owner = SimpleNamespace(
        runtime=object(),
        use_f16_projection_head_kv_decode=True,
        use_f16_attention_quad_decode=True,
        use_f16_output_add_rmsnorm_decode=True,
    )

    with _q8_weight_only_decode_owner(
        owner,
        sidecar,
        "all",
        frozenset({0}),
        raw_library=object(),
        t16_library=object(),
    ) as counters:
        assert owner.use_f16_projection_head_kv_decode is False
        assert owner.use_f16_attention_quad_decode is False
        assert owner.use_f16_output_add_rmsnorm_decode is False
        runner.launch_f16_weight_linear(
            gate_weight,
            11,
            21,
            1,
            2048,
            2048,
            stream=7,
        )
        runner.launch_f16_weight_linear(
            output,
            12,
            22,
            1,
            2048,
            3072,
            stream=7,
        )
        runner.launch_f16_weight_linear_triple(
            q,
            k,
            v,
            13,
            23,
            24,
            25,
            1,
            3072,
            3072,
            512,
            512,
            stream=7,
        )
        runner.launch_f16_weight_linear(
            gate_weight,
            14,
            26,
            2,
            2048,
            2048,
            stream=7,
        )

    raw_calls = [call for call in calls if call[0] == "raw_q8"]
    t16_calls = [call for call in calls if call[0] == "t16_q8"]
    assert [call[1][1] for call in raw_calls] == [104, 101, 102, 103]
    assert all(call[1][0] in (11, 13) for call in raw_calls)
    assert len(t16_calls) == 1
    assert t16_calls[0][1][:3] == (12, 205, 22)
    assert calls[-1][0] == "exact_single"
    assert counters == {
        "q8_weight_gate": 1,
        "q8_weight_output": 1,
        "q8_weight_qkv": 3,
        "exact_single": 1,
        "exact_triple": 0,
    }
    assert owner.use_f16_projection_head_kv_decode is True
    assert owner.use_f16_attention_quad_decode is True
    assert owner.use_f16_output_add_rmsnorm_decode is True

    calls.clear()
    with _q8_weight_only_decode_owner(
        owner,
        sidecar,
        "q",
        frozenset({0}),
        raw_library=object(),
        t16_library=object(),
    ) as counters:
        runner.launch_f16_weight_linear_triple(
            q,
            k,
            v,
            13,
            23,
            24,
            25,
            1,
            3072,
            3072,
            512,
            512,
            stream=7,
        )

    assert [call[0] for call in calls] == [
        "raw_q8",
        "exact_single",
        "exact_single",
    ]
    assert counters == {
        "q8_weight_gate": 0,
        "q8_weight_output": 0,
        "q8_weight_qkv": 1,
        "exact_single": 2,
        "exact_triple": 0,
    }
