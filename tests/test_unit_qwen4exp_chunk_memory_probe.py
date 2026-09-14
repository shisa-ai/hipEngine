from scripts.qwen4exp_chunk_memory_probe import allocation_margins
import pytest


def test_allowance_separates_scratch_from_reserve():
    plan = dict(device_weight_bytes=100,staging_bytes=2,kv_bytes=10,
                index_bytes=3,runtime_state_bytes=5,scratch_bytes=20,
                reserve_bytes=30,required_bytes=170)
    assert allocation_margins(plan,130) == dict(
        explicit_components_bytes=120,observed_remainder_bytes=10,
        scratch_allowance_bytes=20,scratch_margin_bytes=10,
        including_reserve_margin_bytes=40)
    assert allocation_margins(plan,145)["scratch_margin_bytes"] == -5


def test_bounded_probe_context_does_not_silently_use_native():
    from scripts.qwen4exp_chunk_memory_probe import resolve_context_length

    assert resolve_context_length(None, 262144) == 262144
    assert resolve_context_length(4352, 262144) == 4352
    with pytest.raises(ValueError):
        resolve_context_length(0, 262144)


@pytest.mark.parametrize("hidden,ffn,expected_width", [(2560, 640, 2560), (512, 640, 1280)])
def test_probe_prepares_both_full_capacity_risk_queues(hidden, ffn, expected_width):
    from types import SimpleNamespace
    from scripts.qwen4exp_chunk_memory_probe import prepare_lazy_group_risk

    calls = []

    def ensure(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(nbytes=4), SimpleNamespace(
            nbytes=kwargs["compact_rows"] * kwargs["out_features_total"] * 4)

    runner = SimpleNamespace(
        config=SimpleNamespace(hidden_size=hidden, expert_feed_forward_length=ffn,
                               expert_used_count=10),
        prefill_chunk_size=2048, max_sequence_length=1024,
        gdn_prefill_scratch=SimpleNamespace(moe=SimpleNamespace(ensure_group_risk_buffers=ensure)),
        qsa_prefill_scratch=SimpleNamespace(moe=SimpleNamespace(ensure_group_risk_buffers=ensure)))
    records = prepare_lazy_group_risk(runner)
    assert calls == [{"compact_rows": 10240, "out_features_total": expected_width}] * 2
    assert [row["owner"] for row in records] == ["gdn_prefill_scratch", "qsa_prefill_scratch"]
    assert sum(row["nbytes"] for row in records) == 2 * (4 + 10240 * expected_width * 4)
