from scripts.qwen4exp_chunk_memory_probe import allocation_margins


def test_allowance_separates_scratch_from_reserve():
    plan = dict(device_weight_bytes=100,staging_bytes=2,kv_bytes=10,
                index_bytes=3,runtime_state_bytes=5,scratch_bytes=20,
                reserve_bytes=30,required_bytes=170)
    assert allocation_margins(plan,130) == dict(
        explicit_components_bytes=120,observed_remainder_bytes=10,
        scratch_allowance_bytes=20,scratch_margin_bytes=10,
        including_reserve_margin_bytes=40)
    assert allocation_margins(plan,145)["scratch_margin_bytes"] == -5
