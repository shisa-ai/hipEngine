from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import _GGUFFullAttentionPrefillScratch

_MIB = 1 << 20


def _fake_runner(backend: str = "hip_gfx1100") -> SimpleNamespace:
    cfg = SimpleNamespace(
        expert_used_count=8,
        is_moe=True,
        expert_count=256,
        expert_shared_feed_forward_length=512,
        ssm_inner_size=4096,
        ssm_conv_kernel=4,
        ssm_time_step_rank=32,
        ssm_state_size=128,
        ssm_group_count=16,
        head_count_kv=2,
        key_length=256,
        rope_dimension_count=64,
        rope_freq_base=10_000_000.0,
        head_count=16,
    )
    return SimpleNamespace(
        backend=backend,
        hidden_size=2048,
        q_width=4096,
        kv_width=512,
        ffn_size=512,
        linear_qkv_width=8192,
        ssm_value_dim=128,
        weights=SimpleNamespace(config=cfg),
    )


def _fake_dense_qwen36_runner() -> SimpleNamespace:
    cfg = SimpleNamespace(
        expert_used_count=0,
        is_moe=False,
        expert_count=0,
        expert_shared_feed_forward_length=0,
        ssm_inner_size=6144,
        ssm_conv_kernel=4,
        ssm_time_step_rank=48,
        ssm_state_size=128,
        ssm_group_count=16,
        head_count_kv=4,
        key_length=256,
        rope_dimension_count=64,
        rope_freq_base=10_000_000.0,
        head_count=24,
    )
    return SimpleNamespace(
        backend="hip_gfx1100",
        hidden_size=5120,
        q_width=6144,
        kv_width=1024,
        ffn_size=17408,
        linear_qkv_width=10240,
        ssm_value_dim=128,
        weights=SimpleNamespace(
            config=cfg,
            model_name="Qwen3.6-27B",
            file_type_name="MOSTLY_Q4_K_M",
        ),
    )


def _install_fake_device(monkeypatch):
    next_ptr = 0x10000000
    allocations: list[DeviceBuffer] = []

    def fake_malloc(nbytes: int, *, runtime):
        nonlocal next_ptr
        size = int(nbytes)
        buffer = DeviceBuffer(ptr=next_ptr, nbytes=size)
        next_ptr += max(256, ((size + 255) // 256) * 256 + 256)
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    monkeypatch.setattr(gguf_runner, "copy_host_to_device", lambda *args, **kwargs: None)
    return allocations


def _clear_diagnostic_environment(monkeypatch) -> None:
    for name in tuple(gguf_runner.os.environ):
        if name.startswith("HIPENGINE_GGUF_VERIFY_") or name in {
            "HIPENGINE_GGUF_GDN_PREFILL_MODE",
            "HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A",
            "HIPENGINE_GGUF_T16_SELECTED_DP4A",
            "HIPENGINE_GGUF_RAW_SELECTED_DP4A",
            "HIPENGINE_GGUF_DENSE_Q8_DP4A",
            "HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL",
            "HIPENGINE_GGUF_DENSE_Q8_DP4A_SHARED",
            "HIPENGINE_GGUF_DENSE_Q8_DP4A_F32",
        }:
            monkeypatch.delenv(name, raising=False)


def test_unequal_q4_pair_owner_is_model_backend_scoped(monkeypatch) -> None:
    runner = _fake_dense_qwen36_runner()
    assert gguf_runner._gguf_q4_t16_unequal_pair_prefill_applies(runner)
    runner.backend = "hip_gfx1151"
    assert not gguf_runner._gguf_q4_t16_unequal_pair_prefill_applies(runner)
    runner.backend = "hip_gfx1100"
    runner.weights.model_name = "other"
    assert not gguf_runner._gguf_q4_t16_unequal_pair_prefill_applies(runner)
    runner.weights.model_name = "Qwen3.6-27B"
    runner.weights.config.is_moe = True
    assert not gguf_runner._gguf_q4_t16_unequal_pair_prefill_applies(runner)
    runner.weights.config.is_moe = False
    monkeypatch.setattr(gguf_runner, "backend_package_capability", lambda *args: {})
    assert not gguf_runner._gguf_q4_t16_unequal_pair_prefill_applies(runner)
    monkeypatch.setattr(gguf_runner, "backend_package_capability", lambda *args: True)
    assert not gguf_runner._gguf_q4_t16_unequal_pair_prefill_applies(runner)


def test_gfx1100_dense_qwen36_prefill_scratch_uses_model_scoped_liveness_arena(
    monkeypatch,
) -> None:
    _install_fake_device(monkeypatch)
    _clear_diagnostic_environment(monkeypatch)

    scratch = _GGUFFullAttentionPrefillScratch.allocate(
        _fake_dense_qwen36_runner(),
        rows=768,
        capacity=768,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )

    assert scratch.allocation_mode == "liveness_aliased"
    source_f16_policy = gguf_runner._gguf_t16_f16_rocblas_prefill_policy(
        _fake_dense_qwen36_runner()
    )
    assert source_f16_policy is not None
    assert source_f16_policy["gguf_q5_k_t16_v1"][(6_144, 5_120)] == {
        512: 1_280,
        1_024: 1_280,
        4_096: 1_024,
    }
    assert source_f16_policy["gguf_q4_k_t16_v1"][(5_120, 12_288)] == {
        512: 2_048,
        1_024: 512,
    }
    assert source_f16_policy["max_rows_by_quant_shape"] == {
        "gguf_q4_k_t16_v1": {(5_120, 12_288): 2_047},
    }
    assert source_f16_policy["pair_only_second_operand_policies"] == {
        (
            "gguf_q6_k_t16_qmicro_planar_v1",
            5_120,
            10_240,
            "gguf_q4_k_t16_v1",
            6_144,
        ): {
            (512, 1_023): (
                2_048,
                "f16_rocblas_t16_pair_bf16_bf16_out",
                False,
            ),
            (1_024, 2_047): (
                512,
                "f16_rocblas_t16_pair_bf16_bf16_out",
                False,
            ),
        },
    }
    assert source_f16_policy["linear_variant_intervals_by_quant"] == {
        "gguf_q4_k_t16_v1": {
            (17_408, 5_120): {
                (512, 4_096): "f16_rocblas_t16_pair_bf16_bf16_out",
            },
            (5_120, 1_024): {
                (512, 1_024): "f16_rocblas_t16_pair_bf16_bf16_out",
                (4_096, 4_096): "f16_rocblas_t16_pair_bf16_bf16_out",
            },
            (6_144, 5_120): {
                (512, 768): "f16_rocblas_t16_pair_bf16_bf16_out",
            },
            (5_120, 12_288): {
                (512, 2_047): "f16_rocblas_t16_pair_bf16_bf16_out",
            },
        },
        "gguf_q5_k_t16_v1": {
            (6_144, 5_120): {
                (512, 4_096): "f16_rocblas_t16_octet_bf16_bf16_out",
            },
        },
    }
    # Q4/Q5/Q6 source-F16 prefill shares three liveness-aliased transient
    # planes while preserving each sole resident T16 weight allocation. Q5
    # recurrent output and dense FFN down cast their dead BF16 inputs in place,
    # so K6,144 admission does not grow the K5,120 activation workspace.
    assert sum(buffer.nbytes for buffer in scratch.buffers) <= 104 * _MIB
    assert max(buffer.nbytes for buffer in scratch.buffers) <= 103 * _MIB
    assert scratch.q6_f16_x.ptr != 0
    assert scratch.q6_f16_x.nbytes == 768 * 5_120 * 2
    assert scratch.q6_f16_weight.ptr != 0
    assert scratch.q6_f16_out.ptr != 0
    # The package-default compact-peer GDN route stores normalized Q/K once
    # per K head, while V remains per V head.
    compact_qk_bytes = 768 * 16 * 128 * 4
    value_bytes = 768 * 48 * 128 * 4
    assert scratch.prefill_query.nbytes == compact_qk_bytes
    assert scratch.prefill_key.nbytes == compact_qk_bytes
    assert scratch.prefill_value.nbytes == value_bytes
    assert scratch.ffn_gate_up.ptr != 0
    # Dense SiLU reads gate/up before replacing the gate half in place, so the
    # down-projection input reuses the dead gate plane exactly.
    assert scratch.ffn_intermediate.ptr == scratch.ffn_gate_up.ptr
    assert scratch.ffn_intermediate.nbytes * 2 == scratch.ffn_gate_up.nbytes
    assert scratch.ffn_down.ptr != 0
    assert scratch.moe_q8_1 == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.moe_shared_gate == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.moe_shared_intermediate == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.moe_down_out == DeviceBuffer(ptr=0, nbytes=0)
    offsets = dict(scratch.allocation_offsets)
    lifetimes = dict(scratch.allocation_lifetimes)
    assert offsets
    gate_up_offset, gate_up_size = offsets["ffn_gate_up"]
    intermediate_offset, intermediate_size = offsets["ffn_intermediate"]
    down_offset, down_size = offsets["ffn_down"]
    assert (intermediate_offset, intermediate_size) == (
        gate_up_offset,
        gate_up_size // 2,
    )
    assert dict(scratch.allocation_inplace_aliases) == {
        "ffn_intermediate": "ffn_gate_up"
    }
    assert (
        intermediate_offset + intermediate_size <= down_offset
        or down_offset + down_size <= intermediate_offset
    )
    # Attention-output F16 arithmetic owns full-attention stage 5-6. Q5
    # recurrent-output F16 arithmetic also owns the shared weight/output planes
    # at linear stage 5-6 while casting its dead K6,144 input in place.
    for name in ("q6_f16_x", "q6_f16_weight", "q6_f16_out"):
        assert ("full", 5, 6) in lifetimes[name]
    for name in ("q6_f16_weight", "q6_f16_out"):
        assert ("linear", 5, 6) in lifetimes[name]

    entries = list(offsets.items())
    for index, (name_a, (offset_a, size_a)) in enumerate(entries):
        for name_b, (offset_b, size_b) in entries[index + 1 :]:
            allocations_conflict = gguf_runner._prefill_scratch_allocations_conflict(
                name_a,
                offset_a,
                size_a,
                name_b,
                offset_b,
                size_b,
                lifetimes=lifetimes,
                allocation_subranges=scratch.allocation_subranges,
            )
            inplace_aliases = dict(scratch.allocation_inplace_aliases)
            intentional_inplace_alias = (
                inplace_aliases.get(name_a) == name_b
                or inplace_aliases.get(name_b) == name_a
            )
            assert not (allocations_conflict and not intentional_inplace_alias), (
                f"live dense scratch buffers overlap: {name_a}={offsets[name_a]}, "
                f"{name_b}={offsets[name_b]}"
            )


def test_gfx1100_dense_qwen36_hidden_reuse_is_long_row_only(monkeypatch) -> None:
    allocations = _install_fake_device(monkeypatch)
    runner = _fake_dense_qwen36_runner()

    short_a, short_b = gguf_runner._allocate_prefill_hidden_buffers(
        runner,
        rows=64,
        nbytes=64 * 5_120 * 2,
        runtime=SimpleNamespace(),
    )
    assert short_a.ptr != short_b.ptr
    assert len(allocations) == 2

    long_a, long_b = gguf_runner._allocate_prefill_hidden_buffers(
        runner,
        rows=4_096,
        nbytes=4_096 * 5_120 * 2,
        runtime=SimpleNamespace(),
    )
    assert long_a.ptr == long_b.ptr
    assert len(allocations) == 3


def test_gfx1100_dense_qwen36_recoloring_is_long_row_only() -> None:
    runner = _fake_dense_qwen36_runner()

    assert (
        gguf_runner._gguf_prefill_scratch_priority_min_live_stages(
            runner,
            rows=64,
        )
        is None
    )
    assert gguf_runner._gguf_prefill_scratch_priority_min_live_stages(
        runner,
        rows=4_096,
    ) == 5


def test_gfx1100_dense_qwen36_split_gdn_reuses_dead_conv_output_scratch(
    monkeypatch,
) -> None:
    _install_fake_device(monkeypatch)
    _clear_diagnostic_environment(monkeypatch)

    scratch = _GGUFFullAttentionPrefillScratch.allocate(
        _fake_dense_qwen36_runner(),
        rows=4_096,
        capacity=4_352,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )

    assert scratch.rows == 4_096
    assert scratch.linear_qkv_f32.nbytes == 4_096 * 10_240 * 4
    assert scratch.full_query_raw.nbytes == 4_096 * 6_144 * 4
    assert scratch.ffn_gate_up.nbytes == 2 * 4_096 * 17_408 * 2
    assert scratch.ffn_intermediate.ptr == scratch.ffn_gate_up.ptr
    value_bytes = 4_096 * 48 * 128 * 4
    assert scratch.prefill_value.nbytes == value_bytes
    assert scratch.recurrent_out.nbytes == value_bytes
    assert scratch.allocation_lifetimes["conv_out"] == (("linear", 2, 4),)
    conv_offset, conv_bytes = scratch.allocation_offsets["conv_out"]
    recurrent_offset, recurrent_bytes = scratch.allocation_offsets["recurrent_out"]
    assert max(conv_offset, recurrent_offset) < min(
        conv_offset + conv_bytes,
        recurrent_offset + recurrent_bytes,
    )
    assert sum(buffer.nbytes for buffer in scratch.buffers) <= 380 * _MIB
    assert max(buffer.nbytes for buffer in scratch.buffers) == 372 * _MIB + 384 * 1_024
    assert scratch.allocation_offsets["attn_out"] == (
        48 * _MIB + 64 * 1_024,
        40 * _MIB,
    )


def test_gfx1100_explicit_peer_gdn_keeps_full_qk_scratch_fallback(monkeypatch) -> None:
    _install_fake_device(monkeypatch)
    _clear_diagnostic_environment(monkeypatch)
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", "chain_peer_wave32")

    scratch = _GGUFFullAttentionPrefillScratch.allocate(
        _fake_dense_qwen36_runner(),
        rows=768,
        capacity=768,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )

    full_qkv_bytes = 768 * 48 * 128 * 4
    assert scratch.prefill_query.nbytes == full_qkv_bytes
    assert scratch.prefill_key.nbytes == full_qkv_bytes
    assert scratch.prefill_value.nbytes == full_qkv_bytes


@pytest.mark.parametrize(
    ("scratch_rows", "request_rows", "expected_intervals"),
    (
        (768, None, {(512, 1_023)}),
        (1_024, None, {(1_024, 2_047)}),
        (2_048, None, set()),
        (4_096, None, set()),
        (836, 517, {(512, 1_023)}),
        (2_176, 1_024, {(1_024, 2_047)}),
        (4_224, 2_048, set()),
        (4_224, 4_096, set()),
    ),
)
def test_gfx1100_pair_only_source_f16_owner_keeps_current_row_interval(
    monkeypatch,
    scratch_rows: int,
    request_rows: int | None,
    expected_intervals: set[tuple[int, int]],
) -> None:
    runner = _fake_dense_qwen36_runner()
    runner._cast_library = lambda: "cast-library"
    scratch = SimpleNamespace(
        rows=scratch_rows,
        q6_f16_x=DeviceBuffer(ptr=0x10000000, nbytes=1 << 40),
        q6_f16_weight=DeviceBuffer(ptr=0x20000000, nbytes=1 << 40),
        q6_f16_out=DeviceBuffer(ptr=0x30000000, nbytes=1 << 40),
    )
    fake_rocblas = SimpleNamespace(version_string=lambda: "unqualified")
    session = SimpleNamespace(
        runner=runner,
        _bulk_prefill_scratch=scratch,
        use_q6_f16_rocblas_prefill=None,
        _q6_f16_rocblas_prefill_library="dequant-library",
        _q6_f16_rocblas=fake_rocblas,
        compiler_version=None,
        require_cached_build=False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "q6_t16_f16_rocblas_prefill_session",
        lambda owner: owner,
    )

    owner = gguf_runner.Qwen35GGUFResidentSession._q6_f16_rocblas_prefill_context(
        session, request_rows=request_rows
    )

    pair_only = owner.pair_only_second_operand_policies
    assert pair_only is not None
    intervals = next(iter(pair_only.values()), {})
    assert set(intervals) == expected_intervals


def test_gfx1100_source_f16_owner_keeps_exact_fallback_beyond_scratch_rows(
    monkeypatch,
) -> None:
    runner = _fake_dense_qwen36_runner()
    scratch = SimpleNamespace(rows=768)
    session = SimpleNamespace(
        runner=runner,
        _bulk_prefill_scratch=scratch,
        use_q6_f16_rocblas_prefill=None,
    )
    fallback = object()
    monkeypatch.setattr(
        gguf_runner,
        "q6_t16_f16_rocblas_prefill_session",
        lambda owner: fallback if owner is None else owner,
    )

    owner = gguf_runner.Qwen35GGUFResidentSession._q6_f16_rocblas_prefill_context(
        session, request_rows=769
    )

    assert owner is fallback


def test_gfx1100_source_f16_owner_rejects_nonpositive_request_rows() -> None:
    session = SimpleNamespace(
        runner=_fake_dense_qwen36_runner(),
        _bulk_prefill_scratch=SimpleNamespace(rows=768),
        use_q6_f16_rocblas_prefill=None,
    )

    with pytest.raises(ValueError, match="request rows must be positive"):
        gguf_runner.Qwen35GGUFResidentSession._q6_f16_rocblas_prefill_context(
            session, request_rows=0
        )


def test_gfx1100_peer_prefill_scratch_uses_bounded_liveness_arena(monkeypatch) -> None:
    allocations = _install_fake_device(monkeypatch)
    _clear_diagnostic_environment(monkeypatch)

    scratch = _GGUFFullAttentionPrefillScratch.allocate(
        _fake_runner(),
        rows=4096,
        capacity=4352,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )

    assert scratch.allocation_mode == "liveness_aliased"
    assert sum(buffer.nbytes for buffer in scratch.buffers) <= 512 * _MIB
    assert max(buffer.nbytes for buffer in scratch.buffers) <= 512 * _MIB
    assert scratch.prefill_query.ptr != 0
    assert scratch.prefill_key.ptr != 0
    assert scratch.prefill_value.ptr != 0
    assert scratch.linear_z_f32 == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.moe_down_out_f32 == DeviceBuffer(ptr=0, nbytes=0)

    offsets = dict(scratch.allocation_offsets)
    lifetimes = dict(scratch.allocation_lifetimes)
    assert offsets
    assert offsets.keys() == lifetimes.keys()
    entries = list(offsets.items())
    for index, (name_a, (offset_a, size_a)) in enumerate(entries):
        for name_b, (offset_b, size_b) in entries[index + 1 :]:
            lifetimes_overlap = gguf_runner._prefill_scratch_lifetimes_overlap(
                lifetimes[name_a],
                lifetimes[name_b],
            )
            ranges_overlap = offset_a < offset_b + size_b and offset_b < offset_a + size_a
            assert not (lifetimes_overlap and ranges_overlap), (
                f"live scratch buffers overlap: {name_a}={offsets[name_a]}, "
                f"{name_b}={offsets[name_b]}"
            )

    # One arena owner plus small independently-owned metadata allocations.
    arena = max(allocations, key=lambda buffer: buffer.nbytes)
    assert arena in scratch.buffers
    for name, (offset, size) in offsets.items():
        field = getattr(scratch, name)
        assert field.ptr == arena.ptr + offset
        assert field.nbytes == size


@pytest.mark.parametrize(
    "mode",
    ("chain_lds32_direct", "chain_lds32_direct_nonvolatile", "exact"),
)
def test_gfx1100_explicit_exact_direct_liveness_omits_materialized_qkv(
    monkeypatch, mode: str
) -> None:
    _install_fake_device(monkeypatch)
    _clear_diagnostic_environment(monkeypatch)
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", mode)

    scratch = _GGUFFullAttentionPrefillScratch.allocate(
        _fake_runner(),
        rows=4096,
        capacity=4352,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )

    assert scratch.allocation_mode == "liveness_aliased"
    assert scratch.prefill_query == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.prefill_key == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.prefill_value == DeviceBuffer(ptr=0, nbytes=0)


def test_gfx1151_right_sized_short_scratch_uses_owner_slots(monkeypatch) -> None:
    _install_fake_device(monkeypatch)
    _clear_diagnostic_environment(monkeypatch)

    scratch = _GGUFFullAttentionPrefillScratch.allocate(
        _fake_runner("hip_gfx1151"),
        rows=768,
        capacity=768,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )

    assert scratch.allocation_mode == "liveness_aliased"
    assert sum(buffer.nbytes for buffer in scratch.buffers) == 69_790_760
    assert len(set(scratch.allocation_groups.values())) == 21
    assert scratch.moe_down_out_f32 == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.conv_out.ptr == scratch.moe_down_out.ptr
    assert scratch.linear_qkv_f32.ptr != scratch.conv_out.ptr
    assert scratch.allocation_offsets


def test_gfx1151_short_diagnostics_keep_dedicated_scratch_fallback(monkeypatch) -> None:
    _install_fake_device(monkeypatch)
    _clear_diagnostic_environment(monkeypatch)
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE", "1")

    scratch = _GGUFFullAttentionPrefillScratch.allocate(
        _fake_runner("hip_gfx1151"),
        rows=768,
        capacity=768,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )

    assert scratch.allocation_mode == "dedicated"
    assert sum(buffer.nbytes for buffer in scratch.buffers) == 355_182_664
    assert scratch.moe_down_out_f32.ptr != 0
    assert not scratch.allocation_offsets
    assert not scratch.allocation_groups


def test_gfx1151_exact_prefill_scratch_uses_bounded_liveness_owners(monkeypatch) -> None:
    allocations = _install_fake_device(monkeypatch)
    _clear_diagnostic_environment(monkeypatch)

    scratch = _GGUFFullAttentionPrefillScratch.allocate(
        _fake_runner("hip_gfx1151"),
        rows=4096,
        capacity=4352,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )

    assert scratch.allocation_mode == "liveness_aliased"
    assert sum(buffer.nbytes for buffer in scratch.buffers) <= 384 * _MIB
    assert max(buffer.nbytes for buffer in scratch.buffers) <= 384 * _MIB
    assert scratch.prefill_query == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.prefill_key == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.prefill_value == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.linear_qkv_f32.ptr != 0
    assert scratch.conv_out.ptr != 0
    assert scratch.moe_down_out.ptr != 0
    assert scratch.moe_down_out_f32 == DeviceBuffer(ptr=0, nbytes=0)

    offsets = dict(scratch.allocation_offsets)
    lifetimes = dict(scratch.allocation_lifetimes)
    # SH-M2 graph-colors route/stage-disjoint fields into allocator-owned
    # slots. The 128-MiB conv and post-attention outputs reuse one owner, while
    # simultaneously-live linear_qkv_f32 remains on another owner.
    groups = dict(scratch.allocation_groups)
    assert groups["conv_out"] == groups["moe_down_out"]
    assert groups["conv_out"].startswith("owner_slot_")
    assert scratch.conv_out.ptr == scratch.moe_down_out.ptr
    assert scratch.linear_qkv_f32.ptr != scratch.conv_out.ptr
    linear_qkv_offset, linear_qkv_size = offsets["linear_qkv_f32"]
    conv_offset, conv_size = offsets["conv_out"]
    assert linear_qkv_offset >= conv_offset + conv_size or conv_offset >= linear_qkv_offset + linear_qkv_size
    entries = list(offsets.items())
    for index, (name_a, (offset_a, size_a)) in enumerate(entries):
        for name_b, (offset_b, size_b) in entries[index + 1 :]:
            lifetimes_overlap = gguf_runner._prefill_scratch_lifetimes_overlap(
                lifetimes[name_a],
                lifetimes[name_b],
            )
            ranges_overlap = offset_a < offset_b + size_b and offset_b < offset_a + size_a
            assert not (lifetimes_overlap and ranges_overlap), (
                f"live scratch buffers overlap: {name_a}={offsets[name_a]}, "
                f"{name_b}={offsets[name_b]}"
            )

    largest_owner = max(allocations, key=lambda buffer: buffer.nbytes)
    assert largest_owner in scratch.buffers
    assert largest_owner.nbytes <= 384 * _MIB


def test_prefill_scratch_keeps_dedicated_layout_for_diagnostics_and_unvalidated_backend(monkeypatch) -> None:
    _install_fake_device(monkeypatch)
    _clear_diagnostic_environment(monkeypatch)
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE", "1")

    diagnostic = _GGUFFullAttentionPrefillScratch.allocate(
        _fake_runner(),
        rows=4096,
        capacity=4352,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )
    assert diagnostic.allocation_mode == "dedicated"
    assert sum(buffer.nbytes for buffer in diagnostic.buffers) > 1700 * _MIB
    assert diagnostic.moe_down_out_f32.ptr != 0
    assert not diagnostic.allocation_offsets

    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE")
    unvalidated = _GGUFFullAttentionPrefillScratch.allocate(
        _fake_runner("cpu_reference"),
        rows=4096,
        capacity=4352,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )
    assert unvalidated.allocation_mode == "dedicated"
    assert sum(buffer.nbytes for buffer in unvalidated.buffers) > 1700 * _MIB
