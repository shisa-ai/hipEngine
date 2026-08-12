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
    # Q6 source-F16 prefill adds three liveness-aliased transient planes while
    # preserving the sole resident T16 weight allocation. FFN down safely casts
    # its dead BF16 input in place, so the shared arena does not grow here.
    assert sum(buffer.nbytes for buffer in scratch.buffers) <= 115 * _MIB
    assert max(buffer.nbytes for buffer in scratch.buffers) <= 113 * _MIB
    assert scratch.q6_f16_x.ptr != 0
    assert scratch.q6_f16_weight.ptr != 0
    assert scratch.q6_f16_out.ptr != 0
    assert scratch.ffn_gate_up.ptr != 0
    assert scratch.ffn_intermediate.ptr != 0
    assert scratch.ffn_down.ptr != 0
    assert scratch.moe_q8_1 == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.moe_shared_gate == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.moe_shared_intermediate == DeviceBuffer(ptr=0, nbytes=0)
    assert scratch.moe_down_out == DeviceBuffer(ptr=0, nbytes=0)
    offsets = dict(scratch.allocation_offsets)
    lifetimes = dict(scratch.allocation_lifetimes)
    assert offsets
    intermediate_offset, intermediate_size = offsets["ffn_intermediate"]
    down_offset, down_size = offsets["ffn_down"]
    assert (
        intermediate_offset + intermediate_size <= down_offset
        or down_offset + down_size <= intermediate_offset
    )
    entries = list(offsets.items())
    for index, (name_a, (offset_a, size_a)) in enumerate(entries):
        for name_b, (offset_b, size_b) in entries[index + 1 :]:
            lifetimes_overlap = gguf_runner._prefill_scratch_lifetimes_overlap(
                lifetimes[name_a],
                lifetimes[name_b],
            )
            ranges_overlap = (
                offset_a < offset_b + size_b
                and offset_b < offset_a + size_a
            )
            assert not (lifetimes_overlap and ranges_overlap), (
                f"live dense scratch buffers overlap: {name_a}={offsets[name_a]}, "
                f"{name_b}={offsets[name_b]}"
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
