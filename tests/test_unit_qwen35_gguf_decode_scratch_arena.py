from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.policy import QWEN35_DENSE_H5120_GEOMETRY
from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION
from hipengine.runtime import qwen35_gguf_runner as gguf_runner


def _fake_dense_qwen36_runner() -> SimpleNamespace:
    cfg = SimpleNamespace(
        context_length=32_768,
        layer_types=tuple([LINEAR_ATTENTION] * 48 + [FULL_ATTENTION] * 16),
        expert_used_count=0,
        is_moe=False,
        expert_count=0,
        expert_shared_feed_forward_length=0,
        ssm_inner_size=6_144,
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
        hidden_size=5_120,
        q_width=6_144,
        kv_width=1_024,
        ffn_size=17_408,
        linear_qkv_width=10_240,
        ssm_value_dim=128,
        weights=SimpleNamespace(
            config=cfg,
            geometry=QWEN35_DENSE_H5120_GEOMETRY,
            model_name="arbitrary-finetune-name",
            file_type_name="MOSTLY_Q4_K_M",
        ),
    )


class _FakeRuntime:
    def memset(self, *args, **kwargs) -> None:
        return None


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


def _logical_device_buffers(scratch) -> tuple[DeviceBuffer, ...]:
    found: dict[tuple[int, int], DeviceBuffer] = {}
    physical_owners = {(int(buffer.ptr), int(buffer.nbytes)) for buffer in scratch.buffers}
    for name, value in vars(scratch).items():
        if name == "buffers":
            continue
        values = value if isinstance(value, tuple) else (value,)
        for candidate in values:
            if not isinstance(candidate, DeviceBuffer):
                continue
            key = (int(candidate.ptr), int(candidate.nbytes))
            if key not in physical_owners:
                found[key] = candidate
    return tuple(found.values())


def test_dense_qwen36_private_c1_decode_scratch_uses_one_physical_owner(
    monkeypatch,
) -> None:
    allocations = _install_fake_device(monkeypatch)

    scratch = gguf_runner._FullStackScratch.allocate(
        _fake_dense_qwen36_runner(),
        runtime=_FakeRuntime(),
        max_sequence_length=640,
        max_batch_size=1,
        use_single_arena=True,
    )

    assert scratch.allocation_mode == "single_arena"
    assert len(allocations) == 1
    assert scratch.buffers == (allocations[0],)
    logical = _logical_device_buffers(scratch)
    assert len(logical) == 188
    owner_start = int(allocations[0].ptr)
    owner_end = owner_start + int(allocations[0].nbytes)
    ranges = sorted(
        (int(buffer.ptr), int(buffer.ptr) + int(buffer.nbytes)) for buffer in logical
    )
    assert all(owner_start <= start < end <= owner_end for start, end in ranges)
    assert all(end <= next_start for (_, end), (next_start, _) in zip(ranges, ranges[1:]))
    assert allocations[0].nbytes - sum(buffer.nbytes for buffer in logical) < 188 * 256


def test_dense_qwen36_decode_scratch_keeps_dedicated_fallback(monkeypatch) -> None:
    allocations = _install_fake_device(monkeypatch)

    scratch = gguf_runner._FullStackScratch.allocate(
        _fake_dense_qwen36_runner(),
        runtime=_FakeRuntime(),
        max_sequence_length=640,
        max_batch_size=1,
        use_single_arena=False,
    )

    assert scratch.allocation_mode == "dedicated"
    assert len(allocations) == 188
    assert len(scratch.buffers) == 188


def test_dense_qwen36_decode_scratch_owner_denial_falls_back(monkeypatch) -> None:
    next_ptr = 0x10000000
    allocations: list[DeviceBuffer] = []
    denied = False

    def fake_malloc(nbytes: int, *, runtime):
        nonlocal denied, next_ptr
        if not denied:
            denied = True
            raise MemoryError("deny single owner")
        size = int(nbytes)
        buffer = DeviceBuffer(ptr=next_ptr, nbytes=size)
        next_ptr += max(256, ((size + 255) // 256) * 256 + 256)
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    monkeypatch.setattr(gguf_runner, "copy_host_to_device", lambda *args, **kwargs: None)

    scratch = gguf_runner._FullStackScratch.allocate(
        _fake_dense_qwen36_runner(),
        runtime=_FakeRuntime(),
        max_sequence_length=640,
        max_batch_size=1,
        use_single_arena=True,
    )

    assert denied
    assert scratch.allocation_mode == "dedicated"
    assert len(allocations) == 188
    assert len(scratch.buffers) == 188


def test_private_c1_decode_scratch_arena_is_geometry_policy_scoped(monkeypatch) -> None:
    policy = {
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {"enabled": True},
    }
    monkeypatch.delenv(
        "HIPENGINE_GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA",
        raising=False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "backend_package_capability",
        lambda backend, name, default=None: policy
        if name == "GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA_POLICIES"
        else default,
    )

    common = {
        "backend": "hip_gfx1100",
        "geometry": QWEN35_DENSE_H5120_GEOMETRY,
        "file_type_name": "MOSTLY_Q4_K_M",
    }
    assert gguf_runner._resolve_gguf_private_c1_decode_scratch_arena(
        **common,
        max_batch_size=1,
        has_shared_runner=False,
    ) == (True, "private_c1_geometry_policy")
    assert gguf_runner._resolve_gguf_private_c1_decode_scratch_arena(
        **common,
        max_batch_size=2,
        has_shared_runner=False,
        requested=True,
    ) == (False, "multi_row_fallback")
    assert gguf_runner._resolve_gguf_private_c1_decode_scratch_arena(
        **common,
        max_batch_size=1,
        has_shared_runner=True,
        requested=True,
    ) == (False, "shared_runner_fallback")
    assert gguf_runner._resolve_gguf_private_c1_decode_scratch_arena(
        backend="hip_gfx1100",
        geometry=replace(QWEN35_DENSE_H5120_GEOMETRY, head_count=23),
        file_type_name="MOSTLY_Q4_K_M",
        max_batch_size=1,
        has_shared_runner=False,
        requested=True,
    ) == (False, "backend_capability_fallback")
    monkeypatch.setenv("HIPENGINE_GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA", "0")
    assert gguf_runner._resolve_gguf_private_c1_decode_scratch_arena(
        **common,
        max_batch_size=1,
        has_shared_runner=False,
    ) == (False, "disabled")
