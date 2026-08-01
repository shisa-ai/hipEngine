from __future__ import annotations

from types import SimpleNamespace

from hipengine.core.hip import HipMemcpyKind
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION


_DENSE_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS"
_H6A_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6A_ROLE_VARIANTS"
_GLOBAL_ROLE = "global_m128_c4096_first_fill_exact"
_GLOBAL_CANDIDATE = "global_context_rows_dense_initial_cached_exact_spans"
_H6N_GLOBAL = "global_context_rows_dense_initial_fixed512_cached_exact_spans"
_SWA_ROLE = "swa_qrow4_m128_c512_no_wrap_exact"
_SWA_CANDIDATE = "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
_H5R = "swa_context_rows_qrow4_cached_exact_spans"
_H5U = "global_context_rows_cached_exact_spans"
_SOURCE_POLICY = {
    _GLOBAL_ROLE: _H6N_GLOBAL,
    _SWA_ROLE: _SWA_CANDIDATE,
}
_H6A_POLICY = {
    _GLOBAL_ROLE: _GLOBAL_CANDIDATE,
    _SWA_ROLE: _SWA_CANDIDATE,
}
_H5R_POLICY = {_SWA_ROLE: _H5R}


class _FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x10000000
        self.allocations: dict[int, int] = {}

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(0x1000, int(nbytes) + 0x100)
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.allocations.pop(int(ptr), None)

    def memcpy(self, dst: int, src: int, count: int, kind: HipMemcpyKind) -> None:
        del src, count
        assert kind == HipMemcpyKind.HOST_TO_DEVICE
        assert any(
            base <= int(dst) < base + max(size, 1)
            for base, size in self.allocations.items()
        )

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        del value, nbytes
        assert any(
            base <= int(dst) < base + max(size, 1)
            for base, size in self.allocations.items()
        )


def _production_config() -> SimpleNamespace:
    layer_types = tuple(
        FULL_ATTENTION if layer_id % 4 == 0 else SLIDING_ATTENTION
        for layer_id in range(48)
    )
    return SimpleNamespace(
        block_count=48,
        layer_types=layer_types,
        head_counts=tuple(
            48 if attention_type == FULL_ATTENTION else 72
            for attention_type in layer_types
        ),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )


def test_h6a_dense_initial_leaves_remain_registered_rollbacks(monkeypatch) -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    assert getattr(hip_gfx1100, _H6A_CAPABILITY) == _SOURCE_POLICY
    assert hip_gfx1100.LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS == _H5R_POLICY
    assert not hasattr(hip_gfx1151, _DENSE_CAPABILITY)
    assert not hasattr(hip_gfx1151, _H6A_CAPABILITY)
    monkeypatch.setattr(hip_gfx1100, _DENSE_CAPABILITY, _SOURCE_POLICY)

    runtime = _FakeRuntime()
    cache = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    try:
        assert cache.prefill_preappend_role_scoped is True
        assert cache.prefill_preappend_role_variants == _SOURCE_POLICY
        for layer_id in (0, 1):
            cache.position = -1
            cache.prepare_rows(tuple(range(128)))
            assert cache.can_preappend_attention_prefill(layer_id, 128)
            cache.discard_rows()
        for variant in (
            _GLOBAL_CANDIDATE,
            _H6N_GLOBAL,
            _SWA_CANDIDATE,
            _H5R,
            _H5U,
        ):
            assert is_registered(
                KernelKey(
                    "hip_gfx1100",
                    "laguna_attention_prefill",
                    "bf16",
                    variant,
                )
            )
    finally:
        cache.free()
    assert runtime.allocations == {}

    monkeypatch.setattr(hip_gfx1100, _DENSE_CAPABILITY, _H6A_POLICY)
    rollback_runtime = _FakeRuntime()
    rollback = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=rollback_runtime,
    )
    try:
        assert rollback.prefill_preappend_role_variants == _H6A_POLICY
        rollback.prepare_rows(tuple(range(128)))
        assert rollback.can_preappend_attention_prefill(0, 128)
        assert rollback.can_preappend_attention_prefill(1, 128)
        rollback.discard_rows()
    finally:
        rollback.free()
    assert rollback_runtime.allocations == {}
