from __future__ import annotations

from types import SimpleNamespace

from hipengine.core.hip import HipMemcpyKind
from hipengine.kernels.registry import KernelKey, is_registered, registered_keys
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION

_DENSE_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS"
_GLOBAL_ROLE = "global_m128_c4096_first_fill_exact"
_H6A_GLOBAL = "global_context_rows_dense_initial_cached_exact_spans"
_H6N_GLOBAL = "global_context_rows_dense_initial_fixed512_cached_exact_spans"
_SWA_ROLE = "swa_qrow4_m128_c512_no_wrap_exact"
_H6A_SWA = "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
_H6W_SWA = (
    "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
)
_H5R_SWA = "swa_context_rows_qrow4_cached_exact_spans"
_ROLLBACK_POLICY = {
    _GLOBAL_ROLE: _H6A_GLOBAL,
    _SWA_ROLE: _H6A_SWA,
}
_PRODUCTION_POLICY = {
    _GLOBAL_ROLE: _H6N_GLOBAL,
    _SWA_ROLE: _H6W_SWA,
}
_H5R_POLICY = {_SWA_ROLE: _H5R_SWA}
_PRODUCTION_WORKSPACE_BYTES = 161_120_256


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


def _qualifies(cache, layer_id: int, start: int, rows: int) -> bool:
    cache.position = start - 1
    cache.prepare_rows(tuple(range(start, start + rows)))
    try:
        return cache.can_preappend_attention_prefill(layer_id, rows)
    finally:
        cache.discard_rows()


def test_h6n_source_default_promotes_only_fixed512_global_role(monkeypatch) -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        register_laguna_kv_attention_kernels,
    )
    from hipengine.runtime import laguna_kv as module
    from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch

    register_laguna_kv_attention_kernels()
    assert getattr(hip_gfx1100, _DENSE_CAPABILITY) == _PRODUCTION_POLICY
    assert hip_gfx1100.LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS == _H5R_POLICY
    assert not hasattr(hip_gfx1151, _DENSE_CAPABILITY)
    for variant in (_H6A_GLOBAL, _H6N_GLOBAL, _H6A_SWA, _H6W_SWA, _H5R_SWA):
        assert is_registered(
            KernelKey(
                "hip_gfx1100",
                "laguna_attention_prefill",
                "bf16",
                variant,
            )
        )
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            "laguna_attention_prefill",
            "bf16",
            _H6N_GLOBAL,
        )
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == _PRODUCTION_WORKSPACE_BYTES
    keys_before = registered_keys()

    runtime = _FakeRuntime()
    source = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    try:
        assert source.prefill_preappend_role_scoped is True
        assert source.prefill_preappend_role_variants == _PRODUCTION_POLICY
        for layer_id in (0, 1):
            for start in (0, 128, 256, 384):
                assert _qualifies(source, layer_id, start, 128)
        assert not _qualifies(source, 0, 512, 128)
        assert not _qualifies(source, 0, 0, 64)
        source._dense_initial_metadata_valid = False
        assert not _qualifies(source, 0, 0, 128)
        assert not _qualifies(source, 1, 0, 128)
    finally:
        source.free()
    assert runtime.allocations == {}
    assert registered_keys() == keys_before

    monkeypatch.setattr(hip_gfx1100, _DENSE_CAPABILITY, _ROLLBACK_POLICY)
    rollback_runtime = _FakeRuntime()
    rollback = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=rollback_runtime,
    )
    try:
        assert rollback.prefill_preappend_role_variants == _ROLLBACK_POLICY
        assert _qualifies(rollback, 0, 0, 128)
        assert _qualifies(rollback, 1, 0, 128)
    finally:
        rollback.free()
    assert rollback_runtime.allocations == {}

    monkeypatch.setattr(hip_gfx1100, _DENSE_CAPABILITY, _PRODUCTION_POLICY)
    explicit_runtime = _FakeRuntime()
    explicit = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=explicit_runtime,
        global_prefill_variant="global_context_rows_qrow2_online_spans",
    )
    try:
        assert explicit.prefill_preappend_role_variants == {_SWA_ROLE: _H6W_SWA}
        assert not _qualifies(explicit, 0, 0, 128)
        assert _qualifies(explicit, 1, 0, 128)
    finally:
        explicit.free()
    assert explicit_runtime.allocations == {}

    original_is_registered = module.is_registered
    monkeypatch.setattr(
        module,
        "is_registered",
        lambda key: False if key.variant == _H6N_GLOBAL else original_is_registered(key),
    )
    missing_runtime = _FakeRuntime()
    missing = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=missing_runtime,
    )
    try:
        assert missing.prefill_preappend_role_variants == {_SWA_ROLE: _H6W_SWA}
        assert not _qualifies(missing, 0, 0, 128)
        assert _qualifies(missing, 1, 0, 128)
    finally:
        missing.free()
    assert missing_runtime.allocations == {}
