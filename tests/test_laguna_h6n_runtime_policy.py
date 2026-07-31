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
_SOURCE_POLICY = {
    _GLOBAL_ROLE: _H6A_GLOBAL,
    _SWA_ROLE: _H6A_SWA,
}
_CANDIDATE_POLICY = {
    _GLOBAL_ROLE: _H6N_GLOBAL,
    _SWA_ROLE: _H6A_SWA,
}
_GLOBAL_RETAINED = "global_context_rows_spans"
_SWA_RETAINED = "swa_context_rows_qrow4_m128_c256_exact_spans"


class _FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x10000000
        self.allocations: dict[int, int] = {}
        self.malloc_calls = 0

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
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


def _install_fake_dispatch(cache, calls: list[tuple[str, dict[str, object]]]) -> None:
    def resolve(layer: str, variant: str):
        calls.append((f"resolve:{layer}:{variant}", {}))

        def launch(*args: object, **kwargs: object) -> None:
            del args
            calls.append((f"launch:{variant}", dict(kwargs)))

        return launch

    cache._resolve = resolve


def _dispatch(cache, layer_id: int, start: int, rows: int) -> bool:
    cache.position = start - 1
    cache.prepare_rows(tuple(range(start, start + rows)))
    qualified = cache.can_preappend_attention_prefill(layer_id, rows)
    if qualified:
        cache.append_rows(layer_id, 0x2000, 0x3000, rows)
        cache.attend_prefill_cached(
            layer_id,
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            rows,
        )
    else:
        cache.attend_prefill(
            layer_id,
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            rows,
        )
        cache.append_rows(layer_id, 0x2000, 0x3000, rows)
    cache.discard_rows()
    return qualified


def test_h6n_bounded_default_off_runtime_owner_and_fallbacks(monkeypatch) -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.runtime import laguna_kv as module
    from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch

    assert getattr(hip_gfx1100, _DENSE_CAPABILITY) == _SOURCE_POLICY
    assert not hasattr(hip_gfx1151, _DENSE_CAPABILITY)
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "laguna_attention_prefill",
            "bf16",
            _H6N_GLOBAL,
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
    ) == 161_120_256
    keys_before = registered_keys()

    monkeypatch.setattr(
        hip_gfx1100,
        _DENSE_CAPABILITY,
        _CANDIDATE_POLICY,
    )
    runtime = _FakeRuntime()
    cache = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    calls: list[tuple[str, dict[str, object]]] = []
    _install_fake_dispatch(cache, calls)
    try:
        assert cache.prefill_preappend_role_scoped is True
        assert cache.prefill_preappend_role_variants == _CANDIDATE_POLICY
        for layer_id, candidate in ((0, _H6N_GLOBAL), (1, _H6A_SWA)):
            for start in (0, 128, 256, 384):
                before = len(calls)
                assert _dispatch(cache, layer_id, start, 128)
                launches = [
                    call for call in calls[before:] if call[0].startswith("launch:")
                ]
                writer = (
                    "global_f32_rows_spans"
                    if layer_id == 0
                    else "swa_f32_rows_spans"
                )
                assert [call[0] for call in launches] == [
                    f"launch:{writer}",
                    f"launch:{candidate}",
                ]
                assert launches[-1][1]["start_position"] == start

        for layer_id, start, rows, expected in (
            (0, 512, 128, _GLOBAL_RETAINED),
            (0, 0, 64, _GLOBAL_RETAINED),
            (1, 64, 128, _SWA_RETAINED),
            (1, 0, 64, _SWA_RETAINED),
        ):
            before = len(calls)
            assert not _dispatch(cache, layer_id, start, rows)
            launches = [
                call[0] for call in calls[before:] if call[0].startswith("launch:")
            ]
            assert launches[0] == f"launch:{expected}"

        cache._dense_initial_metadata_valid = False
        before = len(calls)
        assert not _dispatch(cache, 0, 0, 128)
        launches = [
            call[0] for call in calls[before:] if call[0].startswith("launch:")
        ]
        assert f"launch:{_H6N_GLOBAL}" not in launches
    finally:
        cache.free()
    assert runtime.allocations == {}
    assert registered_keys() == keys_before

    explicit_runtime = _FakeRuntime()
    explicit = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=explicit_runtime,
        global_prefill_variant="global_context_rows_qrow2_online_spans",
    )
    try:
        assert explicit.prefill_preappend_role_variants == {_SWA_ROLE: _H6A_SWA}
        explicit.position = -1
        explicit.prepare_rows(tuple(range(128)))
        assert not explicit.can_preappend_attention_prefill(0, 128)
        assert explicit.can_preappend_attention_prefill(1, 128)
        explicit.discard_rows()
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
        assert missing.prefill_preappend_role_variants == {_SWA_ROLE: _H6A_SWA}
        missing.position = -1
        missing.prepare_rows(tuple(range(128)))
        assert not missing.can_preappend_attention_prefill(0, 128)
        assert missing.can_preappend_attention_prefill(1, 128)
        missing.discard_rows()
    finally:
        missing.free()
    assert missing_runtime.allocations == {}
